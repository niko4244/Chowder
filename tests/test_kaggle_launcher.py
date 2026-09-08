"""Regression tests for `chowder.kaggle_launcher`.

No GPU, no CUDA, no real torch import required for most of this file --
`capture_environment_fingerprint` and `run_kaggle_parent_evaluation`
accept injectable fakes for exactly that reason. No network access.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from chowder.kaggle_launcher import (
    KAGGLE_T4_USABLE_VRAM_GIB,
    KagglePreflightError,
    PRECISION_DIVERGENCE_REASON,
    REFERENCE_PEAK_GPU_MIB_PARENT_A,
    build_kaggle_eval_spec,
    capture_environment_fingerprint,
    estimate_required_vram_gib,
    local_parent_for_kaggle,
    preflight_vram_headroom,
    run_kaggle_parent_evaluation,
)
from chowder.parent_eval import PARENT_DIMENSIONS, ParentEvalSpec, ParentSuiteSpec, ParentTokenizerEvidence
from chowder.parent_tournament import LocalParent, ParentTournamentError

# Parent A's real, manifest-recorded total (docs/QWEN38_SPARSE_PROGRAM.md /
# the real retry7 manifest): 51.75 GiB.
PARENT_A_REAL_TOTAL_WEIGHT_BYTES = 55_563_006_776


def _build_spec(*, precision: str = "bf16") -> ParentEvalSpec:
    suites = [
        ParentSuiteSpec(name=f"suite-{dim}-v1", dimension=dim, dataset=f"synthetic://{dim}", max_new_tokens=256)
        for dim in PARENT_DIMENSIONS
    ]
    return ParentEvalSpec(suites=tuple(suites), precision=precision, quantization="4bit")


def test_estimate_required_vram_gib_prefers_real_reference_measurement():
    estimate = estimate_required_vram_gib(
        PARENT_A_REAL_TOTAL_WEIGHT_BYTES, reference_peak_gpu_mib_sampled=REFERENCE_PEAK_GPU_MIB_PARENT_A
    )
    expected = (REFERENCE_PEAK_GPU_MIB_PARENT_A / 1024) * 1.10
    assert estimate == pytest.approx(expected)
    # Real evidence: this is roughly the entire nominal capacity of a
    # 16 GiB-class card -- documented, not hidden.
    assert estimate > 16.0


def test_estimate_required_vram_gib_falls_back_to_formula_without_reference():
    estimate = estimate_required_vram_gib(PARENT_A_REAL_TOTAL_WEIGHT_BYTES)
    expected = (PARENT_A_REAL_TOTAL_WEIGHT_BYTES * 0.25 / 2**30) * 1.35
    assert estimate == pytest.approx(expected)


def test_estimate_required_vram_gib_rejects_non_positive_inputs():
    with pytest.raises(ValueError):
        estimate_required_vram_gib(0)
    with pytest.raises(ValueError):
        estimate_required_vram_gib(100, reference_peak_gpu_mib_sampled=0)


def test_preflight_vram_headroom_raises_when_insufficient():
    with pytest.raises(KagglePreflightError, match="exceeds the usable T4 ceiling"):
        preflight_vram_headroom(20.0, free_vram_gib_fn=lambda: 30.0, usable_ceiling_gib=KAGGLE_T4_USABLE_VRAM_GIB)


def test_preflight_vram_headroom_respects_ceiling_even_when_free_reports_more():
    # A live measurement reporting 100 GiB free (implausible for a T4, but
    # exercising the code path) must not bypass the usable ceiling.
    with pytest.raises(KagglePreflightError):
        preflight_vram_headroom(14.0, free_vram_gib_fn=lambda: 100.0, usable_ceiling_gib=13.0)


def test_preflight_vram_headroom_passes_when_sufficient():
    headroom = preflight_vram_headroom(10.0, free_vram_gib_fn=lambda: 14.0, usable_ceiling_gib=14.8)
    assert headroom == 14.0


def test_build_kaggle_eval_spec_changes_only_precision():
    reference = _build_spec(precision="bf16")
    kaggle = build_kaggle_eval_spec(reference)
    assert kaggle.precision == "fp16"
    assert kaggle.quantization == reference.quantization
    assert kaggle.max_model_len == reference.max_model_len
    assert kaggle.require_thinking_efficiency_telemetry == reference.require_thinking_efficiency_telemetry
    assert kaggle.suites == reference.suites
    assert kaggle.digest() != reference.digest()

    reference_dict = reference.to_dict()
    kaggle_dict = kaggle.to_dict()
    del reference_dict["precision"]
    del kaggle_dict["precision"]
    assert reference_dict == kaggle_dict


def test_local_parent_for_kaggle_builds_expected_paths(tmp_path):
    model_dir = tmp_path / "input" / "obliteratus-qwen38-27b"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}")
    manifest_path = tmp_path / "input" / "obliteratus-qwen38-27b.manifest.json"
    manifest_path.write_text("{}")

    parent = local_parent_for_kaggle(
        label="parent-c-obliteratus", revision="a" * 40, kaggle_dataset_root=model_dir
    )
    assert parent.local_path == str(model_dir)
    assert parent.manifest_path == str(manifest_path)


def test_local_parent_for_kaggle_raises_when_directory_missing(tmp_path):
    with pytest.raises(ParentTournamentError):
        local_parent_for_kaggle(
            label="parent-c-obliteratus", revision="a" * 40, kaggle_dataset_root=tmp_path / "never-created"
        )


def test_capture_environment_fingerprint_with_gpu():
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_name=lambda i: "Tesla T4",
        ),
        version=SimpleNamespace(cuda="12.4"),
    )
    fingerprint = capture_environment_fingerprint(
        chowder_commit_sha="b" * 40,
        tokenizer_identity_sha256="c" * 64,
        quantization="4bit",
        dtype="float16",
        device_map_summary='{"": 0}',
        torch_module=fake_torch,
    )
    assert fingerprint.gpu_models == ("Tesla T4",)
    assert fingerprint.gpu_count == 1
    assert fingerprint.cuda_runtime_version == "12.4"
    assert fingerprint.chowder_commit_sha == "b" * 40
    payload = json.dumps(fingerprint.to_dict())
    assert "Tesla T4" in payload


def test_capture_environment_fingerprint_without_gpu():
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    fingerprint = capture_environment_fingerprint(
        chowder_commit_sha="d" * 40,
        tokenizer_identity_sha256=None,
        quantization="none",
        dtype="float32",
        device_map_summary="cpu",
        torch_module=fake_torch,
    )
    assert fingerprint.gpu_models == ()
    assert fingerprint.gpu_count == 0
    assert fingerprint.cuda_runtime_version is None


def test_precision_divergence_reason_names_the_real_check():
    assert "bf16" in PRECISION_DIVERGENCE_REASON
    assert "fp16" in PRECISION_DIVERGENCE_REASON
    assert "base_text_worker._dtype" in PRECISION_DIVERGENCE_REASON


def test_run_kaggle_parent_evaluation_wires_spec_and_preflight(tmp_path):
    model_dir = tmp_path / "input" / "parent-c"
    model_dir.mkdir(parents=True)
    manifest_path = tmp_path / "input" / "parent-c.manifest.json"
    manifest_path.write_text("{}")
    parent = LocalParent(
        label="parent-c-obliteratus", revision="a" * 40, local_path=str(model_dir), manifest_path=str(manifest_path)
    )
    reference_spec = _build_spec(precision="bf16")
    tokenizer = ParentTokenizerEvidence(tokenizer_class="Qwen2Tokenizer", vocab_size=151936, identity_sha256="e" * 64)

    calls = []

    def fake_evaluate_parent(registry, parent_, spec, *, tokenizer, output_root, device, quantization, precision, seed, timeout_seconds):
        calls.append(
            dict(
                parent=parent_, spec=spec, tokenizer=tokenizer, output_root=output_root,
                device=device, quantization=quantization, precision=precision, seed=seed,
                timeout_seconds=timeout_seconds,
            )
        )
        return "FAKE_RESULT"

    result, kaggle_spec = run_kaggle_parent_evaluation(
        registry=object(),
        parent=parent,
        reference_spec=reference_spec,
        tokenizer=tokenizer,
        total_weight_bytes=PARENT_A_REAL_TOTAL_WEIGHT_BYTES,
        free_vram_gib_fn=lambda: 100.0,  # generous fake free VRAM
        output_root=tmp_path / "out",
        seed=20260907,
        reference_peak_gpu_mib_sampled=100,  # tiny synthetic reference so the preflight passes in this test
        evaluate_parent_fn=fake_evaluate_parent,
    )
    assert result == "FAKE_RESULT"
    assert kaggle_spec.precision == "fp16"
    assert len(calls) == 1
    assert calls[0]["precision"] == "fp16"
    assert calls[0]["device"] == "cuda:0"
    assert calls[0]["tokenizer"] is tokenizer
    assert calls[0]["seed"] == 20260907


def test_run_kaggle_parent_evaluation_refuses_before_evaluate_when_vram_insufficient(tmp_path):
    model_dir = tmp_path / "input" / "parent-c"
    model_dir.mkdir(parents=True)
    manifest_path = tmp_path / "input" / "parent-c.manifest.json"
    manifest_path.write_text("{}")
    parent = LocalParent(
        label="parent-c-obliteratus", revision="a" * 40, local_path=str(model_dir), manifest_path=str(manifest_path)
    )
    reference_spec = _build_spec(precision="bf16")
    tokenizer = ParentTokenizerEvidence(tokenizer_class="Qwen2Tokenizer", vocab_size=151936, identity_sha256="e" * 64)

    calls = []

    def fake_evaluate_parent(*args, **kwargs):
        calls.append(1)
        return "SHOULD_NOT_BE_CALLED"

    with pytest.raises(KagglePreflightError):
        run_kaggle_parent_evaluation(
            registry=object(),
            parent=parent,
            reference_spec=reference_spec,
            tokenizer=tokenizer,
            total_weight_bytes=PARENT_A_REAL_TOTAL_WEIGHT_BYTES,
            free_vram_gib_fn=lambda: 2.0,  # far too little
            output_root=tmp_path / "out",
            seed=20260907,
            evaluate_parent_fn=fake_evaluate_parent,
        )
    assert calls == []
