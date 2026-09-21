"""Regression tests for `chowder.kaggle_equivalence`.

Builds real `ParentEvalReport`-shaped evidence via the unmodified
`parent_eval.aggregate_parent_result`, and writes real
`predictions-*.jsonl` files to temp directories -- the same on-disk shape
`parent_tournament.evaluate_parent` produces -- so these tests double as
a fidelity check against the real evidence/predictions shape. No
network, no GPU, no Kaggle access.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from chowder.kaggle_equivalence import (
    BackendEquivalenceError,
    BackendFingerprint,
    build_equivalence_report,
    qualify_backend,
)
from chowder.parent_eval import PARENT_DIMENSIONS, ParentEvalSpec, ParentSuiteSpec, aggregate_parent_result

VALID_SHA = "a" * 40


def _suite_name(dimension: str) -> str:
    return f"suite-{dimension}-v1"


def _build_spec(*, precision: str = "bf16") -> ParentEvalSpec:
    suites = [
        ParentSuiteSpec(name=_suite_name(dim), dimension=dim, dataset=f"synthetic://{dim}", max_new_tokens=256)
        for dim in PARENT_DIMENSIONS
    ]
    return ParentEvalSpec(suites=tuple(suites), precision=precision)


def _fingerprint(**overrides) -> BackendFingerprint:
    defaults = dict(
        python_version="3.11.9",
        torch_version="2.11.0+cu128",
        transformers_version="5.16.1",
        bitsandbytes_version="0.50.2",
        accelerate_version="1.6.0",
        cuda_runtime_version="12.8",
        gpu_models=("NVIDIA GeForce RTX 5060 Ti",),
        gpu_count=1,
        device_map_summary='{"": 0}',
        quantization="4bit",
        dtype="bfloat16",
        tokenizer_identity_sha256="t" * 64,
        chowder_commit_sha=VALID_SHA,
    )
    defaults.update(overrides)
    return BackendFingerprint(**defaults)


def _write_predictions(directory: Path, *, items: dict[str, list[dict]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for suite_name, rows in items.items():
        path = directory / f"predictions-{suite_name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")


def _default_items(spec: ParentEvalSpec, *, prediction_suffix: str = "") -> dict[str, list[dict]]:
    items = {}
    for suite in spec.suites:
        rows = []
        for i in range(6):
            rows.append(
                {
                    "prompt": f"prompt for {suite.name} item {i}",
                    "expected": "42",
                    "prediction": f"reasoning...\n</think>\n\n42{prediction_suffix}",
                    "score": 1.0,
                }
            )
        items[suite.name] = rows
    return items


def _report(spec: ParentEvalSpec, *, label: str, revision: str = VALID_SHA, suite_digest_seed: str = "v1") -> dict:
    metrics = {suite.name: 1.0 for suite in spec.suites}
    suite_evidence = {suite.name: {"holdout_fingerprints_sha256": f"{suite_digest_seed}-{suite.name}"} for suite in spec.suites}
    report = aggregate_parent_result(spec=spec, base_model=label, revision=revision, metrics=metrics, evidence={})
    return report.to_dict(), suite_evidence


def _tokenizer(identity: str = "id-1") -> dict:
    return {"tokenizer_class": "Qwen2Tokenizer", "vocab_size": 151936, "identity_sha256": identity}


def test_perfect_equivalence_qualifies(tmp_path):
    spec = _build_spec()
    local_report, local_suite_evidence = _report(spec, label="parent-a-qwen38-27b-official")
    kaggle_report, kaggle_suite_evidence = _report(spec, label="parent-a-qwen38-27b-official")
    local_dir = tmp_path / "local"
    kaggle_dir = tmp_path / "kaggle"
    _write_predictions(local_dir, items=_default_items(spec))
    _write_predictions(kaggle_dir, items=_default_items(spec))

    report = build_equivalence_report(
        parent_label="parent-a-qwen38-27b-official",
        local_report=local_report,
        kaggle_report=kaggle_report,
        local_suite_evidence=local_suite_evidence,
        kaggle_suite_evidence=kaggle_suite_evidence,
        local_predictions_dir=local_dir,
        kaggle_predictions_dir=kaggle_dir,
        local_tokenizer=_tokenizer(),
        kaggle_tokenizer=_tokenizer(),
        local_environment=_fingerprint(),
        kaggle_environment=_fingerprint(gpu_models=("Tesla T4",), cuda_runtime_version="12.4"),
    )
    assert report.all_scores_equal
    assert report.all_answers_equal
    assert report.protocol_digest_equal

    record = qualify_backend(report, qualification_id="local-rtx5060ti_vs_kaggle-t4x2")
    assert record.status == "qualified"
    assert record.is_qualified
    assert record.item_total_count == 54
    assert record.item_score_agreement_count == 54


def test_precision_difference_qualifies_with_declared_reason(tmp_path):
    local_spec = _build_spec(precision="bf16")
    kaggle_spec = _build_spec(precision="fp16")  # T4 cannot run bf16 -- see module docstring
    local_report, local_suite_evidence = _report(local_spec, label="parent-a-qwen38-27b-official")
    kaggle_report, kaggle_suite_evidence = _report(kaggle_spec, label="parent-a-qwen38-27b-official")
    assert local_report["evaluation_protocol_sha256"] != kaggle_report["evaluation_protocol_sha256"]

    local_dir = tmp_path / "local"
    kaggle_dir = tmp_path / "kaggle"
    _write_predictions(local_dir, items=_default_items(local_spec))
    _write_predictions(kaggle_dir, items=_default_items(kaggle_spec))

    report = build_equivalence_report(
        parent_label="parent-a-qwen38-27b-official",
        local_report=local_report,
        kaggle_report=kaggle_report,
        local_suite_evidence=local_suite_evidence,
        kaggle_suite_evidence=kaggle_suite_evidence,
        local_predictions_dir=local_dir,
        kaggle_predictions_dir=kaggle_dir,
        local_tokenizer=_tokenizer(),
        kaggle_tokenizer=_tokenizer(),
        local_environment=_fingerprint(),
        kaggle_environment=_fingerprint(gpu_models=("Tesla T4",), dtype="float16"),
        declared_digest_divergence_reasons=(
            "precision: bf16 (local) vs fp16 (kaggle) -- T4 lacks bf16 tensor core "
            "support (torch.cuda.is_bf16_supported() is False); base_text_worker._dtype "
            "raises on bf16 there",
        ),
    )
    record = qualify_backend(report, qualification_id="local-rtx5060ti_vs_kaggle-t4x2")
    assert record.status == "qualified_with_acknowledged_differences"
    assert record.is_qualified
    assert not record.protocol_digest_equal
    assert record.declared_digest_divergence_reasons


def test_undeclared_protocol_mismatch_not_qualified(tmp_path):
    local_spec = _build_spec(precision="bf16")
    kaggle_spec = _build_spec(precision="fp16")
    local_report, local_suite_evidence = _report(local_spec, label="parent-a-qwen38-27b-official")
    kaggle_report, kaggle_suite_evidence = _report(kaggle_spec, label="parent-a-qwen38-27b-official")
    local_dir = tmp_path / "local"
    kaggle_dir = tmp_path / "kaggle"
    _write_predictions(local_dir, items=_default_items(local_spec))
    _write_predictions(kaggle_dir, items=_default_items(kaggle_spec))

    report = build_equivalence_report(
        parent_label="parent-a-qwen38-27b-official",
        local_report=local_report,
        kaggle_report=kaggle_report,
        local_suite_evidence=local_suite_evidence,
        kaggle_suite_evidence=kaggle_suite_evidence,
        local_predictions_dir=local_dir,
        kaggle_predictions_dir=kaggle_dir,
        local_tokenizer=_tokenizer(),
        kaggle_tokenizer=_tokenizer(),
        local_environment=_fingerprint(),
        kaggle_environment=_fingerprint(gpu_models=("Tesla T4",)),
        # no declared_digest_divergence_reasons supplied
    )
    record = qualify_backend(report, qualification_id="local-rtx5060ti_vs_kaggle-t4x2")
    assert record.status == "not_qualified"
    assert not record.is_qualified
    assert "undeclared" in record.detail or "no declared_digest_divergence_reasons" in record.detail


def test_item_score_mismatch_fails_closed(tmp_path):
    spec = _build_spec()
    local_report, local_suite_evidence = _report(spec, label="parent-a-qwen38-27b-official")
    kaggle_report, kaggle_suite_evidence = _report(spec, label="parent-a-qwen38-27b-official")
    local_dir = tmp_path / "local"
    kaggle_dir = tmp_path / "kaggle"
    local_items = _default_items(spec)
    kaggle_items = _default_items(spec)
    # Flip one Kaggle item's score/prediction.
    kaggle_items[_suite_name("reasoning")][0]["score"] = 0.0
    kaggle_items[_suite_name("reasoning")][0]["prediction"] = "reasoning...\n</think>\n\nwrong"
    _write_predictions(local_dir, items=local_items)
    _write_predictions(kaggle_dir, items=kaggle_items)

    report = build_equivalence_report(
        parent_label="parent-a-qwen38-27b-official",
        local_report=local_report,
        kaggle_report=kaggle_report,
        local_suite_evidence=local_suite_evidence,
        kaggle_suite_evidence=kaggle_suite_evidence,
        local_predictions_dir=local_dir,
        kaggle_predictions_dir=kaggle_dir,
        local_tokenizer=_tokenizer(),
        kaggle_tokenizer=_tokenizer(),
        local_environment=_fingerprint(),
        kaggle_environment=_fingerprint(gpu_models=("Tesla T4",)),
    )
    assert not report.all_scores_equal
    record = qualify_backend(report, qualification_id="local-rtx5060ti_vs_kaggle-t4x2")
    assert record.status == "not_qualified"
    assert record.item_score_agreement_count == 53
    assert "1/54" in record.detail


def test_suite_digest_mismatch_fails_closed(tmp_path):
    spec = _build_spec()
    local_report, local_suite_evidence = _report(spec, label="parent-a-qwen38-27b-official", suite_digest_seed="v1")
    kaggle_report, kaggle_suite_evidence = _report(spec, label="parent-a-qwen38-27b-official", suite_digest_seed="DIFFERENT")
    local_dir = tmp_path / "local"
    kaggle_dir = tmp_path / "kaggle"
    _write_predictions(local_dir, items=_default_items(spec))
    _write_predictions(kaggle_dir, items=_default_items(spec))

    report = build_equivalence_report(
        parent_label="parent-a-qwen38-27b-official",
        local_report=local_report,
        kaggle_report=kaggle_report,
        local_suite_evidence=local_suite_evidence,
        kaggle_suite_evidence=kaggle_suite_evidence,
        local_predictions_dir=local_dir,
        kaggle_predictions_dir=kaggle_dir,
        local_tokenizer=_tokenizer(),
        kaggle_tokenizer=_tokenizer(),
        local_environment=_fingerprint(),
        kaggle_environment=_fingerprint(gpu_models=("Tesla T4",)),
    )
    assert not report.suite_digest_equal
    record = qualify_backend(report, qualification_id="local-rtx5060ti_vs_kaggle-t4x2")
    assert record.status == "not_qualified"
    assert "suite content digest" in record.detail


def test_tokenizer_mismatch_fails_closed(tmp_path):
    spec = _build_spec()
    local_report, local_suite_evidence = _report(spec, label="parent-a-qwen38-27b-official")
    kaggle_report, kaggle_suite_evidence = _report(spec, label="parent-a-qwen38-27b-official")
    local_dir = tmp_path / "local"
    kaggle_dir = tmp_path / "kaggle"
    _write_predictions(local_dir, items=_default_items(spec))
    _write_predictions(kaggle_dir, items=_default_items(spec))

    report = build_equivalence_report(
        parent_label="parent-a-qwen38-27b-official",
        local_report=local_report,
        kaggle_report=kaggle_report,
        local_suite_evidence=local_suite_evidence,
        kaggle_suite_evidence=kaggle_suite_evidence,
        local_predictions_dir=local_dir,
        kaggle_predictions_dir=kaggle_dir,
        local_tokenizer=_tokenizer(identity="id-1"),
        kaggle_tokenizer=_tokenizer(identity="id-2"),
        local_environment=_fingerprint(),
        kaggle_environment=_fingerprint(gpu_models=("Tesla T4",)),
    )
    assert not report.tokenizer_identity_equal
    record = qualify_backend(report, qualification_id="local-rtx5060ti_vs_kaggle-t4x2")
    assert record.status == "not_qualified"
    assert "tokenizer identity" in record.detail


def test_mismatched_suite_sets_raises():
    with pytest.raises(BackendEquivalenceError):
        from chowder.kaggle_equivalence import compare_items
        import tempfile

        with tempfile.TemporaryDirectory() as local_dir, tempfile.TemporaryDirectory() as kaggle_dir:
            _write_predictions(
                Path(local_dir),
                items={"suite-reasoning-v1": [{"prompt": "p", "expected": "e", "prediction": "e", "score": 1.0}]},
            )
            _write_predictions(
                Path(kaggle_dir),
                items={"suite-coding-v1": [{"prompt": "p", "expected": "e", "prediction": "e", "score": 1.0}]},
            )
            compare_items(local_dir, kaggle_dir)


def test_prompt_text_mismatch_raises(tmp_path):
    from chowder.kaggle_equivalence import compare_items

    local_dir = tmp_path / "local"
    kaggle_dir = tmp_path / "kaggle"
    _write_predictions(
        local_dir, items={"suite-reasoning-v1": [{"prompt": "prompt A", "expected": "e", "prediction": "e", "score": 1.0}]}
    )
    _write_predictions(
        kaggle_dir, items={"suite-reasoning-v1": [{"prompt": "prompt B", "expected": "e", "prediction": "e", "score": 1.0}]}
    )
    with pytest.raises(BackendEquivalenceError, match="prompt text differs"):
        compare_items(local_dir, kaggle_dir)


def test_harmless_formatting_difference_documented(tmp_path):
    spec = _build_spec()
    local_report, local_suite_evidence = _report(spec, label="parent-a-qwen38-27b-official")
    kaggle_report, kaggle_suite_evidence = _report(spec, label="parent-a-qwen38-27b-official")
    local_dir = tmp_path / "local"
    kaggle_dir = tmp_path / "kaggle"
    local_items = _default_items(spec)
    kaggle_items = _default_items(spec, prediction_suffix=".")  # trailing period, still scores 1.0
    _write_predictions(local_dir, items=local_items)
    _write_predictions(kaggle_dir, items=kaggle_items)

    report = build_equivalence_report(
        parent_label="parent-a-qwen38-27b-official",
        local_report=local_report,
        kaggle_report=kaggle_report,
        local_suite_evidence=local_suite_evidence,
        kaggle_suite_evidence=kaggle_suite_evidence,
        local_predictions_dir=local_dir,
        kaggle_predictions_dir=kaggle_dir,
        local_tokenizer=_tokenizer(),
        kaggle_tokenizer=_tokenizer(),
        local_environment=_fingerprint(),
        kaggle_environment=_fingerprint(gpu_models=("Tesla T4",)),
    )
    assert report.all_scores_equal
    assert not report.all_answers_equal  # "42" vs "42."
    record = qualify_backend(report, qualification_id="local-rtx5060ti_vs_kaggle-t4x2")
    assert record.status == "qualified_with_acknowledged_differences"
    assert "raw answers" in record.detail


def test_truncation_difference_detected(tmp_path):
    spec = _build_spec()
    local_report, local_suite_evidence = _report(spec, label="parent-a-qwen38-27b-official")
    kaggle_report, kaggle_suite_evidence = _report(spec, label="parent-a-qwen38-27b-official")
    local_dir = tmp_path / "local"
    kaggle_dir = tmp_path / "kaggle"
    local_items = _default_items(spec)
    kaggle_items = _default_items(spec)
    # Kaggle never closes </think> on the first two reasoning items.
    for i in range(2):
        kaggle_items[_suite_name("reasoning")][i]["prediction"] = "<think> still thinking, ran out of budget"
        kaggle_items[_suite_name("reasoning")][i]["score"] = 0.0
        local_items[_suite_name("reasoning")][i]["score"] = 0.0
        local_items[_suite_name("reasoning")][i]["prediction"] = "reasoning...\n</think>\n\nwrong"
    _write_predictions(local_dir, items=local_items)
    _write_predictions(kaggle_dir, items=kaggle_items)

    report = build_equivalence_report(
        parent_label="parent-a-qwen38-27b-official",
        local_report=local_report,
        kaggle_report=kaggle_report,
        local_suite_evidence=local_suite_evidence,
        kaggle_suite_evidence=kaggle_suite_evidence,
        local_predictions_dir=local_dir,
        kaggle_predictions_dir=kaggle_dir,
        local_tokenizer=_tokenizer(),
        kaggle_tokenizer=_tokenizer(),
        local_environment=_fingerprint(),
        kaggle_environment=_fingerprint(gpu_models=("Tesla T4",)),
    )
    assert report.any_systematic_truncation_difference
    reasoning_items = [c for c in report.item_comparisons if c.suite == _suite_name("reasoning")]
    assert reasoning_items[0].kaggle_truncated is True
    assert reasoning_items[0].local_truncated is False
    assert reasoning_items[0].truncation_difference is True


def test_empty_comparison_not_qualified():
    from chowder.kaggle_equivalence import EquivalenceReport

    report = EquivalenceReport(
        parent_label="parent-a-qwen38-27b-official",
        generated_at="2026-01-01T00:00:00+00:00",
        local_protocol_sha256="a" * 64,
        kaggle_protocol_sha256="a" * 64,
        protocol_digest_equal=True,
        local_suite_digests={},
        kaggle_suite_digests={},
        suite_digest_equal=False,
        local_tokenizer=None,
        kaggle_tokenizer=None,
        tokenizer_identity_equal=False,
        item_comparisons=(),
        dimension_comparisons=(),
        local_capability_mean=None,
        kaggle_capability_mean=None,
        local_behavior_mean=None,
        kaggle_behavior_mean=None,
        declared_digest_divergence_reasons=(),
        local_environment=_fingerprint(),
        kaggle_environment=_fingerprint(),
    )
    record = qualify_backend(report, qualification_id="local-rtx5060ti_vs_kaggle-t4x2")
    assert record.status == "not_qualified"
    assert record.item_total_count == 0


def test_backend_fingerprint_from_dict_round_trips():
    from chowder.kaggle_equivalence import BackendFingerprint

    original = _fingerprint(gpu_models=("Tesla T4", "Tesla T4"), gpu_count=2)
    reloaded = BackendFingerprint.from_dict(json.loads(json.dumps(original.to_dict())))
    assert reloaded == original


def test_backend_fingerprint_validates_commit_sha():
    with pytest.raises(ValueError):
        _fingerprint(chowder_commit_sha="not-a-sha")


def test_deterministic_digest(tmp_path):
    spec = _build_spec()
    local_report, local_suite_evidence = _report(spec, label="parent-a-qwen38-27b-official")
    kaggle_report, kaggle_suite_evidence = _report(spec, label="parent-a-qwen38-27b-official")
    local_dir = tmp_path / "local"
    kaggle_dir = tmp_path / "kaggle"
    _write_predictions(local_dir, items=_default_items(spec))
    _write_predictions(kaggle_dir, items=_default_items(spec))

    def _build():
        return build_equivalence_report(
            parent_label="parent-a-qwen38-27b-official",
            local_report=local_report,
            kaggle_report=kaggle_report,
            local_suite_evidence=local_suite_evidence,
            kaggle_suite_evidence=kaggle_suite_evidence,
            local_predictions_dir=local_dir,
            kaggle_predictions_dir=kaggle_dir,
            local_tokenizer=_tokenizer(),
            kaggle_tokenizer=_tokenizer(),
            local_environment=_fingerprint(),
            kaggle_environment=_fingerprint(gpu_models=("Tesla T4",)),
            now="2026-09-07T00:00:00+00:00",
        )

    report_1 = _build()
    report_2 = _build()
    assert report_1.digest() == report_2.digest()
    record_1 = qualify_backend(report_1, qualification_id="q", now="2026-09-07T00:00:01+00:00")
    record_2 = qualify_backend(report_2, qualification_id="q", now="2026-09-07T00:00:01+00:00")
    assert record_1.digest() == record_2.digest()
