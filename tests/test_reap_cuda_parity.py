"""CUDA parity orchestration tests stay entirely on the CPU (including fake probes)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chowder_batch import reap_cuda_parity as parity
from chowder_batch.reap_host_guard import Snapshot


def _snapshot(timestamp=100.0, *, services=False, gpu=14.0, commit=36.0):
    return Snapshot(
        timestamp=timestamp, ram_free_gib=32.0, virtual_free_gib=commit,
        diskC_free_gib=5.0, diskF_free_gib=30.0, gpu_free_gib=gpu,
        gpu_temp_c=40.0, services_conflict=services,
    )


def test_admission_uses_three_passive_samples_and_no_gpu_runtime():
    probes = iter([_snapshot(100), _snapshot(130), _snapshot(160)])
    sleeps = []
    samples = parity.sample_admission(snapshot_fn=lambda: next(probes),
                                      sleep_fn=sleeps.append)
    assert len(samples) == 3
    assert sleeps == [30, 30]
    assert parity.admission_decision(samples).action == "ADMIT"


@pytest.mark.parametrize("samples", [
    [_snapshot(100), _snapshot(130, services=True), _snapshot(160)],
    [_snapshot(100), _snapshot(130, gpu=2.0), _snapshot(160)],
    [_snapshot(100), _snapshot(130, commit=2.0), _snapshot(160)],
])
def test_admission_refuses_busy_or_insufficient_hosts(samples):
    assert parity.admission_decision(samples).action == "REFUSE"


def test_runtime_refuses_new_service_interference():
    assert parity.runtime_decision([_snapshot(160, services=True)]).action == "ABORT"


def test_nonexistent_or_existing_receipt_never_claims_parity(tmp_path):
    out = tmp_path / "parity.json"
    result = parity.main([
        "--model-dir", str(tmp_path / "no-model"),
        "--calibration", str(tmp_path / "no-calibration"),
        "--observation", str(tmp_path / "no-observation"),
        "--out", str(out),
    ])
    assert result != 0
    assert not out.exists()
    out.write_text("existing evidence", encoding="utf-8")
    assert parity.main([
        "--model-dir", str(tmp_path / "no-model"),
        "--calibration", str(tmp_path / "no-calibration"),
        "--observation", str(tmp_path / "no-observation"),
        "--out", str(out),
        "--execute",
    ]) != 0
    assert out.read_text(encoding="utf-8") == "existing evidence"


def test_exclusive_receipt_publication_is_atomic_and_preserves_prior_evidence(tmp_path):
    target = tmp_path / "measured.json"
    parity.publish_receipt(target, {"schema": parity.SCHEMA, "status": "MEASURED_NUMERIC_DIFFERENCES"})
    value = json.loads(target.read_text(encoding="utf-8"))
    assert value["status"] == "MEASURED_NUMERIC_DIFFERENCES"
    with pytest.raises(FileExistsError):
        parity.publish_receipt(target, {"status": "MEASURED_EXACT_BF16_MATCH"})
    assert json.loads(target.read_text(encoding="utf-8")) == value
    assert not list(tmp_path.glob("*.tmp"))


def test_windows_exclusive_cuda_receipt_publish_survives_no_hardlink_volume(
    tmp_path, monkeypatch,
):
    """Reproduce an external Windows volume with no hardlink support, CPU only."""
    target = tmp_path / "cuda-metric.json"
    monkeypatch.setattr(parity, "_windows_exclusive_rename", lambda: True)

    def no_links(*_args, **_kwargs):
        raise OSError("hard links unsupported on this volume")

    monkeypatch.setattr(parity.os, "link", no_links)
    value = {"schema": parity.SCHEMA, "status": "MEASURED_NUMERIC_DIFFERENCES",
             "scientific_gate": "NOT_REGISTERED"}
    parity.publish_receipt(target, value)
    assert json.loads(target.read_text(encoding="utf-8")) == value
    assert not list(tmp_path.glob(".*.tmp"))


def test_windows_exclusive_cuda_publish_refuses_existing_receipt(tmp_path, monkeypatch):
    target = tmp_path / "original.json"
    original = b"already sealed old numerical evidence\n"
    target.write_bytes(original)
    monkeypatch.setattr(parity, "_windows_exclusive_rename", lambda: True)
    real_rename = Path.rename

    def windows_no_replace(self, new):
        if Path(new) == target and target.exists():
            raise FileExistsError("Windows atomic publication cannot replace")
        return real_rename(self, new)

    monkeypatch.setattr(Path, "rename", windows_no_replace)
    with pytest.raises(FileExistsError, match="cannot replace"):
        parity.publish_receipt(target, {"status": "MEASURED_EXACT_BF16_MATCH"})
    assert target.read_bytes() == original
    assert not list(tmp_path.glob(".*.tmp"))


def test_frozen_prefix_rejects_substituted_calibration_without_tokenizer(tmp_path):
    calibration = tmp_path / "calibration.jsonl"
    calibration.write_text(json.dumps({"text": "different input", "source": "fake"}) + "\n")
    class ShouldNotTokenize:
        def __call__(self, *_args, **_kwargs):
            pytest.fail("unauthenticated calibration was passed to the tokenizer")

    with pytest.raises(ValueError, match="frozen"):
        parity.frozen_prefix(calibration, ShouldNotTokenize(), limit=8)


def test_original_provenance_refuses_changed_hf_decoder_before_cuda(tmp_path, monkeypatch):
    from chowder_batch import reap_hidden_checkpoint

    observation = tmp_path / "historical"
    observation.mkdir()
    model = tmp_path / "original-model"
    model.mkdir()
    frozen = tmp_path / "calibration.jsonl"
    recorded = {
        "checkpoint_receipt": {
            "source_identity": {
                "model_dir": str(model), "weights_dir": str(model),
                "calibration_sha256": parity.FROZEN_V2_BASE_SHA256,
                "trusted_upstream_pin": {"upstream": "synthetic"},
                "source_sha256": {"modeling_qwen3_5_moe.py": "0" * 64,
                                  "reap_observe.py": "0" * 64},
            },
            "source_identity_sha256": "f" * 64,
        },
        "calibration_ppl": 15.0,
    }
    (observation / "observation.json").write_text(json.dumps(recorded))
    monkeypatch.setattr(reap_hidden_checkpoint, "validate_observation_receipt",
                        lambda *_args, **_kwargs: None)
    monkeypatch.setattr(parity, "validate_registered_source_metadata",
                        lambda *_args, **_kwargs: {"upstream": "synthetic"})
    with pytest.raises(ValueError, match="HF decoder or observer hook changed"):
        parity.original_provenance(model=model, weights=model,
                                   observation=observation, calibration=frozen)


def test_numeric_comparison_is_descriptive_and_never_invents_tolerances():
    torch = pytest.importorskip("torch")
    a = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    assert parity.tensor_difference(a, a)["exact_equal"] is True
    assert parity.tensor_difference(a, a)["different_elements"] == 0
    b = torch.tensor([1.0, 2.015625], dtype=torch.bfloat16)
    result = parity.tensor_difference(a, b)
    assert result["exact_equal"] is False
    assert result["different_elements"] == 1
    assert result["max_abs_diff"] > 0
    with pytest.raises(ValueError, match="non-finite"):
        parity.tensor_difference(a, torch.tensor([1.0, float("nan")], dtype=torch.bfloat16))


@pytest.mark.parametrize("block_type", ["linear_attention", "full_attention"])
def test_tiny_real_hf_decoder_reference_vs_observer_without_cuda(block_type):
    torch = pytest.importorskip("torch")
    M = pytest.importorskip("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe")
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
        Qwen3_5MoeTextConfig,
    )

    config = Qwen3_5MoeTextConfig(
        vocab_size=47, hidden_size=32, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=2,
        moe_intermediate_size=8, shared_expert_intermediate_size=8,
        num_experts=4, num_experts_per_tok=2, layer_types=[block_type],
    )
    config._attn_implementation = "sdpa"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261025)
        layer = M.Qwen3_5MoeTextModel(config).layers[0].to(torch.bfloat16).eval()
        hidden = torch.randn(1, 6, 32).to(torch.bfloat16)
    outputs = parity.compare_decoder_layer(
        layer=layer, reference=hidden, observed=hidden.clone(),
        config=config, hf_modeling=M, layer_id=0,
    )
    ref, obs, evidence = outputs
    torch.testing.assert_close(ref, obs, rtol=0, atol=0)
    assert evidence["activation_difference"]["exact_equal"]
    assert evidence["routed_tokens"] == 12
    assert evidence["layer"] == 0
    assert evidence["block_type"] == block_type
    assert "forward" not in vars(layer.mlp.experts)  # unmodified HF reference restored
