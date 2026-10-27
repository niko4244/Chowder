"""Source-only RED/GREEN tests: never load original weights, GPU or model code."""

import hashlib
import json
from pathlib import Path

import pytest

from chowder_batch import reap_bf16_cuda_all40_research_20261027 as full


def sample_receipt(model: Path, layer: int) -> bytes:
    return json.dumps({
        "schema": full.LAYER_SCHEMA,
        "status": "SOURCE_BOUND_SINGLE_LAYER_BF16_PARITY_RESEARCH_ONLY",
        "original_model": str(model.resolve()), "original_layer": layer,
        "num_layers_in_original": 40, "input_shape": [1, 2, 2048],
        "input_dtype": "torch.bfloat16", "fixed_input": "sin(arange*0.03125)*0.5",
        "gpu_device": "NVIDIA GeForce RTX 5060 Ti",
        "original_windows_native_gguf_gate_eligible": False,
        "entire_40_layer_logit_parity_proven": False,
        "quality_gate": "UNKNOWN", "gen0_status": "BLOCKED",
        "chosen": None, "release_approved": False,
        "source_verification": {
            "original_frozen_full_source_proof_sha256": full.FROZEN_PROOF_SHA256,
            "unread_shards_rehashed_this_run": False,
            "verified_live_selected_shard_sha256": {"model-00003-of-00026.safetensors": "a" * 64}},
        "physical_gpu_pre": {"index": 0, "free_mib": 14000,
                             "name": "NVIDIA GeForce RTX 5060 Ti", "total_mib": 16311},
        "physical_gpu_post": {"index": 0, "free_mib": 13500,
                              "name": "NVIDIA GeForce RTX 5060 Ti", "total_mib": 16311},
        "parity": {"elements": 4096, "exact_equal_elements": 2800,
                   "max_abs_error": 0.015625, "mean_abs_error": 0.0001,
                   "rmse": 0.001, "max_relative_error_eps1e-3": 1.0},
    }, sort_keys=True).encode()


def test_original_layer_source_only_receipt_is_not_release_evidence(tmp_path):
    row = full.verify_layer_receipt(sample_receipt(tmp_path, 9), model=tmp_path, layer=9)
    assert row["layer"] == 9
    assert row["exact_equal_elements"] == 2800
    assert row["host_gpu_free_pre_mib"] == 14000


@pytest.mark.parametrize("mutator", [
    lambda r: r.update(status="QUALIFIED_ALL40_RELEASE_PASS"),
    lambda r: r.update(original_layer=10),
    lambda r: r.update(num_layers_in_original=39),
    lambda r: r.update(original_windows_native_gguf_gate_eligible=True),
    lambda r: r.update(entire_40_layer_logit_parity_proven=True),
    lambda r: r.update(quality_gate="PASS"),
    lambda r: r.update(gen0_status="PASS"),
    lambda r: r.update(release_approved=True),
    lambda r: r["physical_gpu_pre"].update(free_mib=8191),
    lambda r: r["physical_gpu_post"].update(name="NVIDIA Tesla T4"),
    lambda r: r["source_verification"].update(
        original_frozen_full_source_proof_sha256="x" * 64),
    lambda r: r["source_verification"].update(unread_shards_rehashed_this_run=True),
    lambda r: r["parity"].update(exact_equal_elements=4097),
    lambda r: r["parity"].update(max_abs_error=float("inf")),
])
def test_source_proof_gate_adversarial_mutation_is_refused(tmp_path, mutator):
    payload = json.loads(sample_receipt(tmp_path, 9))
    mutator(payload)
    with pytest.raises(ValueError):
        full.verify_layer_receipt(json.dumps(payload).encode(), model=tmp_path, layer=9)


def test_duplicate_keys_nonfinite_and_missing_layers_refuse(tmp_path):
    with pytest.raises(ValueError, match="repeated"):
        full.verify_layer_receipt(b'{"schema":"x","schema":"y"}', model=tmp_path, layer=9)
    with pytest.raises(ValueError):
        full.verify_layer_receipt(b'{"v":NaN}', model=tmp_path, layer=9)
    with pytest.raises(ValueError):
        full.verify_layer_receipt(sample_receipt(tmp_path, 9), model=tmp_path, layer=40)
    path = tmp_path / "9.json"
    path.write_bytes(sample_receipt(tmp_path, 9))
    got, digest = full._receipt(tmp_path, 9, path)
    assert got["layer"] == 9
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
