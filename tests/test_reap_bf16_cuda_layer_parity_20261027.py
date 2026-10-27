"""CPU-only synthetic RED/GREEN checks. No models, CUDA or services are used."""

import hashlib
import json

import pytest
import torch

from chowder_batch import reap_bf16_cuda_layer_parity_20261027 as parity


def test_identical_cpu_bf16_output_is_exact():
    value = torch.arange(24, dtype=torch.float32).to(torch.bfloat16).view(1, 3, 8)
    result = parity.compare_finite(value, value.clone())
    assert result["elements"] == result["exact_equal_elements"] == 24
    assert result["max_abs_error"] == result["rmse"] == 0


def test_different_outputs_produce_real_errors():
    a = torch.tensor([1, 2, 3], dtype=torch.bfloat16)
    b = torch.tensor([1, 4, 3], dtype=torch.bfloat16)
    result = parity.compare_finite(a, b)
    assert result["exact_equal_elements"] == 2
    assert result["max_abs_error"] == 2
    assert result["mean_abs_error"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("mutant", [
    lambda a: torch.ones(4, dtype=torch.bfloat16),
    lambda a: a.float(),
    lambda a: torch.tensor([1.0, float("nan")], dtype=torch.bfloat16),
    lambda a: torch.tensor([float("inf"), 1.0], dtype=torch.bfloat16),
    lambda a: torch.empty(0, dtype=torch.bfloat16),
])
def test_output_wrong_shape_dtype_nonfinite_or_empty_is_refused(mutant):
    source = torch.ones(2, dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        parity.compare_finite(source, mutant(source))


def test_forged_model_geometry_refused_without_reading_shards(tmp_path):
    parent = tmp_path / "pruned"
    parent.mkdir()
    (parent / "config.json").write_text(json.dumps({"text_config": {
        "num_hidden_layers": 40, "num_experts": 128,
        "num_experts_per_tok": 8, "hidden_size": 2048}}))
    (parent / "reap-prune.json").write_text(json.dumps({"method": "reap", "keep": 128, "of": 256}))
    (parent / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"model.language_model.layers.3.layer": "model-00001-of-00026.safetensors"}}))
    with pytest.raises(ValueError, match="missing or unsafe"):
        parity.verify_model_metadata(parent, 3)
    (parent / "model-00001-of-00026.safetensors").write_bytes(b"fixed benign bytes")
    cfg, sha, inventory = parity.verify_model_metadata(parent, 3)
    assert cfg["num_experts"] == 128
    assert len(sha["config.json"]) == 64
    assert inventory["tensor_keys"] == 1
    (parent / "reap-prune.json").write_text(json.dumps({"method": "reap", "keep": 160, "of": 256}))
    with pytest.raises(ValueError, match="40/128/top8"):
        parity.verify_model_metadata(parent, 3)


@pytest.mark.parametrize("layer", [-1, 40, True, 3.0])
def test_nonoriginal_layer_index_refused(tmp_path, layer):
    with pytest.raises(ValueError, match="40 decoder layers"):
        parity.verify_model_metadata(tmp_path, layer)


def test_source_pinned_selected_shard_bytes_refuse_resealed_forgery(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors.index.json").write_text("{}")
    (model / "reap-prune.json").write_text("{}")
    shard = "model-00003-of-00026.safetensors"
    (model / shard).write_bytes(b"original immutable test payload")
    metadata = {
        "model.safetensors.index.json": parity.digest_file(model / "model.safetensors.index.json"),
        "reap-prune.json": parity.digest_file(model / "reap-prune.json"),
    }
    names = {f"model-{i:05d}-of-00026.safetensors": "1" * 64
             for i in range(1, 27)}
    names[shard] = parity.digest_file(model / shard)
    frozen = {"schema": "chowder.reap128-v2-pruned-bf16-verification.v1",
              "status": "FULL_SOURCE_TENSOR_EQUIVALENCE_PASS",
              "candidate": str(model), "shards": 26,
              "candidate_shard_sha256": names,
              "candidate_index_sha256": metadata["model.safetensors.index.json"],
              "candidate_prune_receipt_sha256": metadata["reap-prune.json"],
              "original_source": "frozen source", "selection_sha256": "a" * 64,
              "original_observation_sha256": "b" * 64}
    file = tmp_path / "frozen.json"
    file.write_text(json.dumps(frozen))
    monkeypatch.setattr(parity, "FROZEN_PROOF_PATH", file)
    monkeypatch.setattr(parity, "FROZEN_PROOF_SHA256", parity.digest_file(file))
    inv = {"read_shard_names": [shard]}
    result = parity.verify_frozen_layer_ancestry(model, metadata, inv)
    assert result["verified_live_selected_shard_sha256"][shard] == names[shard]
    assert result["unread_shards_rehashed_this_run"] is False
    (model / shard).write_bytes(b"forged bytes, same source filename")
    with pytest.raises(ValueError, match="SHA drift"):
        parity.verify_frozen_layer_ancestry(model, metadata, inv)
    (model / shard).write_bytes(b"original immutable test payload")
    file.write_text(json.dumps({**frozen, "original_source": "attacker source"}))
    with pytest.raises(ValueError, match="git-frozen"):
        parity.verify_frozen_layer_ancestry(model, metadata, inv)


def test_frozen_proof_cannot_be_replaced_by_a_same_size_receipt(tmp_path, monkeypatch):
    file = tmp_path / "proof.json"
    file.write_bytes(b"{}")
    monkeypatch.setattr(parity, "FROZEN_PROOF_PATH", file)
    monkeypatch.setattr(parity, "FROZEN_PROOF_SHA256", hashlib.sha256(b"[]").hexdigest())
    with pytest.raises(ValueError, match="git-frozen"):
        parity.verify_frozen_layer_ancestry(tmp_path, {}, {})


def test_physical_host_gpu_census_requires_original_gpu0_and_real_headroom():
    raw = ("0, GPU-01234567-aaaa-bbbb-cccc-0123456789ab, "
           "NVIDIA GeForce RTX 5060 Ti, 16311, 5322, 10729\n")
    observed = parity.parse_physical_gpu_census(raw, "NVIDIA GeForce RTX 5060 Ti")
    assert observed["free_mib"] == 10729
    for fake in (
        raw.replace("10729", "8191"),
        raw.replace("RTX 5060 Ti", "RTX 2060"),
        raw.replace("0, GPU", "1, GPU"),
        raw.replace("GPU-", ""),
        raw.replace("16311", "12000"),
        raw.replace("5322", "20000"),
    ):
        with pytest.raises((ValueError, RuntimeError)):
            parity.parse_physical_gpu_census(fake, "NVIDIA GeForce RTX 5060 Ti")
    with pytest.raises(ValueError, match="ambiguous|malformed"):
        parity.parse_physical_gpu_census(raw + raw, "NVIDIA GeForce RTX 5060 Ti")
