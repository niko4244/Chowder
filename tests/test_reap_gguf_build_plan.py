"""GGUF conversion must be blocked without verified BF16 lineage."""

from __future__ import annotations

import json

import pytest

from chowder_batch.reap_gguf_build_plan import plan


def _call(tmp_path, receipt):
    model = tmp_path / "candidate"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({
        "text_config": {"num_hidden_layers": 40, "num_experts": 128},
    }))
    (model / "model.safetensors.index.json").write_text("{}")
    (model / "reap-prune.json").write_text("{}")
    proof = tmp_path / "verification.json"
    proof.write_text(json.dumps(receipt))
    checkout = tmp_path / "llama"
    checkout.mkdir()
    (checkout / "convert_hf_to_gguf.py").write_text("x")
    quantizer = tmp_path / "quant.exe"
    quantizer.write_bytes(b"stub")
    return plan(
        candidate=model, verification=proof,
        converter=checkout / "convert_hf_to_gguf.py",
        llama_checkout=checkout, quantizer=quantizer,
        bf16_out=tmp_path / "new-bf16.gguf",
        q8_out=tmp_path / "new-q8.gguf",
        q4_out=tmp_path / "new-q4.gguf",
    )


def test_gguf_plan_refuses_unsigned_model_even_when_directory_exists(tmp_path):
    with pytest.raises(ValueError, match="full source-exact"):
        _call(tmp_path, {"schema": "chowder.reap128-v2-pruned-bf16-verification.v1",
                         "status": "HEADER_AND_EXPERT_SAMPLE_PASS"})


def test_gguf_plan_requires_all_26_shards_verified(tmp_path):
    with pytest.raises(ValueError, match="full source-exact"):
        _call(tmp_path, {"schema": "chowder.reap128-v2-pruned-bf16-verification.v1",
                         "status": "FULL_SOURCE_TENSOR_EQUIVALENCE_PASS",
                         "candidate": str(tmp_path / "candidate"),
                         "candidate_shard_sha256": {"only_one": "0" * 64}})
