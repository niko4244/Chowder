"""Bounds on real CPU-only model loading, without opening production weights."""

from __future__ import annotations

from pathlib import Path

import pytest

from chowder_batch import reap_streaming_bf16_smoke as smoke


def test_empty_smoke_prompt_rejected_before_model_loading(tmp_path):
    with pytest.raises(ValueError, match="cannot be empty"):
        smoke.run(tmp_path / "missing-model", prompt="")


def test_non_cpu_visible_device_rejected_before_model_loading(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="CPU-only"):
        smoke.run(tmp_path / "missing-model", prompt="Hello")


def test_missing_model_does_not_publish_false_success(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    receipt = tmp_path / "never-publish.json"
    result = smoke.main([
        "--model", str(Path(tmp_path / "missing")),
        "--prompt", "The answer is",
        "--out", str(receipt),
    ])
    assert result != 0
    assert not receipt.exists()


def test_existing_receipt_cannot_be_overwritten(tmp_path):
    receipt = tmp_path / "owner.json"
    receipt.write_text("owner evidence")
    with pytest.raises(SystemExit):
        smoke.main(["--model", str(tmp_path), "--out", str(receipt)])
    assert receipt.read_text() == "owner evidence"


def test_40_layer_cpu_streaming_inference_loads_sharded_synthetic_checkpoint(
    tmp_path, monkeypatch,
):
    """End-to-end positive execution with tiny BF16 tensors, not original weights."""
    import json

    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file
    from transformers import AutoTokenizer
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as M

    cfg = Qwen3_5MoeTextConfig(
        vocab_size=67, hidden_size=32, num_hidden_layers=40,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=2,
        moe_intermediate_size=8, shared_expert_intermediate_size=8,
        num_experts=128, num_experts_per_tok=8,
        layer_types=["full_attention" if i % 4 == 3 else "linear_attention"
                     for i in range(40)],
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261025)
        reference = M.Qwen3_5MoeTextModel(cfg).to(torch.bfloat16).eval()
        weights = {"model.language_model." + key: value.detach().contiguous()
                   for key, value in reference.state_dict().items()}
        weights["lm_head.weight"] = torch.randn(
            cfg.vocab_size, cfg.hidden_size, dtype=torch.bfloat16,
        )
    del reference
    model = tmp_path / "synthetic"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"text_config": cfg.to_dict()}))
    shard = "model-00001-of-00001.safetensors"
    save_file(weights, str(model / shard))
    (model / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {key: shard for key in weights},
    }))
    del weights

    class TinyTokenizer:
        def __call__(self, prompt, add_special_tokens=False):
            assert not add_special_tokens
            return {"input_ids": [3, 4, 5, 6]}

        def decode(self, tokens):
            return f"tiny-token-{tokens[0]}"

    monkeypatch.setattr(
        AutoTokenizer, "from_pretrained", lambda *args, **kwargs: TinyTokenizer(),
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    progress = tmp_path / "owned-progress"
    progress.mkdir()
    answer = smoke.run(model, prompt="synthetic test", max_tokens=8,
                       progress_dir=progress, full_verification_sha256="a" * 64)
    assert answer["status"] == "CPU_BF16_END_TO_END_SMOKE_PASS_NOT_QUALITY_GATE"
    assert answer["layer_count"] == 40
    assert answer["layer_types"] == {"linear_attention": 30, "full_attention": 10}
    assert len(answer["final_logits_sha256"]) == 64
    assert len(answer["top_5_token_ids"]) == 5
    assert answer["resumed_after_layer"] == -1
    assert len(list(progress.glob("layer-*.receipt.json"))) == 4
    resumed = smoke.run(model, prompt="synthetic test", max_tokens=8,
                        progress_dir=progress, full_verification_sha256="a" * 64)
    assert resumed["resumed_after_layer"] == 29
    assert resumed["final_logits_sha256"] == answer["final_logits_sha256"]
    with pytest.raises(ValueError, match="stale/corrupt"):
        smoke.run(model, prompt="synthetic test", max_tokens=8,
                  progress_dir=progress, full_verification_sha256="b" * 64)
