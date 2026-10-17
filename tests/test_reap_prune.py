"""reap_prune keeps the top-K experts by REAP score and keeps router rows aligned."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
E, H, I, LAYERS, TOPK = 8, 4, 3, 2, 2


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    model = tmp_path / "model"
    model.mkdir()
    tensors = {}
    for layer in range(LAYERS):
        p = f"model.language_model.layers.{layer}.mlp."
        # Row e of every expert tensor is filled with e, so kept rows are readable.
        tensors[p + "experts.gate_up_proj"] = torch.arange(E, dtype=torch.float32).view(E, 1, 1).expand(E, 2 * I, H).contiguous()
        tensors[p + "experts.down_proj"] = torch.arange(E, dtype=torch.float32).view(E, 1, 1).expand(E, H, I).contiguous()
        tensors[p + "gate.weight"] = torch.arange(E, dtype=torch.float32).view(E, 1).expand(E, H).contiguous()
        tensors[p + "shared_expert_gate.weight"] = torch.ones(1, H)
    tensors["mtp.fc.weight"] = torch.ones(H, 2 * H)
    tensors["lm_head.weight"] = torch.ones(5, H)
    save_file(tensors, str(model / "model-00001-of-00001.safetensors"))
    (model / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {k: "model-00001-of-00001.safetensors" for k in tensors}}), encoding="utf-8")
    (model / "config.json").write_text(json.dumps({"text_config": {
        "num_hidden_layers": LAYERS, "num_experts": E, "num_experts_per_tok": TOPK, "mtp_num_hidden_layers": 1}}),
        encoding="utf-8")
    obs = tmp_path / "obs"
    obs.mkdir()
    for layer in range(LAYERS):
        count = torch.full((E,), 10.0, dtype=torch.float64)
        # Layer 0 prefers high indices, layer 1 low indices; expert 5 never routed in layer 1.
        reap = torch.arange(E, dtype=torch.float64) * 10 if layer == 0 else (E - torch.arange(E, dtype=torch.float64)) * 10
        if layer == 1:
            count[0] = 0.0
            reap[0] = 0.0
        torch.save({"count": count, "reap_sum": reap, "gate_sum": count, "norm_sum": count}, obs / f"layer-{layer:02d}.pt")
    return model, obs


def _run(model: Path, obs: Path, out: Path, method: str = "reap", keep: int = 4) -> None:
    subprocess.run([sys.executable, str(ROOT / "chowder_batch" / "reap_prune.py"), "--model-dir", str(model),
                    "--observations", str(obs), "--keep", str(keep), "--method", method, "--out", str(out)],
                   check=True, capture_output=True)


def test_reap_keeps_top_k_by_score_with_aligned_router_rows(tmp_path: Path) -> None:
    model, obs = _fixture(tmp_path)
    out = tmp_path / "out"
    _run(model, obs, out)
    pruned = load_file(str(out / "model-00001-of-00001.safetensors"))
    record = json.loads((out / "reap-prune.json").read_text(encoding="utf-8"))
    assert record["kept_experts"]["0"] == [4, 5, 6, 7]
    # Layer 1 scores fall with index, and expert 0 was never routed, so it is dropped first.
    assert record["kept_experts"]["1"] == [1, 2, 3, 4]
    for layer, expected in ((0, [4, 5, 6, 7]), (1, [1, 2, 3, 4])):
        p = f"model.language_model.layers.{layer}.mlp."
        assert pruned[p + "experts.gate_up_proj"].shape == (4, 2 * I, H)
        assert pruned[p + "experts.down_proj"].shape == (4, H, I)
        for name in ("experts.gate_up_proj", "experts.down_proj", "gate.weight"):
            rows = pruned[p + name].reshape(4, -1)[:, 0].tolist()
            assert rows == expected, name
        assert torch.equal(pruned[p + "shared_expert_gate.weight"], torch.ones(1, H))
    assert not any(k.startswith("mtp.") for k in pruned)
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))["text_config"]
    assert config["num_experts"] == 4 and config["mtp_num_hidden_layers"] == 0
    index = json.loads((out / "model.safetensors.index.json").read_text(encoding="utf-8"))
    assert set(index["weight_map"]) == set(pruned)


def test_random_control_keeps_the_same_count_per_layer_and_is_seeded(tmp_path: Path) -> None:
    model, obs = _fixture(tmp_path)
    _run(model, obs, tmp_path / "r1", method="random")
    _run(model, obs, tmp_path / "r2", method="random")
    a = json.loads((tmp_path / "r1" / "reap-prune.json").read_text(encoding="utf-8"))["kept_experts"]
    b = json.loads((tmp_path / "r2" / "reap-prune.json").read_text(encoding="utf-8"))["kept_experts"]
    assert a == b
    assert all(len(v) == 4 for v in a.values())
