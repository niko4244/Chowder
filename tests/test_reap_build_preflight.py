"""Synthetic CPU-only qualification of the immutable REAP pruning plan."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from chowder_batch import reap_build_preflight as preflight  # noqa: E402


def _save(path: Path, document: object) -> None:
    path.write_text(json.dumps(document), encoding="utf-8")


def _fixture(tmp_path: Path, mix: str = "v2") -> dict[str, Path]:
    model = tmp_path / "model"
    weights = tmp_path / "weights"
    calibration = tmp_path / "calibration"
    observations = tmp_path / "observations"
    for directory in (model, weights, calibration, observations):
        directory.mkdir()
    _save(model / "config.json", {
        "model_type": "qwen3_5_moe",
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "num_hidden_layers": 40,
            "num_experts": 256,
            "num_experts_per_tok": 8,
        },
    })
    shard = "model-00001-of-00001.safetensors"
    (weights / shard).write_bytes(b"metadata-only; the preflight must not open these bytes")
    _save(model / "model.safetensors.index.json", {"weight_map": {
        f"model.language_model.layers.{i}.mlp.{key}": shard
        for i in range(40)
        for key in ("experts.gate_up_proj", "experts.down_proj", "gate.weight")
    }})
    count_rows = 3 if mix == "v2" else 384
    with (calibration / "calibration.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for i in range(count_rows):
            stream.write(json.dumps({"source": "swe_smith_trajectories", "n_tokens": 256,
                                     "text": f"synthetic calibration line {i}"}) + "\n")
    digest = hashlib.sha256((calibration / "calibration.jsonl").read_bytes()).hexdigest()
    manifest = {
        "rows": count_rows,
        "tokens": count_rows * 256 if mix == "v2" else 651861,
        "max_tokens": 2048,
        "calibration_sha256": digest,
        "sources": {"swe_smith_trajectories": {"kept": count_rows}},
    }
    if mix == "v2":
        manifest.update({"mix": "v2", "ornith": "absent: no admitted Ornith curriculum supplied"})
    _save(calibration / "calibration-manifest.json", manifest)
    layers = []
    for layer in range(40):
        counts = torch.zeros(256, dtype=torch.float64)
        counts[:8] = 1
        stats = {"count": counts, "gate_sum": counts * 0.1,
                 "norm_sum": counts * 0.5, "reap_sum": counts * 0.05}
        torch.save(stats, observations / f"layer-{layer:02d}.pt")
        layers.append({"layer": layer, "routed_tokens": 8, "block_type": "linear_attention"})
    _save(observations / "observation.json", {
        "layers": layers,
        "n_seqs": count_rows,
        "n_tokens": count_rows * 256,
        "calibration_sha256": digest,
        "calibration_ppl": 10.0,
    })
    return {"model": model, "weights": weights, "calibration": calibration,
            "observations": observations, "out": tmp_path / "future-pruned-model"}


def _plan(paths: dict[str, Path], arm: str = "reap128_v2") -> dict:
    return preflight.build_plan(
        arm=arm, model_dir=paths["model"], weights_dir=paths["weights"],
        calibration=paths["calibration"], observations=paths["observations"],
        out=paths["out"],
    )


@pytest.mark.parametrize(("arm", "keep", "predecessors"), [
    ("reap128_v2", 128, ["reap128"]),
    ("reap160_v2", 160, ["reap128", "reap128_v2", "reap160"]),
])
def test_valid_v2_plan_is_deterministic_and_never_builds(tmp_path, arm, keep, predecessors):
    paths = _fixture(tmp_path)
    plan = _plan(paths, arm)
    assert plan == _plan(paths, arm)
    assert plan["status"] == "PREREQUISITES_VALIDATED_NOT_BUILT"
    assert plan["keep_experts_per_layer"] == keep
    assert plan["frozen_order_predecessors"] == predecessors
    assert plan["required_calibration"] == "v2"
    assert plan["layers"] == 40 and plan["model_experts_per_layer"] == 256
    assert plan["prune_command_not_executed"][-6:] == [
        "--keep", str(keep), "--method", "reap", "--out", str(paths["out"].resolve()),
    ]
    assert not paths["out"].exists()
    assert plan["qualification"].startswith("NOT_EVALUATED")


def test_valid_v1_reap160_plan_uses_frozen_v1_calibration(tmp_path, monkeypatch):
    paths = _fixture(tmp_path, mix="v1")
    # The fixture is synthetic, so substitute only its preregistered digest;
    # production remains pinned to the real full v1 SHA256.
    monkeypatch.setattr(preflight, "FROZEN_V1_SHA256",
                        hashlib.sha256((paths["calibration"] / "calibration.jsonl").read_bytes()).hexdigest())
    plan = _plan(paths, "reap160")
    assert plan["keep_experts_per_layer"] == 160
    assert plan["required_calibration"] == "v1"
    assert plan["frozen_order_predecessors"] == ["reap128", "reap128_v2"]


def test_cli_emits_json_without_writing_output_or_model_shards(tmp_path):
    paths = _fixture(tmp_path)
    script = Path(preflight.__file__)
    command = [sys.executable, str(script), "--arm", "reap128_v2",
               "--model-dir", str(paths["model"]), "--weights-dir", str(paths["weights"]),
               "--calibration", str(paths["calibration"]),
               "--observations", str(paths["observations"]), "--out", str(paths["out"])]
    first = subprocess.run(command, cwd=script.parents[1], capture_output=True, text=True, check=False)
    second = subprocess.run(command, cwd=script.parents[1], capture_output=True, text=True, check=False)
    assert first.returncode == second.returncode == 0, first.stderr
    assert first.stdout == second.stdout
    assert json.loads(first.stdout)["keep_experts_per_layer"] == 128
    assert not paths["out"].exists()
    assert (paths["weights"] / "model-00001-of-00001.safetensors").read_bytes().startswith(b"metadata-only")


@pytest.mark.parametrize("damage", [
    "missing_d0", "limited_observer", "bad_ppl", "wrong_digest", "wrong_mix",
    "worked_ornith", "missing_stats", "truncated_stats", "nan_saliency",
    "corrupt_stats",
    "negative_count", "unrouted_saliency", "wrong_token_total", "pruned_model",
    "missing_router_key", "missing_shard", "output_exists", "output_inside_observer",
])
def test_preflight_refuses_incomplete_or_incompatible_inputs(tmp_path, damage):
    paths = _fixture(tmp_path)
    observation = paths["observations"] / "observation.json"
    manifest_file = paths["calibration"] / "calibration-manifest.json"
    config_file = paths["model"] / "config.json"
    index_file = paths["model"] / "model.safetensors.index.json"
    rows = json.loads(observation.read_text(encoding="utf-8"))
    if damage == "missing_d0":
        observation.unlink()
    elif damage == "limited_observer":
        rows["n_seqs"] = 2
        _save(observation, rows)
    elif damage == "bad_ppl":
        rows["calibration_ppl"] = 22.0
        _save(observation, rows)
    elif damage == "wrong_digest":
        rows["calibration_sha256"] = "0" * 64
        _save(observation, rows)
    elif damage in {"wrong_mix", "worked_ornith"}:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        if damage == "wrong_mix":
            manifest["mix"] = "v1"
        else:
            manifest["ornith"] = {"render": "worked"}
        _save(manifest_file, manifest)
    elif damage == "missing_stats":
        (paths["observations"] / "layer-37.pt").unlink()
    elif damage == "corrupt_stats":
        (paths["observations"] / "layer-20.pt").write_bytes(b"not a tensor archive")
    elif damage in {"truncated_stats", "nan_saliency", "negative_count", "unrouted_saliency"}:
        file = paths["observations"] / "layer-20.pt"
        stats = torch.load(file, weights_only=True)
        if damage == "truncated_stats":
            stats["reap_sum"] = torch.zeros(255)
        elif damage == "nan_saliency":
            stats["reap_sum"][0] = float("nan")
        elif damage == "negative_count":
            stats["count"][0] = -1
        else:
            stats["reap_sum"][200] = 9.0
        torch.save(stats, file)
    elif damage == "wrong_token_total":
        rows["layers"][20]["routed_tokens"] = 7
        _save(observation, rows)
    elif damage == "pruned_model":
        config = json.loads(config_file.read_text(encoding="utf-8"))
        config["text_config"]["num_experts"] = 160
        _save(config_file, config)
    elif damage == "missing_router_key":
        index = json.loads(index_file.read_text(encoding="utf-8"))
        index["weight_map"].pop("model.language_model.layers.19.mlp.gate.weight")
        _save(index_file, index)
    elif damage == "missing_shard":
        (paths["weights"] / "model-00001-of-00001.safetensors").unlink()
    elif damage == "output_exists":
        paths["out"].mkdir()
    else:
        paths["out"] = paths["observations"] / "future-checkpoint"
    with pytest.raises(preflight.PreflightError):
        _plan(paths)


def test_v1_requires_v1_and_v2_requires_v2(tmp_path):
    paths = _fixture(tmp_path)
    with pytest.raises(preflight.PreflightError, match="v1"):
        _plan(paths, "reap160")
    with pytest.raises(preflight.PreflightError, match="arm must be"):
        _plan(paths, "reap128")


def test_cli_refuses_without_writing_or_claiming_gate_pass(tmp_path):
    paths = _fixture(tmp_path)
    (paths["observations"] / "layer-00.pt").unlink()
    script = Path(preflight.__file__)
    completed = subprocess.run(
        [sys.executable, str(script), "--arm", "reap160_v2",
         "--calibration", str(paths["calibration"]), "--observations", str(paths["observations"]),
         "--model-dir", str(paths["model"]), "--weights-dir", str(paths["weights"]),
         "--out", str(paths["out"])], cwd=script.parents[1], capture_output=True, text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "layer-00.pt" in completed.stderr
    assert completed.stdout == "" and not paths["out"].exists()
