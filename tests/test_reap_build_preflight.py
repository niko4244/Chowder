"""Synthetic CPU-only qualification of the immutable REAP pruning plan."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from chowder_batch import reap_build_preflight as preflight  # noqa: E402


@pytest.fixture(autouse=True)
def _synthetic_frozen_base_identity(monkeypatch):
    # The three-row fixture stands in for the frozen 704-row base ONLY inside
    # this test process. Production CLI has no pin override.
    rows = "".join(
        json.dumps({"source": "swe_smith_trajectories", "n_tokens": 256,
                    "text": f"synthetic calibration line {i}"}) + "\n"
        for i in range(3)
    )
    monkeypatch.setattr(preflight, "FROZEN_V2_BASE_ROWS", 3)
    monkeypatch.setattr(preflight, "FROZEN_V2_BASE_SHA256",
                        hashlib.sha256(rows.encode("utf-8")).hexdigest())


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


def test_self_consistent_unregistered_v2_is_not_frozen_identity(tmp_path, monkeypatch):
    paths = _fixture(tmp_path)
    # Even a perfectly consistent observer is inadmissible against the
    # production frozen 704-row SHA, with no production override.
    monkeypatch.setattr(preflight, "FROZEN_V2_BASE_ROWS", 704)
    monkeypatch.setattr(preflight, "FROZEN_V2_BASE_SHA256",
                        "f6a09b694ae43a6d8b44b5180a861856501295eabf3d5297eb9749a008cdd929")
    with pytest.raises(preflight.PreflightError, match="frozen v2 base"):
        _plan(paths)


def _append_ornith(paths: dict[str, Path], n: int = 2) -> Path:
    admitted = paths["calibration"] / "admitted.jsonl"
    admitted.write_text('{"admitted":true,"lane":"sft","prompt":"p","expected_answer":"42"}\n')
    rows_path = paths["calibration"] / "calibration.jsonl"
    with rows_path.open("a", encoding="utf-8", newline="\n") as stream:
        for i in range(n):
            stream.write(json.dumps({"source": "ornith_curriculum", "n_tokens": 256,
                                     "text": f"synthetic bare Ornith row {i}"}) + "\n")
    manifest_path = paths["calibration"] / "calibration-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sources"]["ornith_curriculum"] = {"kept": n}
    manifest["ornith"] = {"render": "bare", "path": str(admitted),
                          "sha256": hashlib.sha256(admitted.read_bytes()).hexdigest()}
    manifest["rows"] += n
    manifest["tokens"] += 256 * n
    manifest["calibration_sha256"] = hashlib.sha256(rows_path.read_bytes()).hexdigest()
    _save(manifest_path, manifest)
    observer_path = paths["observations"] / "observation.json"
    observer = json.loads(observer_path.read_text(encoding="utf-8"))
    observer["n_seqs"] += n
    observer["n_tokens"] += 256 * n
    observer["calibration_sha256"] = manifest["calibration_sha256"]
    _save(observer_path, observer)
    return admitted


@pytest.mark.parametrize("added_rows", [0, 2])
def test_registered_base_accepts_optional_bare_admission_suffix(tmp_path, added_rows):
    paths = _fixture(tmp_path)
    admitted = _append_ornith(paths, n=added_rows)
    plan = _plan(paths)
    assert plan["calibration"]["frozen_base_rows"] == 3
    assert plan["calibration"]["ornith_suffix_rows"] == added_rows
    assert admitted.exists() and not paths["out"].exists()


@pytest.mark.parametrize("damage", ["forged_base", "forged_suffix", "bad_admission_sha",
                                    "overquota"])
def test_registered_base_refuses_unattested_changes_even_with_selfconsistent_manifest(
    tmp_path, damage
):
    paths = _fixture(tmp_path)
    _append_ornith(paths, n=65 if damage == "overquota" else 2)
    rows_path = paths["calibration"] / "calibration.jsonl"
    manifest_path = paths["calibration"] / "calibration-manifest.json"
    observer_path = paths["observations"] / "observation.json"
    manifest = json.loads(manifest_path.read_text())
    if damage == "bad_admission_sha":
        manifest["ornith"]["sha256"] = "0" * 64
    elif damage != "overquota":
        rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
        if damage == "forged_base":
            rows[0]["text"] = "replaced original frozen corpus"
        else:
            rows[-1]["source"] = "swe_smith_trajectories"
            manifest["sources"]["swe_smith_trajectories"]["kept"] += 1
            manifest["sources"]["ornith_curriculum"]["kept"] -= 1
        rows_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        manifest["calibration_sha256"] = hashlib.sha256(rows_path.read_bytes()).hexdigest()
        observed = json.loads(observer_path.read_text())
        observed["calibration_sha256"] = manifest["calibration_sha256"]
        _save(observer_path, observed)
    _save(manifest_path, manifest)
    with pytest.raises(preflight.PreflightError, match="frozen v2"):
        _plan(paths)


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


def test_production_cli_refuses_synthetic_base_without_writing_outputs_or_weights(tmp_path):
    paths = _fixture(tmp_path)
    script = Path(preflight.__file__)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    # Exercise the documented script entry point, not the import-friendly
    # test runner's current sys.path or an inherited PYTHONPATH.
    command = [sys.executable, "-B", "chowder_batch/reap_build_preflight.py", "--arm", "reap128_v2",
               "--model-dir", str(paths["model"]), "--weights-dir", str(paths["weights"]),
               "--calibration", str(paths["calibration"]),
               "--observations", str(paths["observations"]), "--out", str(paths["out"])]
    first = subprocess.run(command, cwd=script.parents[1], env=environment,
                           capture_output=True, text=True, check=False)
    second = subprocess.run(command, cwd=script.parents[1], env=environment,
                            capture_output=True, text=True, check=False)
    assert first.returncode == second.returncode == 2
    assert first.stdout == second.stdout == ""
    assert "frozen v2 base identity mismatch" in first.stderr
    assert not paths["out"].exists()
    assert (paths["weights"] / "model-00001-of-00001.safetensors").read_bytes().startswith(b"metadata-only")


@pytest.mark.parametrize("entry", [
    ["chowder_batch/reap_build_preflight.py"],
    ["-m", "chowder_batch.reap_build_preflight"],
])
def test_preflight_direct_script_and_module_help_without_pythonpath(entry):
    repo = Path(preflight.__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, "-B", *entry, "--help"], cwd=repo, env=environment,
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--observations" in completed.stdout
    assert "--out" in completed.stdout


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
    # Production has the real 704-row pin, so this deliberately synthetic
    # corpus is refused before the missing saliency layer is even inspected.
    assert "frozen v2 base identity mismatch" in completed.stderr
    assert completed.stdout == "" and not paths["out"].exists()
