"""CPU-only REAP128_v2 ranking, provenance, and mapping checks."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from chowder_batch import reap_rank_experts as rank  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, monkeypatch) -> tuple[Path, Path, Path, Path, Path]:
    model, obs, calib = (tmp_path / name for name in ("model", "observations", "calibration"))
    for d in (model, obs, calib):
        d.mkdir()
    (model / "config.json").write_text('{"text_config":{"num_hidden_layers":40}}')
    (model / "model.safetensors.index.json").write_text('{"weight_map":{}}')
    (obs / "observation.json").write_text(
        json.dumps({"n_seqs": 704, "calibration_sha256": "calibration-frozen"})
    )
    for i in range(40):
        counts = torch.ones(256, dtype=torch.float64)
        # Deliberate ties at the boundary to verify expert-id stable sorting.
        reap = torch.tensor([float(j // 2) for j in range(256)], dtype=torch.float64)
        torch.save({"count": counts, "reap_sum": reap, "gate_sum": counts,
                    "norm_sum": reap}, obs / f"layer-{i:02d}.pt")

    def preflight(**kwargs):
        assert kwargs["arm"] == "reap128_v2"
        return {
            "calibration": {"observer_sha256": _sha(obs / "observation.json"),
                            "calibration_sha256": "calibration-frozen"},
            "model": {"directory": str(model.resolve()), "weights_directory": str(model.resolve()),
                      "config_sha256": _sha(model / "config.json"),
                      "index_sha256": _sha(model / "model.safetensors.index.json")},
            "saliency_layer_digest_sha256": "synthetic-digest",
            "output_directory_intended_only": str(tmp_path / "future-pruned"),
            "frozen_order_predecessors": ["reap128"],
            "qualification": "NOT_EVALUATED",
        }

    monkeypatch.setattr(rank, "build_plan", preflight)
    return model, obs, calib, tmp_path / "future-pruned", tmp_path / "selection.json"


def _generate(paths):
    model, obs, calib, out, _ = paths
    return rank.generate(model_dir=model, weights_dir=model, calibration=calib,
                         observations=obs, prune_out=out)


def _validate(paths, document):
    model, obs, _, _, _ = paths
    kept = rank.kept_experts(obs, 40, 256, 128, "reap", 20261017)
    rank.validate_selection(document, model_dir=model, weights_dir=model,
                            observations=obs, kept=kept)


def test_40x128_selection_is_deterministic_and_mapped(tmp_path, monkeypatch):
    paths = _fixture(tmp_path, monkeypatch)
    first, again = _generate(paths), _generate(paths)
    assert first == again
    assert first["status"] == "EXPERTS_SELECTED_NOT_BUILT_OR_QUALIFIED"
    assert set(first["layers"]) == {str(i) for i in range(40)}
    for info in first["layers"].values():
        expected = list(range(128, 256))
        assert info["retained_source_experts"] == expected
        assert info["ranked_source_experts"][:4] == [254, 255, 252, 253]
        assert info["ranked_source_experts"][126:130] == [128, 129, 126, 127]
        assert len(info["destination_to_source_expert"]) == 128
        assert info["source_to_destination_expert"] == [-1] * 128 + list(range(128))
        assert info["routed_token_assignments"] == 256
        assert info["experts_used"] == 256
        assert info["score_boundary_margin"] == 1.0
    _validate(paths, first)


@pytest.mark.parametrize("damage", ["rank", "kept", "mapping", "stats", "observer",
                                    "config", "missing_layer", "path"])
def test_selection_rejects_mismatched_provenance_or_mapping(tmp_path, monkeypatch, damage):
    paths = _fixture(tmp_path, monkeypatch)
    selection = _generate(paths)
    model, obs, _, _, _ = paths
    if damage == "rank":
        selection["layers"]["0"]["ranked_source_experts"].reverse()
    elif damage == "kept":
        selection["layers"]["0"]["retained_source_experts"][0] = 0
    elif damage == "mapping":
        selection["layers"]["0"]["source_to_destination_expert"][128] = 2
    elif damage == "stats":
        with (obs / "layer-00.pt").open("ab") as f:
            f.write(b"changed after selection")
    elif damage == "observer":
        (obs / "observation.json").write_text('{"n_seqs":704,"calibration_sha256":"other"}')
    elif damage == "config":
        (model / "config.json").write_text('{"text_config":{"num_hidden_layers":39}}')
    elif damage == "missing_layer":
        selection["layers"].pop("39")
    else:
        selection["source_model_dir"] = str(tmp_path / "other-model")
    with pytest.raises((ValueError, OSError)):
        _validate(paths, selection)


@pytest.mark.parametrize("entry", [["-m", "chowder_batch.reap_prune"],
                                    ["chowder_batch/reap_prune.py"]])
def test_direct_pruner_cannot_consume_40x256_unsealed_stats(tmp_path, entry):
    model, obs = tmp_path / "model", tmp_path / "obs"
    model.mkdir()
    obs.mkdir()
    (model / "config.json").write_text(json.dumps({"text_config": {
        "num_hidden_layers": 40, "num_experts": 256, "num_experts_per_tok": 8
    }}))
    cmd = [sys.executable, "-B", *entry,
           "--model-dir", str(model), "--observations", str(obs), "--keep", "128",
           "--method", "reap", "--out", str(tmp_path / "out")]
    root = Path(__file__).resolve().parents[1]
    missing = subprocess.run(cmd, cwd=root, capture_output=True, text=True, check=False)
    assert missing.returncode and "requires observation.json" in missing.stderr
    (obs / "observation.json").write_text('{"n_seqs":704}')
    unsealed = subprocess.run(cmd, cwd=root, capture_output=True, text=True, check=False)
    assert unsealed.returncode and "registered BF16 D0 requires checkpointed" in unsealed.stderr
    (obs / "observation.json").write_text('{"n_seqs":384,"calibration_sha256":"forged"}')
    unregistered_v1 = subprocess.run(cmd, cwd=root, capture_output=True, text=True, check=False)
    assert unregistered_v1.returncode and "wrong frozen calibration" in unregistered_v1.stderr
    assert not (tmp_path / "out").exists()
