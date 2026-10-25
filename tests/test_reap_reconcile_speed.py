"""No-network immutable probe reconciliation; NEVER upgrades frozen gates."""

from __future__ import annotations

import hashlib
import json

import pytest

from chowder_batch.reap_reconcile_speed import FROZEN_FLAGS, reconcile


def _make(tmp_path):
    arm = tmp_path / "reap160_q4"
    baseline = tmp_path / "dense_q4"
    arm.mkdir()
    baseline.mkdir()
    for folder, tps in ((arm, 40.0), (baseline, 10.0)):
        rows = [
            {"id": str(i), "correct": True, "terminated": True,
             "trigram_ratio": 0.95, "predicted_per_second": tps}
            for i in range(200)
        ]
        (folder / "gsm8k.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
    gguf = "F:/test-reap160-q4.gguf"
    (arm / "serve.json").write_text(json.dumps({
        "gguf": gguf, "gguf_sha256": "a" * 64,
        "flags": FROZEN_FLAGS, "experts_offloaded": None,
    }), encoding="utf-8")
    raw = (
        "0.00 I common_fit_params: successfully fit params to free device memory\n"
        "0.01 I load_tensors: offloaded 41/41 layers to GPU\n"
    )
    (arm / "probe.log").write_text(raw, encoding="utf-8")
    (arm / "placement.json").write_text(json.dumps({
        "gguf": gguf, "flags": FROZEN_FLAGS, "log_verbosity": "4",
        "experts_offloaded": 0,
        "offload_log_lines": ["common_fit_params: successfully fit",
                              "load_tensors: offloaded 41/41 layers to GPU"],
    }), encoding="utf-8")
    return arm


def test_separate_probe_corroborates_but_cannot_promote(tmp_path):
    folder = _make(tmp_path)
    result = reconcile(tmp_path)
    assert result["status"] == "CORROBORATED_SEPARATE_LOAD_PROBE_NOT_CANONICAL_GATE"
    assert result["speed_ratio"] == 4.0
    assert result["speed_ratio_passes"] is True
    assert result["original_serve_offload_status"] is None
    assert result["separate_probe_all_gpu_layers"] == 41
    assert "remains UNKNOWN" in result["what_is_not_proven"]
    for path, value in result["input_sha256"].items():
        assert hashlib.sha256(__import__("pathlib").Path(path).read_bytes()).hexdigest() == value
    assert not (folder / "gates.json").exists()


@pytest.mark.parametrize("damage", ["missing_fit", "partial", "changed_flags", "wrong_model",
                                    "missing_speeds"])
def test_reconciliation_fails_closed_on_unbound_or_incomplete_evidence(tmp_path, damage):
    folder = _make(tmp_path)
    if damage == "missing_fit":
        path = folder / "probe.log"
        path.write_text(path.read_text().replace(
            "common_fit_params: successfully fit params to free device memory", "not fitted"))
    elif damage == "partial":
        path = folder / "probe.log"
        path.write_text(path.read_text().replace("41/41", "30/41"))
    elif damage == "changed_flags":
        path = folder / "placement.json"
        d = json.loads(path.read_text())
        d["flags"] = ["--fit", "off"]
        path.write_text(json.dumps(d))
    elif damage == "wrong_model":
        path = folder / "placement.json"
        d = json.loads(path.read_text())
        d["gguf"] = "F:/wrong-model.gguf"
        path.write_text(json.dumps(d))
    else:
        (folder / "gsm8k.jsonl").write_text("{}\n")
    from chowder_batch.reap_gates import GateError
    with pytest.raises((ValueError, KeyError, GateError)):
        reconcile(tmp_path)
