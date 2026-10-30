"""Resource contracts for an OWNED, isolated CPU-only BF16 prune launch."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chowder_batch import reap_prune_supervised as sup


def _state(ram=26.0, commit=32.0, disk=70.0):
    return sup.HostState(
        ram_available_gib=ram,
        commit_available_gib=commit,
        output_disk_free_gib=disk,
        process_count=600,
    )


def _sizing():
    return {
        "projected_pruned_tensor_bytes": 35 * sup.GIB,
        "projected_largest_output_shard_bytes": 3 * sup.GIB,
    }


def test_admit_requires_both_memory_types_and_disk():
    assert sup.admit(_state(), peak_gib=7, sizing=_sizing()) == ()
    assert any("physical" in reason for reason in sup.admit(
        _state(ram=14.99), peak_gib=7, sizing=_sizing()
    ))
    assert any("commit" in reason for reason in sup.admit(
        _state(commit=14.99), peak_gib=7, sizing=_sizing()
    ))
    assert any("disk" in reason for reason in sup.admit(
        _state(disk=42.99), peak_gib=7, sizing=_sizing()
    ))
    assert sup.admit(_state(ram=15, commit=15, disk=43), peak_gib=7, sizing=_sizing()) == ()


def test_runtime_abort_checks_independent_safety_floors():
    assert not sup.runtime_problems(_state())
    assert "physical RAM safety floor crossed" in sup.runtime_problems(_state(ram=6))
    assert "Windows commit safety floor crossed" in sup.runtime_problems(_state(commit=7))
    assert "disk safety floor crossed" in sup.runtime_problems(_state(disk=7))


def test_default_preflight_cannot_execute_and_execute_requires_new_log(
    tmp_path: Path, monkeypatch
):
    import chowder_batch.reap_original_bf16_pin as pin

    model = tmp_path / "model"
    calibration = tmp_path / "calibration"
    observation = tmp_path / "obs"
    out = tmp_path / "future-model"
    selection_dir = tmp_path / "selection"
    selection = selection_dir / "selection.json"
    for item in (model, calibration, observation, selection_dir):
        item.mkdir()
    selection.write_text("{}")
    plan = {
        "model": {"directory": str(model)},
        "selection_sha256": "selection-sha",
        "calibration": {"observer_sha256": "obs-sha"},
        "build_capacity": _sizing(),
        "prune_command_not_executed": ["python", "chowder_batch/reap_prune.py"],
    }
    monkeypatch.setattr(sup, "build_plan", lambda **kwargs: plan)
    monkeypatch.setattr(sup, "host_state", lambda path: _state())
    monkeypatch.setattr(pin, "load_original_pin", lambda: {
        "weight_shards": {"one": {"size": 2 * sup.GIB}},
    })
    called = []
    monkeypatch.setattr(sup, "_run_owned", lambda *args, **kwargs: called.append(1) or 0)
    args = [
        "--model-dir", str(model), "--weights-dir", str(model),
        "--calibration", str(calibration), "--observations", str(observation),
        "--selection", str(selection), "--out", str(out),
    ]
    assert sup.main(args) == 0
    assert not called
    with pytest.raises(SystemExit):
        sup.main([*args, "--execute"])
    assert not called
    log_path = tmp_path / "owned-job.log"
    assert sup.main([*args, "--execute", "--log", str(log_path)]) == 0
    assert called == [1]
    assert json.loads(log_path.read_text().splitlines()[0])["status"] == "ADMITTED_NOT_EXECUTED"
    assert not out.exists()
    # The same immutable log must never be reused.
    with pytest.raises(SystemExit):
        sup.main([*args, "--execute", "--log", str(log_path)])


def test_supervisor_fail_closed_when_commit_admission_fails(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(sup, "build_plan", lambda **kwargs: {
        "model": {}, "selection_sha256": "id",
        "calibration": {"observer_sha256": "id"},
        "build_capacity": _sizing(),
        "prune_command_not_executed": [],
    })
    monkeypatch.setattr(sup, "host_state", lambda path: _state(commit=3))
    monkeypatch.setattr(sup, "memory_required_gib", lambda sizing: 6.5)
    monkeypatch.setattr(sup, "_run_owned", lambda *args, **kwargs: pytest.fail("launched"))
    args = ["--model-dir", str(tmp_path), "--weights-dir", str(tmp_path),
            "--calibration", str(tmp_path), "--observations", str(tmp_path),
            "--selection", str(tmp_path / "selection.json"),
            "--out", str(tmp_path / "candidate"),
            "--execute", "--log", str(tmp_path / "log")]
    assert sup.main(args) == 2
    assert not (tmp_path / "log").exists()
