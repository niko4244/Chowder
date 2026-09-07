"""Tests for the tournament's load-stability machinery: the commit-headroom
preflight gate and the native-crash retry in `_run_worker`.

These are unit tests over monkeypatched OS/process boundaries -- no GPU,
no model load, no real subprocess. The evidence motivating each behavior
is in the module docstring of `parent_tournament.py` and docs/HANDOFF.md;
CI pins the contract, the lab machine proved the mechanism.
"""

from __future__ import annotations

import json

import pytest

import chowder.parent_tournament as pt


# ---- _enforce_commit_headroom ------------------------------------------------


def test_gate_passes_when_headroom_sufficient(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pt, "_commit_state", lambda: (30.0, 120.0))  # 90 free
    assert pt._enforce_commit_headroom() == pytest.approx(90.0)


def test_gate_fails_with_measured_numbers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pt, "_commit_state", lambda: (75.0, 119.0))  # 44 free
    with pytest.raises(pt.ParentTournamentError) as excinfo:
        pt._enforce_commit_headroom()
    msg = str(excinfo.value)
    assert "44.0 GiB" in msg          # measured headroom, verbatim
    assert "75.0/119.0" in msg        # measured used/limit
    assert "CHOWDER_MIN_COMMIT_HEADROOM_GIB" in msg  # actionable override


def test_gate_threshold_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHOWDER_MIN_COMMIT_HEADROOM_GIB", "10")
    monkeypatch.setattr(pt, "_MIN_COMMIT_HEADROOM_GIB", 10.0)
    monkeypatch.setattr(pt, "_commit_state", lambda: (100.0, 119.0))
    assert pt._enforce_commit_headroom() == pytest.approx(19.0)


def test_gate_skipped_when_measurement_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-Windows (or failed measurement): gate passes, returns NaN."""
    monkeypatch.setattr(pt, "_commit_state", lambda: None)
    result = pt._enforce_commit_headroom()
    assert result != result  # NaN


# ---- _run_worker retry + observability ---------------------------------------


class _FakeProc:
    def __init__(self, returncode: int, stdout_text: str = ""):
        self.returncode = returncode
        self.stdout = stdout_text


def _make_run(monkeypatch: pytest.MonkeyPatch, tmp_path, returncodes: list[int]):
    """Fake subprocess.run: each call pops the next returncode; a 0 exit
    writes a valid result file. Records the env each call received."""
    calls: list[dict] = []

    def fake_run(command, stdout=None, stderr=None, text=None, timeout=None, env=None, **kwargs):
        # The VRAM sampler thread concurrently calls subprocess.run for
        # nvidia-smi; answer it inertly without consuming the scripted
        # worker returncodes.
        if "base_text_worker" not in str(command):
            return _FakeProc(0, stdout_text="")
        calls.append({"env": env, "returncodes_left": list(returncodes)})
        rc = returncodes.pop(0)
        if rc == 0:
            result_path = tmp_path / "eval-result.json"
            result_path.write_text(json.dumps({"ok": True}) + "\n", encoding="utf-8")
        return _FakeProc(rc)

    monkeypatch.setattr(pt.subprocess, "run", fake_run)
    return calls


def _monkeypatch_gate(monkeypatch: pytest.MonkeyPatch, values: list[float]) -> list[int]:
    gate_calls = {"n": 0}

    def gate():
        idx = min(gate_calls["n"], len(values) - 1)
        gate_calls["n"] += 1
        v = values[idx]
        if v < 0:
            raise pt.ParentTournamentError("gate refused")
        return v

    monkeypatch.setattr(pt, "_enforce_commit_headroom", gate)
    return gate_calls


def test_worker_succeeds_first_attempt(monkeypatch, tmp_path) -> None:
    calls = _make_run(monkeypatch, tmp_path, [0])
    _monkeypatch_gate(monkeypatch, [90.0])
    result = pt._run_worker({"model": "x"}, tmp_path, timeout_seconds=60)
    assert result["ok"] is True
    assert result["worker_attempts"] == 1
    assert result["commit_headroom_gib_at_launch"] == 90.0
    assert len(calls) == 1
    assert calls[0]["env"]["PYTHONUNBUFFERED"] == "1"


def test_worker_retries_after_native_crash(monkeypatch, tmp_path) -> None:
    calls = _make_run(monkeypatch, tmp_path, [3221225477, 0])
    _monkeypatch_gate(monkeypatch, [90.0, 90.0])
    result = pt._run_worker({"model": "x"}, tmp_path, timeout_seconds=60)
    assert result["ok"] is True
    assert result["worker_attempts"] == 2
    assert len(calls) == 2


def test_worker_retries_posix_sigsegv_too(monkeypatch, tmp_path) -> None:
    _make_run(monkeypatch, tmp_path, [139, 0])
    _monkeypatch_gate(monkeypatch, [90.0, 90.0])
    result = pt._run_worker({"model": "x"}, tmp_path, timeout_seconds=60)
    assert result["worker_attempts"] == 2


def test_worker_gives_up_after_bounded_retries(monkeypatch, tmp_path) -> None:
    _make_run(monkeypatch, tmp_path, [3221225477, 3221225477, 3221225477])
    _monkeypatch_gate(monkeypatch, [90.0, 90.0, 90.0])
    with pytest.raises(pt.ParentTournamentError) as excinfo:
        pt._run_worker({"model": "x"}, tmp_path, timeout_seconds=60)
    msg = str(excinfo.value)
    assert "attempt 3" in msg
    assert "3221225477" in msg


def test_worker_rechecks_gate_before_each_retry(monkeypatch, tmp_path) -> None:
    """A retry must not relaunch blind: if pressure got worse, the gate
    must fail loudly instead of segfaulting again."""
    _make_run(monkeypatch, tmp_path, [3221225477, 0])
    gate_state = _monkeypatch_gate(monkeypatch, [90.0, -1.0])
    with pytest.raises(pt.ParentTournamentError, match="gate refused"):
        pt._run_worker({"model": "x"}, tmp_path, timeout_seconds=60)
    assert gate_state["n"] == 2  # initial launch + pre-retry recheck


def test_worker_non_native_failure_is_not_retried(monkeypatch, tmp_path) -> None:
    """Exit code 1 (python exception) is deterministic: no retry."""
    calls = _make_run(monkeypatch, tmp_path, [1])
    _monkeypatch_gate(monkeypatch, [90.0])
    with pytest.raises(pt.ParentTournamentError, match="attempt 1"):
        pt._run_worker({"model": "x"}, tmp_path, timeout_seconds=60)
    assert len(calls) == 1
