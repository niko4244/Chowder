"""Tests for llama-server lifecycle management (manager owns only what it starts)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from chowder.llama_server_manager import (
    LlamaServerError,
    LlamaServerManager,
    ServerHandle,
    ServerSpec,
)


def _fake_bin(tmp_path: Path) -> Path:
    bin_path = tmp_path / "llama-server.exe"
    bin_path.write_bytes(b"MZ fake")
    return bin_path


def _spec(tmp_path: Path, **overrides) -> ServerSpec:
    defaults = dict(name="t", model_path=_fake_bin(tmp_path), port=9, gpu_index=None)
    defaults.update(overrides)
    return ServerSpec(**defaults)


def _handle(name: str, pid: int, spec: ServerSpec) -> ServerHandle:
    return ServerHandle(spec=spec, pid=pid, started_at=0.0, log_path=Path(f"{name}.log"))


@pytest.fixture()
def manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LlamaServerManager:
    monkeypatch.setattr("chowder.llama_server_manager.DEFAULT_SERVER_BIN", _fake_bin(tmp_path))
    return LlamaServerManager(server_bin=_fake_bin(tmp_path), state_dir=tmp_path / "state")


def test_start_refuses_when_port_occupied(manager, tmp_path):
    manager._health_ok = lambda url: True  # port already answers
    with pytest.raises(LlamaServerError, match="already answers"):
        manager.start(_spec(tmp_path, name="collide"))


def test_start_refuses_when_same_name_alive(manager, tmp_path, monkeypatch):
    manager._health_ok = lambda url: False
    monkeypatch.setattr(manager, "_pid_alive", lambda pid: True)
    manager._save_handle(_handle("dup", pid=999999999, spec=_spec(tmp_path, name="dup", port=12)))
    with pytest.raises(LlamaServerError, match="already running"):
        manager.start(_spec(tmp_path, name="dup", port=12))


def test_start_refuses_when_vram_does_not_fit(manager, tmp_path, monkeypatch):
    manager._health_ok = lambda url: False
    monkeypatch.setattr(
        "chowder.llama_server_manager.query_gpu_vram_mib", lambda: {0: (5000, 24576)}
    )
    spec = _spec(tmp_path, name="vram", gpu_index=0, expected_vram_mib=1_000_000)
    with pytest.raises(LlamaServerError, match="VRAM arbitration"):
        manager.start(spec)


def test_start_succeeds_and_records_state(manager, tmp_path, monkeypatch):
    proc = type(
        "P", (), {"pid": 4242, "poll": lambda self: None, "kill": lambda self: None, "wait": lambda self, t=0: None}
    )()
    seen = {}
    monkeypatch.setattr(
        "chowder.llama_server_manager.subprocess.Popen",
        lambda cmd, **k: seen.update(cmd=cmd, env=k.get("env")) or proc,
    )
    calls = {"n": 0}

    def health(url: str) -> bool:
        calls["n"] += 1
        return calls["n"] >= 2  # call 1: port guard; call 2: startup poll

    manager._health_ok = health
    handle = manager.start(_spec(tmp_path, name="ok", port=10))
    assert handle.pid == 4242
    state = json.loads((Path(manager.state_dir) / "ok.json").read_text(encoding="utf-8"))
    assert state["pid"] == 4242
    assert state["spec"]["port"] == 10
    assert seen["cmd"][0].endswith("llama-server.exe")


def test_start_failure_cleans_up(manager, tmp_path, monkeypatch):
    killed = {"n": 0}
    proc = type(
        "P",
        (),
        {
            "pid": 77,
            "poll": lambda self: 1,
            "returncode": 1,
            "kill": lambda self: killed.__setitem__("n", killed["n"] + 1),
            "wait": lambda self, t=0: None,
        },
    )()
    monkeypatch.setattr("chowder.llama_server_manager.subprocess.Popen", lambda cmd, **k: proc)
    manager._health_ok = lambda url: False
    with pytest.raises(LlamaServerError, match="exited during startup"):
        manager.start(_spec(tmp_path, name="dead"))
    assert killed["n"] == 0  # exited on its own; no kill attempted
    assert not (Path(manager.state_dir) / "dead.json").exists()


def test_stop_removes_stale_record(manager, tmp_path):
    manager._save_handle(_handle("stale", pid=5999999, spec=_spec(tmp_path, name="stale", port=13)))
    result = manager.stop("stale")
    assert result["stopped"] is True
    assert "stale record removed" in result["reason"]
    assert not (Path(manager.state_dir) / "stale.json").exists()


def test_stop_terminates_live_pid(manager, tmp_path, monkeypatch):
    alive = {"4242": True}

    def fake_taskkill(*a, **k):
        alive["4242"] = False  # taskkill succeeds

    monkeypatch.setattr("chowder.llama_server_manager.subprocess.run", fake_taskkill)
    manager._save_handle(_handle("live", pid=4242, spec=_spec(tmp_path, name="live", port=14)))
    monkeypatch.setattr(manager, "_pid_alive", lambda pid: alive[str(pid)])
    result = manager.stop("live")
    assert result["stopped"] is True
    assert not (Path(manager.state_dir) / "live.json").exists()


def test_stop_unknown_name(manager):
    result = manager.stop("never-started")
    assert result["stopped"] is False


def test_status_reports_liveness(manager, tmp_path, monkeypatch):
    manager._save_handle(_handle("s1", pid=1, spec=_spec(tmp_path, name="s1", port=11)))
    monkeypatch.setattr(manager, "_pid_alive", lambda pid: pid == 1)
    manager._health_ok = lambda url: False
    rows = manager.status()
    assert len(rows) == 1
    assert rows[0]["name"] == "s1"
    assert rows[0]["alive"] is True
    assert rows[0]["healthy"] is False


def test_vram_estimate_full_offload(tmp_path):
    spec = _spec(tmp_path, n_gpu_layers=99)
    size_mib = spec.model_path.stat().st_size / (1024 * 1024)
    assert spec.estimate_vram_mib() == int(size_mib * 1.2)


def test_vram_estimate_unknown_for_partial_offload(tmp_path):
    spec = _spec(tmp_path, n_gpu_layers=20)
    assert spec.estimate_vram_mib() is None
