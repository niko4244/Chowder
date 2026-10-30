"""Source-only fail-closed checks for the registered Windows Q4 watchdog.

No model loads, GPU operations, existing service changes, or generated code
execution. All subprocess calls below are mocked.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

from chowder_batch import reap_q4_windows_benchmark_watchdog_20261026 as q4


@pytest.mark.parametrize(("resource", "limit"), [
    ("ram_available_gib", 7.0),
    ("commit_available_gib", 8.0),
    ("disk_free_gib", 8.0),
])
def test_original_runtime_floors_never_weakened(resource, limit):
    all_safe = {"ram_available_gib": 7.0, "commit_available_gib": 8.0,
                "disk_free_gib": 8.0, "gpu_free_mib": 500}
    assert not q4.unsafe(all_safe)
    below = dict(all_safe, **{resource: limit - 0.001})
    assert q4.unsafe(below)


@pytest.mark.parametrize("reported", ["14000\n12000\n", "", "not-an-int\n"])
def test_gpu_identity_or_value_malformed_refused(monkeypatch, reported):
    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(stdout=reported)

    monkeypatch.setattr(q4.subprocess, "run", fake_run)
    with pytest.raises((ValueError, IndexError)):
        q4.free_vram_mib()


def test_owned_tree_kill_never_uses_any_foreign_pid(monkeypatch):
    commands = []

    class Owned:
        pid = 33445

        def poll(self):
            return None

        def wait(self, timeout):
            return 0

    def fake_run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(q4.subprocess, "run", fake_run)
    q4.kill_owned_tree(Owned())
    assert commands == [["taskkill", "/PID", "33445", "/T", "/F"]]


def test_refused_native_admission_never_creates_output(tmp_path):
    receipt = tmp_path / "blocked.json"
    receipt.write_text(json.dumps({
        "schema": "reap128-v2-verified-host-admission.v1",
        "decision": {"status": "REFUSED"},
    }))
    target = tmp_path / "new-native-evidence"
    with pytest.raises(ValueError, match="did not pass"):
        q4.run(out=target, admission=receipt)
    assert not target.exists()


def test_eligible_receipt_but_lost_live_memory_refuses_before_model(monkeypatch, tmp_path):
    receipt = tmp_path / "admitted.json"
    limit = {"ram_available_gib": 29.315, "commit_available_gib": 34.315,
             "gpu_free_mib": 12288, "disk_free_gib": 8}
    receipt.write_text(json.dumps({
        "schema": "reap128-v2-verified-host-admission.v1",
        "decision": {
            "status": "ADMITTED_TO_ATTEMPT_NOT_SCIENTIFICALLY_QUALIFIED",
            "minimum_ram_gib": limit["ram_available_gib"],
            "minimum_commit_gib": limit["commit_available_gib"],
            "minimum_gpu_mib": limit["gpu_free_mib"],
            "minimum_disk_gib": limit["disk_free_gib"],
        },
        "current": limit,
    }))
    model = tmp_path / "fake-model.gguf"
    model.write_bytes(b"fake; never loaded")
    server = tmp_path / "fake-server.exe"
    server.write_bytes(b"fake; never executed")
    monkeypatch.setattr(q4, "GGUF", model)
    monkeypatch.setattr(q4, "SERVER", server)
    monkeypatch.setattr(
        q4, "sample",
        lambda _: {**limit, "ram_available_gib": 29.314},
    )
    target = tmp_path / "evidence"
    with pytest.raises(ValueError, match="admission lost before launch"):
        q4.run(out=target, admission=receipt)
    assert not target.exists()


@pytest.mark.parametrize(("resource", "poison"), [
    ("ram_available_gib", math.nan),
    ("commit_available_gib", math.inf),
    ("disk_free_gib", -math.inf),
    ("ram_available_gib", True),
    ("commit_available_gib", "11.0"),
    ("gpu_free_mib", -1),
])
def test_runtime_watchdog_poisoned_measurement_is_an_abort(resource, poison):
    row = {"ram_available_gib": 10.0, "commit_available_gib": 12.0,
           "disk_free_gib": 15.0, "gpu_free_mib": 3500}
    row[resource] = poison
    assert q4.unsafe(row), "invalid telemetry must never bypass runtime floor"


def test_runtime_watchdog_refuses_missing_resource_fields():
    assert q4.unsafe({"ram_available_gib": 10.0, "commit_available_gib": 15.0})


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_invalid_watchdog_sample_still_writes_finite_abort_evidence(bad):
    row = {"ram_available_gib": bad, "commit_available_gib": 10.0,
           "disk_free_gib": 50.0, "gpu_free_mib": 2000}
    failures = q4.unsafe(row)
    saved = q4.bounded_telemetry_record(row, failures)
    assert saved["resource_monitor_abort"] is True
    assert "ram_available_gib" in saved["invalid_measurements"]
    assert "NaN" not in json.dumps(saved, allow_nan=False)


def test_native_terminal_refuses_resident_child_server(monkeypatch):
    monkeypatch.setattr(q4.psutil, "pid_exists", lambda pid: True)
    monkeypatch.setattr(q4.psutil, "net_connections", lambda **_: [])
    with pytest.raises(RuntimeError, match="resident|native|shutdown"):
        q4.verify_native_shutdown(777, q4.PORT)


def test_native_terminal_refuses_port_rebound_by_any_process(monkeypatch):
    monkeypatch.setattr(q4.psutil, "pid_exists", lambda pid: False)
    monkeypatch.setattr(q4.psutil, "net_connections", lambda **_: [
        SimpleNamespace(status=q4.psutil.CONN_LISTEN, laddr=("127.0.0.1", q4.PORT))
    ])
    with pytest.raises(RuntimeError, match="port|listener|shutdown"):
        q4.verify_native_shutdown(777, q4.PORT)


def test_native_terminal_accepts_only_no_pid_and_no_listener(monkeypatch):
    monkeypatch.setattr(q4.psutil, "pid_exists", lambda pid: False)
    monkeypatch.setattr(q4.psutil, "net_connections", lambda **_: [])
    assert q4.verify_native_shutdown(777, q4.PORT) is True
