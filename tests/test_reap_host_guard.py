"""Offline host-safety decisions and actual CPU subprocess termination tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from chowder_batch import reap_host_guard as guard
from chowder_batch import reap_observe_supervised as supervised

ARCHIVE = Path(__file__).parent / "data" / "reap_r2_host_126.jsonl"


def _sample(at=1000.0, *, ram=20, commit=30, births=(),
            services=False, errors=0, gpu=14, temp=55) -> guard.Snapshot:
    return guard.Snapshot(
        timestamp=at, ram_free_gib=ram, virtual_free_gib=commit,
        diskC_free_gib=9, diskF_free_gib=80,
        gpu_free_gib=gpu, gpu_temp_c=temp,
        processes=tuple(births), services_conflict=services,
        census_errors=errors,
    )


def _proc(pid: int, at: float, *, parent=1, gb=0.1):
    return guard.ProcessInfo(
        pid, at, parent, "node", int(gb * guard.GIB), int(gb * guard.GIB),
    )


def test_exact_126_real_r2_host_samples_abort_at_123_without_false_early_guard(monkeypatch):
    # This code path must remain fully offline; replay cannot query the live OS
    # or cause observer/service/process changes.
    def forbidden(*args, **kwargs):
        raise AssertionError("replay attempted to access live host")
    monkeypatch.setattr(guard, "collect_snapshot", forbidden)
    rows = guard.replay_timeline(ARCHIVE)
    assert rows["samples"] == 126
    first = rows["first_abort"]
    assert first["sample"] == 123
    assert first["timestamp_utc"] == "2026-10-24T05:54:53.6901049Z"
    assert first["decision"]["action"] == "ABORT"
    assert "physical_ram_below_5_gib" in first["decision"]["reasons"]
    assert first["decision"]["ram_decline_60s_gib"] == pytest.approx(2.95)


def test_guard_replay_cli_never_requires_or_invokes_launch(tmp_path):
    cmd = [sys.executable, "-B", "-m", "chowder_batch.reap_observe_supervised",
           "--replay", str(ARCHIVE)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["first_abort"]["sample"] == 123
    not_launched = subprocess.run(
        [sys.executable, "-B", "-m", "chowder_batch.reap_observe_supervised"],
        capture_output=True, text=True, timeout=20,
    )
    assert not_launched.returncode != 0
    assert "--launch" in not_launched.stderr


def test_three_stable_30_second_prelaunch_samples_admit_only_when_healthy():
    history = [_sample(1000), _sample(1030), _sample(1060)]
    assert guard.assess_admission(history).action == "ADMIT"
    assert guard.assess_admission(history[:2]).action == "REFUSE"
    assert guard.assess_admission(history[:1] * 3).action == "REFUSE"
    history = [_sample(1000, ram=20), _sample(1030, ram=18),
               _sample(1060, ram=18)]
    assert "physical_ram_falling_over_60s" in guard.assess_admission(history).reasons
    history = [_sample(1000), _sample(1030, commit=20),
               _sample(1060)]
    assert "free_commit_below_24_gib" in guard.assess_admission(history).reasons
    history = [_sample(1000), _sample(1030, services=True), _sample(1060)]
    assert "protected_service_ports_listening" in guard.assess_admission(history).reasons
    history = [_sample(1000), _sample(1030, gpu=11.9), _sample(1060)]
    assert "gpu_free_below_12_gib" in guard.assess_admission(history).reasons


def test_foreign_process_fanout_counted_by_pid_and_start_time():
    newborn = tuple(_proc(pid, 1050 + pid % 3, gb=0.1) for pid in range(200, 212))
    samples = [_sample(1000), _sample(1030),
               _sample(1060, births=newborn)]
    verdict = guard.assess_admission(samples)
    assert verdict.action == "REFUSE"
    assert verdict.new_foreign_processes_60s == 12
    assert "foreign_process_fanout_over_60s" in verdict.reasons
    # A producer subprocess of our own observer is NOT a competing actor.
    descendants = (_proc(200, 1059, parent=44), _proc(201, 1059, parent=200))
    verdict = guard.assess_admission(
        [_sample(1000), _sample(1030),
         _sample(1060, births=descendants)], owner_pids=frozenset({44}),
    )
    assert verdict.new_foreign_processes_60s == 0
    # A reused PID born within the interval is a new process identity.
    old = _proc(200, 600)
    new = _proc(200, 1055)
    assert old.key != new.key
    assert guard._foreign_births(_sample(1060, births=(old, new)),
                                 frozenset()) == [new]


def test_birth_alarm_alone_not_destructive_without_ram_decline():
    newborn = tuple(_proc(pid, 1045, gb=0.1) for pid in range(300, 313))
    healthy = [_sample(1000, ram=8), _sample(1030, ram=8),
               _sample(1060, ram=8, births=newborn)]
    result = guard.assess_runtime(healthy)
    assert result.action == "CONTINUE"
    assert result.new_foreign_processes_60s == 13
    assert "foreign_process_birth_burst" in result.warnings
    assert "foreign_new_processes_private_memory_over_1_gib" in result.warnings
    alarm = [_sample(1000, ram=9), _sample(1030, ram=8),
             _sample(1060, ram=6.5, births=newborn)]
    result = guard.assess_runtime(alarm)
    assert result.action == "ABORT"
    assert "competing_process_fanout_and_ram_decline" in result.reasons
    # Memory alone must abort; absence of PID history cannot hide a cliff.
    no_births = [_sample(1000, ram=8), _sample(1030, ram=7),
                 _sample(1060, ram=4.5)]
    assert "physical_ram_below_5_gib" in guard.assess_runtime(no_births).reasons


def test_telemetry_failures_and_disk_gpu_thresholds_fail_closed():
    assert guard.assess_runtime([]).action == "ABORT"
    for sample, reason in (
        (_sample(1060, gpu=None), "gpu_memory_unavailable_or_low"),
        (_sample(1060, temp=None), "gpu_telemetry_unavailable_or_hot"),
        (_sample(1060, errors=26), "process_telemetry_substantially_incomplete"),
        (_sample(1060, commit=5), "free_commit_below_6_gib"),
    ):
        assert reason in guard.assess_runtime([sample]).reasons
    assert guard.assess_runtime([_sample(1060, errors=2)]).action == "CONTINUE"


def test_supervisor_owned_child_only_stopping_and_missing_receipt(tmp_path):
    class FakeChild:
        def __init__(self):
            self.terminated = 0
            self.killed = 0
        def poll(self): return None
        def terminate(self): self.terminated += 1
        def wait(self, timeout=None): return 7
        def kill(self): self.killed += 1

    worker = FakeChild()
    supervised._owned_only_stop(worker)
    assert worker.terminated == 1 and worker.killed == 0
    with pytest.raises(FileNotFoundError):
        supervised._check_observation_complete(tmp_path, tmp_path / "cal.jsonl")


def test_live_supervisor_refuses_unhealthy_prelaunch_without_starting_any_worker(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(supervised.sys, "platform", "win32")
    monkeypatch.setattr(supervised, "_validate_paths", lambda args: None)
    monkeypatch.setattr(supervised.time, "sleep", lambda _: None)
    iterator = iter((
        _sample(1000, ram=10),
        _sample(1030, ram=9),
        _sample(1060, ram=8),
    ))
    monkeypatch.setattr(supervised.guard, "collect_snapshot", lambda: next(iterator))
    def never_start(*args, **kwargs):
        raise AssertionError("unsafe prelaunch attempted to launch observer")
    monkeypatch.setattr(supervised.subprocess, "Popen", never_start)
    args = SimpleNamespace(
        log_dir=tmp_path / "new-logs", model_dir=tmp_path / "unused-model",
        weights_dir=None, calibration=tmp_path / "unused-cal",
        out=tmp_path / "unused-result", scratch=tmp_path / "unused-scratch",
        resume=False,
    )
    assert supervised._run(args) == 2
    state = json.loads((args.log_dir / "supervisor-state.json").read_text())
    assert state["registered_d0_proven"] is False
    assert state["stage"] == "INCOMPLETE_D0"
    assert "physical_ram_below_16_gib" in state["reason"]
    assert len((args.log_dir / "resources.jsonl").read_text().splitlines()) == 3


@pytest.mark.parametrize("crash_phase,expected_generation", [
    ("partial_row", 0),
    ("before_current", 0),
    ("after_current", 1),
    ("after_slot_wrap_partial", 2),
    ("before_slot_wrap_fsync", 2),
])
def test_actual_abrupt_process_exit_releases_lock_and_recovers_commits(
    tmp_path, crash_phase, expected_generation,
):
    pytest.importorskip("torch")
    pytest.importorskip("numpy")
    parent = Path(__file__).parents[1]
    scratch = tmp_path / crash_phase
    ready = tmp_path / f"{crash_phase}.ready"
    code = r'''
import os, pathlib, sys, time, torch
from chowder_batch import reap_hidden_checkpoint as cp
path=pathlib.Path(sys.argv[1])
ready=pathlib.Path(sys.argv[2])
stage=sys.argv[3]
cp.shutil.disk_usage=lambda _: type("Disk",(),{"free":100*cp.GIB})()
identity={"source_sha256":"a"*64,"token_ids_sha256":"b"*64}
o=cp.BF16CheckpointHidden([2,1],4,path,identity)
o[0]=torch.ones((1,2,4),dtype=torch.bfloat16)
o[1]=torch.ones((1,1,4),dtype=torch.bfloat16)
o.finish_embeddings()
o.begin_layer(0)
o[0]=torch.full((1,2,4),2,dtype=torch.bfloat16)
if stage=="partial_row":
    ready.write_text("ready")
    while True: time.sleep(0.2)
o[1]=torch.full((1,1,4),2,dtype=torch.bfloat16)
stats={k:torch.ones(3,dtype=torch.float64) for k in ("count","gate_sum","norm_sum","reap_sum")}
if stage in ("after_slot_wrap_partial", "before_slot_wrap_fsync"):
    o.finish_layer(0,stats,{"layer":0})
    o.begin_layer(1)
    o[0]=torch.full((1,2,4),3,dtype=torch.bfloat16)
    o[1]=torch.full((1,1,4),3,dtype=torch.bfloat16)
    o.finish_layer(1,stats,{"layer":1})
    o.begin_layer(2)
    o[0]=torch.full((1,2,4),4,dtype=torch.bfloat16)
    if stage=="after_slot_wrap_partial":
        ready.write_text("ready")
        while True: time.sleep(0.2)
    o[1]=torch.full((1,1,4),4,dtype=torch.bfloat16)
    original=cp._sync_file
    def interrupted_sync(path):
        if path.name=="state-B.bf16": os._exit(73)
        return original(path)
    cp._sync_file=interrupted_sync
    o.finish_layer(2,stats,{"layer":2})
if stage in ("before_current","after_current"):
    atomic=cp._atomic_bytes
    def split(path,payload):
        if path.name=="CURRENT" and b'"generation":1' in payload:
            if stage=="before_current": os._exit(71)
            atomic(path,payload)
            os._exit(72)
        return atomic(path,payload)
    cp._atomic_bytes=split
o.finish_layer(0,stats,{"layer":0})
'''
    child = subprocess.Popen(
        [sys.executable, "-B", "-c", code, str(scratch), str(ready), crash_phase],
        cwd=parent, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        if crash_phase in ("partial_row", "after_slot_wrap_partial"):
            deadline = time.monotonic() + 25
            while time.monotonic() < deadline and not ready.exists() and child.poll() is None:
                time.sleep(0.1)
            assert ready.exists(), child.stderr.read().decode(errors="replace")
            child.kill()
            assert child.wait(timeout=10) != 0
        else:
            assert child.wait(timeout=25) in (71, 72, 73), child.stderr.read().decode(
                errors="replace")
        from chowder_batch import reap_hidden_checkpoint as cp
        # The production storage admission does not apply when resuming.
        obj = cp.BF16CheckpointHidden(
            [2, 1], 4, scratch,
            {"source_sha256": "a" * 64, "token_ids_sha256": "b" * 64},
            resume=True,
        )
        assert obj.generation == expected_generation
        obj.begin_layer(expected_generation)
        assert obj[0].flatten().tolist() == [
            float(expected_generation + 1)
        ] * 8
        obj.close()
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=10)
