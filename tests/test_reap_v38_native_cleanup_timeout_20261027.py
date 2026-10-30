"""Native VM teardown slowness must never weaken externally witnessed safety."""

import subprocess

import pytest

from chowder_batch import reap_humaneval_v38_actual_execution_20261027 as runner


FIXED_OWNER = {"container_pid": 12345,
               "container_pid_start_ticks": 6789,
               "cgroup_path": "/sys/fs/cgroup/libpod-verified-owner"}


def test_two_transient_ssh_timeouts_then_original_vm_host_extinction(monkeypatch):
    calls = []

    def real_original_witness(binding):
        calls.append(binding)
        if len(calls) < 3:
            raise subprocess.TimeoutExpired("podman machine ssh", 15)
        return {"same_owned_processes_live": [], "cgroup_procs": []}

    monkeypatch.setattr(runner.v37, "_vm_postcheck", real_original_witness)
    assert runner._outside_host_cleanup_retry(FIXED_OWNER) == {
        "same_owned_processes_live": [], "cgroup_procs": []}
    assert len(calls) == 3
    assert all(c["child"] == {"pid": 12345, "start_ticks": 6789} for c in calls)


def test_three_host_timeouts_still_refused(monkeypatch):
    monkeypatch.setattr(runner.v37, "_vm_postcheck",
                        lambda _: (_ for _ in ()).throw(
                            subprocess.TimeoutExpired("podman machine ssh", 15)))
    with pytest.raises(subprocess.TimeoutExpired):
        runner._outside_host_cleanup_retry(FIXED_OWNER)


@pytest.mark.parametrize("returned", [
    {"same_owned_processes_live": [12345], "cgroup_procs": []},
    {"same_owned_processes_live": [], "cgroup_procs": ["12345"]},
    {"same_owned_processes_live": None, "cgroup_procs": []},
    {"same_owned_processes_live": [], "cgroup_procs": None},
    {},
])
def test_cannot_admit_forged_or_live_process_extinction(monkeypatch, returned):
    monkeypatch.setattr(runner.v37, "_vm_postcheck", lambda _: returned)
    with pytest.raises(RuntimeError, match="not verified"):
        runner._outside_host_cleanup_retry(FIXED_OWNER)


def test_real_kernel_safety_rejection_not_masked_by_retry(monkeypatch):
    calls = []

    def witness(_):
        calls.append(True)
        raise RuntimeError("real host found live descendant")

    monkeypatch.setattr(runner.v37, "_vm_postcheck", witness)
    with pytest.raises(RuntimeError, match="live descendant"):
        runner._outside_host_cleanup_retry(FIXED_OWNER)
    assert len(calls) == 1


def test_unchanged_model_candidate_wall_time_and_sealing_contract():
    assert "p.communicate(timeout=10)" in runner.V38_LOADER
    assert "F_SEAL_SEAL" in runner.V38_LOADER
    assert "F_GET_SEALS" in runner.V38_LOADER
    assert runner.APPROVED_V38_RESEARCH_LOADER_SHA256 == runner._sha(
        runner.V38_LOADER.encode())
    assert runner.APPROVED_V38_RESEARCH_HOST_OBSERVER_SHA256 == runner._sha(
        runner.V38_VM_SOURCE_OBSERVER.encode())
