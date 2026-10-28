"""CPU-only source/Win32 adversarial completion-port and helper provenance tests."""

import copy
from pathlib import Path
import subprocess
import sys
import time

import pytest

from chowder_batch import reap_v2_frozen_wave_20261027 as wave
from chowder_batch import reap_v2_live_job_lineage_guard_20261028 as lineage
from chowder_batch import reap_v2_phase51_job_event_witness_20261028 as job_events

MODEL = Path("G:/reap128-v2-q4_k_m-20261025.gguf")
SERVER = Path("F:/llama.cpp/bin/llama-server.exe")
OUT = Path("C:/private-frozen-source-evidence")
STUDY = wave.study_for("reap128_v2_q4")
ROOT = 1401


def fake_provenance():
    command = wave.command(STUDY, model=MODEL, server=SERVER, output=OUT, port=8790)
    known = {
        ROOT: {"pid": ROOT, "birth": 3000.0, "parent": 1,
               "argv": command, "image_path": sys.executable,
               "job_member": True},
        1402: {"pid": 1402, "birth": 3001.0, "parent": ROOT,
               "argv": [str(SERVER), *wave.FLAGS, "--port", "8790", "-m", str(MODEL)],
               "image_path": str(SERVER), "job_member": True},
        1403: {"pid": 1403, "birth": 3002.0, "parent": ROOT,
               "argv": [sys.executable,
                        str(wave.REPO / "chowder_batch" / "reap_eval.py"),
                        *command[command.index("--") + 1:]],
               "image_path": sys.executable, "job_member": True},
    }
    return known


def packet(births):
    return {
        "schema": job_events.SCHEMA, "status": "OBSERVED_ONLY_NOT_APPROVED",
        "events": [{"message": job_events.NEW_PROCESS, "pid": k,
                    "birth_identity_captured": True} for k in births],
        "births": births, "faults": [],
    }


def reconcile(journal, ledger):
    return lineage.reconcile_event_births(
        journal, ledger=ledger, study=STUDY, model=MODEL, server=SERVER,
        output=OUT, port=8790,
    )


def test_complete_exact_original_pid_birth_roles_only_become_source_matches():
    ledger = lineage.LineageLedger(root_pid=ROOT, root_birth=3000.0)
    honest = packet(fake_provenance())
    result = reconcile(honest, ledger)
    assert result["status"] == "EVENT_BIRTHS_SOURCE_MATCHED_NOT_EXHAUSTIVE"
    assert len(ledger.pid_births) == 3
    assert result["job_total_processes_compared"] is False
    assert result["gen0"]["chosen"] is None


@pytest.mark.parametrize("fault", [
    "foreign_image", "other_parent", "wrong_argv", "not_a_job_member",
    "lost_birth", "duplicate_birth", "foreign_event", "pid_reused",
    "one_bogus_fault", "missing_event", "unknown_event",
])
def test_event_packets_never_self_approve_unknown_or_replaced_children(fault):
    evidence = packet(fake_provenance())
    ledger = lineage.LineageLedger(root_pid=ROOT, root_birth=3000.0)
    if fault == "foreign_image":
        evidence["births"][1402]["image_path"] = "C:/malware/llama-server.exe"
    elif fault == "other_parent":
        evidence["births"][1403]["parent"] = 1
    elif fault == "wrong_argv":
        evidence["births"][1402]["argv"].append("--fit-off")
    elif fault == "not_a_job_member":
        evidence["births"][1403]["job_member"] = False
    elif fault == "lost_birth":
        evidence["events"][2]["birth_identity_captured"] = False
    elif fault == "duplicate_birth":
        evidence["events"].append(copy.deepcopy(evidence["events"][1]))
    elif fault == "foreign_event":
        evidence["events"].append({
            "message": job_events.NEW_PROCESS, "pid": 8888,
            "birth_identity_captured": False,
        })
    elif fault == "pid_reused":
        ledger.pid_births[1402] = 2990.0
    elif fault == "one_bogus_fault":
        evidence["faults"].append("UNTRUSTED_PROCESS_IMAGE")
    elif fault == "missing_event":
        evidence["events"].pop()
    elif fault == "unknown_event":
        evidence["events"].append({"message": 101, "pid": 8888})
    refused = reconcile(evidence, ledger)
    assert refused["status"] == "REFUSED", (fault, refused)
    assert ledger.pid_births == ({1402: 2990.0} if fault == "pid_reused" else {})


def test_missing_short_helper_birth_is_not_invented_from_native_total():
    ledger = lineage.LineageLedger(root_pid=ROOT, root_birth=3000.0)
    observed = packet(fake_provenance())
    assert reconcile(observed, ledger)["status"].startswith("EVENT_BIRTHS_SOURCE")
    assert len(ledger.pid_births) == 3
    assert len(ledger.pid_births) != 4
    # Four cumulative process births without the fourth complete live
    # image/argv/parent/membership are NOT a passing private-Job lifetime.
    missing = copy.deepcopy(observed)
    missing["events"].append({
        "message": job_events.NEW_PROCESS, "pid": 1404,
        "birth_identity_captured": False,
    })
    assert reconcile(missing, ledger)["status"] == "REFUSED"


@pytest.mark.parametrize("bad_pid", ["01401", "0001401", "1401 "])
def test_json_birth_pid_alias_is_rejected_before_frozen_source_join(bad_pid):
    evidence = packet(fake_provenance())
    evidence["births"] = {str(k): v for k, v in evidence["births"].items()}
    evidence["births"][bad_pid] = evidence["births"].pop(str(ROOT))
    ledger = lineage.LineageLedger(root_pid=ROOT, root_birth=3000.0)
    assert reconcile(evidence, ledger)["status"] == "REFUSED"
    assert ledger.pid_births == {}


def test_json_birth_pid_roundtrip_requires_exact_native_source_roles():
    import json

    evidence = json.loads(json.dumps(packet(fake_provenance())))
    ledger = lineage.LineageLedger(root_pid=ROOT, root_birth=3000.0)
    assert reconcile(evidence, ledger)["status"] == "EVENT_BIRTHS_SOURCE_MATCHED_NOT_EXHAUSTIVE"
    assert set(ledger.pid_births) == {ROOT, 1402, 1403}


def test_unexpected_kernel_reader_exception_fails_closed_without_a_packet():
    reader = job_events.PrivateJobEvents(kernel=object(), job=1234,
                                         member_check=lambda _: True)
    reader._port = 987

    def failed_native(*args, **kwargs):
        raise RuntimeError("synthetic GetQueuedCompletionStatus failure")

    reader._native = failed_native
    reader._poll()
    assert "PRIVATE_JOB_EVENT_READER_UNEXPECTED_EXCEPTION" in reader.snapshot()["faults"]
    assert "PRIVATE_JOB_EVENT_READER_STOPPED_UNEXPECTEDLY" in reader.snapshot()["faults"]
    assert reader.snapshot()["gen0"]["chosen"] is None


def test_terminal_stop_requires_fresh_empty_polls_not_prior_idle_history(monkeypatch):
    reader = job_events.PrivateJobEvents(kernel=object(), job=1234,
                                         member_check=lambda _: True)
    # A long-lived model server may have accumulated thousands of idle polls
    # before the final native process exits and queues its exit notification.
    reader._quiet_polls = 999
    observed = []

    class RunningReader:
        def is_alive(self):
            return not reader._stop.is_set()

        def join(self, timeout):
            assert reader._stop.is_set()

    reader._thread = RunningReader()

    def fresh_timeout(_seconds):
        observed.append(reader._quiet_polls)
        with reader._lock:
            reader._quiet_polls += 1

    monkeypatch.setattr(job_events.time, "sleep", fresh_timeout)
    receipt = reader.stop()
    assert observed == [0, 1]
    assert receipt["faults"] == []
    assert receipt["gen0"]["release_approved"] is False


def test_handle_close_failure_still_closes_other_owned_handles_and_port():
    class CloseHandle:
        def __init__(self):
            self.called = []

        def __call__(self, handle):
            self.called.append(handle)
            return int(handle != 11)

    closer = CloseHandle()
    kernel = type("Kernel", (), {})()
    kernel.CloseHandle = closer
    reader = job_events.PrivateJobEvents(kernel=kernel, job=1234,
                                         member_check=lambda _: True)
    reader._port = 22
    reader._process_handles = {1: 11, 2: 12}
    with pytest.raises(job_events.EventWitnessRefused,
                       match="ORIGINAL_JOB_PROCESS_HANDLE_CLOSE_FAILED"):
        reader.close()
    assert closer.called == [11, 12, 22]
    assert reader._port is None
    assert reader._process_handles == {1: 11}


def test_real_disposable_private_win32_job_receives_ephemeral_child_events():
    if sys.platform != "win32":
        pytest.skip("real Win32 Job notification ABI")
    from chowder_batch.reap_v2_owner_bound_runtime_supervisor_20261027 import _WindowsOwnedJob

    owner = _WindowsOwnedJob()
    original_call = owner._call

    def audited_native_call(name, *args, **kwargs):
        if name == "ResumeThread":
            assert owner.active is True
            assert owner.event_witness is not None
            assert owner.event_witness._thread is not None
            assert owner.event_witness._thread.is_alive()
        return original_call(name, *args, **kwargs)

    owner._call = audited_native_call
    # This is an original-model-free CPU-only process tree, NEVER a native
    # frozen evaluation, foreign process, CUDA workload or approval.
    code = (
        'import subprocess,sys,time;'
        'subprocess.run([sys.executable,"-c","pass"],check=True);'
        'time.sleep(0.35)'
    )
    try:
        owner.start([sys.executable, "-c", code], Path.cwd())
        assert owner.exit_code() is None
        time.sleep(0.85)
        native = owner.lifetime_process_counts()
        receipt = owner.event_witness.snapshot()
        sources = receipt["births"]
        new = [m for m in receipt["events"] if m["message"] == job_events.NEW_PROCESS]
        assert native["total_processes"] >= 3
        assert len(new) >= 2
        assert any(s["pid"] == owner.pid for s in sources.values())
        assert receipt["gen0"]["release_approved"] is False
        assert receipt["status"] == "OBSERVED_ONLY_NOT_APPROVED"
        # NEW/EXIT delivery is not guaranteed by Windows; event shortfall
        # cannot be interpreted as a complete process or source proof.
        assert len(new) <= native["total_processes"]
    finally:
        if owner.active:
            owner.close()
