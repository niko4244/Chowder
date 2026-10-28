"""CPU-only Phase50 negative and benign Windows conhost natural-drain tests."""

from pathlib import Path

import pytest

from chowder_batch import reap_v2_live_job_lineage_guard_20261028 as lineage
from chowder_batch import reap_v2_phase50_postexit_job_drain_20261028 as drain


class OwnedJob:
    pid = 110

    def __init__(self, states, total):
        self.states = list(states)
        self.next = 0
        self.current = states[0]
        self.total = total

    def member_ids(self):
        if self.next < len(self.states):
            self.current = self.states[self.next]
            self.next += 1
        return list(self.current)

    def lifetime_process_counts(self):
        return {
            "total_processes": self.total,
            "active_processes": len(self.current),
            "total_page_fault_count": 10,
        }


def _ledger():
    ledger = lineage.LineageLedger(root_pid=110, root_birth=1000.0)
    ledger.full_active_witnessed = True
    ledger.roles = {"root": (110, 1000.0), "server": (111, 1001.0),
                    "evaluator": (112, 1002.0)}
    ledger.pid_births = {110: 1000.0, 111: 1001.0, 112: 1002.0}
    return ledger


def _observe(job, ledger, capture, polls=4):
    return drain.witness_natural_drain(
        job, ledger=ledger, study=object(), model=Path("G:/original.gguf"),
        server=Path("F:/native.exe"), output=Path("C:/scratch"), port=8790,
        capture=capture, polls=polls, sleep=lambda _: None,
    )


def test_os_conhost_after_root_zero_drains_naturally_without_forced_kill():
    ledger = _ledger()
    job = OwnedJob([[110, 113], [], [], []], total=4)
    events = []

    def source(owner, **kw):
        events.append(kw["phase"])
        assert kw["phase"] == "shutdown"
        kw["ledger"].pid_births[113] = 1003.0
        return {"status": "PENDING_SHUTDOWN", "roles": {"root": 110},
                "reasons": []}

    good = _observe(job, ledger, source)
    assert good["status"] == "NATURALLY_DRAINED"
    assert good["pre_cleanup_live_owned_descendant_count"] == 0
    assert good["all_private_job_lifetime_processes_source_witnessed"] is True
    assert good["polls"] == 2 and events == ["shutdown"]
    assert good["gen0"]["chosen"] is None
    assert not hasattr(drain, "kill") and not hasattr(drain, "terminate")


def test_post_root_eval_or_native_server_still_alive_must_refuse_without_grace():
    ledger = _ledger()
    job = OwnedJob([[110, 111]], total=3)
    out = _observe(job, ledger, lambda *a, **k: {
        "status": "PENDING_SHUTDOWN", "roles": {"server": 111}, "reasons": []})
    assert out["status"] == "REFUSED"
    assert out["reason"] == "ROOT_EXITED_WITH_RUNNING_NATIVE_OR_EVALUATOR"
    assert out["pre_cleanup_live_owned_descendant_count"] == 1


def test_unseen_short_lived_child_still_blocks_when_job_has_drained():
    job = OwnedJob([[], [], []], total=4)
    out = _observe(job, _ledger(), lambda *a, **k: pytest.fail("no live member"))
    assert out["status"] == "REFUSED"
    assert out["reason"] == "UNSEEN_JOB_HELPER_LIFETIME_PROCESS"


def test_conhost_never_witnessed_or_foreign_child_fails_closed():
    job = OwnedJob([[110, 113], [], [], []], total=4)
    unknown = _observe(job, _ledger(), lambda *a, **k: {
        "status": "REFUSED", "roles": {}, "reasons": ["UNEXPECTED_JOB_CHILD"]})
    assert unknown["reason"] == "POSTEXIT_NATIVE_JOB_LINEAGE_REFUSED"


@pytest.mark.parametrize("status,roles", [
    ("PENDING_SHUTDOWN", {"evaluator": 112}),
    ("OWNER_BOUND_ACTIVE_OBSERVED_ONLY", {"root": 110}),
])
def test_inconsistent_source_process_phase_cannot_produce_natural_witness(status, roles):
    job = OwnedJob([[110, 113]], total=4)
    out = _observe(job, _ledger(), lambda *a, **k: {
        "status": status, "roles": roles, "reasons": []})
    assert out["status"] == "REFUSED"


def test_persistent_legitimate_console_child_not_silently_forced_to_success():
    ledger = _ledger()
    job = OwnedJob([[110, 113]] * 5, total=4)
    def source(owner, **kw):
        kw["ledger"].pid_births[113] = 1003.0
        return {"status": "PENDING_SHUTDOWN", "roles": {"root": 110}, "reasons": []}
    report = _observe(job, ledger, source)
    assert report["status"] == "REFUSED"
    assert report["reason"] == "NATIVE_CONSOLE_POSTEXIT_GRACE_EXCEEDED"
    assert report["polls"] == 4


def test_never_witnessed_original_full_native_lineage_must_not_wait():
    ledger = _ledger()
    ledger.full_active_witnessed = False
    job = OwnedJob([[]], total=3)
    result = _observe(job, ledger, lambda *a, **k: pytest.fail("unapproved"))
    assert result["reason"] == "FULL_ACTIVE_NATIVE_LINEAGE_NEVER_WITNESSED"
