"""Read-only bounded natural Windows Job shutdown after frozen wrapper exit.

An owned suspended-process Job may temporarily contain the real System32
conhost.exe *after* the Python root exits with zero. Never kill that OS helper
and then mistake forced post-close extinction for a clean native experiment.
Only a previously witnessed full native lineage may enter this observer.

This module has NO terminate/kill/close methods and cannot authorize a model
launch, benchmark row, statistical gate, release or Gen-0 selection.
"""

from __future__ import annotations

import time
from pathlib import Path

from chowder_batch import reap_v2_live_job_lineage_guard_20261028 as lineage

SCHEMA = "chowder.reap-v2-phase50-bounded-natural-owned-job-drain.v1"
MAX_POLLS = 40
POLL_SECONDS = 0.05


def _report(status: str, reason: str | None, members: list[int],
            accounting: dict | None, checks: int) -> dict:
    return {
        "schema": SCHEMA, "status": status,
        "reason": reason,
        "pre_cleanup_live_owned_descendant_count": len(members),
        "pre_cleanup_private_job_lifetime_accounting": accounting,
        "all_private_job_lifetime_processes_source_witnessed": status == "NATURALLY_DRAINED",
        "pre_cleanup_natural_owned_tree_extinction_witnessed": status == "NATURALLY_DRAINED",
        "polls": checks,
        "execution_approved": False,
        "gen0": {"status": "BLOCKED", "chosen": None, "release_approved": False},
    }


def witness_natural_drain(
    owner: object,
    *,
    ledger: lineage.LineageLedger,
    study: object,
    model: Path,
    server: Path,
    output: Path,
    port: int,
    capture=None,
    refresh_births=None,
    polls: int = MAX_POLLS,
    sleep=None,
) -> dict:
    """Allow only natural root/conhost drain, never running native/eval children.

    Source of lineage evidence is the original private Job handle plus fresh
    Win32 process-image/parent/argv evidence, not a JSON receipt. A short-lived
    missing child is caught by native Job lifetime totals and fails closed.
    """
    if (not isinstance(ledger, lineage.LineageLedger)
            or ledger.full_active_witnessed is not True
            or type(polls) is not int or not 1 <= polls <= MAX_POLLS):
        return _report("REFUSED", "FULL_ACTIVE_NATIVE_LINEAGE_NEVER_WITNESSED", [], None, 0)
    probe = capture if capture is not None else lineage.capture_live_job_lineage
    sleeper = sleep if sleep is not None else time.sleep
    last_members: list[int] = []
    latest: dict | None = None
    for index in range(polls):
        try:
            now = owner.member_ids()
            latest = owner.lifetime_process_counts()
            if (not isinstance(now, list) or len(now) != len(set(now))
                    or not isinstance(latest, dict)
                    or type(latest.get("active_processes")) is not int
                    or type(latest.get("total_processes")) is not int
                    or latest["active_processes"] < 0
                    or latest["total_processes"] < latest["active_processes"]):
                return _report("REFUSED", "PRIVATE_JOB_POSTEXIT_ACCOUNTING_INVALID",
                               last_members, latest, index + 1)
            last_members = [pid for pid in now if pid != owner.pid]
            if not now and latest["active_processes"] == 0:
                # Two independently fresh native handle reads ensure this is
                # not a torn empty census while a conhost process is exiting.
                second = owner.member_ids()
                second_counts = owner.lifetime_process_counts()
                if (second or second_counts["active_processes"] != 0
                        or second_counts["total_processes"] != latest["total_processes"]):
                    continue
                if refresh_births is not None:
                    checkpoint = refresh_births()
                    if (not isinstance(checkpoint, dict)
                            or checkpoint.get("status") ==
                            "REFUSED"):
                        return _report(
                            "REFUSED", "PRIVATE_JOB_COMPLETION_BIRTH_SOURCE_REFUSED",
                            [], latest, index + 1,
                        )
                if latest["total_processes"] != len(ledger.pid_births):
                    # The OS may deliver already-queued NEW_PROCESS packets
                    # just after the final Job active count became zero.
                    # Only a newly source-vetted PID+birth/image/argv from
                    # the original completion port can close this deficit;
                    # merely waiting for or counting a raw event cannot.
                    if refresh_births is not None and index + 1 < polls:
                        sleeper(POLL_SECONDS)
                        continue
                    return _report("REFUSED", "UNSEEN_JOB_HELPER_LIFETIME_PROCESS",
                                   [], latest, index + 1)
                return _report("NATURALLY_DRAINED", None, [], latest, index + 1)
            proof = probe(
                owner, study=study, model=model, server=server,
                output=output, port=port, phase="shutdown", ledger=ledger,
            )
            if not isinstance(proof, dict) or proof.get("status") == "REFUSED":
                return _report(
                    "REFUSED", "POSTEXIT_NATIVE_JOB_LINEAGE_REFUSED",
                    last_members, latest, index + 1,
                )
            if proof.get("status") not in (
                "PENDING_SHUTDOWN", "PENDING_TRANSIENT_JOB_CENSUS",
                "JOB_MEMBERS_EMPTY_NEEDS_POSTCLOSE_WITNESS",
            ):
                return _report(
                    "REFUSED", "UNRECOGNIZED_POSTEXIT_JOB_LINEAGE",
                    last_members, latest, index + 1,
                )
            roles = proof.get("roles")
            if (not isinstance(roles, dict)
                    or any(name in roles for name in ("server", "evaluator"))):
                return _report(
                    "REFUSED", "ROOT_EXITED_WITH_RUNNING_NATIVE_OR_EVALUATOR",
                    last_members, latest, index + 1,
                )
        except (OSError, ValueError, TypeError, AttributeError, RuntimeError):
            return _report("REFUSED", "NATIVE_POSTEXIT_JOB_CENSUS_UNAVAILABLE",
                           last_members, latest, index + 1)
        if index + 1 < polls:
            sleeper(POLL_SECONDS)
    return _report("REFUSED", "NATIVE_CONSOLE_POSTEXIT_GRACE_EXCEEDED",
                   last_members, latest, polls)
