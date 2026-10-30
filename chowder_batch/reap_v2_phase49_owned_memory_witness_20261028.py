"""Bounded Windows private-Job memory observations for one original native run.

Read-only psutil counters are collected ONLY for the live identities already
established by the fresh native Job/argv/parent census. They do not measure
per-process GPU allocations, actual disk paging, or a whole-lifetime peak.
No process is started, terminated, suspended, configured or signaled here.
"""

from __future__ import annotations

import math
import sys
import time

import psutil

from chowder_batch import reap_v2_owner_bound_runtime_supervisor_20261027 as owner_api
from chowder_batch import reap_v2_live_job_lineage_guard_20261028 as lineage

SCHEMA = "chowder.reap-v2-phase49-owned-native-memory-witness.v1"
ROLES = frozenset({"root", "server", "evaluator"})


class MemoryWitnessRefused(ValueError):
    """Fail closed on missing original owner identity or a torn process sample."""


def _finite_counter(value: object, field: str) -> int:
    if type(value) is not int or value < 0 or not math.isfinite(value):
        raise MemoryWitnessRefused("UNTRUSTED_MEMORY_COUNTER_" + field)
    return value


def sample_owned_memory(owner: object, ledger: object, *, process_factory=None) -> dict:
    """Sample precisely the registered active private Job role identities.

    Requires real Windows owner handles in production. process_factory is
    injected ONLY for deterministic CPU unit tests, never an approval source.
    """
    if (sys.platform != "win32"
            or type(owner) is not owner_api._WindowsOwnedJob
            or not owner.active or not owner.job or not owner.process
            or type(ledger) is not lineage.LineageLedger
            or ledger.full_active_witnessed is not True
            or set(ledger.roles) != ROLES):
        raise MemoryWitnessRefused("ACTIVE_NATIVE_PRIVATE_JOB_ROLES_REQUIRED")
    if not owner.owner_valid():
        raise MemoryWitnessRefused("PRIVATE_OWNER_HANDLE_OR_BIRTH_CHANGED")
    factory = process_factory if process_factory is not None else psutil.Process
    records = {}
    try:
        before = set(owner.member_ids())
        for role in sorted(ROLES):
            pid, birth = ledger.roles[role]
            if (type(pid) is not int or pid <= 0
                    or type(birth) not in (int, float)
                    or not math.isfinite(birth)
                    or pid not in before or not owner.is_member(pid)):
                raise MemoryWitnessRefused("ROLE_PID_NOT_OWNED_" + role)
            process = factory(pid)
            if abs(process.create_time() - birth) > 0.002:
                raise MemoryWitnessRefused("ROLE_PID_REUSED_" + role)
            info = process.memory_info()
            counters = {}
            for key in ("rss", "peak_wset", "private", "pagefile", "num_page_faults"):
                counters[key] = _finite_counter(getattr(info, key, None), key)
            if counters["peak_wset"] < counters["rss"]:
                raise MemoryWitnessRefused("WINDOWS_PEAK_WORKING_SET_BELOW_CURRENT")
            if not owner.is_member(pid) or abs(process.create_time() - birth) > 0.002:
                raise MemoryWitnessRefused("ROLE_CHANGED_DURING_MEMORY_SAMPLE_" + role)
            records[role] = {"pid": pid, "create_time": birth, **counters}
        after = set(owner.member_ids())
        if before != after or not owner.owner_valid():
            raise MemoryWitnessRefused("PRIVATE_JOB_MEMBERS_CHANGED_DURING_MEMORY_SAMPLE")
    except (psutil.Error, OSError, ValueError, OverflowError, TypeError, AttributeError) as error:
        if isinstance(error, MemoryWitnessRefused):
            raise
        raise MemoryWitnessRefused("PRIVATE_JOB_READ_ONLY_COUNTERS_UNAVAILABLE") from error
    return {
        "schema": SCHEMA,
        "status": "OWNED_LIVE_MEMORY_SAMPLE_NOT_SCIENTIFIC_PEAK",
        "sample_monotonic": time.monotonic(),
        "roles": records,
        "sampled_job_working_set_sum_bytes": sum(x["rss"] for x in records.values()),
        "sampled_job_private_commit_sum_bytes": sum(x["private"] for x in records.values()),
        "native_server_sampled_working_set_bytes": records["server"]["rss"],
        "native_server_reported_peak_working_set_so_far_bytes": records["server"]["peak_wset"],
        "native_server_private_commit_bytes": records["server"]["private"],
        "native_server_page_faults_cumulative": records["server"]["num_page_faults"],
        "hard_page_faults_or_pagefile_io_measured": False,
        "native_server_gpu_allocation_measured": False,
        "whole_lifetime_peak_proven": False,
        "gen0": {"status": "BLOCKED", "chosen": None, "release_approved": False},
    }


def summarize_samples(samples: list[dict]) -> dict:
    """Aggregates only genuine previously witnessed active role samples."""
    if not isinstance(samples, list) or not samples:
        return {
            "status": "UNKNOWN_NO_FULL_JOB_MEMORY_SAMPLE",
            "measured_sample_count": 0,
            "gen0": {"status": "BLOCKED", "chosen": None, "release_approved": False},
        }
    times = []
    for sample in samples:
        if (not isinstance(sample, dict)
                or sample.get("schema") != SCHEMA
                or sample.get("status") != "OWNED_LIVE_MEMORY_SAMPLE_NOT_SCIENTIFIC_PEAK"
                or set(sample.get("roles", {})) != ROLES
                or sample.get("whole_lifetime_peak_proven") is not False):
            raise MemoryWitnessRefused("MEMORY_SAMPLE_SOURCE_OR_ROLE_DRIFT")
        tick = sample.get("sample_monotonic")
        if type(tick) not in (int, float) or not math.isfinite(tick) or (times and tick <= times[-1]):
            raise MemoryWitnessRefused("MEMORY_SAMPLE_CLOCK_INVALID")
        times.append(tick)
    return {
        "status": "BOUNDED_NATIVE_JOB_MEMORY_SAMPLES_NOT_ISOLATED_GPU_PEAK",
        "measured_sample_count": len(samples),
        "duration_seconds": times[-1] - times[0],
        "max_sampled_job_working_set_sum_bytes": max(s["sampled_job_working_set_sum_bytes"] for s in samples),
        "max_sampled_job_private_commit_sum_bytes": max(s["sampled_job_private_commit_sum_bytes"] for s in samples),
        "max_sampled_native_working_set_bytes": max(s["native_server_sampled_working_set_bytes"] for s in samples),
        "max_native_reported_peak_working_set_so_far_bytes": max(
            s["native_server_reported_peak_working_set_so_far_bytes"] for s in samples
        ),
        "max_sampled_native_private_commit_bytes": max(s["native_server_private_commit_bytes"] for s in samples),
        "native_gpu_allocation_peak": None,
        "whole_lifetime_peak_proven": False,
        "hard_page_faults_or_pagefile_disk_io_proven": False,
        "gen0": {"status": "BLOCKED", "chosen": None, "release_approved": False},
    }
