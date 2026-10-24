"""Fail-closed host-resource admission and observer runtime decisions.

Only pure decisions and read-only snapshots live here. Never kill processes,
stop services or initialize CUDA in this module. An external supervisor may
terminate *its own* launched observer on an ABORT decision.
"""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

GIB = 1024**3
MIN_ADMIT_RAM_GIB = 16.0
MIN_ADMIT_COMMIT_GIB = 24.0
MIN_ADMIT_GPU_FREE_GIB = 12.0
MAX_ADMIT_RAM_DECLINE_GIB = 1.0
MAX_FOREIGN_BIRTHS_60S = 11
MIN_RUNTIME_RAM_GIB = 5.0
MIN_RUNTIME_COMMIT_GIB = 6.0
MIN_RUNTIME_DISK_C_GIB = 1.5
MIN_RUNTIME_DISK_F_GIB = 4.0
MIN_RUNTIME_GPU_FREE_GIB = 0.65
MAX_RUNTIME_GPU_TEMP_C = 88.0
_MAX_PROCESS_RECORDS = 5_000


@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    started: float
    parent_pid: int
    name: str
    private_bytes: int
    working_set_bytes: int

    @property
    def key(self) -> tuple[int, float]:
        return self.pid, self.started


@dataclass(frozen=True)
class Snapshot:
    timestamp: float
    ram_free_gib: float
    virtual_free_gib: float
    diskC_free_gib: float
    diskF_free_gib: float
    gpu_free_gib: float | None
    gpu_temp_c: float | None
    processes: tuple[ProcessInfo, ...] = ()
    census_errors: int = 0
    services_conflict: bool = False

    def as_record(self) -> dict:
        result = asdict(self)
        result["timestamp_utc"] = __import__("datetime").datetime.fromtimestamp(
            self.timestamp, tz=__import__("datetime").timezone.utc
        ).isoformat()
        result["processes"] = [asdict(p) for p in self.processes]
        return result


@dataclass(frozen=True)
class Decision:
    action: str  # ADMIT, REFUSE, CONTINUE, ABORT
    reasons: tuple[str, ...]
    new_foreign_processes_60s: int = 0
    new_foreign_private_gib: float = 0.0
    ram_decline_60s_gib: float = 0.0
    warnings: tuple[str, ...] = ()

    def as_record(self) -> dict:
        return asdict(self)


def _memory_windows() -> tuple[float, float]:
    """Windows available physical memory and remaining system commit charge."""
    if sys.platform != "win32":
        raise RuntimeError("full host admission requires Windows memory counters")

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    return status.ullAvailPhys / GIB, status.ullAvailPageFile / GIB


def _gpu_snapshot() -> tuple[float, float]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free,temperature.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=8,
    )
    line = result.stdout.splitlines()[0]
    free, temp = (float(v.strip()) for v in line.split(","))
    return free / 1024, temp


def _services_conflict(ports: Iterable[int] = (8767, 11434)) -> bool:
    for port in ports:
        with socket.socket() as sock:
            sock.settimeout(0.3)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return True
    return False


def _process_census() -> tuple[tuple[ProcessInfo, ...], int]:
    """Read-only best-effort census; no process handle remains open."""
    try:
        import psutil
    except ImportError as exc:
        raise RuntimeError("psutil required for per-PID resource supervision") from exc
    records: list[ProcessInfo] = []
    inaccessible = 0
    for proc in psutil.process_iter(["pid", "ppid", "name", "create_time",
                                    "memory_info"]):
        try:
            info = proc.info
            if info["create_time"] is None or info["ppid"] is None:
                inaccessible += 1
                continue
            memory = info["memory_info"]
            if memory is None:
                inaccessible += 1
                continue
            records.append(ProcessInfo(
                pid=int(info["pid"]), parent_pid=int(info["ppid"]),
                started=float(info["create_time"]),
                name=str(info["name"] or "?")[:80],
                private_bytes=int(getattr(memory, "private", getattr(memory, "vms", 0))),
                working_set_bytes=int(getattr(memory, "rss", 0)),
            ))
        except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError, TypeError):
            inaccessible += 1
        if len(records) > _MAX_PROCESS_RECORDS:
            raise RuntimeError("process census exceeded safe bounded limit")
    return tuple(records), inaccessible


def collect_snapshot(*, now: float | None = None) -> Snapshot:
    ram, commit = _memory_windows()
    gpu_free, gpu_temp = _gpu_snapshot()
    processes, errors = _process_census()
    return Snapshot(
        timestamp=time.time() if now is None else now,
        ram_free_gib=ram, virtual_free_gib=commit,
        diskC_free_gib=shutil.disk_usage("C:\\").free / GIB,
        diskF_free_gib=shutil.disk_usage("F:\\").free / GIB,
        gpu_free_gib=gpu_free, gpu_temp_c=gpu_temp,
        processes=processes, census_errors=errors,
        services_conflict=_services_conflict(),
    )


def _excluded_pids(processes: tuple[ProcessInfo, ...],
                   owner_pids: frozenset[int]) -> set[int]:
    excluded = set(owner_pids)
    by_parent: dict[int, list[int]] = {}
    for p in processes:
        by_parent.setdefault(p.parent_pid, []).append(p.pid)
    stack = list(owner_pids)
    while stack:
        for child in by_parent.get(stack.pop(), []):
            if child not in excluded:
                excluded.add(child)
                stack.append(child)
    return excluded


def _foreign_births(snapshot: Snapshot, owner_pids: frozenset[int]) -> list[ProcessInfo]:
    excluded = _excluded_pids(snapshot.processes, owner_pids)
    return [
        p for p in snapshot.processes
        if p.pid not in excluded and 0 <= snapshot.timestamp - p.started <= 60
    ]


def _ram_decline(samples: list[Snapshot], *, seconds: float) -> float:
    latest = samples[-1]
    # A 30-second supervisor has a 1-2 second sampling jitter; select the
    # nearest preceding sample between (seconds-18) and (seconds+18) seconds.
    matches = [s for s in samples[:-1]
               if seconds - 18 <= latest.timestamp - s.timestamp <= seconds + 18]
    if not matches:
        return 0.0
    old = min(matches, key=lambda s: abs(latest.timestamp - s.timestamp - seconds))
    return max(0.0, old.ram_free_gib - latest.ram_free_gib)


def assess_admission(samples: list[Snapshot],
                     *, owner_pids: frozenset[int] = frozenset()) -> Decision:
    if len(samples) != 3 or samples[0].timestamp >= samples[1].timestamp or (
        samples[1].timestamp >= samples[2].timestamp
    ):
        return Decision("REFUSE", ("need_three_ordered_host_samples",))
    reasons = []
    if samples[-1].timestamp - samples[0].timestamp < 55:
        reasons.append("need_60s_prelaunch_observation")
    if min(s.ram_free_gib for s in samples) < MIN_ADMIT_RAM_GIB:
        reasons.append("physical_ram_below_16_gib")
    if min(s.virtual_free_gib for s in samples) < MIN_ADMIT_COMMIT_GIB:
        reasons.append("free_commit_below_24_gib")
    if samples[0].ram_free_gib - samples[-1].ram_free_gib > MAX_ADMIT_RAM_DECLINE_GIB:
        reasons.append("physical_ram_falling_over_60s")
    foreign = _foreign_births(samples[-1], owner_pids)
    if len(foreign) > MAX_FOREIGN_BIRTHS_60S:
        reasons.append("foreign_process_fanout_over_60s")
    if any(s.services_conflict for s in samples):
        reasons.append("protected_service_ports_listening")
    if any(s.census_errors >= 25 for s in samples):
        reasons.append("process_telemetry_substantially_incomplete")
    if any(s.gpu_free_gib is None or s.gpu_temp_c is None for s in samples):
        reasons.append("gpu_telemetry_unavailable")
    if any(s.gpu_free_gib is not None and s.gpu_free_gib < MIN_ADMIT_GPU_FREE_GIB
           for s in samples):
        reasons.append("gpu_free_below_12_gib")
    if any(s.diskC_free_gib < 2 or s.diskF_free_gib < 14 for s in samples):
        reasons.append("insufficient_checkpoint_and_log_disk_headroom")
    return Decision("REFUSE" if reasons else "ADMIT", tuple(reasons),
                    len(foreign), sum(p.private_bytes for p in foreign) / GIB,
                    _ram_decline(samples, seconds=60))


def assess_runtime(samples: list[Snapshot],
                   *, owner_pids: frozenset[int] = frozenset()) -> Decision:
    if not samples:
        return Decision("ABORT", ("host_telemetry_missing",))
    current = samples[-1]
    foreign = _foreign_births(current, owner_pids)
    private_gib = sum(p.private_bytes for p in foreign) / GIB
    decline60 = _ram_decline(samples, seconds=60)
    decline30 = _ram_decline(samples, seconds=30)
    reasons = []
    warnings = []
    if len(foreign) >= 12:
        warnings.append("foreign_process_birth_burst")
    if private_gib >= 1:
        warnings.append("foreign_new_processes_private_memory_over_1_gib")
    if current.ram_free_gib < MIN_RUNTIME_RAM_GIB:
        reasons.append("physical_ram_below_5_gib")
    if current.ram_free_gib < 6 and decline60 >= 2.5:
        reasons.append("rapid_host_ram_decline")
    if len(foreign) >= 12 and current.ram_free_gib < 7 and decline30 >= 1:
        reasons.append("competing_process_fanout_and_ram_decline")
    if current.virtual_free_gib < MIN_RUNTIME_COMMIT_GIB:
        reasons.append("free_commit_below_6_gib")
    if current.diskC_free_gib < MIN_RUNTIME_DISK_C_GIB:
        reasons.append("disk_c_below_1_5_gib")
    if current.diskF_free_gib < MIN_RUNTIME_DISK_F_GIB:
        reasons.append("disk_f_below_4_gib")
    if current.gpu_free_gib is None or current.gpu_free_gib < MIN_RUNTIME_GPU_FREE_GIB:
        reasons.append("gpu_memory_unavailable_or_low")
    if current.gpu_temp_c is None or current.gpu_temp_c >= MAX_RUNTIME_GPU_TEMP_C:
        reasons.append("gpu_telemetry_unavailable_or_hot")
    if current.services_conflict:
        reasons.append("protected_services_reactivated")
    if current.census_errors >= 25:
        reasons.append("process_telemetry_substantially_incomplete")
    return Decision("ABORT" if reasons else "CONTINUE", tuple(reasons),
                    len(foreign), private_gib, decline60, tuple(warnings))


def replay_timeline(path: Path) -> dict:
    """Offline replay; NO live process/GPU probes or launching anything."""
    items = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()
             if line.strip()]
    if not items:
        raise ValueError("resource replay timeline is empty")
    from datetime import datetime
    history: list[Snapshot] = []
    first_abort = None
    for i, raw in enumerate(items, 1):
        stamp = raw["timestamp_utc"]
        timestamp = datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp()
        history.append(Snapshot(
            timestamp=timestamp, ram_free_gib=float(raw["ram_free_gib"]),
            virtual_free_gib=float(raw["virtual_free_gib"]),
            diskC_free_gib=float(raw["diskC_free_gib"]),
            diskF_free_gib=float(raw["diskF_free_gib"]),
            gpu_free_gib=raw.get("gpu_free_gib"),
            gpu_temp_c=raw.get("gpu_temp_c"),
        ))
        decision = assess_runtime(history[-4:])
        if first_abort is None and decision.action == "ABORT":
            first_abort = {"sample": i, "timestamp_utc": stamp,
                           "decision": decision.as_record()}
    return {"samples": len(history), "first_abort": first_abort}
