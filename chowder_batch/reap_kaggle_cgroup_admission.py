"""Fail-closed read-only Linux/cgroup REAP model-load preflight.

This is a *provisional* external execution safeguard, not an Amendment 4 gate
or a replacement for the required Windows available-commit witness. Clean file
cache is diagnostic; it is never optimistically added to unused cgroup memory.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

GIB = 1024**3
GPU_FLOOR_MIB = 12288
PHYSICAL_RUNTIME_FLOOR_GIB = 7
CGROUP_RUNTIME_FLOOR_GIB = 8
DISK_RUNTIME_FLOOR_GIB = 8
PROVISIONAL_MODEL_LOAD_HEADROOM_GIB = 24


def _counter(raw: str, *, label: str) -> int:
    if not isinstance(raw, str) or not re.fullmatch(r"[0-9]+", raw.strip()):
        raise ValueError(f"{label} must be a bounded nonnegative byte counter")
    return int(raw.strip())


def _stat_map(raw: str, *, name: str) -> dict[str, int]:
    if not isinstance(raw, str):
        raise ValueError(f"{name} is not a text counter file")
    values: dict[str, int] = {}
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) != 2 or parts[0] in values or not re.fullmatch(r"[a-z_]+", parts[0]):
            raise ValueError(f"{name} has malformed or duplicate counters")
        values[parts[0]] = _counter(parts[1], label=parts[0])
    return values


def analyze_cgroup(
    *, limit: str, current: str, peak: str, memory_stat: str,
    memory_events: str, physical_free_gib: float, gpu_free_mib: list[int],
    disk_free_gib: float,
) -> dict:
    """Report original floors and conservative, separately labeled admission.

    `memory.max` must be finite. A no-swap Linux cgroup is *not* a Windows
    Windows available-commit measurement. Source/prompt/model identity is
    checked elsewhere and is not inferred by this function.
    """
    cap = _counter(limit, label="cgroup memory.max")
    used = _counter(current, label="cgroup memory.current")
    high = _counter(peak, label="cgroup memory.peak")
    if cap == 0 or used > cap or high < used:
        raise ValueError("cgroup memory counters disagree")
    stat = _stat_map(memory_stat, name="memory.stat")
    events = _stat_map(memory_events, name="memory.events")
    if "file" not in stat or "anon" not in stat or "oom_kill" not in events:
        raise ValueError("required cgroup memory attribution unavailable")
    if stat["file"] > used or stat["anon"] > used:
        raise ValueError("invalid file/anonymous cgroup attribution")
    for name, value in (("physical RAM", physical_free_gib), ("disk", disk_free_gib)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} is nonfinite/invalid")
    if not isinstance(gpu_free_mib, list) or not gpu_free_mib:
        raise ValueError("the actual GPU inventory is unavailable")
    if any(type(x) is not int or x < 0 for x in gpu_free_mib):
        raise ValueError("invalid GPU resource figure")

    unused = (cap - used) / GIB
    reasons = []
    if physical_free_gib < PHYSICAL_RUNTIME_FLOOR_GIB:
        reasons.append("physical RAM below original runtime floor")
    if unused < CGROUP_RUNTIME_FLOOR_GIB:
        reasons.append("cgroup headroom below conservative 8 GiB commit surrogate")
    if physical_free_gib < PROVISIONAL_MODEL_LOAD_HEADROOM_GIB or unused < PROVISIONAL_MODEL_LOAD_HEADROOM_GIB:
        reasons.append("below conservative provisional 24 GiB native-load admission")
    if any(x < GPU_FLOOR_MIB for x in gpu_free_mib):
        reasons.append("GPU free VRAM below registered start admission")
    if disk_free_gib < DISK_RUNTIME_FLOOR_GIB:
        reasons.append("disk below model-run reserve")
    if events["oom_kill"]:
        reasons.append("container memory.events already records an OOM kill")
    return {
        "status": "REFUSED" if reasons else "MAY_ATTEMPT_PROVISIONAL_SMOKE_ONLY",
        "reasons": reasons,
        "cgroup_unused_gib": unused,
        "cgroup_peak_gib": high / GIB,
        "file_cache_gib_descriptive_not_guaranteed_reclaimable": stat["file"] / GIB,
        "anonymous_gib": stat["anon"] / GIB,
        "oom_kill_count": events["oom_kill"],
        "registered_windows_commit_attested": False,
        "scientific_selection": "UNKNOWN",
        "no_process_launched": True,
    }


def read_cgroup_files(base: Path = Path("/sys/fs/cgroup")) -> dict[str, str]:
    """Read only known control files, never write to cgroup or reclaim knobs."""
    return {
        "limit": (base / "memory.max").read_text(encoding="ascii").strip(),
        "current": (base / "memory.current").read_text(encoding="ascii").strip(),
        "peak": (base / "memory.peak").read_text(encoding="ascii").strip(),
        "memory_stat": (base / "memory.stat").read_text(encoding="ascii"),
        "memory_events": (base / "memory.events").read_text(encoding="ascii"),
    }
