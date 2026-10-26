"""Fail-closed read-only Linux/cgroup REAP model-load preflight.

This is a *provisional* external execution safeguard, not an Amendment 4 gate
or a replacement for the required Windows available-commit witness. Clean file
cache is diagnostic; it is never optimistically added to unused cgroup memory.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

GIB = 1024**3
GPU_FLOOR_MIB = 12288
PHYSICAL_RUNTIME_FLOOR_GIB = 7
CGROUP_RUNTIME_FLOOR_GIB = 8
DISK_RUNTIME_FLOOR_GIB = 8
PROVISIONAL_MODEL_LOAD_HEADROOM_GIB = 24
ORIGINAL_DENSE_ABORT_RECEIPT_SHA = "b2ef325bb9c3165439e669326746fdb9165a5a66975890c99cbfd56dda8c6c0a"
ORIGINAL_DENSE_Q4_SHA = "2c6d9466129e8802839566b35728e2506aaf1e8705cd137009e2f08feb9dd049"
ORIGINAL_FLAGS = ["--fit", "on", "-np", "4", "-c", "16384", "--jinja"]


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


def analyze_original_dense_replay(
    receipt_bytes: bytes, *, current_cgroup_unused_gib: float,
    physical_free_gib: float, gpu_free_mib: list[int], disk_free_gib: float,
) -> dict:
    """Evidence-bound admission for a *new* frozen dense Q4 200-row attempt.

    The original dense first-200 load exhausted container headroom BEFORE
    reaching a running model. An unchanged environment must not be re-launched.
    Model-load demand is a measured lower bound, not a successful-fit estimate.
    An additional 4 GiB operating allowance above a stricter 10 GiB cgroup
    stop guard is an admission heuristic, never evidence of model quality/fit.
    """
    if (not isinstance(receipt_bytes, bytes) or len(receipt_bytes) > 2 * 1024 * 1024
            or hashlib.sha256(receipt_bytes).hexdigest() != ORIGINAL_DENSE_ABORT_RECEIPT_SHA):
        raise ValueError("only the exact original dense aborted experiment receipt is admissible")
    record = json.loads(receipt_bytes)
    if (record.get("schema") != "reap-v2-dense-q4-kaggle-first200-provisional.v1"
            or record.get("status") != "REFUSED_OR_ABORTED"
            or record.get("data", {}).get("dense_q4_sha256") != ORIGINAL_DENSE_Q4_SHA
            or record.get("data", {}).get("requested_cohort_n") != 200
            or record.get("flags") != ORIGINAL_FLAGS
            or "10 GiB cgroup" not in record.get("error", "")):
        raise ValueError("dense load observation lacks frozen model/flags/abort identity")
    samples = record.get("samples")
    initial = record.get("initial", {}).get("cgroup_unused_gib")
    if not isinstance(samples, list) or len(samples) < 2:
        raise ValueError("at least two observed dense load resource samples required")

    def valid_amount(value: object, name: str) -> float:
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid/nonfinite {name}")
        return float(value)

    start = valid_amount(initial, "original initial unused cgroup GiB")
    smallest = min(valid_amount(row["cgroup_unused_gib"], "original sample")
                   for row in samples if isinstance(row, dict))
    if start <= smallest:
        raise ValueError("original aborted dense receipt has no measured cgroup consumption")
    consumed = start - smallest
    # Observed *incomplete* load + 10 GiB stricter stop + 4 GiB post-load
    # allowance. This is a minimum for contemplating another research test,
    # NOT a prediction of model fit, Windows available commit, or speed PASS.
    minimum_start = consumed + 10 + 4
    current = valid_amount(current_cgroup_unused_gib, "current cgroup available GiB")
    ram = valid_amount(physical_free_gib, "current available RAM GiB")
    disk = valid_amount(disk_free_gib, "current disk free GiB")
    if not isinstance(gpu_free_mib, list) or not gpu_free_mib:
        raise ValueError("no independently measured GPU inventory")
    if any(type(v) is not int or v < 0 for v in gpu_free_mib):
        raise ValueError("invalid GPU free VRAM measurement")
    reasons = []
    if current < minimum_start:
        reasons.append("new cgroup headroom below observed incomplete dense load plus guarded reserve")
    if ram < 7:
        reasons.append("below original 7 GiB free RAM runtime floor")
    if disk < 8:
        reasons.append("below independent 8 GiB disk guard")
    if any(v < GPU_FLOOR_MIB for v in gpu_free_mib):
        reasons.append("below original registered 12 GiB GPU start admission")
    return {
        "status": "REFUSED" if reasons else "MAY_ATTEMPT_NEW_PROVISIONAL_DENSE_LOAD_ONLY",
        "reasons": reasons,
        "original_dense_200_speed_certified": False,
        "original_windows_commit_attested": False,
        "source_receipt_sha256": ORIGINAL_DENSE_ABORT_RECEIPT_SHA,
        "observed_incomplete_load_consumed_gib": consumed,
        "minimum_provisional_next_initial_headroom_gib": minimum_start,
        "original_runtime_ram_floor_gib": 7,
        "original_runtime_commit_floor_gib": 8,
        "no_process_launched": True,
    }
