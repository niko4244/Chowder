"""Measure REAL active original Q4 AutomationBench Windows RAM/commit/page pressure.

Read-only, process-identity-bound native telemetry. Never launches, terminates,
pauses or alters a model, evaluator, Podman, Ollama, GPU driver or benchmark.
Host-wide GPU figures are NOT the model's exclusive VRAM allocation.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time

import psutil

from chowder_batch import reap_q4_automationbench_full600_resumable_20261027 as run

SCHEMA = "chowder.reap128-v2-original600-windows-native-resource-observation.v1"
GGUF = str(run.q4.GGUF).lower()
ROOT_ARGUMENT = "--out"
NATIVE_ARGUMENT = str(run.q4.SERVER).lower()
MAX_SAMPLES = 120
STARTUP_BOUND = 96 * 1024


class _PerformanceInformation(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("CommitTotal", ctypes.c_size_t),
        ("CommitLimit", ctypes.c_size_t),
        ("CommitPeak", ctypes.c_size_t),
        ("PhysicalTotal", ctypes.c_size_t),
        ("PhysicalAvailable", ctypes.c_size_t),
        ("SystemCache", ctypes.c_size_t),
        ("KernelTotal", ctypes.c_size_t),
        ("KernelPaged", ctypes.c_size_t),
        ("KernelNonpaged", ctypes.c_size_t),
        ("PageSize", ctypes.c_size_t),
        ("HandleCount", ctypes.c_ulong),
        ("ProcessCount", ctypes.c_ulong),
        ("ThreadCount", ctypes.c_ulong),
    ]


def windows_memory() -> dict:
    """Call the native Windows GetPerformanceInfo counters, not virtual reserve."""
    if sys.platform != "win32":
        raise RuntimeError("native Windows committed-memory observations required")
    kernel = _PerformanceInformation()
    kernel.cb = ctypes.sizeof(kernel)
    api = ctypes.WinDLL("psapi", use_last_error=True).GetPerformanceInfo
    api.argtypes = [ctypes.POINTER(_PerformanceInformation), ctypes.c_ulong]
    api.restype = ctypes.c_int
    if not api(ctypes.byref(kernel), kernel.cb):
        raise RuntimeError("native Windows GetPerformanceInfo returned no counters")
    page = int(kernel.PageSize)
    if (page < 1024 or page > 65536 or kernel.CommitTotal > kernel.CommitLimit
            or kernel.PhysicalAvailable > kernel.PhysicalTotal):
        raise RuntimeError("Windows physical/commit native counters violate invariants")
    return {
        "physical_total_bytes": int(kernel.PhysicalTotal) * page,
        "physical_available_bytes": int(kernel.PhysicalAvailable) * page,
        "commit_total_bytes": int(kernel.CommitTotal) * page,
        "commit_limit_bytes": int(kernel.CommitLimit) * page,
        "commit_headroom_bytes": (int(kernel.CommitLimit) - int(kernel.CommitTotal)) * page,
        "commit_peak_bytes": int(kernel.CommitPeak) * page,
        "system_cache_bytes": int(kernel.SystemCache) * page,
        "page_size_bytes": page,
        "process_count": int(kernel.ProcessCount),
    }


def _name_and_argv(p: psutil.Process) -> tuple[str, list[str]]:
    return p.name().lower(), [str(a).lower() for a in p.cmdline()]


def identify_owned(root: Path) -> tuple[psutil.Process, psutil.Process]:
    """Require the original singleton evaluator PID and its own llama process."""
    possible: list[psutil.Process] = []
    for p in psutil.process_iter(("pid", "name")):
        try:
            name, argv = _name_and_argv(p)
            if (name == "python.exe"
                    and "-m" in argv
                    and "chowder_batch.reap_q4_automationbench_full600_resumable_20261027" in argv
                    and ROOT_ARGUMENT in argv
                    and str(root).lower() == argv[argv.index(ROOT_ARGUMENT) + 1]):
                possible.append(p)
        except (IndexError, psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if len(possible) != 1:
        raise RuntimeError("exactly one original600 source-owned evaluator must be running")
    parent = possible[0]
    possible_native = []
    for p in parent.children(recursive=False):
        try:
            name, argv = _name_and_argv(p)
            if (name == "llama-server.exe" and argv[0] == NATIVE_ARGUMENT
                    and "-m" in argv and argv[argv.index("-m") + 1] == GGUF
                    and "--port" in argv
                    and argv[argv.index("--port") + 1] == str(run.q4.PORT)
                    and argv[1:1 + len(run.q4.FLAGS)] ==
                    [x.lower() for x in run.q4.FLAGS]):
                possible_native.append(p)
        except (IndexError, psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if len(possible_native) != 1:
        raise RuntimeError("exact original attached Windows native Q4 process absent")
    return parent, possible_native[0]


def gpu_host() -> dict:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,memory.total,memory.used,memory.free,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=8, check=False,
    )
    if result.returncode:
        raise RuntimeError("physical Nvidia GPU telemetry unavailable")
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError("single original physical Windows GPU expected")
    data = [x.strip() for x in rows[0].split(",")]
    if len(data) != 5 or not data[0].startswith("GPU-"):
        raise RuntimeError("incomplete physical Nvidia VRAM observation")
    total, used, free, busy = map(int, data[1:])
    # Windows WDDM/driver reserved VRAM is real but not accounted for in
    # nvidia-smi's used/free pair. Never silently invent 260MiB model memory.
    reserved = total - used - free
    if not (total >= 12000 and 0 <= used <= total and 0 <= free <= total
            and 0 <= reserved <= 2048 and 0 <= busy <= 100):
        raise RuntimeError("implausible WDDM host GPU memory/compute measurements")
    return {"physical_gpu_uuid": data[0], "host_gpu_total_mib": total,
            "host_gpu_used_mib": used, "host_gpu_free_mib": free,
            "host_gpu_unaccounted_wddm_reserved_mib": reserved,
            "host_gpu_utilization_pct": busy}


def original_native_startup(log: bytes) -> dict:
    """Verify native Q4 architecture and *measured allocated* KV and compute.

    Parse only the startup prefix, never dynamic world-state tool transcripts.
    An explicit zero CPU expert offload is required; model GGUF and all original
    process arguments were already bound before this private read-only step.
    """
    if not isinstance(log, bytes) or not 0 < len(log) <= STARTUP_BOUND:
        raise ValueError("bounded authentic native startup source required")
    text = log.decode("utf-8", errors="strict")
    if ("loading model '" + str(run.q4.GGUF) + "'") not in text:
        raise ValueError("original native startup source is not the GGUF owner")
    if "n_ctx                 = 73728" not in text:
        raise ValueError("source log lacks exact original experimental context")
    if "offloaded 41/41 layers to GPU" not in text or "offloading 39 repeating layers" not in text:
        raise ValueError("GPU placement or CPU expert offload is not independently verified")
    if "n_expert              = 128" not in text or "n_expert_used         = 8" not in text:
        raise ValueError("registered original expert computation changed")
    patterns = {
        "cuda_model_buffer_mib": r"CUDA0 model buffer size\s*=\s*([\d.]+) MiB",
        "cpu_mapped_model_buffer_mib": r"CPU_Mapped model buffer size\s*=\s*([\d.]+) MiB",
        "cuda_kv_buffer_mib": r"CUDA0 KV buffer size\s*=\s*([\d.]+) MiB",
        "cuda_recurrent_state_buffer_mib": r"CUDA0 RS buffer size\s*=\s*([\d.]+) MiB",
        "cuda_compute_buffer_mib": r"CUDA0 compute buffer size\s*=\s*([\d.]+) MiB",
        "host_compute_buffer_mib": r"CUDA_Host compute buffer size\s*=\s*([\d.]+) MiB",
        "prompt_cache_limit_mib": r"prompt cache is enabled, size limit:\s*([\d.]+) MiB",
    }
    amounts = {}
    for key, expression in patterns.items():
        found = re.findall(expression, text)
        if len(found) != 1:
            raise ValueError("missing/ambiguous original allocator startup counter: " + key)
        value = float(found[0])
        if not math.isfinite(value) or not 0 < value < 32768:
            raise ValueError("implausible original allocator startup counter: " + key)
        amounts[key] = value
    if (not 1400 <= amounts["cuda_kv_buffer_mib"] <= 1480
            or not 10600 <= amounts["cuda_model_buffer_mib"] <= 10700
            or not 50 <= amounts["cuda_recurrent_state_buffer_mib"] <= 70
            or not 8000 <= amounts["prompt_cache_limit_mib"] <= 8300
            or "73728 cells,  10 layers" not in text):
        raise ValueError("frozen source-matched original allocator geometry mismatch")
    return {
        "original_gguf_sha256": run.q4.GGUF_SHA,
        "source_startup_prefix_sha256": hashlib.sha256(log).hexdigest(),
        "native_cuda_offloaded_layers": 41,
        "native_cuda_total_layers": 41,
        "native_cpu_expert_offload": False,
        "active_experts_per_token_per_moe_layer": 8,
        "stored_experts_per_moe_layer": 128,
        "full_attention_kv_layers": 10,
        "native_context_tokens": 73728,
        "slots": 1,
        "gpu_kv_type": "f16",
        "measured_original_cuda_allocator_mib": amounts,
        "cache_limit_is_an_allocation_ceiling_not_observed_peak": True,
    }


def sample(root: Path, parent: psutil.Process, native: psutil.Process,
           parent_created: float, native_created: float, owner_sha: str) -> dict:
    """Reject PID reuse, model-root divergence and partial source ownership."""
    if (parent.create_time() != parent_created or native.create_time() != native_created
            or native.ppid() != parent.pid or
            hashlib.sha256((root / "owner.json").read_bytes()).hexdigest() != owner_sha):
        raise RuntimeError("original source owner changed/PID reused during observation")
    seen_parent, seen_native = identify_owned(root)
    if seen_parent.pid != parent.pid or seen_native.pid != native.pid:
        raise RuntimeError("original model/evaluator process changed while sampling")
    win = windows_memory()
    native_info = native.memory_info()
    parent_info = parent.memory_info()
    counters = gpu_host()
    cnt = sum(1 for i in range(600) if (root / f"task-{i:03}" / "receipt.json").is_file())
    counters.update({
        "observed_unix_seconds": time.time(),
        "original_evaluator_pid": parent.pid, "original_native_pid": native.pid,
        "source_owner_sha256": owner_sha, "completed_receipt_files": cnt,
        "native_wset_bytes": int(native_info.rss),
        "native_private_bytes": int(native_info.private),
        "native_page_faults_cumulative": int(native_info.num_page_faults),
        "native_pagefile_bytes": int(native_info.pagefile),
        "evaluator_wset_bytes": int(parent_info.rss),
        "evaluator_private_bytes": int(parent_info.private),
        **win,
    })
    return counters


def summarize(rows: list[dict], *, source_root: Path, owner_sha: str,
              parent_created: float, native_created: float,
              native_allocator: dict | None = None) -> dict:
    if not 2 <= len(rows) <= MAX_SAMPLES:
        raise ValueError("genuine repeated Windows resource samples required")
    times = [x["observed_unix_seconds"] for x in rows]
    if any(not math.isfinite(t) for t in times) or sorted(times) != times:
        raise ValueError("observations are not chronological")
    for key in ("original_evaluator_pid", "original_native_pid", "physical_gpu_uuid",
                "source_owner_sha256"):
        if len({x[key] for x in rows}) != 1:
            raise ValueError("source processes or physical GPU changed within session")
    faults = [r["native_page_faults_cumulative"] for r in rows]
    if any(b < a for a, b in zip(faults, faults[1:])):
        raise ValueError("source process page-fault counter reset / PID reused")
    return {
        "schema": SCHEMA,
        "status": "GENUINE_LIVE_ORIGINAL_Q4_600_WINDOWS_TELEMETRY_NOT_EXCLUSIVE_PEAK",
        "source_root": str(source_root), "source_owner_sha256": owner_sha,
        "original_model_sha256": run.q4.GGUF_SHA,
        "native_original_flags": run.q4.FLAGS,
        "original_evaluator_pid": rows[0]["original_evaluator_pid"],
        "original_evaluator_start_time": parent_created,
        "native_original_pid": rows[0]["original_native_pid"],
        "native_original_start_time": native_created,
        "samples": len(rows), "sampled_period_seconds": times[-1] - times[0],
        "max_native_wset_bytes": max(r["native_wset_bytes"] for r in rows),
        "max_native_private_bytes": max(r["native_private_bytes"] for r in rows),
        "max_native_pagefile_bytes": max(r["native_pagefile_bytes"] for r in rows),
        "native_page_fault_count_delta": faults[-1] - faults[0],
        "source_bound_native_allocator": native_allocator,
        "min_physical_host_available_bytes": min(r["physical_available_bytes"] for r in rows),
        "min_native_windows_commit_headroom_bytes": min(r["commit_headroom_bytes"] for r in rows),
        "max_native_windows_commit_total_bytes": max(r["commit_total_bytes"] for r in rows),
        "max_host_gpu_used_mib": max(r["host_gpu_used_mib"] for r in rows),
        "min_host_gpu_free_mib": min(r["host_gpu_free_mib"] for r in rows),
        "max_host_gpu_unaccounted_wddm_reserved_mib":
            max(r["host_gpu_unaccounted_wddm_reserved_mib"] for r in rows),
        "host_gpu_used_is_model_exclusive": False,
        "first_completed_official_item_count": rows[0]["completed_receipt_files"],
        "last_completed_official_item_count": rows[-1]["completed_receipt_files"],
        "raw_native_windows_observations": rows,
        "frozen_gen0_speed_gate_proven": False,
        "release_approved": False, "chosen": None, "gen0_status": "BLOCKED",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--interval", type=float, default=3.0)
    args = parser.parse_args()
    try:
        if (not args.original_run.is_absolute() or not args.original_run.is_dir()
                or args.original_run.is_symlink() or
                not args.out.is_absolute() or args.out.exists() or args.out.is_symlink()
                or not args.out.parent.is_dir() or args.out.suffix != ".json"
                or type(args.samples) is not int or not 2 <= args.samples <= MAX_SAMPLES
                or not 1 <= args.interval <= 30):
            raise ValueError("original read-only source and exclusive finite telemetry arguments required")
        owner = run._json(args.original_run / "owner.json")
        if owner.get("model_sha256") != run.q4.GGUF_SHA or owner.get("native_flags") != run.q4.FLAGS:
            raise ValueError("original immutable actual-model proof owner is missing")
        owner_hash = hashlib.sha256((args.original_run / "owner.json").read_bytes()).hexdigest()
        p, n = identify_owned(args.original_run)
        logs = list(args.original_run.glob("native-session-*.private.log"))
        if len(logs) != 1 or logs[0].is_symlink():
            raise ValueError("only one original native-session startup log can be owned")
        with logs[0].open("rb") as stream:
            startup = stream.read(STARTUP_BOUND)
        allocator = original_native_startup(startup)
        pt, nt = p.create_time(), n.create_time()
        rows = []
        for i in range(args.samples):
            if i:
                time.sleep(args.interval)
            rows.append(sample(args.original_run, p, n, pt, nt, owner_hash))
        result = summarize(rows, source_root=args.original_run, owner_sha=owner_hash,
                           parent_created=pt, native_created=nt, native_allocator=allocator)
        with args.out.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(result, stream, sort_keys=True, allow_nan=False, indent=2)
            stream.write("\n")
        print(json.dumps({k: v for k, v in result.items() if k != "raw_native_windows_observations"},
                         sort_keys=True, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError,
            psutil.Error, KeyError, TypeError) as error:
        print(json.dumps({"status": "NO_RESOURCE_TELEMETRY_CERTIFICATION",
                          "reason": type(error).__name__ + ": " + str(error)[:350],
                          "native_model_not_interrupted": True,
                          "gen0_status": "BLOCKED"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
