"""Bounded, nonintrusive native Windows/GPU resource witness for REAP admission.

Read-only per-sample snapshots, host-wide VRAM and Windows commit counters.
Never terminate, suspend, tune or attach to another service; do not confuse
the max of a small sample with the actual experiment lifetime's peak.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

SCHEMA = "chowder.reap-v2-windows-native-bounded-resource-witness.v1"


class PERFORMANCE_INFORMATION(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD),
                *( (field, ctypes.c_size_t) for field in (
                    "CommitTotal", "CommitLimit", "CommitPeak", "PhysicalTotal",
                    "PhysicalAvailable", "SystemCache", "KernelTotal", "KernelPaged",
                    "KernelNonpaged", "PageSize")),
                ("HandleCount", wintypes.DWORD), ("ProcessCount", wintypes.DWORD),
                ("ThreadCount", wintypes.DWORD)]


def windows_commit() -> dict:
    if os.name != "nt":
        raise RuntimeError("original native resource witness requires Windows")
    counters = PERFORMANCE_INFORMATION()
    counters.cb = ctypes.sizeof(counters)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetPerformanceInfo.argtypes = [ctypes.POINTER(PERFORMANCE_INFORMATION), wintypes.DWORD]
    psapi.GetPerformanceInfo.restype = wintypes.BOOL
    if not psapi.GetPerformanceInfo(ctypes.byref(counters), counters.cb):
        raise OSError(ctypes.get_last_error(), "GetPerformanceInfo unavailable")
    page = counters.PageSize
    if page < 4096 or counters.CommitLimit <= counters.CommitTotal:
        raise RuntimeError("Windows commit limit reached or invalid")
    return {
        "ram_total_bytes": counters.PhysicalTotal * page,
        "ram_available_bytes": counters.PhysicalAvailable * page,
        "commit_total_bytes": counters.CommitTotal * page,
        "commit_limit_bytes": counters.CommitLimit * page,
        "commit_free_bytes": (counters.CommitLimit - counters.CommitTotal) * page,
        "commit_peak_since_boot_bytes": counters.CommitPeak * page,
    }


def gpu_snapshot() -> list[dict]:
    command = ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
               "--format=csv,noheader,nounits"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=12, check=True)
    rows = []
    for raw in result.stdout.splitlines():
        if not raw.strip():
            continue
        parts = [item.strip() for item in raw.split(",")]
        if len(parts) != 6:
            raise ValueError("unexpected NVIDIA GPU accounting schema")
        index, name, total, used, free, utilization = parts
        vals = (int(total), int(used), int(free), int(utilization))
        if any(v < 0 for v in vals) or not 0 <= vals[-1] <= 100:
            raise ValueError("invalid NVIDIA GPU telemetry")
        if vals[1] > vals[0] or vals[2] > vals[0] or vals[1] + vals[2] > vals[0] + 16:
            raise ValueError("physically impossible NVIDIA GPU memory telemetry")
        # Under Windows WDDM, total-(used+free) can be reserved/unavailable
        # physical VRAM (260 MiB observed in the original native allocator
        # study). This is NOT free capacity and may not be silently credited.
        rows.append({"index": int(index), "name": name,
                     "total_mib": vals[0], "used_mib": vals[1],
                     "free_mib": vals[2], "utilization_percent": vals[3],
                     "wddm_unaccounted_mib": max(0, vals[0] - vals[1] - vals[2])})
    if not rows:
        raise RuntimeError("no NVIDIA GPU telemetry available")
    return rows


def process_snapshot() -> list[dict]:
    running = []
    for proc in psutil.process_iter(["pid", "name", "cmdline", "memory_info"]):
        try:
            name = (proc.info["name"] or "").lower()
            cmd = [str(arg).lower() for arg in (proc.info["cmdline"] or ())]
            if "llama-server" in name:
                classification = "native_llama_server"
            elif name in ("ollama.exe", "ollama"):
                classification = "ollama_service"
            elif name.startswith("python") and any("lb_run.py" in arg for arg in cmd):
                classification = "livebench_evaluator"
            elif name.lower() == "freebuff.exe":
                classification = "freebuff_app"
            else:
                continue
            m = proc.info["memory_info"]
            if m is None:
                raise ValueError("unavailable process resident memory")
            running.append({"pid": proc.pid, "role": classification,
                            "working_set_bytes": m.rss,
                            "private_bytes": getattr(m, "private", None),
                            # Record only frozen-relevant flags; never copy arbitrary
                            # command lines, credentials or Freebuff settings.
                            "frozen_fit_flags_present": all(
                                v in cmd for v in ("--fit", "-np", "-c", "--jinja")
                            ) if classification == "native_llama_server" else None,
                            "one_slot_present": "-np" in cmd and "1" in cmd if
                            classification == "native_llama_server" else None})
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
        except psutil.AccessDenied as exc:
            raise RuntimeError("protected native process inventory not readable") from exc
    return sorted(running, key=lambda r: r["pid"])


def snapshot() -> dict:
    return {"monotonic_ns": time.monotonic_ns(),
            "physical_and_commit": windows_commit(),
            "gpus": gpu_snapshot(), "processes": process_snapshot()}


def summarize(samples: list[dict]) -> dict:
    if not 1 <= len(samples) <= 16:
        raise ValueError("bounded observation requires 1-16 authentic samples")
    for sample in samples:
        if not sample.get("gpus") or not sample.get("physical_and_commit"):
            raise ValueError("missing physical commit or GPU census")
    keys = ("ram_available_bytes", "commit_free_bytes")
    return {"schema": SCHEMA,
            "status": "BOUNDED_HOST_MEASUREMENT_NOT_EXPERIMENT_PEAK",
            "sample_count": len(samples),
            "min_available": {key: min(r["physical_and_commit"][key] for r in samples)
                              for key in keys},
            "sampled_max_host_gpu_used_mib": max(
                gpu["used_mib"] for row in samples for gpu in row["gpus"]),
            "sampled_min_host_gpu_free_mib": min(
                gpu["free_mib"] for row in samples for gpu in row["gpus"]),
            "sampled_max_native_llama_working_set_bytes": max(
                (proc["working_set_bytes"] for row in samples for proc in row["processes"]
                 if proc["role"] == "native_llama_server"), default=None),
            "samples": samples,
            "model_inference_launched": False,
            "isolated_model_peak_ram_or_vram_known": False,
            "full_frozen_serving_stability_proven": False,
            "gen0": {"status": "BLOCKED", "chosen": None, "release_approved": False}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--interval-seconds", type=float, default=1.0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        if (not 1 <= args.samples <= 16
                or not 0 <= args.interval_seconds <= 10):
            raise ValueError("bounded read-only sampling: <=16 samples, <=10 seconds interval")
        # Early exclusive output check avoids running an expensive observation
        # that could not be saved without overwriting another contributor.
        if args.out and (args.out.exists() or args.out.is_symlink()):
            raise ValueError("resource witness output already exists")
        samples = []
        for index in range(args.samples):
            if index:
                time.sleep(args.interval_seconds)
            samples.append(snapshot())
        result = summarize(samples)
        if args.out:
            with args.out.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
        print(json.dumps({key: value for key, value in result.items() if key != "samples"},
                         sort_keys=True, allow_nan=False))
        return 0
    except (OSError, RuntimeError, ValueError, TypeError, subprocess.SubprocessError) as error:
        print(json.dumps({"status": "REFUSED_RESOURCE_WITNESS", "reason": str(error)[:250],
                          "gen0": {"status": "BLOCKED", "chosen": None, "release_approved": False}}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
