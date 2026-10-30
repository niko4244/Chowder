"""Supervise only this run's registered CPU REAP128_v2 BF16 construction.

Default is admission-only. --execute starts one OWNED Python pruner, with no
CUDA-visible device. No existing service, unrelated process, or original
checkpoint is ever stopped, modified or cleaned up. A resource abort signals
only the process we launched. A crashed job may leave a private staging folder;
it must never be silently treated as a published model.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from chowder_batch.reap_build_preflight import build_plan

GIB = 1024 ** 3
RAM_BUFFER_GIB = 8.0
COMMIT_BUFFER_GIB = 8.0
RUNTIME_RAM_FLOOR_GIB = 7.0
RUNTIME_COMMIT_FLOOR_GIB = 8.0
RUNTIME_DISK_FLOOR_GIB = 8.0
POLL_SECONDS = 5


@dataclass(frozen=True)
class HostState:
    ram_available_gib: float
    commit_available_gib: float
    output_disk_free_gib: float
    process_count: int


def host_state(out_parent: Path) -> HostState:
    """Read-only OS physical, commit and storage counters; not pagefile proxy."""
    import shutil

    if sys.platform != "win32":
        raise RuntimeError("frozen model construction supervisor is Windows-only")

    class Performance(ctypes.Structure):
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

    perf = Performance()
    perf.cb = ctypes.sizeof(Performance)
    if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(perf), perf.cb):
        raise ctypes.WinError()
    return HostState(
        ram_available_gib=perf.PhysicalAvailable * perf.PageSize / GIB,
        commit_available_gib=(perf.CommitLimit - perf.CommitTotal) * perf.PageSize / GIB,
        output_disk_free_gib=shutil.disk_usage(out_parent).free / GIB,
        process_count=perf.ProcessCount,
    )


def memory_required_gib(sizing: dict) -> float:
    """Conservative per-shard payload peak: largest live source + output."""
    # Each output shard is at most source shard bytes; source input geometry
    # bounds are independently enforced by the registered original pin.
    # The largest original weight shard is known from the versioned upstream
    # source pin, not estimated from the current host.
    from chowder_batch.reap_original_bf16_pin import load_original_pin

    max_source = max(int(v["size"]) for v in load_original_pin()["weight_shards"].values())
    max_output = int(sizing["projected_largest_output_shard_bytes"])
    return (max_source + max_output) / GIB


def admit(state: HostState, *, peak_gib: float, sizing: dict) -> tuple[str, ...]:
    problems = []
    if state.ram_available_gib < peak_gib + RAM_BUFFER_GIB:
        problems.append("available physical RAM is below estimated peak plus 8 GiB reserve")
    if state.commit_available_gib < peak_gib + COMMIT_BUFFER_GIB:
        problems.append("available Windows commit is below estimated peak plus 8 GiB reserve")
    needed = (sizing["projected_pruned_tensor_bytes"] + 8 * GIB) / GIB
    if state.output_disk_free_gib < needed:
        problems.append("insufficient target disk space including 8 GiB reserve")
    return tuple(problems)


def runtime_problems(state: HostState) -> tuple[str, ...]:
    problems = []
    if state.ram_available_gib < RUNTIME_RAM_FLOOR_GIB:
        problems.append("physical RAM safety floor crossed")
    if state.commit_available_gib < RUNTIME_COMMIT_FLOOR_GIB:
        problems.append("Windows commit safety floor crossed")
    if state.output_disk_free_gib < RUNTIME_DISK_FLOOR_GIB:
        problems.append("disk safety floor crossed")
    return tuple(problems)


def _read_lines(stream, items: queue.Queue) -> None:
    try:
        for line in stream:
            items.put(("output", line.rstrip("\n")))
    finally:
        items.put(("eof", ""))


def _run_owned(command: list[str], *, out: Path, log) -> int:
    """Foreground life-cycle ownership; terminate/kill ONLY the child PID."""
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "-1"
    environment["OMP_NUM_THREADS"] = "2"
    environment["MKL_NUM_THREADS"] = "2"
    environment["OPENBLAS_NUM_THREADS"] = "2"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [sys.executable, "-u", "-B", *command[1:]]
    queue_lines: queue.Queue = queue.Queue()
    with subprocess.Popen(
        command, cwd=Path(__file__).resolve().parents[1], env=environment,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, creationflags=subprocess.CREATE_NO_WINDOW,
    ) as child:
        print(json.dumps({"event": "OWNED_PRUNER_STARTED", "pid": child.pid}), flush=True)
        thread = threading.Thread(
            target=_read_lines, args=(child.stdout, queue_lines), daemon=True,
        )
        thread.start()
        reason = None
        while child.poll() is None:
            try:
                kind, line = queue_lines.get(timeout=POLL_SECONDS)
            except queue.Empty:
                kind, line = None, None
            if kind == "output":
                log.write(line + "\n")
                log.flush()
                print(line, flush=True)
            if kind == "eof":
                continue
            host = host_state(out.parent)
            violations = runtime_problems(host)
            if violations:
                reason = "; ".join(violations)
                log.write(f"OWNED_PRUNER_ABORT: {reason}\n")
                log.flush()
                # This process is the ONLY process under our ownership.
                child.terminate()
                try:
                    child.wait(timeout=12)
                except subprocess.TimeoutExpired:
                    child.kill()
                break
        thread.join(timeout=10)
        while not queue_lines.empty():
            kind, line = queue_lines.get_nowait()
            if kind == "output":
                log.write(line + "\n")
                print(line, flush=True)
        return 3 if reason else int(child.wait())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("model-dir", "weights-dir", "calibration", "observations", "selection", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--execute", action="store_true",
                        help="start the source-bound, CPU-only supervised prune")
    parser.add_argument("--log", type=Path,
                        help="new log file, outside source and destination trees")
    args = parser.parse_args(argv)
    try:
        plan = build_plan(
            arm="reap128_v2", model_dir=args.model_dir, weights_dir=args.weights_dir,
            calibration=args.calibration, observations=args.observations,
            selection=args.selection, out=args.out,
        )
        sizing = plan["build_capacity"]
        if sizing is None:
            raise ValueError("registered REAP128_v2 sizing/identity unavailable")
        state = host_state(args.out.parent)
        peak = memory_required_gib(sizing)
        problems = admit(state, peak_gib=peak, sizing=sizing)
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"status": "REFUSED", "reason": str(exc)}), flush=True)
        return 2
    receipt = {
        "status": "REFUSED" if problems else "ADMITTED_NOT_EXECUTED",
        "reasons": problems,
        "model": plan["model"], "selection_sha256": plan["selection_sha256"],
        "observation_sha256": plan["calibration"]["observer_sha256"],
        "build_capacity": sizing, "peak_estimate_gib": peak,
        "host": vars(state), "command": plan["prune_command_not_executed"],
    }
    print(json.dumps(receipt, sort_keys=True, allow_nan=False), flush=True)
    if problems:
        return 2
    if not args.execute:
        return 0
    if args.log is None:
        parser.error("--execute requires --log for immutable build evidence")
    log_path = args.log.resolve()
    inputs = [
        args.model_dir.resolve(), args.weights_dir.resolve(), args.observations.resolve(),
        args.calibration.resolve(), args.out.resolve(), args.selection.resolve().parent,
    ]
    if (not log_path.parent.is_dir() or log_path.exists()
            or any(log_path.is_relative_to(p) for p in inputs)):
        parser.error("log must be a fresh file outside model, observation and output inputs")
    # Recheck admission immediately before actually launching the owned process.
    reasons = admit(host_state(args.out.parent), peak_gib=peak, sizing=sizing)
    if reasons:
        print(json.dumps({"status": "REFUSED", "reasons": reasons}), flush=True)
        return 2
    with log_path.open("x", encoding="utf-8") as log:
        log.write(json.dumps(receipt, sort_keys=True) + "\n")
        log.flush()
        code = _run_owned(plan["prune_command_not_executed"], out=args.out, log=log)
        log.write(json.dumps({"event": "OWNED_PRUNER_EXIT", "returncode": code,
                              "output_published": args.out.is_dir()}) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
