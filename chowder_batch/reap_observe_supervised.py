"""Explicit operator-launched, host-guarded frozen BF16 REAP observation.

Does not stop, restart, configure, or restore any unrelated service. An abort
may terminate only the Python observer child started by this supervisor.
There is intentionally no implicit launch, auto-retry or background mode.

An operator must separately arrange safe model-service maintenance, choose
new unique paths and request --launch. A resume requires existing checkpoint.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

try:
    from chowder_batch import reap_host_guard as guard
except ImportError:
    import reap_host_guard as guard


class Refused(RuntimeError):
    pass


class _WindowsObserverJob:
    """Kill the *owned* observer automatically if supervisor exits/dies."""

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise Refused("Windows observer Job Object unavailable")
        kernel = ctypes.windll.kernel32

        class BasicLimits(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", ctypes.c_uint32),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", ctypes.c_uint32),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", ctypes.c_uint32),
                ("SchedulingClass", ctypes.c_uint32),
            ]

        class IOCounters(ctypes.Structure):
            _fields_ = [(field, ctypes.c_uint64) for field in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
            )]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BasicLimits),
                ("IoInfo", IOCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
        kernel.CreateJobObjectW.restype = ctypes.c_void_p
        kernel.SetInformationJobObject.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32,
        ]
        kernel.SetInformationJobObject.restype = ctypes.c_int
        kernel.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        kernel.AssignProcessToJobObject.restype = ctypes.c_int
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel.CloseHandle.restype = ctypes.c_int
        self.kernel = kernel
        self.handle = kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise Refused("cannot create kill-on-close observer Job Object")
        limits = ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE
        if not kernel.SetInformationJobObject(
            self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits),
        ):
            self.close()
            raise Refused("cannot configure kill-on-close observer Job Object")

    def attach(self, worker: subprocess.Popen) -> None:
        if not self.kernel.AssignProcessToJobObject(
            self.handle, ctypes.c_void_p(int(worker._handle)),
        ):
            raise Refused("cannot attach observer to kill-on-close Job Object")

    def close(self) -> None:
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def _atomic_json(path: Path, data: dict) -> None:
    payload = (json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("xb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _append_jsonl(path: Path, data: dict) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(data, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _sample_record(snapshot: guard.Snapshot, decision: guard.Decision,
                   *, owner_keys: frozenset[tuple[int, float]]) -> dict:
    foreign = guard._foreign_births(snapshot, owner_keys)
    largest = sorted(foreign, key=lambda process: process.private_bytes, reverse=True)[:12]
    return {
        "timestamp_utc": snapshot.as_record()["timestamp_utc"],
        "ram_free_gib": snapshot.ram_free_gib,
        "virtual_free_gib": snapshot.virtual_free_gib,
        "diskC_free_gib": snapshot.diskC_free_gib,
        "diskF_free_gib": snapshot.diskF_free_gib,
        "gpu_free_gib": snapshot.gpu_free_gib,
        "gpu_temp_c": snapshot.gpu_temp_c,
        "services_conflict": snapshot.services_conflict,
        "process_census_total": len(snapshot.processes),
        "process_census_errors": snapshot.census_errors,
        "foreign_birth_count_60s": decision.new_foreign_processes_60s,
        "foreign_birth_private_gib_60s": round(decision.new_foreign_private_gib, 3),
        "host_ram_decline_60s_gib": round(decision.ram_decline_60s_gib, 3),
        "foreign_births_top_private": [
            {"pid": p.pid, "created_epoch": p.started, "parent_pid": p.parent_pid,
             "name": p.name, "private_gib": round(p.private_bytes / guard.GIB, 3),
             "working_set_gib": round(p.working_set_bytes / guard.GIB, 3)}
            for p in largest
        ],
        "decision": decision.as_record(),
    }


def _check_observation_complete(out: Path, calibration: Path) -> dict:
    """Check a completed full D0 receipt, not merely an exit-zero process."""
    from chowder_batch.reap_hidden_checkpoint import validate_observation_receipt
    from chowder_batch.reap_select import _validate_v2_observation

    receipt = out / "observation.json"
    observed = json.loads(receipt.read_text(encoding="utf-8"))
    if (observed.get("n_seqs") != 704
            or observed.get("n_tokens") != 1_192_632
            or not isinstance(observed.get("layers"), list)
            or len(observed["layers"]) != 40
            or not isinstance(observed.get("checkpoint_receipt"), dict)):
        raise Refused("observation lacks the complete frozen checkpoint receipt")
    validate_observation_receipt(
        out, observed, expected_generation=40, require_registered_original=True,
    )
    _validate_v2_observation(calibration.parent, receipt)
    value = observed.get("calibration_ppl")
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise Refused("full observation did not measure finite positive D0 perplexity")
    return {
        "status": "D0_PASS" if value < 20 else "D0_MEASURED_FAIL",
        "calibration_ppl": value,
        "n_seqs": 704,
        "layers": 40,
        "receipt_path": str(receipt),
    }


def _observation_command(args: argparse.Namespace) -> list[str]:
    root = Path(__file__).resolve().parents[1]
    command = [
        str(root / "chowder_batch" / "reap_observe.py"),
        "--model-dir", str(args.model_dir),
        "--calibration", str(args.calibration),
        "--out", str(args.out),
    ]
    if args.weights_dir:
        command.extend(("--weights-dir", str(args.weights_dir)))
    command.extend((
        "--resume-hidden-disk-dir" if args.resume else "--hidden-disk-dir",
        str(args.scratch),
    ))
    return [
        sys.executable, "-u", "-B",
        str(root / "chowder_batch" / "reap_observer_bootstrap.py"),
        str(args.log_dir / "JOB_ATTACHED.ready"), *command,
    ]


def _validate_paths(args: argparse.Namespace) -> None:
    if not args.model_dir.is_dir() or not args.calibration.is_file():
        raise Refused("model or frozen calibration source does not exist")
    if args.weights_dir and not args.weights_dir.is_dir():
        raise Refused("weights source does not exist")
    from chowder_batch.reap_v2_source_identity import FROZEN_V2_BASE_SHA256
    from hashlib import sha256

    if sha256(args.calibration.read_bytes()).hexdigest() != FROZEN_V2_BASE_SHA256:
        raise Refused("frozen 704-row calibration SHA256 mismatch")
    from chowder_batch.reap_original_bf16_pin import validate_registered_source_metadata
    validate_registered_source_metadata(
        args.model_dir, args.weights_dir or args.model_dir,
    )
    if args.resume:
        if (not args.out.is_dir() or not args.scratch.is_dir()
                or not (args.scratch / "CURRENT").is_file()
                or (args.out / "observation.json").exists()):
            raise Refused("resume requires a committed unfinished BF16 experiment")
    elif args.out.exists() or args.scratch.exists():
        raise Refused("fresh experiment must use new output and scratch paths")
    if args.out.resolve() == args.scratch.resolve() or (
        args.out.resolve().is_relative_to(args.scratch.resolve())
        or args.scratch.resolve().is_relative_to(args.out.resolve())
    ):
        raise Refused("observation and scratch paths must be disjoint")
    if args.log_dir.exists() or not args.log_dir.parent.is_dir():
        raise Refused("supervisor logs require a new directory with existing parent")
    protected = (args.model_dir, args.calibration.parent,
                 args.weights_dir or args.model_dir, args.out, args.scratch)
    for source in protected:
        a, b = args.log_dir.resolve(), source.resolve()
        if a == b or a.is_relative_to(b) or b.is_relative_to(a):
            raise Refused("supervisor logs overlap protected inputs/outputs")


def _owned_only_stop(worker: subprocess.Popen, *, grace_seconds: int = 15) -> None:
    """Never enumerate/terminate other PIDs, or start/stop unrelated services."""
    if worker.poll() is not None:
        return
    worker.terminate()
    try:
        worker.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        worker.kill()
        worker.wait(timeout=grace_seconds)


def _run(args: argparse.Namespace) -> int:
    if sys.platform != "win32":
        raise Refused("host memory supervisor is currently admitted on Windows only")
    _validate_paths(args)
    args.log_dir.mkdir(exist_ok=False)
    state_file = args.log_dir / "supervisor-state.json"
    timeline = args.log_dir / "resources.jsonl"
    owner_keys = frozenset({guard.process_identity(os.getpid())})
    worker: subprocess.Popen | None = None
    job: _WindowsObserverJob | None = None
    verdict = {"stage": "PRELAUNCH", "reason": "NOT_ADMITTED",
               "observer_pid": None, "registered_d0_proven": False}
    _atomic_json(state_file, verdict)
    try:
        snapshots: list[guard.Snapshot] = []
        admission_started = time.monotonic()
        for index in range(3):
            delay = admission_started + index * 30 - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            sample_started = time.monotonic()
            snap = guard.collect_snapshot()
            if time.monotonic() - sample_started > 25:
                raise Refused("prelaunch resource census exceeded 25-second sampling budget")
            snapshots.append(snap)
            interim = guard.assess_admission(snapshots, owner_keys=owner_keys)
            _append_jsonl(timeline, _sample_record(snap, interim, owner_keys=owner_keys))
        admission = guard.assess_admission(snapshots, owner_keys=owner_keys)
        if admission.action != "ADMIT":
            raise Refused(f"host prelaunch refusal: {', '.join(admission.reasons)}")
        # Close the sensor/port TOCTOU between 60s admission and Popen.
        last_sample_started = time.monotonic()
        recent = guard.collect_snapshot()
        if time.monotonic() - last_sample_started > 25:
            raise Refused("last-moment host census exceeded 25-second budget")
        renewed = guard.assess_admission(
            [snapshots[0], snapshots[1], recent], owner_keys=owner_keys,
        )
        _append_jsonl(timeline, _sample_record(recent, renewed, owner_keys=owner_keys))
        if renewed.action != "ADMIT":
            raise Refused("last-moment host admission refusal: " + ", ".join(renewed.reasons))
        command = _observation_command(args)
        verdict.update(stage="ADMITTED", reason="60s_RAM_COMMIT_PROCESS_GATE_PASS")
        _atomic_json(state_file, verdict)
        job = _WindowsObserverJob()
        with (args.log_dir / "observer.stdout.log").open("wb") as stdout, (
            args.log_dir / "observer.stderr.log"
        ).open("wb") as stderr:
            worker = subprocess.Popen(
                command, cwd=Path(__file__).resolve().parents[1],
                stdout=stdout, stderr=stderr,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            # The tiny bootstrap remains idle until a successfully configured
            # Job Object owns it. A supervisor crash closes the OS job handle.
            job.attach(worker)
            owner_keys = frozenset({*owner_keys, guard.process_identity(worker.pid)})
            _atomic_json(args.log_dir / "JOB_ATTACHED.ready", {
                "schema": "chowder.reap-observer-job-attachment.v1",
                "worker_pid": worker.pid,
            })
            verdict.update(stage="RUNNING", reason="INCOMPLETE_D0",
                           observer_pid=worker.pid)
            _atomic_json(state_file, verdict)
            history: list[guard.Snapshot] = []
            start = time.monotonic()
            next_tick = start + 30
            while worker.poll() is None:
                try:
                    worker.wait(timeout=max(0.01, next_tick - time.monotonic()))
                except subprocess.TimeoutExpired:
                    pass
                if worker.poll() is not None:
                    break
                sample_started = time.monotonic()
                try:
                    snap = guard.collect_snapshot()
                except Exception as exc:
                    raise Refused(f"resource telemetry failed closed: {type(exc).__name__}") from exc
                if time.monotonic() - sample_started > 25:
                    raise Refused("resource census exceeded 25-second monitoring budget")
                history.append(snap)
                history = history[-4:]
                decision = guard.assess_runtime(history, owner_keys=owner_keys)
                _append_jsonl(timeline, _sample_record(snap, decision, owner_keys=owner_keys))
                if decision.action == "ABORT":
                    raise Refused("RESOURCE_SAFETY_ABORT: " + ", ".join(decision.reasons))
                if (args.log_dir / "STOP").exists():
                    raise Refused("operator STOP file in supervisor log directory")
                if time.monotonic() - start >= 8 * 60 * 60:
                    raise Refused("8-hour observer safety limit reached")
                next_tick += 30
                if time.monotonic() > next_tick + 5:
                    raise Refused("runtime resource samples could not maintain 30s cadence")
            exit_code = worker.wait()
            if exit_code != 0:
                raise Refused(f"observer process failed with exit code {exit_code}")
            # Never publish a supervised PASS based only on a just-exited child:
            # a final memory cliff could have happened between poll intervals.
            final_started = time.monotonic()
            final_sample = guard.collect_snapshot()
            if time.monotonic() - final_started > 25:
                raise Refused("final host-resource witness exceeded sampling budget")
            # Worker has exited and may no longer appear in process census.
            supervisor_key = frozenset({guard.process_identity(os.getpid())})
            final_history = [*history, final_sample][-4:]
            final_safety = guard.assess_runtime(final_history, owner_keys=supervisor_key)
            _append_jsonl(timeline, _sample_record(
                final_sample, final_safety, owner_keys=supervisor_key,
            ))
            if final_safety.action == "ABORT":
                raise Refused("final host-resource witness unsafe: " + ", ".join(final_safety.reasons))
            final = _check_observation_complete(args.out, args.calibration)
            verdict.update(stage=final["status"], reason="complete_frozen_observation",
                           measured=final, registered_d0_proven=(final["status"] == "D0_PASS"))
            return 0 if final["status"] == "D0_PASS" else 3
    except Exception as exc:
        verdict.update(stage="INCOMPLETE_D0", reason=str(exc),
                       registered_d0_proven=False)
        return 2
    finally:
        if worker is not None:
            _owned_only_stop(worker)
        if job is not None:
            job.close()
        _atomic_json(state_file, verdict)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path,
                        help="pure offline historic resource replay, no launch")
    parser.add_argument("--launch", action="store_true",
                        help="explicit authorization to run an observer")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--weights-dir", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--scratch", type=Path)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.replay:
        if args.launch:
            parser.error("--replay and --launch cannot be combined")
        print(json.dumps(guard.replay_timeline(args.replay), indent=2))
        return 0
    if not args.launch:
        parser.error("observer requires explicit --launch (use --replay for offline tests)")
    if any(getattr(args, name) is None for name in (
        "model_dir", "calibration", "out", "scratch", "log_dir",
    )):
        parser.error("--launch requires --model-dir --calibration --out --scratch --log-dir")
    try:
        return _run(args)
    except Exception as exc:
        print(f"OBSERVER_NOT_LAUNCHED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
