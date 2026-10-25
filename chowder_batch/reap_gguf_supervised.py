"""Execute frozen BF16→GGUF→Q8_0+Q4_K_M CPU stages with owned-child guards.

The default is read-only admission. Explicit --execute and a new --log are
required before launching any subprocess. All intended destinations must be
fresh; each stage writes an exclusive private sibling file and atomically
publishes only on successful process exit plus minimal GGUF magic/size check.
Never stop or change shared model services, GPUs, or another process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from chowder_batch.reap_gguf_build_plan import plan
from chowder_batch.reap_prune_supervised import host_state

GIB = 1024 ** 3
MIN_START_RAM_GIB = 12.0
MIN_START_COMMIT_GIB = 16.0
RUNTIME_RAM_GIB = 7.0
RUNTIME_COMMIT_GIB = 8.0
RUNTIME_FREE_DISK_GIB = 6.0
POLL_SEC = 5


def resource_problems(paths: list[Path], *, initial: bool = False) -> list[str]:
    """Read-only Windows memory/commit and per-volume space."""
    state = host_state(paths[0].parent)
    reasons = []
    if state.ram_available_gib < (MIN_START_RAM_GIB if initial else RUNTIME_RAM_GIB):
        reasons.append("insufficient unowned physical RAM")
    if state.commit_available_gib < (MIN_START_COMMIT_GIB if initial else RUNTIME_COMMIT_GIB):
        reasons.append("insufficient free Windows commit")
    from shutil import disk_usage
    for dest in paths:
        free = disk_usage(dest.parent).free / GIB
        if free < RUNTIME_FREE_DISK_GIB:
            reasons.append(f"volume floor breached at {dest.parent}")
    return reasons


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _minimal_gguf_check(path: Path) -> None:
    if not path.is_file() or path.stat().st_size < 1024 * 1024:
        raise ValueError(f"missing/incomplete GGUF stage: {path}")
    with path.open("rb") as f:
        magic = f.read(4)
        version = int.from_bytes(f.read(4), "little")
    if magic != b"GGUF" or version not in (2, 3):
        raise ValueError(f"GGUF stage header magic/version invalid: {path}")


def _reader(stream, sink: queue.Queue) -> None:
    try:
        for line in stream:
            sink.put(("line", line.rstrip("\n")))
    finally:
        sink.put(("eof", ""))


def _owned_stage(command: list[str], *, drives: list[Path], log) -> int:
    """Only the exact child this function creates can be terminated."""
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    pending: queue.Queue = queue.Queue()
    with subprocess.Popen(
        command, cwd=Path(__file__).resolve().parents[1],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=env, text=True, bufsize=1,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    ) as child:
        print(json.dumps({"event": "OWNED_CONVERSION_STARTED",
                          "pid": child.pid, "binary": command[0]}), flush=True)
        worker = threading.Thread(target=_reader, args=(child.stdout, pending), daemon=True)
        worker.start()
        aborted = None
        while child.poll() is None:
            try:
                kind, line = pending.get(timeout=POLL_SEC)
            except queue.Empty:
                kind, line = "", ""
            if kind == "line":
                log.write(line + "\n")
                log.flush()
                if "error" in line.lower() or "writing" in line.lower() or "output" in line.lower():
                    print(line[:300], flush=True)
            problems = resource_problems(drives)
            if problems:
                aborted = "; ".join(problems)
                log.write(f"OWNED_CONVERSION_ABORT: {aborted}\n")
                log.flush()
                child.terminate()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                break
        worker.join(timeout=10)
        while not pending.empty():
            kind, line = pending.get_nowait()
            if kind == "line":
                log.write(line + "\n")
        return 3 if aborted else int(child.wait())


def build(*, specification: dict, log_path: Path, execute: bool) -> dict:
    commands = [
        specification["conversion_command"],
        specification["q8_quality_command"],
        specification["q4_fit_command"],
    ]
    destinations = [
        Path(specification["capacity"][key]["intended"])
        for key in ("bf16", "q8_0", "q4_k_m")
    ]
    reasons = resource_problems(destinations, initial=True)
    if reasons:
        raise ValueError("GGUF CPU conversion admission refused: " + "; ".join(reasons))
    if not execute:
        return {"status": "ADMITTED_NOT_EXECUTED", "phases": 3,
                "frozen_candidate": specification["candidate"]}
    if not log_path.parent.is_dir() or log_path.exists():
        raise ValueError("conversion audit log must be new with an existing parent")
    roots = [Path(specification["candidate"]), *destinations]
    if any(log_path.resolve().is_relative_to(path.resolve()) for path in roots):
        raise ValueError("log must not live inside model or staged outputs")
    phase_results = []
    with log_path.open("x", encoding="utf-8", newline="\n") as log:
        log.write(json.dumps({"event": "FROZEN_GGUF_BUILD_START", "spec": specification},
                             sort_keys=True) + "\n")
        log.flush()
        for phase, (original, out) in enumerate(zip(commands, destinations)):
            if out.exists():
                raise ValueError(f"output already exists, refusing conversion: {out}")
            if resource_problems(destinations, initial=True):
                raise ValueError(f"insufficient resources for conversion phase {phase}")
            temporary = out.with_name(f".{out.stem}.owned-{uuid.uuid4().hex}.gguf")
            if temporary.exists():
                raise ValueError("private output unexpectedly exists")
            command = list(original)
            if phase == 0:
                position = command.index("--outfile") + 1
                command[position] = str(temporary)
            else:
                command[2] = str(temporary)
            log.write(json.dumps({"event": "OWNED_STAGE_START", "phase": phase,
                                  "command": command}) + "\n")
            log.flush()
            status = _owned_stage(command, drives=destinations, log=log)
            if status:
                raise RuntimeError(f"owned GGUF phase {phase} returned {status}; "
                                   f"private partial file retained at {temporary}")
            _minimal_gguf_check(temporary)
            digest = _digest(temporary)
            size = temporary.stat().st_size
            if out.exists():
                raise ValueError("another process created final output; refuse overwrite")
            temporary.rename(out)
            result = {
                "phase": phase, "output": str(out), "sha256": digest,
                "size_bytes": size, "status": "PUBLISHED_GGUF_NOT_QUALITY_QUALIFIED",
            }
            phase_results.append(result)
            log.write(json.dumps({"event": "OWNED_STAGE_PUBLISHED", **result}) + "\n")
            log.flush()
            print(json.dumps({"event": "OWNED_STAGE_PUBLISHED", **result}), flush=True)
    return {"status": "GGUF_CPU_BUILDS_COMPLETED_NOT_QUALITY_QUALIFIED",
            "phases": phase_results}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate", "verification", "converter", "llama-checkout", "quantizer",
                 "bf16-out", "q8-out", "q4-out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        spec = plan(
            candidate=args.candidate, verification=args.verification,
            converter=args.converter, llama_checkout=args.llama_checkout,
            quantizer=args.quantizer, bf16_out=args.bf16_out,
            q8_out=args.q8_out, q4_out=args.q4_out,
        )
        if args.execute and args.log is None:
            raise ValueError("--execute requires an exclusive --log path")
        result = build(specification=spec,
                       log_path=args.log if args.log is not None else Path("not-used.log"),
                       execute=args.execute)
        print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError) as exc:
        print(f"frozen GGUF supervisor refused: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
