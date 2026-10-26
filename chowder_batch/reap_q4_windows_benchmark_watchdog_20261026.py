"""Windows-only, fail-closed original REAP v2 Q4 first-200 serving watchdog.

Never alters frozen serving/evaluator parameters. Requires the registered
admission check, supervises only the owned wrapper process tree, preserves
all partial attempts separately, and cannot authorize Gen-0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from chowder_batch.reap_prune_supervised import host_state
from chowder_batch.reap_q4_resource_admission import (
    ORIGINAL_COMMIT_RUNTIME_FLOOR_GIB,
    ORIGINAL_DISK_FLOOR_GIB,
    ORIGINAL_RAM_RUNTIME_FLOOR_GIB,
)

GGUF = Path("G:/reap128-v2-q4_k_m-20261025.gguf")
SERVER = Path("F:/llama.cpp/bin/llama-server.exe")
ARM = "reap128_v2_q4"
MODEL_SHA = "2d64b786461b99a335fdbbdf7c4d3a6758b21409bce755751714ab7cd8f63bf7"
PORT = 8790
POLL_SECONDS = 3


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def free_vram_mib() -> int:
    call = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=15, check=True,
    )
    values = call.stdout.strip().splitlines()
    if len(values) != 1:
        raise ValueError("expected exactly one GPU; refuse ambiguous VRAM")
    return int(values[0].strip())


def sample(parent: Path) -> dict:
    state = host_state(parent)
    return {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "ram_available_gib": state.ram_available_gib,
        "commit_available_gib": state.commit_available_gib,
        "disk_free_gib": state.output_disk_free_gib,
        "gpu_free_mib": free_vram_mib(),
    }


def unsafe(row: dict) -> list[str]:
    problems = []
    if row["ram_available_gib"] < ORIGINAL_RAM_RUNTIME_FLOOR_GIB:
        problems.append("physical RAM below 7 GiB safety floor")
    if row["commit_available_gib"] < ORIGINAL_COMMIT_RUNTIME_FLOOR_GIB:
        problems.append("available Windows commit below 8 GiB safety floor")
    if row["disk_free_gib"] < ORIGINAL_DISK_FLOOR_GIB:
        problems.append("disk below 8 GiB safety floor")
    return problems


def kill_owned_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    subprocess.run(
        ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
        capture_output=True, timeout=30, check=False,
    )
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def run(*, out: Path, admission: Path) -> dict:
    if sys.platform != "win32":
        raise RuntimeError("original registered benchmark requires Windows")
    if out.is_symlink() or out.exists() or not out.parent.is_dir():
        raise ValueError("benchmark evidence directory must be exclusive")
    if admission.is_symlink() or not admission.is_file():
        raise ValueError("original admission receipt missing")
    prior = json.loads(admission.read_text(encoding="utf8"))
    if (prior.get("schema") != "reap128-v2-verified-host-admission.v1"
            or prior.get("decision", {}).get("status")
            != "ADMITTED_TO_ATTEMPT_NOT_SCIENTIFICALLY_QUALIFIED"):
        raise ValueError("frozen original source-bound admission did not pass")
    if not GGUF.is_file() or not SERVER.is_file():
        raise ValueError("original GGUF/native server absent")
    from chowder_batch.reap_serve import _require_free_loopback_port
    _require_free_loopback_port(PORT)
    # Recheck immediate independent live admission floors before launching.
    start = sample(out.parent)
    before = prior["current"]
    for key, minimum in (
        ("ram_available_gib", prior["decision"]["minimum_ram_gib"]),
        ("commit_available_gib", prior["decision"]["minimum_commit_gib"]),
        ("gpu_free_mib", prior["decision"]["minimum_gpu_mib"]),
        ("disk_free_gib", prior["decision"]["minimum_disk_gib"]),
    ):
        if start[key] < minimum:
            raise ValueError(f"source-bound admission lost before launch: {key}")
        if before[key] < minimum:
            raise ValueError(f"source-bound admission receipt was not actually eligible: {key}")
    out.mkdir(mode=0o700)
    command = [
        sys.executable, "-B", str(Path(__file__).with_name("reap_serve.py")),
        "--gguf", GGUF.as_posix(), "--arm", ARM, "--port", str(PORT),
        "--out", str(out), "--server", str(SERVER), "--timeout", "900", "--",
        "--port", str(PORT), "--arm", ARM, "--suites", "gsm8k",
        "--out", str(out), "--limit", "200", "--parallel", "4",
    ]
    metadata = {
        "schema": "chowder.reap-q4-windows-monitored-first200.v1",
        "status": "ABORT_OR_INCOMPLETE",
        "command": command, "model_sha256_expected": MODEL_SHA,
        "admission_sha256": sha(admission), "first_sample": start,
        "physical_floor_gib": ORIGINAL_RAM_RUNTIME_FLOOR_GIB,
        "commit_floor_gib": ORIGINAL_COMMIT_RUNTIME_FLOOR_GIB,
        "disk_floor_gib": ORIGINAL_DISK_FLOOR_GIB,
        "gen0_status": "BLOCKED", "chosen": None, "release_approved": False,
    }
    samples_path = out / "resource-samples.jsonl"
    started = time.monotonic()
    proc = None
    try:
        with (out / "benchmark-console.private.log").open("xb") as log, samples_path.open(
            "x", encoding="utf8",
        ) as telemetry:
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            metadata["owned_wrapper_pid"] = proc.pid
            while proc.poll() is None:
                try:
                    row = sample(out)
                    failures = unsafe(row)
                except (OSError, ValueError, subprocess.SubprocessError) as exc:
                    row = {"time_utc": datetime.now(timezone.utc).isoformat(),
                           "error": type(exc).__name__}
                    failures = ["resource telemetry missing; fail closed"]
                telemetry.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                telemetry.flush()
                os.fsync(telemetry.fileno())
                if failures:
                    metadata["abort_reasons"] = failures
                    kill_owned_tree(proc)
                    break
                time.sleep(POLL_SECONDS)
            if proc.poll() is None:
                kill_owned_tree(proc)
            metadata["child_exit_code"] = proc.wait(timeout=30)
        metadata["elapsed_wall_seconds"] = round(time.monotonic() - started, 3)
        metadata["sample_count"] = len(samples_path.read_bytes().splitlines())
        arm_dir = out / ARM
        serving = arm_dir / "serve.json"
        results = arm_dir / "gsm8k.jsonl"
        summary = arm_dir / "summary.json"
        if serving.is_file():
            payload = json.loads(serving.read_text(encoding="utf8"))
            metadata["serve_json_sha256"] = sha(serving)
            metadata["source_bound_speed_session"] = payload.get("speed_session")
        if results.is_file():
            metadata["partial_or_complete_rows"] = len(results.read_bytes().splitlines())
            metadata["rows_sha256"] = sha(results)
        if summary.is_file():
            metadata["summary_sha256"] = sha(summary)
        if (metadata["child_exit_code"] == 0 and metadata.get("partial_or_complete_rows") == 200
                and isinstance(metadata.get("source_bound_speed_session"), dict)):
            metadata["status"] = "COMPLETE_SOURCE_BOUND_200_CANDIDATE_ONLY"
        else:
            metadata["status"] = "INCOMPLETE_OR_ABORTED_NO_SPEED_CERTIFICATION"
    finally:
        if proc is not None:
            kill_owned_tree(proc)
        with (out / "watchdog-terminal.json").open("x", encoding="utf8") as stream:
            json.dump(metadata, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(out=args.out, admission=args.admission)
    print(json.dumps({key: result.get(key) for key in (
        "status", "elapsed_wall_seconds", "sample_count", "partial_or_complete_rows",
        "abort_reasons", "source_bound_speed_session", "gen0_status",
    )}, sort_keys=True))
    return 0 if result["status"] == "COMPLETE_SOURCE_BOUND_200_CANDIDATE_ONLY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
