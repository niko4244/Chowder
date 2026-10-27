"""Original frozen Windows Q4 full-1,319 GSM8K admission and live watchdog.

Prospectively reuses the *original* serving/evaluator and exact source. No
HumanEval candidate programs are run. This is an independent Windows Gate0/1
candidate only, never a Gen-0 selection, speed ratio, or changed contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

from chowder_batch import reap_eval, reap_serve
from chowder_batch import reap_q4_windows_benchmark_watchdog_20261026 as q4

FULL = 1319
EXPECTED_IDS = {f"gsm8k-{i}" for i in range(FULL)}


def source_admission(admission: dict, fresh: dict) -> dict:
    """Require a newly source-recomputed original Q4 start admission."""
    decision = q4.verify_admission_source(admission)
    current = admission["current"]
    if q4.unsafe(fresh):
        raise ValueError("original full1319 fresh host telemetry unsafe")
    for key, target in (
        ("ram_available_gib", decision["minimum_ram_gib"]),
        ("commit_available_gib", decision["minimum_commit_gib"]),
        ("gpu_free_mib", decision["minimum_gpu_mib"]),
        ("disk_free_gib", decision["minimum_disk_gib"]),
    ):
        if (type(current.get(key)) not in (int, float)
                or type(fresh.get(key)) not in (int, float)
                or not math.isfinite(current[key]) or not math.isfinite(fresh[key])
                or min(current[key], fresh[key]) < target):
            raise ValueError(f"original Q4 full1319 source-bound admission refused: {key}")
    return decision


def _json(path: Path) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("original sealed receipt missing or oversized")
    data = json.loads(path.read_bytes(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    if type(data) is not dict:
        raise ValueError("original sealed receipt must be a dict")
    return data


def complete_original_gsm8k(arm: Path, *, model_session: str) -> dict:
    """Independently re-score all 1,319 saved authentic model completions."""
    if arm.is_symlink() or not arm.is_dir():
        raise ValueError("original model output folder absent or symlink")
    serving = _json(arm / "serve.json")
    source_witness = _json(arm / "gsm8k.inputs.json")
    summary = _json(arm / "summary.json")
    rows_path = arm / "gsm8k.jsonl"
    if (rows_path.is_symlink() or not rows_path.is_file()
            or rows_path.stat().st_size > 64 * 1024 * 1024):
        raise ValueError("original full1319 source-bound rows missing/unbounded")
    if (serving.get("gguf_sha256") != q4.MODEL_SHA
            or serving.get("flags") != reap_serve.FLAGS
            or serving.get("eval_session_id") != model_session
            or serving.get("server_port") != q4.PORT
            or type(serving.get("server_pid")) is not int
            or serving.get("measured_experts_offloaded") != 0):
        raise ValueError("original native Q4 serving identity, flags or expert offload mismatch")
    reap_serve._verify_offload_receipt(serving, arm / "server.log")
    actual_source = reap_eval._load("gsm8k")
    expected_sha = reap_eval.FROZEN_V2_QUALITY_SOURCES["gsm8k"][1]
    if (len(actual_source) != FULL or reap_eval._digest(actual_source) != expected_sha
            or source_witness.get("source_items_sha256") != expected_sha
            or source_witness.get("source_items_n") != FULL
            or source_witness.get("suite") != "gsm8k"
            or source_witness.get("model") != {k: serving[k]
                                               for k in ("gguf", "gguf_sha256", "flags")}):
        raise ValueError("original 1,319 canonical GSM8K source/input witness mismatch")
    raw = rows_path.read_bytes()
    rows_sha = hashlib.sha256(raw).hexdigest()
    if not raw.endswith(b"\n"):
        raise ValueError("original full1319 row ledger truncated")
    rows = [json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            for line in raw.splitlines()]
    ids = [r.get("id") if type(r) is dict else None for r in rows]
    if len(rows) != FULL or set(ids) != EXPECTED_IDS or len(set(ids)) != FULL:
        raise ValueError("original frozen full1319 GSM8K IDs missing/duplicated")
    by_id = {r["id"]: r for r in actual_source}
    passed = 0
    naturally_stopped = 0
    degenerate = 0
    total_output_tokens = 0
    for row in rows:
        if (type(row) is not dict or set(row) != {
                "id", "correct", "terminated", "trigram_ratio", "completion_tokens",
                "seconds", "predicted_per_second", "text"}
                or type(row["correct"]) is not bool or type(row["terminated"]) is not bool
                or type(row["text"]) is not str or type(row["completion_tokens"]) is not int
                or row["completion_tokens"] < 0):
            raise ValueError("original frozen GSM8K model output row fields malformed")
        for key in ("trigram_ratio", "seconds", "predicted_per_second"):
            x = row[key]
            if type(x) not in (int, float) or not math.isfinite(x) or x < 0:
                raise ValueError("original GSM8K output numeric row malformed/nonfinite")
        rescored = reap_eval._score("gsm8k", by_id[row["id"]], {
            "finish_reason": "stop" if row["terminated"] else "length",
            "text": row["text"], "completion_tokens": row["completion_tokens"],
            "seconds": row["seconds"],
            "timings": {"predicted_per_second": row["predicted_per_second"]},
        })
        if any(rescored[k] != row[k] for k in ("id", "correct", "terminated", "trigram_ratio")):
            raise ValueError("original scorer does not reproduce saved Q4 result")
        passed += int(row["correct"])
        naturally_stopped += int(row["terminated"])
        degenerate += int(row["trigram_ratio"] < 0.5)
        total_output_tokens += row["completion_tokens"]
    official = summary.get("gsm8k")
    if (type(official) is not dict or official.get("n") != FULL
            or official.get("file_sha256") != rows_sha
            or official.get("accuracy") != passed / FULL
            or official.get("terminated_rate") != naturally_stopped / FULL
            or official.get("degenerate_rate") != degenerate / FULL):
        raise ValueError("full original canonical GSM8K 1,319 summary/scorer differs")
    return {"population": FULL, "correct": passed, "naturally_terminated": naturally_stopped,
            "degenerate": degenerate, "completion_tokens": total_output_tokens,
            "rows_sha256": rows_sha, "source_sha256": expected_sha,
            "serving_receipt_sha256": q4.sha(arm / "serve.json"),
            "native_offload_verified": True, "all_rows_rescored": True}


def run(out: Path, admission: Path) -> dict:
    if sys.platform != "win32":
        raise RuntimeError("registered full original Q4 qualification requires Windows")
    if out.exists() or out.is_symlink() or not out.is_absolute() or not out.parent.is_dir():
        raise ValueError("full1319 original evidence output must be a new exclusive directory")
    if admission.is_symlink() or not admission.is_file() or admission.stat().st_size > 32768:
        raise ValueError("original source-bound admission receipt absent/too large")
    prior = _json(admission)
    if not q4.GGUF.is_file() or not q4.SERVER.is_file():
        raise ValueError("original Q4 GGUF/native server file absent")
    reap_serve._require_free_loopback_port(q4.PORT)
    first = q4.sample(out.parent)
    decision = source_admission(prior, first)
    out.mkdir(mode=0o700)
    command = [sys.executable, "-B", str(Path(reap_serve.__file__)),
               "--gguf", q4.GGUF.as_posix(), "--arm", q4.ARM,
               "--port", str(q4.PORT), "--out", str(out),
               "--server", str(q4.SERVER), "--timeout", "900", "--",
               "--port", str(q4.PORT), "--arm", q4.ARM, "--suites", "gsm8k",
               "--out", str(out), "--limit", str(FULL), "--parallel", "4"]
    receipt = {"schema": "chowder.reap128-v2-q4-windows-full1319.v1",
               "status": "ABORT_OR_UNVERIFIED", "arm": q4.ARM,
               "command": command, "admission_sha256": q4.sha(admission),
               "starting_resources": first, "source_bound_start": decision,
               "frozen_flags": reap_serve.FLAGS, "original_model_sha256": q4.MODEL_SHA,
               "gen0_status": "BLOCKED", "chosen": None, "release_approved": False}
    child = None
    start = time.monotonic()
    samples = out / "resource-samples.jsonl"
    try:
        with (out / "benchmark-console.private.log").open("xb") as log, samples.open("x", encoding="utf8") as telemetry:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT)
            receipt["owned_wrapper_pid"] = child.pid
            count = 0
            while child.poll() is None:
                try:
                    observed = q4.sample(out)
                    violations = q4.unsafe(observed)
                except (OSError, ValueError, subprocess.SubprocessError) as exc:
                    observed = {"time_utc": datetime.now(timezone.utc).isoformat(),
                                "monitor_error": type(exc).__name__}
                    violations = ["full1319 watchdog telemetry missing; abort"]
                telemetry.write(json.dumps(q4.bounded_telemetry_record(observed, violations),
                                           sort_keys=True, allow_nan=False) + "\n")
                telemetry.flush()
                os.fsync(telemetry.fileno())
                count += 1
                if violations:
                    receipt["abort_reasons"] = violations
                    q4.kill_owned_tree(child)
                    break
                time.sleep(q4.POLL_SECONDS)
            if child.poll() is None:
                q4.kill_owned_tree(child)
            receipt["child_exit_code"] = child.wait(timeout=30)
        receipt["sample_count"] = count
        receipt["elapsed_wall_seconds"] = round(time.monotonic() - start, 3)
        arm = out / q4.ARM
        serving_path = arm / "serve.json"
        if serving_path.is_file():
            serving = _json(serving_path)
            receipt["native_shutdown_verified"] = q4.verify_native_shutdown(
                serving.get("server_pid"), q4.PORT)
        if (receipt["child_exit_code"] == 0 and not receipt.get("abort_reasons")
                and receipt.get("native_shutdown_verified") is True):
            checked = complete_original_gsm8k(arm, model_session=serving["eval_session_id"])
            receipt["source_bound_scoring"] = checked
            receipt["status"] = "COMPLETE_ORIGINAL_WINDOWS_Q4_FULL1319_CANDIDATE_ONLY"
        else:
            receipt["status"] = "INCOMPLETE_OR_ABORTED_NO_GATE_CERTIFICATION"
    finally:
        if child is not None:
            q4.kill_owned_tree(child)
        path = out / "full1319-terminal.json"
        with path.open("x", encoding="utf8") as stream:
            json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = run(args.out, args.admission)
    print(json.dumps({k: receipt.get(k) for k in ("status", "sample_count", "elapsed_wall_seconds",
                                                   "abort_reasons", "source_bound_scoring", "gen0_status")},
                     sort_keys=True))
    return 0 if receipt["status"] == "COMPLETE_ORIGINAL_WINDOWS_Q4_FULL1319_CANDIDATE_ONLY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
