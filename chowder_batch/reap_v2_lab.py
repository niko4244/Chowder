"""REAP128_v2 experimental test lane, separate from frozen Gen-0 selection.

The Q4 model is UNBLOCKED_FOR_TESTING (human experiment authorization), NOT
scientifically approved. A short official-source GSM8K-only serving smoke may
run after fresh, unchanged historical resource admission on a free Windows
host. No HumanEval generated Python is ever executed here. No Gen-0 decision,
release evidence, model source, protected service or frozen scorer is edited.

Status: python -m chowder_batch.reap_v2_lab
Run:    python -m chowder_batch.reap_v2_lab --execute --out F:/reap-lab/new-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from chowder_batch import reap_q4_resource_admission as admission
from chowder_batch import reap_q4_windows_benchmark_watchdog_20261026 as watchdog

ARM = "reap128_v2_q4"
EXPERIMENTAL_STATUS = "UNBLOCKED_FOR_TESTING"
MAX_SMOKE_ITEMS = 16
POLL_SECONDS = 2
MAX_RUN_SECONDS = 1800
DEFAULT_PORT = 8891


def testing_policy() -> dict:
    """No dependency on Gate0/Gate1/Gate2 or release selection for a lab run."""
    return {
        "schema": "chowder.reap128-v2-independent-experimental-testing.v1",
        "status": EXPERIMENTAL_STATUS,
        "selected_for_experimental_testing": ARM,
        "experimental_testing_authorized": True,
        "research_variants_unblocked": ["BF16", "Q8_0", "Q4_K_M"],
        "supervised_local_smoke_variant": "Q4_K_M",
        "registered_model_sha256": watchdog.MODEL_SHA,
        "registered_serving_flags": ["--fit", "on", "-np", "4", "-c", "16384", "--jinja"],
        "allowed_suites": ["gsm8k"],
        "no_model_generated_python_execution": True,
        "production_gen0_status": "BLOCKED",
        "production_chosen": None,
        "release_approved": False,
        "safety_admission_required": True,
    }


def _exclusive_output(out: Path) -> None:
    """Never write the originals, published model area or current worktree."""
    if not out.is_absolute() or not out.parent.is_dir():
        raise ValueError("choose a fresh absolute experiment path under an existing output folder")
    if out.exists() or out.is_symlink() or out.parent.is_symlink():
        raise ValueError("experiment output path must be new and exclusive")
    forbidden = [Path(__file__).resolve().parents[1], watchdog.GGUF.parent,
                 admission.ORIGINAL_SERVER_LOG.parent, watchdog.SERVER.parent]
    safe_out = out.resolve(strict=False)
    if any(safe_out == path.resolve() or safe_out.is_relative_to(path.resolve())
           for path in forbidden):
        raise ValueError("experiment output cannot be inside protected research or model paths")


def _fresh_admission(out_parent: Path) -> tuple[dict, dict]:
    """Read original source bytes and actual live hardware; don't trust a PASS flag."""
    history = admission.validate_aborted_speed_history(
        admission.ORIGINAL_ABORT, expected_sha256=admission.ORIGINAL_ABORT_SHA256)
    cache = admission.parse_host_prompt_cache(
        admission.ORIGINAL_SERVER_LOG, expected_sha256=admission.ORIGINAL_LOG_SHA256)
    state = watchdog.sample(out_parent)
    if watchdog.unsafe(state):
        raise ValueError("live runtime RAM/commit/disk/VRAM measurement unsafe")
    decision = admission.admit_host(
        history, ram_available_gib=state["ram_available_gib"],
        commit_available_gib=state["commit_available_gib"],
        gpu_free_mib=state["gpu_free_mib"],
        disk_free_gib=state["disk_free_gib"],
        potential_remaining_cache_gib=cache["potential_remaining_gib"],
    )
    return decision, state


def _run_command(out: Path, *, limit: int, port: int) -> list[str]:
    """Use existing hashed serving wrapper and frozen official GSM8K source."""
    root = Path(__file__).resolve().parents[1]
    return [sys.executable, "-B", str(root / "chowder_batch" / "reap_serve.py"),
            "--gguf", str(watchdog.GGUF), "--server", str(watchdog.SERVER),
            "--arm", ARM, "--port", str(port), "--out", str(out),
            "--", "--port", str(port), "--arm", ARM, "--suites", "gsm8k",
            "--limit", str(limit), "--parallel", "4", "--out", str(out)]


def _verify_smoke_results(out: Path, *, limit: int, port: int) -> dict:
    """A successful wrapper return is NOT sufficient proof of model inference."""
    arm_dir = out / ARM
    serve = arm_dir / "serve.json"
    input_witness = arm_dir / "gsm8k.inputs.json"
    row_path = arm_dir / "gsm8k.jsonl"
    summary_path = arm_dir / "summary.json"
    if any(not path.is_file() or path.is_symlink() or path.stat().st_size > 16 * 1024 * 1024
           for path in (serve, input_witness, row_path, summary_path)):
        raise ValueError("completed model smoke lacks bounded native source/row witnesses")
    try:
        serving = json.loads(serve.read_bytes())
        witness = json.loads(input_witness.read_bytes())
        summary = json.loads(summary_path.read_bytes())
        raw_rows = row_path.read_bytes()
        rows = [json.loads(line) for line in raw_rows.splitlines()]
    except (ValueError, UnicodeError, OSError) as exc:
        raise ValueError("completed model smoke has unreadable native source/row witness") from exc
    expected_flags = testing_policy()["registered_serving_flags"]
    if (not isinstance(serving, dict)
            or serving.get("gguf_sha256") != watchdog.MODEL_SHA
            or serving.get("flags") != expected_flags
            or type(serving.get("server_pid")) is not int
            or serving["server_pid"] <= 0
            or type(serving.get("server_port")) is not int
            or serving["server_port"] != port
            or type(serving.get("eval_session_id")) is not str
            or len(serving["eval_session_id"]) != 32
            or not isinstance(witness, dict)
            or witness.get("suite") != "gsm8k"
            or witness.get("source_items_n") != 1319
            or witness.get("model", {}).get("gguf_sha256") != watchdog.MODEL_SHA
            or witness.get("model", {}).get("flags") != expected_flags
            or not isinstance(summary, dict)
            or not isinstance(summary.get("gsm8k"), dict)
            or type(summary["gsm8k"].get("n")) is not int
            or summary["gsm8k"]["n"] != limit
            or summary["gsm8k"].get("file_sha256") != hashlib.sha256(raw_rows).hexdigest()
            or len(rows) != limit):
        raise ValueError("model smoke source, exact flags, serving session or row count mismatch")
    if ({row.get("id") for row in rows if isinstance(row, dict)}
            != {f"gsm8k-{i}" for i in range(limit)}
            or any(not isinstance(row, dict)
           or type(row.get("correct")) is not bool
           or type(row.get("terminated")) is not bool
           or type(row.get("text")) is not str
           for row in rows)):
        raise ValueError("model smoke scored rows are missing, duplicated or malformed")
    # A completed laboratory test must not leave a private model listener
    # running. Inspect only; never kill a reused or unrelated PID/port.
    watchdog.verify_native_shutdown(serving["server_pid"], port)
    return {"smoke_rows_verified": limit,
            "source_sha256": witness.get("source_items_sha256"),
            "serving_session_id": serving["eval_session_id"],
            "correct_items_observed": sum(row["correct"] for row in rows)}


def _execution_preflight(out: Path, *, limit: int, port: int) -> tuple[dict, list[str]]:
    if sys.platform != "win32":
        raise ValueError("this supervised local Q4 test requires Windows; use an eligible remote host")
    if type(limit) is not int or not 1 <= limit <= MAX_SMOKE_ITEMS:
        raise ValueError("experimental smoke requires 1..16 GSM8K items; HumanEval is prohibited")
    if type(port) is not int or not 1024 <= port <= 65535:
        raise ValueError("experimental private loopback port out of range")
    if port in {11434, 7050, 8766, 8767, 8790}:
        raise ValueError("cannot use original or protected services' ports")
    _exclusive_output(out)
    if (not watchdog.GGUF.is_file() or watchdog.GGUF.is_symlink()
            or not watchdog.SERVER.is_file() or watchdog.SERVER.is_symlink()):
        raise ValueError("registered Q4 model or pinned native server missing")
    # Independent source-bound identity is re-hashed inside reap_serve before
    # the first model request and checked again at end of the same session.
    # Resolve physical/commit/VRAM refusal BEFORE hashing 11.56 GB of model
    # weights. The native wrapper will independently hash those same bytes.
    decision, _ = _fresh_admission(out.parent)
    if decision["status"] != "ADMITTED_TO_ATTEMPT_NOT_SCIENTIFICALLY_QUALIFIED":
        raise ValueError("experiment hardware admission REFUSED: " + "; ".join(decision["reasons"]))
    if watchdog.sha(watchdog.GGUF) != watchdog.MODEL_SHA:
        raise ValueError("untrusted or modified registered REAP Q4 model bytes")
    # The serving wrapper checks exclusive loopback port ownership at launch.
    return decision, _run_command(out, limit=limit, port=port)


def run_experiment(out: Path, *, limit: int = 4, port: int = DEFAULT_PORT) -> dict:
    """Bounded same-session inference only, using existing supervised wrapper."""
    approved, command = _execution_preflight(out, limit=limit, port=port)
    # Source-only plan and the real runtime both use the same admission logic.
    # Recheck just before launch: an approved old screenshot isn't sufficient.
    refreshed, _ = _fresh_admission(out.parent)
    if refreshed["status"] != approved["status"]:
        raise ValueError("experiment admission changed before launch")
    out.mkdir(mode=0o700)
    proc = None
    aborted: list[str] = []
    started = time.monotonic()
    log_path = out / "experimental-wrapper-console.private.log"
    try:
        with log_path.open("xb") as stream:
            proc = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, shell=False)
            while proc.poll() is None:
                if time.monotonic() - started > MAX_RUN_SECONDS:
                    aborted.append("experiment elapsed 30-minute limit")
                else:
                    try:
                        observation = watchdog.sample(out)
                        aborted.extend(watchdog.unsafe(observation))
                    except (ValueError, OSError, subprocess.SubprocessError):
                        aborted.append("resource monitoring unavailable, abort")
                if aborted:
                    watchdog.kill_owned_tree(proc)
                    break
                try:
                    proc.wait(timeout=POLL_SECONDS)
                except subprocess.TimeoutExpired:
                    continue
            code = proc.wait(timeout=15)
        evidence = _verify_smoke_results(out, limit=limit, port=port) if code == 0 and not aborted else {}
        # Never convert model-test results into Amendment 4 quality or release.
        return {**testing_policy(), "status": (
            "EXPERIMENTAL_SMOKE_FINISHED" if code == 0 and not aborted
            else "EXPERIMENTAL_SMOKE_ABORTED_OR_FAILED"),
            "experiment_exit_code": code, "runtime_abort_reasons": aborted,
            "smoke_items_requested": limit, "official_score_certified": False,
            "experiment_output": str(out), "release_approved": False,
            **evidence}
    except BaseException:
        if proc is not None and proc.poll() is None:
            watchdog.kill_owned_tree(proc)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="only after verified host admission")
    parser.add_argument("--out", type=Path, help="new exclusive lab output directory")
    parser.add_argument("--limit", type=int, default=4, help="1..16 GSM8K, not HumanEval")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(testing_policy(), sort_keys=True))
        return 0
    if args.out is None:
        parser.error("--execute needs a fresh --out directory")
    try:
        verdict = run_experiment(args.out, limit=args.limit, port=args.port)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        verdict = {**testing_policy(), "status": "TEST_EXECUTION_REFUSED",
                   "reason": str(exc), "official_score_certified": False}
        print(json.dumps(verdict, sort_keys=True))
        return 2
    print(json.dumps(verdict, sort_keys=True))
    return 0 if verdict["status"] == "EXPERIMENTAL_SMOKE_FINISHED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
