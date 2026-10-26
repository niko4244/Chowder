"""Separate, NON-GEN-0 extended-context original-Q4 public AutomationBench trial.

Only native context geometry changes: original four 4096-token slots are NOT
claimed; this experiment is one 32768-token slot. Model GGUF, source, official
benchmark harness/scorer and protected scientific receipts remain immutable.
No model-generated code/shell is executed (official in-memory WorldState tools).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.request

from chowder_batch import reap_q4_automationbench as base
from chowder_batch import reap_q4_windows_benchmark_watchdog_20261026 as watchdog
from chowder_batch import reap_q4_resource_admission as admission_policy
from chowder_batch import reap_serve

PROFILE = "NONCOMPARABLE_Q4_EXPANDED_CONTEXT_1X32768"
FLAGS = ["--fit", "on", "-np", "1", "-c", "32768", "--jinja"]
PORT = base.PORT
# Native expected KV doubles from original 16k context (320MiB) to 32k
# (640MiB): budget another 512 MiB start VRAM above the original 12GiB gate.
EXTRA_GPU_MIB = 512
MAX_START_SECONDS = 360
MAX_TASK_SECONDS = 3600
MALFORMED_TOOL_ABORT_COUNT = 5


def _preflight(out: Path, *, count: int, task_name: str | None = None) -> dict:
    """Conservative ORIGINAL frozen first-start gate, plus VRAM for extra KV."""
    if (type(count) is not int or count not in (1, base.SHORT_PUBLIC_COUNT, base.FULL_TASK_COUNT)
            or (count == 1) != (task_name is not None)
            or (task_name is not None and task_name not in base.SHORT_PUBLIC_TASKS)):
        raise ValueError("only one pinned named task, all six pinned tasks or all 600")
    base._output_check(out)
    admission, measured = base._fresh_admission(out.parent)
    extra_gpu = watchdog.free_vram_mib()
    if (admission["status"] != "ADMITTED_TO_ATTEMPT_NOT_SCIENTIFICALLY_QUALIFIED"
            or measured["gpu_free_mib"] < admission_policy.ORIGINAL_GPU_START_FLOOR_MIB + EXTRA_GPU_MIB
            or extra_gpu < admission_policy.ORIGINAL_GPU_START_FLOOR_MIB + EXTRA_GPU_MIB):
        return {"status": "REFUSED_RESOURCE_ADMISSION", "reason": admission.get("reasons"),
                "current": measured, "requires_extra_gpu_mib": EXTRA_GPU_MIB}
    return {"status": "READY_TO_ATTEMPT_NONCOMPARABLE_EXPLORATORY",
            "original_admission": admission, "measured": measured,
            "extra_gpu_mib": EXTRA_GPU_MIB}


def _commands(out: Path, *, count: int, task_name: str | None = None) -> tuple[list[str], list[str]]:
    original_server, official = base._commands(out, count=count, port=PORT)
    expected = [str(watchdog.SERVER), *reap_serve.FLAGS]
    if original_server[:len(expected)] != expected:
        raise ValueError("source registered serving command changed")
    extended_server = [original_server[0], *FLAGS, *original_server[len(expected):]]
    if ("--fit", "on") not in tuple(zip(extended_server[1::2], extended_server[2::2])):
        raise ValueError("native --fit on missing")
    if task_name is not None:
        if count != 1 or task_name not in base.SHORT_PUBLIC_TASKS:
            raise ValueError("individual task must be source-pinned")
        official.extend(("--tasks", task_name))
    return extended_server, official


def _verify_expanded_geometry(log_path: Path) -> dict:
    """Read native *observed* slot geometry, never infer from command alone."""
    if log_path.is_symlink() or not 0 < log_path.stat().st_size < 64 * 1024 * 1024:
        raise ValueError("native expanded start log unavailable")
    raw = log_path.read_bytes()
    content = raw.decode("utf-8", errors="replace")
    if not re.search(r"n_ctx\s*=\s*32768\b", content):
        raise ValueError("native overall context not actually 32768")
    if not re.search(r"n_ctx_seq\s*=\s*32768\b", content):
        raise ValueError("native per-sequence context not actually 32768")
    if not re.search(r"n_seq_max\s*=\s*1\b", content):
        raise ValueError("native concurrency not actually one slot")
    return {"native_log_sha256": hashlib.sha256(raw).hexdigest(),
            "profile": PROFILE, "observed_sequences": 1,
            "observed_total_context": 32768, "observed_context_per_seq": 32768}


def _repeated_tool_json_errors(log_path: Path) -> int:
    """Fail fast on repeated *native* 500s without repairing model responses."""
    if not log_path.is_file() or log_path.is_symlink() or log_path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("native log unavailable for bounded tool-call watchdog")
    return log_path.read_bytes().count(b"Failed to parse tool call arguments as JSON")


def _run(out: Path, *, count: int, task_name: str | None = None) -> dict:
    if os.name != "nt":
        raise ValueError("must use admitted native Windows Q4 host")
    admission = _preflight(out, count=count, task_name=task_name)
    if admission["status"] != "READY_TO_ATTEMPT_NONCOMPARABLE_EXPLORATORY":
        return admission
    source = base._source_check()
    if (not watchdog.GGUF.is_file() or watchdog.GGUF.is_symlink()
            or not watchdog.SERVER.is_file() or watchdog.SERVER.is_symlink()
            or watchdog.sha(watchdog.GGUF) != watchdog.MODEL_SHA):
        raise ValueError("source GGUF SHA or native server identity mismatch")
    reap_serve._require_free_loopback_port(PORT)
    recheck = _preflight(out, count=count, task_name=task_name)
    if recheck["status"] != "READY_TO_ATTEMPT_NONCOMPARABLE_EXPLORATORY":
        raise ValueError("resource admission changed before native launch")
    native_command, bench_command = _commands(out, count=count, task_name=task_name)
    out.mkdir(mode=0o700)
    manifest = {"profile": PROFILE, "noncomparable_to_frozen_4_way": True,
                "gen0_status": "BLOCKED", "chosen": None, "release_approved": False,
                "gguf_sha256": watchdog.MODEL_SHA, "official_source": source,
                "original_flags": reap_serve.FLAGS, "experimental_flags": FLAGS,
                "official_evaluator_command": bench_command,
                "native_command": native_command, "admission": recheck,
                "count": count, "selected_task": task_name,
                "no_model_generated_program_execution": True}
    (out / "experimental-manifest.private.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf8")
    native = None
    evaluator = None
    result = None
    started = time.monotonic()
    telemetry = (out / "resources.private.jsonl").open("x", encoding="utf8")
    try:
        with (out / "native.private.log").open("xb") as native_log:
            native = subprocess.Popen(
                native_command, cwd=out, env={**os.environ, "LLAMA_ARG_LOG_VERBOSITY": "4"},
                stdin=subprocess.DEVNULL, stdout=native_log, stderr=subprocess.STDOUT,
                shell=False, creationflags=subprocess.CREATE_NO_WINDOW,
            )
            while True:
                base._resource_watch(native, out, started=started, timeout=MAX_START_SECONDS,
                                     log=telemetry, phase="expanded_context_native_load")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as reply:
                        if reply.status == 200:
                            break
                except OSError:
                    pass
                time.sleep(3)
            served_model = base._served_model(PORT, native)
            cuda = base._verify_native_cuda_startup(out / "native.private.log")
            context = _verify_expanded_geometry(out / "native.private.log")
            manifest.update({"native_cuda": cuda, "observed_context": context,
                             "served_model": served_model})
            bench_command[bench_command.index("--model") + 1] = served_model
            (out / "experimental-manifest.private.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True), encoding="utf8")
            limited_env = {key: value for key, value in os.environ.items()
                           if key.upper() in ("PATH", "SYSTEMROOT", "TEMP", "TMP", "APPDATA",
                                              "LOCALAPPDATA", "USERPROFILE")}
            limited_env.update({"OPENAI_API_KEY": "local-source-bound-only", "HF_HUB_OFFLINE": "1",
                                "PYTHON_DOTENV_DISABLED": "1", "PYTHONNOUSERSITE": "1"})
            with (out / "official-benchmark.private.log").open("xb") as official_log:
                evaluator = subprocess.Popen(
                    bench_command, cwd=base.BENCH_ROOT, env=limited_env,
                    stdin=subprocess.DEVNULL, stdout=official_log, stderr=subprocess.STDOUT,
                    shell=False, creationflags=subprocess.CREATE_NO_WINDOW,
                )
                while evaluator.poll() is None:
                    base._resource_watch(native, out, started=started, timeout=MAX_TASK_SECONDS,
                                         log=telemetry, phase="expanded_context_official_evaluator")
                    if _repeated_tool_json_errors(out / "native.private.log") >= MALFORMED_TOOL_ABORT_COUNT:
                        (out / "native-tool-json-fail-closed.private.json").write_text(
                            json.dumps({"status": "REPEATED_MALFORMED_MODEL_TOOL_JSON_UNSCORED",
                                        "threshold": MALFORMED_TOOL_ABORT_COUNT,
                                        "profile": PROFILE, "selected_task": task_name,
                                        "gen0_status": "BLOCKED"}), encoding="utf8")
                        raise RuntimeError("native rejected repeated malformed model tool-call arguments")
                    time.sleep(3)
                if evaluator.returncode != 0:
                    raise RuntimeError("official expanded experiment evaluator failed; not scored")
                raw = out / "public-automationbench.private.json"
                if (not raw.is_file() or raw.is_symlink()
                        or not 0 < raw.stat().st_size < base.MAX_PRIVATE_JSON_BYTES):
                    raise ValueError("official benchmark output absent or oversized")
                result = base._triage_official_output(raw, count=count, served_id=served_model)
                if task_name is not None:
                    official_tasks = json.loads(raw.read_bytes())["tasks"]
                    if len(official_tasks) != 1 or official_tasks[0]["name"] != task_name:
                        raise ValueError("official single-task output identity mismatch")
    finally:
        try:
            if evaluator is not None and evaluator.poll() is None:
                watchdog.kill_owned_tree(evaluator)
            if native is not None and native.poll() is None:
                watchdog.kill_owned_tree(native)
            if native is not None:
                watchdog.verify_native_shutdown(native.pid, PORT)
        finally:
            telemetry.close()
    return {"status": result["status"], "census": result, "private_path": str(out),
            "profile": PROFILE, "gen0_status": "BLOCKED", "chosen": None,
            "release_approved": False, "official_private_leaderboard_comparable": False}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--count", type=int, choices=(base.SHORT_PUBLIC_COUNT, base.FULL_TASK_COUNT), default=base.SHORT_PUBLIC_COUNT)
    p.add_argument("--task", choices=base.SHORT_PUBLIC_TASKS)
    args = p.parse_args()
    count = 1 if args.task is not None else args.count
    try:
        result = (_run(args.out, count=count, task_name=args.task) if args.execute
                  else _preflight(args.out, count=count, task_name=args.task))
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        result = {"status": "REFUSED", "reason": type(exc).__name__, "profile": PROFILE,
                  "gen0_status": "BLOCKED", "chosen": None, "release_approved": False}
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] in ("READY_TO_ATTEMPT_NONCOMPARABLE_EXPLORATORY",
                                     "RAW_COMPLETED_PUBLIC_ROLLOUTS_NEED_INDEPENDENT_ROW_AUDIT",
                                     "ALL_OFFICIAL_ROLLOUTS_ABORTED_UNSCORED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
