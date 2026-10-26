"""Frozen-model Q4 -> official public AutomationBench v1.0.6 experimental lane.

The original BF16/Q8/Q4 scientific selection remains BLOCKED. This launches
only a new source-bound private public-task experiment, never mutates the
original scientific rows or executes generated Python/shell. Default command
is read-only. Model inference requires the original Q4 Windows admission,
verified official source/runtime and an exclusive output folder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import TextIO
import urllib.request

from chowder_batch import reap_serve
from chowder_batch import reap_q4_windows_benchmark_watchdog_20261026 as watchdog
from chowder_batch.reap_v2_lab import _fresh_admission

BENCH_ROOT = Path("F:/reap-v2-public-automationbench-source-20261026")
BENCH_COMMIT = "4a8e1061254004d9dac807054eed33fad7d1ff14"
BENCH_PYTHON = BENCH_ROOT / ".venv/Scripts/python.exe"
PUBLIC_600_PROMPTS_SHA256 = "1c8feeec1f680b05b392a5d00fc634edb143c3e7bdd152c476af50cebcec57b5"
MODEL = "reap128_v2_q4"
PORT = 8891
PRIVATE_MAX_SMOKE = 1
FULL_TASK_COUNT = 600
HEALTH_TIMEOUT_SECONDS = 300
MAX_SMOKE_WALL_SECONDS = 1800
PRIVATE_ROOT = Path("F:/reap-v2-automationbench-private-20261026")
MAX_PRIVATE_JSON_BYTES = 256 * 1024 * 1024


def _source_check() -> dict:
    """Bind the real public 600 task source, Python3.13, CLI and evaluator."""
    if BENCH_ROOT.is_symlink() or not BENCH_ROOT.is_dir():
        raise ValueError("official AutomationBench source is unavailable")
    revision = subprocess.run(
        ["git", "-C", str(BENCH_ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True, timeout=15, check=True,
    ).stdout.strip()
    if revision != BENCH_COMMIT:
        raise ValueError("official AutomationBench Git commit changed")
    # A clean HEAD alone does not exclude an untracked sitecustomize.py or
    # untracked package shadow that Python imports before the official harness.
    dirty = subprocess.run(
        ["git", "-C", str(BENCH_ROOT), "status", "--porcelain=v1",
         "--untracked-files=all"],
        capture_output=True, text=True, timeout=20, check=True,
    )
    if dirty.stdout.strip():
        raise ValueError("official benchmark source has modified or untracked files")
    # Git ignore rules can hide credential files and import-time shadows from
    # the porcelain output; explicitly reject them before Python imports.
    for relative in (".env", "sitecustomize.py", "usercustomize.py",
                     ".venv/Lib/site-packages/sitecustomize.py",
                     ".venv/Lib/site-packages/usercustomize.py"):
        if (BENCH_ROOT / relative).exists():
            raise ValueError("official benchmark checkout has local execution override")
    if not BENCH_PYTHON.is_file() or BENCH_PYTHON.is_symlink():
        raise ValueError("private official Python3.13 runtime unavailable")
    version = subprocess.run([str(BENCH_PYTHON), "--version"],
                             capture_output=True, text=True, timeout=15, check=True)
    if not version.stdout.strip().startswith("Python 3.13."):
        raise ValueError("official scorer Python3.13 version mismatch")
    script = (
        "from automationbench.domains import DEFAULT_DOMAINS,get_combined_dataset;"
        "import hashlib,json;"
        "d=get_combined_dataset(DEFAULT_DOMAINS);"
        "rows=[(str(x['example_id']),str(x['prompt'])) for x in d];"
        "raw=json.dumps(rows,ensure_ascii=False,separators=(',',':')).encode('utf8');"
        "print(json.dumps({'n':len(rows),'domains':DEFAULT_DOMAINS,"
        "'source_sha256':hashlib.sha256(raw).hexdigest()}))"
    )
    data = subprocess.run([str(BENCH_PYTHON), "-I", "-c", script],
                          capture_output=True, text=True, cwd=BENCH_ROOT,
                          timeout=360, check=True)
    source = json.loads(data.stdout)
    if (source.get("n") != FULL_TASK_COUNT
            or source.get("domains") != ["sales", "marketing", "operations",
                                         "support", "finance", "hr"]
            or source.get("source_sha256") != PUBLIC_600_PROMPTS_SHA256):
        raise ValueError("public benchmark population/source integrity mismatch")
    return {"source_git_commit": revision,
            "public_source_sha256": PUBLIC_600_PROMPTS_SHA256,
            "public_tasks": FULL_TASK_COUNT, "python": version.stdout.strip(),
            "public_not_private_leaderboard": True}


def _commands(out: Path, *, count: int, port: int = PORT) -> tuple[list[str], list[str]]:
    if type(count) is not int or count not in (PRIVATE_MAX_SMOKE, FULL_TASK_COUNT):
        raise ValueError("use exactly one discovery task or all 600 public tasks")
    if type(port) is not int or port != PORT:
        raise ValueError("reserved private loopback port required")
    native = [str(watchdog.SERVER), *reap_serve.FLAGS, "--host", "127.0.0.1",
              "--port", str(port),
              "-m", str(watchdog.GGUF)]
    command = [
        str(BENCH_PYTHON), "-I", "-m", "automationbench.scripts.eval",
        "--model", MODEL, "--base-url", f"http://127.0.0.1:{port}/v1",
        "--api-key-var", "OPENAI_API_KEY", "--api", "chat_completions",
        "--toolset", "api", "--domains", "all",
        "--num-examples", str(count), "--max-concurrent", "1",
        "--max-steps", "50", "--no-ensure-complete",
        "--export-json", str(out / "public-automationbench.private.json"),
    ]
    return native, command


def _output_check(out: Path) -> None:
    if (not out.is_absolute() or out.exists() or out.is_symlink()
            or out.drive.upper() != "F:" or not out.parent.is_dir()):
        raise ValueError("choose an exclusive new absolute private experiment directory")
    # Prevent a junction or symlink ancestor from silently redirecting private
    # model responses into a worktree, model directory or foreign volume.
    for ancestor in (out, *out.parents):
        if ancestor.exists() and (ancestor.is_symlink()
                or (ancestor.stat().st_file_attributes & 0x400)):
            raise ValueError("private output ancestor is a reparse point")
    candidate = out.resolve(strict=False)
    root = PRIVATE_ROOT.resolve(strict=False)
    if candidate == root or not candidate.is_relative_to(root):
        raise ValueError("private experiment output must be inside dedicated F: root")
    banned = (Path(__file__).resolve().parents[1], BENCH_ROOT,
              watchdog.GGUF.parent, watchdog.SERVER.parent)
    if any(candidate == path.resolve() or candidate.is_relative_to(path.resolve())
           for path in banned):
        raise ValueError("cannot store private experiments inside source/model worktrees")


def _read_only_status(out: Path, *, count: int) -> dict:
    _commands(out, count=count)
    _output_check(out)
    admitted, live = _fresh_admission(out.parent)
    # Do not hash 11.56 GB or create an output directory on an inadmissible host.
    return {"status": "READY_TO_ATTEMPT" if admitted["status"].startswith("ADMITTED_")
            else "REFUSED_RESOURCE_ADMISSION",
            "official_suite": "AutomationBench v1.0.6 PUBLIC",
            "count_requested": count, "original_model_sha256": watchdog.MODEL_SHA,
            "serving_flags": reap_serve.FLAGS,
            "admission": admitted, "current": live,
            "gen0_status": "BLOCKED", "chosen": None, "release_approved": False}


def _resource_watch(proc: subprocess.Popen, out: Path, *, started: float, timeout: int,
                    log: TextIO | None = None, phase: str = "unspecified") -> None:
    if proc.poll() is not None:
        raise RuntimeError("private benchmark/native server terminated unexpectedly")
    if time.monotonic() - started > timeout:
        raise TimeoutError("bounded private experiment exceeded wall-clock deadline")
    snapshot = watchdog.sample(out)
    failures = watchdog.unsafe(snapshot)
    if log is not None:
        # Keep actual physical/commit/VRAM pressure as private, append-only,
        # fsynced provenance for every observed live load/evaluator interval.
        try:
            record = {"phase": phase, "owned_native_pid": proc.pid,
                      "elapsed_seconds": round(time.monotonic() - started, 3),
                      "measurement": snapshot, "failures": failures}
            encoded = json.dumps(record, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("invalid live resource telemetry cannot be recorded") from exc
        log.write(encoded + "\n")
        log.flush()
        os.fsync(log.fileno())
    if failures:
        raise RuntimeError("original Q4 runtime memory/commit/disk floor breached")


def _served_model(port: int, native: subprocess.Popen) -> str:
    """Use only the single model actually reported by this owned native listener."""
    reap_serve._require_owned_listener(native, port)
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as r:
        if r.status != 200 or int(r.headers.get("Content-Length", "0")) > 65536:
            raise ValueError("native model inventory unavailable or oversized")
        data = json.loads(r.read(65537))
    items = data.get("data") if isinstance(data, dict) else None
    if not isinstance(items, list) or len(items) != 1:
        raise ValueError("native server must expose exactly one served model")
    item = items[0]
    identity = item.get("id") if isinstance(item, dict) else None
    if not isinstance(identity, str) or not identity or len(identity) > 512:
        raise ValueError("native server reported malformed model ID")
    # Never silently present a different model as the source-hashed REAP arm.
    if identity not in (watchdog.GGUF.name, str(watchdog.GGUF), MODEL):
        raise ValueError("native model ID differs from registered Q4")
    return identity


def _verify_native_cuda_startup(log_path: Path) -> dict:
    """Re-derive fitted 41/41 CUDA layers and zero CPU experts from native bytes."""
    if log_path.is_symlink() or not 0 < log_path.stat().st_size < 64 * 1024 * 1024:
        raise ValueError("missing or unbounded native startup log")
    startup = log_path.read_bytes()
    lines, offloaded = reap_serve.parse_offload(startup.decode("utf-8", errors="replace"))
    if offloaded != 0 or not any("offloaded 41/41 layers to GPU" in line for line in lines):
        raise ValueError("native 41/41 CUDA or zero-CPU-expert proof absent")
    return {"startup_sha256": hashlib.sha256(startup).hexdigest(),
            "startup_bytes": len(startup), "layers_cuda": "41/41",
            "measured_experts_offloaded": 0}


def run_private_public_benchmark(out: Path, *, count: int = PRIVATE_MAX_SMOKE) -> dict:
    if os.name != "nt":
        raise ValueError("original registered Q4 Windows GPU host required")
    status = _read_only_status(out, count=count)
    if status["status"] != "READY_TO_ATTEMPT":
        return status
    approved = _source_check()
    if (not watchdog.GGUF.is_file() or not watchdog.SERVER.is_file()
            or watchdog.GGUF.is_symlink() or watchdog.SERVER.is_symlink()
            or watchdog.sha(watchdog.GGUF) != watchdog.MODEL_SHA):
        raise ValueError("original immutable Q4 model or native server identity mismatch")
    # Bind the actual runtime to an unused local port before any source writes.
    reap_serve._require_free_loopback_port(PORT)
    confirmed = _read_only_status(out, count=count)
    if confirmed["status"] != "READY_TO_ATTEMPT":
        raise ValueError("live admission changed immediately before launch")
    server_cmd, bench_cmd = _commands(out, count=count)
    out.mkdir(mode=0o700)
    private_manifest = {**approved, **confirmed,
                        "server_command": server_cmd,
                        "benchmark_command": bench_cmd,
                        "score_source": "official_public_task_completed_correctly",
                        "official_private_leaderboard_comparable": False}
    (out / "manifest.private.json").write_text(
        json.dumps(private_manifest, sort_keys=True, indent=2), encoding="utf-8")
    native: subprocess.Popen | None = None
    task: subprocess.Popen | None = None
    ok = False
    started = time.monotonic()
    resource_log = (out / "resource-samples.private.jsonl").open("x", encoding="utf8")
    try:
        with (out / "native.private.log").open("xb") as log:
            native = subprocess.Popen(server_cmd, cwd=out, stdin=subprocess.DEVNULL,
                                      stdout=log, stderr=subprocess.STDOUT, shell=False,
                                      env={**os.environ,
                                           "LLAMA_ARG_LOG_VERBOSITY": reap_serve.LOG_VERBOSITY},
                                      creationflags=subprocess.CREATE_NO_WINDOW)
            while True:
                _resource_watch(native, out, started=started,
                                timeout=HEALTH_TIMEOUT_SECONDS, log=resource_log,
                                phase="native_model_load")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                        if r.status == 200:
                            break
                except OSError:
                    pass
                time.sleep(3)
            served_id = _served_model(PORT, native)
            startup = _verify_native_cuda_startup(out / "native.private.log")
            # The original serving flags are unchanged. Use the verified model
            # ID presented by the native API instead of guessing an alias.
            bench_cmd[bench_cmd.index("--model") + 1] = served_id
            private_manifest["served_model_id"] = served_id
            private_manifest["native_cuda_witness"] = startup
            (out / "manifest.private.json").write_text(
                json.dumps(private_manifest, sort_keys=True, indent=2), encoding="utf-8")
            # Do not export cloud/provider credentials to simulated local tools.
            minimal_env = {name: value for name, value in os.environ.items()
                           if name.upper() in ("PATH", "SYSTEMROOT", "TEMP", "TMP",
                                               "APPDATA", "LOCALAPPDATA", "USERPROFILE")}
            minimal_env["OPENAI_API_KEY"] = "local-source-bound-only"
            minimal_env["HF_HUB_OFFLINE"] = "1"
            minimal_env["PYTHON_DOTENV_DISABLED"] = "1"
            minimal_env["PYTHONNOUSERSITE"] = "1"
            with (out / "official-benchmark.private.log").open("xb") as bench_log:
                task = subprocess.Popen(bench_cmd, cwd=BENCH_ROOT, env=minimal_env,
                                        stdin=subprocess.DEVNULL, stdout=bench_log,
                                        stderr=subprocess.STDOUT, shell=False,
                                        creationflags=subprocess.CREATE_NO_WINDOW)
                max_wall = MAX_SMOKE_WALL_SECONDS if count == PRIVATE_MAX_SMOKE else 12 * 3600
                while task.poll() is None:
                    _resource_watch(native, out, started=started, timeout=max_wall,
                                    log=resource_log, phase="official_task_evaluation")
                    time.sleep(3)
                if task.returncode != 0:
                    raise RuntimeError("official benchmark exited nonzero; not a scored result")
                raw = out / "public-automationbench.private.json"
                if (not raw.is_file() or raw.is_symlink()
                        or not 0 < raw.stat().st_size <= MAX_PRIVATE_JSON_BYTES):
                    raise ValueError("official benchmark result missing or unbounded")
                ok = True
    finally:
        try:
            # Kill only processes created by this runner, never protected services.
            if task is not None and task.poll() is None:
                watchdog.kill_owned_tree(task)
            if native is not None and native.poll() is None:
                watchdog.kill_owned_tree(native)
            if native is not None:
                watchdog.verify_native_shutdown(native.pid, PORT)
        finally:
            resource_log.close()
    return {"status": "RAW_PUBLIC_BENCHMARK_NEEDS_INDEPENDENT_ROW_AUDIT" if ok
            else "UNVERIFIED_INCOMPLETE",
            "public_requested": count, "private_output": str(out),
            "public_leaderboard_equivalent": False,
            "gen0_status": "BLOCKED", "chosen": None, "release_approved": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, choices=(1, FULL_TASK_COUNT), default=1)
    args = parser.parse_args(argv)
    try:
        result = (run_private_public_benchmark(args.out, count=args.count) if args.execute
                  else _read_only_status(args.out, count=args.count))
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        result = {"status": "REFUSED", "reason": type(exc).__name__,
                  "scientific_gen0": "BLOCKED", "chosen": None,
                  "release_approved": False}
    print(json.dumps(result, sort_keys=True, default=str))
    return 0 if result["status"] in ("READY_TO_ATTEMPT",
                                     "RAW_PUBLIC_BENCHMARK_NEEDS_INDEPENDENT_ROW_AUDIT") else 2


if __name__ == "__main__":
    raise SystemExit(main())
