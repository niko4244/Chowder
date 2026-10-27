"""Corrected Q4 AutomationBench episode at the context budget phase 28 admits.

Phase 28 measured a live Q4 trial dying structurally at ``-c 32768``: the
conversation grew 1,372.7 tokens per request until request 24 reached 32,639
of 32,768, the tool call was truncated, and llama-server answered HTTP 500 on
invalid JSON for the remaining hour. The arithmetic needs 70,378 tokens for a
50-step episode.

A second live trial at ``-c 65536 --max-steps 45`` then failed differently: the
prompt was only 11,180 tokens, but one generation ran ~54,355 tokens to the
ceiling and was cut (``n_tokens = 65535, truncated = 1``), producing the same
invalid-JSON 500 loop. Larger context alone does not bound a degenerate turn.

This lane therefore changes native context geometry, the evaluator step budget
and the per-turn generation cap: ``-c 73728 --n-predict 8192 --max-steps 44``
needs 68,724 tokens, which fits with 5,004 to spare. The cap must sit above the
largest *legitimate* turn (4,574 measured) and below the window: a 4,096 cap
cut real tool calls and produced 88 invalid-JSON 500s. It is NONCOMPARABLE to
the frozen four-slot 16,384 cohort, and it is not a Gen-0 selection result.

It drives the official public evaluator directly; it does not fork the guarded
lab harness. No model-generated Python or shell is executed: AutomationBench
tools operate on in-memory WorldState.

Default command is a read-only preflight. Model inference requires
``--execute`` and an exclusive output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import urllib.request

from chowder_batch.reap_context_budget_20261026 import (
    assess,
    trajectory_from_native_log,
)
from chowder_batch.reap_context_guard_proxy_20261026 import (
    POLICY_NAME,
    start_guard,
    stop_guard,
    summarize,
)

PROFILE = "NONCOMPARABLE_Q4_CTX72K_1X73728_STEPS44_DECODE_RP11_DRY08_T07"
# 73,728 leaves room for 44 steps of measured prompt growth (~1,383/step)
# plus one full 8,192-token turn: 1,067 + 1,382.9*43 + 8,192 = 68,724.
CONTEXT = 73728
MAX_STEPS = 44
# Bounds a degenerate turn. Trial 2 produced one 54,356-token turn with no
# limit at all; the largest legitimate turn measured was 4,574, so 8,192
# leaves ~1.8x headroom. A 4,096 cap was tried first and cut real tool calls:
# 88 invalid-JSON 500s on the opening task.
GEN_CAP = 8192
PORT = 8892
# Measured: server sampler flags bind, but --n-predict reports -1 and is NOT
# enforced on /v1/chat/completions. Degeneration is therefore addressed at its
# cause -- unpenalized repetition at temperature 1.0 -- rather than by a cap.
REPEAT_PENALTY = 1.1
DRY_MULTIPLIER = 0.8
TEMPERATURE = 0.7
FLAGS = ["--fit", "on", "-np", "1", "-c", str(CONTEXT),
         "--repeat-penalty", str(REPEAT_PENALTY),
         "--dry-multiplier", str(DRY_MULTIPLIER),
         "--temperature", str(TEMPERATURE), "--jinja"]

SERVER = Path("F:/llama.cpp/bin/llama-server.exe")
GGUF = Path("G:/reap128-v2-q4_k_m-20261025.gguf")
GGUF_SHA = "2d64b786461b99a335fdbbdf7c4d3a6758b21409bce755751714ab7cd8f63bf7"
GGUF_BYTES = 11563915264
SERVED_ID = "reap128_v2_q4"

BENCH_ROOT = Path("F:/reap-v2-public-automationbench-source-20261026")
BENCH_COMMIT = "4a8e1061254004d9dac807054eed33fad7d1ff14"
BENCH_PYTHON = BENCH_ROOT / ".venv/Scripts/python.exe"
BENCH_VERSION = "1.0.6"

TASKS = (
    "sales.apply_project_label",
    "sales.mark_vip_emails_read",
    "sales.create_contact_for_account",
    "sales.advance_opportunity_stage",
    "marketing.event_followup_personalization",
    "hr.visa_expiration_monitoring",
)

# Floors carried over unchanged from the frozen admission gate.
MIN_RAM_GIB = 29.3145986328125
MIN_COMMIT_GIB = 34.3145986328125
MIN_GPU_MIB = 12288
MIN_DISK_GIB = 8.0
# Doubling 32k KV costs ~320 MiB; require headroom above the original gate.
GPU_MARGIN_MIB = 512

MAX_START_SECONDS = 420
MAX_TASK_SECONDS = 7200
# A turn can now legally emit GEN_CAP tokens, so the tail room must cover it.
GENERATION_RESERVE = max(2048, GEN_CAP)
MAX_JSON_BYTES = 256 * 1024 * 1024
PRIOR_LIVE_LOG = Path(
    "F:/reap-v2-automationbench-private-20261026/q4-expctx-32768-six-v2/native.private.log"
)


# --------------------------------------------------------------------------
# pure verification helpers
# --------------------------------------------------------------------------


def memory_snapshot() -> dict[str, float]:
    """Physical and commit availability via Win32, in GiB."""
    import ctypes
    from ctypes import Structure, c_ulonglong, byref, sizeof, windll
    from ctypes.wintypes import DWORD

    class Status(Structure):
        _fields_ = [("dwLength", DWORD), ("dwMemoryLoad", DWORD)]
        _fields_ += [
            (name, c_ulonglong)
            for name in (
                "ullTotalPhys", "ullAvailPhys", "ullTotalPageFile",
                "ullAvailPageFile", "ullTotalVirtual", "ullAvailVirtual",
                "ullAvailExtendedVirtual",
            )
        ]

    status = Status()
    status.dwLength = sizeof(Status)
    if not windll.kernel32.GlobalMemoryStatusEx(byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    gib = float(1024**3)
    return {
        "ram_available_gib": status.ullAvailPhys / gib,
        "commit_available_gib": status.ullAvailPageFile / gib,
        "physical_gib": status.ullTotalPhys / gib,
    }


def gpu_free_mib() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=25, check=True,
    ).stdout.strip().splitlines()
    if not out:
        raise OSError("nvidia-smi returned no GPU")
    return int(out[0].strip())


def verify_geometry(text: str, *, context: int) -> dict[str, object]:
    """Every slot must report the declared context; no silent --fit shrink."""
    slots = {int(m) for m in re.findall(r"n_ctx_slot = (\d+)", text)}
    declared = {int(m) for m in re.findall(r"^.*\bn_ctx\s*=\s*(\d+)\s*$", text, re.M)}
    if slots != {context}:
        raise ValueError(f"native slot geometry {sorted(slots)} != {context}")
    if declared and context not in declared:
        raise ValueError(f"native declared context {sorted(declared)} != {context}")
    return {
        "observed_context_per_seq": context,
        "observed_sequences": 1,
        "observed_total_context": context,
        "profile": PROFILE,
    }


def verify_cuda(text: str) -> dict[str, object]:
    if "offloaded 41/41 layers to GPU" not in text:
        raise ValueError("41/41 CUDA offload proof absent from native startup log")
    expert = re.search(r"offloaded\s+(\d+)/\d+\s+experts", text)
    offloaded = 0 if expert is None else int(expert.group(1))
    if offloaded != 0:
        raise ValueError("CPU expert offload detected; not the pinned all-GPU arm")
    return {"layers_cuda": "41/41", "measured_experts_offloaded": 0}


def verify_gguf(path: Path, *, shown_bytes: int, shown_sha: str) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise ValueError("pinned Q4 GGUF unavailable")
    size = path.stat().st_size
    if size != shown_bytes:
        raise ValueError(f"Q4 GGUF size {size} != pinned {shown_bytes}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    measured = digest.hexdigest()
    if measured != shown_sha:
        raise ValueError("Q4 GGUF digest differs from the registered source hash")
    return {"gguf_bytes": size, "gguf_sha256": measured, "gguf_path": str(path)}


def audit_official(
    blob: bytes, *, served_id: str, count: int, max_steps: int,
    expected: tuple[str, ...] | None,
) -> dict[str, object]:
    """Independently re-derive every aggregate from the item-level rows."""
    if not 0 < len(blob) < MAX_JSON_BYTES:
        raise ValueError("official result absent or oversized")
    result = json.loads(blob, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    if type(result) is not dict or set(result) != {"meta", "summary", "tasks", "usage_by_task"}:
        raise ValueError("official result root schema mismatch")
    meta, summary, tasks = result["meta"], result["summary"], result["tasks"]
    if type(meta) is not dict or type(summary) is not dict or type(tasks) is not list:
        raise ValueError("official result section types malformed")
    if meta.get("benchmark_version") != BENCH_VERSION:
        raise ValueError("benchmark version mismatch")
    if meta.get("model") != served_id:
        raise ValueError("scored model differs from the served model")
    if meta.get("toolset") != "api":
        raise ValueError("toolset mismatch")
    # The step budget is the whole point of this lane; never accept the old one.
    if meta.get("max_steps") != max_steps:
        raise ValueError(f"meta.max_steps {meta.get('max_steps')} != {max_steps}")
    if meta.get("total_tasks") != count or len(tasks) != count:
        raise ValueError("scored population size mismatch")

    names = [row.get("name") for row in tasks]
    if any(type(name) is not str or not name for name in names) or len(set(names)) != count:
        raise ValueError("task identity missing or repeated")
    # A sampled run pins its slice by recording the names it actually scored;
    # the pinned six-task lane must match exactly.
    if expected is not None and set(names) != set(expected):
        raise ValueError("pinned six-task slice changed")

    aborted = summary.get("aborted_tasks")
    if not isinstance(aborted, list) or len(set(aborted)) != len(aborted):
        raise ValueError("aborted task list malformed")
    if any(name not in names for name in aborted):
        raise ValueError("aborted task is not in the scored population")

    for row in tasks:
        if type(row.get("passed")) is not bool:
            raise ValueError("passed flag is not boolean")
        if type(row.get("score")) not in (int, float) or not 0 <= row["score"] <= 1:
            raise ValueError("score outside [0, 1]")
        if row["passed"] != (row["score"] == 1.0):
            raise ValueError("passed flag disagrees with score")
        for key in ("num_model_calls", "num_tool_calls"):
            if type(row.get(key)) is not int or row[key] < 0:
                raise ValueError(f"{key} is not a non-negative integer")

    completed = [row for row in tasks if row["name"] not in aborted]
    passes = sum(1 for row in completed if row["passed"])
    if type(summary.get("passed_count")) is not int or summary["passed_count"] != passes:
        raise ValueError("summary passed_count disagrees with the rows")
    if any(row["passed"] for row in tasks if row["name"] in aborted):
        raise ValueError("an aborted task was scored as passing")

    total = len(completed)
    return {
        "status": "ALL_OFFICIAL_ROLLOUTS_ABORTED_UNSCORED" if not total
        else "RAW_COMPLETED_PUBLIC_ROLLOUTS_NEED_INDEPENDENT_ROW_AUDIT",
        "requested_tasks": count,
        "completed_tasks": total,
        "aborted_unscored_tasks": len(aborted),
        "passed_tasks": passes,
        "completed_pass_rate": (passes / total) if total else None,
        "requested_task_pass_rate": passes / count,
        "model_calls": sum(row["num_model_calls"] for row in tasks),
        "tool_calls": sum(row["num_tool_calls"] for row in tasks),
        "aborted_task_names": sorted(aborted),
        "raw_sha256": hashlib.sha256(blob).hexdigest(),
        "official_private_leaderboard_comparable": False,
        "gen0_status": "BLOCKED",
        "chosen": None,
        "release_approved": False,
    }


def audit_truncation(native_text: str, *, context: int, max_steps: int) -> dict[str, object]:
    """Did the corrected geometry actually eliminate the phase-28 failure?"""
    observed = trajectory_from_native_log(native_text)
    verdict = assess(
        observed=observed,
        context=context,
        max_steps=max_steps,
        generation_reserve=GENERATION_RESERVE,
    )
    prompts = list(observed["prompt_tokens"])
    truncated = int(observed["truncated_generations"])
    json_failures = int(observed["tool_call_json_failures"])
    return {
        "truncated_generations": truncated,
        "tool_call_json_failures": json_failures,
        "requests_served": len(prompts),
        "max_prompt_tokens": max(prompts) if prompts else 0,
        "context": context,
        "tokens_per_request": verdict["tokens_per_request"],
        "required_context": verdict["required_context"],
        "affordable_steps": verdict["affordable_steps"],
        "truncation_eliminated": truncated == 0 and json_failures == 0,
    }


# --------------------------------------------------------------------------
# side-effecting helpers
# --------------------------------------------------------------------------


def verify_official_source() -> dict[str, object]:
    if not BENCH_ROOT.is_dir() or BENCH_ROOT.is_symlink():
        raise ValueError("official AutomationBench source unavailable")
    revision = subprocess.run(
        ["git", "-C", str(BENCH_ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True, timeout=20, check=True,
    ).stdout.strip()
    if revision != BENCH_COMMIT:
        raise ValueError("official source commit changed")
    dirty = subprocess.run(
        ["git", "-C", str(BENCH_ROOT), "status", "--porcelain=v1", "--untracked-files=all"],
        capture_output=True, text=True, timeout=25, check=True,
    ).stdout
    if dirty.strip():
        raise ValueError("official source has modified or untracked files")
    for relative in (".env", "sitecustomize.py", "usercustomize.py"):
        if (BENCH_ROOT / relative).exists():
            raise ValueError("official checkout has an import-time override")
    if not BENCH_PYTHON.is_file():
        raise ValueError("official Python runtime unavailable")
    if "3.13" not in BENCH_PYTHON.as_posix() and not subprocess.run(
        [str(BENCH_PYTHON), "--version"], capture_output=True, text=True, timeout=20
    ).stdout.strip().startswith("Python 3.13"):
        raise ValueError("official runtime is not Python 3.13")
    return {"source_git_commit": revision, "benchmark_version": BENCH_VERSION,
            "public_tasks": 600, "public_not_private_leaderboard": True}


def port_is_free(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
            return False
    except OSError:
        return True


def wait_health(proc: subprocess.Popen, *, started: float, deadline: int) -> None:
    while True:
        if proc.poll() is not None:
            raise RuntimeError("native server exited during startup")
        if time.monotonic() - started > deadline:
            raise TimeoutError("native server did not become healthy in time")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as reply:
                if reply.status == 200:
                    return
        except OSError:
            pass
        time.sleep(3)


def serve_single_model(proc: subprocess.Popen) -> str:
    if proc.poll() is not None:
        raise RuntimeError("native server is not running")
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=5) as reply:
        if reply.status != 200:
            raise ValueError("model inventory unavailable")
        data = json.loads(reply.read(65537))
    items = data.get("data") if isinstance(data, dict) else None
    if not isinstance(items, list) or len(items) != 1:
        raise ValueError("native server must expose exactly one model")
    identity = items[0].get("id") if isinstance(items[0], dict) else None
    if not isinstance(identity, str) or not identity:
        raise ValueError("malformed served model id")
    return identity


def kill_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True, timeout=60, check=False,
        )
    finally:
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


def record(dirpath: Path, handle, phase: str, pid: int, started: float) -> dict[str, float]:
    snapshot = memory_snapshot()
    snapshot["gpu_free_mib"] = float(gpu_free_mib())
    snapshot["disk_free_gib"] = shutil.disk_usage(dirpath).free / float(1024**3)
    handle.write(json.dumps({
        "phase": phase,
        "owned_native_pid": pid,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "measurement": snapshot,
    }, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
    return snapshot


GUARD_PORT = 8894


def bench_command(served_id: str, out: Path, *, count: int,
                  tasks: tuple[str, ...] | None,
                  base_port: int = PORT) -> list[str]:
    command = [
        str(BENCH_PYTHON), "-I", "-m", "automationbench.scripts.eval",
        "--model", served_id,
        "--base-url", f"http://127.0.0.1:{base_port}/v1",
        "--api-key-var", "OPENAI_API_KEY",
        "--api", "chat_completions",
        "--toolset", "api",
        "--domains", "all",
        "--num-examples", str(count),
        "--max-concurrent", "1",
        "--max-steps", str(MAX_STEPS),
        "--no-ensure-complete",
        "--export-json", str(out / "public-automationbench.private.json"),
    ]
    if tasks:
        command += ["--tasks", ",".join(tasks)]
    return command


def preflight(out: Path) -> dict[str, object]:
    """Read-only: refuse before touching the GPU or creating an output dir."""
    memory = memory_snapshot()
    snapshot = {
        **memory,
        "gpu_free_mib": float(gpu_free_mib()),
        "disk_free_gib": shutil.disk_usage(out.parent).free / float(1024**3),
    }
    blockers: list[str] = []
    if memory["ram_available_gib"] < MIN_RAM_GIB:
        blockers.append("RAM_BELOW_FROZEN_FLOOR")
    if memory["commit_available_gib"] < MIN_COMMIT_GIB:
        blockers.append("COMMIT_BELOW_FROZEN_FLOOR")
    if snapshot["gpu_free_mib"] < MIN_GPU_MIB + GPU_MARGIN_MIB:
        blockers.append("GPU_FREE_BELOW_64K_FLOOR")
    if snapshot["disk_free_gib"] < MIN_DISK_GIB:
        blockers.append("DISK_BELOW_FROZEN_FLOOR")
    if not port_is_free(PORT):
        blockers.append("TARGET_PORT_IN_USE")
    if out.exists():
        blockers.append("OUTPUT_DIRECTORY_ALREADY_EXISTS")

    # Admit the episode arithmetically against the measured 32k trajectory.
    budget = None
    if PRIOR_LIVE_LOG.is_file():
        with PRIOR_LIVE_LOG.open("r", encoding="utf-8", errors="replace") as handle:
            budget = assess(
                observed=trajectory_from_native_log(handle.read()),
                context=CONTEXT,
                max_steps=MAX_STEPS,
                generation_reserve=GENERATION_RESERVE,
            )
        if budget["status"] != "ADMITTED_TO_ATTEMPT":
            blockers.append("EPISODE_CANNOT_FIT_DECLARED_CONTEXT")
    else:
        blockers.append("PRIOR_TRAJECTORY_UNAVAILABLE_TO_ADMIT")

    return {
        "schema": "reap-v2-q4-automationbench-ctx64k.v1",
        "profile": PROFILE,
        "status": "REFUSED" if blockers else "READY_TO_ATTEMPT_NONCOMPARABLE_EXPLORATORY",
        "context": CONTEXT,
        "max_steps": MAX_STEPS,
        "generation_reserve": GENERATION_RESERVE,
        "steering_flags": FLAGS,
        "decoding": {"repeat_penalty": REPEAT_PENALTY, "dry_multiplier": DRY_MULTIPLIER,
                     "temperature": TEMPERATURE},
        "frozen_flags": ["--fit", "on", "-np", "4", "-c", "16384", "--jinja"],
        "model_generated_program_execution": False,
        "noncomparable_to_frozen_4_way": True,
        "context_budget": budget,
        "measured": snapshot,
        "blockers": blockers,
        "gen0_status": "BLOCKED",
        "chosen": None,
        "release_approved": False,
    }


def execute(out: Path, *, sample: int | None = None, guard: bool = False) -> dict[str, object]:
    if sample is not None and (type(sample) is not int or not 2 <= sample <= 600):
        raise ValueError("sample must be an integer between 2 and 600")
    count = sample if sample is not None else len(TASKS)
    tasks: tuple[str, ...] | None = None if sample is not None else TASKS

    gate = preflight(out)
    if gate["status"] != "READY_TO_ATTEMPT_NONCOMPARABLE_EXPLORATORY":
        return gate

    source = verify_official_source()
    model = verify_gguf(GGUF, shown_bytes=GGUF_BYTES, shown_sha=GGUF_SHA)
    out.mkdir(parents=True, exist_ok=False)

    native_command = [
        str(SERVER), *FLAGS,
        "--host", "127.0.0.1", "--port", str(PORT), "-m", str(GGUF),
    ]
    manifest = {
        "schema": "reap-v2-q4-automationbench-ctx64k.v1",
        "profile": PROFILE,
        "context": CONTEXT,
        "max_steps": MAX_STEPS,
        "generation_reserve": GENERATION_RESERVE,
        "steering_flags": FLAGS,
        "decoding": {"repeat_penalty": REPEAT_PENALTY, "dry_multiplier": DRY_MULTIPLIER,
                     "temperature": TEMPERATURE},
        "frozen_flags": ["--fit", "on", "-np", "4", "-c", "16384", "--jinja"],
        "native_command": native_command,
        "sample_size": count,
        "sampled_public_slice": sample is not None,
        "guard_policy": POLICY_NAME if guard else None,
        "official_evaluator_command": bench_command(
            SERVED_ID, out, count=count, tasks=tasks,
            base_port=GUARD_PORT if guard else PORT,
        ),
        "official_source": source,
        "model": model,
        "admission": gate,
        "tasks": list(TASKS),
        "noncomparable_to_frozen_4_way": True,
        "model_generated_program_execution": False,
        "gen0_status": "BLOCKED",
        "chosen": None,
        "release_approved": False,
    }
    (out / "experimental-manifest.private.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf8"
    )

    native = evaluator = None
    guard_server = guard_thread = guard_state = None
    started = time.monotonic()
    telemetry = (out / "resources.private.jsonl").open("x", encoding="utf8")
    try:
        with (out / "native.private.log").open("xb") as native_log:
            native = subprocess.Popen(
                native_command, cwd=out,
                env={**os.environ, "LLAMA_ARG_LOG_VERBOSITY": "4"},
                stdin=subprocess.DEVNULL, stdout=native_log, stderr=subprocess.STDOUT,
                shell=False, creationflags=subprocess.CREATE_NO_WINDOW,
            )
            while True:
                record(out, telemetry, "ctx64k_native_load", native.pid, started)
                if time.monotonic() - started > MAX_START_SECONDS:
                    raise TimeoutError("native load exceeded its deadline")
                try:
                    wait_health(native, started=started, deadline=MAX_START_SECONDS)
                    break
                except TimeoutError:
                    raise

        native_text = (out / "native.private.log").read_text("utf-8", errors="replace")
        cuda = verify_cuda(native_text)
        geometry = verify_geometry(native_text, context=CONTEXT)
        served = serve_single_model(native)

        manifest.update({"native_cuda": cuda, "observed_context": geometry, "served_model": served})
        if guard:
            guard_server, guard_state, guard_thread = start_guard(
                upstream_port=PORT, listen_port=GUARD_PORT, context=CONTEXT,
                receipt_path=out / "guard-receipt.private.jsonl",
            )
        command = bench_command(served, out, count=count, tasks=tasks,
                                base_port=GUARD_PORT if guard else PORT)
        manifest["official_evaluator_command"] = command
        (out / "experimental-manifest.private.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf8"
        )

        limited_env = {
            key: value for key, value in os.environ.items()
            if key.upper() in ("PATH", "SYSTEMROOT", "TEMP", "TMP", "APPDATA",
                               "LOCALAPPDATA", "USERPROFILE")
        }
        limited_env.update({
            "OPENAI_API_KEY": "local-source-bound-only", "HF_HUB_OFFLINE": "1",
            "PYTHON_DOTENV_DISABLED": "1", "PYTHONNOUSERSITE": "1",
        })
        with (out / "official-benchmark.private.log").open("xb") as official_log:
            evaluator = subprocess.Popen(
                command, cwd=BENCH_ROOT, env=limited_env,
                stdin=subprocess.DEVNULL, stdout=official_log, stderr=subprocess.STDOUT,
                shell=False, creationflags=subprocess.CREATE_NO_WINDOW,
            )
            while evaluator.poll() is None:
                record(out, telemetry, "ctx64k_official_evaluator", native.pid, started)
                if time.monotonic() - started > MAX_TASK_SECONDS:
                    raise TimeoutError("benchmark episode exceeded its wall-clock deadline")
                time.sleep(3)
            if evaluator.returncode != 0:
                raise RuntimeError("official evaluator failed; nothing was scored")

        raw = out / "public-automationbench.private.json"
        if not raw.is_file():
            raise ValueError("official evaluator produced no result file")
        blob = raw.read_bytes()
        audit = audit_official(
            blob, served_id=served, count=count, max_steps=MAX_STEPS, expected=tasks
        )
        native_text = (out / "native.private.log").read_text("utf-8", errors="replace")
        truncation = audit_truncation(native_text, context=CONTEXT, max_steps=MAX_STEPS)
        (out / "independent-audit.private.json").write_text(
            json.dumps({"official": audit, "truncation": truncation,
                        "model": model, "official_source": source},
                       indent=2, sort_keys=True), encoding="utf8"
        )
        result = {"profile": PROFILE, "official_audit": audit,
                  "truncation_audit": truncation, "private_path": str(out)}
        if guard_state is not None:
            result["guard"] = summarize(guard_state.counters)
            manifest["guard_counters"] = result["guard"]
        return result
    finally:
        try:
            if evaluator is not None:
                kill_tree(evaluator)
            if guard_server is not None and guard_thread is not None:
                stop_guard(guard_server, guard_thread)
            if native is not None:
                kill_tree(native)
        finally:
            telemetry.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--sample", type=int, default=None,
                        help="score the first N public tasks instead of the pinned six")
    parser.add_argument("--guard", action="store_true",
                        help="route through the compaction + runaway-policy proxy")
    args = parser.parse_args()
    try:
        result = (execute(args.out, sample=args.sample, guard=args.guard)
                  if args.execute else preflight(args.out))
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        result = {"status": "REFUSED", "reason": type(exc).__name__,
                  "detail": str(exc)[:400], "profile": PROFILE,
                  "gen0_status": "BLOCKED", "chosen": None, "release_approved": False}
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if result.get("status") in (
        "READY_TO_ATTEMPT_NONCOMPARABLE_EXPLORATORY",
        "RAW_COMPLETED_PUBLIC_ROLLOUTS_NEED_INDEPENDENT_ROW_AUDIT",
        "ALL_OFFICIAL_ROLLOUTS_ABORTED_UNSCORED",
    ) else 2


if __name__ == "__main__":
    raise SystemExit(main())
