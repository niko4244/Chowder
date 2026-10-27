"""Execute the ORIGINAL AutomationBench v1.0.6 public600 on original REAP128_v2 Q4.

This is a genuine model-to-official-evaluator runner, not an audit, fake world
state or simulation. The official scoring code remains git-clean and unchanged.
Each original source row is evaluated in its own official --skip N --num-examples
1 process. A single owned native model server is reused within a batch; after
each item finishes, its ORIGINAL official result and independently recomputed
strict score are saved in immutable, hash-bound per-item receipts. This solves
the official evaluator's save_results=False limitation without substituting a
scorer. Segmented ordering/context and guarded 72k serving are NONCOMPARABLE
to frozen four-slot Gen0 evaluation; no release approval is inferred.

Requirements:
  - Windows, original GGUF hash, original official source and python3.13
  - default admission: GPU >=12800MiB; RAM >=29.3146GiB,
    commit >=34.3146GiB
  - explicit --bypass-conservative-resource-admission permits the previously
    verified runtime minima with actual native allocator-fit GPU measurement,
    but NEVER bypasses existing owned ports, source/score hashes, original
    benchmark tasks, runtime host minimums, or frozen Gen0 gates
  - free 8892/8894; no concurrent model serving
  - same protected guard policy as previous genuine pilot

The default mode is a genuine read-only preflight. "--execute --max-new 600"
runs source-verified original items until all finish, a memory floor is breached,
or an 8,000-second global owned-runtime lease expires. Resumption rehashes every
previously scored original receipt; never changes model, prompts or rubric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from chowder_batch import reap_q4_automationbench_ctx64k_20261026 as q4
from chowder_batch import reap_q8_automationbench_kaggle_20261026 as source_ids
from chowder_batch.reap_context_guard_proxy_20261026 import start_guard, stop_guard

SCHEMA = "chowder.reap128-v2-q4-original-automationbench600-checkpointed.v1"
POPULATION = 600
MAX_RUN_SECONDS = 8000
MAX_ITEM_SECONDS = 2400
ORIGINAL_RAM_RUNTIME_FLOOR_GIB = 7
ORIGINAL_COMMIT_RUNTIME_FLOOR_GIB = 8
ORIGINAL_DISK_RUNTIME_FLOOR_GIB = 8
# Measured in source-pinned REAL Q4 llama-server native startup (MiB):
# CUDA model 10619.88 + f16 KV 1440 + recurrent 62.81 + compute 172.05.
# A WDDM host observation showed another 260MiB unaccounted reserved.
# This minimum is physical GPU allocator fit, NOT the original conservative
# 12,288+512MiB prestart threshold; it is exploratory only.
MEASURED_NATIVE_CUDA_REQUIRED_MIB = 10619.88 + 1440 + 62.81 + 172.05
MEASURED_WDDM_UNACCOUNTED_MIB = 260
OVERRIDE_GPU_MIN_FREE_MIB = int(
    MEASURED_NATIVE_CUDA_REQUIRED_MIB + MEASURED_WDDM_UNACCOUNTED_MIB + 1
)
OVERRIDABLE_PREFLIGHT_BLOCKERS = frozenset({
    "RAM_BELOW_FROZEN_FLOOR", "COMMIT_BELOW_FROZEN_FLOOR",
    "GPU_FREE_BELOW_64K_FLOOR",
})


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for blob in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(blob)
    return h.hexdigest()


def _json(path: Path) -> dict:
    def unique(items: list[tuple[str, object]]) -> dict:
        data = {}
        for k, v in items:
            if k in data:
                raise ValueError("duplicate evidence JSON key")
            data[k] = v
        return data

    return json.loads(path.read_bytes(), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def _canonical(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def original_task_census() -> tuple[list[dict], str]:
    """Import ONLY the pinned trusted original Python3.13 official task loader."""
    script = (
        "from automationbench.domains import DEFAULT_DOMAINS,get_combined_dataset;"
        "from automationbench.scripts.eval import _task_name;"
        "import hashlib,json;"
        "d=get_combined_dataset(DEFAULT_DOMAINS);"
        "rows=[(str(x['example_id']),str(x['prompt'])) for x in d];"
        "digest=hashlib.sha256(json.dumps(rows,ensure_ascii=False,separators=(',',':')).encode()).hexdigest();"
        "print('REAP_CANONICAL_TASKS='+json.dumps({'sha':digest,'items':["
        "{'id':str(x['example_id']),'name':_task_name(x)} for x in d]},separators=(',',':')))"
    )
    env = {k: v for k, v in os.environ.items()
           if k.upper() in ("PATH", "SYSTEMROOT", "TEMP", "TMP", "APPDATA",
                            "LOCALAPPDATA", "USERPROFILE", "HOMEDRIVE",
                            "HOMEPATH", "HF_HOME", "HF_DATASETS_CACHE")}
    env.update({"HF_HUB_OFFLINE": "1", "PYTHON_DOTENV_DISABLED": "1",
                "PYTHONNOUSERSITE": "1"})
    outcome = subprocess.run([str(q4.BENCH_PYTHON), "-I", "-c", script],
                             cwd=q4.BENCH_ROOT, env=env,
                             capture_output=True, text=True, timeout=120, check=True)
    lines = [row.removeprefix("REAP_CANONICAL_TASKS=") for row in
             outcome.stdout.splitlines() if row.startswith("REAP_CANONICAL_TASKS=")]
    if len(lines) != 1:
        raise ValueError("original 600 task source loader did not emit one identity census")
    data = json.loads(lines[0])
    if (data.get("sha") != source_ids.PUBLIC_PROMPTS_SHA256
            or not isinstance(data.get("items"), list) or len(data["items"]) != POPULATION
            or any(type(t.get("id")) is not str or type(t.get("name")) is not str
                   for t in data["items"]) or
            len({t["name"] for t in data["items"]}) != POPULATION
            or len({t["id"] for t in data["items"]}) != 570):
        raise ValueError("original 600 official task identities no longer match frozen source")
    return data["items"], data["sha"]


def exact_owner(tasks: list[dict], source_sha: str, model_sha: str,
                official_commit: str) -> dict:
    return {
        "schema": SCHEMA, "status": "ORIGINAL_Q4_PUBLIC600_IN_PROGRESS",
        "model_gguf": str(q4.GGUF), "model_sha256": model_sha,
        "official_benchmark_commit": official_commit,
        "official_version": q4.BENCH_VERSION,
        "original_task_prompt_sha256": source_sha,
        "original_task_population": POPULATION, "tasks": tasks,
        "native_flags": q4.FLAGS, "context_guard_policy": q4.POLICY_NAME,
        "official_evaluator": "automationbench.scripts.eval",
        "official_toolset": "api", "official_scoring": "original binary WorldState verifier",
        "segmented_reproduction_not_private_leaderboard": True,
        "frozen_scientific_gate_eligible": False, "gen0_status": "BLOCKED",
        "chosen": None, "release_approved": False,
    }


def verify_item(itemdir: Path, index: int, expected: dict, owner_hash: str,
                served_id: str) -> dict | None:
    receipt_path, raw_path = itemdir / "receipt.json", itemdir / "public-automationbench.private.json"
    if not receipt_path.exists():
        if itemdir.exists():
            # An aborted item is *not* resumable as a successfully graded original.
            raise ValueError(f"original item {index} left incomplete evidence; manual review required")
        return None
    if any(p.is_symlink() for p in (itemdir, receipt_path, raw_path)):
        raise ValueError("foreign/symlinked scored item evidence")
    saved = _json(receipt_path)
    if (saved.get("schema") != SCHEMA or saved.get("owner_sha256") != owner_hash
            or saved.get("original_index") != index
            or saved.get("task") != expected or saved.get("served_id") != served_id
            or saved.get("raw_sha256") != _digest(raw_path)
            or saved.get("status") != "ORIGINAL_OFFICIAL_TASK_SCORED"):
        raise ValueError("source-bound per-item checkpoint drift")
    recomputed = q4.audit_official(raw_path.read_bytes(), served_id=served_id,
                                  count=1, max_steps=q4.MAX_STEPS,
                                  expected=(expected["name"],))
    if (recomputed != saved.get("independent_score")
            or recomputed["completed_tasks"] != 1):
        raise ValueError("saved score is not independently reproducible")
    return saved


def quarantine_unfinished_item(itemdir: Path) -> Path:
    """Preserve incomplete ORIGINAL private output without ever scoring it.

    A killed/unhealthy evaluator may have produced a partial official JSON/log.
    Save those exact bytes under an immutable retry-specific sibling, then
    start the original official task from scratch in a new exclusive folder.
    Never use this path when any purported scored receipt exists.
    """
    if not itemdir.is_dir() or itemdir.is_symlink() or (itemdir / "receipt.json").exists():
        raise ValueError("not an unscored owned incomplete task")
    children = list(itemdir.iterdir())
    allowed = {"public-automationbench.private.json", "official-evaluator.private.log"}
    if (not children or any(p.is_symlink() or not p.is_file() or p.name not in allowed
                            for p in children)):
        raise ValueError("foreign files in an unfinished task; no automatic retry")
    evidence = [(p.name, _digest(p)) for p in sorted(children)]
    fingerprint = hashlib.sha256(_canonical({"incomplete_files": evidence})).hexdigest()[:16]
    sibling = itemdir.with_name(itemdir.name + ".unfinished-" + fingerprint)
    if sibling.exists() or sibling.is_symlink():
        raise ValueError("duplicate unscored retry archive exists")
    itemdir.rename(sibling)
    return sibling


def _safe_resource(out: Path) -> dict:
    state = q4.memory_snapshot()
    state["disk_free_gib"] = shutil.disk_usage(out).free / (1024 ** 3)
    if (state["ram_available_gib"] < ORIGINAL_RAM_RUNTIME_FLOOR_GIB
            or state["commit_available_gib"] < ORIGINAL_COMMIT_RUNTIME_FLOOR_GIB
            or state["disk_free_gib"] < ORIGINAL_DISK_RUNTIME_FLOOR_GIB):
        raise RuntimeError("original running host RAM/commit/disk safety floor breached")
    return state


def _native_prestart(out: Path, *, bypass_conservative: bool = False) -> dict:
    """Explicitly opt out of conservative floor, not physical allocator fit.

    This applies solely to the noncomparable experimental tool-evaluator arm,
    never original frozen 4-slot release Gate0/Gate1/Gate2. At ~200MiB free
    VRAM while an instance is running, *both* modes categorically refuse a
    second model load. This is intentional and not an arbitrary policy limit.
    """
    state = _safe_resource(out)
    state["gpu_free_mib"] = q4.gpu_free_mib()
    if bypass_conservative:
        if (state["gpu_free_mib"] < OVERRIDE_GPU_MIN_FREE_MIB
                or state["ram_available_gib"] < ORIGINAL_RAM_RUNTIME_FLOOR_GIB
                or state["commit_available_gib"] < ORIGINAL_COMMIT_RUNTIME_FLOOR_GIB):
            raise RuntimeError("physical native allocator fit or running OS reserve unavailable")
        state["prestart_admission"] = "EXPLICIT_CONSERVATIVE_FLOOR_BYPASS_NONCOMPARABLE"
        state["measured_required_native_gpu_mib"] = OVERRIDE_GPU_MIN_FREE_MIB
    elif (state["ram_available_gib"] < q4.MIN_RAM_GIB
          or state["commit_available_gib"] < q4.MIN_COMMIT_GIB
          or state["gpu_free_mib"] < q4.MIN_GPU_MIB + q4.GPU_MARGIN_MIB):
        raise RuntimeError("original full600 resumed native load below 72k model floors")
    return state


def _override_read_only_gate(gate: dict, *, bypass_conservative: bool) -> dict:
    """Keep all provenance/context/port blockers even with resource opt-out."""
    if not bypass_conservative:
        return gate
    other = [b for b in gate["blockers"] if b not in OVERRIDABLE_PREFLIGHT_BLOCKERS]
    measured = gate["measured"]
    allowed = (measured["ram_available_gib"] >= ORIGINAL_RAM_RUNTIME_FLOOR_GIB
               and measured["commit_available_gib"] >= ORIGINAL_COMMIT_RUNTIME_FLOOR_GIB
               and measured["disk_free_gib"] >= ORIGINAL_DISK_RUNTIME_FLOOR_GIB
               and measured["gpu_free_mib"] >= OVERRIDE_GPU_MIN_FREE_MIB)
    if not allowed:
        other.append("PHYSICAL_ALLOCATION_OR_RUNNING_OS_RESERVE_UNAVAILABLE")
    return {**gate, "blockers": other,
            "status": "READY_TO_ATTEMPT_NONCOMPARABLE_EXPLORATORY"
            if not other else "REFUSED",
            "explicit_conservative_resource_bypass": True,
            "resource_bypass_only_not_security_or_scientific_bypass": True,
            "gpu_native_measured_allocator_fit_floor_mib": OVERRIDE_GPU_MIN_FREE_MIB,
            "ignored_only_conservative_resource_blockers": sorted(
                set(gate["blockers"]) & OVERRIDABLE_PREFLIGHT_BLOCKERS),
            "frozen_gen0_status": "BLOCKED",
            "release_approved": False}


def run(out: Path, *, execute: bool, max_new: int = 600,
        bypass_conservative: bool = False) -> dict:
    if not isinstance(max_new, int) or isinstance(max_new, bool) or not 1 <= max_new <= 600:
        raise ValueError("max_new must be within full original population")
    if not execute:
        return {"status": "READ_ONLY_ORIGINAL_600_PREFLIGHT",
                "decision": _override_read_only_gate(
                    q4.preflight(out), bypass_conservative=bypass_conservative),
                "model_executed": False}
    source = q4.verify_official_source()
    tasks, source_sha = original_task_census()
    # Registered exact original model hash, checked before *any* model inference.
    model = q4.verify_gguf(q4.GGUF, shown_bytes=q4.GGUF_BYTES, shown_sha=q4.GGUF_SHA)
    owner = exact_owner(tasks, source_sha, model["gguf_sha256"],
                        source["source_git_commit"])
    owner_hash = hashlib.sha256(_canonical(owner)).hexdigest()
    if out.exists():
        if out.is_symlink() or _json(out / "owner.json") != owner:
            raise ValueError("cannot resume another source/model/serving owner")
    else:
        admission = _override_read_only_gate(
            q4.preflight(out), bypass_conservative=bypass_conservative)
        if admission["status"] != "READY_TO_ATTEMPT_NONCOMPARABLE_EXPLORATORY":
            return {"status": "REFUSED_NATIVE_RESOURCE_ADMISSION", "admission": admission}
        out.mkdir(mode=0o700)
        with (out / "owner.json").open("xb") as f:
            f.write(_canonical(owner))
    existing = sorted(p.name for p in out.iterdir() if p.is_dir())
    for name in existing:
        prior = name.split(".unfinished-", maxsplit=1)
        if (not prior[0].startswith("task-") or not prior[0][5:].isdigit()
                or (len(prior) == 2 and (
                    len(prior[1]) != 16 or
                    any(ch not in "0123456789abcdef" for ch in prior[1])))):
            raise ValueError("unowned directory in private original benchmark")
    if not q4.port_is_free(q4.PORT) or not q4.port_is_free(q4.GUARD_PORT):
        raise ValueError("original native Q4/guard ports already in use")
    prestart = _native_prestart(out, bypass_conservative=bypass_conservative)
    if bypass_conservative:
        admission_path = out / f"explicit-conservative-resource-bypass-{int(time.time())}.private.json"
        with admission_path.open("x", encoding="utf-8") as admission_log:
            json.dump({
                "schema": SCHEMA, "mode": "NONCOMPARABLE_EXPLORATORY_ONLY",
                "explicit_bypass_of_only_conservative_prestart_thresholds": True,
                "native_measurement": prestart, "model_sha256": q4.GGUF_SHA,
                "owner_sha256": owner_hash, "frozen_gen0_status": "BLOCKED",
                "release_approved": False,
            }, admission_log, sort_keys=True)
            admission_log.write("\n")
    native = None
    server = None
    thread = None
    started = time.monotonic()
    created = 0
    try:
        with (out / f"native-session-{int(time.time())}.private.log").open("xb") as log:
            native = subprocess.Popen([str(q4.SERVER), *q4.FLAGS, "--host",
                "127.0.0.1", "--port", str(q4.PORT), "-m", str(q4.GGUF)],
                cwd=out, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, shell=False,
                env={**os.environ, "LLAMA_ARG_LOG_VERBOSITY": "4"},
                creationflags=subprocess.CREATE_NO_WINDOW)
            q4.wait_health(native, started=started, deadline=q4.MAX_START_SECONDS)
        server_log = max(out.glob("native-session-*.private.log"), key=lambda f:f.stat().st_mtime)
        evidence = server_log.read_text(encoding="utf8", errors="replace")
        q4.verify_cuda(evidence)
        q4.verify_geometry(evidence, context=q4.CONTEXT)
        served_id = q4.serve_single_model(native)
        # Bind resume to native served id and same original GGUF.
        if served_id != str(q4.GGUF):
            raise ValueError("native serving identity is not original GGUF")
        server, guard_state, thread = start_guard(
            upstream_port=q4.PORT, listen_port=q4.GUARD_PORT,
            context=q4.CONTEXT, receipt_path=out /
            f"guard-session-{int(time.time())}.private.jsonl")
        for idx, task in enumerate(tasks):
            itemdir = out / f"task-{idx:03}"
            if itemdir.is_dir() and not (itemdir / "receipt.json").exists():
                archived = quarantine_unfinished_item(itemdir)
                print(json.dumps({"event": "ORIGINAL_UNSCORED_ATTEMPT_PRESERVED",
                                  "original_index": idx,
                                  "archive": str(archived)}), flush=True)
            found = verify_item(itemdir, idx, task, owner_hash, served_id)
            if found is not None:
                continue
            if created >= max_new or time.monotonic() - started >= MAX_RUN_SECONDS:
                break
            itemdir.mkdir(mode=0o700, exist_ok=False)
            command = q4.bench_command(served_id, itemdir, count=1, tasks=None,
                                       base_port=q4.GUARD_PORT)
            command.extend(["--skip", str(idx)])
            env = {k:v for k,v in os.environ.items() if k.upper() in
                   ("PATH","SYSTEMROOT","TEMP","TMP","APPDATA","LOCALAPPDATA","USERPROFILE")}
            env.update({"OPENAI_API_KEY":"local-original-only",
                        "HF_HUB_OFFLINE":"1", "PYTHON_DOTENV_DISABLED":"1",
                        "PYTHONNOUSERSITE":"1"})
            command_log = itemdir / "official-evaluator.private.log"
            with command_log.open("xb") as log:
                item_start = time.monotonic()
                child = subprocess.Popen(command, cwd=q4.BENCH_ROOT, env=env,
                                         stdin=subprocess.DEVNULL, stdout=log,
                                         stderr=subprocess.STDOUT, shell=False,
                                         creationflags=subprocess.CREATE_NO_WINDOW)
                try:
                    while child.poll() is None:
                        _safe_resource(out)
                        if (native.poll() is not None or
                                time.monotonic() - started >= MAX_RUN_SECONDS or
                                time.monotonic() - item_start > MAX_ITEM_SECONDS):
                            raise TimeoutError("original owned native/item deadline exceeded")
                        time.sleep(3)
                    if child.returncode:
                        raise RuntimeError("original per-item official evaluator aborted")
                finally:
                    q4.kill_tree(child)
            raw_file = itemdir / "public-automationbench.private.json"
            score = q4.audit_official(raw_file.read_bytes(), served_id=served_id,
                                     count=1, max_steps=q4.MAX_STEPS,
                                     expected=(task["name"],))
            if score["completed_tasks"] != 1:
                raise RuntimeError("aborted original task is not scored or checkpointed")
            record = {"schema": SCHEMA, "owner_sha256": owner_hash,
                      "original_index": idx, "task": task,
                      "served_id": served_id, "status": "ORIGINAL_OFFICIAL_TASK_SCORED",
                      "raw_sha256": _digest(raw_file),
                      "independent_score": score}
            with (itemdir / "receipt.json").open("xb") as stream:
                stream.write(_canonical(record))
            created += 1
            print(json.dumps({"original_evaluated_index":idx,
                              "completed_score":score["passed_tasks"],
                              "checkpoint_sha256":_digest(itemdir/"receipt.json"),
                              "population": POPULATION}), flush=True)
    finally:
        if server is not None and thread is not None:
            stop_guard(server, thread)
        if native is not None:
            q4.kill_tree(native)
    verified = []
    for idx, item in enumerate(tasks):
        found = verify_item(out / f"task-{idx:03}", idx, item, owner_hash, served_id)
        if found is not None:
            verified.append(found)
    return {
        "status": "ORIGINAL_PUBLIC600_COMPLETED_REPRODUCTION" if len(verified)==POPULATION
                  else "ORIGINAL_PUBLIC600_PARTIAL_REAL_EXECUTION_RESUMABLE",
        "original_items_completed":len(verified), "original_population":POPULATION,
        "new_completed_this_session":created,
        "actual_passes":sum(v["independent_score"]["passed_tasks"] for v in verified),
        "source_prompt_sha256":source_sha, "model_sha256":q4.GGUF_SHA,
        "official_benchmark_version":q4.BENCH_VERSION,
        "original_frozen_gen0_serving_comparable":False,
        "gen0_status":"BLOCKED","chosen":None,"release_approved":False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-new", type=int, default=600)
    parser.add_argument("--bypass-conservative-resource-admission", action="store_true",
                        help="Experimental Q4 native source only; retains real allocator "
                             "fit, running OS reserves, provenance, and all original gates.")
    args = parser.parse_args()
    try:
        result = run(args.out, execute=args.execute, max_new=args.max_new,
                     bypass_conservative=args.bypass_conservative_resource_admission)
        print(json.dumps(result, sort_keys=True, indent=2), flush=True)
        return 0 if result["status"] != "REFUSED_NATIVE_RESOURCE_ADMISSION" else 2
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError,
            TimeoutError, KeyError) as exc:
        print(json.dumps({"status":"REFUSED_OR_INTERRUPTED_NO_NEW_SCORE",
                          "reason":str(exc)[:400], "gen0_status":"BLOCKED",
                          "release_approved":False}), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
