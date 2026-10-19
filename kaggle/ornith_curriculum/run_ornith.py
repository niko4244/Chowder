"""Kaggle script kernel: one Ornith curriculum round (propose -> cross-check -> admit).

Reproduces C:\\llm-work\\ornith\\run-ornith-run3.ps1 on a 2x T4 kernel: Ornith-1.5-9B
proposes, Qwythos-27B cross-checks only code-valid proposals, hermes admits with the
difficulty gate skipped. One model is served at a time. The llama.cpp build, pinned
download and sha256 check are the tested helpers from kaggle/reap_baseline/run_baseline.py,
shipped in the attached dataset next to hermes and ornith_phase.py.

Outputs (under /kaggle/working): run.log, provenance.json, curriculum/{proposals,
candidates,admitted,rejected}.jsonl and curriculum/manifest.json.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
import urllib.request
from pathlib import Path

DATASET = Path("/kaggle/input/ornith-hermes")
sys.path.insert(0, str(DATASET))

ROUNDS = 96
N_TASKS = 4
SEED = 20261119
MAX_TOKENS = 6144
PROPOSE_BUDGET_S = 6 * 3600
HEALTH_TIMEOUT_S = 15 * 60

MODELS_JSON = """__MODELS__"""  # make_kernel.py inlines models.json here
PROPOSER_ARGS = ["--alias", "ornith-1.5-9b", "--jinja", "--ctx-size", "12288", "--n-gpu-layers", "999",
                 "--flash-attn", "on"]
CROSSCHECK_ARGS = ["--alias", "qwythos-27b", "--jinja", "--ctx-size", "8192", "--fit", "on", "--flash-attn", "on",
                   "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"]
WORK = Path("/kaggle/working")
CUR = WORK / "curriculum"
FACTS: dict = {"rounds_requested": ROUNDS}


def download(model: dict, scratch: Path, rb) -> Path:
    target = scratch / model["file"]
    stamp = target.with_suffix(target.suffix + ".sha256")
    if target.exists() and stamp.exists() and stamp.read_text().strip() == model["sha256"]:
        return target
    from huggingface_hub import hf_hub_download

    rb.log(f"downloading {model['hf_repo']}@{model['revision']} :: {model['file']}")
    path = Path(hf_hub_download(repo_id=model["hf_repo"], filename=model["file"], revision=model["revision"],
                                local_dir=str(scratch)))
    stamp.write_text(rb.verify_sha256(path, model["sha256"]) + "\n")
    FACTS[f"{model['alias']}_sha256"] = model["sha256"]
    rb.log(f"sha256 verified for {model['file']}")
    return path


def serve(server: Path, gguf: Path, port: int, args: list[str], rb) -> subprocess.Popen:
    log = open(WORK / f"server-{port}.log", "w")
    proc = subprocess.Popen([str(server), "-m", str(gguf), "--host", "127.0.0.1", "--port", str(port), *args],
                            stdout=log, stderr=subprocess.STDOUT)
    deadline = time.time() + HEALTH_TIMEOUT_S
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server on {port} exited {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200:
                    rb.log(f"server ready on {port}")
                    return proc
        except OSError:
            pass
        time.sleep(5)
    proc.kill()
    raise RuntimeError(f"llama-server on {port} never became healthy")


def stop(proc: subprocess.Popen | None) -> None:
    if proc and proc.poll() is None:
        proc.kill()
        proc.wait(timeout=60)


def run_budgeted(cmd: list[str], budget_s: float, out: Path, env: dict | None = None) -> bool:
    """Run ``cmd``; on timeout kill it and carry on. Fail if ``out`` is missing or empty.

    Returns True when the command finished within the budget.
    """
    try:
        done = subprocess.run(cmd, timeout=budget_s, env=env)
        finished = True
        if done.returncode != 0:
            raise RuntimeError(f"{cmd[1:3]} exited {done.returncode}")
    except subprocess.TimeoutExpired:
        finished = False
    if not out.exists() or not out.read_text(encoding="utf-8").strip():
        raise RuntimeError(f"{out.name} is missing or empty")
    return finished


def main() -> int:
    import run_baseline as rb

    started = dt.datetime.now(dt.timezone.utc).isoformat()
    status, error, proc = "ok", None, None
    env = dict(os.environ, PYTHONPATH=str(DATASET))
    py = sys.executable
    phase = str(DATASET / "ornith_phase.py")
    try:
        CUR.mkdir(parents=True, exist_ok=True)
        scratch = rb.scratch_dir()
        server = rb.build_llama_server(scratch)
        models = json.loads(MODELS_JSON)
        proposer, checker = (download(m, scratch, rb) for m in models)

        proc = serve(server, proposer, 8108, PROPOSER_ARGS, rb)
        FACTS["propose_finished_in_budget"] = run_budgeted(
            [py, phase, "propose", "--endpoint", "http://127.0.0.1:8108/v1/chat/completions", "--rounds", str(ROUNDS),
             "--n-tasks", str(N_TASKS), "--seed", str(SEED), "--max-tokens", str(MAX_TOKENS),
             "--out", str(CUR / "proposals.jsonl")], PROPOSE_BUDGET_S, CUR / "proposals.jsonl", env)
        stop(proc)

        proc = serve(server, checker, 8110, CROSSCHECK_ARGS, rb)
        rb.run("crosscheck", [py, phase, "crosscheck", "--endpoint", "http://127.0.0.1:8110/v1/chat/completions",
                              "--proposals", str(CUR / "proposals.jsonl"), "--out", str(CUR / "candidates.jsonl"),
                              "--only-code-valid"], env=env)
        stop(proc)
        proc = None

        rb.run("admit", [py, "-m", "hermes", "admit", "--candidates", str(CUR / "candidates.jsonl"),
                         "--out", str(CUR / "admitted.jsonl"), "--rejection-log", str(CUR / "rejected.jsonl"),
                         "--skip-difficulty"], env=env, cwd=str(DATASET))
        admitted = CUR / "admitted.jsonl"
        lines = lambda p: [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []
        manifest = {
            "admitted": "admitted.jsonl",
            "sha256": hashlib.sha256(admitted.read_bytes()).hexdigest() if admitted.exists() else "",
            "hermes_commit": json.loads((DATASET / "manifest.json").read_text())["hermes_commit"],
            "difficulty": "skipped: llama-server gives no prompt logprobs",
            "rounds_requested": ROUNDS,
            "propose_finished_in_budget": FACTS["propose_finished_in_budget"],
            "proposed": len(lines(CUR / "proposals.jsonl")),
            "cross_checked": len(lines(CUR / "candidates.jsonl")),
            "admitted_count": len(lines(admitted)),
        }
        (CUR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        rb.log(f"DONE admitted {manifest['admitted_count']} sha256={manifest['sha256']}")
    except Exception as exc:
        status, error = "failed", f"{type(exc).__name__}: {exc}"
        rb.log(f"FAILED: {error}")
        traceback.print_exc()
    finally:
        stop(proc)
    (WORK / "provenance.json").write_text(json.dumps({
        "models": MODELS_JSON, "facts": FACTS, "llama_cpp_commit": rb.FACTS.get("llama_cpp_commit"),
        "gpus": rb.gpu_names(), "exit_codes": rb.EXIT_CODES, "status": status, "error": error,
        "started_utc": started, "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat()}, indent=2) + "\n")
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
