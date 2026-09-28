"""Kaggle kernel script: evaluation arms on 2xT4, one worker per GPU, in parallel.

Pushed by `python -m chowder.kaggle_dispatch eval`, which renders the JOB block
(pinned commit, prompts mount, eval protocol, and a list of jobs, each an arm
+ optional adapter kernel mount + row range + GPU). Jobs on the same GPU run in
sequence; the two GPUs run concurrently. Each job runs the production
`chowder.evaluators.transformers_text_worker` on its row range and writes
/kaggle/working/<arm>-<start>-<end>/{predictions-*.jsonl,result.json}; the
adapter actually loaded is sha256-recorded in eval_job.json. Row ranges of one
arm are merged by the operator in row order.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

JOB = json.loads(r'''__CHOWDER_JOB__''')

WORK = Path("/kaggle/working")
RECORD = WORK / "eval_job.json"
record: dict = {"job": JOB, "stages": {}, "jobs": []}
lock = threading.Lock()


def save() -> None:
    with lock:
        RECORD.write_text(json.dumps(record, indent=2), encoding="utf-8")


def stage(name: str, fn):
    started = time.time()
    try:
        out = fn()
        record["stages"][name] = {"ok": True, "seconds": round(time.time() - started, 1), "result": out}
        return out
    except Exception as exc:  # noqa: BLE001 -- every stage leaves evidence
        record["stages"][name] = {"ok": False, "seconds": round(time.time() - started, 1),
                                  "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()}
        raise
    finally:
        save()


def _mount(name: str) -> Path:
    base = Path("/kaggle/input")
    hits = [p for depth in ("", "*/", "*/*/", "*/*/*/") for p in base.glob(f"{depth}{name}") if p.is_dir()]
    if not hits:
        raise FileNotFoundError(f"mount {name!r} not found; /kaggle/input has {sorted(map(str, base.glob('*/*/*')))[:50]}")
    return hits[0]


def install() -> dict:
    repo = Path("/tmp/Chowder")
    subprocess.run(["git", "clone", "-q", JOB["repo_url"], str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "checkout", "-q", JOB["commit"]], check=True)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-q", "-y", "torchao"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(repo), "transformers>=5.12,<6",
                    "peft>=0.20,<0.21", "accelerate>=1,<2", "math-verify==0.9.0"], check=True)
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != JOB["commit"]:
        raise RuntimeError(f"checked out {head}, expected {JOB['commit']}")
    return {"repo": str(repo), "commit": head}


def locate() -> dict:
    prompts = sorted(_mount(JOB["prompts_mount"]).rglob(JOB["prompts_file"]))
    if not prompts:
        raise FileNotFoundError(f"{JOB['prompts_file']} not in mount {JOB['prompts_mount']}")
    adapters = {}
    for job in JOB["jobs"]:
        if job.get("adapter_mount") and job["arm"] not in adapters:
            # The final adapter dir: has weights, and is not a trainer checkpoint.
            cands = [p.parent for p in _mount(job["adapter_mount"]).rglob("adapter_model.safetensors")
                     if not any(part.startswith("checkpoint-") for part in p.parts)]
            if not cands:
                raise FileNotFoundError(f"no final adapter in mount {job['adapter_mount']}")
            chosen = min(cands, key=lambda p: len(p.parts))
            weights = (chosen / "adapter_model.safetensors").read_bytes()
            adapters[job["arm"]] = {"dir": str(chosen), "sha256": hashlib.sha256(weights).hexdigest()}
    return {"prompts": str(prompts[0]), "adapters": adapters}


def run_job(job: dict, prompts: Path, adapters: dict, repo: str) -> None:
    start, end = job["rows"]
    out = WORK / f"{job['arm']}-{start}-{end}"
    out.mkdir(parents=True, exist_ok=True)
    rows = prompts.read_text(encoding="utf-8").splitlines()[start:end]
    (out / "rows.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    spec = {
        "base_model": JOB["base_model"], "revision": JOB["revision"],
        "adapter_dir": adapters[job["arm"]]["dir"] if job["arm"] in adapters else None,
        "output_dir": str(out), "precision": JOB["precision"], "quantization": "none", "device": "auto",
        "seed": JOB["seed"], "offline": False,
        "suites": [{"name": JOB["suite_name"], "dataset": str(out / "rows.jsonl"), "prompt_field": "prompt",
                    "expected_field": "expected", "scoring": JOB["scoring"], "max_new_tokens": JOB["max_new_tokens"],
                    "use_chat_template": True, "batch_size": JOB["batch_size"]}],
    }
    (out / "eval_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": f"{repo}/src", "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
           "CUDA_VISIBLE_DEVICES": str(job["gpu"])}
    t0 = time.time()
    with open(out / "worker_stdout.log", "w", encoding="utf-8") as log:
        rc = subprocess.run([sys.executable, "-m", "chowder.evaluators.transformers_text_worker",
                             "--spec", str(out / "eval_spec.json"), "--result", str(out / "result.json")],
                            stdout=log, stderr=subprocess.STDOUT, env=env).returncode
    entry = {**job, "rc": rc, "seconds": round(time.time() - t0, 1), "output": str(out)}
    with lock:
        record["jobs"].append(entry)
    save()


def run_all(located: dict, repo: str) -> dict:
    prompts = Path(located["prompts"])
    per_gpu: dict[int, list] = {}
    for job in JOB["jobs"]:
        per_gpu.setdefault(int(job["gpu"]), []).append(job)
    threads = [threading.Thread(target=lambda js=js: [run_job(j, prompts, located["adapters"], repo) for j in js])
               for js in per_gpu.values()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    failed = [j for j in record["jobs"] if j["rc"] != 0]
    if failed:
        raise RuntimeError(f"{len(failed)} job(s) failed: {[(j['arm'], j['rows'], j['rc']) for j in failed]}")
    return {"jobs": len(record["jobs"])}


if __name__ == "__main__":
    inst = stage("install", install)
    loc = stage("locate", locate)
    stage("run", lambda: run_all(loc, inst["repo"]))
