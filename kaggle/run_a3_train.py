"""Kaggle kernel script: train the A3 teacher-free recipe on 2xT4 (fp16, DDP).

Pushed by `python -m chowder.kaggle_dispatch train`, which renders the JOB
block below (pinned commit, data mount, optional resume source). Stages are
recorded to /kaggle/working/a3_job.json even when a later stage fails:

1. hardware -- two T4s visible, bf16 unsupported (why the recipe is fp16).
2. install  -- clone Chowder at the pinned commit, editable install.
3. data     -- the private dataset mount holds train.jsonl/dev.jsonl; the
               recipe's pinned sha256 digests are enforced by train_pilot.
4. resume   -- newest checkpoint-N from an attached previous-run output.
5. train    -- train_pilot --devices 0,1 --save-steps, log to train.log.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

JOB = json.loads(r'''__CHOWDER_JOB__''')

WORK = Path("/kaggle/working")
RECORD = WORK / "a3_job.json"
record: dict = {"job": JOB, "stages": {}}


def save() -> None:
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
        save()
        raise
    finally:
        save()


def hardware() -> dict:
    import torch

    gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if len(gpus) < 2:
        raise RuntimeError(f"expected 2 GPUs, found {gpus}")
    return {"gpus": gpus, "bf16_supported": torch.cuda.is_bf16_supported(), "torch": torch.__version__}


def install() -> dict:
    # Outside /kaggle/working: everything there is exported as kernel output,
    # and a repo clone made proof-1's output pull time out.
    repo = Path("/tmp/Chowder")
    subprocess.run(["git", "clone", "-q", JOB["repo_url"], str(repo)], check=True)
    # The image's torchao 0.10 makes transformers/peft>=5.12 raise at import
    # (they require >=0.16 if present). A3 uses no torchao; removing it is
    # safer than installing a torchao built for a different torch.
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-q", "-y", "torchao"], check=False)
    subprocess.run(["git", "-C", str(repo), "checkout", "-q", JOB["commit"]], check=True)
    # Chowder's [train] pins minus torch/torchao: keep the image's CUDA torch,
    # but the worker needs transformers>=5.12 (the image ships 5.0).
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(repo), "transformers>=5.12,<6",
                    "peft>=0.20,<0.21", "datasets>=4,<5", "accelerate>=1,<2"], check=True)
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != JOB["commit"]:
        raise RuntimeError(f"checked out {head}, expected {JOB['commit']}")
    return {"repo": str(repo), "commit": head}


def _mounted(pattern: str, mount: str) -> list[Path]:
    # Mount layout differs across Kaggle images (/kaggle/input/<slug> vs nested
    # owner paths), so search by name and keep hits under the requested mount.
    return sorted(p for p in Path("/kaggle/input").rglob(pattern) if mount in p.parts)


def data() -> dict:
    hits = _mounted("train.jsonl", JOB["data_mount"])
    if not hits:
        raise FileNotFoundError(f"no train.jsonl under a /kaggle/input/**/{JOB['data_mount']} mount")
    data_dir = hits[0].parent
    if not (data_dir / "dev.jsonl").is_file():
        raise FileNotFoundError(f"no dev.jsonl beside {hits[0]}")
    return {"data_dir": str(data_dir)}


def resume() -> dict:
    if not JOB.get("resume_mount"):
        return {"resume_from": None}
    ckpts = sorted((p for p in _mounted("checkpoint-*", JOB["resume_mount"]) if p.is_dir()),
                   key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1)
    if not ckpts:
        raise FileNotFoundError(f"resume requested but no checkpoint-* under {JOB['resume_mount']}")
    return {"resume_from": str(ckpts[-1])}


def train(repo: str, data_dir: str, resume_from: str | None) -> dict:
    cmd = [sys.executable, f"{repo}/experiments/teacher_free_distill/train_pilot.py",
           "--recipe", f"{repo}/{JOB['recipe']}", "--data-dir", data_dir,
           "--output", str(WORK / "cond_a3_kaggle"), "--devices", "0,1",
           "--save-steps", str(JOB["save_steps"]), "--yes"]
    if resume_from:
        cmd += ["--resume-from-checkpoint", resume_from]
    if JOB.get("max_steps"):
        cmd += ["--max-steps", str(JOB["max_steps"])]
    env = {"PYTHONPATH": f"{repo}/src", "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    import os
    import shutil

    with open(WORK / "train.log", "w", encoding="utf-8") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **env}).returncode
    # Checkpoints land under a hidden .chowder/ run dir; copy them somewhere a
    # later kernel is sure to see when this run is attached as its input.
    kept = []
    for ck in (WORK / "cond_a3_kaggle").rglob("checkpoint-*"):
        if ck.is_dir():
            shutil.copytree(ck, WORK / "checkpoints" / ck.name, dirs_exist_ok=True)
            kept.append(ck.name)
    if rc != 0:
        raise RuntimeError(f"train_pilot exited {rc}; see train.log (checkpoints kept: {kept})")
    return {"rc": rc, "command": cmd, "checkpoints": sorted(kept)}


if __name__ == "__main__":
    stage("hardware", hardware)
    inst = stage("install", install)
    d = stage("data", data)
    r = stage("resume", resume)
    stage("train", lambda: train(inst["repo"], d["data_dir"], r["resume_from"]))
