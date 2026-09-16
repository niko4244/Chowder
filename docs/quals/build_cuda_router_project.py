"""Build the P11 rung-2 CUDA project from the hash-pinned CPU builder.

Per the committed preregistration, the workload is identical to the CPU pilot
except the device: the corpora and the tiny base (E=4, k=2) come from the
pinned builder (`build_tiny_router_pilot.py`,
SHA-256 bfa954ee...22e6, verified before this run). This script writes a CUDA
project file only; it changes no model bytes, no corpus bytes, and no
recipe field other than the device.
"""

from __future__ import annotations

import json
from pathlib import Path

CPU_RUN = Path(r"C:/Users/nikma/Chowder-Protected/runs/2026-09-13-router-healing-p8-p10")
CUDA_RUN = Path(r"C:/Users/nikma/Chowder-Protected/runs/2026-09-13-router-healing-p11-cuda")

BASE_DIR = CPU_RUN / "tiny-qwen3-moe-e4-k2"
CORPUS = CPU_RUN / "router-corpus.txt"
HOLDOUT = CPU_RUN / "router-holdout.txt"
PROJECT = CUDA_RUN / "router-project-cuda.json"

cpu_project = json.loads((CPU_RUN / "router-project.json").read_text(encoding="utf-8"))
project = json.loads(json.dumps(cpu_project))  # deep copy

project["name"] = "router-healing-tiny-cuda"
project["work_dir"] = str(CUDA_RUN / "work")
project["registry_path"] = str(CUDA_RUN / "work" / "runs.db")
backend = project["config"]["backend"]["router_healing"]
backend["device"] = "cuda"

PROJECT.write_text(json.dumps(project, indent=2) + "\n", encoding="utf-8", newline="\n")
print(
    json.dumps(
        {
            "project": str(PROJECT),
            "base_dir": str(BASE_DIR),
            "device": backend["device"],
            "max_steps": backend["max_steps"],
            "learning_rate": backend["learning_rate"],
            "seq_len": backend["seq_len"],
            "batch_size": backend["batch_size"],
            "seed": backend["seed"],
        },
        indent=2,
    )
)
