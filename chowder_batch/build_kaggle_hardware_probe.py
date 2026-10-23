"""Build an independently executable Kaggle CPU/GPU/TPU hardware-only probe.

This builder does not authenticate, upload, start sessions, consume quota,
download weights, or alter the frozen experimental design. It creates a new
notebook and metadata from known code under one account supplied by the owner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SOURCE = Path(__file__).with_name("kaggle_hardware_probe.py")
ACCELERATORS = {
    "cpu": {"enable_gpu": False, "enable_tpu": False, "machine_shape": "None"},
    "gpu": {"enable_gpu": True, "enable_tpu": False, "machine_shape": "NvidiaTeslaT4"},
    "tpu": {"enable_gpu": False, "enable_tpu": True, "machine_shape": "TpuV5E8"},
}


def generate(*, dest: Path, kernel_id: str, accelerator: str) -> dict[str, Any]:
    if accelerator not in ACCELERATORS:
        raise ValueError("unsupported accelerator")
    if (len(kernel_id.split("/")) != 2 or
            any(not part or not all(c.isalnum() or c in "-_" for c in part)
                for part in kernel_id.split("/"))):
        raise ValueError("kernel id must be username/new-private-notebook-slug")
    if dest.exists():
        raise FileExistsError("probe staging destination already exists; refusing overwrite")
    source = SOURCE.read_text(encoding="utf-8")
    compile(source, str(SOURCE), "exec")
    code = (source + "\n\n"
            "if __name__ == '__main__':\n"
            "    raise SystemExit(main(["
            + repr("--accelerator") + ", " + repr(accelerator) + ", "
            + repr("--out") + ", "
            + repr("/kaggle/working/chowder-hardware-receipt.json") + "]))\n")
    # The source's own __main__ also calls main(); strip only the guard in the
    # generated notebook so the probe runs exactly once with fixed arguments.
    marker = '\nif __name__ == "__main__":\n    raise SystemExit(main())\n'
    if source.count(marker) != 1:
        raise ValueError("probe source entrypoint must have exactly one known guard")
    code = source.replace(marker, "\n") + code[len(source):]
    compile(code, "kaggle_hardware_notebook.py", "exec")
    notebook: dict[str, Any] = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [
                "# Chowder Kaggle hardware admission\n",
                "This hardware-only check does not load weights, train, select a model, "
                "or claim frozen BF16 equivalence.\n",
            ]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
             "source": code.splitlines(keepends=True)},
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                    "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    metadata = {
        "id": kernel_id,
        # Kaggle currently derives the canonical notebook slug from title
        # even when id is provided; keep them identical to avoid an accidental
        # second notebook with a different slug on remote push.
        "title": kernel_id.split("/")[1],
        "code_file": "chowder-hardware-probe.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_internet": False,
        "dataset_sources": [],
        "kernel_sources": [],
        "model_sources": [],
        "competition_sources": [],
        **ACCELERATORS[accelerator],
    }
    dest.mkdir(parents=True, exist_ok=False)
    (dest / "chowder-hardware-probe.ipynb").write_text(
        json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
    (dest / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {"staging": str(dest), "kernel_id": kernel_id, "accelerator": accelerator,
            "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "remote_job_started": False,
            "model_loaded": False, "registered_gate_passed": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--kernel-id", required=True)
    parser.add_argument("--accelerator", choices=sorted(ACCELERATORS), default="cpu")
    args = parser.parse_args()
    print(json.dumps(generate(dest=args.out_dir, kernel_id=args.kernel_id,
                              accelerator=args.accelerator), sort_keys=True))


if __name__ == "__main__":
    main()
