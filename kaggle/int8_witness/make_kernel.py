"""Write the Kaggle kernel upload directory for the int8 capacity probe.

    python kaggle/int8_witness/make_kernel.py --owner <kaggle-user> --out <dir>

It writes ``<dir>/probe.py`` (verbatim from this lane) and
``<dir>/kernel-metadata.json``: private, GPU, internet, with the code dataset
mounted. It writes files and prints what it wrote; it never calls the ``kaggle``
CLI (see README.md).

``enable_gpu`` only asks for *a* GPU. The probe reports which card it got and
whether that card supports bf16, because the spec is bf16 and a T4/P100 session
is not the card the pre-registration was written for.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROBE = HERE / "probe.py"
DATASET_SLUG = "chowder-moe-int8"
KERNEL_SLUG = "chowder-int8-placement-capacity"


def kernel_metadata(owner: str) -> dict:
    return {
        "id": f"{owner}/{KERNEL_SLUG}",
        "title": "Chowder int8 placement capacity probe",
        "code_file": "probe.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [f"{owner}/{DATASET_SLUG}"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--owner", required=True, help="Kaggle username that will own the kernel"
    )
    parser.add_argument(
        "--out", required=True, help="directory to write the kernel upload into"
    )
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        PROBE, out / "probe.py"
    )  # verbatim: the kernel runs this file, not a rewrite
    metadata = kernel_metadata(args.owner)
    (out / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"{out} -> {metadata['id']} (private, GPU, internet, mounts {metadata['dataset_sources'][0]})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
