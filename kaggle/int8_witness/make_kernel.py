"""Write a Kaggle kernel upload directory for this lane.

    python kaggle/int8_witness/make_kernel.py --owner <kaggle-user> --kind capacity --out <dir>
    python kaggle/int8_witness/make_kernel.py --owner <kaggle-user> --kind step     --out <dir>

``capacity`` writes ``probe.py``: what a session gives the int8 placement. ``step``
writes ``run_step.py``: real steps of that placement on reap160, which also needs
the checkpoint dataset. It copies the script verbatim and writes
``kernel-metadata.json`` (private, GPU, internet, code dataset mounted, and the
checkpoint too for ``step``). It never calls the ``kaggle`` CLI (see README.md).

``enable_gpu`` only asks for *a* GPU; these kernels report which card they got,
because the spec is bf16 and the shapes this account can request are T4/P100.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE_DATASET = "chowder-moe-int8"
MODEL_DATASET = "qwen36-reap160-bf16"

KINDS = {
    "capacity": {
        # The site derives the slug from the title, and a metadata id that does not
        # resolve to it is refused with a 409 on the second push.
        "slug": "chowder-int8-placement-capacity-probe",
        "title": "Chowder int8 placement capacity probe",
        "script": "probe.py",
        "datasets": [CODE_DATASET],
    },
    "step": {
        "slug": "chowder-int8-witness-step",
        "title": "Chowder int8 witness step",
        "script": "run_step.py",
        "datasets": [CODE_DATASET, MODEL_DATASET],
    },
}


def kernel_metadata(owner: str, kind: str) -> dict:
    spec = KINDS[kind]
    return {
        "id": f"{owner}/{spec['slug']}",
        "title": spec["title"],
        "code_file": spec["script"],
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [f"{owner}/{name}" for name in spec["datasets"]],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--owner", required=True, help="Kaggle username that will own the kernel")
    parser.add_argument("--kind", required=True, choices=sorted(KINDS), help="which kernel to write")
    parser.add_argument("--out", required=True, help="directory to write the kernel upload into")
    args = parser.parse_args(argv)

    spec = KINDS[args.kind]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(HERE / spec["script"], out / spec["script"])  # verbatim, never a rewrite
    metadata = kernel_metadata(args.owner, args.kind)
    (out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"{out} -> {metadata['id']} (private, GPU, internet, mounts {', '.join(metadata['dataset_sources'])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
