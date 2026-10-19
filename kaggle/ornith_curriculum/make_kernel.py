"""Write the Ornith curriculum kernel folder: run_ornith.py (models inlined) + kernel-metadata.json.

    python kaggle/ornith_curriculum/make_kernel.py --owner <kaggle-user> --out <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
KERNEL_SLUG = "ornith-curriculum-run3"
DATASET_SLUG = "ornith-hermes"


def build(owner: str, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    models = json.dumps(json.loads((HERE / "models.json").read_text(encoding="utf-8")))
    script = (HERE / "run_ornith.py").read_text(encoding="utf-8")
    if script.count('"""__MODELS__"""') != 1:
        raise SystemExit("run_ornith.py must contain exactly one __MODELS__ placeholder")
    (out / "run_ornith.py").write_text(script.replace('"""__MODELS__"""', repr(models)), encoding="utf-8")
    meta = {
        "id": f"{owner}/{KERNEL_SLUG}",
        "title": "Ornith curriculum run3",
        "code_file": "run_ornith.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [f"{owner}/{DATASET_SLUG}"],
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--owner", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    print(f"{build(a.owner, Path(a.out))} -> {a.owner}/{KERNEL_SLUG} (gpu, internet, private, mounts {a.owner}/{DATASET_SLUG})")


if __name__ == "__main__":
    main()
