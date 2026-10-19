"""Build the private Kaggle dataset for the Ornith kernel: hermes/, ornith_phase.py, run_baseline.py.

    python kaggle/ornith_curriculum/make_dataset.py --owner <kaggle-user> \
        --hermes-dir <repo>/hermes --ornith-phase <path>/ornith_phase.py --out <dir>

Upload with ``kaggle datasets create -p <dir> --dir-mode zip``: without --dir-mode the
CLI silently skips the hermes/ subdirectory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN_BASELINE = HERE.parent / "reap_baseline" / "run_baseline.py"
DATASET_SLUG = "ornith-hermes"


def build(owner: str, hermes_dir: Path, ornith_phase: Path, out: Path) -> dict:
    if (out / "hermes").exists():
        shutil.rmtree(out / "hermes")
    (out / "hermes").mkdir(parents=True)
    for src in sorted(hermes_dir.glob("*.py")):
        shutil.copy2(src, out / "hermes" / src.name)
    shutil.copy2(ornith_phase, out / "ornith_phase.py")
    shutil.copy2(RUN_BASELINE, out / "run_baseline.py")
    commit = subprocess.run(["git", "-C", str(hermes_dir.parent), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    files = {p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(out.rglob("*.py"))}
    manifest = {"hermes_commit": commit, "files": files}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (out / "dataset-metadata.json").write_text(json.dumps(
        {"id": f"{owner}/{DATASET_SLUG}", "title": "ornith-hermes", "licenses": [{"name": "other"}]},
        indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--owner", required=True)
    p.add_argument("--hermes-dir", required=True)
    p.add_argument("--ornith-phase", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    m = build(a.owner, Path(a.hermes_dir), Path(a.ornith_phase), Path(a.out))
    for name, digest in m["files"].items():
        print(f"{digest}  {name}")
    print(f"hermes {m['hermes_commit']} -> {a.owner}/{DATASET_SLUG} (private)")


if __name__ == "__main__":
    main()
