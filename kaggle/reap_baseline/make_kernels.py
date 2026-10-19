"""Write the per-arm Kaggle kernel upload directories (script kernels).

    python kaggle/reap_baseline/make_kernels.py --owner <kaggle-user> --out <dir>

For every arm in ``baselines.json`` this writes ``<dir>/<arm>/run_baseline.py``
(the kernel script, with ``ARM`` and ``BASELINE`` rewritten from the arm table)
and ``<dir>/<arm>/kernel-metadata.json`` (private, GPU, internet, and the
private eval-code dataset mounted read-only).

It writes files and prints what it wrote. It never calls the ``kaggle`` CLI:
uploading is the orchestrator's step (see README.md).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "run_baseline.py"
BASELINES = HERE / "baselines.json"
DATASET_SLUG = "chowder-reap-eval"
ARM_LINE = re.compile(r'^ARM = "[^"]*"$', re.M)
BASELINE_LINE = re.compile(r"^BASELINE = \{.*\}$", re.M)


def load_baselines(path: Path = BASELINES) -> list[dict]:
    entries = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path} must hold a non-empty list of arms")
    keys = {"arm", "hf_repo", "revision", "file", "sha256"}
    for entry in entries:
        if set(entry) != keys:
            raise ValueError(f"{path}: arm {entry.get('arm')!r} must have exactly {sorted(keys)}")
    return entries


def kernel_slug(arm: str) -> str:
    return "reap-baseline-" + arm.replace("_", "-")


def rewrite(source: str, entry: dict) -> str:
    """Set the arm identity in the kernel template; refuse a template it does not recognise."""
    if not ARM_LINE.search(source) or not BASELINE_LINE.search(source):
        raise ValueError(f"{TEMPLATE} has no rewritable ARM/BASELINE line")
    # lambda replacements: the JSON carries no escapes today, and this keeps it that way.
    out = ARM_LINE.sub(lambda _: f'ARM = "{entry["arm"]}"', source, count=1)
    out = BASELINE_LINE.sub(lambda _: "BASELINE = " + json.dumps(entry, sort_keys=True), out, count=1)
    if f'ARM = "{entry["arm"]}"' not in out:
        raise ValueError(f"ARM was not rewritten to {entry['arm']!r}")
    return out


def kernel_metadata(owner: str, arm: str) -> dict:
    return {
        "id": f"{owner}/{kernel_slug(arm)}",
        "title": f"REAP baseline {arm}",
        "code_file": "run_baseline.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [f"{owner}/{DATASET_SLUG}"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--owner", required=True, help="Kaggle username that will own the kernels")
    parser.add_argument("--out", required=True, help="directory to write <arm>/ kernel directories into")
    args = parser.parse_args(argv)

    template = TEMPLATE.read_text(encoding="utf-8")
    out = Path(args.out)
    for entry in load_baselines():
        arm_dir = out / entry["arm"]
        arm_dir.mkdir(parents=True, exist_ok=True)
        (arm_dir / "run_baseline.py").write_text(rewrite(template, entry), encoding="utf-8")
        metadata = kernel_metadata(args.owner, entry["arm"])
        (arm_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"{arm_dir} -> {metadata['id']} (gpu, internet, private, mounts {metadata['dataset_sources'][0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
