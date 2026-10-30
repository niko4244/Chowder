"""Copy frozen historical REAP baselines into a new isolated v2 evaluation root.

The historical root remains immutable. New v2 runs write only the new root,
with original baseline bytes copied and hash-attested, never resumed in place.
No provider, model, GPU, network, or service interaction is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

BASELINE_ARMS = (
    "dense", "dense_q4", "random128", "reap128", "reap128_q4", "reap160", "reap160_q4",
)
COPIED_FILES = (
    "summary.json", "gsm8k.jsonl", "humaneval.jsonl",
    "serve.json", "placement.json", "probe.log",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create(*, historical: Path, intended: Path) -> dict:
    historical = historical.resolve()
    intended = intended.resolve()
    if (not historical.is_dir() or not intended.parent.is_dir()
            or intended.exists() or intended == historical
            or intended.is_relative_to(historical) or historical.is_relative_to(intended)):
        raise ValueError("new evaluation root must be disjoint, new, and have an existing parent")
    from chowder_batch.reap_select import _attested_rows

    for arm, suite, count in (
        ("dense", "gsm8k", 1319), ("dense", "humaneval", 164),
        ("random128", "gsm8k", 1319), ("dense_q4", "gsm8k", 200),
        ("reap128", "gsm8k", 1319), ("reap128", "humaneval", 164),
        ("reap160", "gsm8k", 1319), ("reap160", "humaneval", 164),
    ):
        if _attested_rows(historical, arm, suite, count) is None:
            raise ValueError(f"historical baseline missing certified {arm}/{suite}")
    inventory = {}
    # Stage all new files and atomically publish only the complete copy.
    with tempfile.TemporaryDirectory(
        dir=intended.parent, prefix=f".{intended.name}.building-",
    ) as temp:
        staged = Path(temp)
        for arm in BASELINE_ARMS:
            if not (historical / arm).is_dir():
                continue
            folder = staged / arm
            folder.mkdir()
            for filename in COPIED_FILES:
                original = historical / arm / filename
                if original.is_file():
                    if original.is_symlink() or original.stat().st_size > 16 * 1024 * 1024:
                        raise ValueError(f"unsafe or oversized historical artifact: {original}")
                    copied = folder / filename
                    shutil.copyfile(original, copied)
                    digest = _sha(original)
                    if _sha(copied) != digest:
                        raise ValueError(f"frozen historical copy changed: {original}")
                    inventory[f"{arm}/{filename}"] = digest
        receipt = {
            "schema": "chowder.reap-v2-isolated-eval-workspace.v1",
            "status": "FROZEN_HISTORICAL_BASELINES_COPIED_NOT_EVALUATED",
            "historical_source": str(historical),
            "new_eval_root": str(intended),
            "source_sha256": inventory,
            "v2_evaluation_missing": True,
            "release_approved": False,
            "note": "Missing historical .inputs.json makes historical reruns unsafe; "
                    "new v2 sessions must use their own new input witnesses.",
        }
        (staged / "frozen-historical-baselines.json").write_text(
            json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8",
        )
        if intended.exists():
            raise ValueError("evaluation workspace appeared while copying; refuse publication")
        staged.rename(intended)
    return receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = create(historical=args.historical, intended=args.out)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"frozen evaluation workspace refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": result["status"], "path": result["new_eval_root"],
        "files": len(result["source_sha256"]),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
