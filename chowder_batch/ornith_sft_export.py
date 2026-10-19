"""Write admitted Ornith curriculum rows as chat SFT data for the custom-MoE track.

One chat row per admitted task, the assistant turn being the *worked* solution --
``reap_calibration.worked_answer`` -- so the text REAP calibrates on when run with
``--ornith-render worked`` is exactly the text the MoE is trained on. Rows are
deduplicated on ``proposal_sha256``: the curriculum may admit the same proposal
twice under different task ids, and SFT should see it once.

Usage:
    python chowder_batch/ornith_sft_export.py --admitted <file> --sha256 <pin> --out <jsonl>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # a batch script, not an installed console entry
    sys.path.insert(0, str(ROOT))

from chowder_batch.reap_calibration import ornith_assistant, ornith_rows  # noqa: E402

SOURCE = "ornith_curriculum"


def sft_rows(admitted: str, sha256: str) -> list[dict]:
    """The export rows, in file order: same filters as calibration, one row per proposal."""
    seen: set[str] = set()
    rows = []
    for row in ornith_rows(admitted, sha256):
        key = row["proposal_sha256"]
        if key:
            if key in seen:
                continue
            seen.add(key)
        # ponytail: a row with no proposal id cannot be checked for duplication, so it is kept.
        # Ceiling: an unidentifiable admitted row repeats as many times as it is listed.
        # Upgrade: dedupe on a hash of the prompt when proposal_sha256 is absent.
        rows.append({
            "messages": [{"role": "user", "content": row["prompt"]},
                         {"role": "assistant", "content": ornith_assistant(row, "worked")}],
            "source": SOURCE,
            "proposal_sha256": key,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--admitted", required=True, help="a frozen Ornith admitted.jsonl")
    parser.add_argument("--sha256", required=True, help="the pinned sha256 of --admitted")
    parser.add_argument("--out", required=True, help="the SFT jsonl to write")
    args = parser.parse_args()

    rows = sft_rows(args.admitted, args.sha256)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows")


if __name__ == "__main__":
    main()
