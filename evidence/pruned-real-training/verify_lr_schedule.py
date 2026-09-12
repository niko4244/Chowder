"""CLI over `chowder.schedule_audit`: which LR schedule did a run actually follow?

The maths used to live here. It now lives in `src/chowder/schedule_audit.py`, with
its behaviour pinned by `tests/test_schedule_audit.py` (including this run's real
500-step trajectory as a fixture), so this file is a thin reader over the sampled
trajectory. Two copies of a rule drift -- that is exactly the defect this session
fixed for the two text scorers, and there is no reason to recreate it here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\nikma\Chowder-router-healing\src")

from chowder.schedule_audit import identify_schedule  # noqa: E402

TRAJECTORY = Path(r"F:\llm-models\_a4b\lr-trajectory.jsonl")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory", default=str(TRAJECTORY))
    ap.add_argument("--expect", default="cosine")
    ap.add_argument("--peak-lr", type=float, default=2e-4)
    ap.add_argument("--warmup-steps", type=int, default=0)
    args = ap.parse_args()

    path = Path(args.trajectory)
    if not path.is_file():
        print(f"no trajectory at {path}")
        return 2
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    observed = [
        (r["step"], float(r["learning_rate"]))
        for r in rows
        if isinstance(r.get("learning_rate"), (int, float))
    ]
    if not observed:
        print("trajectory carries no learning_rate values")
        return 2
    total = next((r["max_steps"] for r in rows if isinstance(r.get("max_steps"), int)), 500)

    verdict = identify_schedule(
        observed, peak_lr=args.peak_lr, total_steps=total, warmup_steps=args.warmup_steps
    )
    print(f"samples {verdict.samples}  steps {observed[0][0]}..{observed[-1][0]}  "
          f"max_steps {total}")
    print(f"observed LR  first {observed[0][1]:.6e}  last {observed[-1][1]:.6e}")
    print(f"best fit {verdict.best_fit} at offset {verdict.offset}: "
          f"residual {verdict.residual_fraction_of_peak*100:.4f}% of peak, "
          f"separation {verdict.separation:.1f}x")
    print(f"\nVERDICT: {verdict.schedule}")
    if verdict.matches(args.expect):
        print(f"=> matches the pre-registered {args.expect} schedule")
        return 0
    print(f"=> DOES NOT positively support the pre-registered {args.expect} schedule")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
