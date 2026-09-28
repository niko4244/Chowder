"""Apply A4_PLAN.md's pre-registered promotion gate to the saved predictions.

A4 is preferred over A3 only if, on paired rows:
  1. MATH-150: A4 - A3 95% bootstrap CI lower bound > 0;
  2. GSM8K:    A4 - A3 CI lower bound > -0.05;
  3. EOS rate on both benchmarks not lower than A3's.
Everything is re-scored here from raw predictions with the current scorer
(EOS-gated, boxed-aware; math_verify_match for MATH), never read from a
worker's own metric. Decision is always recorded as requires_operator_review.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from chowder.evaluators.scoring import score


def load(paths: list[Path]) -> list[dict]:
    rows = []
    for p in paths:  # row ranges of one arm, concatenated in the order given
        rows += [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows


def scored(rows: list[dict], scoring: str) -> list[float]:
    return [score(r["prediction"], r["expected"], scoring, finished=r["eos_terminated"]) for r in rows]


def paired(a: list[float], b: list[float], seed: int = 2026, samples: int = 10000) -> dict:
    d = [y - x for x, y in zip(a, b)]
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(samples))
    return {"n": len(d), "delta": sum(d) / len(d), "ci95": [boots[int(0.025 * samples)], boots[int(0.975 * samples)]],
            "wins": d.count(1.0), "losses": d.count(-1.0)}


def arm_summary(rows: list[dict], s: list[float]) -> dict:
    return {"n": len(rows), "correct": sum(s), "accuracy": sum(s) / len(rows),
            "eos_rate": sum(bool(r["eos_terminated"]) for r in rows) / len(rows)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gsm8k-base", type=Path, required=True)
    ap.add_argument("--gsm8k-a3", type=Path, required=True)
    ap.add_argument("--gsm8k-a4", type=Path, required=True)
    ap.add_argument("--math-base", type=Path, nargs="+", required=True)
    ap.add_argument("--math-a3", type=Path, nargs="+", required=True)
    ap.add_argument("--math-a4", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    report: dict = {"gate": "A4_PLAN.md pre-registered", "benchmarks": {}}
    for bench, scoring, (b, a3, a4) in (
        ("gsm8k", "final_number_match", ([args.gsm8k_base], [args.gsm8k_a3], [args.gsm8k_a4])),
        ("math150", "math_verify_match", (args.math_base, args.math_a3, args.math_a4)),
    ):
        rows = {name: load(paths) for name, paths in (("base", b), ("a3", a3), ("a4", a4))}
        n = {len(v) for v in rows.values()}
        if len(n) != 1 or any(x["prompt"] != y["prompt"] for x, y in zip(rows["a3"], rows["a4"])):
            raise SystemExit(f"{bench}: arms are not row-aligned ({ {k: len(v) for k, v in rows.items()} })")
        sc = {name: scored(v, scoring) for name, v in rows.items()}
        report["benchmarks"][bench] = {
            "arms": {name: arm_summary(rows[name], sc[name]) for name in rows},
            "a3_vs_base": paired(sc["base"], sc["a3"]),
            "a4_vs_base": paired(sc["base"], sc["a4"]),
            "a4_vs_a3": paired(sc["a3"], sc["a4"]),
        }
    m, g = report["benchmarks"]["math150"], report["benchmarks"]["gsm8k"]
    checks = {
        "1_math150_a4_minus_a3_ci_low_gt_0": m["a4_vs_a3"]["ci95"][0] > 0,
        "2_gsm8k_a4_minus_a3_ci_low_gt_-0.05": g["a4_vs_a3"]["ci95"][0] > -0.05,
        "3_eos_rate_not_lower": all(bm["arms"]["a4"]["eos_rate"] >= bm["arms"]["a3"]["eos_rate"] for bm in (m, g)),
    }
    report["checks"] = checks
    report["a4_preferred"] = all(checks.values())
    report["decision"] = "requires_operator_review"
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"checks": checks, "a4_preferred": report["a4_preferred"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
