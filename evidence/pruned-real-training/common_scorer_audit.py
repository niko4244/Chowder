"""Re-score every prediction file under BOTH scoring rules, symmetrically.

Why this exists: `final_number_match` behaves differently in the two text-evaluator
workers (docs/PRUNED_9B_REAL_TRAINING_CORRECTION.md). The automatic baseline runs
through `base_text_worker`, which discards an unclosed `<think>` block before
extracting a number; the candidate runs through `transformers_text_worker`, which
reads the raw text. The run's two sides are therefore not comparable as recorded.

This applies BOTH rules to EVERY prediction file, so a symmetric comparison exists
for whichever rule is chosen, and the spread between them is visible rather than
hidden. It is read-only against the run directory and needs no GPU, so it can run
while the candidate eval is still pending.

It also characterises the generations, because the baseline's 0.00 turned out to be
a scorer artifact rather than a content judgement, and that distinction cannot be
made from a score alone:

  * unclosed `<think>`      -- the model never produced an answer span at all
  * duplicate-line ratio    -- degeneration into repetition
  * hit the token cap       -- no EOS, so the budget was exhausted

A 0-vs-0 symmetric result is uninformative, not a finding. Say so if it happens.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\nikma\Chowder-router-healing\src")

RUN = Path(r"F:\llm-models\_a4b\realtrain-gsm8k")
OUT = Path(r"F:\llm-models\_a4b\common-scorer-audit.json")
CAP_TOKENS = 768  # the pre-registered max_new_tokens


def duplicate_line_ratio(text: str) -> float:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) < 2:
        return 0.0
    seen: set[str] = set()
    dupes = 0
    for line in lines:
        if line in seen:
            dupes += 1
        seen.add(line)
    return dupes / len(lines)


def main() -> int:
    from chowder.evaluators.base_text_worker import _score as score_base
    from chowder.evaluators.transformers_text_worker import _score as score_candidate

    evals = RUN / ".chowder" / "evals"
    if not evals.is_dir():
        raise SystemExit(f"no eval directory at {evals}")

    report: dict = {
        "run": str(RUN),
        "note": (
            "Both rules applied to every file. 'base' discards an unclosed <think> "
            "block before extracting a number; 'candidate' reads raw text. The "
            "recorded score is whichever rule that side's worker used."
        ),
        "suites": {},
    }

    for suite_dir in sorted(evals.iterdir()):
        predictions = suite_dir / "predictions-gsm8k.jsonl"
        if not predictions.is_file():
            continue
        rows = [json.loads(l) for l in predictions.read_text(encoding="utf-8").splitlines() if l.strip()]
        n = len(rows)

        recorded = sum(float(r.get("score", 0.0)) for r in rows)
        as_base = sum(score_base(r["prediction"], r["expected"], "final_number_match") for r in rows)
        as_cand = sum(score_candidate(r["prediction"], r["expected"], "final_number_match") for r in rows)

        opened = sum(1 for r in rows if "<think>" in r["prediction"])
        unclosed = sum(1 for r in rows if "<think>" in r["prediction"] and "</think>" not in r["prediction"])
        dup = [duplicate_line_ratio(r["prediction"]) for r in rows]
        degenerate = sum(1 for d in dup if d >= 0.5)

        # recorded metric the run itself will report, if the suite finished
        result_file = suite_dir / "eval-result.json"
        official = None
        adapter_loaded = None
        if result_file.is_file():
            data = json.loads(result_file.read_text(encoding="utf-8"))
            official = data.get("metrics", {}).get("gsm8k")
            adapter_loaded = data.get("model_provenance", {}).get("adapter_loaded")

        disagreements = [
            {
                "index": i,
                "expected": r["expected"],
                "base": score_base(r["prediction"], r["expected"], "final_number_match"),
                "candidate": score_candidate(r["prediction"], r["expected"], "final_number_match"),
                "tail": r["prediction"][-90:],
            }
            for i, r in enumerate(rows)
            if score_base(r["prediction"], r["expected"], "final_number_match")
            != score_candidate(r["prediction"], r["expected"], "final_number_match")
        ]

        report["suites"][suite_dir.name] = {
            "rows": n,
            "complete": n == 50,
            "adapter_loaded": adapter_loaded,
            "recorded_metric": official,
            "recorded_sum": recorded,
            "scored_as_base_rule": as_base,
            "scored_as_candidate_rule": as_cand,
            "base_accuracy": round(as_base / n, 4) if n else None,
            "candidate_accuracy": round(as_cand / n, 4) if n else None,
            "generations": {
                "opened_think": opened,
                "unclosed_think": unclosed,
                "mean_duplicate_line_ratio": round(sum(dup) / n, 4) if n else None,
                "degenerate_half_or_more_duplicate_lines": degenerate,
            },
            "rule_disagreements": disagreements,
        }

    print(f"{'suite':<42}{'rows':>5}{'base':>7}{'cand':>7}{'unclosed':>10}{'degen':>7}")
    for name, s in report["suites"].items():
        g = s["generations"]
        print(f"{name:<42}{s['rows']:>5}{s['scored_as_base_rule']:>7.0f}"
              f"{s['scored_as_candidate_rule']:>7.0f}{g['unclosed_think']:>10}"
              f"{g['degenerate_half_or_more_duplicate_lines']:>7}")

    sides = list(report["suites"].values())
    if len(sides) >= 2:
        base_vals = [s["scored_as_base_rule"] for s in sides]
        cand_vals = [s["scored_as_candidate_rule"] for s in sides]
        report["symmetric_comparison"] = {
            "base_rule_both_sides": base_vals,
            "candidate_rule_both_sides": cand_vals,
            "informative": not (max(base_vals) == 0 and max(cand_vals) == 0),
        }
    else:
        report["symmetric_comparison"] = {
            "status": "pending",
            "note": "only one side has predictions so far; rerun when the candidate eval finishes",
        }

    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
