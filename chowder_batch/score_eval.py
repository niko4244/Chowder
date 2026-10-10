"""Scores captured model outputs (from student_eval.py --model) against fixtures.

One command:  python score_eval.py eval_results.jsonl
  - runs the part-4 checker for each row's fixture on the captured answer
  - writes <input>_scored.jsonl with per-row pass/fail
  - prints per-fixture and per-family results
Baselines through the same scoring path:
  python score_eval.py --baseline gold   (expect 8/8)
  python score_eval.py --baseline anti   (expect 0/8)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PARTS = ["eval_part1.py", "eval_part2.py", "eval_part3.py",
         "eval_part4.py", "eval_part5.py"]


def load_harness():
    """Assemble the eval parts into one namespace, exactly like student_eval."""
    ns = {"__name__": "chowder_score_eval"}
    for part in PARTS:
        path = os.path.join(HERE, part)
        with open(path, encoding="utf-8") as f:
            code = compile(f.read(), part, "exec")
        ns["__file__"] = path  # parts read __file__ at module level (all same dir)
        exec(code, ns)
    return ns


def fixtures_in_order(ns):
    return [ns["fixture_planner"](), ns["fixture_importer"](), ns["fixture_stopper"](),
            ns["fixture_decimals"](), ns["fixture_urljoin"](), ns["fixture_decimal_mul"](),
            ns["fixture_repeat"](), ns["fixture_state"](),
            ns["fixture_spark_envelope_basic"](), ns["fixture_spark_envelope_gated_loop"](),
            ns["fixture_spark_structured_args"]()]


def score_rows(rows, fixtures, checkers):
    by_id = {fx["fixture_id"]: fx for fx in fixtures}
    scored, unknown = [], []
    for row in rows:
        fid = row.get("fixture_id")
        if fid not in by_id:
            unknown.append(fid)
            continue
        passed = bool(checkers[fid](row.get("answer", ""), by_id[fid]))
        scored.append({**row, "scored": True, "passed": passed})
    return scored, unknown


def summarize(scored):
    fam_total, fam_pass = {}, {}
    for row in scored:
        fam = row.get("family", "?")
        fam_total[fam] = fam_total.get(fam, 0) + 1
        fam_pass[fam] = fam_pass.get(fam, 0) + (1 if row["passed"] else 0)
    total = len(scored)
    npass = sum(1 for r in scored if r["passed"])
    return total, npass, fam_total, fam_pass


def write_scored(rows, out_path):
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Score captured eval outputs against fixtures")
    ap.add_argument("captured", nargs="?", help="captured results jsonl from student_eval.py --model")
    ap.add_argument("--out", default=None, help="scored output path (default <captured>_scored.jsonl)")
    ap.add_argument("--baseline", choices=["gold", "anti"], default=None,
                    help="score the fixtures' own gold/anti answers through the scoring path")
    args = ap.parse_args()
    if not args.captured and not args.baseline:
        ap.error("provide a captured jsonl or --baseline gold|anti")

    ns = load_harness()
    fixtures = fixtures_in_order(ns)
    checkers = ns["CHECKERS"]

    if args.baseline:
        key = "gold_answer" if args.baseline == "gold" else "anti_answer"
        rows = [{"fixture_id": fx["fixture_id"], "family": fx["family"],
                 "answer": fx[key]} for fx in fixtures]
        out_default = "baseline_%s_scored.jsonl" % args.baseline
    else:
        with open(args.captured, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        out_default = os.path.splitext(args.captured)[0] + "_scored.jsonl"

    scored, unknown = score_rows(rows, fixtures, checkers)
    out_path = args.out or out_default
    write_scored(scored, out_path)

    for row in scored:
        print("%-8s %-32s %s" % (row["fixture_id"], row.get("family", ""),
                                 "PASS" if row["passed"] else "FAIL"))
    if unknown:
        print("UNKNOWN fixture_ids (not scored): %s" % sorted(set(unknown)))
    total, npass, fam_total, fam_pass = summarize(scored)
    print("---")
    for fam in sorted(fam_total):
        print("%-34s %d/%d" % (fam, fam_pass[fam], fam_total[fam]))
    print("TOTAL %d/%d" % (npass, total))
    print("scored rows written to", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
