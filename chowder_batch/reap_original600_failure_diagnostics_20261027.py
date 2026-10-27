"""Read-only, independently rescored original AutomationBench600 failure diagnostics.

Report only descriptive properties of authenticated ORIGINAL scored tasks. The
active/unfinished task remains UNKNOWN; no partial-prefix metric is an official
full-population score or an eligible frozen four-slot Gen-0 comparison.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from chowder_batch import reap_original600_live_independent_audit_20261027 as audit
from chowder_batch import reap_q4_automationbench_full600_resumable_20261027 as runner

SCHEMA = "chowder.reap128-v2-original600-authenticated-failure-diagnostics.v1"


def inspect(root: Path) -> dict:
    """Revalidate original receipts before and after reading each official raw row."""
    source = audit.inspect(root)
    row_counts: Counter[str] = Counter()
    positive_by_type: dict[str, Counter[str]] = {}
    all_stops: Counter[str] = Counter()
    positive_partial = 0
    failed_with_no_positive = 0
    excluded_violations = 0
    steps_at_budget = 0
    refused = 0
    assertions = {"positive": 0, "positive_passed": 0,
                  "excluded": 0, "excluded_passed": 0}
    individual: list[dict] = []
    for evidence in source["rows"]:
        index = evidence["index"]
        raw_path = root / f"task-{index:03d}" / "public-automationbench.private.json"
        expected_hash = evidence["original_scorer_json_sha256"]
        if audit.digest(raw_path) != expected_hash:
            raise ValueError(f"source-bound task {index} changed before diagnosis")
        raw = runner._json(raw_path)
        if audit.digest(raw_path) != expected_hash:
            raise ValueError(f"source-bound task {index} changed during diagnosis")
        row = raw["tasks"][0]
        if row["name"] != evidence["name"] or row["passed"] is not evidence["strict_pass"]:
            raise ValueError(f"source-bound task {index} no longer matches original scorer")
        if type(row.get("steps")) is not int or not 0 <= row["steps"] <= runner.q4.MAX_STEPS:
            raise ValueError("official step count malformed")
        if type(row.get("died_on_refusal")) is not bool:
            raise ValueError("official refusal state malformed")
        stops = row.get("stop_reasons")
        if not isinstance(stops, list) or any(type(s) is not str for s in stops):
            raise ValueError("official stop reasons malformed")
        positive = [a for a in row["assertion_results"] if not a["excluded"]]
        excluded = [a for a in row["assertion_results"] if a["excluded"]]
        p_passed = sum(a["passed"] for a in positive)
        x_passed = sum(a["passed"] for a in excluded)
        if (p_passed != evidence["positive_passed"]
                or len(positive) != evidence["positive_total"]):
            raise ValueError("positive assertion totals changed after original audit")
        assertions["positive"] += len(positive)
        assertions["positive_passed"] += p_passed
        assertions["excluded"] += len(excluded)
        assertions["excluded_passed"] += x_passed
        if row["steps"] == runner.q4.MAX_STEPS:
            steps_at_budget += 1
        refused += int(row["died_on_refusal"])
        excluded_violations += int(x_passed != len(excluded))
        if not row["passed"]:
            if p_passed == 0:
                failed_with_no_positive += 1
            elif p_passed < len(positive):
                positive_partial += 1
        for a in row["assertion_results"]:
            typ = a.get("type")
            if type(typ) is not str or not typ or len(typ) > 120:
                raise ValueError("original assertion type invalid")
            count = positive_by_type.setdefault(typ, Counter())
            kind = "excluded" if a["excluded"] else "positive"
            count[kind] += 1
            count[kind + "_passed"] += int(a["passed"])
        all_stops.update(stops)
        row_counts["strict_pass" if row["passed"] else "strict_fail"] += 1
        individual.append({
            "index": index, "name": evidence["name"],
            "strict_pass": row["passed"], "steps": row["steps"],
            "at_original_44_step_limit": row["steps"] == runner.q4.MAX_STEPS,
            "died_on_refusal": row["died_on_refusal"],
            "positive_total": len(positive), "positive_passed": p_passed,
            "excluded_total": len(excluded), "excluded_passed": x_passed,
            "original_raw_sha256": expected_hash,
            "original_receipt_sha256": evidence["checkpoint_sha256"],
        })
    if (assertions["positive"] != source["totals"]["positive_assertions"]
            or assertions["positive_passed"] != source["totals"]["positive_assertions_passed"]
            or assertions["excluded"] != source["totals"]["excluded_assertions"]
            or assertions["excluded_passed"] != source["totals"]["excluded_assertions_passed"]
            or row_counts["strict_pass"] != source["totals"]["strict_passes"]):
        raise ValueError("failure analysis disagrees with independent original scorer")
    return {
        "schema": SCHEMA, "status": "AUTHENTIC_PARTIAL_PREFIX_DESCRIPTIVE_ONLY"
        if len(individual) < runner.POPULATION else "AUTHENTIC_FULL600_DESCRIPTIVE_ONLY",
        "source_commit": source["official_source_commit"],
        "model_sha256": source["model_gguf_sha256"],
        "owner_sha256": source["owner_sha256"],
        "scored": len(individual), "population": runner.POPULATION,
        "active_unscored": source["active_unscored_indices"],
        "strict_passes": source["totals"]["strict_passes"],
        "strict_fails": row_counts["strict_fail"],
        "failed_with_no_positive_satisfied": failed_with_no_positive,
        "failed_with_some_positive_satisfied": positive_partial,
        "excluded_requirement_violations": excluded_violations,
        "at_original_44_step_limit": steps_at_budget,
        "original_refusal_terminations": refused,
        "stop_reasons_original": dict(sorted(all_stops.items())),
        "assertions": assertions,
        "by_original_assertion_type": {
            k: dict(sorted(v.items())) for k, v in sorted(positive_by_type.items())
        },
        "items": individual,
        "prefix_nonrandom_no_full600_confidence_interval": True,
        "noncomparable_to_frozen_four_slot": True,
        "frozen_gen0_status": "BLOCKED", "chosen": None,
        "release_approved": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-run", required=True, type=Path)
    parser.add_argument("--snapshot", type=Path,
                        help="Write only a new exclusive original-score-derived F: receipt")
    args = parser.parse_args()
    try:
        result = inspect(args.original_run)
        if args.snapshot is not None:
            with args.snapshot.open("xb") as stream:
                stream.write(runner._canonical(result))
        print(json.dumps({k: v for k, v in result.items()
                          if k not in ("items", "by_original_assertion_type")},
                         indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"status": "REFUSED_SOURCE_BOUND_DIAGNOSIS",
                          "error": str(error)[:300], "no_model_rerun": True,
                          "frozen_gen0_status": "BLOCKED"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
