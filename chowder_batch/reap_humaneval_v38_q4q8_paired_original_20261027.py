"""Authentic independently executed original Q8-versus-Q4 HumanEval research.

Paired per-item outcomes are derived exclusively from independent
model/source/program/witness-audited ORIGINAL generated-code executions.
Q8 is live and often incomplete. Prefix pairs must never masquerade as
complete or frozen four-slot same-session Gate2 evidence.

No model inference, executable code, native Podman, source mutation, or
optional Amendment4 test occurs here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

from chowder_batch import reap_v38_independent_receipt_audit_worker13_20261027 as independent
from chowder_batch import reap_humaneval_v38_full_164_completion_20261027 as completion
from chowder_batch import reap_humaneval_v38_actual_execution_20261027 as original

SCHEMA = "chowder.original-human164-q4-q8-paired-independent-research.v1"


def verified_pair(q4: dict, q8: dict, *, q8_full_certified: bool) -> dict:
    """Cross-match exact canonical items, never separate or shuffled tasks."""
    if q4.get("arm") != "q4" or q8.get("arm") != "q8":
        raise ValueError("original model source/arm pairing mismatch")
    if (q4.get("verified_contiguous_rows") != 164
            or q4.get("status") != "COMPLETE_164_RESEARCH_ONLY"
            or q4.get("full_164_independent_correct") is None
            or q8.get("status") not in ("VERIFIED_PREFIX_ONLY", "COMPLETE_164_RESEARCH_ONLY")):
        raise ValueError("authentic complete Q4 and source-bound Q8 prefix required")
    n = q8.get("verified_contiguous_rows")
    if type(n) is not int or not 0 < n <= 164:
        raise ValueError("no authentic original paired observations")
    if not isinstance(q4.get("rows"), list) or len(q4["rows"]) != 164:
        raise ValueError("original complete Q4 case identities unavailable")
    if not isinstance(q8.get("rows"), list) or len(q8["rows"]) != n:
        raise ValueError("original Q8 scored case identities unavailable")
    if (q4.get("unknown_remaining_rows") != 0
            or q4.get("full_164_denominator") != 164
            or q4.get("full_164_independent_correct")
                != q4.get("verified_independent_correct_prefix")
            or q8.get("unknown_remaining_rows") != 164 - n
            or type(q4.get("verified_actual_executions")) is not int
            or type(q8.get("verified_actual_executions")) is not int
            or type(q4.get("verified_source_unterminated_nonpasses")) is not int
            or type(q8.get("verified_source_unterminated_nonpasses")) is not int
            or q4["verified_actual_executions"] + q4["verified_source_unterminated_nonpasses"] != 164
            or q8["verified_actual_executions"] + q8["verified_source_unterminated_nonpasses"] != n):
        raise ValueError("independent original population/termination metadata inconsistent")
    if q4["original_model_sha256"] == q8["original_model_sha256"]:
        raise ValueError("Q8 source must be independently distinct from Q4 GGUF")
    if q4["original_rows_sha256"] == q8["original_rows_sha256"]:
        raise ValueError("original Q8 generated-response cohort cannot be Q4 alias")
    if q4["ordered_program_manifest_sha256"] == q8["ordered_program_manifest_sha256"]:
        raise ValueError("quantization variants cannot reuse original Q4 executable cohort")
    if q4.get("gen0_status") != "BLOCKED" or q8.get("gen0_status") != "BLOCKED":
        raise ValueError("unexpected frozen scientific approval in research evidence")
    complete = n == 164
    if complete and (not q8_full_certified
                     or q8["status"] != "COMPLETE_164_RESEARCH_ONLY"
                     or q8.get("unknown_remaining_rows") != 0
                     or q8.get("full_164_denominator") != 164):
        raise ValueError("Q8 complete claim lacks independent completed certificate")
    if not complete and q8_full_certified:
        raise ValueError("partial Q8 cannot be labeled full independently certified")
    if q4["verified_independent_correct_prefix"] != sum(
            type(row.get("research_correct")) is bool and row["research_correct"]
            for row in q4["rows"]):
        raise ValueError("original Q4 outcomes inconsistent with independent witness")
    if q8["verified_independent_correct_prefix"] != sum(
            type(row.get("research_correct")) is bool and row["research_correct"]
            for row in q8["rows"]):
        raise ValueError("original Q8 outcomes inconsistent with independent witness")
    original_missing = {
        arm: {row["id"] for row in original.original_program_manifest(arm)[
            "source_rows_in_original_order"] if row["candidate_program_sha256"] is None}
        for arm in ("q4", "q8")
    }
    if (sum(r.get("program_sha256") is None for r in q4["rows"])
            != q4["verified_source_unterminated_nonpasses"]
            or sum(r.get("program_sha256") is None for r in q8["rows"])
                != q8["verified_source_unterminated_nonpasses"]):
        raise ValueError("source-confirmed unterminated program census mismatch")
    paired, both, q8_only, q4_only, neither = [], 0, 0, 0, 0
    for i in range(n):
        a, b = q4["rows"][i], q8["rows"][i]
        if (type(a.get("index")) is not int or a["index"] != i
                or type(b.get("index")) is not int or b["index"] != i
                or a.get("id") != b.get("id")
                or type(a.get("research_correct")) is not bool
                or type(b.get("research_correct")) is not bool
                # The registered generator did NOT terminate on two original
                # Q4 cases. The independent manifest uses None for their
                # nonexistent executable bytes; that is a genuine nonpass,
                # never a missing/fabricated code SHA. Same for any Q8
                # source-confirmed unterminated case.
                or not (a.get("program_sha256") is None and not a["research_correct"]
                        and a["id"] in original_missing["q4"]
                        or isinstance(a.get("program_sha256"), str)
                        and re.fullmatch(r"[0-9a-f]{64}", a["program_sha256"]))
                or not (b.get("program_sha256") is None and not b["research_correct"]
                        and b["id"] in original_missing["q8"]
                        or isinstance(b.get("program_sha256"), str)
                        and re.fullmatch(r"[0-9a-f]{64}", b["program_sha256"]))):
            raise ValueError("Q4/Q8 exact canonical same-case pairing violated")
        c4, c8 = a["research_correct"], b["research_correct"]
        both += int(c4 and c8)
        q8_only += int(c8 and not c4)
        q4_only += int(c4 and not c8)
        neither += int(not c4 and not c8)
        paired.append({"index": i, "id": a["id"],
                       "q4_original_program_sha256": a["program_sha256"],
                       "q8_original_program_sha256": b["program_sha256"],
                       "q4_independent_correct": c4, "q8_independent_correct": c8})
    observed_delta = (q8_only - q4_only) / n
    # Diagnostic only; not the frozen Gate1/Gate2 or Amendment4 statistic.
    # Population ends on a streaming prefix, so its CI is NOT confirmatory.
    s2 = (q8_only + q4_only - n * observed_delta * observed_delta) / (n - 1) if n > 1 else None
    diagnostic = (observed_delta - 1.645 * math.sqrt(max(0.0, s2 / n))
                  if s2 is not None else None)
    return {
        "schema": SCHEMA,
        "status": ("FULL_164_INDEPENDENT_RESEARCH_PAIRED_Q8_Q4_ONLY"
                   if complete else "PARTIAL_PREFIX_SOURCE_BOUND_NOT_A_FULL_BENCHMARK"),
        "population": 164,
        "paired_original_case_count": n,
        "remaining_unscored_q8_original_cases": 164 - n,
        "original_q4_model_sha256": q4["original_model_sha256"],
        "original_q8_model_sha256": q8["original_model_sha256"],
        "original_q4_serving_session": q4["original_session"],
        "original_q8_serving_session": q8["original_session"],
        "original_q4_generated_rows_sha256": q4["original_rows_sha256"],
        "original_q8_generated_rows_sha256": q8["original_rows_sha256"],
        "original_q4_witnessed_journal_sha256": q4["journal_snapshot_sha256"],
        "original_q8_witnessed_journal_sha256": q8["journal_snapshot_sha256"],
        "original_q4_correct_in_paired_items": both + q4_only,
        "original_q8_correct_in_paired_items": both + q8_only,
        "both_correct": both, "q8_only_correct": q8_only,
        "q4_only_correct": q4_only, "both_incorrect": neither,
        "observed_difference_q8_minus_q4_on_verified_original_prefix": observed_delta,
        "diagnostic_sample_sd_wald_lower_bound_not_scientific_gate": diagnostic,
        "per_item_original_source_bound_pairs": paired,
        "is_confirmatory_quantization_comparison": False,
        "is_original_frozen_Q8_vs_dense_gate2": False,
        "is_amendment4_paired_mcnemar_ci": False,
        "release_approved": False,
        "gen0_status": "BLOCKED", "chosen": None,
    }


def audit_actual() -> dict:
    root4, root8 = completion.ROOTS["q4"], completion.ROOTS["q8"]
    verified4 = independent.audit_live("q4", root4 / "q4-actual-independent-per-case.private.jsonl")
    verified8 = independent.audit_live("q8", root8 / "q8-actual-independent-per-case.private.jsonl")
    # Authenticate completed Q4 against its exact separate final certificate,
    # not just its last mutable file or model self-reported original grade.
    complete4 = completion.certify("q4")
    if (complete4["full_journal_sha256"] != verified4["journal_snapshot_sha256"]
            or complete4["independent_correct"] != verified4["verified_independent_correct_prefix"]):
        raise ValueError("Q4 complete source changed between two independent reads")
    full8 = verified8["verified_contiguous_rows"] == 164
    if full8:
        complete8 = completion.certify("q8")
        if (complete8["full_journal_sha256"] != verified8["journal_snapshot_sha256"]
                or complete8["independent_correct"] != verified8["verified_independent_correct_prefix"]):
            raise ValueError("Q8 complete source changed between independent reads")
    return verified_pair(verified4, verified8, q8_full_certified=full8)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, help="Write NEW exclusive private evidence F:")
    args = parser.parse_args()
    try:
        result = audit_actual()
        if args.snapshot:
            if (not args.snapshot.is_absolute() or args.snapshot.exists()
                    or args.snapshot.is_symlink()
                    or args.snapshot.parent.resolve() != Path("F:/").resolve()):
                raise ValueError("exclusive F: research evidence destination required")
            with args.snapshot.open("xb") as stream:
                content = (json.dumps(result, sort_keys=True, indent=2) + "\n").encode()
                stream.write(content)
                stream.flush()
                import os
                os.fsync(stream.fileno())
            print("IMMUTABLE_PAIRED_RESEARCH_SHA256", hashlib.sha256(content).hexdigest())
        print(json.dumps({k: v for k, v in result.items() if k != "per_item_original_source_bound_pairs"},
                         sort_keys=True, indent=2))
        return 0
    except (ValueError, OSError, KeyError, TypeError, UnicodeError) as exc:
        print(json.dumps({"status": "REFUSED_SOURCE_BOUND_Q4_Q8_PAIR",
                          "reason": str(exc)[:320], "gen0_status": "BLOCKED"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
