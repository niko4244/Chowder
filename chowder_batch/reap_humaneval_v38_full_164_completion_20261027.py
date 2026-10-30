"""Independently authenticate completed original HumanEval164 *research* scores.

The existing whole-journal auditor verifies every actual sealed-memfd execution
and original source/model/task/serving-session witness. This separate gate
checks that a post-run completion certificate matches that independently
recomputed full score exactly. Never starts model/code, never promotes a V38
research result to frozen V2 Gate2, and refuses any partial 164 population.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from chowder_batch import reap_v38_independent_receipt_audit_worker13_20261027 as independent
from chowder_batch import reap_humaneval_v38_actual_execution_20261027 as original

ROOTS = {
    "q4": Path("F:/reap-v2-human164-v38-private-q4-sealed-original-20261027-1325"),
    "q8": Path("F:/reap-v2-human164-v38-private-q8-sealed-original-20261027-1329"),
}
SCHEMA = "chowder.reap128-v2-human164-completion-source-cross-certificate.v1"
COMPLETE_SCHEMA = "chowder.reap-v2-original-human164-independent-research.v38"
COMPLETE_STATUS = "COMPLETE_164_ORIGINAL_CODE_ISOLATED_RESEARCH_ONLY_NOT_FROZEN_GATE2"


def validate_completion_json(arm: str, raw: bytes, audited: dict, *, row_count: int) -> dict:
    """Fail closed on certificate forgery, scientific gate promotion or truncation."""
    if arm not in ROOTS or len(raw) > 64 * 1024 or not raw:
        raise ValueError("wrong arm, empty or oversized original completion certificate")
    summary = independent._strict_json(raw)
    manifest = original.original_program_manifest(arm)
    expected = {
        "schema": COMPLETE_SCHEMA,
        "status": COMPLETE_STATUS,
        "arm": manifest["arm"],
        "original_model_sha256": audited["original_model_sha256"],
        "original_rows_sha256": audited["original_rows_sha256"],
        "canonical_sha256": original.CANONICAL_SHA256,
        "candidate_identity_manifest_sha256": audited["ordered_program_manifest_sha256"],
        "original_rows": 164,
        "executed_rows": audited["verified_actual_executions"],
        "unterminated_rows_source_skipped": audited["verified_source_unterminated_nonpasses"],
        "independently_research_scored_correct": audited["verified_independent_correct_prefix"],
        "per_case_evidence_sha256": audited["journal_snapshot_sha256"],
        "original_v2_attestation_approved": False,
        "windows_native_gate_eligible": False,
        "gen0_status": "BLOCKED",
        "chosen": None,
        "release_approved": False,
    }
    if set(summary) != set(expected) | {"independent_score"}:
        raise ValueError("original final score certificate fields added/missing")
    for key, value in expected.items():
        if type(summary[key]) is not type(value) or summary[key] != value:
            raise ValueError("independent original full-suite certificate mismatch: " + key)
    score = summary["independent_score"]
    if (type(score) is not float or not math.isfinite(score)
            or score != audited["verified_independent_correct_prefix"] / 164):
        raise ValueError("original research accuracy not equal to independent exact count")
    if (row_count != 164 or audited["verified_contiguous_rows"] != 164
            or audited["status"] != "COMPLETE_164_RESEARCH_ONLY"
            or audited["unknown_remaining_rows"] != 0
            or audited["full_164_denominator"] != 164
            or audited["full_164_independent_correct"] != summary["independently_research_scored_correct"]
            or audited["verified_actual_executions"] + audited["verified_source_unterminated_nonpasses"] != 164):
        raise ValueError("not a fully source-bound 164 HumanEval research score")
    return {
        "schema": SCHEMA,
        "status": "INDEPENDENTLY_VERIFIED_COMPLETE_164_RESEARCH_ONLY",
        "arm": arm,
        "model_sha256": audited["original_model_sha256"],
        "original_source_rows_sha256": audited["original_rows_sha256"],
        "canonical_source_sha256": original.CANONICAL_SHA256,
        "original_serving_session": audited["original_session"],
        "source_program_population_sha256": audited["ordered_program_manifest_sha256"],
        "scored_original_rows": 164,
        "actual_independently_witnessed_executions": audited["verified_actual_executions"],
        "original_unterminated_nonpasses": audited["verified_source_unterminated_nonpasses"],
        "independent_correct": audited["verified_independent_correct_prefix"],
        "independent_accuracy": score,
        "incorrect_case_ids": [r["id"] for r in audited["rows"] if not r["research_correct"]],
        "full_journal_sha256": audited["journal_snapshot_sha256"],
        "completion_receipt_sha256": hashlib.sha256(raw).hexdigest(),
        "outside_vm_host_witness_source_sha256": audited["reviewed_observer_sha256"],
        "immutable_memfd_loader_source_sha256": audited["reviewed_loader_sha256"],
        "original_frozen_v2_gate2_eligible": False,
        "dense_paired_gate2_complete": False,
        "gen0_status": "BLOCKED", "chosen": None, "release_approved": False,
    }


def certify(arm: str) -> dict:
    if arm not in ROOTS:
        raise ValueError("only original registered Q4/Q8 cohorts supported")
    root = ROOTS[arm]
    if not root.is_dir() or root.is_symlink():
        raise ValueError("original physical model research root missing")
    journal = root / (arm + "-actual-independent-per-case.private.jsonl")
    complete = root / (arm + "-complete.private.json")
    if not complete.is_file() or complete.is_symlink():
        raise ValueError("no authentic final 164-case completion certificate yet")
    # Independently replay original model source against EVERY original row,
    # candidate SHA, memfd seals, host sandbox and cleanup witness.
    verified = independent.audit_live(arm, journal)
    raw = complete.read_bytes()
    return validate_completion_json(arm, raw, verified,
                                    row_count=verified["verified_contiguous_rows"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["q4", "q8"], required=True)
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    try:
        result = certify(args.arm)
        if args.snapshot is not None:
            if (not args.snapshot.is_absolute() or args.snapshot.exists()
                    or args.snapshot.is_symlink()
                    or args.snapshot.parent.resolve() != Path("F:/").resolve()):
                raise ValueError("new exclusive private F: result path required")
            with args.snapshot.open("xb") as stream:
                stream.write((json.dumps(result, sort_keys=True, indent=2) + "\n").encode())
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0
    except (ValueError, OSError, UnicodeError, TypeError, KeyError, OverflowError) as exc:
        print(json.dumps({"status": "REFUSED_UNTRUSTED_FULL_HUMANEVAL_164_CERTIFICATE",
                          "reason": str(exc)[:400], "gen0_status": "BLOCKED"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
