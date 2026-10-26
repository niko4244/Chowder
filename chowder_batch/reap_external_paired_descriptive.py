"""Source-bound *descriptive* pairing of independently audited external runs.

Does not calculate a confidence interval, certify noninferiority, or modify
frozen Amendment4 selection. Input must be previously independently audited,
complete, provisional Linux experiments, never aborted or merged sessions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

from chowder_batch.reap_external_full_audit import MODELS

STATUS = "COMPLETE_LINUX_COHORT_VERIFIED_PROVISIONAL_NOT_WINDOWS_GATE"
SOURCE_COUNTS = {"gsm8k": 1319, "humaneval": 164}
MODEL_SHA = {arm: fields[3] for arm, fields in MODELS.items()}


def _read_file(path: Path, limit: int) -> bytes:
    if (path.is_symlink() or not path.is_file() or path.stat().st_size == 0
            or path.stat().st_size > limit):
        raise ValueError("missing, symlinked, empty or oversized private evidence file")
    return path.read_bytes()


def read_verified_cohort(suite: str, audit: dict, raw: bytes, *, expected_n: int) -> dict:
    if suite not in SOURCE_COUNTS or expected_n <= 1 or expected_n > SOURCE_COUNTS[suite]:
        raise ValueError("invalid canonical benchmark cohort scope")
    if (not isinstance(audit, dict)
            or audit.get("schema") != "chowder.reap-v2-original-population-external-independent-audit.v1"
            or audit.get("status") != STATUS
            or audit.get("model") not in MODEL_SHA
            or audit.get("model_sha256") != MODEL_SHA[audit["model"]]
            or audit.get("original_windows_gate_eligible") is not False
            or audit.get("speed_ratio_certified") is not False
            or audit.get("gen0_status") != "BLOCKED"
            or audit.get("chosen") is not None
            or audit.get("release_approved") is not False
            or not isinstance(audit.get("session_id"), str)
            or not re.fullmatch(r"[0-9a-f]{32}", audit["session_id"])
            or not isinstance(audit.get("original_machine_receipt_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", audit["original_machine_receipt_sha256"])
            or not isinstance(audit.get("original_native_cuda_placement"), dict)
            or audit["original_native_cuda_placement"].get("native_gpu_offloaded_layers") != "41/41"
            or audit["original_native_cuda_placement"].get("n_seq_max") != 4
            or audit["original_native_cuda_placement"].get("n_ctx") != 16384
            or audit["original_native_cuda_placement"].get("n_ctx_seq") != 4096
            or audit["original_native_cuda_placement"].get("total_kv_buffer_mib") != 320.0
            or audit["original_native_cuda_placement"].get("loopback_bound") is not True):
        raise ValueError("external run is not an independently complete audited original session")
    if audit["model"] != "reap128_v2_q4":
        witness = audit.get("original_measured_postload_resource_witness")
        if (not isinstance(witness, dict)
                or witness.get("status") != "COMPLETE_LINUX_SOURCE_MEASURED_POSTLOAD_RESOURCE_WITNESS"
                or witness.get("original_windows_commit_attested") is not False
                or witness.get("model_fit_windows_gate_eligible") is not False
                or type(witness.get("monitored_sample_count")) is not int
                or witness["monitored_sample_count"] < 2):
            raise ValueError("external Q8/control resource source-audit attestation is missing")
        for key, minimum in (
            ("minimum_measured_cgroup_unused_gib", 10),
            ("minimum_measured_physical_available_gib", 7),
            ("measured_postload_recovery_gib", 4),
            ("measured_file_cache_decrease_gib", 4),
        ):
            value = witness.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
                raise ValueError("external Q8/control resource witness has invalid measured safety values")
        gpu = witness.get("initial_actual_gpu_free_mib")
        if (not isinstance(gpu, list) or len(gpu) != 2
                or any(type(v) is not int or v < 12288 for v in gpu)):
            raise ValueError("external Q8/control resource native start VRAM admission absent")
    # Original HumanEval scoring executes generated programs. A complete
    # original-scored cohort alone is not an independently replayed code outcome.
    # The full auditor emits this proof only after checking the separate
    # 164/164 replay's exact row SHA, serving session and correctness count.
    # Never describe an unreplayed arm as independently verified paired code.
    if suite == "humaneval":
        replay = audit.get("human_eval_independent_execution")
        if (not isinstance(replay, str) or not re.fullmatch(
            r"PASS_FULL_164_LOW_PRIVILEGE_REEXECUTION_SHA256:[0-9a-f]{64}", replay
        )):
            raise ValueError("HumanEval paired comparison requires an independent full-164 code execution receipt for each arm")
    summary = audit.get("suites", {}).get(suite)
    if (not isinstance(summary, dict) or type(summary.get("n")) is not int
            or summary["n"] != expected_n
            or type(summary.get("correct")) is not int
            or type(summary.get("terminated")) is not int
            or not isinstance(summary.get("row_sha256"), str)):
        raise ValueError("original complete audited suite summary missing")
    if (not isinstance(raw, bytes) or len(raw) > 64 * 1024 * 1024 or not raw.endswith(b"\n")
            or hashlib.sha256(raw).hexdigest() != summary["row_sha256"]):
        raise ValueError("immutable complete audited source JSONL hash mismatch")
    try:
        rows = [json.loads(line) for line in raw.splitlines()]
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("external original scored rows cannot be decoded") from exc
    if len(rows) != expected_n or not all(isinstance(row, dict) for row in rows):
        raise ValueError("incomplete or malformed original result population")
    expected_ids = ({f"gsm8k-{i}" for i in range(expected_n)} if suite == "gsm8k"
                    else {f"HumanEval/{i}" for i in range(expected_n)})
    by_id: dict[str, dict] = {}
    for row in rows:
        ident = row.get("id")
        if (type(ident) is not str or ident in by_id
                or type(row.get("correct")) is not bool
                or type(row.get("terminated")) is not bool
                or row.get("eval_session_id") != audit["session_id"]
                or (row["correct"] and not row["terminated"])):
            raise ValueError("original row id/session/outcome/termination integrity failed")
        by_id[ident] = row
    if (set(by_id) != expected_ids
            or sum(row["correct"] for row in rows) != summary["correct"]
            or sum(row["terminated"] for row in rows) != summary["terminated"]):
        raise ValueError("original audited cohort IDs or scorer aggregates changed")
    return by_id


def compare(suite: str, candidate_audit: dict, candidate_raw: bytes,
            reference_audit: dict, reference_raw: bytes, *, expected_n: int) -> dict:
    if (candidate_audit.get("model") == reference_audit.get("model")
            or candidate_audit.get("session_id") == reference_audit.get("session_id")):
        raise ValueError("different original model identities and independent sessions required")
    if {candidate_audit.get("model"), reference_audit.get("model")} - set(MODEL_SHA):
        raise ValueError("unrecognized original compared models")
    lhs = read_verified_cohort(suite, candidate_audit, candidate_raw, expected_n=expected_n)
    rhs = read_verified_cohort(suite, reference_audit, reference_raw, expected_n=expected_n)
    both_correct = both_wrong = only_candidate = only_reference = 0
    for ident, row in lhs.items():
        candidate, reference = row["correct"], rhs[ident]["correct"]
        both_correct += candidate and reference
        both_wrong += not candidate and not reference
        only_candidate += candidate and not reference
        only_reference += not candidate and reference
    if sum((both_correct, both_wrong, only_candidate, only_reference)) != expected_n:
        raise ValueError("paired original outcome partition failed")
    return {
        "suite": suite, "n": expected_n,
        "candidate_model": candidate_audit["model"],
        "reference_model": reference_audit["model"],
        "candidate_model_sha256": candidate_audit["model_sha256"],
        "reference_model_sha256": reference_audit["model_sha256"],
        "candidate_session": candidate_audit["session_id"],
        "reference_session": reference_audit["session_id"],
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "candidate_only_correct": only_candidate,
        "reference_only_correct": only_reference,
        "candidate_accuracy": (both_correct + only_candidate) / expected_n,
        "reference_accuracy": (both_correct + only_reference) / expected_n,
        "delta_accuracy": (only_candidate - only_reference) / expected_n,
        "paired_confidence_interval": None,
        "amendment4_eligibility": False,
        "speed_ratio_certified": False,
        "statistical_method": "NONE_DESCRIPTIVE_ONLY_UNRESOLVED_MCNEMAR_VS_WALD",
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate-audit", type=Path, required=True)
    p.add_argument("--candidate-gsm8k", type=Path, required=True)
    p.add_argument("--candidate-humaneval", type=Path, required=True)
    p.add_argument("--reference-audit", type=Path, required=True)
    p.add_argument("--reference-gsm8k", type=Path, required=True)
    p.add_argument("--reference-humaneval", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    try:
        if args.out.exists() or args.out.is_symlink() or not args.out.parent.is_dir():
            raise ValueError("comparison report target must be new exclusive path")
        candidate = json.loads(_read_file(args.candidate_audit, 2 * 1024 * 1024))
        reference = json.loads(_read_file(args.reference_audit, 2 * 1024 * 1024))
        suites = {
            name: compare(name, candidate, _read_file(candidate_path, 64 * 1024 * 1024),
                          reference, _read_file(reference_path, 64 * 1024 * 1024),
                          expected_n=SOURCE_COUNTS[name])
            for name, candidate_path, reference_path in (
                ("gsm8k", args.candidate_gsm8k, args.reference_gsm8k),
                ("humaneval", args.candidate_humaneval, args.reference_humaneval))
        }
        report = {
            "schema": "reap-v2-sourcebound-external-paired-descriptive.v1",
            "status": "PROVISIONAL_LINUX_FULL_COHORT_DESCRIPTIVE_NO_GATE",
            "audit_sha256": {
                "candidate": hashlib.sha256(_read_file(args.candidate_audit, 2 * 1024 * 1024)).hexdigest(),
                "reference": hashlib.sha256(_read_file(args.reference_audit, 2 * 1024 * 1024)).hexdigest(),
            },
            "suites": suites,
            "gen0_status": "BLOCKED", "chosen": None, "release_approved": False,
        }
        with args.out.open("x", encoding="utf8", newline="\n") as target:
            json.dump(report, target, indent=2, sort_keys=True, allow_nan=False)
            target.write("\n")
        print(json.dumps({"status": report["status"], "suites": suites}, sort_keys=True))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"source-bound paired descriptive report refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
