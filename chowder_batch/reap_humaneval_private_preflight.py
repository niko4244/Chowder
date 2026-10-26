"""Read-only, no-execution source admission for private 164-task HumanEval replay.

This does NOT execute model-generated Python, establish a secure sandbox,
attest Windows equivalence or turn Linux-provisional evidence into a Gen-0 gate.
It requires the original independently generated full external audit, then
binds each future Q8/control 164-task replay to that audit's exact SHA,
server session, native receipt, model, canonical source and frozen scorer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from chowder_batch.reap_external_full_audit import (
    EVALUATOR_SHA, MODELS, Q8_SHA, RANDOM_SHA, SCORER_SHA,
    _check_machine, _check_postload_resource_evidence, _strict_json,
)

CANONICAL_HUMANEVAL_RAW_SHA = "5144d31cf337b25ce262b780ed103b128b1e9d289d3c0d15f4fe34933bb705d5"
ELIGIBLE_ARMS = ("reap128_v2_q8", "random128_v2_q8")
AUDIT_SCHEMA = "chowder.reap-v2-original-population-external-independent-audit.v1"
AUDIT_STATUS = "COMPLETE_LINUX_COHORT_VERIFIED_PROVISIONAL_NOT_WINDOWS_GATE"
NO_REPLAY_YET = "PENDING_INDEPENDENT_LOW_PRIVILEGE_CODE_EXECUTION"


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_file(path: Path, maximum: int) -> bytes:
    if (not path.is_absolute() or path.is_symlink() or not path.is_file()
            or path.stat().st_size == 0 or path.stat().st_size > maximum):
        raise ValueError("private source path missing, linked, relative, empty or unbounded")
    return path.read_bytes()


def _json_object(raw: bytes) -> dict:
    try:
        result = _strict_json(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("private source JSON invalid or unbounded") from exc
    if not isinstance(result, dict):
        raise ValueError("expected a private source JSON object")
    return result


def preflight(*, arm: str, machine: dict, machine_sha: str,
              trusted_audit: dict, audit_sha: str,
              canonical_raw: bytes, rows_raw: bytes,
              evaluator_sha: str, scorer_sha: str) -> dict:
    """Verify admission with no generation, subprocess, or file modification."""
    if arm not in ELIGIBLE_ARMS:
        raise ValueError("only original complete Q8 or independent random128 replay is eligible")
    if evaluator_sha != EVALUATOR_SHA or scorer_sha != SCORER_SHA:
        raise ValueError("original frozen HumanEval evaluator/scorer SHA mismatch")
    if not isinstance(machine, dict) or not isinstance(trusted_audit, dict):
        raise ValueError("native machine/audit provenance required")
    if (not isinstance(machine_sha, str) or not re.fullmatch(r"[a-f0-9]{64}", machine_sha)
            or not isinstance(audit_sha, str) or not re.fullmatch(r"[a-f0-9]{64}", audit_sha)):
        raise ValueError("original independently hashed receipts missing")
    _check_machine(machine, arm)
    verified_resource = _check_postload_resource_evidence(machine)
    session = machine.get("session_id")
    if (trusted_audit.get("schema") != AUDIT_SCHEMA
            or trusted_audit.get("status") != AUDIT_STATUS
            or trusted_audit.get("model") != arm
            or trusted_audit.get("model_sha256") != MODELS[arm][3]
            or trusted_audit.get("original_machine_receipt_sha256") != machine_sha
            or trusted_audit.get("session_id") != session
            or trusted_audit.get("original_windows_gate_eligible") is not False
            or trusted_audit.get("speed_ratio_certified") is not False
            or trusted_audit.get("gen0_status") != "BLOCKED"
            or trusted_audit.get("chosen") is not None
            or trusted_audit.get("release_approved") is not False
            or trusted_audit.get("human_eval_independent_execution") != NO_REPLAY_YET
            or trusted_audit.get("original_measured_postload_resource_witness") != verified_resource):
        raise ValueError("trusted independent full-audit provenance is not bound to original model/session/resources")
    native = trusted_audit.get("original_native_cuda_placement")
    if (not isinstance(native, dict)
            or native.get("native_gpu_offloaded_layers") != "41/41"
            or native.get("n_seq_max") != 4
            or native.get("n_ctx") != 16384
            or native.get("n_ctx_seq") != 4096
            or native.get("total_kv_buffer_mib") != 320.0
            or native.get("loopback_bound") is not True):
        raise ValueError("independently attested frozen native four-way CUDA model placement missing")
    try:
        suites = trusted_audit["suites"]
        orig_suites = machine["suite_results"]
        if (set(suites) != {"gsm8k", "humaneval"}
                or set(orig_suites) != {"gsm8k", "humaneval"}):
            raise ValueError("original full paired suite coverage missing")
        for suite, required in (("gsm8k", 1319), ("humaneval", 164)):
            if (type(suites[suite]["n"]) is not int or suites[suite]["n"] != required
                    or type(orig_suites[suite]["n"]) is not int
                    or orig_suites[suite]["n"] != required
                    or orig_suites[suite]["session_id"] != session
                    or suites[suite]["row_sha256"] != orig_suites[suite]["file_sha256"]):
                raise ValueError("original full immutable row count/session/hash mismatch")
        summary = suites["humaneval"]
        correct = summary["correct"]
        terminated = summary["terminated"]
        if (type(correct) is not int or type(terminated) is not int
                or not 0 <= correct <= terminated <= 164):
            raise ValueError("invalid original independently audited HumanEval results")
    except (KeyError, TypeError) as exc:
        raise ValueError("missing complete independent suite source evidence") from exc

    if (not isinstance(canonical_raw, bytes) or len(canonical_raw) > 2 * 1024 * 1024
            or _digest(canonical_raw) != CANONICAL_HUMANEVAL_RAW_SHA):
        raise ValueError("original canonical HumanEval raw SHA or size mismatch")
    if (not isinstance(rows_raw, bytes) or not 0 < len(rows_raw) <= 64 * 1024 * 1024
            or not rows_raw.endswith(b"\n")
            or _digest(rows_raw) != summary["row_sha256"]):
        raise ValueError("original complete independent audited result row SHA/format mismatch")
    try:
        canon = _strict_json(canonical_raw)
        rows = [_strict_json(line) for line in rows_raw.splitlines()]
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("original full HumanEval source/result cannot be decoded") from exc
    if (not isinstance(canon, list) or len(canon) != 164
            or len(rows) != 164 or not all(isinstance(row, dict) for row in rows)):
        raise ValueError("164 canonical original HumanEval source/rows missing")
    canonical_ids = {f"HumanEval/{i}" for i in range(164)}
    if (not all(isinstance(row, dict) for row in canon)
            or {row.get("id") for row in canon} != canonical_ids
            or not all(isinstance(row.get(key), str) and row[key]
                       for row in canon for key in ("header", "test", "entry_point"))):
        raise ValueError("original canonical 164 HumanEval tests missing or ambiguous")
    seen: set[str] = set()
    by_id: dict[str, dict] = {}
    for row in rows:
        ident = row.get("id")
        if (not isinstance(ident, str) or ident in seen
                or ident not in canonical_ids
                or row.get("eval_session_id") != session
                or type(row.get("correct")) is not bool
                or type(row.get("terminated")) is not bool
                or not isinstance(row.get("text"), str)
                or (row["correct"] and not row["terminated"])):
            raise ValueError("original HumanEval row/session/outcome integrity missing")
        seen.add(ident)
        by_id[ident] = row
    if (seen != canonical_ids
            or sum(row["correct"] for row in rows) != correct
            or sum(row["terminated"] for row in rows) != terminated):
        raise ValueError("original independently audited HumanEval aggregates or IDs mismatch")
    bools = [{"id": item["id"], "correct": by_id[item["id"]]["correct"],
              "terminated": by_id[item["id"]]["terminated"]} for item in canon]
    return {
        "schema": "chowder.reap-v2-independent-private-humaneval-preflight.v1",
        "status": "PREFLIGHT_PRIVATE_164_SOURCE_BOUND_SANDBOX_REQUIRED",
        "arm": arm, "model_sha256": MODELS[arm][3],
        "session_id": session, "machine_sha256": machine_sha,
        "trusted_full_audit_sha256": audit_sha,
        "canonical_humaneval_raw_sha256": _digest(canonical_raw),
        "original_humaneval_row_sha256": _digest(rows_raw),
        "original_evaluator_sha256": evaluator_sha,
        "original_scorer_sha256": scorer_sha,
        "ordered_original_boolean_outcome_sha256": _digest(
            json.dumps(bools, sort_keys=True, separators=(",", ":")).encode()
        ),
        "source_rows": 164, "reported_correct": correct,
        "reported_terminated": terminated,
        "candidate_code_executed": False,
        "sandbox_verified": False,
        "original_windows_gate_eligible": False,
        "gen0_status": "BLOCKED", "chosen": None, "release_approved": False,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=ELIGIBLE_ARMS, required=True)
    p.add_argument("--machine", type=Path, required=True)
    p.add_argument("--trusted-full-audit", type=Path, required=True)
    p.add_argument("--humaneval", type=Path, required=True)
    p.add_argument("--canonical-humaneval", type=Path, required=True)
    p.add_argument("--original-evaluator", type=Path, required=True)
    p.add_argument("--original-scorer", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    try:
        if a.out.exists() or a.out.is_symlink() or not a.out.parent.is_dir():
            raise ValueError("private preflight output path must be new and exclusive")
        machine_bytes = _read_file(a.machine, 2 * 1024 * 1024)
        audit_bytes = _read_file(a.trusted_full_audit, 2 * 1024 * 1024)
        response = preflight(
            arm=a.arm,
            machine=_json_object(machine_bytes), machine_sha=_digest(machine_bytes),
            trusted_audit=_json_object(audit_bytes), audit_sha=_digest(audit_bytes),
            canonical_raw=_read_file(a.canonical_humaneval, 2 * 1024 * 1024),
            rows_raw=_read_file(a.humaneval, 64 * 1024 * 1024),
            evaluator_sha=_digest(_read_file(a.original_evaluator, 2 * 1024 * 1024)),
            scorer_sha=_digest(_read_file(a.original_scorer, 2 * 1024 * 1024)),
        )
        with a.out.open("x", encoding="utf8", newline="\n") as target:
            json.dump(response, target, sort_keys=True, indent=2, allow_nan=False)
            target.write("\n")
        print(json.dumps({"status": response["status"], "arm": a.arm,
                          "candidate_code_executed": False}, sort_keys=True))
        return 0
    except (ValueError, OSError, TypeError, KeyError, UnicodeError) as exc:
        print(f"independent original HumanEval pre-execution admission refused: {exc}",
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
