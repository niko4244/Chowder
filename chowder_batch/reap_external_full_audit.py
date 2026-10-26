"""Read-only independent audit of complete, frozen-population external REAP rows.

This is NOT an Amendment 4 Windows serving-owner attestation. It cannot turn
Kaggle quality/control/speed results into eligible Gen-0 selector inputs.
Never executes generated HumanEval code: an independently isolated execution
receipt must be provided and verified separately for that score cross-check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

FROZEN_FLAGS = ["--fit", "on", "-np", "4", "-c", "16384", "--jinja"]
Q4_SHA = "2d64b786461b99a335fdbbdf7c4d3a6758b21409bce755751714ab7cd8f63bf7"
Q8_SHA = "89dcaf33b61f59c83a31b718ba98b73c33437d0320ae0cc56737a5d7e43519c6"
RANDOM_SHA = "601871e00403658708eb4eebf3b05f9dc54a93b326e31f2f87b48cb59f92241b"
SERVER_SHA = "f10032228f1d80b0f6639cd3bcdcc7194bbc6638368641cac7f1396775b5186d"
EVALUATOR_SHA = "9736714f21834a45c23599a50fa92ef3bebc6cf70967aca7e2329d26b4d73ee1"
SCORER_SHA = "d2e59aca5c0a621bac8d4979d522c9b6107f811610567d21c92a2b53960860eb"
MODELS = {
    "reap128_v2_q4": (
        "reap-v2-q4-kaggle-complete-provisional-eval.v1",
        "COMPLETE_FULL_DATA_PROVISIONAL_NOT_FROZEN_WINDOWS_GATE", "q4_sha256", Q4_SHA,
    ),
    "reap128_v2_q8": (
        "reap-v2-q8-original-full-with-measured-postload-cache-reclaim.v1",
        "COMPLETE_Q8_RECLAIM_FULL_PROVISIONAL_NOT_FROZEN_WINDOWS_GATE", "q8_sha256", Q8_SHA,
    ),
    "random128_v2_q8": (
        "reap-v2-frozen-independent-random128-full-postload-reclaim.v1",
        "COMPLETE_RANDOM128_RECLAIM_FULL_PROVISIONAL_NOT_FROZEN_WINDOWS_GATE",
        "random128_sha256", RANDOM_SHA,
    ),
}


def _check_machine(receipt: dict, arm: str) -> None:
    from chowder_batch.reap_eval import FROZEN_V2_QUALITY_SOURCES

    if arm not in MODELS or not isinstance(receipt, dict):
        raise ValueError("only the original registered Q4/Q8/random128 external model is permitted")
    schema, status, key, digest = MODELS[arm]
    if (receipt.get("schema") != schema or receipt.get("status") != status
            or receipt.get("flags") != FROZEN_FLAGS
            or receipt.get("scientific_selection") != "UNKNOWN"
            or receipt.get("not_official_windows_serving_attestation") is not True
            or not isinstance(receipt.get("session_id"), str)
            or not re.fullmatch(r"[0-9a-f]{32}", receipt["session_id"])):
        raise ValueError("unqualified/aborted/model-mixed/fabricated original external session")
    data = receipt.get("data")
    sources = receipt.get("source_sha256")
    if (not isinstance(data, dict) or data.get(key) != digest
            or data.get("server_binary_sha256") != SERVER_SHA
            or not isinstance(sources, dict)
            or sources.get("eval") != EVALUATOR_SHA
            or sources.get("score") != SCORER_SHA
            or any(sources.get(suite + "_full") != source_sha
                   for suite, (_, source_sha) in FROZEN_V2_QUALITY_SOURCES.items())):
        raise ValueError("frozen original model, evaluator or canonical input source SHA mismatch")
    if not isinstance(receipt.get("suite_results"), dict):
        raise ValueError("no complete external suite results in original serving session")


def _number(value: object, label: str, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
        raise ValueError(f"nonfinite, negative or invalid {label}")


def _check_rows(suite: str, source: list[dict], rows: list[dict], session: str) -> dict:
    from chowder.evaluators.scoring import score
    from chowder_batch.reap_eval import _trigram_ratio

    if suite not in ("gsm8k", "humaneval") or len(rows) != len(source) or not source:
        raise ValueError("full canonical cohort missing rows")
    ids = {item["id"]: item for item in source}
    if len(ids) != len(source) or len({row.get("id") for row in rows}) != len(rows):
        raise ValueError("duplicate canonical cohort IDs")
    if {row.get("id") for row in rows} != set(ids):
        raise ValueError("missing/foreign cohort ID or wrong population")
    correct = terminated = total_tokens = 0
    for row in rows:
        if (row.get("eval_session_id") != session
                or type(row.get("correct")) is not bool
                or type(row.get("terminated")) is not bool
                or not isinstance(row.get("text"), str)
                or type(row.get("completion_tokens")) is not int):
            raise ValueError("stale session or impossible scored row fields")
        for field in ("completion_tokens", "seconds", "predicted_per_second", "trigram_ratio"):
            _number(row.get(field), field, nullable=field == "predicted_per_second")
        if not 0 <= row["trigram_ratio"] <= 1:
            raise ValueError("out-of-range trigram ratio")
        if row["trigram_ratio"] != round(_trigram_ratio(row["text"]), 4):
            raise ValueError("source output degeneration metric does not recompute")
        if row["correct"] and not row["terminated"]:
            raise ValueError("source evaluator cannot score an unterminated answer correct")
        if suite == "gsm8k":
            expected = bool(row["terminated"] and score(
                row["text"], ids[row["id"]]["expected"], "final_number_match") == 1.0)
            if row["correct"] != expected:
                raise ValueError("GSM8K original final-answer scorer cannot reproduce saved row")
        correct += row["correct"]
        terminated += row["terminated"]
        total_tokens += row["completion_tokens"]
    return {"n": len(rows), "correct": correct, "terminated": terminated,
            "accuracy": correct / len(rows), "completion_tokens": total_tokens}


def _read_json(path: Path, *, limit: int) -> tuple[dict, str]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError("external private evidence file missing, linked or unbounded")
    raw = path.read_bytes()
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError("external provenance JSON is not an object")
    return obj, hashlib.sha256(raw).hexdigest()


def _read_jsonl(path: Path) -> tuple[list[dict], str]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("unbounded/missing/symlink original scored cohort")
    raw = path.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("unflushed original output JSONL")
    rows = [json.loads(line) for line in raw.splitlines()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("original scored rows must be structured JSON objects")
    return rows, hashlib.sha256(raw).hexdigest()


def audit(machine: Path, gsm8k: Path, humaneval: Path, arm: str,
          independent_humaneval: Path | None = None) -> dict:
    from chowder_batch.reap_eval import FROZEN_V2_QUALITY_SOURCES, _digest, _load

    receipt, receipt_sha = _read_json(machine, limit=2 * 1024 * 1024)
    _check_machine(receipt, arm)
    if set(receipt["suite_results"]) != {"gsm8k", "humaneval"}:
        raise ValueError("no complete, unique full-suite machine summary")
    session = receipt["session_id"]
    verified = {}
    for suite, path in (("gsm8k", gsm8k), ("humaneval", humaneval)):
        original = _load(suite)
        expected_n, original_sha = FROZEN_V2_QUALITY_SOURCES[suite]
        if len(original) != expected_n or _digest(original) != original_sha:
            raise ValueError("locally loaded frozen input population does not match preregistration")
        rows, row_sha = _read_jsonl(path)
        result = _check_rows(suite, original, rows, session)
        summary = receipt["suite_results"][suite]
        if (not isinstance(summary, dict) or summary.get("file_sha256") != row_sha
                or summary.get("session_id") != session or summary.get("n") != expected_n):
            raise ValueError("original complete session summary and byte-level rows disagree")
        for field, calculated in (("accuracy", result["accuracy"]),
                                  ("terminated_rate", result["terminated"] / expected_n),
                                  ("mean_completion_tokens", result["completion_tokens"] / expected_n)):
            _number(summary.get(field), field)
            if not math.isclose(summary[field], calculated, abs_tol=1e-10, rel_tol=0):
                raise ValueError(f"original external suite summary {field} fails independent replay")
        wall = summary.get("suite_elapsed_wall_seconds")
        _number(wall, "actual suite wall seconds")
        if wall <= 0:
            raise ValueError("no measured complete-cohort wall time")
        verified[suite] = {**result, "row_sha256": row_sha,
                           "suite_wall_seconds": wall,
                           "generated_tokens_per_wall_second": result["completion_tokens"] / wall}
    replay = "PENDING_INDEPENDENT_LOW_PRIVILEGE_CODE_EXECUTION"
    if independent_humaneval is not None:
        independent, replay_sha = _read_json(independent_humaneval, limit=2 * 1024 * 1024)
        if (independent.get("status") != "PASS_INDEPENDENT_ISOLATED_HUMANEVAL_FULL_164_PROVISIONAL_LINUX"
                or independent.get("original_row_sha256") != verified["humaneval"]["row_sha256"]
                or independent.get("session_id") != session
                or independent.get("independent_full_rows") != 164
                or independent.get("matched_recorded_scores") != 164
                or independent.get("correct") != verified["humaneval"]["correct"]):
            raise ValueError("independent HumanEval re-execution outcome/session/row hash mismatch")
        replay = f"PASS_FULL_164_LOW_PRIVILEGE_REEXECUTION_SHA256:{replay_sha}"
    return {
        "schema": "chowder.reap-v2-original-population-external-independent-audit.v1",
        "status": "COMPLETE_LINUX_COHORT_VERIFIED_PROVISIONAL_NOT_WINDOWS_GATE",
        "model": arm,
        "model_sha256": MODELS[arm][3],
        "session_id": session,
        "original_machine_receipt_sha256": receipt_sha,
        "human_eval_independent_execution": replay,
        "suites": verified,
        "original_windows_gate_eligible": False,
        "speed_ratio_certified": False,
        "gen0_status": "BLOCKED",
        "chosen": None,
        "release_approved": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", type=Path, required=True)
    parser.add_argument("--gsm8k", type=Path, required=True)
    parser.add_argument("--humaneval", type=Path, required=True)
    parser.add_argument("--model", choices=list(MODELS), required=True)
    parser.add_argument("--independent-humaneval", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.out.is_symlink() or args.out.exists() or not args.out.parent.is_dir():
            raise ValueError("original external evidence audit output must be a new exclusive path")
        result = audit(args.machine, args.gsm8k, args.humaneval,
                       args.model, args.independent_humaneval)
        with args.out.open("x", encoding="utf8", newline="\n") as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"status": result["status"], "model": result["model"],
                          "session_id": result["session_id"],
                          "suites": result["suites"],
                          "human_eval_independent_execution": result["human_eval_independent_execution"]}))
        return 0
    except (ValueError, OSError, TypeError, KeyError, UnicodeError) as exc:
        print(f"external original-frozen cohort audit refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
