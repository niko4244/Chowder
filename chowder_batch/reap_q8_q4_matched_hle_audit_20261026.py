"""Independent read-only exact-ID paired HLE research audit (NOT official HLE).

Fail closed until a Q8 private 128-row cohort has actually completed on the
same data, prompts, sampler, scorer and protocol manifest as the completed Q4
cohort. Only aggregate correctness leaves the private result directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from chowder_batch import reap_q8_matched_hle_kaggle_20261026 as pinned


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json_file(p: Path) -> dict:
    if not p.is_file() or p.is_symlink() or p.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("missing or oversized private HLE research receipt")
    parsed = json.loads(p.read_bytes(), parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
    if type(parsed) is not dict:
        raise ValueError("HLE research receipt is not an object")
    return parsed


def _cohort_rows(dir_path: Path, pattern: str, expected_sha: str | None = None) -> tuple[list[dict], str]:
    candidates = list(dir_path.glob(pattern))
    if len(candidates) != 1 or not candidates[0].is_file() or candidates[0].is_symlink():
        raise ValueError("HLE result population missing or ambiguous")
    raw = candidates[0].read_bytes()
    if not 0 < len(raw) <= 4 * 1024 * 1024:
        raise ValueError("HLE population missing or unbounded")
    digest = sha(raw)
    if expected_sha is not None and digest != expected_sha:
        raise ValueError("HLE original row SHA does not match its bound receipt")
    rows = [json.loads(line, parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
            for line in raw.splitlines() if line.strip()]
    if len(rows) != 128 or any(type(x) is not dict for x in rows):
        raise ValueError("HLE research requires all 128 completed source rows")
    return rows, digest


def compare_rows(q4_rows: list[dict], q8_rows: list[dict], *, expected_ids: set[str]) -> dict:
    """Exact source-ID paired comparison, no heuristic matches or missing rows."""
    if len(q4_rows) != len(q8_rows) or len(q4_rows) != len(expected_ids):
        raise ValueError("paired HLE population sizes differ")
    arms = []
    for rows in (q4_rows, q8_rows):
        by_id = {}
        sessions = set()
        for row in rows:
            if (type(row) is not dict or type(row.get("id")) is not str
                    or type(row.get("correct")) is not bool
                    or row.get("graded") is not True
                    or row.get("truncated") is not False
                    or type(row.get("completion_tokens")) is not int
                    or row["completion_tokens"] < 0
                    or type(row.get("answer_type")) is not str
                    or row["answer_type"] not in ("exactMatch", "multipleChoice")
                    or type(row.get("eval_session_id")) is not str
                    or not row["eval_session_id"]):
                raise ValueError("HLE row is ungraded, truncated or invalid")
            if row["id"] in by_id:
                raise ValueError("HLE duplicated source ID")
            by_id[row["id"]] = row
            sessions.add(row["eval_session_id"])
        if set(by_id) != expected_ids or len(sessions) != 1:
            raise ValueError("HLE source-ID or serving-session match failed")
        arms.append(by_id)
    q4, q8 = arms
    partition = {"both_correct": 0, "q8_only_correct": 0,
                 "q4_only_correct": 0, "both_incorrect": 0}
    for item_id in expected_ids:
        a, b = q4[item_id], q8[item_id]
        if a["answer_type"] != b["answer_type"]:
            raise ValueError("HLE per-question answer-type changed")
        if a["correct"] and b["correct"]:
            partition["both_correct"] += 1
        elif b["correct"]:
            partition["q8_only_correct"] += 1
        elif a["correct"]:
            partition["q4_only_correct"] += 1
        else:
            partition["both_incorrect"] += 1
    n = len(expected_ids)
    q4_correct = partition["both_correct"] + partition["q4_only_correct"]
    q8_correct = partition["both_correct"] + partition["q8_only_correct"]
    return {"population": n, "q4_correct": q4_correct, "q8_correct": q8_correct,
            "q4_accuracy": q4_correct / n, "q8_accuracy": q8_correct / n,
            "q8_minus_q4_percentage_points": 100 * (q8_correct - q4_correct) / n,
            "partition": partition,
            "metric": "non_official_deterministic_normalized_text_only_HLE_match",
            "official_hle_judge_score": "UNAVAILABLE", "gen0_status": "BLOCKED",
            "chosen": None, "release_approved": False}


def audit(q8_dir: Path, *, q4_dir: Path = pinned.Q4_ROWS_DIR) -> dict:
    if not q8_dir.is_dir() or q8_dir.is_symlink():
        raise ValueError("exact private Q8 Kaggle output folder required")
    q4_receipt = _json_file(q4_dir / "reap-v2-q4-hle-textonly-research.json")
    q8_receipt = _json_file(q8_dir / "reap-v2-q8-matched128-hle-textonly-research.json")
    if (q4_receipt.get("status") != "BUDGET_TRUNCATED_Q4_HLE_TEXT_ONLY_RESEARCH_NOT_OFFICIAL_JUDGE"
            or q8_receipt.get("status") != "COMPLETE_Q8_MATCHED_HLE_TEXT_ONLY_RESEARCH_NOT_OFFICIAL_JUDGE"
            or q4_receipt.get("protocol_sha256") != q8_receipt.get("protocol_sha256")
            or q4_receipt.get("protocol_file_sha256") != q8_receipt.get("protocol_file_sha256")
            or q4_receipt.get("protocol") != q8_receipt.get("protocol")
            or q4_receipt.get("protocol_file_sha256") != "cbacf0bb2215fce041ca62e991002d2772bee04891bc5e1773d3b0e2c44c49b2"
            or q4_receipt.get("protocol_sha256") != "ebdb19e818257c21335889f9bea51c9939293444d315ba0211e21d2c599e70d8"
            or q4_receipt.get("data", {}).get("q4_gguf_sha256") != "2d64b786461b99a335fdbbdf7c4d3a6758b21409bce755751714ab7cd8f63bf7"
            or q8_receipt.get("data", {}).get("q8_gguf_sha256") != pinned.Q8_SHA256
            or q8_receipt.get("matched_q4_row_sha256") != pinned.Q4_ROWS_SHA256
            or q8_receipt.get("official_hle_score_claimed") is not False
            or q4_receipt.get("official_hle_score_claimed") is not False
            or q8_receipt.get("scientific_selection") != "BLOCKED"):
        raise ValueError("Q4↔Q8 original HLE protocol/model/science receipt mismatch")
    q4_rows, q4_hash = _cohort_rows(q4_dir, "reap-q4-hle-textonly-*.jsonl", pinned.Q4_ROWS_SHA256)
    q8_rows, q8_hash = _cohort_rows(q8_dir, "reap-q8-matched128-hle-textonly-*.jsonl")
    q8_summary_files = list(q8_dir.glob("reap-q8-matched128-hle-textonly-*-summary.json"))
    if len(q8_summary_files) != 1:
        raise ValueError("single Q8 paired summary missing")
    summary = _json_file(q8_summary_files[0])
    if (summary.get("row_file_sha256") != q8_hash
            or summary.get("completed_n") != 128
            or summary.get("planned_n") != 128
            or summary.get("not_attempted_n") != 0
            or summary.get("graded_n") != 128
            or summary.get("ungradeable_n") != 0
            or summary.get("truncated_n") != 0
            or summary.get("item_errors") != []
            or summary.get("correct") != sum(x["correct"] is True for x in q8_rows)
            or summary != q8_receipt.get("summary")):
        raise ValueError("Q8 matched HLE complete-summary integrity mismatch")
    result = compare_rows(q4_rows, q8_rows, expected_ids=set(pinned._source_bound_rows()))
    if result["q4_correct"] != 3:
        raise ValueError("pinned original completed Q4 correct count changed")
    result["q4_row_sha256"] = q4_hash
    result["q8_row_sha256"] = q8_hash
    result["q8_model_sha256"] = pinned.Q8_SHA256
    result["protocol_sha256"] = q8_receipt["protocol_sha256"]
    result["status"] = "INDEPENDENT_MATCHED_128_RESEARCH_COUNTS_VALIDATED_NOT_OFFICIAL_HLE"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q8", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = audit(args.q8)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        result = {"status": "BLOCKED_UNVERIFIED_HLE_PAIR", "reason": type(exc).__name__,
                  "gen0_status": "BLOCKED", "chosen": None, "release_approved": False}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("INDEPENDENT_MATCHED_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
