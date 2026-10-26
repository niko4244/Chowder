"""Re-derive and audit a collected Q4 text-only HLE kernel run.

The kernel writes its own summary; this auditor does not trust it. It reloads the
pinned HLE population, re-derives every aggregate from the item-level rows, and
refuses the run if the rows and the kernel's summary disagree, if any row is off
the frozen canonical prefix, if any row exceeds the frozen completion ceiling, or
if the protocol digest is not the frozen one.

It never executes generated code, never calls a judge and never prints HLE
question or answer text.

Usage:
  python scripts/reap_v2_hle_collect.py --kernel-dir DIR [--out audit.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re

import pyarrow.parquet as pq

from chowder_batch import reap_hle_protocol as P

SESSION = re.compile(r"^[0-9a-f]{32}$")
COMPLETE = {"COMPLETE_Q4_HLE_TEXT_ONLY_RESEARCH_NOT_OFFICIAL_JUDGE",
            "BUDGET_TRUNCATED_Q4_HLE_TEXT_ONLY_RESEARCH_NOT_OFFICIAL_JUDGE"}


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def find(directory: pathlib.Path, pattern: str) -> pathlib.Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {pattern} in {directory}, found {len(matches)}")
    return matches[0]


def population(parquet: pathlib.Path) -> list[dict]:
    if parquet.stat().st_size != P.HLE_PARQUET_BYTES or sha256(parquet) != P.HLE_PARQUET_SHA256:
        raise ValueError("pinned HLE parquet bytes/SHA mismatch")
    rows = pq.read_table(parquet, columns=["id", "question", "image", "answer", "answer_type"]).to_pylist()
    text_rows = P.text_only_rows(rows)
    if len(text_rows) != 2158:
        raise ValueError(f"text-only HLE population changed: {len(text_rows)}")
    return P.canonical_order(text_rows)[:P.PLANNED_MAX_SAMPLES]


def audit(kernel_dir: pathlib.Path, parquet: pathlib.Path) -> dict:
    receipt = json.loads(find(kernel_dir, "reap-v2-q4-hle-textonly-research.json").read_text(encoding="utf-8"))
    problems: list[str] = []

    if receipt.get("status") not in COMPLETE:
        problems.append(f"kernel status is {receipt.get('status')!r}, not a completed run")
    if receipt.get("protocol_sha256") != P.frozen_manifest()["protocol_sha256"]:
        problems.append("receipt protocol digest is not the frozen protocol")
    session = receipt.get("session_id")
    if not isinstance(session, str) or not SESSION.match(session):
        problems.append("receipt session id is missing or malformed")
    for flag, expected in (("official_hle_score_claimed", False), ("amendment4_gate_eligible", False)):
        if receipt.get(flag) is not expected:
            problems.append(f"receipt {flag} is not {expected}")
    if receipt.get("chosen") is not None or receipt.get("release_approved") is not False:
        problems.append("receipt claims a selection or release approval")

    rows_path = find(kernel_dir, "reap-q4-hle-textonly-*.jsonl")
    summary = receipt.get("summary") or {}
    if summary.get("row_file_sha256") != sha256(rows_path):
        problems.append("row file digest disagrees with the kernel summary")

    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    planned = population(parquet)
    allowed = {row["id"]: row for row in planned}
    expected_prefix = [row["id"] for row in planned]

    seen: set[str] = set()
    for index, row in enumerate(rows):
        item_id = row.get("id")
        if item_id not in allowed:
            problems.append(f"row {index} id is outside the frozen canonical prefix")
            continue
        if item_id in seen:
            problems.append(f"row {index} repeats id {item_id}")
        seen.add(item_id)
        if row.get("eval_session_id") != session:
            problems.append(f"row {index} carries a foreign session")
        tokens = row.get("completion_tokens")
        if not isinstance(tokens, int) or tokens < 0 or tokens > P.MAX_COMPLETION_TOKENS:
            problems.append(f"row {index} completion tokens outside the frozen ceiling")
        if row.get("answer_type") != allowed[item_id]["answer_type"]:
            problems.append(f"row {index} answer_type differs from the pinned population")
        if not isinstance(row.get("correct"), bool) or not isinstance(row.get("graded"), bool):
            problems.append(f"row {index} verdict fields are not booleans")
        if row.get("truncated") is not (tokens >= P.MAX_COMPLETION_TOKENS - 1):
            problems.append(f"row {index} truncation flag disagrees with its token count")

    if [row["id"] for row in rows if row.get("id") in seen] != expected_prefix[:len(rows)]:
        problems.append("rows are not a prefix of the frozen canonical order")

    derived = P.summarize(rows, len(planned))
    for field in ("completed_n", "graded_n", "ungradeable_n", "correct", "accuracy", "truncated_n"):
        if summary.get(field) != derived[field]:
            problems.append(f"kernel summary {field}={summary.get(field)!r} but rows give {derived[field]!r}")
    if summary.get("population_text_only_n") != len(planned):
        problems.append("kernel summary population differs from the pinned text-only population")

    stats = {
        "rows_checked": len(rows),
        "mean_completion_tokens": (round(sum(r["completion_tokens"] for r in rows) / len(rows), 3) if rows else None),
        "total_completion_tokens": sum(r["completion_tokens"] for r in rows),
        "generation_wall_seconds": round(sum(r["seconds"] for r in rows), 3) if rows else None,
        "generated_tokens_per_wall_second": (
            round(derived["accuracy"] is not None and
                  sum(r["completion_tokens"] for r in rows) / sum(r["seconds"] for r in rows), 4)
            if rows and sum(r["seconds"] for r in rows) > 0 else None
        ),
        "item_errors": summary.get("item_errors"),
        "not_attempted_n": summary.get("not_attempted_n"),
    }
    return {
        "schema": "reap-v2-q4-hle-textonly-independent-audit.v1",
        "status": "AUDIT_PASSED_DETERMINISTIC_NON_OFFICIAL" if not problems else "AUDIT_REFUSED",
        "problems": problems,
        "kernel_status": receipt.get("status"),
        "session_id": session,
        "protocol_sha256": receipt.get("protocol_sha256"),
        "row_file_sha256": sha256(rows_path),
        "derived": derived,
        "derived_stats": stats,
        "population_text_only_n": len(planned),
        "official_hle_score": "UNAVAILABLE",
        "official_judge_invoked": False,
        "scientific_selection": "BLOCKED",
        "chosen": None,
        "amendment4_gate_eligible": False,
        "interpretation": (
            "Deterministic non-official match on a text-only prefix of the official split. "
            "Not an official HLE score, not comparable to vendor full-set no-tools figures, "
            "and not eligible for the frozen Gen-0 selector."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-dir", type=pathlib.Path, required=True)
    parser.add_argument("--parquet", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, default=None)
    args = parser.parse_args()
    report = audit(args.kernel_dir, args.parquet)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0 if report["status"].startswith("AUDIT_PASSED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
