"""Recheck immutable original Q8/Q4 rows; seal a PRIVATE item-level bit witness.

No model launch, no generated-code execution and no release/Gen-0 selection.
The witnessed scores are the original scorer's booleans, not a new HumanEval
sandbox execution certificate or official frontier leaderboard scores.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


SOURCES = {
    "q8": {
        "session": "95876063255e45159f57e29306e0e885",
        "base": Path("F:/reap-v2-q8-postload-full-complete-original-evidence-20261026-phase4"),
        "prefix": "reap-q8-reclaim-full-95876063255e45159f57e29306e0e885",
        "sha": {"gsm8k": "9ff2b418fecabdc419d94b2a06f6d9b115bbbd4bfba59ccae2d0d9f81e8ef749",
                "humaneval": "cd3c9d1577611a032b36e69a3f96bd272d118819074352503f143bcd68cac301"},
        "audit": Path("F:/reap-v2-phase26-original-q8-source-audit-20261026.json"),
    },
    "q4": {
        "session": "a2a6b83e345145839cc333561b2f3baf",
        "base": Path("F:/reap-v2-kaggle-q4-full-complete-private-evidence-20261026"),
        "prefix": "reap-v2-a2a6b83e345145839cc333561b2f3baf",
        "sha": {"gsm8k": "3b09d337f567729192e9c302ec698cf1e34d7451538847939e3bf9a36a9557db",
                "humaneval": "92ec6439be77a7dbeb1a80448a0bb664e3cb6113a9cc0d356f7dbe300db69c43"},
        "audit": Path("F:/reap-v2-phase26-original-q4-source-audit-20261026.json"),
    },
}
SIZES = {"gsm8k": 1319, "humaneval": 164}
PRIVATE_PARENT = Path("F:/reap-v2-automationbench-private-20261026")


def file_sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def reverify_rows(raw: bytes, *, digest: str, session: str, suite: str) -> list[dict]:
    """Reject forged IDs, row aliases, nonfinite numerics and wrong sessions."""
    if file_sha(raw) != digest or not raw.endswith(b"\n"):
        raise ValueError("source bytes/hash or final newline changed")
    expected = SIZES[suite]
    items = []
    for line in raw.splitlines():
        row = json.loads(line, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite")))
        if not isinstance(row, dict) or set(row) != {
            "id", "correct", "terminated", "trigram_ratio", "completion_tokens",
            "seconds", "predicted_per_second", "text", "eval_session_id",
        }:
            raise ValueError("unexpected original row fields")
        if (type(row["correct"]) is not bool or type(row["terminated"]) is not bool
                or type(row["completion_tokens"]) is not int or row["completion_tokens"] < 0
                or not isinstance(row["text"], str) or row["eval_session_id"] != session
                or not isinstance(row["id"], str)):
            raise ValueError("invalid original booleans/row/session")
        for key in ("seconds", "predicted_per_second", "trigram_ratio"):
            value = row[key]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("invalid/nonfinite original numeric row")
        # Never leak completion text into newly generated tables.
        items.append({"id": row["id"], "correct": row["correct"],
                      "terminated": row["terminated"], "completion_tokens": row["completion_tokens"],
                      "seconds": row["seconds"]})
    if len(items) != expected or len({r["id"] for r in items}) != expected:
        raise ValueError("original complete unique item population absent")
    if suite == "gsm8k" and {r["id"] for r in items} != {f"gsm8k-{i}" for i in range(1319)}:
        raise ValueError("GSM8K exact original item indexing changed")
    if suite == "humaneval":
        if {r["id"] for r in items} != {f"HumanEval/{i}" for i in range(164)}:
            raise ValueError("HumanEval exact 164 item indexing changed")
    return items


def load_source(model: str, suite: str) -> tuple[list[dict], dict]:
    spec = SOURCES[model]
    audit_file = spec["audit"]
    if audit_file.is_symlink() or not audit_file.is_file():
        raise ValueError("source bound independent audit absent")
    audit = json.loads(audit_file.read_bytes())
    expected = audit.get("suites", {}).get(suite, {})
    if (audit.get("gen0_status") != "BLOCKED" or audit.get("release_approved") is not False
            or audit.get("session_id") != spec["session"]
            or expected.get("n") != SIZES[suite]
            or expected.get("row_sha256") != spec["sha"][suite]):
        raise ValueError("historical independent audit source identity changed")
    path = spec["base"] / f"{spec['prefix']}-{suite}.jsonl"
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("private source row file missing/unbounded")
    rows = reverify_rows(path.read_bytes(), digest=spec["sha"][suite],
                         session=spec["session"], suite=suite)
    correct = sum(r["correct"] for r in rows)
    terminated = sum(r["terminated"] for r in rows)
    tokens = sum(r["completion_tokens"] for r in rows)
    if (correct != expected.get("correct") or terminated != expected.get("terminated")
            or tokens != expected.get("completion_tokens")):
        raise ValueError("recount differs from original independent scored audit")
    return rows, {"n": len(rows), "correct": correct, "terminated": terminated,
                  "completion_tokens": tokens, "wall_seconds": expected["suite_wall_seconds"],
                  "tokens_per_wall_second": expected["generated_tokens_per_wall_second"],
                  "source_sha256": spec["sha"][suite], "session": spec["session"]}


def paired(rows8: list[dict], rows4: list[dict]) -> dict:
    a = {r["id"]: r for r in rows8}
    b = {r["id"]: r for r in rows4}
    if set(a) != set(b):
        raise ValueError("not a matched original source population")
    counts = {"both_correct": 0, "both_wrong": 0, "q8_only": 0, "q4_only": 0}
    for name in a:
        c8, c4 = a[name]["correct"], b[name]["correct"]
        key = "both_correct" if c8 and c4 else (
            "both_wrong" if not c8 and not c4 else "q8_only" if c8 else "q4_only")
        counts[key] += 1
    return counts


def run(out: Path) -> dict:
    parent = PRIVATE_PARENT.resolve(strict=False)
    candidate = out.resolve(strict=False)
    if (not out.is_absolute() or out.exists() or out.is_symlink()
            or candidate == parent or not candidate.is_relative_to(parent)
            or not out.parent.is_dir() or out.parent.is_symlink()):
        raise ValueError("exclusive private result directory beneath private F: root required")
    cohorts = {}
    public = {}
    for model in ("q8", "q4"):
        for suite in SIZES:
            rows, summary = load_source(model, suite)
            cohorts[(model, suite)] = rows
            public[f"{model}_{suite}"] = summary
    matched = paired(cohorts["q8", "gsm8k"], cohorts["q4", "gsm8k"])
    if matched != {"both_correct": 1198, "both_wrong": 68, "q8_only": 22, "q4_only": 31}:
        raise ValueError("frozen paired original 1319-item witness changed")
    out.mkdir(mode=0o700)
    path = out / "original-q8-q4-item-level.private.csv"
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["suite", "id", "q8_correct", "q8_terminated", "q8_tokens",
                         "q4_correct", "q4_terminated", "q4_tokens"])
        for suite in SIZES:
            b = {r["id"]: r for r in cohorts["q4", suite]}
            for a in cohorts["q8", suite]:
                c = b[a["id"]]
                writer.writerow([suite, a["id"], int(a["correct"]), int(a["terminated"]),
                                 a["completion_tokens"], int(c["correct"]),
                                 int(c["terminated"]), c["completion_tokens"]])
    receipt = {"status": "ORIGINAL_LINUX_COMPLETE_REVERIFIED_DESCRIPTIVE_ONLY",
               "source": "original frozen evaluator booleans, not new inference",
               "human_eval_independent_sandbox": "NOT_PROVEN",
               "full_original_item_rows_per_model": sum(SIZES.values()),
               "paired_gsm8k": matched, "cohorts": public,
               "private_item_witness_sha256": file_sha(path.read_bytes()),
               "gen0_status": "BLOCKED", "chosen": None, "release_approved": False}
    (out / "original-q8-q4-recount.private.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.out)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
