"""Did the model ever address the record the assertion grades?

Phase 36/37 left one live explanation for the zero-scoring note and message
families: the write is well-formed but aimed at the wrong record, so no payload
repair can recover it. A pass rate cannot distinguish that from "right record,
wrong content", because both score zero.

This audit reads only the official evaluator exports (no inference, no scoring
change) and classifies each graded failure by whether the identifier the
assertion grades was ever present in the model's own requests:

- ``target_requested``      the required id appears in at least one ``api_fetch``
                            call, so the record *was* addressed.
- ``target_never_requested`` the id appears in no request at all, so the record
                            was never addressed.

It is deliberately coarse. A substring hit does not prove the call was the right
operation on the right collection, only that the identifier was in play; the
class is evidence about *targeting*, not proof about intent.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

# Assertion parameters that name the record/message the assertion grades.
TARGET_KEYS = ("record_id", "parent_id", "campaign_id", "contact_id", "to")

REQUESTED = "target_requested"
NEVER = "target_never_requested"


def request_corpus(task: Mapping[str, Any]) -> str:
    """Every url/params/body string the model sent, concatenated."""
    parts: list[str] = []
    for message in task.get("messages") or []:
        for call in message.get("tool_calls") or []:
            try:
                entry = json.loads(call) if isinstance(call, str) else call
            except ValueError:
                continue
            if not isinstance(entry, dict):
                continue
            raw = entry.get("arguments")
            try:
                arguments = json.loads(raw) if isinstance(raw, str) else (raw or {})
            except ValueError:
                # Keep the damaged payload: an id inside it still counts as addressed.
                parts.append(str(raw))
                continue
            if isinstance(arguments, dict):
                parts.extend(str(v) for v in arguments.values() if v is not None)
            else:
                parts.append(str(arguments))
    return "\n".join(parts)


def classify_task(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Classify each graded assertion of one task by target addressing."""
    corpus = request_corpus(task)
    rows: list[dict[str, Any]] = []
    for assertion in task.get("assertion_results") or []:
        if assertion.get("excluded"):
            continue
        params = assertion.get("params") or {}
        targets = [(key, str(params[key])) for key in TARGET_KEYS if params.get(key)]
        if not targets:
            continue
        addressed = [(key, value) for key, value in targets if value in corpus]
        rows.append({
            "task": task.get("name"),
            "assertion": assertion.get("type"),
            "passed": bool(assertion.get("passed")),
            "targets": [f"{k}={v}" for k, v in targets],
            "missing_targets": [f"{k}={v}" for k, v in targets if value_missing(corpus, v)],
            "class": REQUESTED if addressed else NEVER,
        })
    return rows


def value_missing(corpus: str, value: str) -> bool:
    return value not in corpus


def summarize(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    failed = [r for r in rows if not r["passed"]]
    by_type: dict[str, Counter[str]] = defaultdict(Counter)
    for row in failed:
        by_type[str(row["assertion"])][str(row["class"])] += 1
    return {
        "graded_with_target": len(list(rows)),
        "failed_with_target": len(failed),
        "failed_target_never_requested": sum(1 for r in failed if r["class"] == NEVER),
        "failed_target_requested": sum(1 for r in failed if r["class"] == REQUESTED),
        "by_assertion": {k: dict(v) for k, v in sorted(by_type.items())},
    }


def audit_directory(path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    tasks = 0
    for item in sorted(path.glob("task-*")):
        raw = item / "public-automationbench.private.json"
        if not raw.is_file():
            continue
        payload = json.loads(raw.read_bytes())
        for task in payload.get("tasks") or []:
            tasks += 1
            rows.extend(classify_task(task))
    summary = summarize(rows)
    summary["tasks"] = tasks
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(audit_directory(args.out_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
