"""Independent internal call-trace consistency witness for original guard.v3.

Source-pinned official AutomationBench outputs contain assistant messages,
steps, and stop-reason records in addition to their reported model-call count.
Check those three independent *fields* on the 88 scored outputs and the two
receiptless archives. They are NOT independent execution attestations: all
fields originate from the same historical evaluator JSON.

No inference, original evidence writes, score promotion, or gate approval.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from chowder_batch import reap_original600_guardv3_crossledger_forensics_20261029 as ledger
from chowder_batch import reap_original600_guardv3_git_pinned_reader_20261027 as historical
from chowder_batch import reap_original600_guardv3_native_journal_witness_20261029 as native
from chowder_batch.reap_original600_guardv3_stable_snapshot_20261029 import stable_snapshot


SCHEMA = "chowder.reap-original600-guardv3-official-call-trace.v1"


class CallTraceError(ValueError):
    """A historical output cannot support its declared model-call count."""


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def inspect_official_trace(raw: bytes, *, task_name: str) -> dict:
    """Count actual message roles, steps and stop records, not just labels."""
    if not 0 < len(raw) <= ledger.MAX_OFFICIAL_JSON:
        raise CallTraceError("official call-trace output outside bounded size")
    obj = native._strict_json(raw)
    tasks, usage = obj.get("tasks"), obj.get("usage_by_task")
    if (type(tasks) is not list or len(tasks) != 1
            or type(usage) is not list or len(usage) != 1
            or type(tasks[0]) is not dict or type(usage[0]) is not dict
            or tasks[0].get("name") != task_name
            or usage[0].get("task_name") != task_name):
        raise CallTraceError("official task/usage identity invalid")
    task = tasks[0]
    calls, steps = task.get("num_model_calls"), task.get("steps")
    messages = task.get("messages")
    stops, finishes = task.get("stop_reasons"), task.get("finish_reasons")
    if (type(calls) is not int or not 0 <= calls <= 10_000
            or type(steps) is not int or not 0 <= steps <= 10_000
            or type(messages) is not list or type(stops) is not list
            or type(finishes) is not list
            or any(type(value) is not str for value in stops + finishes)):
        raise CallTraceError("official call-trace fields invalid")
    roles = []
    for message in messages:
        if (type(message) is not dict
                or type(message.get("role")) is not str
                or message.get("role") not in {"system", "user", "assistant", "tool"}
                or "content" not in message):
            raise CallTraceError("official message role/content invalid")
        roles.append(message["role"])
    assistant = roles.count("assistant")
    if calls != assistant or calls != steps or calls != len(stops):
        raise CallTraceError("official model calls disagree with assistant/step/stop trace")
    if len(finishes) > calls:
        raise CallTraceError("official finish-reason count exceeds model calls")
    return {
        "model_calls": calls, "assistant_messages": assistant,
        "steps": steps, "stop_reasons": len(stops),
        "finish_reasons": len(finishes),
        "message_role_sequence_sha256": _sha(json.dumps(
            roles, separators=(",", ":"), allow_nan=False).encode()),
    }


def reconcile_trace_snapshot(snapshot: Path, *, scored: dict, owner: dict) -> dict:
    """Verify the complete pinned cross-ledger population before tracing it."""
    cross = ledger.reconcile_snapshot(snapshot, scored=scored, owner=owner)
    if (cross["completed_scored_tasks"] != 88 or cross["strict_passes"] != 3
            or cross["receiptless_archived_official_outputs"] != 2
            or cross["receiptless_archived_reported_model_calls"] != 79
            or cross["remaining_unreconciled_count_difference"] != 18):
        raise CallTraceError("historical cross-ledger population changed")

    rows = []
    for row in scored["rows"]:
        index = row["index"]
        raw = (snapshot / f"task-{index:03d}" /
               "public-automationbench.private.json").read_bytes()
        if _sha(raw) != row["original_scorer_json_sha256"]:
            raise CallTraceError("scored original JSON SHA drift")
        trace = inspect_official_trace(raw, task_name=owner["tasks"][index]["name"])
        if trace["model_calls"] != row["model_calls"]:
            raise CallTraceError("scored trace disagrees with original scorer")
        rows.append(["scored", index, _sha(raw), trace])

    for name, expected in sorted(ledger.ARCHIVED.items()):
        index = int(ledger.UNFINISHED.fullmatch(name).group(1))
        raw = (snapshot / name / "public-automationbench.private.json").read_bytes()
        if _sha(raw) != expected:
            raise CallTraceError("receiptless archived original JSON SHA drift")
        trace = inspect_official_trace(raw, task_name=owner["tasks"][index]["name"])
        rows.append(["receiptless_unscored", index, _sha(raw), trace])

    scored_rows = [r for r in rows if r[0] == "scored"]
    archive_rows = [r for r in rows if r[0] == "receiptless_unscored"]
    if (len(scored_rows) != 88 or len(archive_rows) != 2
            or sum(r[3]["model_calls"] for r in scored_rows) != cross["scored_task_model_calls"]
            or sum(r[3]["model_calls"] for r in archive_rows) != 79):
        raise CallTraceError("source-pinned call-trace totals disagree")

    return {
        "schema": SCHEMA, "historical_source_commit": historical.COMMIT,
        "historical_owner_sha256": historical.OWNER_SHA,
        "source_crossledger_certificate_sha256": cross["crossledger_certificate_sha256"],
        "scored_tasks": len(scored_rows), "strict_passes": 3,
        "receiptless_unscored_archives": len(archive_rows),
        "scored_model_calls_correlated": sum(r[3]["model_calls"] for r in scored_rows),
        "receiptless_model_calls_correlated": sum(r[3]["model_calls"] for r in archive_rows),
        "all_assistant_messages": sum(r[3]["assistant_messages"] for r in rows),
        "all_steps": sum(r[3]["steps"] for r in rows),
        "all_stop_reasons": sum(r[3]["stop_reasons"] for r in rows),
        "all_finish_reasons": sum(r[3]["finish_reasons"] for r in rows),
        "original_unreconciled_guard_request_difference": 18,
        "official_trace_certificate_sha256": _sha(json.dumps(
            rows, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()),
        "native_http_request_response_authenticated": False,
        "independent_execution_attestation": False,
        "receiptless_archives_scored": False,
        "research_only": True, "frozen_four_slot_comparable": False,
        "gen0_status": "BLOCKED", "chosen": None, "release_approved": False,
    }


def verify_original_call_traces() -> dict:
    historical._require_assertions_enabled()
    with stable_snapshot(historical.ROOT, historical.REPO) as snapshot:
        owner_raw = (snapshot / "owner.json").read_bytes()
        if _sha(owner_raw) != historical.OWNER_SHA:
            raise CallTraceError("original owner SHA drift")
        owner = native._strict_json(owner_raw)
        _, scorer = historical.historical_modules()
        scored = scorer.inspect(snapshot)
        return reconcile_trace_snapshot(snapshot, scored=scored, owner=owner)


def main() -> int:
    try:
        print(json.dumps(verify_original_call_traces(), indent=2, sort_keys=True))
        return 0
    except (CallTraceError, ledger.CrossLedgerError, native.NativeJournalError,
            ValueError, OSError, KeyError, TypeError, AttributeError) as exc:
        print(json.dumps({"status": "REFUSED_ORIGINAL_GUARDV3_OFFICIAL_CALL_TRACE",
                          "reason": str(exc)[:350], "gen0_status": "BLOCKED",
                          "chosen": None, "release_approved": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
