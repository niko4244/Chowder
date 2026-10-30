"""Source-locked historical guard.v3 cross-ledger call accounting (CPU only).

The six guard journals count requests, while original AutomationBench reports
model calls per completed task. Receiptless archived evaluator outputs are
*not* scored tasks, even when they contain a completed public JSON. This
module accounts for the difference without inventing a per-request mapping.

The original F: tree is read through a single Windows handle-locked snapshot.
No inference, original evidence writes, guard-policy migration, or Gen-0
approval is performed. The observed arithmetic residual is not a claim that
those requests came from any particular task or client.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from chowder_batch import reap_original600_guardv3_git_pinned_reader_20261027 as historical
from chowder_batch import reap_original600_guardv3_native_journal_witness_20261029 as native
from chowder_batch.reap_original600_guardv3_stable_snapshot_20261029 import stable_snapshot


SCHEMA = "chowder.reap-original600-guardv3-crossledger-forensics.v1"
ARCHIVED = {
    "task-023.unfinished-006d2455d554050f":
        "130a511126b1d67f679ef55ac3f7822e803e90cab8c11b8c93a96db8ce0a0c5e",
    "task-024.unfinished-611c6174f3c6faab":
        "b721c45992041043f04c7d7aa7b1417934935c3f1711026621cfc3f6876e3508",
}
ARCHIVED_LOG_SHA = {
    "task-023.unfinished-006d2455d554050f":
        "1617dc5491dbddaf9e5b425381dd3ad82f285e81e43dae06275ef2f026997c5e",
    "task-024.unfinished-611c6174f3c6faab":
        "619f8f6610282b110be34188995f8ba6fd331030cb03868c2c8ab7d229a85d11",
}
LOG_ONLY = {
    "task-041.unfinished-66f67d1d459fd5e7",
    "task-087.unfinished-881ca4bf7d04bd0f",
}
LOG_ONLY_SHA = {
    "task-041.unfinished-66f67d1d459fd5e7":
        "51bee54d2db9490e1412208039f6fa70fdadd31772b5d6aaee654011f8413e7a",
    "task-087.unfinished-881ca4bf7d04bd0f":
        "f64f93655c2e02433d5d026151dedcbee4ca8eebc902942fbcf9cdd68b9b3672",
}
UNFINISHED = re.compile(r"task-(\d{3})\.unfinished-[0-9a-f]{16}\Z")
MAX_OFFICIAL_JSON = 4 * 1024 * 1024


class CrossLedgerError(ValueError):
    """Historical evidence is incomplete, changed, or overinterpreted."""


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _archive_fingerprint(files: dict[str, bytes]) -> str:
    """Reproduce the exact Git-pinned runner's quarantine-name digest."""
    evidence = [(name, _sha(raw)) for name, raw in sorted(files.items())]
    canonical = (json.dumps({"incomplete_files": evidence}, sort_keys=True,
                            indent=2, allow_nan=False) + "\n").encode()
    return _sha(canonical)[:16]


def _check_quarantine(name: str, files: dict[str, bytes]) -> None:
    match = UNFINISHED.fullmatch(name)
    if match is None or name.rsplit(".unfinished-", 1)[1] != _archive_fingerprint(files):
        raise CrossLedgerError("original quarantine fingerprint does not match files")


def _official_calls(raw: bytes, *, task_name: str) -> int:
    """Parse archived official output without treating it as a score receipt."""
    if not 0 < len(raw) <= MAX_OFFICIAL_JSON:
        raise CrossLedgerError("archived official output outside bounded size")
    result = native._strict_json(raw)
    tasks, usage = result.get("tasks"), result.get("usage_by_task")
    if (type(tasks) is not list or len(tasks) != 1
            or type(usage) is not list or len(usage) != 1
            or type(tasks[0]) is not dict or type(usage[0]) is not dict
            or tasks[0].get("name") != task_name
            or usage[0].get("task_name") != task_name):
        raise CrossLedgerError("archived original task/usage identity mismatch")
    calls = tasks[0].get("num_model_calls")
    if type(calls) is not int or not 0 <= calls <= 10_000:
        raise CrossLedgerError("archived original model-call count invalid")
    return calls


def reconcile_snapshot(snapshot: Path, *, scored: dict, owner: dict) -> dict:
    """Reconcile one locked copy; never attribute requests to individual tasks."""
    tasks = owner.get("tasks")
    if type(tasks) is not list or len(tasks) != 600:
        raise CrossLedgerError("original owner population invalid")
    rows = scored.get("rows")
    if (type(rows) is not list or len(rows) != 88
            or scored.get("completed_scored_tasks") != 88
            or scored.get("totals", {}).get("strict_passes") != 3
            or scored.get("owner_sha256") != historical.OWNER_SHA
            or scored.get("frozen_gen0_status") != "BLOCKED"
            or scored.get("release_approved") is not False):
        raise CrossLedgerError("historical source-locked scorer state changed")

    recorded = []
    for index, row in enumerate(rows):
        if type(row) is not dict or row.get("index") != index:
            raise CrossLedgerError("historical scored task ordering invalid")
        if row.get("name") != tasks[index].get("name"):
            raise CrossLedgerError("historical scored task identity invalid")
        source = snapshot / f"task-{index:03d}" / "public-automationbench.private.json"
        raw = source.read_bytes()
        if _sha(raw) != row.get("original_scorer_json_sha256"):
            raise CrossLedgerError("historical scored official JSON changed")
        calls = _official_calls(raw, task_name=tasks[index]["name"])
        if calls != row.get("model_calls"):
            raise CrossLedgerError("independent scored model-call count differs")
        recorded.append((index, calls, _sha(raw)))
    scored_calls = sum(row[1] for row in recorded)
    if scored_calls != scored["totals"].get("model_calls"):
        raise CrossLedgerError("historical scorer aggregate model-call mismatch")

    directories = {p.name for p in snapshot.iterdir() if p.is_dir()}
    scored_names = {f"task-{index:03d}" for index in range(len(rows))}
    unfinished = {name for name in directories if UNFINISHED.fullmatch(name)}
    if (unfinished != set(ARCHIVED) | LOG_ONLY
            or directories != scored_names | unfinished):
        raise CrossLedgerError("unexpected original task/archive directory inventory")

    archived = []
    if set(ARCHIVED) != set(ARCHIVED_LOG_SHA) or LOG_ONLY != set(LOG_ONLY_SHA):
        raise CrossLedgerError("historical archive source-hash inventory incomplete")
    for name, expected_sha in sorted(ARCHIVED.items()):
        directory = snapshot / name
        if (directory / "receipt.json").exists():
            raise CrossLedgerError("receiptless archive unexpectedly scored")
        if {p.name for p in directory.iterdir()} != {
                "official-evaluator.private.log", "public-automationbench.private.json"}:
            raise CrossLedgerError("receiptless archive file inventory changed")
        index = int(UNFINISHED.fullmatch(name).group(1))
        raw = (directory / "public-automationbench.private.json").read_bytes()
        log = (directory / "official-evaluator.private.log").read_bytes()
        if _sha(raw) != expected_sha:
            raise CrossLedgerError("historical receiptless official JSON SHA drift")
        if _sha(log) != ARCHIVED_LOG_SHA[name]:
            raise CrossLedgerError("historical receiptless evaluator log SHA drift")
        _check_quarantine(name, {"official-evaluator.private.log": log,
                                 "public-automationbench.private.json": raw})
        calls = _official_calls(raw, task_name=tasks[index]["name"])
        archived.append((index, calls, _sha(raw)))
    for name in LOG_ONLY:
        directory = snapshot / name
        if (directory / "receipt.json").exists() or (
                directory / "public-automationbench.private.json").exists():
            raise CrossLedgerError("log-only archive has unverified score/output")
        if {p.name for p in directory.iterdir()} != {"official-evaluator.private.log"}:
            raise CrossLedgerError("log-only archive file inventory changed")
        log = (directory / "official-evaluator.private.log").read_bytes()
        if _sha(log) != LOG_ONLY_SHA[name]:
            raise CrossLedgerError("historical log-only evaluator SHA drift")
        _check_quarantine(name, {"official-evaluator.private.log": log})

    native_names = {p.name for p in snapshot.glob("native-session-*.private.log")}
    guard_names = {p.name for p in snapshot.glob("guard-session-*.private.jsonl")}
    if (native_names != {f"native-session-{n}.private.log" for n, _ in native.SESSIONS}
            | {native.UNSERVED_LOG}
            or guard_names != {f"guard-session-{g}.private.jsonl" for _, g in native.SESSIONS}):
        raise CrossLedgerError("historical session inventory changed")
    unserved = (snapshot / native.UNSERVED_LOG).read_bytes()
    native._require_pinned(unserved, native.UNSERVED_SHA, native.UNSERVED_LOG)
    if native.parse_native(unserved, allow_empty=True)["launched"]:
        raise CrossLedgerError("unserved native session contains model requests")
    guard_requests = 0
    excess_native = 0
    for native_id, guard_id in native.SESSIONS:
        native_raw = (snapshot / f"native-session-{native_id}.private.log").read_bytes()
        guard_raw = (snapshot / f"guard-session-{guard_id}.private.jsonl").read_bytes()
        native._require_pinned(native_raw, native.PINNED_NATIVE[native_id], native_id)
        native._require_pinned(guard_raw, native.PINNED_GUARD[guard_id], guard_id)
        if (b"offloaded 41/41 layers to GPU" not in native_raw
                or b"loading model 'G:\\reap128-v2-q4_k_m-20261025.gguf'" not in native_raw):
            raise CrossLedgerError("original Q4 native serving identity missing")
        session = native.reconcile_pair(native_raw, guard_raw)
        guard_requests += session["requests"]
        excess_native += session["unmatched_native_launches"]

    archived_calls = sum(row[1] for row in archived)
    difference = guard_requests - scored_calls
    residual = difference - archived_calls
    if residual < 0:
        raise CrossLedgerError("archived model-call counts exceed guard request ledger")
    certificate = {
        "scored": recorded, "archived_receiptless": archived,
        "guard_requests": guard_requests, "unmatched_native_launches": excess_native,
        "arithmetic_residual": residual,
    }
    return {
        "schema": SCHEMA, "historical_source_commit": historical.COMMIT,
        "historical_owner_sha256": historical.OWNER_SHA,
        "completed_scored_tasks": len(rows), "strict_passes": 3,
        "scored_task_model_calls": scored_calls,
        "receiptless_archived_official_outputs": len(archived),
        "receiptless_archived_reported_model_calls": archived_calls,
        "log_only_unfinished_archives": len(LOG_ONLY),
        "guard_journal_requests": guard_requests,
        "guard_minus_scored_call_count": difference,
        "remaining_unreconciled_count_difference": residual,
        "unmatched_native_launches": excess_native,
        "crossledger_certificate_sha256": _sha(json.dumps(
            certificate, sort_keys=True, separators=(",", ":"),
            allow_nan=False).encode()),
        "per_request_attribution_established": False,
        "receiptless_archives_scored": False,
        "guard_calls_equivalent_to_official_model_calls_proven": False,
        "research_only": True, "frozen_four_slot_comparable": False,
        "gen0_status": "BLOCKED", "chosen": None, "release_approved": False,
    }


def verify_original_crossledger() -> dict:
    historical._require_assertions_enabled()
    with stable_snapshot(historical.ROOT, historical.REPO) as snapshot:
        owner_raw = (snapshot / "owner.json").read_bytes()
        if _sha(owner_raw) != historical.OWNER_SHA:
            raise CrossLedgerError("original guard.v3 owner SHA drift")
        owner = native._strict_json(owner_raw)
        _, scorer = historical.historical_modules()
        scored = scorer.inspect(snapshot)
        return reconcile_snapshot(snapshot, scored=scored, owner=owner)


def main() -> int:
    try:
        print(json.dumps(verify_original_crossledger(), indent=2, sort_keys=True))
        return 0
    except (CrossLedgerError, native.NativeJournalError, ValueError, OSError,
            KeyError, TypeError) as exc:
        print(json.dumps({"status": "REFUSED_ORIGINAL_GUARDV3_CROSSLEDGER",
                          "reason": str(exc)[:350], "gen0_status": "BLOCKED",
                          "chosen": None, "release_approved": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
