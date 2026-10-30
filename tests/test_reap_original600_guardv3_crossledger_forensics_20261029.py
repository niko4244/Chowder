"""Adversarial synthetic evidence for the historical guard.v3 call ledger."""

from __future__ import annotations

import json

import pytest

from chowder_batch import reap_original600_guardv3_crossledger_forensics_20261029 as audit


def _raw(name: str, calls: int) -> bytes:
    return json.dumps({"tasks": [{"name": name, "num_model_calls": calls}],
                       "usage_by_task": [{"task_name": name}]},
                      sort_keys=True, separators=(",", ":")).encode()


@pytest.fixture
def case(tmp_path, monkeypatch):
    owner = {"tasks": [{"name": f"canonical-{i}"} for i in range(600)]}
    rows = []
    for i in range(88):
        directory = tmp_path / f"task-{i:03d}"
        directory.mkdir()
        raw = _raw(owner["tasks"][i]["name"], 1)
        (directory / "public-automationbench.private.json").write_bytes(raw)
        rows.append({"index": i, "name": owner["tasks"][i]["name"],
                     "model_calls": 1, "original_scorer_json_sha256": audit._sha(raw)})
    scored = {"rows": rows, "completed_scored_tasks": 88,
              "totals": {"strict_passes": 3, "model_calls": 88},
              "owner_sha256": audit.historical.OWNER_SHA,
              "frozen_gen0_status": "BLOCKED", "release_approved": False}
    archives = {}
    archive_logs = {}
    for i, count in ((23, 2), (24, 3)):
        raw = _raw(owner["tasks"][i]["name"], count)
        log = f"original incomplete task {i}\n".encode()
        fingerprint = audit._archive_fingerprint({
            "official-evaluator.private.log": log,
            "public-automationbench.private.json": raw})
        name = f"task-{i:03d}.unfinished-{fingerprint}"
        directory = tmp_path / name
        directory.mkdir()
        (directory / "public-automationbench.private.json").write_bytes(raw)
        (directory / "official-evaluator.private.log").write_bytes(log)
        archives[name] = audit._sha(raw)
        archive_logs[name] = audit._sha(log)
    monkeypatch.setattr(audit, "ARCHIVED", archives)
    monkeypatch.setattr(audit, "ARCHIVED_LOG_SHA", archive_logs)
    log_only = b"original unfinished task 41\n"
    log_name = "task-041.unfinished-" + audit._archive_fingerprint({
        "official-evaluator.private.log": log_only})
    monkeypatch.setattr(audit, "LOG_ONLY", {log_name})
    monkeypatch.setattr(audit, "LOG_ONLY_SHA", {log_name: audit._sha(log_only)})
    (tmp_path / log_name).mkdir()
    (tmp_path / log_name / "official-evaluator.private.log").write_bytes(log_only)

    native_raw = (b"offloaded 41/41 layers to GPU\n"
                  b"loading model 'G:\\reap128-v2-q4_k_m-20261025.gguf'\n")
    guard_raw = b"guard\n"
    unserved = b"unserved\n"
    (tmp_path / "native-session-1.private.log").write_bytes(native_raw)
    (tmp_path / "guard-session-2.private.jsonl").write_bytes(guard_raw)
    (tmp_path / "native-session-3.private.log").write_bytes(unserved)
    monkeypatch.setattr(audit.native, "SESSIONS", (("1", "2"),))
    monkeypatch.setattr(audit.native, "UNSERVED_LOG", "native-session-3.private.log")
    monkeypatch.setattr(audit.native, "UNSERVED_SHA", audit._sha(unserved))
    monkeypatch.setattr(audit.native, "PINNED_NATIVE", {"1": audit._sha(native_raw)})
    monkeypatch.setattr(audit.native, "PINNED_GUARD", {"2": audit._sha(guard_raw)})
    monkeypatch.setattr(audit.native, "parse_native", lambda raw, **kw: {"launched": 0})
    monkeypatch.setattr(audit.native, "reconcile_pair", lambda n, g: {
        "requests": 100, "unmatched_native_launches": 1})
    return tmp_path, scored, owner


def _verify(case):
    root, scored, owner = case
    return audit.reconcile_snapshot(root, scored=scored, owner=owner)


def test_original_arithmetic_keeps_unassigned_requests_and_unscored_archives(case):
    result = _verify(case)
    assert result["completed_scored_tasks"] == 88
    assert result["scored_task_model_calls"] == 88
    assert result["receiptless_archived_reported_model_calls"] == 5
    assert result["guard_minus_scored_call_count"] == 12
    assert result["remaining_unreconciled_count_difference"] == 7
    assert result["unmatched_native_launches"] == 1
    assert result["receiptless_archives_scored"] is False
    assert result["per_request_attribution_established"] is False
    assert result["gen0_status"] == "BLOCKED"
    assert result["chosen"] is None and result["release_approved"] is False
    assert len(result["crossledger_certificate_sha256"]) == 64


@pytest.mark.parametrize("bad", [
    b'{"tasks":[],"tasks":[],"usage_by_task":[]}',
    b'{"tasks":[{"name":"x","num_model_calls":NaN}],"usage_by_task":[]}',
    _raw("wrong-task", 3),
    _raw("canonical-23", -1),
    _raw("canonical-23", True),
    _raw("canonical-23", 10001),
    b'{"tasks":{},"usage_by_task":[]}',
    b'{"tasks":[{"name":"canonical-23","num_model_calls":3}],"usage_by_task":[{"task_name":"wrong"}]}',
])
def test_archived_official_json_rejects_malformed_and_forged_counts(bad):
    with pytest.raises((audit.CrossLedgerError, audit.native.NativeJournalError)):
        audit._official_calls(bad, task_name="canonical-23")


def test_modified_scored_call_count_rejected(case):
    root, scored, owner = case
    scored["rows"][3]["model_calls"] = 2
    with pytest.raises(audit.CrossLedgerError, match="model-call count"):
        audit.reconcile_snapshot(root, scored=scored, owner=owner)


def test_modified_scored_source_bytes_rejected(case):
    root, scored, owner = case
    (root / "task-003" / "public-automationbench.private.json").write_bytes(
        _raw("canonical-3", 2))
    with pytest.raises(audit.CrossLedgerError, match="official JSON changed"):
        audit.reconcile_snapshot(root, scored=scored, owner=owner)


def test_archive_with_receipt_cannot_be_scored(case):
    root, scored, owner = case
    name = next(iter(audit.ARCHIVED))
    (root / name / "receipt.json").write_text("{}")
    with pytest.raises(audit.CrossLedgerError, match="unexpectedly scored"):
        audit.reconcile_snapshot(root, scored=scored, owner=owner)


def test_unexpected_archive_rejected(case):
    root, scored, owner = case
    (root / ("task-042.unfinished-" + "c" * 16)).mkdir()
    with pytest.raises(audit.CrossLedgerError, match="directory inventory"):
        audit.reconcile_snapshot(root, scored=scored, owner=owner)


def test_missing_guard_session_rejected(case):
    root, scored, owner = case
    (root / "guard-session-2.private.jsonl").unlink()
    with pytest.raises(audit.CrossLedgerError, match="session inventory"):
        audit.reconcile_snapshot(root, scored=scored, owner=owner)


def test_modified_guard_journal_rejected(case):
    root, scored, owner = case
    (root / "guard-session-2.private.jsonl").write_bytes(b"forged\n")
    with pytest.raises(audit.native.NativeJournalError, match="SHA drift"):
        audit.reconcile_snapshot(root, scored=scored, owner=owner)


def test_log_only_archive_cannot_gain_unverified_output(case):
    root, scored, owner = case
    name = next(iter(audit.LOG_ONLY))
    (root / name / "public-automationbench.private.json").write_bytes(b"{}")
    with pytest.raises(audit.CrossLedgerError, match="unverified score/output"):
        audit.reconcile_snapshot(root, scored=scored, owner=owner)


def test_negative_arithmetic_residual_refuses_not_fabricates(case, monkeypatch):
    root, scored, owner = case
    monkeypatch.setattr(audit.native, "reconcile_pair", lambda n, g: {
        "requests": 90, "unmatched_native_launches": 1})
    with pytest.raises(audit.CrossLedgerError, match="exceed guard request"):
        audit.reconcile_snapshot(root, scored=scored, owner=owner)


def test_archive_raw_hash_drift_rejected(case):
    root, scored, owner = case
    name = next(iter(audit.ARCHIVED))
    path = root / name / "public-automationbench.private.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(audit.CrossLedgerError, match="SHA drift"):
        audit.reconcile_snapshot(root, scored=scored, owner=owner)


def test_archived_evaluator_log_drift_rejected(case):
    root, scored, owner = case
    name = next(iter(audit.ARCHIVED))
    (root / name / "official-evaluator.private.log").write_bytes(b"forged")
    with pytest.raises(audit.CrossLedgerError, match="evaluator log SHA drift"):
        audit.reconcile_snapshot(root, scored=scored, owner=owner)


def test_log_only_archive_missing_log_rejected(case):
    root, scored, owner = case
    name = next(iter(audit.LOG_ONLY))
    (root / name / "official-evaluator.private.log").unlink()
    with pytest.raises(audit.CrossLedgerError, match="file inventory"):
        audit.reconcile_snapshot(root, scored=scored, owner=owner)


def test_log_only_archive_modified_log_rejected(case):
    root, scored, owner = case
    name = next(iter(audit.LOG_ONLY))
    (root / name / "official-evaluator.private.log").write_bytes(b"forged")
    with pytest.raises(audit.CrossLedgerError, match="evaluator SHA drift"):
        audit.reconcile_snapshot(root, scored=scored, owner=owner)


def test_quarantine_name_fingerprint_mismatch_rejected(case, monkeypatch):
    root, scored, owner = case
    name = next(iter(audit.ARCHIVED))
    changed = name.rsplit("-", 1)[0] + "-" + "f" * 16
    (root / name).rename(root / changed)
    monkeypatch.setattr(audit, "ARCHIVED", {changed: audit.ARCHIVED[name],
                                            **{k: v for k, v in audit.ARCHIVED.items() if k != name}})
    monkeypatch.setattr(audit, "ARCHIVED_LOG_SHA", {
        changed: audit.ARCHIVED_LOG_SHA[name],
        **{k: v for k, v in audit.ARCHIVED_LOG_SHA.items() if k != name}})
    with pytest.raises(audit.CrossLedgerError, match="quarantine fingerprint"):
        audit.reconcile_snapshot(root, scored=scored, owner=owner)


def test_receiptless_archive_extra_file_rejected(case):
    root, scored, owner = case
    name = next(iter(audit.ARCHIVED))
    (root / name / "unexpected.private.log").write_bytes(b"unexpected")
    with pytest.raises(audit.CrossLedgerError, match="file inventory"):
        audit.reconcile_snapshot(root, scored=scored, owner=owner)


def test_source_owner_drift_rejected(case):
    root, scored, owner = case
    scored["owner_sha256"] = "0" * 64
    with pytest.raises(audit.CrossLedgerError, match="scorer state"):
        audit.reconcile_snapshot(root, scored=scored, owner=owner)
