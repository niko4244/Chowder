"""Adversarial CPU-only tests of historical official model-call trace evidence."""

from __future__ import annotations

import json

import pytest

from chowder_batch import reap_original600_guardv3_official_call_trace_witness_20261029 as witness


def _raw(name: str, calls: int, *, steps=None, messages=None, stops=None,
         finishes=None) -> bytes:
    if messages is None:
        messages = ([{"role": "system", "content": "rules"},
                     {"role": "user", "content": "task"}]
                    + [{"role": "assistant", "content": "reply"} for _ in range(calls)])
    return json.dumps({
        "tasks": [{"name": name, "num_model_calls": calls,
                   "steps": calls if steps is None else steps,
                   "messages": messages,
                   "stop_reasons": ["stop"] * calls if stops is None else stops,
                   "finish_reasons": ["stop"] * max(0, calls - 1)
                   if finishes is None else finishes}],
        "usage_by_task": [{"task_name": name}],
    }, sort_keys=True, separators=(",", ":")).encode()


def test_independent_fields_support_model_call_count():
    result = witness.inspect_official_trace(_raw("canonical-0", 3),
                                            task_name="canonical-0")
    assert result["model_calls"] == 3
    assert result["assistant_messages"] == 3
    assert result["steps"] == 3
    assert result["stop_reasons"] == 3
    assert result["finish_reasons"] == 2
    assert len(result["message_role_sequence_sha256"]) == 64


@pytest.mark.parametrize("bad", [
    _raw("canonical-0", 3, steps=2),
    _raw("canonical-0", 3, steps=True),
    _raw("canonical-0", 3, messages=[{"role": "assistant", "content": "x"}]),
    _raw("canonical-0", 3, messages=[{"role": "assistant"}] * 3),
    _raw("canonical-0", 3, messages=[{"role": ["assistant"], "content": "x"}] * 3),
    _raw("canonical-0", 3, messages=[{"role": "tool", "content": "x"}] * 3),
    _raw("canonical-0", 3, stops=["stop", "stop"]),
    _raw("canonical-0", 3, stops=["stop", None, "stop"]),
    _raw("canonical-0", 3, finishes=["stop"] * 4),
    _raw("wrong-task", 3),
    b'{"tasks":[],"tasks":[],"usage_by_task":[]}',
    b'{"tasks":[{"name":"canonical-0","num_model_calls":NaN}],"usage_by_task":[]}',
    b"{}",
])
def test_rejects_false_call_trace_or_malformed_original(bad):
    with pytest.raises((witness.CallTraceError, witness.native.NativeJournalError)):
        witness.inspect_official_trace(bad, task_name="canonical-0")


@pytest.fixture
def population(tmp_path, monkeypatch):
    owner = {"tasks": [{"name": f"canonical-{i}"} for i in range(600)]}
    scored = {"rows": []}
    for i in range(88):
        raw = _raw(owner["tasks"][i]["name"], 1)
        folder = tmp_path / f"task-{i:03d}"
        folder.mkdir()
        (folder / "public-automationbench.private.json").write_bytes(raw)
        scored["rows"].append({"index": i, "model_calls": 1,
                               "original_scorer_json_sha256": witness._sha(raw)})
    archives = {}
    for i, calls in ((23, 44), (24, 35)):
        name = f"task-{i:03d}.unfinished-" + str(i) * 8
        folder = tmp_path / name
        folder.mkdir()
        raw = _raw(owner["tasks"][i]["name"], calls)
        (folder / "public-automationbench.private.json").write_bytes(raw)
        archives[name] = witness._sha(raw)
    monkeypatch.setattr(witness.ledger, "ARCHIVED", archives)
    monkeypatch.setattr(witness.ledger, "reconcile_snapshot", lambda *a, **kw: {
        "completed_scored_tasks": 88, "strict_passes": 3,
        "receiptless_archived_official_outputs": 2,
        "receiptless_archived_reported_model_calls": 79,
        "remaining_unreconciled_count_difference": 18,
        "scored_task_model_calls": 88,
        "crossledger_certificate_sha256": "c" * 64,
    })
    return tmp_path, scored, owner


def test_population_cannot_promote_receiptless_archives(population):
    root, scored, owner = population
    result = witness.reconcile_trace_snapshot(root, scored=scored, owner=owner)
    assert result["scored_tasks"] == 88
    assert result["receiptless_unscored_archives"] == 2
    assert result["scored_model_calls_correlated"] == 88
    assert result["receiptless_model_calls_correlated"] == 79
    assert result["all_assistant_messages"] == 167
    assert result["all_steps"] == 167
    assert result["all_stop_reasons"] == 167
    assert result["original_unreconciled_guard_request_difference"] == 18
    assert result["independent_execution_attestation"] is False
    assert result["receiptless_archives_scored"] is False
    assert result["gen0_status"] == "BLOCKED"
    assert result["chosen"] is None and result["release_approved"] is False


def test_tampered_scored_trace_rejected_even_when_sha_is_replaced(population):
    root, scored, owner = population
    raw = _raw("canonical-7", 1, steps=2)
    (root / "task-007" / "public-automationbench.private.json").write_bytes(raw)
    scored["rows"][7]["original_scorer_json_sha256"] = witness._sha(raw)
    with pytest.raises(witness.CallTraceError, match="disagree"):
        witness.reconcile_trace_snapshot(root, scored=scored, owner=owner)


def test_archived_sha_drift_rejected(population):
    root, scored, owner = population
    name = next(iter(witness.ledger.ARCHIVED))
    (root / name / "public-automationbench.private.json").write_bytes(b"{}")
    with pytest.raises(witness.CallTraceError, match="SHA drift"):
        witness.reconcile_trace_snapshot(root, scored=scored, owner=owner)


def test_source_crossledger_population_drift_rejected(population, monkeypatch):
    root, scored, owner = population
    monkeypatch.setattr(witness.ledger, "reconcile_snapshot", lambda *a, **kw: {
        "completed_scored_tasks": 89, "strict_passes": 3,
        "receiptless_archived_official_outputs": 2,
        "receiptless_archived_reported_model_calls": 79,
        "remaining_unreconciled_count_difference": 18,
    })
    with pytest.raises(witness.CallTraceError, match="population changed"):
        witness.reconcile_trace_snapshot(root, scored=scored, owner=owner)
