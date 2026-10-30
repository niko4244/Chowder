"""Original source-bound diagnostics are descriptive, never substitute scores."""

import json

import pytest

from chowder_batch import reap_original600_failure_diagnostics_20261027 as d


def fixture(tmp_path, monkeypatch, *, scored=True):
    root = tmp_path
    row = {
        "name": "original.task-000", "passed": False,
        "steps": 44, "died_on_refusal": False,
        "stop_reasons": ["tool_calls", "length"],
        "assertion_results": [
            {"type": "slack_message_in_channel", "passed": True, "excluded": False},
            {"type": "salesforce_note_exists", "passed": False, "excluded": False},
            {"type": "gmail_message_not_sent_to", "passed": True, "excluded": True},
        ],
    }
    entries = []
    if scored:
        item = root / "task-000"
        item.mkdir()
        raw = item / "public-automationbench.private.json"
        raw.write_text(json.dumps({"tasks": [row]}), encoding="utf-8")
        entries.append({
            "index": 0, "name": "original.task-000", "strict_pass": False,
            "positive_passed": 1, "positive_total": 2,
            "original_scorer_json_sha256": d.audit.digest(raw),
            "checkpoint_sha256": "a" * 64,
        })
    totals = {
        "positive_assertions": 2 if scored else 0,
        "positive_assertions_passed": 1 if scored else 0,
        "excluded_assertions": 1 if scored else 0,
        "excluded_assertions_passed": 1 if scored else 0,
        "strict_passes": 0,
    }
    source = {
        "rows": entries, "totals": totals,
        "official_source_commit": "b" * 40,
        "model_gguf_sha256": "c" * 64,
        "owner_sha256": "d" * 64,
        "active_unscored_indices": [1 if scored else 0],
    }
    monkeypatch.setattr(d.audit, "inspect", lambda _: source)
    return root, row, source


def test_unscored_task_not_reported_as_zero_success(tmp_path, monkeypatch):
    root, _, _ = fixture(tmp_path, monkeypatch, scored=False)
    report = d.inspect(root)
    assert report["scored"] == 0
    assert report["status"] == "AUTHENTIC_PARTIAL_PREFIX_DESCRIPTIVE_ONLY"
    assert report["active_unscored"] == [0]
    assert report["prefix_nonrandom_no_full600_confidence_interval"] is True
    assert report["frozen_gen0_status"] == "BLOCKED"
    assert report["chosen"] is None
    assert report["release_approved"] is False


def test_source_bound_original_failure_taxonomy(tmp_path, monkeypatch):
    root, _, _ = fixture(tmp_path, monkeypatch)
    report = d.inspect(root)
    assert report["scored"] == 1
    assert report["strict_passes"] == 0
    assert report["strict_fails"] == 1
    assert report["failed_with_some_positive_satisfied"] == 1
    assert report["failed_with_no_positive_satisfied"] == 0
    assert report["at_original_44_step_limit"] == 1
    assert report["original_refusal_terminations"] == 0
    assert report["excluded_requirement_violations"] == 0
    assert report["assertions"] == {
        "positive": 2, "positive_passed": 1,
        "excluded": 1, "excluded_passed": 1,
    }
    assert report["stop_reasons_original"]["length"] == 1
    assert report["by_original_assertion_type"]["salesforce_note_exists"] == {
        "positive": 1, "positive_passed": 0,
    }
    assert report["noncomparable_to_frozen_four_slot"] is True


def test_hash_tampering_refused(tmp_path, monkeypatch):
    root, _, source = fixture(tmp_path, monkeypatch)
    source["rows"][0]["original_scorer_json_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="changed before diagnosis"):
        d.inspect(root)


@pytest.mark.parametrize("field,value,reason", [
    ("steps", -1, "step count"),
    ("steps", 45, "step count"),
    ("steps", True, "step count"),
    ("died_on_refusal", "false", "refusal"),
    ("stop_reasons", [1], "stop reasons"),
])
def test_forged_official_metrics_refused(tmp_path, monkeypatch, field, value, reason):
    root, row, source = fixture(tmp_path, monkeypatch)
    row[field] = value
    raw = root / "task-000" / "public-automationbench.private.json"
    raw.write_text(json.dumps({"tasks": [row]}), encoding="utf-8")
    source["rows"][0]["original_scorer_json_sha256"] = d.audit.digest(raw)
    with pytest.raises(ValueError, match=reason):
        d.inspect(root)


def test_assertion_type_nonstring_refused(tmp_path, monkeypatch):
    root, row, source = fixture(tmp_path, monkeypatch)
    row["assertion_results"][0]["type"] = []
    raw = root / "task-000" / "public-automationbench.private.json"
    raw.write_text(json.dumps({"tasks": [row]}), encoding="utf-8")
    source["rows"][0]["original_scorer_json_sha256"] = d.audit.digest(raw)
    with pytest.raises(ValueError, match="assertion type invalid"):
        d.inspect(root)


def test_owner_verified_audit_totals_unchanged(tmp_path, monkeypatch):
    root, _, source = fixture(tmp_path, monkeypatch)
    source["totals"]["positive_assertions"] += 1
    with pytest.raises(ValueError, match="disagrees with independent"):
        d.inspect(root)
