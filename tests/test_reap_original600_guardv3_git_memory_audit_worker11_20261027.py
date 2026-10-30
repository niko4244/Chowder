"""Adversarial source-lock tests for the immutable original guard.v3 scorer."""

import pytest

from chowder_batch import reap_original600_guardv3_git_memory_audit_worker11_20261027 as target


def test_actual_original_scoring_and_restoration():
    original = target.audit.run
    policy = original.q4.POLICY_NAME
    result = target.inspect()
    assert result["completed_scored_tasks"] >= 31
    assert result["totals"]["tool_calls"] >= 1157
    assert result["historical_guard_policy"] == "guard.v3"
    assert result["historical_git_runner_source_sha256"] == target.PINNED[target.RUNNER]
    assert result["noncomparable_to_frozen_four_slot"] is True
    assert result["release_approved"] is False
    assert target.audit.run is original
    assert original.q4.POLICY_NAME == policy


@pytest.mark.parametrize("name", list(target.PINNED))
def test_tampered_git_source_is_refused(name, monkeypatch):
    monkeypatch.setitem(target.PINNED, name, "0" * 64)
    with pytest.raises(ValueError, match="original Git source identity drift"):
        target.pinned_sources()


def test_forged_owner_is_refused(monkeypatch):
    monkeypatch.setattr(target, "OWNER_SHA", "f" * 64)
    with pytest.raises(ValueError, match="historical original source/owner"):
        target.inspect()


def test_scoring_exception_restores_imported_module(monkeypatch):
    original = target.audit.run
    policy = original.q4.POLICY_NAME

    def fail(_):
        assert target.audit.run is not original
        assert target.audit.run.q4.POLICY_NAME == "guard.v3"
        raise ValueError("malformed original receipt")

    monkeypatch.setattr(target.audit, "inspect", fail)
    with pytest.raises(ValueError, match="malformed original receipt"):
        target.inspect()
    assert target.audit.run is original
    assert original.q4.POLICY_NAME == policy


def test_source_load_does_not_mutate_live_runner(monkeypatch):
    source = target.pinned_sources()[target.RUNNER]
    original = target.audit.run
    frozen = target.frozen_runner(source)
    assert frozen is not original
    assert frozen.verify_item is not original.verify_item
    assert frozen.SCHEMA == original.SCHEMA
