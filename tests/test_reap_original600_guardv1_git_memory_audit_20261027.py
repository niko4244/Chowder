"""Source-pinned historical guard.v1 receipt replay, no model inference."""

from pathlib import Path

import pytest

from chowder_batch import reap_original600_guardv1_git_memory_audit_20261027 as v1


def test_actual_original_43_task_archive_recovers_with_historical_git_runner():
    before_run = v1.scorer.run
    before_policy = before_run.q4.POLICY_NAME
    report = v1.inspect()
    assert report["completed_scored_tasks"] == 43
    assert report["totals"]["strict_passes"] == 0
    assert report["totals"]["positive_assertions_passed"] == 35
    assert report["totals"]["positive_assertions"] == 240
    assert report["totals"]["excluded_assertions_passed"] == 223
    assert report["active_unscored_indices"] == [43]
    assert report["historic_git_commit"] == v1.COMMIT
    assert report["historic_guard_source_sha256"] == v1.SOURCE_SHA[v1.GUARD]
    assert report["model_executed_for_this_audit"] is False
    assert report["frozen_gen0_status"] == "BLOCKED"
    assert report["frozen_four_slot_comparable"] is False
    assert v1.scorer.run is before_run
    assert before_run.q4.POLICY_NAME == before_policy


def test_retired_runner_may_change_live_worktree_without_relabeling_original():
    assert v1.sources()[v1.RUNNER]
    result = v1.inspect()
    assert result["historic_git_commit"] == v1.COMMIT
    assert result["source_lock"].endswith("historical-audit.v1")


def test_historical_source_tampering_fails_closed(monkeypatch):
    monkeypatch.setitem(v1.SOURCE_SHA, v1.GUARD, "0" * 64)
    with pytest.raises(ValueError, match="pinned Git source drift"):
        v1.sources()


def test_historical_owner_tampering_fails_closed(monkeypatch):
    monkeypatch.setattr(v1, "OWNER_SHA", "0" * 64)
    with pytest.raises(ValueError, match="owner SHA drift"):
        v1.inspect()


def test_nonexistent_archive_refused(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(v1, "ROOT", tmp_path / "missing")
    with pytest.raises(ValueError, match="source root"):
        v1.inspect()


def test_original_reader_restores_module_and_policy_if_scorer_refuses(monkeypatch):
    before_run = v1.scorer.run
    before_policy = before_run.q4.POLICY_NAME

    def refuse(_: Path):
        assert v1.scorer.run is not before_run
        assert v1.scorer.run.q4.POLICY_NAME == "guard.v1"
        raise ValueError("historical independent scorer refusal")

    monkeypatch.setattr(v1.scorer, "inspect", refuse)
    with pytest.raises(ValueError, match="independent scorer refusal"):
        v1.inspect()
    assert v1.scorer.run is before_run
    assert before_run.q4.POLICY_NAME == before_policy
