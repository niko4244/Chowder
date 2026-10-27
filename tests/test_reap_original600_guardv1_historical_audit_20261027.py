"""Original guard.v1 must remain auditable without changing guard.v2 source."""

import pytest

from chowder_batch import reap_original600_guardv1_historical_audit_20261027 as original


def test_genuine_immutable_guard_v1_prefix_reaudits_without_inference():
    if not original.ORIGINAL_RUN.is_dir():
        pytest.skip("original private F: evidence unavailable")
    current = original.independent.run.q4.POLICY_NAME
    result = original.audit_original()
    assert result["owner_sha256"] == original.OLD_OWNER_SHA
    assert result["completed_scored_tasks"] >= 43
    assert result["historical_guard_policy"] == "guard.v1"
    assert result["new_guard_v2_compatible_for_pooling"] is False
    assert result["frozen_gen0_status"] == "BLOCKED"
    assert result["release_approved"] is False
    assert result["official_complete_600_claim_allowed"] is False
    assert original.independent.run.q4.POLICY_NAME == current


def test_foreign_benchmark_root_cannot_be_scored_as_original(tmp_path):
    with pytest.raises(ValueError, match="not the registered"):
        original.audit_original(tmp_path)


def test_immutable_owner_hash_mismatch_refused(monkeypatch):
    monkeypatch.setattr(original, "OLD_OWNER_SHA", "0" * 64)
    with pytest.raises(ValueError, match="owner SHA changed"):
        original.audit_original()


def test_frozen_git_original_guard_source_must_match(monkeypatch):
    old = original.SOURCE_HASHES[
        "chowder_batch/reap_context_guard_proxy_20261026.py"]
    monkeypatch.setitem(original.SOURCE_HASHES,
                        "chowder_batch/reap_context_guard_proxy_20261026.py",
                        "0" * 64)
    with pytest.raises(ValueError, match="Git source pin"):
        original.audit_original()
    assert old != "0" * 64


def test_original_scoring_source_must_be_same_as_frozen_commit(monkeypatch):
    old = original.SOURCE_HASHES[
        "chowder_batch/reap_original600_live_independent_audit_20261027.py"]
    monkeypatch.setitem(original.SOURCE_HASHES,
                        "chowder_batch/reap_original600_live_independent_audit_20261027.py",
                        "1" * 64)
    with pytest.raises(ValueError, match="Git source pin"):
        original.audit_original()
    assert old != "1" * 64
