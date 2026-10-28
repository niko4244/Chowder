"""Authentic saved v1 selector decision, not synthetic model results."""

from pathlib import Path

import pytest

from chowder_batch import reap_v2_authentic_amendment4_replay_20261027 as replay


def test_actual_frozen_historical_v1_gate1_fails_and_requires_both_fallbacks():
    result = replay.replay()
    assert result["historical_v1_gate1"] == {
        "status": "FAIL", "lower_bound": -0.03780741197234255,
        "strict_margin": -0.03,
    }
    assert result["authentic_historical_selector"]["status"] == "BLOCKED"
    assert "both REAP128_v2 and REAP160" in result["authentic_historical_selector"]["reason"]
    assert result["current_v2_gate1"] == "UNKNOWN"
    assert result["current_v2_gate2"] == "UNKNOWN"
    assert result["amendment4_optional_nonzero_margin_paired_mcnemar_ci"].startswith("BLOCKED")
    assert result["gen0"] == {"status": "BLOCKED", "chosen": None, "release_approved": False}


def test_forged_historical_gate_artifact_fails_before_selector(tmp_path: Path):
    forged = tmp_path / "copy.json"
    forged.write_text(replay.SOURCE.read_text(encoding="utf-8").replace(
        '"status": "FAIL"', '"status": "PASS"', 1), encoding="utf-8")
    with pytest.raises(replay.FrozenReplayError, match="SHA drift"):
        replay.replay(forged)


def test_missing_or_untrusted_source_does_not_default_to_v2_pass(tmp_path: Path):
    with pytest.raises(replay.FrozenReplayError, match="regular file"):
        replay.replay(tmp_path / "missing.json")
