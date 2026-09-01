import pytest

from chowder.investigation import RemediationOutcome, config_patch_digest
from chowder.replay import ReplayExecutor, ReplayGroundTruth


def test_outcome_for_known_patch():
    good_patch = {"attn_implementation": "eager"}
    truth = ReplayGroundTruth(
        fingerprint_sha256="abc123",
        outcomes={config_patch_digest(good_patch): RemediationOutcome.RESOLVED},
    )
    assert truth.outcome_for(good_patch) is RemediationOutcome.RESOLVED


def test_outcome_for_unlisted_patch_raises():
    truth = ReplayGroundTruth(fingerprint_sha256="abc123", outcomes={})
    with pytest.raises(KeyError, match="no ground truth recorded"):
        truth.outcome_for({"never": "seen"})


def test_lookup_is_order_independent():
    """A config_patch with keys in a different order must still hit the
    same ground-truth entry -- the digest is over canonical JSON, not
    dict-insertion order."""
    patch_a = {"x": 1, "y": 2}
    patch_b = {"y": 2, "x": 1}
    truth = ReplayGroundTruth(
        fingerprint_sha256="abc123",
        outcomes={config_patch_digest(patch_a): RemediationOutcome.RESOLVED},
    )
    assert truth.outcome_for(patch_b) is RemediationOutcome.RESOLVED


def test_structurally_different_patches_do_not_collide():
    truth = ReplayGroundTruth(
        fingerprint_sha256="abc123",
        outcomes={
            config_patch_digest({"a": 1}): RemediationOutcome.RESOLVED,
            config_patch_digest({"a": 2}): RemediationOutcome.DID_NOT_RESOLVE,
        },
    )
    assert truth.outcome_for({"a": 1}) is RemediationOutcome.RESOLVED
    assert truth.outcome_for({"a": 2}) is RemediationOutcome.DID_NOT_RESOLVE


def test_replay_executor_delegates_to_ground_truth():
    patch = {"max_length": 1536}
    truth = ReplayGroundTruth(
        fingerprint_sha256="abc123",
        outcomes={config_patch_digest(patch): RemediationOutcome.PARTIALLY_RESOLVED},
    )
    executor = ReplayExecutor(ground_truth=truth)
    assert executor.run(patch) is RemediationOutcome.PARTIALLY_RESOLVED


def test_replay_executor_unlisted_patch_raises():
    executor = ReplayExecutor(ground_truth=ReplayGroundTruth("abc123", {}))
    with pytest.raises(KeyError):
        executor.run({"unexpected": True})
