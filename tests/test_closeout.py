import pytest

from chowder.closeout import finalize_investigation
from chowder.incident import compute_fingerprint
from chowder.investigation import (
    Investigation,
    RemediationOutcome,
    RemediationRegistry,
    config_patch_digest,
)
from chowder.models import Hypothesis
from chowder.remediation_runner import RemediationExperiment
from chowder.replay import ReplayExecutor, ReplayGroundTruth

from fixtures_incidents import QWEN3_5_CONV1D_NO_ENGINE


def _build_resolved_investigation() -> Investigation:
    """One failed trial, then one that resolves -- a realistic shape, not
    a single-trial happy path."""
    capture = QWEN3_5_CONV1D_NO_ENGINE
    fingerprint = compute_fingerprint(capture)
    wrong_patch = {"attn_implementation": "eager"}
    right_patch = {"cudnn_enabled": False}
    truth = ReplayGroundTruth(
        fingerprint_sha256=fingerprint.fingerprint_sha256,
        outcomes={
            config_patch_digest(wrong_patch): RemediationOutcome.DID_NOT_RESOLVE,
            config_patch_digest(right_patch): RemediationOutcome.RESOLVED,
        },
    )
    experiment = RemediationExperiment(executor=ReplayExecutor(truth))
    investigation = Investigation(
        investigation_id="inv-closeout-test",
        fingerprint=fingerprint,
        capture=capture,
        gpu_hour_budget=5.0,
    )

    wrong_trial = investigation.add_hypothesis(
        Hypothesis(observation="o1", suspected_cause="wrong guess", intervention="force eager"),
        config_patch=wrong_patch,
    )
    wrong_record = experiment.run(investigation, wrong_trial, environment=capture.environment)
    investigation.record_failed_trial(wrong_trial, wrong_record)

    right_trial = investigation.add_hypothesis(
        Hypothesis(observation="o2", suspected_cause="cuDNN kernel gap", intervention="disable cuDNN"),
        config_patch=right_patch,
    )
    right_record = experiment.run(investigation, right_trial, environment=capture.environment)
    investigation.resolve(right_trial, right_record)

    return investigation


def test_finalize_requires_resolved_status():
    capture = QWEN3_5_CONV1D_NO_ENGINE
    investigation = Investigation(
        investigation_id="inv-open",
        fingerprint=compute_fingerprint(capture),
        capture=capture,
        gpu_hour_budget=1.0,
    )
    with pytest.raises(ValueError, match="only a RESOLVED investigation"):
        finalize_investigation(investigation)


def test_finalize_returns_registry_ready_record():
    investigation = _build_resolved_investigation()
    record, _trail = finalize_investigation(investigation)
    assert record.outcome is RemediationOutcome.RESOLVED

    fresh_registry = RemediationRegistry().with_record(record)
    looked_up = fresh_registry.lookup(investigation.fingerprint)
    assert looked_up is record


def test_audit_trail_contains_every_trial_in_order():
    investigation = _build_resolved_investigation()
    _record, trail = finalize_investigation(investigation)
    assert len(trail.trials) == 2
    assert trail.trials[0].hypothesis_intervention == "force eager"
    assert trail.trials[0].outcome is RemediationOutcome.DID_NOT_RESOLVE
    assert trail.trials[1].hypothesis_intervention == "disable cuDNN"
    assert trail.trials[1].outcome is RemediationOutcome.RESOLVED


def test_audit_trail_digest_is_deterministic_and_content_sensitive():
    investigation_a = _build_resolved_investigation()
    investigation_b = _build_resolved_investigation()
    _, trail_a = finalize_investigation(investigation_a)
    _, trail_b = finalize_investigation(investigation_b)
    # Same investigation_id and same trial content in both builds -> same digest.
    assert trail_a.trail_sha256 == trail_b.trail_sha256

    investigation_c = _build_resolved_investigation()
    investigation_c.investigation_id = "inv-closeout-test-different"
    _, trail_c = finalize_investigation(investigation_c)
    assert trail_c.trail_sha256 != trail_a.trail_sha256
