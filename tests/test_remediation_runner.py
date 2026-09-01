from dataclasses import dataclass, field
from typing import Any, Mapping

import pytest

from chowder.incident import compute_fingerprint
from chowder.investigation import HypothesisTrial, Investigation, RemediationOutcome
from chowder.models import Hypothesis
from chowder.remediation_runner import RemediationExperiment
from chowder.replay import GroundTruthMissingError, ReplayExecutor, ReplayGroundTruth

from fixtures_incidents import PEFT_KBIT_PREP_OOM, QWEN3_5_CONV1D_NO_ENGINE


def _investigation(capture, gpu_hour_budget: float = 1.0) -> Investigation:
    return Investigation(
        investigation_id="inv-runner-test",
        fingerprint=compute_fingerprint(capture),
        capture=capture,
        gpu_hour_budget=gpu_hour_budget,
    )


def _trial(config_patch: Mapping[str, Any]) -> HypothesisTrial:
    return HypothesisTrial(
        hypothesis=Hypothesis(
            observation="o", suspected_cause="c", intervention="disable cuDNN"
        ),
        config_patch=config_patch,
    )


def test_correct_patch_resolves():
    good_patch = {"cudnn_enabled": False}
    truth = ReplayGroundTruth(
        fingerprint_sha256="x",
        outcomes={_digest(good_patch): RemediationOutcome.RESOLVED},
    )
    experiment = RemediationExperiment(executor=ReplayExecutor(truth))
    investigation = _investigation(QWEN3_5_CONV1D_NO_ENGINE)
    record = experiment.run(
        investigation,
        _trial(good_patch),
        environment=QWEN3_5_CONV1D_NO_ENGINE.environment,
    )
    assert record.outcome is RemediationOutcome.RESOLVED
    assert record.attempts_used == 1
    assert record.context_sha256 is not None


def test_wrong_patch_does_not_resolve():
    wrong_patch = {"batch_size": 1}
    truth = ReplayGroundTruth(
        fingerprint_sha256="x",
        outcomes={_digest(wrong_patch): RemediationOutcome.DID_NOT_RESOLVE},
    )
    experiment = RemediationExperiment(executor=ReplayExecutor(truth))
    investigation = _investigation(QWEN3_5_CONV1D_NO_ENGINE)
    record = experiment.run(
        investigation,
        _trial(wrong_patch),
        environment=QWEN3_5_CONV1D_NO_ENGINE.environment,
    )
    assert record.outcome is RemediationOutcome.DID_NOT_RESOLVE


def test_unlisted_patch_propagates_ground_truth_missing_not_swallowed():
    truth = ReplayGroundTruth(fingerprint_sha256="x", outcomes={})
    experiment = RemediationExperiment(executor=ReplayExecutor(truth))
    investigation = _investigation(QWEN3_5_CONV1D_NO_ENGINE)
    with pytest.raises(GroundTruthMissingError):
        experiment.run(
            investigation,
            _trial({"never": "defined"}),
            environment=QWEN3_5_CONV1D_NO_ENGINE.environment,
        )


def test_budget_exhausted_before_running_raises():
    investigation = _investigation(QWEN3_5_CONV1D_NO_ENGINE, gpu_hour_budget=0.0)
    truth = ReplayGroundTruth(fingerprint_sha256="x", outcomes={})
    experiment = RemediationExperiment(executor=ReplayExecutor(truth))
    with pytest.raises(ValueError, match="no remaining GPU-hour budget"):
        experiment.run(
            investigation,
            _trial({"x": 1}),
            environment=QWEN3_5_CONV1D_NO_ENGINE.environment,
        )


@dataclass
class _FlakyThenSucceedsExecutor:
    calls: list[Mapping[str, Any]] = field(default_factory=list)

    def run(self, config_patch: Mapping[str, Any]) -> RemediationOutcome:
        self.calls.append(config_patch)
        if len(self.calls) < 2:
            return RemediationOutcome.DID_NOT_RESOLVE
        return RemediationOutcome.RESOLVED


def test_max_attempts_allows_a_later_attempt_to_resolve():
    executor = _FlakyThenSucceedsExecutor()
    experiment = RemediationExperiment(executor=executor, max_attempts=2)
    investigation = _investigation(QWEN3_5_CONV1D_NO_ENGINE)
    record = experiment.run(
        investigation,
        _trial({"x": 1}),
        environment=QWEN3_5_CONV1D_NO_ENGINE.environment,
    )
    assert record.outcome is RemediationOutcome.RESOLVED
    assert record.attempts_used == 2
    assert len(executor.calls) == 2


def test_attempt_loop_never_exceeds_remaining_gpu_budget():
    executor = _FlakyThenSucceedsExecutor()
    experiment = RemediationExperiment(
        executor=executor,
        max_attempts=3,
        gpu_hours_per_attempt=0.05,
    )
    investigation = _investigation(QWEN3_5_CONV1D_NO_ENGINE, gpu_hour_budget=0.05)
    record = experiment.run(
        investigation,
        _trial({"x": 1}),
        environment=QWEN3_5_CONV1D_NO_ENGINE.environment,
    )
    assert record.outcome is RemediationOutcome.DID_NOT_RESOLVE
    assert record.attempts_used == 1
    assert record.gpu_hours_spent == pytest.approx(0.05)
    assert len(executor.calls) == 1


def test_attempt_rejected_when_one_attempt_cannot_fit_budget():
    experiment = RemediationExperiment(
        executor=_FlakyThenSucceedsExecutor(),
        max_attempts=2,
        gpu_hours_per_attempt=0.06,
    )
    investigation = _investigation(QWEN3_5_CONV1D_NO_ENGINE, gpu_hour_budget=0.05)
    with pytest.raises(ValueError, match="insufficient remaining GPU-hour budget"):
        experiment.run(
            investigation,
            _trial({"x": 1}),
            environment=QWEN3_5_CONV1D_NO_ENGINE.environment,
        )


def test_max_attempts_of_one_does_not_retry():
    executor = _FlakyThenSucceedsExecutor()
    experiment = RemediationExperiment(executor=executor, max_attempts=1)
    investigation = _investigation(QWEN3_5_CONV1D_NO_ENGINE)
    record = experiment.run(
        investigation,
        _trial({"x": 1}),
        environment=QWEN3_5_CONV1D_NO_ENGINE.environment,
    )
    assert record.outcome is RemediationOutcome.DID_NOT_RESOLVE
    assert record.attempts_used == 1
    assert len(executor.calls) == 1


@dataclass
class _CrashingExecutor:
    def run(self, config_patch: Mapping[str, Any]) -> RemediationOutcome:
        raise RuntimeError("a genuinely new problem, not in any ground truth")


def test_executor_crash_preserves_full_spawned_incident():
    experiment = RemediationExperiment(executor=_CrashingExecutor())
    investigation = _investigation(PEFT_KBIT_PREP_OOM)
    record = experiment.run(
        investigation,
        _trial({"max_length": 4096}),
        environment=PEFT_KBIT_PREP_OOM.environment,
    )
    assert record.outcome is RemediationOutcome.PARTIALLY_RESOLVED
    assert record.spawned_incident is not None
    assert record.spawned_incident.exception_type == "RuntimeError"
    assert "genuinely new problem" in record.spawned_incident.exception_message
    assert record.spawned_incident.environment == PEFT_KBIT_PREP_OOM.environment
    assert "spawned_signature_kind=" in record.notes


def _digest(patch: Mapping[str, Any]) -> str:
    from chowder.investigation import config_patch_digest

    return config_patch_digest(patch)
