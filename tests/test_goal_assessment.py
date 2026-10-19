from dataclasses import replace

import pytest

from chowder.goal_assessment import (
    GoalAssessment,
    GoalEvidenceRef,
    GoalStatus,
    MeasuredGoalAssessor,
)
from chowder.improvement.constitution import (
    Constitution,
    ConstitutionViolation,
    ProtectedSurface,
)
from chowder.models import Goal, MetricTarget, OptimizationDirection


DIGESTS = {
    "benchmark": "b" * 64,
    "protocol": "c" * 64,
}


def _goal() -> Goal:
    return Goal(
        metrics=(
            MetricTarget("quality", minimum=0.8),
            MetricTarget(
                "latency_ms",
                maximum=100,
                direction=OptimizationDirection.MINIMIZE,
            ),
        ),
        gpu_hour_budget=2,
    )


def _evidence() -> tuple[GoalEvidenceRef, ...]:
    return (GoalEvidenceRef("runs/candidate.json", "a" * 64),)


def test_objective_identity_is_stable_and_resume_requires_exact_match():
    constitution = Constitution()
    identity = constitution.new_objective_identity(
        objective_version="objective-1",
        goal=_goal(),
        benchmark_digest=DIGESTS["benchmark"],
        evaluation_protocol_digest=DIGESTS["protocol"],
    )
    same = constitution.new_objective_identity(
        objective_version="objective-1",
        goal=_goal(),
        benchmark_digest=DIGESTS["benchmark"],
        evaluation_protocol_digest=DIGESTS["protocol"],
    )
    changed = constitution.new_objective_identity(
        objective_version="objective-2",
        goal=_goal(),
        benchmark_digest=DIGESTS["benchmark"],
        evaluation_protocol_digest=DIGESTS["protocol"],
    )

    constitution.assert_resume_allowed(identity, same)
    with pytest.raises(ConstitutionViolation, match="identity changed"):
        constitution.assert_resume_allowed(identity, changed)


def test_protected_change_requires_human_approval_and_new_objective_version():
    constitution = Constitution()
    current = constitution.new_objective_identity(
        objective_version="objective-1",
        goal=_goal(),
        benchmark_digest=DIGESTS["benchmark"],
        evaluation_protocol_digest=DIGESTS["protocol"],
    )
    proposed = constitution.new_objective_identity(
        objective_version="objective-2",
        goal=_goal(),
        benchmark_digest="d" * 64,
        evaluation_protocol_digest=DIGESTS["protocol"],
    )

    with pytest.raises(ConstitutionViolation, match="human approval"):
        constitution.assert_changes_allowed(
            [ProtectedSurface.PROTECTED_BENCHMARK],
            current=current,
            proposed=proposed,
        )
    constitution.assert_changes_allowed(
        [ProtectedSurface.PROTECTED_BENCHMARK],
        current=current,
        proposed=proposed,
        human_approved=True,
    )


def test_approved_protected_change_cannot_reuse_objective_version():
    constitution = Constitution()
    current = constitution.new_objective_identity(
        objective_version="objective-1",
        goal=_goal(),
        benchmark_digest=DIGESTS["benchmark"],
        evaluation_protocol_digest=DIGESTS["protocol"],
    )
    with pytest.raises(ConstitutionViolation, match="new objective version"):
        constitution.assert_changes_allowed(
            [ProtectedSurface.GOAL_THRESHOLDS],
            current=current,
            proposed=current,
            human_approved=True,
        )


def _complete_assessment() -> GoalAssessment:
    return MeasuredGoalAssessor().assess(
        Goal((MetricTarget("quality", minimum=0.8),), gpu_hour_budget=1),
        goal_version="objective-1",
        observed_metrics={"quality": 0.9},
        artifact_identity="candidate-1",
        evaluation_protocol_digest=DIGESTS["protocol"],
        benchmark_digest=DIGESTS["benchmark"],
        evidence=_evidence(),
        measured_at="2026-09-20T00:00:00+00:00",
    )


def test_forged_success_without_metrics_or_evidence_is_rejected():
    with pytest.raises(ValueError, match="at least one metric"):
        GoalAssessment(
            goal_version="objective-1",
            goal_digest="a" * 64,
            evaluation_protocol_digest=DIGESTS["protocol"],
            benchmark_digest=DIGESTS["benchmark"],
            artifact_identity="candidate-1",
            metric_assessments=(),
            status=GoalStatus.MET,
            evidence=(),
            measured_at="2026-09-20T00:00:00+00:00",
        )


def test_forged_overall_status_must_match_metric_statuses():
    assessment = _complete_assessment()
    for status in (GoalStatus.UNMET, GoalStatus.UNKNOWN, GoalStatus.INVALID):
        with pytest.raises(ValueError, match="inconsistent"):
            replace(assessment, status=status)


def test_assessor_reports_met_only_with_complete_finite_evidence():
    assessment = MeasuredGoalAssessor().assess(
        _goal(),
        goal_version="objective-1",
        observed_metrics={"quality": 0.9, "latency_ms": 80},
        artifact_identity="candidate-1",
        evaluation_protocol_digest=DIGESTS["protocol"],
        benchmark_digest=DIGESTS["benchmark"],
        evidence=_evidence(),
        measured_at="2026-09-20T00:00:00+00:00",
    )

    assert assessment.status is GoalStatus.MET
    assert assessment.is_success
    assert assessment.goal_digest
    assert assessment.metric_assessments[0].required_minimum == 0.8
    assert assessment.to_dict()["evidence"] == [
        {"path": "runs/candidate.json", "digest": "a" * 64, "kind": "evaluation"}
    ]


def test_assessor_distinguishes_unmet_unknown_and_invalid_without_inventing_zero():
    assessor = MeasuredGoalAssessor()
    base = dict(
        goal_version="objective-1",
        artifact_identity="candidate-1",
        evaluation_protocol_digest=DIGESTS["protocol"],
        benchmark_digest=DIGESTS["benchmark"],
        evidence=_evidence(),
    )

    unmet = assessor.assess(_goal(), observed_metrics={"quality": 0.7, "latency_ms": 80}, **base)
    assert unmet.status is GoalStatus.UNMET
    assert unmet.metric_assessments[0].observed_value == 0.7

    unknown = assessor.assess(_goal(), observed_metrics={"quality": 0.9}, **base)
    assert unknown.status is GoalStatus.UNKNOWN
    assert unknown.metric_assessments[1].observed_value is None
    assert "MISSING_MEASUREMENT" in unknown.metric_assessments[1].reason_codes

    invalid = assessor.assess(
        _goal(), observed_metrics={"quality": float("nan"), "latency_ms": 80}, **base
    )
    assert invalid.status is GoalStatus.INVALID
    assert "NON_FINITE_MEASUREMENT" in invalid.metric_assessments[0].reason_codes


def test_malformed_protocol_or_benchmark_digest_is_invalid_not_success():
    assessment = MeasuredGoalAssessor().assess(
        Goal((MetricTarget("quality", minimum=0.8),), gpu_hour_budget=1),
        goal_version="objective-1",
        observed_metrics={"quality": 0.9},
        artifact_identity="candidate-1",
        evaluation_protocol_digest="not-a-digest",
        benchmark_digest=DIGESTS["benchmark"],
        evidence=_evidence(),
    )

    assert assessment.status is GoalStatus.INVALID
    assert "INVALID_EVALUATION_PROTOCOL" in assessment.reason_codes
    assert "INVALID_EVALUATION_IDENTITY" in assessment.refusal_codes


def test_missing_protocol_or_evidence_refuses_success_and_preserves_reason_codes():
    assessment = MeasuredGoalAssessor().assess(
        Goal((MetricTarget("quality", minimum=0.8),), gpu_hour_budget=1),
        goal_version="objective-1",
        observed_metrics={"quality": 0.9},
        artifact_identity="candidate-1",
        evaluation_protocol_digest=None,
        benchmark_digest=DIGESTS["benchmark"],
        evidence=(),
    )

    assert assessment.status is GoalStatus.UNKNOWN
    assert not assessment.is_success
    assert "MISSING_EVALUATION_PROTOCOL" in assessment.reason_codes
    assert "INCOMPLETE_EVALUATION_IDENTITY" in assessment.refusal_codes
    assert "INCOMPLETE_EVIDENCE" in assessment.refusal_codes
