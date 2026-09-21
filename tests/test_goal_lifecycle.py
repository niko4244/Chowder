from dataclasses import replace
from hashlib import sha256
import json

import pytest

from chowder.goal_assessment import GoalEvidenceRef, GoalStatus
from chowder.goal_lifecycle import GoalLifecycle, GoalLifecycleError, GoalTerminalState
from chowder.models import ExperimentResult, Goal, MetricTarget
from chowder.registry import RegistryInvariantError, RunRegistry


PROTOCOL = "a" * 64
BENCHMARK = "b" * 64
EVIDENCE = (GoalEvidenceRef("evaluation.json", "e" * 64),)


def _goal() -> Goal:
    return Goal((MetricTarget("quality", minimum=0.8),), gpu_hour_budget=2.0)


def _open(tmp_path, *, version="objective-1", goal=None, benchmark=BENCHMARK):
    registry = RunRegistry(tmp_path / "runs.db")
    lifecycle = GoalLifecycle.open(
        registry,
        objective_version=version,
        goal=goal or _goal(),
        benchmark_digest=benchmark,
        evaluation_protocol_digest=PROTOCOL,
    )
    return registry, lifecycle


def test_parent_already_meets_goal_stops_before_work_and_persists_assessment(tmp_path):
    registry, lifecycle = _open(tmp_path)
    try:
        result = lifecycle.assess_parent(
            observed_metrics={"quality": 0.9},
            artifact_identity="baseline",
            evidence=EVIDENCE,
            evaluation_protocol_digest=PROTOCOL,
            benchmark_digest=BENCHMARK,
            measured_at="2026-09-20T00:00:00+00:00",
        )

        assert result.assessment.status is GoalStatus.MET
        assert result.terminal_state is GoalTerminalState.STOP_GOALS_MET
        assert result.succeeded
        assert result.can_continue is False
        assert len(registry.list_goal_assessments("objective-1")) == 1
        assert registry.list_goal_terminals("objective-1")[0]["terminal_state"] == "STOP_GOALS_MET"
    finally:
        registry.close()


def test_promoted_improvement_below_goal_continues_until_generation_limit(tmp_path):
    registry, lifecycle = _open(tmp_path)
    try:
        parent = lifecycle.assess_parent(
            observed_metrics={"quality": 0.5},
            artifact_identity="baseline",
            evidence=EVIDENCE,
            evaluation_protocol_digest=PROTOCOL,
            benchmark_digest=BENCHMARK,
        )
        assert parent.assessment.status is GoalStatus.UNMET
        assert parent.can_continue

        candidate = lifecycle.assess_candidate(
            observed_metrics={"quality": 0.6},
            artifact_identity="candidate-1",
            evidence=EVIDENCE,
            promoted=True,
            generation_index=1,
            generation_limit=1,
            evaluation_protocol_digest=PROTOCOL,
            benchmark_digest=BENCHMARK,
        )

        assert candidate.assessment.status is GoalStatus.UNMET
        assert candidate.promoted is True
        assert candidate.terminal_state is GoalTerminalState.STOP_GENERATION_LIMIT
        assert candidate.succeeded is False
    finally:
        registry.close()


def test_unknown_or_incomplete_evaluation_stops_uncertain_not_success(tmp_path):
    registry, lifecycle = _open(tmp_path)
    try:
        result = lifecycle.assess_parent(
            observed_metrics={},
            artifact_identity="baseline",
            evidence=(),
            evaluation_protocol_digest=None,
            benchmark_digest=BENCHMARK,
        )

        assert result.assessment.status is GoalStatus.UNKNOWN
        assert result.terminal_state is GoalTerminalState.STOP_UNCERTAIN
        assert result.succeeded is False
        assert "MISSING_MEASUREMENT" in result.assessment.metric_assessments[0].reason_codes
    finally:
        registry.close()


@pytest.mark.parametrize(
    "field",
    ("goal_digest", "evaluation_protocol_digest", "benchmark_digest", "constitution_digest", "goal_version"),
)
def test_registry_rejects_assessment_identity_mismatch(tmp_path, field):
    registry, lifecycle = _open(tmp_path)
    try:
        valid = lifecycle.assess_parent(
            observed_metrics={"quality": 0.5},
            artifact_identity="baseline",
            evidence=EVIDENCE,
            evaluation_protocol_digest=PROTOCOL,
            benchmark_digest=BENCHMARK,
        ).assessment
        replacement = "other-objective" if field == "goal_version" else "d" * 64
        forged = replace(valid, **{field: replacement})
        with pytest.raises(RegistryInvariantError, match="does not match frozen objective|lifecycle"):
            registry.record_goal_assessment("objective-1", forged)
    finally:
        registry.close()


@pytest.mark.parametrize(
    "field",
    ("goal_digest", "evaluation_protocol_digest", "benchmark_digest", "constitution_digest", "goal_version"),
)
def test_resume_rejects_persisted_assessment_identity_mismatch(tmp_path, field):
    registry, lifecycle = _open(tmp_path)
    valid = lifecycle.assess_parent(
        observed_metrics={"quality": 0.5},
        artifact_identity="baseline",
        evidence=EVIDENCE,
        evaluation_protocol_digest=PROTOCOL,
        benchmark_digest=BENCHMARK,
    ).assessment
    replacement = "other-objective" if field == "goal_version" else "d" * 64
    forged = replace(valid, **{field: replacement})
    payload = registry._json(forged.to_dict())
    assessment_id = sha256(payload.encode("utf-8")).hexdigest()
    registry._conn.execute(
        "INSERT INTO goal_assessments "
        "(assessment_id, objective_version, artifact_identity, status, assessment_json, recorded_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (assessment_id, "objective-1", forged.artifact_identity, forged.status.value, payload, "now"),
    )
    registry._conn.commit()
    registry.close()

    with RunRegistry(tmp_path / "runs.db") as resumed_registry:
        with pytest.raises(GoalLifecycleError, match="identity does not match"):
            GoalLifecycle.open(
                resumed_registry,
                objective_version="objective-1",
                goal=_goal(),
                benchmark_digest=BENCHMARK,
                evaluation_protocol_digest=PROTOCOL,
                resume=True,
            )


def test_resume_rehydrates_terminal_result_and_refuses_new_candidate_work(tmp_path):
    registry, lifecycle = _open(tmp_path)
    first = lifecycle.assess_parent(
        observed_metrics={"quality": 0.9},
        artifact_identity="baseline",
        evidence=EVIDENCE,
        evaluation_protocol_digest=PROTOCOL,
        benchmark_digest=BENCHMARK,
        measured_at="2026-09-20T00:00:00+00:00",
    )
    assert first.terminal_state is GoalTerminalState.STOP_GOALS_MET
    registry.close()

    with RunRegistry(tmp_path / "runs.db") as resumed_registry:
        resumed = GoalLifecycle.open(
            resumed_registry,
            objective_version="objective-1",
            goal=_goal(),
            benchmark_digest=BENCHMARK,
            evaluation_protocol_digest=PROTOCOL,
            resume=True,
        )
        assert resumed.identity.goal_digest
        assert resumed.terminal_state is GoalTerminalState.STOP_GOALS_MET
        assert resumed.last_assessment is not None
        assert resumed.last_assessment.status is GoalStatus.MET
        assert resumed.terminal_result().succeeded
        with pytest.raises(GoalLifecycleError, match="already terminated"):
            resumed.assess_candidate(
                observed_metrics={"quality": 0.95},
                artifact_identity="candidate-1",
                evidence=EVIDENCE,
                promoted=True,
                generation_index=1,
                evaluation_protocol_digest=PROTOCOL,
                benchmark_digest=BENCHMARK,
            )

        with pytest.raises(GoalLifecycleError, match="identity changed"):
            GoalLifecycle.open(
                resumed_registry,
                objective_version="objective-1",
                goal=_goal(),
                benchmark_digest="d" * 64,
                evaluation_protocol_digest=PROTOCOL,
                resume=True,
            )


def test_promotion_is_not_completion_and_budget_stop_is_not_success(tmp_path):
    registry, lifecycle = _open(tmp_path)
    try:
        lifecycle.assess_parent(
            observed_metrics={"quality": 0.5},
            artifact_identity="baseline",
            evidence=EVIDENCE,
            evaluation_protocol_digest=PROTOCOL,
            benchmark_digest=BENCHMARK,
        )
        result = lifecycle.assess_candidate(
            observed_metrics={"quality": 0.7},
            artifact_identity="candidate-1",
            evidence=EVIDENCE,
            promoted=True,
            generation_index=1,
            budget_exhausted=True,
            evaluation_protocol_digest=PROTOCOL,
            benchmark_digest=BENCHMARK,
        )

        assert result.assessment.status is GoalStatus.UNMET
        assert result.terminal_state is GoalTerminalState.STOP_BUDGET
        assert result.succeeded is False
    finally:
        registry.close()


def test_lifecycle_identity_is_stable_across_result_assessments(tmp_path):
    registry, lifecycle = _open(tmp_path)
    try:
        lifecycle.assess_parent(
            observed_metrics={"quality": 0.5},
            artifact_identity="baseline",
            evidence=EVIDENCE,
            evaluation_protocol_digest=PROTOCOL,
            benchmark_digest=BENCHMARK,
        )
        result = ExperimentResult(
            "candidate-1",
            {"quality": 0.9},
            0.2,
            artifact_ref="artifact://candidate-1",
            evidence={"evaluation_protocol_sha256": PROTOCOL},
        )
        assessed = lifecycle.assess_result(result, promoted=True, generation_index=1)
        assert assessed.succeeded
        assert assessed.assessment.goal_digest == lifecycle.identity.goal_digest
    finally:
        registry.close()
