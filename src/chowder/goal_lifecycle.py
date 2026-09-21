from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from enum import Enum
from typing import Mapping, Sequence

from .goal_assessment import (
    GoalAssessment,
    GoalEvidenceRef,
    GoalStatus,
    MeasuredGoalAssessor,
)
from .improvement.constitution import Constitution, ConstitutionViolation, ObjectiveIdentity, goal_digest
from .models import ExperimentResult, Goal, MetricTarget
from .protocol import result_protocol_fingerprint
from .registry import RunRegistry


class GoalTerminalState(str, Enum):
    STOP_GOALS_MET = "STOP_GOALS_MET"
    STOP_BUDGET = "STOP_BUDGET"
    STOP_PLATEAU = "STOP_PLATEAU"
    STOP_GENERATION_LIMIT = "STOP_GENERATION_LIMIT"
    STOP_OPERATOR = "STOP_OPERATOR"
    STOP_UNCERTAIN = "STOP_UNCERTAIN"
    REQUIRES_HUMAN_REVIEW = "REQUIRES_HUMAN_REVIEW"


class GoalLifecycleError(ValueError):
    """Raised when a lifecycle transition violates the frozen objective."""


@dataclass(frozen=True)
class GoalLifecycleResult:
    assessment: GoalAssessment
    terminal_state: GoalTerminalState | None
    promoted: bool = False

    @property
    def succeeded(self) -> bool:
        return self.terminal_state is GoalTerminalState.STOP_GOALS_MET

    @property
    def can_continue(self) -> bool:
        return self.terminal_state is None


@dataclass
class GoalLifecycle:
    """Own one frozen objective's assessment and terminal-state transitions."""

    registry: RunRegistry
    goal: Goal
    identity: ObjectiveIdentity
    constitution: Constitution
    assessor: MeasuredGoalAssessor
    terminal_state: GoalTerminalState | None = None
    last_assessment: GoalAssessment | None = None

    @classmethod
    def open(
        cls,
        registry: RunRegistry,
        *,
        objective_version: str,
        goal: Goal,
        benchmark_digest: str,
        evaluation_protocol_digest: str,
        constitution: Constitution | None = None,
        objective_metadata: Mapping[str, object] | None = None,
        resume: bool = False,
    ) -> "GoalLifecycle":
        policy = constitution or Constitution()
        identity = policy.new_objective_identity(
            objective_version=objective_version,
            goal=goal,
            benchmark_digest=benchmark_digest,
            evaluation_protocol_digest=evaluation_protocol_digest,
        )
        metadata = dict(objective_metadata or {})
        frozen_goal_payload = _goal_payload(goal)
        frozen_goal_payload.update(metadata)
        stored = registry.get_goal_objective(objective_version)
        if stored is not None:
            stored_identity = _identity_from_dict(stored["identity"])
            try:
                stored_identity.require_match(identity)
            except ConstitutionViolation as exc:
                raise GoalLifecycleError(str(exc)) from exc
            if not resume:
                raise GoalLifecycleError(
                    f"objective version already exists: {objective_version}; resume explicitly"
                )
            stored_goal = stored["goal"]
            if not isinstance(stored_goal, Mapping):
                raise GoalLifecycleError("persisted objective payload is invalid")
            if any(
                stored_goal.get(key) != value
                for key, value in _goal_payload(goal).items()
            ):
                raise GoalLifecycleError("persisted goal changed; refusing to resume objective")
            for key, value in metadata.items():
                if key not in stored_goal:
                    raise GoalLifecycleError(
                        f"persisted objective is missing {key}; migration required before resume"
                    )
                if stored_goal[key] != value:
                    raise GoalLifecycleError(
                        f"objective identity changed ({key}); refusing to resume objective"
                    )
        else:
            if resume:
                raise GoalLifecycleError(
                    f"cannot resume unknown objective version: {objective_version}"
                )
            registry.record_goal_objective(identity, frozen_goal_payload)

        lifecycle = cls(
            registry=registry,
            goal=goal,
            identity=identity,
            constitution=policy,
            assessor=MeasuredGoalAssessor(),
        )
        if resume:
            lifecycle.last_assessment = registry.latest_goal_assessment(objective_version)
            if lifecycle.last_assessment is not None:
                lifecycle._assert_assessment_identity(lifecycle.last_assessment)
            terminal_raw = registry.latest_goal_terminal(objective_version)
            if terminal_raw is not None:
                if lifecycle.last_assessment is None:
                    raise GoalLifecycleError(
                        "persisted terminal state has no persisted goal assessment"
                    )
                try:
                    lifecycle.terminal_state = GoalTerminalState(terminal_raw)
                except ValueError as exc:
                    raise GoalLifecycleError(
                        f"persisted terminal state is invalid: {terminal_raw!r}"
                    ) from exc
        return lifecycle

    def _assert_assessment_identity(self, assessment: GoalAssessment) -> None:
        fields = (
            ("objective version", self.identity.objective_version, assessment.goal_version),
            ("goal", self.identity.goal_digest, assessment.goal_digest),
            (
                "evaluation protocol",
                self.identity.evaluation_protocol_digest,
                assessment.evaluation_protocol_digest,
            ),
            ("benchmark", self.identity.benchmark_digest, assessment.benchmark_digest),
            (
                "constitution",
                self.identity.constitution_digest,
                assessment.constitution_digest,
            ),
        )
        for label, expected, actual in fields:
            if actual != expected:
                raise GoalLifecycleError(
                    f"persisted assessment {label} identity does not match frozen objective"
                )

    def terminal_result(self) -> GoalLifecycleResult:
        """Return the persisted terminal result for a resumed objective."""
        if self.terminal_state is None or self.last_assessment is None:
            raise GoalLifecycleError("objective has no persisted terminal result")
        return GoalLifecycleResult(
            assessment=self.last_assessment,
            terminal_state=self.terminal_state,
        )

    def assess_parent(
        self,
        *,
        observed_metrics: Mapping[str, object],
        artifact_identity: str,
        evidence: Sequence[GoalEvidenceRef],
        evaluation_protocol_digest: str | None = None,
        benchmark_digest: str | None = None,
        measured_at: str | None = None,
    ) -> GoalLifecycleResult:
        """Assess the current parent before any candidate work is launched."""
        return self._assess(
            observed_metrics=observed_metrics,
            artifact_identity=artifact_identity,
            evidence=evidence,
            evaluation_protocol_digest=evaluation_protocol_digest,
            benchmark_digest=benchmark_digest,
            measured_at=measured_at,
            promoted=False,
            generation_index=0,
            generation_limit=None,
            budget_exhausted=False,
        )

    def assess_parent_result(self, result: ExperimentResult) -> GoalLifecycleResult:
        """Assess a baseline result before candidate work is launched."""
        evidence_digest = _result_evidence_digest(result)
        return self.assess_parent(
            observed_metrics=result.metrics,
            artifact_identity=result.artifact_ref or result.experiment_id,
            evidence=(GoalEvidenceRef(f"result:{result.experiment_id}", evidence_digest),),
            evaluation_protocol_digest=result_protocol_fingerprint(result.evidence),
            benchmark_digest=self.identity.benchmark_digest,
        )

    def assess_result(
        self,
        result: ExperimentResult,
        *,
        promoted: bool,
        generation_index: int,
        generation_limit: int | None = None,
        budget_exhausted: bool = False,
    ) -> GoalLifecycleResult:
        """Assess a persisted experiment result without launching evaluation."""
        return self.assess_candidate(
            observed_metrics=result.metrics,
            artifact_identity=result.artifact_ref or result.experiment_id,
            evidence=(GoalEvidenceRef(f"result:{result.experiment_id}", _result_evidence_digest(result)),),
            promoted=promoted,
            generation_index=generation_index,
            generation_limit=generation_limit,
            budget_exhausted=budget_exhausted,
            evaluation_protocol_digest=result_protocol_fingerprint(result.evidence),
            benchmark_digest=self.identity.benchmark_digest,
        )

    def assess_candidate(
        self,
        *,
        observed_metrics: Mapping[str, object],
        artifact_identity: str,
        evidence: Sequence[GoalEvidenceRef],
        promoted: bool,
        generation_index: int,
        generation_limit: int | None = None,
        budget_exhausted: bool = False,
        evaluation_protocol_digest: str | None = None,
        benchmark_digest: str | None = None,
        measured_at: str | None = None,
    ) -> GoalLifecycleResult:
        """Assess a candidate and turn bounded exhaustion into non-success."""
        if generation_index < 1:
            raise GoalLifecycleError("candidate generation_index must be at least 1")
        if generation_limit is not None and generation_limit < 1:
            raise GoalLifecycleError("generation_limit must be at least 1")
        return self._assess(
            observed_metrics=observed_metrics,
            artifact_identity=artifact_identity,
            evidence=evidence,
            evaluation_protocol_digest=evaluation_protocol_digest,
            benchmark_digest=benchmark_digest,
            measured_at=measured_at,
            promoted=promoted,
            generation_index=generation_index,
            generation_limit=generation_limit,
            budget_exhausted=budget_exhausted,
        )

    def _assess(
        self,
        *,
        observed_metrics: Mapping[str, object],
        artifact_identity: str,
        evidence: Sequence[GoalEvidenceRef],
        evaluation_protocol_digest: str | None,
        benchmark_digest: str | None,
        measured_at: str | None,
        promoted: bool,
        generation_index: int,
        generation_limit: int | None,
        budget_exhausted: bool,
    ) -> GoalLifecycleResult:
        if self.terminal_state is not None:
            raise GoalLifecycleError(
                f"objective already terminated: {self.terminal_state.value}"
            )
        if (
            evaluation_protocol_digest is not None
            and evaluation_protocol_digest != self.identity.evaluation_protocol_digest
        ):
            raise GoalLifecycleError("evaluation protocol changed within the objective")
        if benchmark_digest is not None and benchmark_digest != self.identity.benchmark_digest:
            raise GoalLifecycleError("benchmark changed within the objective")
        protocol_missing = evaluation_protocol_digest is None
        benchmark_missing = benchmark_digest is None
        assessment = self.assessor.assess(
            self.goal,
            goal_version=self.identity.objective_version,
            observed_metrics=observed_metrics,
            artifact_identity=artifact_identity,
            evaluation_protocol_digest=(
                evaluation_protocol_digest or self.identity.evaluation_protocol_digest
            ),
            benchmark_digest=benchmark_digest or self.identity.benchmark_digest,
            constitution_digest=self.identity.constitution_digest,
            evidence=evidence,
            measured_at=measured_at,
        )
        if protocol_missing or benchmark_missing:
            reasons = list(assessment.reason_codes)
            refusals = list(assessment.refusal_codes)
            if protocol_missing:
                reasons.append("MISSING_EVALUATION_PROTOCOL")
            if benchmark_missing:
                reasons.append("MISSING_BENCHMARK_DIGEST")
            refusals.append("INCOMPLETE_EVALUATION_IDENTITY")
            assessment = replace(
                assessment,
                status=(
                    GoalStatus.INVALID
                    if assessment.status is GoalStatus.INVALID
                    else GoalStatus.UNKNOWN
                ),
                reason_codes=tuple(dict.fromkeys(reasons)),
                refusal_codes=tuple(dict.fromkeys(refusals)),
            )
        self._assert_assessment_identity(assessment)
        self.registry.record_goal_assessment(self.identity.objective_version, assessment)
        self.last_assessment = assessment

        terminal: GoalTerminalState | None
        if assessment.status is GoalStatus.MET:
            terminal = GoalTerminalState.STOP_GOALS_MET
        elif assessment.status in {GoalStatus.UNKNOWN, GoalStatus.INVALID}:
            terminal = GoalTerminalState.STOP_UNCERTAIN
        elif budget_exhausted:
            terminal = GoalTerminalState.STOP_BUDGET
        elif generation_index == 0:
            terminal = None
        elif generation_limit is not None and generation_index >= generation_limit:
            terminal = GoalTerminalState.STOP_GENERATION_LIMIT
        elif not promoted:
            terminal = GoalTerminalState.STOP_PLATEAU
        else:
            terminal = None

        if terminal is not None:
            self.terminal_state = terminal
            self.registry.record_goal_terminal(
                self.identity.objective_version,
                terminal,
                assessment.artifact_identity,
            )
        return GoalLifecycleResult(assessment, terminal, promoted=promoted)


def _result_evidence_digest(result: ExperimentResult) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(result.evidence), sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
    ).hexdigest()


def _goal_payload(goal: Goal) -> dict[str, object]:
    return {
        "metrics": [
            {
                "name": metric.name,
                "minimum": metric.minimum,
                "maximum": metric.maximum,
                "weight": metric.weight,
                "regression_tolerance": metric.regression_tolerance,
                "direction": metric.direction.value,
            }
            for metric in goal.metrics
        ],
        "gpu_hour_budget": goal.gpu_hour_budget,
        "max_parallel_candidates": goal.max_parallel_candidates,
        "minimum_promotion_gain": goal.minimum_promotion_gain,
        "require_protocol_match": goal.require_protocol_match,
    }


def _identity_from_dict(payload: Mapping[str, object]) -> ObjectiveIdentity:
    return ObjectiveIdentity(
        objective_version=str(payload["objective_version"]),
        goal_digest=str(payload["goal_digest"]),
        benchmark_digest=str(payload["benchmark_digest"]),
        evaluation_protocol_digest=str(payload["evaluation_protocol_digest"]),
        constitution_digest=str(payload["constitution_digest"]),
    )


__all__ = [
    "GoalLifecycle",
    "GoalLifecycleError",
    "GoalLifecycleResult",
    "GoalTerminalState",
]
