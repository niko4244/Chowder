from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Mapping, Protocol, Sequence

from .improvement.constitution import goal_digest
from .models import Goal, MetricTarget, OptimizationDirection


class GoalStatus(str, Enum):
    MET = "MET"
    UNMET = "UNMET"
    UNKNOWN = "UNKNOWN"
    INVALID = "INVALID"


@dataclass(frozen=True)
class GoalEvidenceRef:
    """A content-addressed pointer to evidence used by an assessment."""

    path: str
    digest: str
    kind: str = "evaluation"

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "GoalEvidenceRef":
        if not isinstance(payload, Mapping):
            raise ValueError("evidence reference must be a mapping")
        path = payload.get("path", "")
        digest = payload.get("digest", "")
        kind = payload.get("kind", "evaluation")
        if not all(isinstance(value, str) for value in (path, digest, kind)):
            raise ValueError("evidence reference fields must be strings")
        return cls(path=path, digest=digest, kind=kind)

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path.strip():
            raise ValueError("evidence path must be non-empty")
        if len(self.digest) != 64 or any(char not in "0123456789abcdef" for char in self.digest):
            raise ValueError("evidence digest must be a lowercase SHA-256 digest")
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise ValueError("evidence kind must be non-empty")


@dataclass(frozen=True)
class GoalMetricAssessment:
    metric_name: str
    direction: OptimizationDirection
    required_minimum: float | None
    required_maximum: float | None
    observed_value: float | None
    regression_tolerance: float
    status: GoalStatus
    evidence: tuple[GoalEvidenceRef, ...] = ()
    reason_codes: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "GoalMetricAssessment":
        if not isinstance(payload, Mapping):
            raise ValueError("metric assessment must be a mapping")
        try:
            direction = OptimizationDirection(payload["direction"])
            status = GoalStatus(payload["status"])
            evidence_payload = payload.get("evidence", ())
            reason_codes = payload.get("reason_codes", ())
            if not isinstance(evidence_payload, (list, tuple)):
                raise ValueError("metric assessment evidence must be a list")
            if not isinstance(reason_codes, (list, tuple)):
                raise ValueError("metric assessment reason_codes must be a list")
            return cls(
                metric_name=payload["metric_name"],
                direction=direction,
                required_minimum=payload.get("required_minimum"),
                required_maximum=payload.get("required_maximum"),
                observed_value=payload.get("observed_value"),
                regression_tolerance=payload["regression_tolerance"],
                status=status,
                evidence=tuple(GoalEvidenceRef.from_dict(item) for item in evidence_payload),
                reason_codes=tuple(str(item) for item in reason_codes),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid metric assessment: {exc}") from exc

    def __post_init__(self) -> None:
        if not isinstance(self.metric_name, str) or not self.metric_name.strip():
            raise ValueError("metric_name must be a non-empty string")
        if not isinstance(self.direction, OptimizationDirection):
            raise ValueError("metric direction must be an OptimizationDirection")
        if not isinstance(self.status, GoalStatus):
            raise ValueError("metric status must be a GoalStatus")
        try:
            target = MetricTarget(
                self.metric_name,
                minimum=self.required_minimum,
                maximum=self.required_maximum,
                regression_tolerance=self.regression_tolerance,
                direction=self.direction,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid metric assessment target: {exc}") from exc

        if self.status in {GoalStatus.MET, GoalStatus.UNMET}:
            if self.observed_value is None or not math.isfinite(float(self.observed_value)):
                raise ValueError(
                    f"{self.status.value} metric {self.metric_name!r} requires a finite observed value"
                )
            if self.status is GoalStatus.MET and not target.target_met(float(self.observed_value)):
                raise ValueError(f"MET metric {self.metric_name!r} is below its target")
            if self.status is GoalStatus.UNMET and target.target_met(float(self.observed_value)):
                raise ValueError(f"UNMET metric {self.metric_name!r} satisfies its target")
            if not self.evidence:
                raise ValueError(
                    f"{self.status.value} metric {self.metric_name!r} requires evidence"
                )
        elif self.status is GoalStatus.UNKNOWN:
            if self.observed_value is not None and not math.isfinite(float(self.observed_value)):
                raise ValueError(
                    f"UNKNOWN metric {self.metric_name!r} cannot contain a non-finite value"
                )
        elif self.status is GoalStatus.INVALID:
            if self.observed_value is not None:
                try:
                    finite = math.isfinite(float(self.observed_value))
                except (TypeError, ValueError):
                    finite = False
                if finite:
                    raise ValueError(
                        f"INVALID metric {self.metric_name!r} cannot contain a finite value"
                    )
        else:
            raise ValueError(f"unsupported metric status: {self.status!r}")


@dataclass(frozen=True)
class GoalAssessment:
    """Immutable result of comparing measured evidence with a frozen goal."""

    goal_version: str
    goal_digest: str
    evaluation_protocol_digest: str | None
    benchmark_digest: str | None
    artifact_identity: str
    metric_assessments: tuple[GoalMetricAssessment, ...]
    status: GoalStatus
    constitution_digest: str | None = None
    evidence: tuple[GoalEvidenceRef, ...] = ()
    measured_at: str = ""
    reason_codes: tuple[str, ...] = ()
    refusal_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.goal_version, str) or not self.goal_version.strip():
            raise ValueError("goal_version must be a non-empty string")
        if not isinstance(self.status, GoalStatus):
            raise ValueError("assessment status must be a GoalStatus")
        if not _is_digest(self.goal_digest):
            raise ValueError("goal_digest must be a lowercase SHA-256 digest")
        if not isinstance(self.artifact_identity, str) or not self.artifact_identity.strip():
            raise ValueError("artifact_identity must be a non-empty string")
        if not self.measured_at:
            raise ValueError("measured_at must be recorded")
        if not self.metric_assessments:
            raise ValueError("goal assessment requires at least one metric assessment")
        metric_names = [metric.metric_name for metric in self.metric_assessments]
        if len(metric_names) != len(set(metric_names)):
            raise ValueError("goal assessment metric names must be unique")

        protocol_valid = self.evaluation_protocol_digest is not None and _is_digest(
            self.evaluation_protocol_digest
        )
        benchmark_valid = self.benchmark_digest is not None and _is_digest(self.benchmark_digest)
        constitution_valid = self.constitution_digest is not None and _is_digest(
            self.constitution_digest
        )
        protocol_missing = self.evaluation_protocol_digest is None
        benchmark_missing = self.benchmark_digest is None
        identity_invalid = (
            self.evaluation_protocol_digest is not None and not protocol_valid
        ) or (self.benchmark_digest is not None and not benchmark_valid) or (
            self.constitution_digest is not None and not constitution_valid
        )
        evidence_complete = bool(self.evidence) and all(
            metric.evidence for metric in self.metric_assessments
        )
        identity_complete = protocol_valid and benchmark_valid and evidence_complete
        metric_statuses = {metric.status for metric in self.metric_assessments}
        identity_refused = "INCOMPLETE_EVALUATION_IDENTITY" in self.refusal_codes
        if GoalStatus.INVALID in metric_statuses or identity_invalid:
            expected_status = GoalStatus.INVALID
        elif GoalStatus.UNKNOWN in metric_statuses or not identity_complete or identity_refused:
            expected_status = GoalStatus.UNKNOWN
        elif GoalStatus.UNMET in metric_statuses:
            expected_status = GoalStatus.UNMET
        else:
            expected_status = GoalStatus.MET
        if self.status is not expected_status:
            raise ValueError(
                f"assessment status {self.status.value!r} is inconsistent; "
                f"expected {expected_status.value!r}"
            )
        if self.status in {GoalStatus.MET, GoalStatus.UNMET} and not identity_complete:
            raise ValueError(
                f"{self.status.value} assessment requires complete evidence and evaluation identity"
            )
        if self.status is GoalStatus.UNKNOWN and not (
            protocol_missing
            or benchmark_missing
            or not evidence_complete
            or identity_refused
        ) and not metric_statuses.intersection({GoalStatus.UNKNOWN}):
            raise ValueError("UNKNOWN assessment has no refusal or unknown measurement basis")

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "GoalAssessment":
        if not isinstance(payload, Mapping):
            raise ValueError("goal assessment must be a mapping")
        try:
            metrics_payload = payload["metrics"]
            evidence_payload = payload.get("evidence", ())
            reason_codes = payload.get("reason_codes", ())
            refusal_codes = payload.get("refusal_codes", ())
            if not isinstance(metrics_payload, (list, tuple)):
                raise ValueError("goal assessment metrics must be a list")
            if not isinstance(evidence_payload, (list, tuple)):
                raise ValueError("goal assessment evidence must be a list")
            if not isinstance(reason_codes, (list, tuple)):
                raise ValueError("goal assessment reason_codes must be a list")
            if not isinstance(refusal_codes, (list, tuple)):
                raise ValueError("goal assessment refusal_codes must be a list")
            return cls(
                goal_version=payload["goal_version"],
                goal_digest=payload["goal_digest"],
                evaluation_protocol_digest=payload.get("evaluation_protocol_digest"),
                benchmark_digest=payload.get("benchmark_digest"),
                artifact_identity=payload["artifact_identity"],
                constitution_digest=payload.get("constitution_digest"),
                metric_assessments=tuple(
                    GoalMetricAssessment.from_dict(item) for item in metrics_payload
                ),
                status=GoalStatus(payload["status"]),
                evidence=tuple(GoalEvidenceRef.from_dict(item) for item in evidence_payload),
                measured_at=payload["measured_at"],
                reason_codes=tuple(str(item) for item in reason_codes),
                refusal_codes=tuple(str(item) for item in refusal_codes),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid goal assessment: {exc}") from exc

    @property
    def is_success(self) -> bool:
        return self.status is GoalStatus.MET

    def require_success(self) -> None:
        if not self.is_success:
            raise ValueError(
                f"goal is not complete: status={self.status.value}; "
                f"reasons={','.join(self.reason_codes) or 'none'}"
            )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible durable representation."""
        return {
            "goal_version": self.goal_version,
            "goal_digest": self.goal_digest,
            "evaluation_protocol_digest": self.evaluation_protocol_digest,
            "benchmark_digest": self.benchmark_digest,
            "constitution_digest": self.constitution_digest,
            "artifact_identity": self.artifact_identity,
            "status": self.status.value,
            "measured_at": self.measured_at,
            "reason_codes": list(self.reason_codes),
            "refusal_codes": list(self.refusal_codes),
            "evidence": [_evidence_dict(ref) for ref in self.evidence],
            "metrics": [
                {
                    "metric_name": metric.metric_name,
                    "direction": metric.direction.value,
                    "required_minimum": metric.required_minimum,
                    "required_maximum": metric.required_maximum,
                    "observed_value": metric.observed_value,
                    "regression_tolerance": metric.regression_tolerance,
                    "status": metric.status.value,
                    "reason_codes": list(metric.reason_codes),
                    "evidence": [_evidence_dict(ref) for ref in metric.evidence],
                }
                for metric in self.metric_assessments
            ],
        }


class GoalAssessor(Protocol):
    """Assessment-only seam; implementations must not launch evaluation."""

    def assess(
        self,
        goal: Goal,
        *,
        goal_version: str,
        observed_metrics: Mapping[str, object],
        artifact_identity: str,
        evaluation_protocol_digest: str | None,
        benchmark_digest: str | None,
        constitution_digest: str | None = None,
        evidence: Sequence[GoalEvidenceRef] = (),
        measured_at: str | None = None,
    ) -> GoalAssessment:
        ...


class MeasuredGoalAssessor:
    """Compare already-measured values with a ``Goal`` without doing I/O."""

    def assess(
        self,
        goal: Goal,
        *,
        goal_version: str,
        observed_metrics: Mapping[str, object],
        artifact_identity: str,
        evaluation_protocol_digest: str | None,
        benchmark_digest: str | None,
        constitution_digest: str | None = None,
        evidence: Sequence[GoalEvidenceRef] = (),
        measured_at: str | None = None,
    ) -> GoalAssessment:
        timestamp = measured_at or datetime.now(timezone.utc).isoformat()
        all_evidence = tuple(evidence)
        assessment_digest = goal_digest(goal)
        metric_assessments: list[GoalMetricAssessment] = []
        overall_reasons: list[str] = []
        refusal_codes: list[str] = []

        if evaluation_protocol_digest is None:
            overall_reasons.append("MISSING_EVALUATION_PROTOCOL")
            refusal_codes.append("INCOMPLETE_EVALUATION_IDENTITY")
        elif not _is_digest(evaluation_protocol_digest):
            overall_reasons.append("INVALID_EVALUATION_PROTOCOL")
            refusal_codes.append("INVALID_EVALUATION_IDENTITY")
        if benchmark_digest is None:
            overall_reasons.append("MISSING_BENCHMARK_DIGEST")
            refusal_codes.append("INCOMPLETE_EVALUATION_IDENTITY")
        elif not _is_digest(benchmark_digest):
            overall_reasons.append("INVALID_BENCHMARK_DIGEST")
            refusal_codes.append("INVALID_EVALUATION_IDENTITY")
        if not all_evidence:
            overall_reasons.append("MISSING_EVIDENCE")
            refusal_codes.append("INCOMPLETE_EVIDENCE")

        for target in goal.metrics:
            reasons: list[str] = []
            raw_value = observed_metrics.get(target.name)
            if target.name not in observed_metrics:
                status = GoalStatus.UNKNOWN
                reasons.append("MISSING_MEASUREMENT")
                observed_value = None
            else:
                try:
                    observed_value = float(raw_value)
                except (TypeError, ValueError):
                    observed_value = None
                    status = GoalStatus.INVALID
                    reasons.append("NON_NUMERIC_MEASUREMENT")
                else:
                    if not math.isfinite(observed_value):
                        status = GoalStatus.INVALID
                        reasons.append("NON_FINITE_MEASUREMENT")
                    elif not all_evidence:
                        status = GoalStatus.UNKNOWN
                        reasons.append("MISSING_EVIDENCE")
                    elif target.target_met(observed_value):
                        status = GoalStatus.MET
                    else:
                        status = GoalStatus.UNMET
                        reasons.append("THRESHOLD_NOT_MET")

            metric_assessments.append(
                GoalMetricAssessment(
                    metric_name=target.name,
                    direction=target.direction,
                    required_minimum=target.minimum,
                    required_maximum=target.maximum,
                    observed_value=observed_value,
                    regression_tolerance=target.regression_tolerance,
                    status=status,
                    evidence=all_evidence,
                    reason_codes=tuple(reasons),
                )
            )

        statuses = {metric.status for metric in metric_assessments}
        identity_invalid = any(
            reason in {"INVALID_EVALUATION_PROTOCOL", "INVALID_BENCHMARK_DIGEST"}
            for reason in overall_reasons
        )
        if GoalStatus.INVALID in statuses or identity_invalid:
            status = GoalStatus.INVALID
            if GoalStatus.INVALID in statuses:
                overall_reasons.append("INVALID_MEASUREMENT")
            if identity_invalid:
                overall_reasons.append("INVALID_EVALUATION_IDENTITY")
        elif GoalStatus.UNKNOWN in statuses or overall_reasons:
            status = GoalStatus.UNKNOWN
        elif GoalStatus.UNMET in statuses:
            status = GoalStatus.UNMET
        else:
            status = GoalStatus.MET

        return GoalAssessment(
            goal_version=goal_version,
            goal_digest=assessment_digest,
            evaluation_protocol_digest=evaluation_protocol_digest,
            benchmark_digest=benchmark_digest,
            constitution_digest=constitution_digest,
            artifact_identity=artifact_identity,
            metric_assessments=tuple(metric_assessments),
            status=status,
            evidence=all_evidence,
            measured_at=timestamp,
            reason_codes=tuple(dict.fromkeys(overall_reasons)),
            refusal_codes=tuple(dict.fromkeys(refusal_codes)),
        )


def _is_digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


def _evidence_dict(ref: GoalEvidenceRef) -> dict[str, str]:
    return {"path": ref.path, "digest": ref.digest, "kind": ref.kind}


__all__ = [
    "GoalAssessment",
    "GoalAssessor",
    "GoalEvidenceRef",
    "GoalMetricAssessment",
    "GoalStatus",
    "MeasuredGoalAssessor",
]
