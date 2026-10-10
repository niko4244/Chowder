from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from ..models import Goal


class ConstitutionViolation(ValueError):
    """Raised when an autonomous action crosses a protected boundary."""


class ProtectedSurface(str, Enum):
    GOAL_THRESHOLDS = "goal_thresholds"
    PROTECTED_BENCHMARK = "protected_benchmark"
    CONTAMINATION_RULES = "contamination_rules"
    EVALUATION_PROTOCOL = "evaluation_protocol"
    PROMOTION_RULES = "promotion_rules"
    INDEPENDENT_JUDGE = "independent_judge"
    BUDGET_CEILINGS = "budget_ceilings"
    PROVENANCE_HASHING = "provenance_hashing"
    SANDBOX_BOUNDARIES = "sandbox_boundaries"
    PERMISSION_POLICY = "permission_policy"
    CONSTITUTION = "constitution"
    RECORDED_EVIDENCE = "recorded_evidence"


@dataclass(frozen=True)
class ObjectiveIdentity:
    """Content identity that must remain stable for a resumable objective."""

    objective_version: str
    goal_digest: str
    benchmark_digest: str
    evaluation_protocol_digest: str
    constitution_digest: str

    def __post_init__(self) -> None:
        for name, value in (
            ("objective_version", self.objective_version),
            ("goal_digest", self.goal_digest),
            ("benchmark_digest", self.benchmark_digest),
            ("evaluation_protocol_digest", self.evaluation_protocol_digest),
            ("constitution_digest", self.constitution_digest),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        for name in (
            "goal_digest",
            "benchmark_digest",
            "evaluation_protocol_digest",
            "constitution_digest",
        ):
            value = getattr(self, name)
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")

    def matches(self, other: ObjectiveIdentity) -> bool:
        return self == other

    def require_match(self, other: ObjectiveIdentity) -> None:
        if not self.matches(other):
            raise ConstitutionViolation(
                "objective identity changed; start a new objective version instead of resuming"
            )


def goal_digest(goal: Goal) -> str:
    """Return a stable digest of the existing canonical ``Goal`` type."""
    payload = {
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
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Constitution:
    """The non-negotiable policy used by an improvement objective."""

    version: str = "1"
    protected_surfaces: frozenset[ProtectedSurface] = frozenset(ProtectedSurface)

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("constitution version must be non-empty")
        unknown = set(self.protected_surfaces) - set(ProtectedSurface)
        if unknown:
            raise ValueError(f"unknown protected surfaces: {sorted(unknown)}")
        missing = set(ProtectedSurface) - set(self.protected_surfaces)
        if missing:
            raise ValueError(f"constitution omitted protected surfaces: {sorted(missing)}")

    def digest(self) -> str:
        payload = {
            "version": self.version,
            "protected_surfaces": sorted(surface.value for surface in self.protected_surfaces),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def assert_changes_allowed(
        self,
        changed_surfaces: Iterable[ProtectedSurface],
        *,
        current: ObjectiveIdentity,
        proposed: ObjectiveIdentity,
        human_approved: bool = False,
    ) -> None:
        """Reject protected changes unless they create an approved new objective."""
        surfaces = frozenset(changed_surfaces)
        unknown = surfaces - set(ProtectedSurface)
        if unknown:
            raise ConstitutionViolation(f"unknown protected surface(s): {sorted(unknown)}")
        if not surfaces:
            current.require_match(proposed)
            return
        if not human_approved:
            raise ConstitutionViolation(
                "protected changes require explicit human approval and a new objective version"
            )
        if proposed.objective_version == current.objective_version:
            raise ConstitutionViolation(
                "protected changes require a new objective version"
            )
        if proposed == current:
            raise ConstitutionViolation(
                "approved protected change must produce a distinct objective identity"
            )

    def assert_resume_allowed(
        self, expected: ObjectiveIdentity, actual: ObjectiveIdentity
    ) -> None:
        expected.require_match(actual)

    def new_objective_identity(
        self,
        *,
        objective_version: str,
        goal: Goal,
        benchmark_digest: str,
        evaluation_protocol_digest: str,
    ) -> ObjectiveIdentity:
        return ObjectiveIdentity(
            objective_version=objective_version,
            goal_digest=goal_digest(goal),
            benchmark_digest=benchmark_digest,
            evaluation_protocol_digest=evaluation_protocol_digest,
            constitution_digest=self.digest(),
        )


__all__ = [
    "Constitution",
    "ConstitutionViolation",
    "ObjectiveIdentity",
    "ProtectedSurface",
    "goal_digest",
]
