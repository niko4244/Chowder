"""Retention as a first-class constrained objective.

The objective 0.5 optimizes is approximately:

    maximize target_capability_gain
    subject to: no protected-capability regression, no general-reasoning
    regression, no termination/degeneration, tool and repair correctness,
    fabrication ceilings, runtime/memory ceilings, contamination and evidence
    integrity.

A model does not promote because one score went up while the model got worse
at everything else -- the arithmetic 0.60->0.78 with reasoning 0.71->0.51
outcome is a refusal, not a promotion with a footnote. Constraints live in
per-campaign, per-model *retention profiles* (named, preregistered, versioned),
not in one global benchmark list: a campaign on a model with no tool surface
should not inherit tool ceilings it cannot measure, and a campaign on a code
model must inherit the repair ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

__all__ = [
    "RetentionConstraint",
    "RetentionProfile",
    "RetentionViolation",
    "evaluate_retention",
]


@dataclass(frozen=True)
class RetentionConstraint:
    """One named constraint a candidate must satisfy to promote.

    ``kind`` selects the comparison:

    * ``max-regression``: dimension's candidate-vs-parent delta may not be
      worse than ``value`` (a negative value permits a small, declared dip);
    * ``absolute-floor``: the candidate's measured value may not fall below
      ``value`` regardless of the parent's.
    """

    dimension: str
    kind: str  # max-regression | absolute-floor
    value: float
    #: Where the measurement must come from: a Tier-3 benchmark. A retention
    #: constraint measured on Tier-1/Tier-2 evidence would let the search
    #: shape its own gate -- so the constraint names a promotion-evidence
    #: benchmark, and enforcing that is the tier wall's job.
    benchmark: str

    def __post_init__(self) -> None:
        # The single kind validation. The manifest loader wraps this error
        # with its own source context rather than re-implementing the rule.
        if self.kind not in ("max-regression", "absolute-floor"):
            raise ValueError(
                f"retention constraint {self.dimension!r} has unknown kind "
                f"{self.kind!r}; a constraint is max-regression or "
                "absolute-floor"
            )


@dataclass(frozen=True)
class RetentionProfile:
    """The preregistered retention constraints of one campaign."""

    profile_id: str
    constraints: tuple[RetentionConstraint, ...]

    def __post_init__(self) -> None:
        if not self.constraints:
            raise ValueError(
                f"retention profile {self.profile_id!r} declares no "
                "constraints: an unconstrained campaign is a capability "
                "trader, and 0.5 is not one"
            )
        dimensions = [c.dimension for c in self.constraints]
        if len(dimensions) != len(set(dimensions)):
            raise ValueError(
                f"retention profile {self.profile_id!r} declares a dimension "
                "twice; each constraint is one decision"
            )


@dataclass(frozen=True)
class RetentionViolation:
    dimension: str
    constraint: RetentionConstraint
    measured: float
    detail: str

    @property
    def code(self) -> str:
        """The machine-readable identifier for this failure shape.

        One owner of the vocabulary a promotion rejection records:
        ``RETENTION_UNMEASURED`` (NaN measured: the constraint could not be
        evaluated at all), ``RETENTION_FLOOR`` (an absolute-floor breach),
        ``RETENTION_REGRESSION`` (a max-regression breach). Renaming or adding
        a shape changes here and nowhere else.
        """
        if self.measured != self.measured:  # NaN: the gate was never measured
            return "RETENTION_UNMEASURED"
        if self.constraint.kind == "absolute-floor":
            return "RETENTION_FLOOR"
        return "RETENTION_REGRESSION"

    @property
    def reason(self) -> str:
        """The rejection reason a promotion records: identifier plus detail."""
        return f"{self.code}: {self.detail}"


def evaluate_retention(
    profile: RetentionProfile,
    *,
    parent_values: Mapping[str, float],
    candidate_values: Mapping[str, float],
) -> tuple[RetentionViolation, ...]:
    """Every constraint the candidate violates; empty means it may promote.

    A dimension with no candidate measurement is a violation by default
    (fail-closed): a gate that cannot be measured is not a gate that was
    passed. The parent's own missing value only fails ``absolute-floor``
    against the candidate value; ``max-regression`` needs both sides and
    refuses on either missing.
    """
    violations: list[RetentionViolation] = []
    for constraint in profile.constraints:
        candidate_value = candidate_values.get(constraint.dimension)
        if candidate_value is None:
            violations.append(
                RetentionViolation(
                    dimension=constraint.dimension,
                    constraint=constraint,
                    measured=float("nan"),
                    detail=(
                        "no candidate measurement for a declared retention "
                        "constraint -- unmeasured is not compliance"
                    ),
                )
            )
            continue
        if constraint.kind == "absolute-floor":
            if candidate_value < constraint.value:
                violations.append(
                    RetentionViolation(
                        dimension=constraint.dimension,
                        constraint=constraint,
                        measured=candidate_value,
                        detail=(
                            f"candidate {candidate_value:g} is below the "
                            f"absolute floor {constraint.value:g}"
                        ),
                    )
                )
            continue
        parent_value = parent_values.get(constraint.dimension)
        if parent_value is None:
            violations.append(
                RetentionViolation(
                    dimension=constraint.dimension,
                    constraint=constraint,
                    measured=candidate_value,
                    detail=(
                        "no parent measurement for a max-regression "
                        "constraint -- the delta cannot be certified"
                    ),
                )
            )
            continue
        delta = candidate_value - parent_value
        if delta < constraint.value - 1e-12:
            violations.append(
                RetentionViolation(
                    dimension=constraint.dimension,
                    constraint=constraint,
                    measured=delta,
                    detail=(
                        f"regression {delta:+g} on {constraint.dimension!r} "
                        f"breaches the declared max-regression {constraint.value:+g}"
                    ),
                )
            )
    return tuple(violations)
