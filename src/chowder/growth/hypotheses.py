"""Hypothesis generation: candidates are experiments, not hyperparameter
combinations.

Before any candidate exists, the loop inspects the model's measured capability
profile and writes an explicit hypothesis per weakness it intends to address:
the observation that motivates it, the suspected cause, the intervention
family and parameters it proposes, what it predicts, what it risks, what must
be measured, and -- preregistered, before the run -- what result would
FALSIFY it. A candidate that cannot name its hypothesis is a recipe with
extra steps; the generator refuses to produce one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .evidence import EvidenceStore, PriorAdjustment, prior_for_family
from .interventions import (
    InterventionFamily,
    InterventionFamilyRefusal,
    assert_maturity_permits,
    family_from_id,
)

__all__ = [
    "Observation",
    "Hypothesis",
    "HypothesisRefusal",
    "generate_hypotheses",
    "hypothesis_candidate_brief",
]


class HypothesisRefusal(RuntimeError):
    """A hypothesis cannot be formed or used the way the caller asked."""


@dataclass(frozen=True)
class Observation:
    """One measured model weakness, straight from the capability profile.

    ``metric`` names the measured dimension, ``value`` its current value,
    ``threshold`` the preregistered line below which it counts as a weakness,
    and ``evidence_ref`` names the artifact that measured it -- an observation
    nobody can trace to a measurement is a hunch, and hunches do not spend
    GPU-hours.
    """

    metric: str
    value: float
    threshold: float
    evidence_ref: str
    direction: str = "min"  # "min": weakness means value > threshold

    def is_weakness(self) -> bool:
        return self.value > self.threshold if self.direction == "min" else self.value < self.threshold


@dataclass(frozen=True)
class Hypothesis:
    """A falsifiable claim about why the model is weak, and what to do."""

    hypothesis_id: str
    observation: Observation
    suspected_cause: str
    family_id: str
    intervention_parameters: Mapping[str, Any]
    predicted_improvement: str
    predicted_risks: tuple[str, ...]
    required_measurements: tuple[str, ...]
    #: Preregistered BEFORE the run: the result that would prove it wrong.
    falsification_criterion: str
    prior: PriorAdjustment | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "observation": {
                "metric": self.observation.metric,
                "value": self.observation.value,
                "threshold": self.observation.threshold,
                "evidence_ref": self.observation.evidence_ref,
            },
            "suspected_cause": self.suspected_cause,
            "family_id": self.family_id,
            "intervention_parameters": dict(self.intervention_parameters),
            "predicted_improvement": self.predicted_improvement,
            "predicted_risks": list(self.predicted_risks),
            "required_measurements": list(self.required_measurements),
            "falsification_criterion": self.falsification_criterion,
            "prior": (
                {
                    "multiplier": self.prior.multiplier,
                    "reasons": list(self.prior.reasons),
                }
                if self.prior is not None
                else None
            ),
        }


def generate_hypotheses(
    observations: Sequence[Observation],
    *,
    evidence_store: EvidenceStore,
    model_family: str,
    architecture: str,
    campaign_policy: Mapping[str, Any],
    family_ids: Sequence[str] | None = None,
    counter_start: int = 0,
) -> tuple[Hypothesis, ...]:
    """One hypothesis per (observation, family) pairing the store permits.

    Every proposal cites its observation; every observation must be a measured
    weakness (`is_weakness()`), not a vibe. Families the evidence store has
    excluded in this scope produce no hypothesis at all -- not a lowered
    prior, a refusal. The prior rides along on each hypothesis so the
    candidate planner can size budgets honestly.
    """
    if not observations:
        return ()
    weaknesses = [o for o in observations if o.is_weakness()]
    if not weaknesses:
        return ()
    families = (
        tuple(family_from_id(fid) for fid in family_ids) if family_ids else ()
    )
    hypotheses: list[Hypothesis] = []
    index = counter_start
    for observation in weaknesses:
        for family in families:
            try:
                assert_maturity_permits(
                    family.maturity,
                    campaign_policy=campaign_policy,
                    family_id=family.family_id,
                )
            except InterventionFamilyRefusal:
                continue
            # Family applicability to the observed failure class: a retention
            # observation does not justify an efficiency family's proposal.
            if not _addresses(family, observation):
                continue
            prior = prior_for_family(
                evidence_store,
                family_id=family.family_id,
                model_family=model_family,
                architecture=architecture,
            )
            if prior.excluded:
                continue
            index += 1
            hypotheses.append(
                Hypothesis(
                    hypothesis_id=f"hyp-{index:03d}-{observation.metric}-{family.family_id}",
                    observation=observation,
                    suspected_cause=_suspected_cause(family, observation),
                    family_id=family.family_id,
                    intervention_parameters={
                        name: spec["range"][0]
                        for name, spec in family.parameters.items()
                    },
                    predicted_improvement=(
                        f"{observation.metric} moves below {observation.threshold:g} "
                        "without regressing the protected profile"
                    ),
                    predicted_risks=family.risks,
                    required_measurements=tuple(
                        family.eval_dimensions
                        or ("target-capability", "retained-capabilities")
                    ),
                    falsification_criterion=(
                        f"{observation.metric} does not improve beyond measurement "
                        "noise at the declared sample size, OR any protected "
                        "retention dimension regresses -- either outcome closes "
                        f"this hypothesis and records it in the evidence store "
                        f"for family {family.family_id}"
                    ),
                    prior=prior,
                )
            )
    return tuple(hypotheses)


def _addresses(family: InterventionFamily, observation: Observation) -> bool:
    """Whether a family plausibly addresses the observed metric.

    Kept deliberately coarse and overridable: it matches the failure class the
    family declares against a small metric-class vocabulary, so a family
    cannot be pointed at a failure it was never declared for.
    """
    metric = observation.metric.lower()
    failure = family.target_failure_class
    if failure == "target-capability-weakness":
        return not metric.startswith(("retention", "termination", "latency", "vram"))
    if failure == "retention-loss":
        return metric.startswith("retention")
    if failure == "factual-weakness":
        return "factual" in metric or "fabrication" in metric
    if failure in ("vram-footprint", "latency", "inference-efficiency"):
        return metric.startswith(("vram", "latency", "throughput"))
    if failure == "long-context-efficiency":
        return "long" in metric or "latency" in metric
    if failure == "agent-runtime-failure":
        # Runtime metrics arrive under either separator convention
        # (runtime_reward from the benchmark, runtime-reward in a profile).
        normalized = metric.replace("-", "_")
        return normalized.startswith(
            ("runtime", "tool", "repair", "premature", "nonexistent")
        )
    if failure == "confidence-calibration":
        return "confidence" in metric or "margin" in metric or "calibration" in metric
    return True


def _suspected_cause(family: InterventionFamily, observation: Observation) -> str:
    return (
        f"the {observation.metric} weakness ({observation.value:g} vs threshold "
        f"{observation.threshold:g}) stems from a cause the {family.name} "
        "family addresses"
    )


def hypothesis_candidate_brief(
    hypothesis: Hypothesis, *, family: InterventionFamily
) -> dict[str, Any]:
    """The brief a candidate planner must satisfy for this hypothesis.

    The planner proposes concrete recipes; this brief carries the hypothesis
    identity, the family's declared parameter space, and the falsification
    rule that must travel with every candidate into the campaign declaration.
    """
    if family.family_id != hypothesis.family_id:
        raise HypothesisRefusal(
            f"brief requested for family {family.family_id!r} but the "
            f"hypothesis proposes {hypothesis.family_id!r}"
        )
    return {
        "hypothesis_id": hypothesis.hypothesis_id,
        "family_id": family.family_id,
        "parameters": {
            name: dict(spec) for name, spec in family.parameters.items()
        },
        "required_measurements": list(hypothesis.required_measurements),
        "falsification_criterion": hypothesis.falsification_criterion,
        "prior_multiplier": hypothesis.prior.multiplier if hypothesis.prior else 1.0,
    }
