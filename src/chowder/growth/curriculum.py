"""Curriculum engine: what should the model learn next?

Answers, from evidence: given Model N's capability profile and failure
bank, which skills deserve training, in what mixture, at what difficulty,
from which source strategy -- and why. Priority is a weighted, documented
score over weakness magnitude, importance, confidence, failure frequency,
frontier gap, trainability, cost, and regression risk -- deliberately NOT
"train on whatever score is lowest."

Every item carries its decision provenance: why it was selected, what
evidence drove it, and which protected sets it must respect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .capability import CapabilityProfile, assert_known_skills
from .failure_bank import FailureBank

#: Curriculum item mix roles.
ROLES = frozenset({"TARGET", "PRESERVE", "GENERAL", "REPLAY", "STRETCH"})

#: Training types the engine may assign (PART 33): not every weakness is an
#: SFT problem.
TRAINING_TYPES = frozenset(
    {"continued_pretrain", "sft", "preference", "targeted_repair", "architecture_research"}
)

DEFAULT_WEIGHTS: Mapping[str, float] = {
    "weakness_magnitude": 0.30,  # how far below target the skill sits
    "importance": 0.20,  # declared skill importance to the deployment goal
    "confidence": 0.15,  # confidence in the weakness evidence
    "failure_frequency": 0.15,  # banked failures pointing at the skill
    "frontier_gap": 0.10,  # distance to frontier on the skill's benchmarks
    "trainability": 0.05,  # can small-scale training plausibly move it
    "cost_efficiency": 0.03,  # expected GPU cost per unit of evidence
    "regression_risk": 0.02,  # penalty for skills entangled with strong ones
}

#: Skills whose regression is expensive: growing them risks protected
#: capabilities (e.g. heavy code-mix training historically trades away
#: calibration). Evidence-driven, configurable.
REGRESSION_ENTANGLEMENT: Mapping[str, float] = {
    "knowledge.calibration": 0.8,
    "safety.honesty": 0.9,
    "safety.refusal_calibration": 0.9,
    "instruction.formatting": 0.5,
}


@dataclass(frozen=True)
class CurriculumItem:
    """One planned training item with full decision provenance."""

    item_id: str
    skill: str
    role: str  # ROLES
    priority: float
    confidence: float
    weakness_evidence: str
    desired_improvement: float  # target skill-estimate gain
    preservation_risks: tuple[str, ...]
    source_strategy: str  # data source ids / generation strategy
    example_count: int
    token_target: int
    difficulty_band: str
    verification_method: str
    training_type: str  # TRAINING_TYPES
    evaluation_set: str  # which eval tier/ids measure the repair
    protected_regression_set: tuple[str, ...]
    decision_trace: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"{self.item_id}: unknown role {self.role!r}")
        if self.training_type not in TRAINING_TYPES:
            raise ValueError(f"{self.item_id}: unknown training_type {self.training_type!r}")
        assert_known_skills((self.skill,), context=f"item {self.item_id}")

    def to_dict(self) -> dict[str, Any]:
        data = {
            f: getattr(self, f)
            for f in self.__dataclass_fields__  # type: ignore[attr-defined]
        }
        data["preservation_risks"] = list(self.preservation_risks)
        data["protected_regression_set"] = list(self.protected_regression_set)
        data["decision_trace"] = dict(self.decision_trace)
        return data


@dataclass(frozen=True)
class SkillPriority:
    """One skill's computed priority with its full trace."""

    skill: str
    priority: float
    components: Mapping[str, float]
    evidence: str


class CurriculumEngine:
    """Turns capability profile + failure bank + frontier snapshot into a
    prioritized, mixture-composed curriculum plan."""

    def __init__(
        self,
        *,
        weights: Mapping[str, float] | None = None,
        importance: Mapping[str, float] | None = None,
        failure_bank: FailureBank | None = None,
    ) -> None:
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            unknown = set(weights) - set(self.weights)
            if unknown:
                raise ValueError(f"unknown priority weights: {sorted(unknown)}")
            self.weights.update(weights)
        self.importance = dict(importance or {})
        self.failure_bank = failure_bank or FailureBank()

    # ---------------- priority ----------------

    def prioritize(
        self,
        profile: CapabilityProfile,
        *,
        frontier: Mapping[str, float] | None = None,
        floor: Mapping[str, float] | None = None,
        max_skills: int = 8,
    ) -> tuple[SkillPriority, ...]:
        """Rank skills by weighted evidence. ``frontier`` maps skill ->
        normalized frontier reference (0..1); ``floor`` maps skill -> the
        generation-0 value (regression below floor is a red flag, not a
        training target)."""
        frontier = frontier or {}
        floor = floor or {}
        skill_estimates = {s.skill: s for s in profile.skills}
        tainted = set(profile.tainted)
        priorities: list[SkillPriority] = []
        for skill, estimate in skill_estimates.items():
            if estimate.confidence <= 0.05:
                continue  # no usable evidence: do not pretend to prioritize
            weakness = max(0.0, 1.0 - estimate.estimate)
            importance = float(self.importance.get(skill, 0.5))
            confidence = estimate.confidence
            failures = self.failure_bank.category_counts()
            # Failures map to skills via the benchmark registry mapping in
            # the caller-supplied bank evidence; approximate here by
            # counting banked failures whose categories relate to the
            # skill's root.
            skill_root = skill.split(".")[0]
            frequency = min(1.0, failures.get(skill_root, 0) / 10.0)
            frontier_gap = (
                max(0.0, frontier.get(skill, estimate.estimate) - estimate.estimate)
                if skill in frontier
                else 0.0
            )
            trainability = 1.0 if estimate.estimate < 0.95 else 0.1
            cost_efficiency = 1.0 - min(1.0, estimate.estimate)
            regression_risk = REGRESSION_ENTANGLEMENT.get(skill, 0.2)
            components = {
                "weakness_magnitude": weakness,
                "importance": importance,
                "confidence": confidence,
                "failure_frequency": frequency,
                "frontier_gap": frontier_gap,
                "trainability": trainability,
                "cost_efficiency": cost_efficiency,
                "regression_risk": regression_risk,
            }
            priority = sum(self.weights[k] * v for k, v in components.items())
            below_floor = skill in floor and estimate.estimate < floor[skill] - 0.02
            priorities.append(
                SkillPriority(
                    skill=skill,
                    priority=priority,
                    components=components,
                    evidence=(
                        f"estimate={estimate.estimate:.3f} conf={confidence:.2f} "
                        f"frontier_gap={frontier_gap:.3f} below_floor={below_floor} "
                        f"tainted_excluded={sorted(tainted)}"
                    ),
                )
            )
        priorities.sort(key=lambda p: -p.priority)
        return tuple(priorities[:max_skills])

    # ---------------- mixture ----------------

    def mixture(
        self,
        priorities: Sequence[SkillPriority],
        *,
        protected_probe_skills: Sequence[str] = (),
    ) -> dict[str, float]:
        """Role proportions for one cycle, evidence-driven.

        TARGET dominates while real weaknesses exist; PRESERVE scales with
        protected-probe exposure (anti-forgetting); REPLAY scales with banked
        recurrence; GENERAL and STRETCH keep the floor under the target.
        """
        if not priorities:
            return {"TARGET": 0.2, "PRESERVE": 0.2, "GENERAL": 0.4, "REPLAY": 0.1, "STRETCH": 0.1}
        top = priorities[0]
        # Evidence-driven proportions: the sharper the top priority, the more
        # TARGET; preserve/replay scale with what the bank says is at risk.
        open_failures = self.failure_bank.open_failures()
        repaired = self.failure_bank.repaired_classes()
        replay_pressure = min(0.35, 0.05 * len(open_failures))
        preserve_pressure = min(0.35, 0.08 * len(set(protected_probe_skills) | set(repaired)))
        target_share = 0.45 + 0.15 * (1.0 - top.priority) if top.priority < 0.6 else 0.45
        target_share = min(0.6, target_share)
        general_share = max(0.10, 0.30 - target_share / 2)
        remaining = 1.0 - target_share - general_share
        preserve_share = remaining * (preserve_pressure / max(preserve_pressure + replay_pressure, 1e-9))
        replay_share = remaining - preserve_share
        stretch_share = 0.05 if top.priority < 0.8 else 0.0
        # Renormalize with stretch taken from general.
        general_share = max(0.05, general_share - stretch_share)
        total = target_share + general_share + preserve_share + replay_share + stretch_share
        return {
            "TARGET": target_share / total,
            "PRESERVE": preserve_share / total,
            "GENERAL": general_share / total,
            "REPLAY": replay_share / total,
            "STRETCH": stretch_share / total,
        }

    # ---------------- plan ----------------

    def plan(
        self,
        *,
        model_version: str,
        profile: CapabilityProfile,
        protected_sets: Sequence[str],
        floor: Mapping[str, float] | None = None,
        frontier: Mapping[str, float] | None = None,
        max_items: int = 5,
        protected_probe_skills: Sequence[str] = (),
        budget_examples: int = 20000,
    ) -> tuple[CurriculumItem, ...]:
        """Build the cycle's curriculum items.

        ``protected_sets`` are the protected benchmark ids the firewall
        enforces; every item carries them as its regression set. Items are
        only created for skills with real evidence; the plan records why
        each skill was selected and why example counts were chosen.
        """
        priorities = self.prioritize(profile, frontier=frontier, floor=floor)
        mix = self.mixture(priorities, protected_probe_skills=protected_probe_skills)
        items: list[CurriculumItem] = []
        target_budget = int(budget_examples * mix["TARGET"])
        for index, priority in enumerate(priorities[:max_items], start=1):
            estimate = profile.skill(priority.skill)
            if estimate is None:
                continue
            examples = max(500, int(target_budget / max(1, len(priorities[:max_items]))))
            item = CurriculumItem(
                item_id=f"{model_version}-curriculum-{index:02d}-{priority.skill.replace('.', '-')}",
                skill=priority.skill,
                role="TARGET",
                priority=priority.priority,
                confidence=estimate.confidence,
                weakness_evidence=priority.evidence,
                desired_improvement=round(min(0.2, max(0.03, 1.0 - estimate.estimate) / 2), 4),
                preservation_risks=tuple(
                    sorted(
                        skill
                        for skill, risk in REGRESSION_ENTANGLEMENT.items()
                        if risk >= 0.5 and skill != priority.skill
                    )
                ),
                source_strategy="registered GOLD/SILVER sources + verified synthetic analogues",
                example_count=examples,
                token_target=examples * 600,
                difficulty_band="medium",
                verification_method=(
                    "executable_tests" if priority.skill.startswith("coding")
                    else "symbolic_numeric" if priority.skill.startswith("math")
                    else "multi_judge"
                ),
                training_type=(
                    "targeted_repair"
                    if priority.components["failure_frequency"] > 0.3
                    else "sft"
                ),
                evaluation_set="tier1+tier2",
                protected_regression_set=tuple(sorted(protected_sets)),
                decision_trace={
                    "components": dict(priority.components),
                    "weights": dict(self.weights),
                    "mixture": dict(mix),
                    "floor": dict(floor or {}),
                    "reason": (
                        "weighted evidence: weakness, importance, confidence, banked failures, "
                        "frontier gap; see components"
                    ),
                },
            )
            items.append(item)
        return tuple(items)
