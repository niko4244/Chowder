"""Fake-compute multi-generation simulation: prove the loop before it spends.

The loop's control logic -- what it does with a promotion, a rejection, a
plateau, an exhausted envelope, a target it must not train -- is the part that
decides whether unattended training is safe to start. Exercising it against real
GPUs costs hours per evidence point and cannot produce a rejection, a regression
or a budget exhaustion on demand.

This module runs the *real* ``GrowthLoop`` against a deterministic stand-in for
compute. Everything above the executor is production code: the durable
``GrowthState``, the selector, the campaign builder, the preregistration freeze,
the budget arithmetic, the plateau rule and the stop/review decisions are the
same objects the real loop uses. Only two things are replaced:

* the executor, which would run a campaign, and
* preparation/readiness, which would validate inputs against a real tree.

The scenarios therefore pin *terminal states* rather than hopes:

| scenario | what it shows |
|---|---|
| A | two promotions, then a plateau at the declared generation limit |
| B | a rejected generation keeps the parent and the next attempt promotes |
| C | a protected regression is rejected and the parent pointer does not move |
| D | an envelope that cannot hold the next campaign launches no training |
| E | a declared structural target goes to human review, unspent |
| F | a repeatedly unsuccessful intervention stops being repeated |

The stand-in is not a model: it is a table of measured outcomes. A scenario says
what compute cost and what the target moved by, and the simulator asserts only
what the loop *decides* from that.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .campaign import CampaignManifest
from .growth_loop import (
    CONTINUE,
    REQUIRES_HUMAN_REVIEW,
    STOP_BUDGET,
    STOP_PLATEAU,
    STOP_SUCCESS,
    STOP_UNCERTAIN,
    CampaignOutcome,
    GrowthLoop,
    LoopDecision,
    LoopRunReport,
)
from .next_campaign import LoopPolicy
from .target_selection import GrowthState, SkillProfile

#: Reused verbatim rather than spelled as literals: the provenance vocabulary is
#: owned by the evaluation result schema, and a simulator that invented its own
#: spelling could disagree with the profile builder without failing.
from chowder.evals.result import MEASURED_PARENT, UNMEASURED

#: Measured outcomes one simulated campaign can end in.
PROMOTED = "PROMOTED"
REJECTED = "REJECTED"
INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True)
class SimulatedCampaign:
    """One planned campaign outcome: what it cost and what it moved."""

    verdict: str
    wall_gpu_hours: float
    target_effect: float | None
    #: Skill -> new measured estimate for the resulting generation. Applied on
    #: promotion only, because a rejected campaign leaves the parent where it is.
    improves: Mapping[str, float] = field(default_factory=dict)
    regressions: tuple[str, ...] = ()
    reason: str = ""


@dataclass(frozen=True)
class Scenario:
    """A named sequence of simulated campaigns and the state it must end in."""

    name: str
    description: str
    campaigns: tuple[SimulatedCampaign, ...]
    expected_action: str
    expected_reason_codes: tuple[str, ...] = ()
    expected_campaigns_run: int = 0
    expected_promotions: int = 0
    expected_parent_version: str = ""
    initial_scores: Mapping[str, float] = field(
        default_factory=lambda: DEFAULT_SKILL_SCORES
    )
    maximum_generations: int = 3
    maximum_total_wall_gpu_hours: float = 6.0
    maximum_consecutive_non_promotions: int = 2
    maximum_same_target_attempts: int = 2
    structural_skills: tuple[str, ...] = ()
    allowed_training_types: tuple[str, ...] = ("targeted_repair", "sft")
    #: How many *different* targets the run must have chosen. A loop that tries
    #: the same unsuccessful intervention again reports fewer.
    expected_distinct_targets: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "campaigns": [
                {
                    "verdict": campaign.verdict,
                    "wall_gpu_hours": campaign.wall_gpu_hours,
                    "target_effect": campaign.target_effect,
                    "improves": dict(campaign.improves),
                    "regressions": list(campaign.regressions),
                }
                for campaign in self.campaigns
            ],
            "expected_action": self.expected_action,
            "expected_reason_codes": list(self.expected_reason_codes),
            "expected_campaigns_run": self.expected_campaigns_run,
            "expected_promotions": self.expected_promotions,
            "expected_parent_version": self.expected_parent_version,
            "initial_scores": dict(self.initial_scores),
        }


class SimulationError(AssertionError):
    """A scenario whose outcome is not the one it declared."""


def skill_profile(scores: Mapping[str, float]) -> SkillProfile:
    """A measured profile with one benchmark per skill (evidence, not a mean).

    Public because it is the one place that says what "a measured capability"
    looks like to the control plane: the loop's tests and the simulation must
    agree on that shape, or they would be proving different things.
    """

    from .target_selection import SkillEstimateEvidence

    def estimate(skill: str, score: float, measured: bool = True) -> Any:
        return SkillEstimateEvidence(
            skill=skill,
            estimate=score if measured else None,
            confidence=0.9 if measured else 0.0,
            uncertainty=0.1 if measured else 1.0,
            supporting_benchmarks=(f"{skill}-bench@2026-01",),
            benchmark_measurements={f"{skill}-bench@2026-01": score} if measured else {},
            n_measurements=16 if measured else 0,
            generation="gen2",
            provenance=MEASURED_PARENT if measured else UNMEASURED,
            aggregation="single benchmark",
        )

    return SkillProfile(
        generation="gen2",
        estimates=tuple(estimate(skill, score) for skill, score in scores.items()),
    )


def _policy(document: Mapping[str, Any]) -> LoopPolicy:
    payload = dict(document)
    payload.setdefault("calibration_benchmarks", [])
    payload.setdefault("reliability_benchmarks", [])
    return LoopPolicy.from_mapping(payload, source="simulator")


class _SimulatedExecutor:
    """Serves the scenario's table, one campaign at a time. No compute."""

    def __init__(self, scenario: Scenario, profile: SkillProfile) -> None:
        self.scenario = scenario
        self.calls: list[str] = []
        self._scores = {
            estimate.skill: float(estimate.estimate or 0.0)
            for estimate in profile.estimates
        }

    def __call__(self, frozen) -> CampaignOutcome:  # noqa: ANN001
        if len(self.calls) >= len(self.scenario.campaigns):
            raise SimulationError(
                f"{self.scenario.name}: the loop asked for a "
                f"{len(self.calls) + 1}th campaign and the scenario declares only "
                f"{len(self.scenario.campaigns)}"
            )
        step = self.scenario.campaigns[len(self.calls)]
        self.calls.append(frozen.cycle_id)
        promoted = step.verdict.upper() == PROMOTED
        if promoted:
            self._scores.update({skill: float(score) for skill, score in step.improves.items()})
        return CampaignOutcome(
            verdict=step.verdict,
            wall_gpu_hours=step.wall_gpu_hours,
            device_gpu_hours=None,
            measured_target_effect=step.target_effect,
            parent_identity=(
                (f"simulated/{frozen.candidate_version}/adapter", "b" * 64)
                if promoted
                else None
            ),
            profile=skill_profile(self._scores).to_dict(),
            failures=(),
            regressions=step.regressions,
            run_root=str(frozen.directory),
            reason=step.reason or step.verdict,
        )


#: The starting measured profile every scenario begins from: a strong skill, a
#: weak one, and a third so a scenario can observe a *different* target being
#: chosen rather than assuming the weakest one always wins.
DEFAULT_SKILL_SCORES: Mapping[str, float] = {
    "math.reasoning": 0.62,
    "instruction.formatting": 0.31,
    "termination.control": 0.44,
}

#: The frozen declaration the simulation starts from, resolved from this file
#: (``<repo>/src/chowder/growth/simulator.py`` -> ``<repo>/docs/gen2/...``).
#: A default, not an assumption: an explicit ``manifest_path`` always wins, and
#: a missing default refuses by name rather than simulating against invented
#: limits.
DEFAULT_PARENT_DECLARATION = (
    Path(__file__).resolve().parents[3] / "docs" / "gen2" / "gen2_campaign.json"
)


def parent_declaration(
    scenario: Scenario, *, root: Path, manifest_path: str | Path | None = None
) -> CampaignManifest:
    """The real frozen declaration, with its state root redirected into ``root``.

    The protection policy, protocol, execution configuration, budget shape and
    promotion policy therefore come from the checked-in declaration rather than
    from a simulator's idea of them; only the location is temporary.
    """
    path = Path(manifest_path or DEFAULT_PARENT_DECLARATION)
    if not path.is_file():
        raise SimulationError(
            f"no frozen parent declaration at {path}; the simulation takes its "
            "policy, protection protocol and budget shape from a real one rather "
            "than from invented limits, so it refuses instead of guessing"
        )
    document = json.loads(path.read_text(encoding="utf-8"))
    document["state_root"] = str(root / "parent-run")
    return CampaignManifest.from_mapping(document, source=f"simulator:{path.name}")


def policy_document(scenario: Scenario, parent: CampaignManifest) -> dict[str, Any]:
    """A loop policy built from the parent declaration plus the scenario's limits."""
    return {
        "maximum_generations": scenario.maximum_generations,
        "maximum_total_wall_gpu_hours": scenario.maximum_total_wall_gpu_hours,
        "maximum_consecutive_non_promotions": scenario.maximum_consecutive_non_promotions,
        "maximum_same_target_attempts": scenario.maximum_same_target_attempts,
        "maximum_candidates": len(parent.recipe_ids),
        "plateau_epsilon": 0.01,
        "allowed_training_types": list(scenario.allowed_training_types),
        "protected_benchmarks": list(parent.protected_benchmarks),
        "broad_benchmarks": list(parent.broad_benchmarks),
        "campaign_budget": {
            "device_gpu_hours_ceiling_per_recipe": parent.budget.device_gpu_hours_ceiling_per_recipe,
            "wall_gpu_hours_ceiling_per_recipe": parent.budget.wall_gpu_hours_ceiling_per_recipe,
            "device_gpu_hours_ceiling_campaign": parent.budget.device_gpu_hours_ceiling_campaign,
            "wall_gpu_hours_ceiling_campaign": parent.budget.wall_gpu_hours_ceiling_campaign,
            "device_time_measured": parent.budget.device_time_measured,
        },
        "protection": parent.protection.to_dict(),
        "evaluation_execution": parent.evaluation_execution.to_dict(),
        "candidate_selection_policy": parent.candidate_selection_policy,
        "stopping_rules": list(parent.stopping_rules),
        "promotion_policy_version": parent.promotion_policy_version,
        "human_review_triggers": ["architecture change", "new external dataset"],
        "structural_skills": list(scenario.structural_skills),
    }


def run_scenario(
    scenario: Scenario,
    *,
    root: str | Path,
    manifest_path: str | Path | None = None,
) -> LoopRunReport:
    """Run one scenario through the real loop and check its terminal state."""
    root = Path(root)
    state = GrowthState(root=root / "growth-state")
    profile = skill_profile(scenario.initial_scores)
    parent = parent_declaration(scenario, root=root, manifest_path=manifest_path)
    policy = _policy(policy_document(scenario, parent))
    executor = _SimulatedExecutor(scenario, profile)
    loop = GrowthLoop(
        policy=policy,
        state=state,
        executor=executor,
        parent_declaration=parent,
        parent_profile=profile,
        prepare=lambda frozen: None,
        readiness=lambda frozen: True,
    )
    report = loop.run()

    _require(
        scenario.name,
        "terminal action",
        report.decision.action,
        scenario.expected_action,
    )
    for code in scenario.expected_reason_codes:
        _require(
            scenario.name,
            "reason code present",
            code in report.decision.reason_codes,
            True,
        )
    _require(
        scenario.name,
        "campaigns run through the executor",
        len(executor.calls),
        scenario.expected_campaigns_run,
    )
    _require(
        scenario.name,
        "promotions",
        sum(1 for record in report.generations if record.verdict.upper() == PROMOTED),
        scenario.expected_promotions,
    )
    if scenario.expected_parent_version:
        _require(
            scenario.name,
            "parent version after the run",
            report.parent_version,
            scenario.expected_parent_version,
        )
    if scenario.expected_distinct_targets:
        _require(
            scenario.name,
            "distinct targets chosen",
            len({record.target_skill for record in report.generations}),
            scenario.expected_distinct_targets,
        )
    # The loop must never leave the session's spend unaccounted.
    declared = sum(
        float(campaign.wall_gpu_hours) for campaign in scenario.campaigns[: len(executor.calls)]
    )
    _require(
        scenario.name,
        "accounted wall GPU-hours equal the measured campaigns",
        round(float(report.budget["spent_wall_gpu_hours"]), 9),
        round(declared, 9),
    )
    return report


def _require(scenario: str, what: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise SimulationError(
            f"scenario {scenario!r}: {what} was {actual!r}, expected {expected!r}"
        )


# --------------------------------------------------------------------------
# the six scenarios
# --------------------------------------------------------------------------
#
# Each table is an *input*, not a claim about a model: it says what one
# simulated campaign cost and how far it moved the target. The assertions are
# about what the loop decides from that, which is the part that must be right
# before unattended training is allowed to start.

SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        name="A-two-promotions-then-a-plateau",
        description=(
            "Two generations improve and promote; the third does not, and the "
            "declared generation limit ends the session."
        ),
        campaigns=(
            SimulatedCampaign(
                verdict=PROMOTED,
                wall_gpu_hours=0.42,
                target_effect=0.18,
                improves={"instruction.formatting": 0.88},
            ),
            SimulatedCampaign(
                verdict=PROMOTED,
                wall_gpu_hours=0.40,
                target_effect=0.15,
                improves={"termination.control": 0.86},
            ),
            SimulatedCampaign(
                verdict=REJECTED,
                wall_gpu_hours=0.38,
                target_effect=0.0,
                reason="target gate not met",
            ),
        ),
        expected_action=STOP_PLATEAU,
        expected_reason_codes=("GENERATION_LIMIT_REACHED",),
        expected_campaigns_run=3,
        expected_promotions=2,
        expected_parent_version="gen4",
    ),
    Scenario(
        name="B-rejected-then-the-next-attempt-promotes",
        description=(
            "A rejected generation keeps the parent pointer, the failure is "
            "remembered, and the next attempt on the same generation promotes."
        ),
        campaigns=(
            SimulatedCampaign(
                verdict=REJECTED,
                wall_gpu_hours=0.39,
                target_effect=-0.02,
                reason="target gate not met",
            ),
            SimulatedCampaign(
                verdict=PROMOTED,
                wall_gpu_hours=0.41,
                target_effect=0.17,
                improves={"instruction.formatting": 0.9, "termination.control": 0.6},
            ),
        ),
        expected_action=STOP_SUCCESS,
        expected_reason_codes=("GENERATION_LIMIT_REACHED",),
        expected_campaigns_run=2,
        expected_promotions=1,
        expected_parent_version="gen3",
        maximum_generations=2,
    ),
    Scenario(
        name="C-protected-regression-does-not-advance-the-parent",
        description=(
            "A promotion is followed by a generation that regresses a protected "
            "capability; it is rejected and the parent pointer stays put."
        ),
        campaigns=(
            SimulatedCampaign(
                verdict=PROMOTED,
                wall_gpu_hours=0.44,
                target_effect=0.2,
                improves={"termination.control": 0.72},
            ),
            SimulatedCampaign(
                verdict=REJECTED,
                wall_gpu_hours=0.43,
                target_effect=0.05,
                regressions=("math.reasoning",),
                reason="protected slice regressed beyond tolerance",
            ),
        ),
        expected_action=STOP_PLATEAU,
        expected_reason_codes=("GENERATION_LIMIT_REACHED",),
        expected_campaigns_run=2,
        expected_promotions=1,
        expected_parent_version="gen3",
        maximum_generations=2,
    ),
    Scenario(
        name="D-envelope-too-small-launches-nothing",
        description=(
            "The remaining session envelope cannot hold the declared campaign "
            "envelope, so no training is launched at all."
        ),
        campaigns=(),
        expected_action=STOP_BUDGET,
        expected_reason_codes=("REMAINING_ENVELOPE_TOO_SMALL",),
        expected_campaigns_run=0,
        expected_promotions=0,
        expected_parent_version="gen2",
        maximum_total_wall_gpu_hours=1.0,
    ),
    Scenario(
        name="E-structural-target-goes-to-review-unspent",
        description=(
            "The policy declares a skill structurally out of reach for this "
            "path, so the loop routes it to a human instead of training."
        ),
        campaigns=(),
        expected_action=REQUIRES_HUMAN_REVIEW,
        expected_reason_codes=("TREATMENT_REQUIRES_REVIEW",),
        expected_campaigns_run=0,
        expected_promotions=0,
        expected_parent_version="gen2",
        structural_skills=("instruction.formatting",),
    ),
    Scenario(
        name="F-a-failing-intervention-is-not-repeated",
        description=(
            "Every attempt fails. The loop walks to a different target and a "
            "different treatment each generation rather than re-running the same "
            "unsuccessful experiment, and the declared consecutive-non-promotion "
            "limit ends the session -- not the generation limit, and not a fourth "
            "attempt at a target that has already failed."
        ),
        campaigns=(
            SimulatedCampaign(verdict=REJECTED, wall_gpu_hours=0.4, target_effect=0.0),
            SimulatedCampaign(verdict=REJECTED, wall_gpu_hours=0.4, target_effect=0.0),
            SimulatedCampaign(verdict=REJECTED, wall_gpu_hours=0.4, target_effect=0.0),
        ),
        expected_action=STOP_PLATEAU,
        # ``CONSECUTIVE_NON_PROMOTIONS`` rather than ``TARGET_EXHAUSTED`` or
        # ``GENERATION_LIMIT_REACHED``: ``detect_plateau`` deliberately refuses to
        # call a target exhausted while an allowed treatment class is still
        # untried on it, so the honest stop is the declared non-promotion limit.
        # What this scenario pins is that the limit is reached with three
        # *distinct* targets (asserted below), never by retrying one.
        expected_reason_codes=("CONSECUTIVE_NON_PROMOTIONS",),
        expected_campaigns_run=3,
        expected_promotions=0,
        expected_parent_version="gen2",
        maximum_same_target_attempts=1,
        maximum_consecutive_non_promotions=3,
        # Two treatments, because the third target's evidence calls for ``sft``:
        # a single-treatment policy would refuse it and stop as human review
        # (scenario E covers that path), which would prove nothing about not
        # repeating a failed intervention.
        allowed_training_types=("targeted_repair", "sft"),
        expected_distinct_targets=3,
    ),
)


def run_all(*, root: str | Path, manifest_path: str | Path | None = None) -> dict[str, Any]:
    """Run every scenario under ``root`` and return their reports."""
    root = Path(root)
    reports: dict[str, Any] = {}
    for scenario in SCENARIOS:
        scenario_root = root / scenario.name
        scenario_root.mkdir(parents=True, exist_ok=True)
        report = run_scenario(scenario, root=scenario_root, manifest_path=manifest_path)
        reports[scenario.name] = report.to_dict()
    return reports
