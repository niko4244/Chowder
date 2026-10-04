"""The adaptive budget ladder: escalation earned by evidence, never by mood.

A campaign starts deterministic -- the declared halving schedule, candidates
in declaration order, zero assumptions. As the evidence store fills, the
ladder may escalate, one rung at a time, and every rung is a *reordering*
inside the declared envelope, never a budget change:

- DETERMINISTIC: the declaration's own schedule and order. Always available,
  even with an empty store.
- PRIOR_WEIGHTED: round-0 entry order follows the evidence store's priors
  (families with measured failures enter last, promising families first).
  Who may *exist* as a candidate is the maturity gate's decision, not this
  ladder's -- a prior of 0.0 refuses rather than silently dropping anyone.
- ADAPTIVE: from the second round on, survivors are ordered by UCB1 over
  training-side efficiency (a training-side score per device GPU-hour).
  Explore-first: an unobserved candidate outranks every observed one.

Two walls this ladder does not cross:

1. **The budget wall.** ``HalvingSchedule`` owns step budgets and survivor
   counts; ``plan_search`` owns the ceilings. The ladder reorders who runs
   and who survives *within* the plan those functions projected. It cannot
   add a round, grow a budget, or raise a ceiling, and it re-verifies every
   reordered plan against the same worst-case totals before returning.
2. **The evidence wall.** The adaptive stage's reward is a *training-side*
   score per GPU-hour. A gate score, a protected benchmark, or any tier-3
   number is survivor- or promotion-tier evidence: feeding one to the search
   order would breach ``eval_isolation``. Expected improvement
   (``expected_improvement.py``) is a *campaign-layer* selector over
   completed, gated runs -- it is deliberately not a search-time stage here,
   both because its reward reads gate scores and because it has not been
   shown to beat UCB1 on this repository's data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:  # pragma: no cover - import cycle guard for type checkers only
    from .candidate_search import SearchPlan

__all__ = [
    "BudgetLadderRefusal",
    "LadderStage",
    "allocate",
    "entry_order",
    "ladder_stage",
    "survivor_orderer",
    "ucb1_order",
]

#: Refusal code for everything this module refuses.
LADDER_SCHEMA = "BUDGET_LADDER"


class BudgetLadderRefusal(RuntimeError):
    """The requested allocation cannot be made inside the declared walls."""


class LadderStage(str, Enum):
    """How much evidence the allocation is allowed to act on."""

    #: The declared schedule and order, no assumptions.
    DETERMINISTIC = "deterministic"
    #: Entry order follows evidence-store priors; survivor order unchanged.
    PRIOR_WEIGHTED = "prior-weighted"
    #: From round 1 on, survivor order follows UCB1 over training-side
    #: efficiency. Entry order still follows priors.
    ADAPTIVE = "adaptive"


#: Campaign-policy values for the ``budget_ladder`` key, and the rung each
#: one permits. The default policy value is the floor.
_POLICY_RUNGS: dict[str, LadderStage] = {
    "deterministic": LadderStage.DETERMINISTIC,
    "prior-weighted": LadderStage.PRIOR_WEIGHTED,
    "adaptive": LadderStage.ADAPTIVE,
}

#: Completed searches recorded before the adaptive rung unlocks. One search
#: is a anecdote, not a distribution; two is the minimum the UCB1 ordering
#: has any business acting on.
_MIN_COMPLETED_SEARCHES_FOR_ADAPTIVE = 2


def ladder_stage(
    *,
    policy: Mapping[str, Any],
    evidence_records: Sequence[Mapping[str, Any]],
    family_ids: Sequence[str],
    completed_searches: int = 0,
) -> tuple[LadderStage, tuple[str, ...]]:
    """The rung this campaign may stand on, and why it may stand there.

    ``policy["budget_ladder"]`` (default ``"deterministic"``) names the rung
    the campaign is *allowed*; the evidence decides how far up it actually
    gets. The reasons tuple records the escalation decision either way, so a
    stage is never an unexplained default.

    Escalation requirements:

    - PRIOR_WEIGHTED: the policy permits it and *every* family in the
      candidate set has at least one evidence record. A family with no
      history is an exploration candidate by definition -- priors cannot
      order what does not exist for everyone.
    - ADAPTIVE: the policy permits it, the prior-weighted conditions hold,
      and at least two completed searches have been recorded.
    """
    requested = policy.get("budget_ladder", "deterministic")
    if requested not in _POLICY_RUNGS:
        raise BudgetLadderRefusal(
            f"{LADDER_SCHEMA}: policy budget_ladder {requested!r} is not one "
            f"of {sorted(_POLICY_RUNGS)}; an unknown allocation policy is a "
            "refusal, not a default"
        )
    permitted = _POLICY_RUNGS[requested]

    reasons: list[str] = [f"policy permits {permitted.value}"]
    if permitted is LadderStage.DETERMINISTIC:
        return (LadderStage.DETERMINISTIC, tuple(reasons))

    families = set(family_ids)
    families_with_history = {
        str(record.get("family_id", ""))
        for record in evidence_records
        if record.get("family_id")
    }
    missing = sorted(families - families_with_history)
    if missing:
        reasons.append(
            f"deterministic: {missing} has no evidence record; priors cannot "
            "order a family they know nothing about"
        )
        return (LadderStage.DETERMINISTIC, tuple(reasons))
    reasons.append("every candidate family has evidence history")

    if permitted is LadderStage.PRIOR_WEIGHTED:
        return (LadderStage.PRIOR_WEIGHTED, tuple(reasons))

    if completed_searches < _MIN_COMPLETED_SEARCHES_FOR_ADAPTIVE:
        reasons.append(
            f"adaptive needs at least {_MIN_COMPLETED_SEARCHES_FOR_ADAPTIVE} "
            f"completed searches recorded, has {completed_searches}"
        )
        return (LadderStage.PRIOR_WEIGHTED, tuple(reasons))
    reasons.append(f"{completed_searches} completed searches recorded")
    return (LadderStage.ADAPTIVE, tuple(reasons))


def entry_order(
    recipe_ids: Sequence[str],
    *,
    priors: Mapping[str, float],
) -> tuple[str, ...]:
    """Round-0 entry order under the evidence priors.

    ``priors`` is keyed by *recipe id* (the campaign maps its families'
    prior multipliers onto its own candidates). A higher multiplier enters
    first; ties break by recipe_id so the order is deterministic whatever
    the declaration order was. A prior of 0.0 (or a missing entry) refuses:
    an architecture-incompatible family is the maturity gate's exclusion,
    and an absent prior means the stage check was skipped -- both are
    failures to refuse, not candidates to drop quietly.
    """
    for recipe_id in recipe_ids:
        multiplier = priors.get(recipe_id)
        if multiplier is None:
            raise BudgetLadderRefusal(
                f"{LADDER_SCHEMA}: candidate {recipe_id!r} has no prior; "
                "prior-weighted entry needs a multiplier for every candidate "
                "(1.0 for an explicitly-unweighted one)"
            )
        if not math.isfinite(multiplier) or multiplier <= 0.0:
            raise BudgetLadderRefusal(
                f"{LADDER_SCHEMA}: candidate {recipe_id!r} has prior "
                f"{multiplier!r}; excluding a candidate is the maturity "
                "gate's decision, and the ladder refuses to make it silently"
            )
    return tuple(
        sorted(recipe_ids, key=lambda recipe_id: (-priors[recipe_id], recipe_id))
    )


def ucb1_order(
    candidates: Sequence[str],
    *,
    efficiency: Mapping[str, tuple[float, int]],
    exploration: float = math.sqrt(2.0),
) -> tuple[str, ...]:
    """UCB1 ordering of ``candidates`` by training-side efficiency.

    ``efficiency[recipe_id]`` is ``(mean_score_per_gpu_hour, observation_count)``
    -- the campaign's own training-side statistic (for example loss
    improvement per device GPU-hour), never a gate or benchmark score. A
    candidate with zero observations outranks every observed one: with no
    measurement, allocation is pure exploration. Ties break by recipe_id.
    """
    total_observations = sum(count for _, count in efficiency.values())
    indexed = {recipe_id: index for index, recipe_id in enumerate(candidates)}

    def sort_key(recipe_id: str) -> tuple[float, str]:
        entry = efficiency.get(recipe_id)
        if entry is None or entry[1] <= 0:
            # Explore-first: infinity sorts above any finite UCB1 score.
            return (-math.inf, recipe_id)
        mean, count = entry
        bonus = exploration * math.sqrt(math.log(max(total_observations, 1)) / count)
        return (-(mean + bonus), recipe_id)

    unknown = [recipe_id for recipe_id in candidates if recipe_id not in indexed]
    if unknown:  # pragma: no cover - defensive, candidates is the input list
        raise BudgetLadderRefusal(
            f"{LADDER_SCHEMA}: {unknown} are not candidates of this round"
        )
    return tuple(sorted(candidates, key=sort_key))


def survivor_orderer(
    *,
    stage: LadderStage,
    efficiency: Mapping[str, tuple[float, int]],
) -> Any:
    """The ``order_survivors`` hook ``run_search`` accepts, for this stage.

    Deterministic and prior-weighted stages get the identity order (the
    preregistered order the plan and the screen already produce); the
    adaptive stage orders each round's runnable survivors by UCB1 over the
    training-side efficiency observations.
    """
    if stage is not LadderStage.ADAPTIVE:
        return lambda recipe_ids, round_row: list(recipe_ids)
    return lambda recipe_ids, round_row: list(
        ucb1_order(recipe_ids, efficiency=efficiency)
    )


@dataclass(frozen=True)
class AllocationReceipt:
    """What the ladder changed, and the totals it verified it did not change."""

    stage: LadderStage
    entry_recipe_ids: tuple[str, ...]
    round_recipe_ids: tuple[tuple[str, ...], ...]
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "entry_recipe_ids": list(self.entry_recipe_ids),
            "round_recipe_ids": [list(ids) for ids in self.round_recipe_ids],
            "reasons": list(self.reasons),
        }


def allocate(
    plan: "SearchPlan",
    *,
    stage: LadderStage,
    priors: Mapping[str, float],
    efficiency: Mapping[str, tuple[float, int]] | None = None,
    reasons: tuple[str, ...] = (),
) -> tuple["SearchPlan", AllocationReceipt]:
    """Reorder a plan's rounds for this stage, inside the plan it was given.

    Returns a plan whose per-round candidate *sets* are identical to the
    input's -- so its worst-case projections, and every ceiling check
    ``plan_search`` already performed, still hold verbatim -- with the
    ordering changed:

    - round 0: ``entry_order`` under the priors (deterministic stage keeps
      the declaration order, since a prior-weighted reorder without prior
      authority would be an unexplained preference);
    - round r >= 1: ``ucb1_order`` over the efficiency observations, filtered
      to that round's candidates (deterministic and prior-weighted stages
      keep the plan's own order).

    The result satisfies ``run_search``'s plan-vs-declaration identity check:
    the schedule dict is untouched, so the budget a round allocates remains
    exactly the budget that was admitted.
    """
    from .candidate_search import SearchPlan as _SearchPlan, SearchRound

    if not plan.declared:
        raise BudgetLadderRefusal(
            f"{LADDER_SCHEMA}: an undeclared search has nothing to allocate"
        )
    round_ids = [row.recipe_ids for row in plan.rounds]
    if stage is LadderStage.DETERMINISTIC:
        receipt = AllocationReceipt(
            stage=stage,
            entry_recipe_ids=round_ids[0] if round_ids else (),
            round_recipe_ids=tuple(round_ids),
            reasons=reasons or ("deterministic: the declared order is the order",),
        )
        return (plan, receipt)

    if stage is LadderStage.ADAPTIVE and efficiency is None:
        raise BudgetLadderRefusal(
            f"{LADDER_SCHEMA}: the adaptive stage orders rounds by UCB1 over "
            "training-side efficiency, and no observations were given"
        )

    ordered_rounds: list[tuple[str, ...]] = []
    for row in plan.rounds:
        ids = row.recipe_ids
        if row.round_index == 0:
            ordered_rounds.append(entry_order(ids, priors=priors))
        elif stage is LadderStage.ADAPTIVE:
            ordered_rounds.append(
                ucb1_order(ids, efficiency=efficiency or {})
            )
        else:
            ordered_rounds.append(ids)

    # Set identity: a reorder that changed *who* runs a round would change
    # the worst-case projection the ceilings were checked against.
    for row, ordered in zip(plan.rounds, ordered_rounds):
        if set(ordered) != set(row.recipe_ids):
            raise BudgetLadderRefusal(
                f"{LADDER_SCHEMA}: reordering round {row.round_index} changed "
                "its candidate set; allocation may reorder the plan, never "
                "rewrite who was admitted"
            )

    reordered = _SearchPlan(
        declared=plan.declared,
        rounds=tuple(
            SearchRound(
                round_index=row.round_index,
                max_steps=row.max_steps,
                recipe_ids=ordered,
                projected_device_gpu_hours=row.projected_device_gpu_hours,
                projected_wall_gpu_hours=row.projected_wall_gpu_hours,
            )
            for row, ordered in zip(plan.rounds, ordered_rounds)
        ),
        total_device_gpu_hours=plan.total_device_gpu_hours,
        total_wall_gpu_hours=plan.total_wall_gpu_hours,
        schedule=plan.schedule,
    )
    stage_reasons = reasons or (
        f"{stage.value}: reordered inside the declared envelope; totals "
        f"{plan.total_device_gpu_hours:.6f} device / "
        f"{plan.total_wall_gpu_hours:.6f} wall GPU-h unchanged"
    )
    receipt = AllocationReceipt(
        stage=stage,
        entry_recipe_ids=ordered_rounds[0],
        round_recipe_ids=tuple(ordered_rounds),
        reasons=stage_reasons,
    )
    return (reordered, receipt)
