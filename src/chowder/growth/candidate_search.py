"""Bounded candidate search over a campaign's declared recipes.

The campaign trains the recipes its declaration names; this decides *with how
much budget, in what order, and how many at a time*. It is successive halving
over the campaign's own attempts: round 0 runs every candidate at a cheap step
budget, the candidates that trained successfully (preregistered order, nothing
else) advance, and each later round runs the survivors at
``step_multiplier`` times the previous round's budget. Only the final round's
results are offered to selection, so no early, cheap round's winner can become
the campaign's candidate.

Three rules are enforced here rather than trusted:

* **the schedule is one owner.** ``successive_halving.HalvingSchedule`` decides
  the per-round step budget and the survivor count, so this module cannot drift
  from the controller that already runs the EvolutionEngine stack;
* **the screen is training-side only.** A candidate advances because its
  attempt succeeded and produced an artifact -- the same fields
  ``cycle.select_candidate`` reads. No protected, target or broad benchmark
  score is visible to the search at any point, so a hyperparameter winner can
  never be chosen on final-gate evidence;
* **the cost is bounded before it is spent.** The whole schedule is projected
  first (worst case: every candidate survives every round), checked against the
  declaration's own envelope *and* the campaign's ceilings, and refused if the
  round budgets it implies exceed the per-recipe ceilings. An undeclared search
  is no search at all: the campaign runs its recipes once, exactly as before.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

from chowder.successive_halving import HalvingSchedule

from .recipe_planner import TrainingRecipe

__all__ = [
    "CandidateSearchRefusal",
    "CandidateSearchDeclaration",
    "SearchRun",
    "SearchRound",
    "SearchPlan",
    "plan_search",
    "run_search",
    "round_recipe",
    "advanced",
    "ProjectCost",
]


class ProjectCost(Protocol):
    """The recipe planner's own projection, reused so costs cannot diverge."""

    def __call__(self, *, seq_len: int, max_steps: int) -> tuple[float, float]: ...

#: Refusal code for everything this module refuses.
SEARCH_SCHEMA = "CANDIDATE_SEARCH"

#: The declared keys, so an unrecognized one refuses instead of being ignored.
_DECLARED_KEYS = frozenset(
    {
        "rounds",
        "initial_max_steps",
        "step_multiplier",
        "survival_fraction",
        "min_survivors",
        "device_gpu_hours_ceiling",
        "wall_gpu_hours_ceiling",
    }
)


class CandidateSearchRefusal(RuntimeError):
    """The declared search cannot be planned or run as declared."""


@dataclass(frozen=True)
class CandidateSearchDeclaration:
    """A campaign's declared, preregistered candidate search.

    Absent (``rounds == 0``) means the campaign is a single pass over its
    recipes -- today's behaviour, stated rather than implied. When declared,
    every bound is required: a search that cannot name its round count, its
    starting budget or its own envelope would be an unbounded allocation, which
    is exactly what a preregistration exists to prevent.
    """

    rounds: int = 0
    initial_max_steps: int = 0
    step_multiplier: float = 2.0
    survival_fraction: float = 0.5
    min_survivors: int = 1
    device_gpu_hours_ceiling: float = 0.0
    wall_gpu_hours_ceiling: float = 0.0

    def __post_init__(self) -> None:
        if self.rounds == 0:
            return
        if self.rounds < 1:
            raise CandidateSearchRefusal(
                f"{SEARCH_SCHEMA}: rounds must be at least 1 when a search is "
                f"declared (0 means \"no search\"), got {self.rounds}"
            )
        if self.initial_max_steps < 1:
            raise CandidateSearchRefusal(
                f"{SEARCH_SCHEMA}: a declared search needs initial_max_steps "
                ">= 1: the cheap round it starts from"
            )
        for name in ("device_gpu_hours_ceiling", "wall_gpu_hours_ceiling"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise CandidateSearchRefusal(
                    f"{SEARCH_SCHEMA}: {name} must be a finite positive number; "
                    "a search with no envelope is an unbounded allocation"
                )
        # The schedule owns the rest of the validation, and raising from its
        # own constructor here means there is one set of rules, not two.
        try:
            self.schedule()
        except ValueError as error:
            raise CandidateSearchRefusal(f"{SEARCH_SCHEMA}: {error}") from error

    @property
    def declared(self) -> bool:
        return self.rounds > 0

    def schedule(self) -> HalvingSchedule:
        """The halving policy this declaration delegates every decision to."""
        return HalvingSchedule(
            initial_max_steps=self.initial_max_steps,
            step_multiplier=self.step_multiplier,
            survival_fraction=self.survival_fraction,
            min_survivors=self.min_survivors,
            max_rounds=self.rounds,
        )

    @classmethod
    def from_mapping(
        cls, document: Mapping[str, Any], *, source: str = "<memory>"
    ) -> "CandidateSearchDeclaration":
        unknown = sorted(set(document) - _DECLARED_KEYS)
        if unknown:
            raise CandidateSearchRefusal(
                f"{source}: unknown candidate_search fields {unknown}; a declared "
                "search key nothing reads is not a declaration"
            )
        return cls(**{key: document[key] for key in _DECLARED_KEYS if key in document})

    def to_dict(self) -> dict[str, Any]:
        return {
            "rounds": self.rounds,
            "initial_max_steps": self.initial_max_steps,
            "step_multiplier": self.step_multiplier,
            "survival_fraction": self.survival_fraction,
            "min_survivors": self.min_survivors,
            "device_gpu_hours_ceiling": self.device_gpu_hours_ceiling,
            "wall_gpu_hours_ceiling": self.wall_gpu_hours_ceiling,
        }


@dataclass(frozen=True)
class SearchRound:
    """One round: which candidates run, and the budget they run at."""

    round_index: int
    max_steps: int
    recipe_ids: tuple[str, ...]
    projected_device_gpu_hours: float
    projected_wall_gpu_hours: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_index": self.round_index,
            "max_steps": self.max_steps,
            "recipe_ids": list(self.recipe_ids),
            "projected_device_gpu_hours": self.projected_device_gpu_hours,
            "projected_wall_gpu_hours": self.projected_wall_gpu_hours,
        }


@dataclass(frozen=True)
class SearchPlan:
    """The whole bounded search, projected before anything is spent."""

    declared: bool
    rounds: tuple[SearchRound, ...] = ()
    total_device_gpu_hours: float = 0.0
    total_wall_gpu_hours: float = 0.0
    schedule: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "declared": self.declared,
            "rounds": [row.to_dict() for row in self.rounds],
            "total_device_gpu_hours": self.total_device_gpu_hours,
            "total_wall_gpu_hours": self.total_wall_gpu_hours,
            "schedule": dict(self.schedule),
        }


def round_recipe(recipe: TrainingRecipe, *, max_steps: int, device: float, wall: float) -> TrainingRecipe:
    """The same proposal at a different round budget.

    A survivor's next round is the *same* recipe with a larger step budget --
    the search's own axis -- so nothing about the proposal changes except the
    budget it is given, and the projections travel with it.
    """
    return dataclasses.replace(
        recipe,
        max_steps=max_steps,
        warmup_steps=max(2, max_steps // 10),
        projected_device_gpu_hours=device,
        projected_wall_gpu_hours=wall,
        notes=(
            f"{recipe.notes}; search round budget {max_steps} steps "
            f"(projected {device:.4f} device / {wall:.4f} wall GPU-h)"
        ),
    )


def plan_search(
    declaration: CandidateSearchDeclaration,
    *,
    recipes: Sequence[TrainingRecipe],
    project_cost: ProjectCost,
    per_recipe_device_ceiling: float,
    per_recipe_wall_ceiling: float,
    campaign_device_ceiling: float,
    campaign_wall_ceiling: float,
) -> SearchPlan:
    """Project every round of the declared search, or refuse it.

    ``project_cost`` is the recipe planner's own projection
    (``seq_len``, ``max_steps`` -> device/wall GPU-hours), so the search and the
    planner cannot disagree about what a step costs. The projection assumes the
    *worst* case that the screen can only improve on: every candidate survives
    every round.

    Refuses on: a round whose budget would exceed the per-recipe ceilings (the
    executor's own admission rule, applied before compute rather than during
    it), a total that exceeds the declaration's envelope, and a total that
    exceeds the campaign's ceilings.
    """
    if not declaration.declared:
        return SearchPlan(declared=False, schedule=declaration.to_dict())
    if not recipes:
        raise CandidateSearchRefusal(
            f"{SEARCH_SCHEMA}: a declared search has no candidate to run; the "
            "campaign's recipe set is what the search ranges over"
        )
    schedule = declaration.schedule()
    rounds: list[SearchRound] = []
    total_device = 0.0
    total_wall = 0.0
    candidate_ids = tuple(recipe.recipe_id for recipe in recipes)
    for round_index in range(declaration.rounds):
        if not candidate_ids:
            break
        max_steps = schedule.round_max_steps(round_index)
        round_device = 0.0
        round_wall = 0.0
        for recipe in recipes:
            if recipe.recipe_id not in candidate_ids:
                continue
            device, wall = project_cost(seq_len=recipe.seq_len, max_steps=max_steps)
            if device > per_recipe_device_ceiling or wall > per_recipe_wall_ceiling:
                raise CandidateSearchRefusal(
                    f"{SEARCH_SCHEMA}: round {round_index} would run "
                    f"{recipe.recipe_id!r} at {max_steps} steps, projecting "
                    f"{device:.6f} device / {wall:.6f} wall GPU-h against the "
                    f"declared per-recipe ceilings "
                    f"{per_recipe_device_ceiling:.6f}/{per_recipe_wall_ceiling:.6f}"
                    "; a round the executor would refuse to admit is not a round"
                )
            round_device += device
            round_wall += wall
        rounds.append(
            SearchRound(
                round_index=round_index,
                max_steps=max_steps,
                recipe_ids=candidate_ids,
                projected_device_gpu_hours=round_device,
                projected_wall_gpu_hours=round_wall,
            )
        )
        total_device += round_device
        total_wall += round_wall
        # Worst case: everyone survives, so the next round is the schedule's own
        # survivor count applied to the candidates that entered this one.
        survivors = schedule.survivors(len(candidate_ids))
        candidate_ids = candidate_ids[:survivors]
    if total_device > declaration.device_gpu_hours_ceiling:
        raise CandidateSearchRefusal(
            f"{SEARCH_SCHEMA}: the declared {declaration.rounds}-round search "
            f"projects {total_device:.6f} device GPU-h against its own declared "
            f"ceiling {declaration.device_gpu_hours_ceiling:.6f}; the search "
            "envelope is a bound, not a preference"
        )
    if total_wall > declaration.wall_gpu_hours_ceiling:
        raise CandidateSearchRefusal(
            f"{SEARCH_SCHEMA}: the declared {declaration.rounds}-round search "
            f"projects {total_wall:.6f} wall GPU-h against its own declared "
            f"ceiling {declaration.wall_gpu_hours_ceiling:.6f}"
        )
    if total_device > campaign_device_ceiling:
        raise CandidateSearchRefusal(
            f"{SEARCH_SCHEMA}: the declared search projects "
            f"{total_device:.6f} device GPU-h against the campaign ceiling "
            f"{campaign_device_ceiling:.6f}; a search must fit the campaign it "
            "belongs to"
        )
    if total_wall > campaign_wall_ceiling:
        raise CandidateSearchRefusal(
            f"{SEARCH_SCHEMA}: the declared search projects "
            f"{total_wall:.6f} wall GPU-h against the campaign ceiling "
            f"{campaign_wall_ceiling:.6f}"
        )
    return SearchPlan(
        declared=True,
        rounds=tuple(rounds),
        total_device_gpu_hours=total_device,
        total_wall_gpu_hours=total_wall,
        schedule=declaration.to_dict(),
    )


#: The training-side fields a candidate may advance on. Exactly the fields
#: ``cycle.select_candidate`` reads, and nothing else: no benchmark score of any
#: kind is reachable from here.
_ADVANCE_FIELDS = ("status", "candidate_succeeded", "artifact_ref")


def advanced(results: Sequence[Mapping[str, Any]], *, survivor_count: int) -> tuple[str, ...]:
    """Which attempts advance, in preregistered order, on training-side evidence.

    No ranking and no score: an attempt advances because it trained and produced
    an artifact. ``first_by_loss`` and every other selection policy still apply
    to the *final* round through ``cycle.select_candidate`` -- they decide the
    winner, not who is allowed to earn a larger budget.
    """
    succeeded = [
        str(row.get("recipe_id", ""))
        for row in results
        if (row.get("candidate_succeeded") is True or row.get("status") == "SUCCEEDED")
        and row.get("artifact_ref")
    ]
    return tuple(succeeded[:survivor_count])


@dataclass(frozen=True)
class SearchRun:
    """What the search actually did: every round, and the final round's results."""

    rounds: tuple[SearchRound, ...]
    round_attempts: tuple[tuple[Mapping[str, Any], ...], ...]
    survivors: tuple[str, ...]
    total_device_gpu_hours: float
    total_wall_gpu_hours: float
    stopped_by: str | None = None

    @property
    def final_results(self) -> tuple[Mapping[str, Any], ...]:
        """Only the final round's attempts: selection may not see earlier ones."""
        return self.round_attempts[-1] if self.round_attempts else ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "rounds": [row.to_dict() for row in self.rounds],
            "survivors": list(self.survivors),
            "total_device_gpu_hours": self.total_device_gpu_hours,
            "total_wall_gpu_hours": self.total_wall_gpu_hours,
            "stopped_by": self.stopped_by,
        }


def run_search(
    plan: SearchPlan,
    *,
    declaration: CandidateSearchDeclaration,
    recipes: Sequence[TrainingRecipe],
    project_cost: ProjectCost,
    run_attempt: Callable[[TrainingRecipe], Mapping[str, Any]],
    on_attempt: Callable[[Mapping[str, Any], SearchRound], None] | None = None,
    should_stop: Callable[[float, float], str | None] | None = None,
) -> SearchRun:
    """Run the planned rounds, screening on training-side evidence only.

    ``run_attempt`` is the campaign's own executor seam (``executor(recipe,
    items)`` with the declared curriculum), so every attempt goes through the
    same binding, admission, accounting and evidence path a single-pass campaign
    uses. ``on_attempt`` lets the caller charge the ledger and record the
    attempt as it happens -- a round that is stopped mid-flight still leaves its
    spend recorded.
    """
    if not plan.declared:
        raise CandidateSearchRefusal(
            f"{SEARCH_SCHEMA}: an undeclared search has no rounds to run; the "
            "campaign's single pass is not a search"
        )
    if dict(plan.schedule) != declaration.to_dict():
        raise CandidateSearchRefusal(
            f"{SEARCH_SCHEMA}: this plan was projected from a different search "
            "declaration than the one it is being run under; the budget a round "
            "allocates and the budget that was admitted must be one decision"
        )
    schedule = declaration.schedule()
    by_id = {recipe.recipe_id: recipe for recipe in recipes}
    planned = set(plan.rounds[0].recipe_ids) if plan.rounds else set()
    unknown = sorted(planned - set(by_id))
    if unknown:
        raise CandidateSearchRefusal(
            f"{SEARCH_SCHEMA}: this plan ranges over {unknown}, which the "
            "campaign's recipe set does not hold; a search may only run the "
            "candidates the campaign declared"
        )
    round_attempts: list[tuple[Mapping[str, Any], ...]] = []
    # Round 0 runs the candidates the *plan* projected, never whatever recipe
    # set this call was handed. The plan is what the ceilings were checked
    # against, so seeding from anything wider would spend a round on candidates
    # no admission rule ever saw.
    survivors: tuple[str, ...] = plan.rounds[0].recipe_ids if plan.rounds else ()
    spent_device = 0.0
    spent_wall = 0.0
    stopped_by: str | None = None
    for row in plan.rounds:
        # The plan's per-round budget is what the search allocates; *who* runs a
        # round is the screen's answer -- the survivors of the round before,
        # in preregistered order -- not the plan's worst-case narrowing. The
        # plan projects the worst case; the run spends the real case.
        candidates = tuple(recipe_id for recipe_id in survivors if recipe_id in by_id)
        if not candidates:
            break
        attempts: list[Mapping[str, Any]] = []
        for recipe_id in candidates:
            recipe = by_id.get(recipe_id)
            if recipe is None:
                raise CandidateSearchRefusal(
                    f"{SEARCH_SCHEMA}: round {row.round_index} names "
                    f"{recipe_id!r}, which the campaign's recipe set does not hold"
                )
            device, wall = project_cost(seq_len=recipe.seq_len, max_steps=row.max_steps)
            evidence = dict(
                run_attempt(round_recipe(recipe, max_steps=row.max_steps, device=device, wall=wall))
            )
            evidence["recipe_id"] = recipe_id
            evidence["search_round"] = row.round_index
            evidence["search_max_steps"] = row.max_steps
            attempts.append(evidence)
            if on_attempt is not None:
                on_attempt(evidence, row)
            spent_device += device
            spent_wall += wall
            if should_stop is not None and stopped_by is None:
                stopped_by = should_stop(spent_device, spent_wall)
                if stopped_by is not None:
                    # The ceiling that stopped this attempt stops the next one
                    # too: finishing the round would spend past the bound that
                    # just tripped, which is what the single pass refuses to do.
                    break
        round_attempts.append(tuple(attempts))
        survivors = advanced(attempts, survivor_count=schedule.survivors(len(attempts)))
        if stopped_by is not None:
            break
    return SearchRun(
        rounds=plan.rounds[: len(round_attempts)],
        round_attempts=tuple(round_attempts),
        survivors=survivors,
        total_device_gpu_hours=spent_device,
        total_wall_gpu_hours=spent_wall,
        stopped_by=stopped_by,
    )
