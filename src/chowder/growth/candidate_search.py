"""Bounded candidate search over a campaign's declared recipes.

The campaign trains the recipes its declaration names; this decides *with how
much budget, in what order, and how many at a time*. It is successive halving
over the campaign's own attempts: round 0 runs every candidate at a cheap step
budget, the candidates that trained successfully (preregistered order, nothing
else) advance, and each later round *continues* the survivors -- resuming the
checkpoint their previous round produced and training only the incremental
steps toward ``step_multiplier`` times the previous round's budget, instead of
restarting the recipe from step 0 and paying for the earlier rounds twice. A
survivor with no checkpoint to continue from, or one whose executor reports it
restarted instead of resuming, ends its lineage rather than advancing. Only the
final round's results are offered to selection, so no early, cheap round's
winner can become the campaign's candidate.

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

from chowder.successive_halving import HalvingSchedule, latest_checkpoint_dir

from .recipe_planner import TrainingRecipe

__all__ = [
    "CandidateSearchRefusal",
    "CandidateSearchDeclaration",
    "SearchRun",
    "SearchRound",
    "SearchPlan",
    "SearchProgress",
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


def round_recipe(
    recipe: TrainingRecipe,
    *,
    max_steps: int,
    device: float,
    wall: float,
    resume_from: str | None = None,
) -> TrainingRecipe:
    """The same proposal at a different round budget.

    A survivor's next round is the *same* recipe with a larger step budget --
    the search's own axis -- so nothing about the proposal changes except the
    budget it is given, and the projections travel with it.

    When ``resume_from`` names the survivor's previous round's checkpoint, the
    round is a *continuation*: the backend resumes optimizer/scheduler/RNG
    state from that checkpoint and trains only the additional steps, instead
    of restarting the recipe from step 0 and paying for the earlier rounds a
    second time. The peft backend excludes ``resume_from_checkpoint`` from the
    checkpoint recipe digest, so the continuation keeps the checkpoint's own
    bound identity -- and its manifest refuses to resume into a checkpoint
    whose recipe, dataset, or base no longer matches.
    """
    continuation = (
        f"; continues from checkpoint {resume_from} toward "
        f"{max_steps} steps" if resume_from else ""
    )
    return dataclasses.replace(
        recipe,
        max_steps=max_steps,
        warmup_steps=max(2, max_steps // 10),
        projected_device_gpu_hours=device,
        projected_wall_gpu_hours=wall,
        resume_from_checkpoint=resume_from,
        notes=(
            f"{recipe.notes}; search round budget {max_steps} steps "
            f"(projected {device:.4f} device / {wall:.4f} wall GPU-h)"
            f"{continuation}"
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

    Rounds after the first are **continuations**: a survivor resumes its own
    previous checkpoint and trains only the steps between the two budgets, so
    a round's projected cost is the projection of that step *delta*, not of a
    full retrain. Progressive allocation that priced round r at the full
    ``max_steps`` cost would double-charge every earlier round and make the
    schedule look unaffordable; priced at the delta it measures what the round
    actually adds. The projection is a floor (checkpoint restore has a real,
    unmodeled cost the executor meters when it happens), and it is stated as
    such in each round's record.

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
    previous_max_steps = 0
    for round_index in range(declaration.rounds):
        if not candidate_ids:
            break
        max_steps = schedule.round_max_steps(round_index)
        # Continuation pricing: rounds after the first train only the step
        # delta between budgets (see the docstring); round 0 trains its full
        # budget from the parent.
        incremental_steps = max(1, max_steps - previous_max_steps)
        round_device = 0.0
        round_wall = 0.0
        for recipe in recipes:
            if recipe.recipe_id not in candidate_ids:
                continue
            device, wall = project_cost(
                seq_len=recipe.seq_len, max_steps=incremental_steps
            )
            if device > per_recipe_device_ceiling or wall > per_recipe_wall_ceiling:
                raise CandidateSearchRefusal(
                    f"{SEARCH_SCHEMA}: round {round_index} would run "
                    f"{recipe.recipe_id!r} at {max_steps} steps "
                    f"({incremental_steps} incremental after "
                    f"{previous_max_steps}), projecting "
                    f"{device:.6f} device / {wall:.6f} wall GPU-h against the "
                    f"declared per-recipe ceilings "
                    f"{per_recipe_device_ceiling:.6f}/{per_recipe_wall_ceiling:.6f}"
                    "; a round the executor would refuse to admit is not a round"
                )
            round_device += device
            round_wall += wall
        previous_max_steps = max_steps
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
    #: Per-candidate cumulative projected spend across the rounds it ran, so
    #: a lineage's full cost is one lookup instead of re-summing attempts.
    candidate_cumulative: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    #: Why a candidate's lineage ended without advancing: no checkpoint to
    #: continue from, or an executor-reported restart instead of a resume.
    lineage_stops: Mapping[str, str] = field(default_factory=dict)

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
            "candidate_cumulative": {
                recipe_id: dict(row) for recipe_id, row in self.candidate_cumulative.items()
            },
            "lineage_stops": dict(self.lineage_stops),
        }


@dataclass(frozen=True)
class SearchProgress:
    """A durable snapshot of a partially-run search, taken between rounds.

    An interrupted search must be resumable, and it must resume *as the same
    search*: the same completed rounds (and their recorded spend), the same
    survivors, and -- because a round after the first is a continuation -- the
    checkpoint identity each survivor continues from. ``from_run`` derives the
    snapshot from a run's own record, so the resume path cannot disagree with
    what the interrupted run actually did.
    """

    rounds: tuple[SearchRound, ...]
    round_attempts: tuple[tuple[Mapping[str, Any], ...], ...]
    survivors: tuple[str, ...]
    spent_device_gpu_hours: float
    spent_wall_gpu_hours: float

    @classmethod
    def from_run(cls, run: SearchRun) -> "SearchProgress":
        if run.stopped_by is None:
            raise CandidateSearchRefusal(
                f"{SEARCH_SCHEMA}: a concluded search has nothing to resume; "
                "progress snapshots exist for interrupted searches"
            )
        return cls(
            rounds=run.rounds,
            round_attempts=run.round_attempts,
            survivors=run.survivors,
            spent_device_gpu_hours=run.total_device_gpu_hours,
            spent_wall_gpu_hours=run.total_wall_gpu_hours,
        )


def _attempt_checkpoint(evidence: Mapping[str, Any]) -> str | None:
    """The checkpoint this attempt produced, from its own artifact record.

    Same resolution the EvolutionEngine controller uses: the highest real
    ``checkpoint-N`` directory the trainer actually wrote under the attempt's
    artifact. ``None`` means the attempt left nothing to continue from.
    """
    artifact_ref = evidence.get("artifact_ref")
    if not artifact_ref:
        return None
    checkpoint = latest_checkpoint_dir(str(artifact_ref))
    return str(checkpoint) if checkpoint is not None else None


def run_search(
    plan: SearchPlan,
    *,
    declaration: CandidateSearchDeclaration,
    recipes: Sequence[TrainingRecipe],
    project_cost: ProjectCost,
    run_attempt: Callable[[TrainingRecipe], Mapping[str, Any]],
    on_attempt: Callable[[Mapping[str, Any], SearchRound], None] | None = None,
    should_stop: Callable[[float, float], str | None] | None = None,
    progress: SearchProgress | None = None,
    order_survivors: Callable[[Sequence[str], SearchRound], Sequence[str]] | None = None,
) -> SearchRun:
    """Run the planned rounds, screening on training-side evidence only.

    ``run_attempt`` is the campaign's own executor seam (``executor(recipe,
    items)`` with the declared curriculum), so every attempt goes through the
    same binding, admission, accounting and evidence path a single-pass campaign
    uses. ``on_attempt`` lets the caller charge the ledger and record the
    attempt as it happens -- a round that is stopped mid-flight still leaves its
    spend recorded.

    **Progressive allocation is continuation, not restart.** From the second
    round on, a survivor's attempt carries ``resume_from_checkpoint`` pointing
    at the checkpoint *its own previous round* produced; the backend restores
    optimizer/scheduler/RNG state from it and trains only the incremental
    steps. Three invariants keep that honest:

    * a survivor with no resolvable checkpoint cannot earn a larger budget --
      the lineage ends there rather than silently restarting from the parent
      (the same rule the EvolutionEngine controller applies);
    * an executor that reports ``resume_state == "not-a-resume"`` for a
      declared continuation has admitted it restarted: the attempt stays in
      the accounting, but its lineage cannot advance, so a silent restart can
      never win progressive allocation;
    * every attempt's evidence records the continuation it declared and the
      checkpoint it produced, so the search's record is auditable round by
      round and an interrupted run resumes from exactly what it recorded.

    ``progress`` resumes an interrupted search: the completed rounds, their
    spend and their survivors are carried over unchanged and the remaining
    rounds continue from the recorded survivors.

    ``order_survivors`` (the budget ladder's adaptive seam) may reorder the
    survivors each round runs in -- the order is what ``advanced`` truncates,
    so it decides who a binding survivor cut keeps. It receives the round's
    runnable ids and the round row, and must return the same ids: a hook that
    drops or adds a candidate refuses, because admission is the plan's
    decision, not the ordering hook's.
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
    completed_rounds = 0
    # survivors maps a candidate to the checkpoint its NEXT round continues
    # from (None for round 0's entrants: they start from the parent).
    survivor_checkpoints: dict[str, str | None] = {}
    spent_device = 0.0
    spent_wall = 0.0
    if progress is not None:
        prior_rounds = len(progress.rounds)
        if prior_rounds >= len(plan.rounds) or (
            prior_rounds > 0
            and [row.to_dict() for row in progress.rounds]
            != [row.to_dict() for row in plan.rounds[:prior_rounds]]
        ):
            raise CandidateSearchRefusal(
                f"{SEARCH_SCHEMA}: the progress snapshot carries {prior_rounds} "
                f"round(s) that do not prefix this {len(plan.rounds)}-round plan; "
                "a search may only resume as the search it was declared as"
            )
        round_attempts = list(progress.round_attempts)
        completed_rounds = prior_rounds
        last_round = progress.round_attempts[-1] if progress.round_attempts else ()
        resumed_survivors = {
            str(evidence.get("recipe_id", "")): _attempt_checkpoint(evidence)
            for evidence in last_round
            if str(evidence.get("recipe_id", ""))
        }
        survivor_checkpoints = resumed_survivors or {
            recipe_id: None for recipe_id in progress.survivors
        }
        spent_device = progress.spent_device_gpu_hours
        spent_wall = progress.spent_wall_gpu_hours
    else:
        # Round 0 runs the candidates the *plan* projected, never whatever
        # recipe set this call was handed. The plan is what the ceilings were
        # checked against, so seeding from anything wider would spend a round
        # on candidates no admission rule ever saw.
        survivor_checkpoints = {
            recipe_id: None for recipe_id in (plan.rounds[0].recipe_ids if plan.rounds else ())
        }

    candidate_cumulative: dict[str, dict[str, float]] = {}
    lineage_stops: dict[str, str] = {}
    stopped_by: str | None = None

    for row in plan.rounds[completed_rounds:]:
        # The plan's per-round budget is what the search allocates; *who* runs
        # a round is the screen's answer -- the survivors of the round before,
        # in preregistered order -- not the plan's worst-case narrowing. The
        # plan projects the worst case; the run spends the real case.
        #
        # A survivor without a checkpoint to continue from ends its lineage
        # here, honestly: continuing it would mean restarting the recipe from
        # step 0 and paying its earlier rounds twice.
        runnable: dict[str, str | None] = {}
        for recipe_id, checkpoint in survivor_checkpoints.items():
            if recipe_id not in by_id:
                continue
            if row.round_index > 0 and not checkpoint:
                lineage_stops.setdefault(
                    recipe_id, "no checkpoint from the previous round to continue from"
                )
                continue
            runnable[recipe_id] = checkpoint
        if not runnable:
            break
        if order_survivors is not None:
            ordered_ids = list(order_survivors(list(runnable), row))
            if set(ordered_ids) != set(runnable):
                raise CandidateSearchRefusal(
                    f"{SEARCH_SCHEMA}: the survivor-ordering hook changed who "
                    f"runs round {row.round_index} ({sorted(ordered_ids)} vs "
                    f"{sorted(runnable)}); allocation may reorder a round, "
                    "never rewrite who was admitted"
                )
            runnable = {recipe_id: runnable[recipe_id] for recipe_id in ordered_ids}
        attempts: list[Mapping[str, Any]] = []
        for recipe_id, checkpoint in runnable.items():
            recipe = by_id[recipe_id]
            # Continuation pricing: this attempt trains the steps between the
            # previous round's budget and this round's (the plan projected the
            # same delta), never the full round budget again.
            previous_steps = plan.rounds[row.round_index - 1].max_steps if row.round_index > 0 else 0
            incremental_steps = max(1, row.max_steps - previous_steps)
            device, wall = project_cost(seq_len=recipe.seq_len, max_steps=incremental_steps)
            evidence = dict(
                run_attempt(
                    round_recipe(
                        recipe,
                        max_steps=row.max_steps,
                        device=device,
                        wall=wall,
                        resume_from=checkpoint,
                    )
                )
            )
            evidence["recipe_id"] = recipe_id
            evidence["search_round"] = row.round_index
            evidence["search_max_steps"] = row.max_steps
            evidence["search_incremental_steps"] = incremental_steps
            evidence["declared_resume_from"] = checkpoint
            evidence["checkpoint_dir"] = _attempt_checkpoint(evidence)
            attempts.append(evidence)
            cumulative = candidate_cumulative.setdefault(
                recipe_id,
                {"device_gpu_hours": 0.0, "wall_gpu_hours": 0.0, "rounds": 0},
            )
            cumulative["device_gpu_hours"] = round(
                cumulative["device_gpu_hours"] + device, 9
            )
            cumulative["wall_gpu_hours"] = round(cumulative["wall_gpu_hours"] + wall, 9)
            cumulative["rounds"] = int(cumulative["rounds"]) + 1
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

        # Who advances: candidates that trained and produced an artifact -- and,
        # when a continuation was declared, did not report restarting. A silent
        # restart must never win progressive allocation.
        eligible = [
            evidence
            for evidence in attempts
            if not (
                evidence.get("declared_resume_from")
                and evidence.get("resume_state") == "not-a-resume"
            )
        ]
        for evidence in attempts:
            if (
                evidence.get("declared_resume_from")
                and evidence.get("resume_state") == "not-a-resume"
                and str(evidence.get("recipe_id")) not in lineage_stops
            ):
                lineage_stops[str(evidence.get("recipe_id"))] = (
                    "the attempt reported it restarted instead of resuming the "
                    "declared checkpoint"
                )
        advanced_ids = advanced(
            eligible, survivor_count=schedule.survivors(len(attempts))
        )
        next_checkpoints: dict[str, str | None] = {}
        for evidence in attempts:
            recipe_id = str(evidence.get("recipe_id", ""))
            if recipe_id in advanced_ids:
                next_checkpoints[recipe_id] = evidence.get("checkpoint_dir") or None
        survivor_checkpoints = next_checkpoints
        if stopped_by is not None:
            break

    return SearchRun(
        rounds=plan.rounds[: len(round_attempts)],
        round_attempts=tuple(round_attempts),
        # A lineage that ended (no checkpoint, or a reported restart) is not a
        # survivor: reporting it as one would offer a dead lineage to the next
        # round's narrowing and misstate what can still continue.
        survivors=tuple(
            recipe_id
            for recipe_id in survivor_checkpoints
            if recipe_id not in lineage_stops
        ),
        total_device_gpu_hours=spent_device,
        total_wall_gpu_hours=spent_wall,
        stopped_by=stopped_by,
        candidate_cumulative=candidate_cumulative,
        lineage_stops=lineage_stops,
    )
