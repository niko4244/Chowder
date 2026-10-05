"""Curriculum, recipes and the cycle, derived from the declaration alone.

``plan_campaign`` is the planning entry point and is also what the readiness
gate plans with, so a plan that cannot be built refuses before any compute.
Nothing here reads a measurement to decide what to train.

Extracted from ``campaign_runner``, which keeps the orchestration and the
two injection seams. Nothing here re-implements a decision another module owns.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing import Any, Sequence

from pathlib import Path
from ..campaign import PROMOTION_POLICY_VERSION, CampaignManifest, CampaignManifestError, settle_campaign, settle_campaign_projection, stops_on_admission_refusal, stops_on_campaign_overrun
from ..curriculum import CurriculumEngine, CurriculumItem
from ..cycle import CycleConfig, CycleOutcome, GrowthCycle, TrainingFn
from ..failure_bank import FailureBank
from ..frontier_reference import SnapshotStore
from ..lineage import GenerationLedger, RegressionMemory
from ..recipe_planner import HardwareBudget, RecipePlanner, TrainingRecipe
from ..training_binding import STATUS_SUCCEEDED, GrowthEnvelope, SubprocessTrainingFn, default_runner, directory_digest
from ..campaign_controllers.contracts import CampaignRunRefusal, FIELD_ENFORCEMENT, NON_BEHAVIORAL_FIELDS, assert_every_field_enforced
from ..campaign_controllers.declared import DECLARED_INPUT_REQUIREMENTS, _require_path, require_declared_inputs, undeclared_inputs
from ..campaign_controllers.evaluation import PARENT_PROFILE_NOT_ATTRIBUTED, _evaluate_candidate, _load_binder, _load_data_registry, _load_hardware_budget, _load_material, _load_profile, _runs_from_report

from ..candidate_search import CandidateSearchRefusal, CandidateSearchDeclaration, SearchPlan, plan_search, run_search
from ..contamination import ContaminationFirewall


@dataclass(frozen=True)
class CampaignPlan:
    """What the declared inputs plan to run, before compute or admission.

    ``recipes`` are the planner's own proposals, in the order the declared
    ``recipe_ids`` select them: ``chowder growth campaign plan`` prints these
    ids so a preregistration can name the recipes it will actually run, and
    the runner refuses a declared id the planner did not propose.

    ``search`` is the declared bounded candidate search projected over those
    same recipes -- the rounds, their step budgets and the worst-case total. It
    is planned here, once, so the plan a command prints, the readiness check
    that admits it and the run that executes it are the same arithmetic.
    """

    items: tuple[CurriculumItem, ...]
    recipes: tuple[TrainingRecipe, ...]
    search: SearchPlan = SearchPlan(declared=False)
    #: Why the declared search could not be projected, when it could not. The
    #: *plan* still exists -- the curriculum and the recipe proposal are what
    #: this object answers for -- and the search's own refusal is carried here
    #: so the check that owns it can report it, rather than every reader of a
    #: plan inheriting a refusal that belongs to one of its parts.
    search_refusal: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "curriculum_items": [item.item_id for item in self.items],
            "candidate_search": self.search.to_dict(),
            "candidate_search_refusal": self.search_refusal,
            "recipes": [
                {
                    "recipe_id": recipe.recipe_id,
                    "projected_device_gpu_hours": float(recipe.projected_device_gpu_hours),
                    "projected_wall_gpu_hours": float(recipe.projected_wall_gpu_hours),
                    "curriculum_item_ids": list(recipe.curriculum_item_ids),
                    "max_steps": recipe.max_steps,
                    "learning_rate": recipe.learning_rate,
                    "lora_rank": recipe.lora_rank,
                }
                for recipe in self.recipes
            ],
        }


class _NullExecutor:
    """Planning needs a cycle, not an execution path: nothing runs in it."""

    firewall = ContaminationFirewall()

    def admit(self, recipe: TrainingRecipe) -> tuple[str, str] | None:  # noqa: ARG002
        return None

    def __call__(self, recipe: TrainingRecipe, items: Sequence[CurriculumItem]) -> Any:
        raise CampaignRunRefusal(
            "the planning cycle was asked to execute a recipe; planning never "
            "starts compute"
        )


def plan_campaign(manifest: CampaignManifest) -> CampaignPlan:
    """Plan the declared campaign through the production cycle, no compute.

    This is the same assembly :func:`run_campaign` executes -- one curriculum
    plan, one recipe proposal, one recipe-selection rule -- so the ids printed
    here are the ids a run will honor.
    """
    assert_every_field_enforced()
    require_declared_inputs(manifest, phase="plan")
    root = Path(manifest.state_root)
    cycle = _build_cycle(
        manifest,
        executor=_NullExecutor(),
        firewall=_NullExecutor.firewall,
        root=root,
    )
    items = cycle.plan_curriculum(_load_profile(manifest))
    recipes = cycle.plan_recipes(items)
    # The declared search is projected over the recipes this campaign will run
    # (the declared ids), never over the planner's whole grid: a search over
    # candidates nobody declared is not this campaign's search. While the ids
    # are still placeholders -- which is how an author learns them, and why
    # planning must work before they are written down -- it is projected over
    # the planner's own proposal instead, and the ``recipe_set`` check is what
    # refuses a declared id the planner never proposed. No *run* is planned
    # this way: ``run_campaign`` selects the declared recipes (raising on an
    # unproposed id) before it asks for this plan.
    by_id = {recipe.recipe_id: recipe for recipe in recipes}
    declared = tuple(
        by_id[recipe_id] for recipe_id in manifest.recipe_ids if recipe_id in by_id
    )
    try:
        search = search_plan_for(manifest, cycle=cycle, recipes=declared or recipes)
    except CandidateSearchRefusal as refusal:
        return CampaignPlan(items=items, recipes=recipes, search_refusal=str(refusal))
    return CampaignPlan(items=items, recipes=recipes, search=search)


def envelope_for(manifest: CampaignManifest) -> GrowthEnvelope:
    """The executor's envelope: the manifest's per-recipe ceilings, unchanged."""
    return GrowthEnvelope(
        device_gpu_hours_ceiling=manifest.budget.device_gpu_hours_ceiling_per_recipe,
        wall_gpu_hours_ceiling=manifest.budget.wall_gpu_hours_ceiling_per_recipe,
        # The composed project declares the same declared wall ceiling it is
        # settled against, so it can never ask for more than the campaign did.
        project_gpu_hour_budget=manifest.budget.wall_gpu_hours_ceiling_per_recipe,
    )


def _build_cycle(
    manifest: CampaignManifest,
    *,
    executor: TrainingFn,
    firewall: ContaminationFirewall,
    root: Path,
) -> GrowthCycle:
    budget = manifest.budget
    config = CycleConfig(
        cycle_id=manifest.cycle_id,
        parent_version=manifest.parent_version,
        candidate_version=manifest.resolved_candidate_version(),
        device_gpu_hours_ceiling=budget.device_gpu_hours_ceiling_campaign,
        target_benchmarks=manifest.target_benchmarks,
        protected_benchmarks=manifest.protected_benchmarks,
        broad_battery=manifest.broad_benchmarks,
        calibration_benchmarks=manifest.calibration_benchmarks,
        reliability_benchmarks=manifest.reliability_benchmarks,
        # The declared recipe set sizes the proposal: the planner proposes
        # exactly what the campaign plans to run, and an id it cannot propose
        # refuses below.
        recipe_count=len(manifest.recipe_ids),
        # The promotion gates bind from the declaration, not from who built
        # the cycle: a manifest that declares them gets the same objects, the
        # same enforcement, and the same load-time refusals a programmatic
        # construction gets. Absent (every manifest predating them) means no
        # gates — the historical behavior, unchanged.
        retention_profile=manifest.retention_profile,
        eval_tier_policy=manifest.eval_tier_policy,
    )
    return GrowthCycle(
        config,
        curriculum=CurriculumEngine(),
        planner=RecipePlanner(
            budget=_load_hardware_budget(manifest),
            max_device_gpu_hours=budget.device_gpu_hours_ceiling_per_recipe,
            max_wall_gpu_hours=budget.wall_gpu_hours_ceiling_per_recipe,
        ),
        failure_bank=FailureBank(),
        firewall=firewall,
        ledger=GenerationLedger(root / "ledger"),
        regression_memory=RegressionMemory(root / "ledger"),
        snapshots=SnapshotStore(root / "ledger"),
        train_fn=executor,
    )


def search_plan_for(
    manifest: CampaignManifest, *, cycle: Any, recipes: Sequence[TrainingRecipe]
) -> SearchPlan:
    """The declared search's own bound, projected through the planner's costs.

    One owner of the arithmetic: the plan command prints this, readiness admits
    it and the run executes it, all calling here. An undeclared search returns
    an undeclared plan -- no rounds, no spend, exactly the single pass every
    manifest predating the field runs.
    """
    declaration = manifest.candidate_search
    if not declaration.declared:
        return SearchPlan(declared=False, schedule=declaration.to_dict())
    return plan_search(
        declaration,
        recipes=recipes,
        project_cost=cycle.planner.project_cost,
        per_recipe_device_ceiling=manifest.budget.device_gpu_hours_ceiling_per_recipe,
        per_recipe_wall_ceiling=manifest.budget.wall_gpu_hours_ceiling_per_recipe,
        campaign_device_ceiling=manifest.budget.device_gpu_hours_ceiling_campaign,
        campaign_wall_ceiling=manifest.budget.wall_gpu_hours_ceiling_campaign,
    )


def _select_recipes(
    manifest: CampaignManifest, proposed: Sequence[TrainingRecipe]
) -> tuple[TrainingRecipe, ...]:
    by_id = {recipe.recipe_id: recipe for recipe in proposed}
    missing = [rid for rid in manifest.recipe_ids if rid not in by_id]
    if missing:
        raise CampaignRunRefusal(
            f"recipe_ids declares {missing}, which the planner did not propose "
            f"(proposed: {sorted(by_id)}); a recipe set is what runs, so the "
            "runner refuses rather than substituting a different recipe"
        )
    return tuple(by_id[rid] for rid in manifest.recipe_ids)
