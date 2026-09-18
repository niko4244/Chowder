"""Campaign runner: a preregistered manifest drives one real generation cycle.

The manifest is the preregistration -- parent identity, promotion sets, recipe
set, budgets, stopping rules, policy version and the input paths a run reads.
This module is the only thing that turns that declaration into a cycle, and it
does so by *composing* the pieces that already own each decision:

* ``training_binding.SubprocessTrainingFn`` owns execution **and admission**:
  the manifest's per-recipe ceilings become its envelope, and the runner asks
  the executor whether a recipe may run rather than comparing costs itself.
* ``cycle.GrowthCycle`` owns sequencing, curriculum planning, recipe proposals,
  candidate selection and the promotion phase.
* ``promotion.evaluate_promotion`` owns the verdict.
* ``compute_cost.settle_cost`` / ``campaign.settle_campaign`` own settlement.
* ``metric_binding.MetricBinder`` owns turning measured rows into declared
  scales, and refuses rows whose provenance does not belong to the role.

Nothing here re-implements any of those, and nothing here invents an input: a
phase that needs a path the manifest did not declare refuses with the field
named, because a default would be a policy nobody preregistered.

Every key a manifest may declare is listed in :data:`FIELD_ENFORCEMENT` with
the behavior it drives. ``assert_every_field_enforced`` fails if the schema and
that table ever diverge, so a new field cannot land as decoration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from chowder.evals.result import BenchmarkRun, EvalReport

from .campaign import (
    PROMOTION_POLICY_VERSION,
    CampaignManifest,
    CampaignManifestError,
    settle_campaign,
    settle_campaign_projection,
    stops_on_admission_refusal,
    stops_on_campaign_overrun,
)
from .capability import CapabilityProfile
from .compute_cost import ComputeCost, CycleCostLedger
from .contamination import ContaminationFirewall
from .curriculum import CurriculumEngine, CurriculumItem
from .cycle import CycleConfig, CycleOutcome, GrowthCycle, TrainingFn
from .data_registry import DataRegistry, DataSource, admit
from .failure_bank import FailureBank
from .frontier_reference import SnapshotStore
from .lineage import GenerationLedger, RegressionMemory
from .metric_binding import MetricBinder, PromotionAssembly
from .recipe_planner import HardwareBudget, RecipePlanner, TrainingRecipe
from .training_binding import (
    STATUS_SUCCEEDED,
    GrowthEnvelope,
    SubprocessTrainingFn,
    default_runner,
    directory_digest,
)

#: What each declared manifest field actually drives. The runner refuses a
#: field it cannot act on, so this table is the proof that no declaration is
#: decorative; keeping it complete is enforced by
#: :func:`assert_every_field_enforced`.
FIELD_ENFORCEMENT: Mapping[str, str] = {
    "cycle_id": "names the run, its durable record and its ledger entry",
    "parent_version": "the generation the candidate is compared against",
    "candidate_version": "the generation the verdict is recorded under (derived when empty)",
    "parent_model_path": "hashed and compared with parent_model_digest before any compute",
    "parent_model_digest": "must equal the parent tree's digest, else the run refuses",
    "state_root": "attempts, registry, ledger, accounting and campaign-run.json live here",
    "target_benchmarks": "the cycle's target set: the improvement gate",
    "protected_benchmarks": "the cycle's protected set: the regression gate",
    "broad_benchmarks": "the cycle's broad battery: no material deterioration",
    "calibration_benchmarks": "the cycle's calibration set: hard gate when declared",
    "reliability_benchmarks": "the cycle's reliability set: hard gate when declared",
    "budget": "per-recipe ceilings become the executor's admission envelope; the campaign ceilings refuse a plan that does not fit and settle the actual cost after compute",
    "recipe_ids": "the recipe set: the planner proposes this many, and an unproposed id refuses",
    "candidate_selection_policy": "the selection order passed to select_candidate",
    "stopping_rules": "each rule must name a behavior that refuses: admission refusal ends the campaign before compute, an overrun rule stops once a ceiling is breached, the contamination rule is the firewall's own refusal, and the threshold rule is the code's frozen constants; any rule outside campaign.STOPPING_RULE_ENFORCEMENT refuses at load",
    "promotion_policy_version": "must be the policy this code implements",
    "contamination_manifest_path": "loaded into the MetricBinder; a declared-but-missing file refuses",
    "project_template_path": "the project template the executor composes each attempt from",
    "training_material_path": "curriculum item -> source and text material; a missing item refuses",
    "data_registry_path": "the admitted data sources the executor may draw on",
    "hardware_budget_path": "the measured local budget the recipe planner projects against",
    "parent_profile_path": "the parent capability profile the curriculum is planned from",
    "parent_eval_report_path": "the parent side of adjudication",
    "candidate_eval_report_path": "the candidate side of adjudication",
    "notes": "documentation only: it drives no behavior and gates nothing",
}

#: The one declared field that deliberately drives nothing.
NON_BEHAVIORAL_FIELDS = frozenset({"notes"})


class CampaignRunRefusal(RuntimeError):
    """The campaign could not be executed as declared.

    Raised before or around compute for a missing input, an unverifiable
    identity, or a declaration the runner cannot honor. A refusal is never
    converted into a default.
    """


@dataclass(frozen=True)
class CampaignPlan:
    """What the declared inputs plan to run, before compute or admission.

    ``recipes`` are the planner's own proposals, in the order the declared
    ``recipe_ids`` select them: ``chowder growth campaign plan`` prints these
    ids so a preregistration can name the recipes it will actually run, and
    the runner refuses a declared id the planner did not propose.
    """

    items: tuple[CurriculumItem, ...]
    recipes: tuple[TrainingRecipe, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "curriculum_items": [item.item_id for item in self.items],
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
    root = Path(manifest.state_root)
    cycle = _build_cycle(
        manifest,
        executor=_NullExecutor(),
        firewall=_NullExecutor.firewall,
        root=root,
    )
    items = cycle.plan_curriculum(_load_profile(manifest))
    return CampaignPlan(items=items, recipes=cycle.plan_recipes(items))


@dataclass(frozen=True)
class CampaignRun:
    """The durable outcome of one manifest-driven campaign run."""

    cycle_id: str
    parent_version: str
    candidate_version: str
    verdict: str  # PROMOTED / REJECTED / INCONCLUSIVE / TAINTED
    phases: tuple[Mapping[str, Any], ...]
    admission: tuple[Mapping[str, Any], ...]
    cost: Mapping[str, Any]
    settlement: Mapping[str, Any]
    ceiling_enforcement: Mapping[str, str]
    promotion: Mapping[str, Any] | None
    record_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "parent_version": self.parent_version,
            "candidate_version": self.candidate_version,
            "verdict": self.verdict,
            "phases": [dict(phase) for phase in self.phases],
            "admission": [dict(entry) for entry in self.admission],
            "cost": dict(self.cost),
            "settlement": dict(self.settlement),
            "ceiling_enforcement": dict(self.ceiling_enforcement),
            "promotion": self.promotion,
            "record_path": self.record_path,
        }


def assert_every_field_enforced(manifest_cls: type = CampaignManifest) -> None:
    """The schema and the enforcement table must describe the same fields."""
    schema = set(getattr(manifest_cls, "__dataclass_fields__"))
    declared = set(FIELD_ENFORCEMENT)
    if schema != declared:
        missing = sorted(schema - declared)
        extra = sorted(declared - schema)
        raise CampaignManifestError(
            "campaign manifest schema and FIELD_ENFORCEMENT disagree "
            f"(unlisted fields: {missing}; unknown fields: {extra}); a declared "
            "field the runner cannot act on would be silently ignored"
        )
    undocumented = sorted(set(FIELD_ENFORCEMENT) - NON_BEHAVIORAL_FIELDS)
    if not undocumented:
        raise CampaignManifestError("FIELD_ENFORCEMENT lists no behavioral fields")


def run_campaign(
    manifest: CampaignManifest,
    *,
    train_fn: Any = None,
    state_root: str | Path | None = None,
) -> CampaignRun:
    """Execute the declared campaign and return its mechanical outcome.

    ``train_fn`` is the execution seam (production: a ``SubprocessTrainingFn``,
    which the runner builds when one is not supplied). It must expose
    ``admit(recipe)``: admission is *its* decision, and a TrainingFn that
    cannot answer for projected cost is refused rather than bypassed.
    """
    assert_every_field_enforced()
    root = Path(state_root or manifest.state_root)
    root.mkdir(parents=True, exist_ok=True)
    phases: list[Mapping[str, Any]] = []

    _verify_parent_identity(manifest)
    phases.append(
        {
            "phase": "identity",
            "verdict": "ok",
            "detail": f"parent tree digest matches {manifest.parent_model_digest[:12]}",
        }
    )

    binder = _load_binder(manifest)
    phases.append(
        {
            "phase": "contamination",
            "verdict": "ok",
            "detail": (
                f"binder loaded from {manifest.contamination_manifest_path}"
                if manifest.contamination_manifest_path
                else "no contamination manifest declared: every row binds UNKNOWN"
            ),
        }
    )

    executor = train_fn if train_fn is not None else build_executor(manifest, state_root=root)
    if not hasattr(executor, "admit"):
        raise CampaignRunRefusal(
            "the training executor exposes no admission seam (`admit(recipe)`), "
            "so projected cost could not control admission; the campaign refuses "
            "rather than running under an unenforced budget"
        )
    firewall = getattr(executor, "firewall", ContaminationFirewall())
    cycle = _build_cycle(manifest, executor=executor, firewall=firewall, root=root)
    candidate_version = manifest.resolved_candidate_version()

    # Plan on the cycle that will execute, not a second assembly of it: one
    # owner of "what runs", so the ids planned are the ids run.
    items = cycle.plan_curriculum(_load_profile(manifest))
    plan = CampaignPlan(items=items, recipes=cycle.plan_recipes(items))
    if not plan.items:
        raise CampaignRunRefusal(
            "the declared parent profile produced no curriculum items, so there "
            "is nothing this campaign may train on"
        )
    recipes = _select_recipes(manifest, plan.recipes)
    phases.append(
        {
            "phase": "plan",
            "verdict": "ok",
            "detail": (
                f"{len(plan.items)} curriculum items -> {len(recipes)} declared recipes "
                f"of {len(manifest.recipe_ids)}"
            ),
        }
    )

    admission: list[Mapping[str, Any]] = []
    for recipe in recipes:
        refusal = executor.admit(recipe)
        entry: dict[str, Any] = {
            "recipe_id": recipe.recipe_id,
            "projected_device_gpu_hours": float(recipe.projected_device_gpu_hours),
            "projected_wall_gpu_hours": float(recipe.projected_wall_gpu_hours),
            "admitted": refusal is None,
        }
        if refusal is not None:
            entry["refused_by"], entry["refusal_reason"] = refusal
        admission.append(entry)
    refused = [entry for entry in admission if not entry["admitted"]]
    detail = "; ".join(
        f"{entry['recipe_id']}: {entry['refusal_reason']}" for entry in refused
    )
    if refused and stops_on_admission_refusal(manifest.stopping_rules):
        phases.append({"phase": "admission", "verdict": "refused", "detail": detail})
        return _refuse(manifest, root, candidate_version, phases, admission)
    admitted_ids = {entry["recipe_id"] for entry in admission if entry["admitted"]}
    recipes = tuple(recipe for recipe in recipes if recipe.recipe_id in admitted_ids)
    if not recipes:
        phases.append({"phase": "admission", "verdict": "refused", "detail": detail})
        return _refuse(manifest, root, candidate_version, phases, admission)
    phases.append(
        {
            "phase": "admission",
            "verdict": "ok" if not refused else "partial",
            "detail": (
                f"{len(recipes)} recipe(s) admitted by the executor before compute"
                + (f"; skipped: {detail}" if refused else "")
            ),
        }
    )

    # Campaign admission: the per-recipe ceilings bound what the executor may
    # start, and these bound what the campaign as a whole plans to spend. A
    # campaign whose own plan does not fit its declared envelope refuses here,
    # before compute, rather than discovering it only when the actuals land.
    projected = ComputeCost(
        device_gpu_hours=sum(recipe.projected_device_gpu_hours for recipe in recipes),
        wall_gpu_hours=sum(recipe.projected_wall_gpu_hours for recipe in recipes),
        source=f"campaign projection ({len(recipes)} admitted recipes)",
    )
    projection = settle_campaign_projection(manifest, projected=projected)
    if not projection.compliant:
        phases.append(
            {
                "phase": "campaign_projection",
                "verdict": "refused",
                "detail": "; ".join(projection.failure_reasons),
            }
        )
        return _refuse(manifest, root, candidate_version, phases, admission)
    phases.append(
        {
            "phase": "campaign_projection",
            "verdict": "ok",
            "detail": (
                f"planned {projected.device_gpu_hours:.6f} device / "
                f"{projected.wall_gpu_hours:.6f} wall within the declared "
                "campaign ceilings"
            ),
        }
    )

    ledger = CycleCostLedger(cycle_id=manifest.cycle_id)
    results: list[Mapping[str, Any]] = []
    stopped_by: str | None = None
    for recipe in recipes:
        evidence = dict(executor(recipe, plan.items))
        evidence["recipe_id"] = recipe.recipe_id
        results.append(evidence)
        cost = _attempt_cost(evidence)
        ledger.add(
            f"{recipe.recipe_id} attempt",
            "failed_attempt" if evidence.get("status") != STATUS_SUCCEEDED else "training",
            cost,
            recipe_id=recipe.recipe_id,
            notes=f"status={evidence.get('status')}",
        )
        if stops_on_campaign_overrun(manifest.stopping_rules):
            running = settle_campaign(manifest, total=ledger.total())
            if not running.compliant:
                stopped_by = "campaign ceiling reached before the remaining recipes"
                break
    accounting_path = root / "cycle_compute_accounting.json"
    accounting_digest = ledger.write(accounting_path)
    total = ledger.total()
    settlement = settle_campaign(manifest, total=total)
    ceiling_enforcement = _ceiling_enforcement(manifest, total)
    phases.append(
        {
            "phase": "settlement",
            "verdict": "ok" if settlement.compliant else "violated",
            "detail": "; ".join(settlement.failure_reasons)
            or f"actual cost within the declared campaign ceilings (digest {accounting_digest[:12]})",
        }
    )
    if stopped_by:
        phases.append(
            {"phase": "stopping", "verdict": "stopped", "detail": stopped_by}
        )

    selected = cycle.select_candidate(
        results, order=manifest.candidate_selection_policy
    )
    phases.append(
        {
            "phase": "selection",
            "verdict": "selected" if selected else "none",
            "detail": (
                f"{selected.get('recipe_id')} by {manifest.candidate_selection_policy}"
                if selected
                else f"no successful candidate under {manifest.candidate_selection_policy}"
            ),
        }
    )

    assembly = _adjudicate(manifest, cycle=cycle, binder=binder, total=total)
    decision = assembly.decision
    phases.append(
        {
            "phase": "promotion",
            "verdict": decision.verdict,
            "detail": "; ".join(decision.reasons) or "predeclared rule",
        }
    )

    verdict = decision.verdict
    if not settlement.compliant:
        # A campaign that blew its own declared envelope does not promote, no
        # matter how the candidate measured: the resource gate is hard, and it
        # is authoritative over the lineage record too. ``finalize`` records a
        # generation only for a PROMOTED decision, so the decision handed to
        # it is the vetoed one -- otherwise a rule verdict computed before
        # settlement could write a promoted generation the campaign refused.
        verdict = "REJECTED"
        phases.append(
            {
                "phase": "resource_veto",
                "verdict": "REJECTED",
                "detail": "; ".join(settlement.failure_reasons),
            }
        )

    outcome = _finalize(
        manifest,
        cycle=cycle,
        decision=_with_resource_veto(decision, settlement),
        selected=selected,
        root=root,
        evaluation_report_ref=manifest.candidate_eval_report_path,
    )
    run = CampaignRun(
        cycle_id=manifest.cycle_id,
        parent_version=manifest.parent_version,
        candidate_version=candidate_version,
        verdict=verdict,
        phases=tuple(phases),
        admission=tuple(admission),
        cost={
            "device_gpu_hours": total.device_gpu_hours,
            "wall_gpu_hours": total.wall_gpu_hours,
            "device_measured": total.device_measured,
            "accounting_path": str(accounting_path),
            "accounting_digest": accounting_digest,
        },
        settlement=settlement.to_dict(),
        ceiling_enforcement=ceiling_enforcement,
        promotion=assembly.to_dict(),
        record_path="",
    )
    return _write_record(run, root, outcome=outcome)


# --------------------------------------------------------------------------
# declared inputs
# --------------------------------------------------------------------------


def _require_path(value: str, field: str, *, purpose: str) -> Path:
    if not value:
        raise CampaignRunRefusal(
            f"the manifest declares no {field}, so {purpose} cannot happen; "
            "declare the path rather than letting the runner guess one"
        )
    path = Path(value)
    if not path.exists():
        raise CampaignRunRefusal(f"{field} {value!r} does not exist ({purpose})")
    return path


def _verify_parent_identity(manifest: CampaignManifest) -> None:
    if not manifest.parent_model_path:
        raise CampaignRunRefusal(
            "the manifest declares no parent_model_path; a candidate without a "
            "hashed parent is not a generation"
        )
    parent = Path(manifest.parent_model_path)
    if not parent.exists():
        raise CampaignRunRefusal(
            f"parent_model_path {manifest.parent_model_path!r} does not exist"
        )
    digest, _entries = directory_digest(parent)
    if digest != manifest.parent_model_digest:
        raise CampaignRunRefusal(
            f"parent_model_digest {manifest.parent_model_digest[:12]} does not match "
            f"the parent tree's digest {digest[:12]}; the campaign would train from a "
            "model other than the one it preregistered"
        )


def _load_binder(manifest: CampaignManifest) -> MetricBinder:
    from .catalog import default_registry

    registry = default_registry()
    if not manifest.contamination_manifest_path:
        return MetricBinder(registry)
    path = _require_path(
        manifest.contamination_manifest_path,
        "contamination_manifest_path",
        purpose="rows cannot be bound against an unchecked firewall",
    )
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise CampaignRunRefusal(
            f"contamination manifest {path} is not a JSON object"
        )
    return MetricBinder.from_manifest(registry, document)


def _load_profile(manifest: CampaignManifest) -> CapabilityProfile:
    path = _require_path(
        manifest.parent_profile_path,
        "parent_profile_path",
        purpose="a curriculum cannot be planned from nothing",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise CampaignRunRefusal(f"parent profile {path} is not a JSON object")
    return CapabilityProfile.from_dict(dict(payload))


def _load_material(manifest: CampaignManifest) -> tuple[dict[str, str], dict[str, list[str]]]:
    path = _require_path(
        manifest.training_material_path,
        "training_material_path",
        purpose="the executor writes the corpus this run trains on",
    )
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise CampaignRunRefusal(f"training material {path} is not a JSON object")
    sources = document.get("sources")
    material = document.get("material")
    if not isinstance(sources, Mapping) or not isinstance(material, Mapping):
        raise CampaignRunRefusal(
            f"training material {path} must declare 'sources' (item -> source id) "
            "and 'material' (item -> text lines)"
        )
    return (
        {str(key): str(value) for key, value in sources.items()},
        {str(key): [str(line) for line in value] for key, value in material.items()},
    )


def _load_data_registry(manifest: CampaignManifest) -> DataRegistry:
    path = _require_path(
        manifest.data_registry_path,
        "data_registry_path",
        purpose="nothing may train on an unadmitted source",
    )
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping) or not isinstance(document.get("sources"), list):
        raise CampaignRunRefusal(
            f"data registry {path} must be an object with a 'sources' list"
        )
    registry = DataRegistry()
    for entry in document["sources"]:
        if not isinstance(entry, Mapping):
            raise CampaignRunRefusal(f"data registry {path} holds a non-object entry")
        source = DataSource(**dict(entry))
        if source.inclusion_decision != "included":
            raise CampaignRunRefusal(
                f"data source {source.source_id!r} is declared "
                f"{source.inclusion_decision!r}; only included sources may train"
            )
        registry.register(
            admit(
                source,
                decision="included",
                reason="declared included in the campaign's data registry",
            )
        )
    return registry


def _load_hardware_budget(manifest: CampaignManifest) -> HardwareBudget:
    path = _require_path(
        manifest.hardware_budget_path,
        "hardware_budget_path",
        purpose="recipes are projected against measured hardware, never guesses",
    )
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise CampaignRunRefusal(f"hardware budget {path} is not a JSON object")
    steps = document.get("measured_step_seconds_at_seq")
    if not isinstance(steps, Mapping):
        raise CampaignRunRefusal(
            f"hardware budget {path} must declare 'measured_step_seconds_at_seq'"
        )
    return HardwareBudget(
        gpu_name=str(document.get("gpu_name", "")),
        vram_gb=float(document.get("vram_gb", 0.0)),
        measured_step_seconds_at_seq={
            int(key): float(value) for key, value in steps.items()
        },
        measured_load_seconds=float(document.get("measured_load_seconds", 0.0)),
        wall_multiplier=float(document.get("wall_multiplier", 3.5)),
    )


def envelope_for(manifest: CampaignManifest) -> GrowthEnvelope:
    """The executor's envelope: the manifest's per-recipe ceilings, unchanged."""
    return GrowthEnvelope(
        device_gpu_hours_ceiling=manifest.budget.device_gpu_hours_ceiling_per_recipe,
        wall_gpu_hours_ceiling=manifest.budget.wall_gpu_hours_ceiling_per_recipe,
        # The composed project declares the same declared wall ceiling it is
        # settled against, so it can never ask for more than the campaign did.
        project_gpu_hour_budget=manifest.budget.wall_gpu_hours_ceiling_per_recipe,
    )


def build_executor(
    manifest: CampaignManifest,
    *,
    state_root: str | Path | None = None,
    runner: Any = None,
) -> SubprocessTrainingFn:
    """Build the production executor from the manifest's declared inputs."""
    root = Path(state_root or manifest.state_root)
    template_path = _require_path(
        manifest.project_template_path,
        "project_template_path",
        purpose="the executor has no project to compose",
    )
    template = json.loads(template_path.read_text(encoding="utf-8"))
    if not isinstance(template, Mapping):
        raise CampaignRunRefusal(f"project template {template_path} is not a JSON object")
    sources, material = _load_material(manifest)
    registry = _load_data_registry(manifest)
    return SubprocessTrainingFn(
        run_root=root / "attempts",
        project_template=template,
        envelope=envelope_for(manifest),
        registry=registry,
        firewall=ContaminationFirewall(),
        sources=sources,
        material=material,
        runner=runner or default_runner,
    )


# --------------------------------------------------------------------------
# cycle assembly and adjudication
# --------------------------------------------------------------------------


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


def _attempt_cost(evidence: Mapping[str, Any]) -> ComputeCost:
    recorded = evidence.get("actual_cost")
    if isinstance(recorded, Mapping):
        return ComputeCost.from_dict(recorded)
    measured = evidence.get("measured_gpu_hours")
    return ComputeCost.from_wall_only(
        float(measured or 0.0), source=f"attempt:{evidence.get('attempt', 'unknown')}"
    )


def _ceiling_enforcement(
    manifest: CampaignManifest, total: ComputeCost
) -> dict[str, str]:
    """Which unit each declared ceiling was actually settled in.

    The device ceiling settles only when the budget declares measured device
    time *and* the campaign's own cost says the figure is a measurement;
    otherwise it is an admission constraint on the projected plan, and the
    record says so instead of implying a certified number.
    """
    settled = manifest.budget.device_time_measured and total.device_measured
    return {
        "device": "settlement:measured" if settled else "admission:projected_plan",
        "wall": "settlement:measured",
        "project": "settlement:measured_wall",
    }


def _adjudicate(
    manifest: CampaignManifest,
    *,
    cycle: GrowthCycle,
    binder: MetricBinder,
    total: ComputeCost,
) -> PromotionAssembly:
    """Bind the declared evaluations and let the predeclared rule decide.

    Absent reports are not an error: the rule then sees no candidate-measured
    rows and answers INCONCLUSIVE, which is the same refusal to certify it
    applies to any other missing evidence.
    """
    parent_runs: tuple[BenchmarkRun, ...] = ()
    if manifest.parent_eval_report_path:
        parent_runs = _runs_from_report(
            manifest.parent_eval_report_path, "parent_eval_report_path"
        )
    candidate_runs: tuple[BenchmarkRun, ...] = ()
    if manifest.candidate_eval_report_path:
        candidate_runs = _runs_from_report(
            manifest.candidate_eval_report_path, "candidate_eval_report_path"
        )
    budget = manifest.budget
    device_settleable = budget.device_time_measured
    return cycle.decide_promotion_from_runs(
        binder,
        candidate_runs=candidate_runs,
        parent_runs=parent_runs,
        # An unmeasured device figure is not a cost reading: reporting one as
        # a measured zero would be the fabrication the settlement rules exist
        # to refuse, so it is reported only when it was measured.
        device_gpu_hours=total.device_gpu_hours if device_settleable else 0.0,
        actual_wall_gpu_hours=total.wall_gpu_hours,
        wall_gpu_hours_ceiling=budget.wall_gpu_hours_ceiling_campaign,
    )


def _runs_from_report(path_value: str, field: str) -> tuple[BenchmarkRun, ...]:
    path = _require_path(path_value, field, purpose="a declared report must exist")
    report = EvalReport.load(path)
    return tuple(report.runs)


def _with_resource_veto(decision: Any, settlement: Any) -> Any:
    """The decision the lineage records, after the campaign's resource gate.

    A compliant settlement leaves the rule's decision untouched. A violated
    one turns it into a REJECTED decision carrying the settlement's machine
    identifiers, so no code path can read a promoted generation out of a
    campaign that failed its own frozen envelope.
    """
    if settlement.compliant:
        return decision
    return replace(
        decision,
        verdict="REJECTED",
        reasons=tuple(decision.reasons) + tuple(settlement.failure_reasons),
    )


def _finalize(
    manifest: CampaignManifest,
    *,
    cycle: GrowthCycle,
    decision: Any,
    selected: Mapping[str, Any] | None,
    root: Path,
    evaluation_report_ref: str,
) -> CycleOutcome:
    return cycle.finalize(
        decision,
        base_model={
            "path": manifest.parent_model_path,
            "sha256": manifest.parent_model_digest,
            "version": manifest.parent_version,
        },
        dataset_manifest_ref=selected.get("evidence_path", "") if selected else "",
        curriculum_manifest_ref=str(root / "cycle_compute_accounting.json"),
        recipe={"recipe_id": selected.get("recipe_id")} if selected else {},
        training_evidence_ref=selected.get("evidence_path", "") if selected else "",
        evaluation_report_ref=evaluation_report_ref,
        notes=f"campaign {manifest.cycle_id} (policy {PROMOTION_POLICY_VERSION})",
    )


def _refuse(
    manifest: CampaignManifest,
    root: Path,
    candidate_version: str,
    phases: list[Mapping[str, Any]],
    admission: list[Mapping[str, Any]],
) -> CampaignRun:
    run = CampaignRun(
        cycle_id=manifest.cycle_id,
        parent_version=manifest.parent_version,
        candidate_version=candidate_version,
        verdict="REFUSED",
        phases=tuple(phases),
        admission=tuple(admission),
        cost={},
        settlement={},
        ceiling_enforcement={},
        promotion=None,
        record_path="",
    )
    return _write_record(run, root, outcome=None)


def _write_record(
    run: CampaignRun, root: Path, *, outcome: CycleOutcome | None
) -> CampaignRun:
    document = run.to_dict()
    if outcome is not None:
        document["cycle_outcome"] = outcome.to_dict()
    path = root / "campaign-run.json"
    path.write_text(json.dumps(document, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return CampaignRun(
        cycle_id=run.cycle_id,
        parent_version=run.parent_version,
        candidate_version=run.candidate_version,
        verdict=run.verdict,
        phases=run.phases,
        admission=run.admission,
        cost=run.cost,
        settlement=run.settlement,
        ceiling_enforcement=run.ceiling_enforcement,
        promotion=run.promotion,
        record_path=str(path),
    )
