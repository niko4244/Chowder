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

The same run root is also the frozen judge's input: the runner materialises
:data:`CERTIFICATION_EVIDENCE` -- the three provenance-bound arms, the
winner's identity and the contamination evidence the run bound -- from the
run's own measurements, so the verdict a campaign records and the verdict the
judge certifies are read from one directory.

The candidate arm is the one measurement the run *produces*: after selection the
runner asks the declared evaluator
(:mod:`chowder.growth.candidate_evaluation`) to measure the artifact it
selected, binds the returned report to that artifact's digest, and adjudicates
and certifies on those rows. A campaign cannot hand the run a pre-existing
candidate report: the manifest key that used to declare one is retired with a
named refusal, because whoever prepared it could not have measured the adapter
this run just made.

Every key a manifest may declare is listed in :data:`FIELD_ENFORCEMENT` with
the behavior it drives. ``assert_every_field_enforced`` fails if the schema and
that table ever diverge, so a new field cannot land as decoration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from chowder.evals.result import BenchmarkRun, EvalReport

from .candidate_evaluation import (
    CANDIDATE_EVALUATION_NOT_PRODUCED,
    CandidateEvaluation,
    CandidateEvaluationRefusal,
    CandidateEvaluator,
    EvaluationRequest,
    coerce_evaluation,
    evaluation_detail,
    validate_candidate_report,
    write_candidate_evaluation,
)

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
from .certification import (
    FAIL,
    PASS,
    UNKNOWN,
    Certification,
    CertificationRow,
    ProtectionPolicy,
    certify_run_root,
)
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
    "base_model_path": "hashed and compared with base_model_digest before any compute",
    "base_model_digest": "must equal the dense base tree's digest, else the run refuses",
    "parent_adapter_path": "hashed and compared with parent_adapter_digest when declared",
    "parent_adapter_digest": "must equal the parent adapter tree's digest, else the run refuses",
    "state_root": "attempts, registry, ledger, accounting, the judged evidence set and campaign-run.json live here",
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
    "parent_eval_report_path": "the parent side of adjudication, and the parent arm of the judged evidence set",
    "baseline_eval_report_path": "the trusted-ancestor (gen0) arm of the judged evidence set: branch protection is judged against it, never against an unresolved parent",
    "protection": "the declared branch-protection policy (trusted ancestor version + slice regression tolerance) the certification gate applies before any lineage record is written",
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


#: The production candidate-evaluator factory. ``None`` in this build: no
#: instrument that measures the selected adapter under the declared protocol is
#: wired yet, so a run that is not given a seam refuses rather than adjudicating
#: on evidence it did not produce (see :func:`build_evaluator`). A build supplies
#: one the same way it supplies a process runner, so the campaign's own path to
#: measured candidate evidence is one function, not a second engine.
default_evaluator_factory: Callable[..., CandidateEvaluator | None] | None = None


#: The inputs each phase requires before it can run, in the order the run needs
#: them. A phase that finds one undeclared refuses with the whole list, so an
#: operator sees every missing input at once instead of one per attempt.
DECLARED_INPUT_REQUIREMENTS: Mapping[str, tuple[tuple[str, str], ...]] = {
    "plan": (
        ("parent_profile_path", "a curriculum cannot be planned from nothing"),
        ("hardware_budget_path", "recipes are projected against measured hardware, never guesses"),
    ),
    "run": (
        ("project_template_path", "the executor has no project to compose"),
        ("training_material_path", "the executor writes the corpus this run trains on"),
        ("data_registry_path", "nothing may train on an unadmitted source"),
        ("hardware_budget_path", "recipes are projected against measured hardware, never guesses"),
        ("parent_profile_path", "a curriculum cannot be planned from nothing"),
        (
            "contamination_manifest_path",
            "rows cannot be bound against an unchecked firewall, so every row would bind UNKNOWN",
        ),
    ),
}


def undeclared_inputs(manifest: CampaignManifest, *, phase: str) -> tuple[str, ...]:
    """Declared-input fields the phase needs that the manifest leaves empty."""
    return tuple(
        field
        for field, _why in DECLARED_INPUT_REQUIREMENTS[phase]
        if not str(getattr(manifest, field)).strip()
    )


def require_declared_inputs(manifest: CampaignManifest, *, phase: str) -> None:
    """Refuse with *every* undeclared input of the phase, not just the first.

    A pre-compute checklist is the honest form of this refusal: the operator
    learns what the campaign still has to declare before any compute starts.
    """
    missing = undeclared_inputs(manifest, phase=phase)
    if not missing:
        return
    reasons = "; ".join(
        f"{field} ({why})"
        for field, why in DECLARED_INPUT_REQUIREMENTS[phase]
        if field in missing
    )
    raise CampaignRunRefusal(
        f"the campaign leaves {len(missing)} declared input(s) the {phase} phase "
        f"requires undeclared: {reasons}; a campaign runs from fully declared "
        "inputs, so name them rather than letting the runner substitute a default"
    )


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
    require_declared_inputs(manifest, phase="plan")
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
    #: The branch-protection certification this run applied before its lineage
    #: record: the trusted-ancestor gate, the protected-slice protocol and the
    #: digest binding of every arm to the bytes it measured.
    certification: Mapping[str, Any]
    #: Which artifact the campaign selected, with the digest of the bytes it
    #: contains: the record names the model this run would promote, so a reader
    #: (and the evaluation that must be bound to it) never has to guess.
    selection: Mapping[str, Any]
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
            "certification": dict(self.certification),
            "selection": dict(self.selection),
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
    eval_fn: CandidateEvaluator | None = None,
    state_root: str | Path | None = None,
) -> CampaignRun:
    """Execute the declared campaign and return its mechanical outcome.

    ``train_fn`` is the execution seam (production: a ``SubprocessTrainingFn``,
    which the runner builds when one is not supplied). It must expose
    ``admit(recipe)``: admission is *its* decision, and a TrainingFn that
    cannot answer for projected cost is refused rather than bypassed.

    ``eval_fn`` is the evaluation seam the candidate arm is produced through.
    It is asked to measure the artifact this run selected, and nothing else can
    supply the candidate side of promotion: a run without it refuses.
    """
    assert_every_field_enforced()
    require_declared_inputs(manifest, phase="run")
    root = Path(state_root or manifest.state_root)
    root.mkdir(parents=True, exist_ok=True)
    phases: list[Mapping[str, Any]] = []

    _verify_parent_identity(manifest)
    phases.append(
        {
            "phase": "identity",
            "verdict": "ok",
            "detail": _identity_detail(manifest),
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
    total = ledger.total()

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
    if not selected:
        # Nothing was produced to evaluate. The accounting is still written --
        # the compute was really spent -- and the run records its refusal.
        accounting_digest = ledger.write(accounting_path)
        phases.append(
            {
                "phase": "candidate_evaluation",
                "verdict": "refused",
                "detail": (
                    f"{CANDIDATE_EVALUATION_NOT_PRODUCED}: the run selected no "
                    "artifact with a measured digest, so there is nothing the "
                    "candidate arm could attest"
                ),
            }
        )
        return _refuse(
            manifest,
            root,
            candidate_version,
            phases,
            admission,
            cost={
                "device_gpu_hours": total.device_gpu_hours,
                "wall_gpu_hours": total.wall_gpu_hours,
                "accounting_path": str(accounting_path),
                "accounting_digest": accounting_digest,
            },
        )

    # The candidate arm is produced here, from the artifact this run selected,
    # and its measured cost is charged before settlement so evaluation spend
    # counts against the campaign's own ceilings like every other leg.
    request, evaluation = _evaluate_candidate(
        manifest, root=root, selected=selected, eval_fn=eval_fn
    )
    ledger.add(
        "candidate evaluation",
        "evaluation",
        evaluation.cost or ComputeCost.zero(source="candidate evaluation"),
        notes=f"{len(evaluation.report.runs)} candidate-measured rows",
    )
    phases.append(
        {
            "phase": "candidate_evaluation",
            "verdict": "measured",
            "detail": evaluation_detail(evaluation, request),
        }
    )

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

    written = write_certification_evidence(
        manifest, root=root, selected=selected, candidate=evaluation
    )
    phases.append(
        {
            "phase": "certification_evidence",
            "verdict": "ok",
            "detail": (
                "wrote " + ", ".join(written)
                if written
                else "the manifest declares no evaluation evidence to materialise"
            ),
        }
    )

    # Certification decides *before* the lineage record exists: the trusted-ancestor
    # and protected-slice gates are the production mechanism the frozen judge also
    # runs, applied to the evidence this run just wrote, so a generation cannot be
    # recorded as promoted and only afterwards found unprotectable.
    certification = certify_before_lineage(manifest, root=root, selected=selected)
    phases.append(
        {
            "phase": "certification",
            "verdict": certification.status,
            "detail": "; ".join(certification.reasons)
            or f"{len(certification.rows)} protection requirements held",
        }
    )

    assembly = _adjudicate(
        manifest,
        cycle=cycle,
        binder=binder,
        total=total,
        candidate_runs=tuple(evaluation.report.runs),
    )
    decision = assembly.decision
    phases.append(
        {
            "phase": "promotion",
            "verdict": decision.verdict,
            "detail": "; ".join(decision.reasons) or "predeclared rule",
        }
    )

    verdict = decision.verdict
    if certification.status != PASS and (certification.status == FAIL or verdict == "PROMOTED"):
        # A refused certification vetoes promotion exactly like a violated
        # resource envelope: the rule's verdict is preserved for the record, and
        # the decision handed to ``finalize`` is the vetoed one, so no ledger row
        # claims a generation the certification refused. A hard breach is a hard
        # failure of the run even when the rule would have said INCONCLUSIVE.
        verdict = "REJECTED" if certification.status == FAIL else "INCONCLUSIVE"
        phases.append(
            {
                "phase": "certification_veto",
                "verdict": verdict,
                "detail": "; ".join(certification.reasons),
            }
        )
        decision = replace(
            decision,
            verdict=verdict,
            reasons=tuple(decision.reasons) + tuple(certification.reasons),
        )
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
        # The report this run produced, at the path the judge reads: the lineage
        # names the measurement that decided it, not an input it was handed.
        evaluation_report_ref=str(root / CERTIFICATION_EVIDENCE["candidate"]),
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
        certification=certification.to_dict(),
        selection=dict(selected) if selected else {},
        promotion=assembly.to_dict(),
        record_path="",
    )
    return _write_record(run, root, outcome=outcome)


# --------------------------------------------------------------------------
# certification evidence: the run writes what the frozen judge reads
# --------------------------------------------------------------------------

#: The artifacts ``docs/gen2/judge_gen2.py`` reads from a run root. The runner
#: writes exactly these, so one run root is both the run's output and the
#: judge's input; ``tests/test_growth_certification_coupling.py`` compares this
#: mapping against the judge's own literals, so the two cannot drift into a
#: boundary that certifies nothing the pipeline produces.
CERTIFICATION_EVIDENCE: Mapping[str, str] = {
    "candidate": "candidate_evaluation.json",
    "parent": "parent_evaluation.json",
    "ancestor": "baseline_evaluation.json",
    "selection": "chosen_candidate.json",
    "contamination": "gen2_contamination_manifest.json",
}

#: Which declared manifest field supplies each arm the run *reads*. The
#: candidate arm is absent here on purpose: it is produced by the run.
_EVIDENCE_ARM_SOURCES: Mapping[str, str] = {
    "parent": "parent_eval_report_path",
    "ancestor": "baseline_eval_report_path",
}


def write_certification_evidence(
    manifest: CampaignManifest,
    *,
    root: Path,
    selected: Mapping[str, Any] | None,
    candidate: CandidateEvaluation | None,
) -> tuple[str, ...]:
    """Materialise the judged artifacts from the run's own evidence.

    Every file here is derived from something the run actually has: the
    candidate arm this run measured over the artifact it selected, the declared
    arms it reads (copied verbatim, so a row's ``measurement_origin`` is the
    evaluator's declaration and not the runner's election), the contamination
    manifest the firewall bound, and the adapter artifact the winning attempt
    produced -- digested over its bytes.

    An input the manifest does not declare produces **no** file rather than a
    placeholder: the judge then reports that gate UNKNOWN, which is the honest
    answer. Nothing here can invent a measurement the run never took, and no
    gate is loosened by writing these: they are the same numbers the run
    already adjudicated with, at the paths the frozen judge reads.

    Every declared input is read and validated *before* the first file is
    written, so a refusal (a declared report that does not exist, or one whose
    rows cannot be parsed) never leaves a half-materialised evidence set for the
    judge to read.
    """
    arms: list[tuple[str, Path, EvalReport]] = []
    for arm, field_name in _EVIDENCE_ARM_SOURCES.items():
        declared = str(getattr(manifest, field_name))
        if not declared:
            continue
        source = _require_path(
            declared, field_name, purpose="the judged evidence set is built from it"
        )
        # An unreadable report must refuse here rather than reach the judge as an
        # arm whose rows cannot be parsed.
        arms.append((arm, source, EvalReport.load(source)))
    if candidate is not None:
        # Produced by this run, so it is written rather than read: nothing here
        # can substitute a report the campaign declared.
        arms.append(("candidate", None, candidate.report))
    artifacts = _measurement_artifacts(arms, root=root)
    contamination: Path | None = None
    if manifest.contamination_manifest_path:
        contamination = _require_path(
            manifest.contamination_manifest_path,
            "contamination_manifest_path",
            purpose="the run's contamination evidence is copied for the judge",
        )
        if not isinstance(json.loads(contamination.read_text(encoding="utf-8")), Mapping):
            raise CampaignRunRefusal(
                f"contamination manifest {contamination} is not a JSON object, so "
                "it cannot be the contamination evidence the judge reads"
            )

    written: list[str] = []
    for arm, source, _report in arms:
        destination = root / CERTIFICATION_EVIDENCE[arm]
        if source is None:
            write_candidate_evaluation(
                CandidateEvaluation(report=_report),
                root=root,
                name=CERTIFICATION_EVIDENCE[arm],
            )
        else:
            destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        written.append(destination.name)

    # The measurements those rows declare, so the run root carries the bytes
    # each row's digest is computed over.
    for origin, relative in artifacts:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(origin.read_bytes())
        written.append(str(relative))

    if contamination is not None:
        destination = root / CERTIFICATION_EVIDENCE["contamination"]
        destination.write_text(contamination.read_text(encoding="utf-8"), encoding="utf-8")
        written.append(destination.name)

    chosen = _chosen_candidate_document(selected)
    if chosen is not None:
        destination = root / CERTIFICATION_EVIDENCE["selection"]
        destination.write_text(
            json.dumps(chosen, indent=2, sort_keys=True), encoding="utf-8"
        )
        written.append(destination.name)
    return tuple(written)


def _measurement_artifacts(
    arms: Sequence[tuple[str, Path | None, EvalReport]],
    *,
    root: Path,
) -> list[tuple[Path, Path]]:
    """Every measurement artifact the arms' rows name, as (origin, relative) pairs.

    A row that names an artifact is bound to it: the judge recomputes the digest
    the row declares over those bytes, so a run root that lacks them carries a
    measurement nobody can verify. A reference that resolves to nothing refuses
    the run -- evidence is not assembled around a missing artifact -- and two arms
    that name the same relative path with different content refuse too, because
    one run root cannot be both.
    """
    artifacts: list[tuple[Path, Path]] = []
    seen: dict[Path, Path] = {}
    for arm, source, report in arms:
        for run in report.runs:
            reference = str(run.raw_artifact_ref or "")
            if not reference:
                continue
            relative = Path(reference)
            if relative.is_absolute():
                # The row names an absolute artifact; the judge hashes it there.
                continue
            if ".." in relative.parts:
                raise CampaignRunRefusal(
                    f"the {arm} arm's {run.benchmark_qualified_id} row names raw "
                    f"artifact {reference!r}, which points outside the run root; the "
                    "judged evidence set cannot follow it"
                )
            if source is None:
                # The candidate arm's relative refs are resolved against the run
                # root: the evaluator wrote its measurements into the run it was
                # asked to measure for, so there is nothing to carry.
                origin = root / relative
                if not origin.is_file():
                    raise CampaignRunRefusal(
                        f"the candidate arm's {run.benchmark_qualified_id} row names "
                        f"raw artifact {reference!r}, which the evaluation did not "
                        "write into the run root, so the measurement it declares "
                        "cannot be verified"
                    )
                previous = seen.get(relative)
                if previous is not None and previous.read_bytes() != origin.read_bytes():
                    raise CampaignRunRefusal(
                        f"two arms name {reference!r} with different content, so one "
                        "run root cannot carry both measurements as verifiable evidence"
                    )
                seen[relative] = origin
                continue
            origin = source.parent / relative
            if not origin.is_file():
                raise CampaignRunRefusal(
                    f"the {arm} arm's {run.benchmark_qualified_id} row names raw "
                    f"artifact {reference!r}, which does not exist next to {source}, "
                    "so the measurement it declares cannot be verified"
                )
            previous = seen.get(relative)
            if previous is not None and previous.read_bytes() != origin.read_bytes():
                raise CampaignRunRefusal(
                    f"two arms name {reference!r} with different content, so one run "
                    "root cannot carry both measurements as verifiable evidence"
                )
            seen[relative] = origin
            artifacts.append((origin, relative))
    return artifacts


def _chosen_candidate_document(
    selected: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """The winner's identity, or None when the run produced no artifact.

    The digest is the one the binding already measured over the artifact's
    bytes, recomputed here only when the attempt recorded none. A candidate
    with no artifact gets no record rather than a formatted placeholder the
    judge would have to reject.
    """
    if not selected:
        return None
    artifact_ref = selected.get("artifact_ref")
    if not isinstance(artifact_ref, str) or not artifact_ref.strip():
        return None
    artifact = Path(artifact_ref)
    if not artifact.exists():
        return None
    digest = selected.get("artifact_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        try:
            digest, _entries = directory_digest(artifact)
        except OSError:
            return None
    return {
        "recipe_id": str(selected.get("recipe_id", "")),
        "artifact_ref": str(artifact),
        "artifact_sha256": digest,
        "attempt": selected.get("attempt"),
    }


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


def _identity_detail(manifest: CampaignManifest) -> str:
    """Exactly which objects were hashed and which bytes matched."""
    detail = f"base tree digest matches {manifest.base_model_digest[:12]}"
    if manifest.has_parent_adapter():
        detail += (
            f"; parent adapter {manifest.parent_adapter_digest[:12]} verified "
            f"over base for {manifest.parent_version}"
        )
    else:
        detail += f"; {manifest.parent_version} is the base itself (no adapter declared)"
    return detail


def _verify_digest(path: Path, field: str, declared: str, *, of: str) -> None:
    """The declared digest must match the bytes on disk, or the run refuses."""
    if not path.exists():
        raise CampaignRunRefusal(f"{field} {str(path)!r} does not exist ({of})")
    digest, _entries = directory_digest(path)
    if digest != declared:
        raise CampaignRunRefusal(
            f"{field} {declared[:12]} does not match the tree's digest {digest[:12]} "
            f"({of}); the campaign would train from a model other than the one it "
            "preregistered"
        )


def _verify_parent_identity(manifest: CampaignManifest) -> None:
    """Verify the base, and the parent adapter separately when declared.

    A base digest and an adapter digest identify different objects, so each is
    checked against its own tree. Overloading one field to mean either is what
    made the gen2 manifest ambiguous.
    """
    _verify_digest(
        Path(manifest.base_model_path),
        "base_model_digest",
        manifest.base_model_digest,
        of="the dense base tree",
    )
    if manifest.has_parent_adapter():
        _verify_digest(
            Path(manifest.parent_adapter_path),
            "parent_adapter_digest",
            manifest.parent_adapter_digest,
            of=f"the {manifest.parent_version} adapter tree",
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


def build_evaluator(
    manifest: CampaignManifest, *, state_root: str | Path | None = None
) -> CandidateEvaluator | None:
    """Build the production candidate evaluator, or ``None`` when none is wired.

    The evaluator is what makes the candidate arm a *run output*: it is asked to
    measure the artifact this run selected, under the protocol the campaign
    declared. This build wires none, and ``None`` is a refusal
    (:data:`CANDIDATE_EVALUATION_NOT_PRODUCED`) rather than a licence to read a
    report from somewhere else -- a campaign that cannot evaluate its own
    candidate has nothing to adjudicate on.
    """
    if default_evaluator_factory is None:
        return None
    return default_evaluator_factory(manifest, state_root=state_root)


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


def certify_before_lineage(
    manifest: CampaignManifest,
    *,
    root: Path,
    selected: Mapping[str, Any] | None,
) -> Certification:
    """Certify the run's own evidence, before the lineage record is written.

    ``_finalize`` records a generation only for a PROMOTED decision, so the
    certification verdict is applied to the decision *first*: a candidate that
    the frozen judge would refuse at audit time -- because the trusted-ancestor
    arm is missing, because a protected slice was never really measured, or
    because the evaluation names bytes the campaign did not select -- cannot be
    recorded as promoted and then audited. The mechanism is the production one
    (:mod:`chowder.growth.certification`), the same code the frozen judge runs,
    so the run and the audit cannot disagree about what the evidence says.

    Arms are read from the judged evidence set this run wrote, never from a
    report handed to the campaign: ``candidate_evaluation.json`` is bound to the
    digest of the artifact the run selected, the parent arm to the frozen parent
    adapter, and the ancestor arm to the frozen base model.
    """
    protection = manifest.protection
    if (
        not protection.trusted_ancestor_version
        or protection.slice_regression_max is None
        or protection.protocol is None
    ):
        return Certification(
            status=UNKNOWN,
            rows=(
                CertificationRow(
                    requirement="a declared branch-protection policy",
                    status=UNKNOWN,
                    detail=(
                        "the campaign declares no protection policy "
                        "(trusted_ancestor_version + slice_regression_max + "
                        "protocol), so it cannot certify that the branch is "
                        "protected"
                    ),
                ),
            ),
            reasons=(
                "the campaign declares no protection policy, so no generation may "
                "be recorded as promoted",
            ),
        )
    root = Path(root)
    arms: dict[str, Path | None] = {
        "candidate": root / CERTIFICATION_EVIDENCE["candidate"],
        "parent": (
            root / CERTIFICATION_EVIDENCE["parent"]
            if (root / CERTIFICATION_EVIDENCE["parent"]).is_file()
            else None
        ),
        "ancestor": (
            root / CERTIFICATION_EVIDENCE["ancestor"]
            if (root / CERTIFICATION_EVIDENCE["ancestor"]).is_file()
            else None
        ),
    }
    ancestor_path = str(manifest.baseline_eval_report_path or "")
    if not ancestor_path:
        arms["ancestor"] = None
    adapter_digests: dict[str, str] = {}
    selected_digest = str((selected or {}).get("artifact_sha256", ""))
    if selected_digest:
        adapter_digests["candidate"] = selected_digest
    if manifest.has_parent_adapter() and manifest.parent_adapter_digest:
        adapter_digests["parent"] = manifest.parent_adapter_digest
    base_digests = {"ancestor": manifest.base_model_digest} if manifest.base_model_digest else {}
    return certify_run_root(
        policy=ProtectionPolicy(
            required_protected=tuple(manifest.protected_benchmarks),
            slice_regression_max=float(protection.slice_regression_max),
            trusted_ancestor_version=protection.trusted_ancestor_version,
            protocol=protection.protocol,
            candidate_version=manifest.resolved_candidate_version(),
            parent_version=manifest.parent_version,
        ),
        run_root=root,
        arms=arms,
        expected_adapter_digests=adapter_digests,
        expected_base_digests=base_digests,
    )


def _evaluate_candidate(
    manifest: CampaignManifest,
    *,
    root: Path,
    selected: Mapping[str, Any],
    eval_fn: CandidateEvaluator | None,
) -> tuple[EvaluationRequest, CandidateEvaluation]:
    """Measure the artifact this run selected, through the declared evaluator.

    Every failure is named: no seam, a seam that raises, a report that is not
    candidate-measured, a report bound to other bytes. None of them is
    recoverable by reading a report from somewhere else -- that path no longer
    exists -- so each is a refusal.
    """
    artifact_ref = str(selected.get("artifact_ref") or "")
    artifact_digest = str(selected.get("artifact_sha256") or "")
    protection = manifest.protection
    evaluator = eval_fn if eval_fn is not None else build_evaluator(
        manifest, state_root=root
    )
    if evaluator is None:
        raise CampaignRunRefusal(
            f"{CANDIDATE_EVALUATION_NOT_PRODUCED}: no candidate evaluator is "
            "wired for this campaign, so it cannot measure the artifact it "
            "selected; the candidate arm is a run output -- a report prepared "
            "outside the run is not accepted as candidate evidence"
        )
    try:
        request = EvaluationRequest(
            cycle_id=manifest.cycle_id,
            candidate_version=manifest.resolved_candidate_version(),
            base_model_path=manifest.base_model_path,
            base_model_digest=manifest.base_model_digest,
            artifact_ref=artifact_ref,
            artifact_sha256=artifact_digest,
            recipe_id=str(selected.get("recipe_id") or ""),
            attempt=str(selected.get("attempt") or ""),
            target_benchmarks=tuple(manifest.target_benchmarks),
            protected_benchmarks=tuple(manifest.protected_benchmarks),
            broad_benchmarks=tuple(manifest.broad_benchmarks),
            calibration_benchmarks=tuple(manifest.calibration_benchmarks),
            reliability_benchmarks=tuple(manifest.reliability_benchmarks),
            protocol=(
                protection.protocol.to_dict() if protection.protocol is not None else {}
            ),
            output_root=str(root),
        )
    except CandidateEvaluationRefusal as refusal:
        raise CampaignRunRefusal(str(refusal)) from refusal
    try:
        returned = evaluator(request)
    except CampaignRunRefusal:
        raise
    except Exception as error:  # the evaluator failed rather than measured
        raise CampaignRunRefusal(
            f"{CANDIDATE_EVALUATION_NOT_PRODUCED}: the candidate evaluator "
            f"failed: {error}"
        ) from error
    try:
        return request, validate_candidate_report(coerce_evaluation(returned), request)
    except CandidateEvaluationRefusal as refusal:
        raise CampaignRunRefusal(str(refusal)) from refusal


def _adjudicate(
    manifest: CampaignManifest,
    *,
    cycle: GrowthCycle,
    binder: MetricBinder,
    total: ComputeCost,
    candidate_runs: tuple[BenchmarkRun, ...],
) -> PromotionAssembly:
    """Bind the run's own candidate measurement and let the declared rule decide.

    ``candidate_runs`` are the rows this run produced by evaluating the artifact
    it selected. The parent side is still a declared input (the parent is the
    generation that already exists); the candidate side never is.
    """
    parent_runs: tuple[BenchmarkRun, ...] = ()
    if manifest.parent_eval_report_path:
        parent_runs = _runs_from_report(
            manifest.parent_eval_report_path, "parent_eval_report_path"
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
            **manifest.model_identity(),
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
    *,
    cost: Mapping[str, Any] | None = None,
) -> CampaignRun:
    run = CampaignRun(
        cycle_id=manifest.cycle_id,
        parent_version=manifest.parent_version,
        candidate_version=candidate_version,
        verdict="REFUSED",
        phases=tuple(phases),
        admission=tuple(admission),
        cost=dict(cost or {}),
        settlement={},
        ceiling_enforcement={},
        certification={},
        selection={},
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
        settlement=run.settlement,            ceiling_enforcement=run.ceiling_enforcement,
            certification=run.certification,
            selection=run.selection,
            promotion=run.promotion,

        record_path=str(path),
    )
