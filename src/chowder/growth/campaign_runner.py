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
from chowder.local_model_manifest import LocalModelManifestError, model_content_digest

from .candidate_evaluation import (
    CANDIDATE_EVALUATION_NOT_PRODUCED,
    CandidateEvaluation,
    CandidateEvaluationRefusal,
    CandidateEvaluator,
    EvaluationRequest,
    coerce_evaluation,
    evaluation_detail,
    validate_candidate_report,
    validate_evaluation_cost,
    write_candidate_evaluation,
)

from .candidate_search import (
    CandidateSearchRefusal,
    CandidateSearchDeclaration,
    SearchPlan,
    plan_search,
    run_search,
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
    digest_of,
)
from .compute_cost import ComputeCost, CycleCostLedger
from .evaluation_binding import EvaluationMaterial, SubprocessEvaluationFn
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

#: The one declared field that deliberately drives nothing.




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
    #: What each attempt produced, compactly. A run that refuses after training
    #: records the spend and the artifacts here: the compute really happened, and
    #: a refusal is evidence rather than a lost run.
    attempts: tuple[Mapping[str, Any], ...] = ()

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
            "attempts": [dict(attempt) for attempt in self.attempts],
            # A refusal names itself here as well as in its phases, so a caller
            # does not have to know which phase refused to report why.
            "refused_by": self.refused_by,
            "refusal_reason": self.refusal_reason,
        }

    @property
    def refused_by(self) -> str:
        """The phase that refused this run, or the empty string."""
        for phase in self.phases:
            if str(phase.get("verdict")) == "refused":
                return str(phase.get("phase"))
        return ""

    @property
    def refusal_reason(self) -> str:
        """Why this run refused, in its own words, or the empty string."""
        for phase in self.phases:
            if str(phase.get("verdict")) == "refused":
                return str(phase.get("detail"))
        return ""




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

    # The declared bounded candidate search, projected before anything is spent.
    # A search that cannot fit its own envelope, or a round the executor would
    # refuse to admit, refuses here -- the same rule the single pass applies to
    # its recipes. Undeclared, this is an undeclared plan and the loop below is
    # exactly the single pass it always was.
    try:
        search = search_plan_for(manifest, cycle=cycle, recipes=recipes)
    except CandidateSearchRefusal as refusal:
        phases.append(
            {"phase": "candidate_search", "verdict": "refused", "detail": str(refusal)}
        )
        return _refuse(manifest, root, candidate_version, phases, admission)
    if search.declared:
        phases.append(
            {
                "phase": "candidate_search",
                "verdict": "ok",
                "detail": (
                    f"{len(search.rounds)} declared round(s) over "
                    f"{len(recipes)} candidate(s), worst case "
                    f"{search.total_device_gpu_hours:.6f} device / "
                    f"{search.total_wall_gpu_hours:.6f} wall GPU-h within the "
                    "declared search envelope"
                ),
            }
        )

    # Campaign admission: the per-recipe ceilings bound what the executor may
    # start, and these bound what the campaign as a whole plans to spend. A
    # campaign whose own plan does not fit its declared envelope refuses here,
    # before compute, rather than discovering it only when the actuals land.
    # A declared search spends its rounds, so it is the search's total -- not a
    # single pass -- that the campaign ceiling has to cover.
    projected = ComputeCost(
        device_gpu_hours=(
            search.total_device_gpu_hours
            if search.declared
            else sum(recipe.projected_device_gpu_hours for recipe in recipes)
        ),
        wall_gpu_hours=(
            search.total_wall_gpu_hours
            if search.declared
            else sum(recipe.projected_wall_gpu_hours for recipe in recipes)
        ),
        source=(
            f"campaign projection ({len(search.rounds)} search rounds over "
            f"{len(recipes)} recipes)"
            if search.declared
            else f"campaign projection ({len(recipes)} admitted recipes)"
        ),
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

    # Pre-compute readiness: everything the run will need *after* training is
    # loaded and checked here, while refusing still costs nothing. The framing
    # is deliberate -- the failure this prevents is spending a training budget
    # and only then discovering that the evidence a verdict needs cannot be
    # read, or that no instrument exists to measure what was just produced.
    try:
        arms_detail = _preflight_arms(manifest)
    except CampaignRunRefusal as refusal:
        phases.append(
            {"phase": "readiness", "verdict": "refused", "detail": str(refusal)}
        )
        return _refuse(manifest, root, candidate_version, phases, admission)

    try:
        evaluator = (
            eval_fn if eval_fn is not None else build_evaluator(manifest, state_root=root)
        )
    except CampaignRunRefusal as refusal:
        phases.append(
            {"phase": "readiness", "verdict": "refused", "detail": str(refusal)}
        )
        return _refuse(manifest, root, candidate_version, phases, admission)
    if evaluator is None:
        phases.append(
            {
                "phase": "readiness",
                "verdict": "refused",
                "detail": (
                    f"{CANDIDATE_EVALUATION_NOT_PRODUCED}: no candidate evaluator "
                    "is available for this campaign, so nothing could measure the "
                    "artifact it is about to train"
                ),
            }
        )
        return _refuse(manifest, root, candidate_version, phases, admission)

    # The evaluator's own admission seam: material coverage, dataset existence
    # and slice length are all knowable before a GPU is touched, and the
    # executor's admission is the mirror image of this one.
    admit_evaluator = getattr(evaluator, "admit", None)
    if callable(admit_evaluator):
        evaluator_refusal = admit_evaluator(
            benchmarks=tuple(manifest.evaluated_benchmarks),
            protocol=manifest.protection.require_protocol(),
        )
        if evaluator_refusal is not None:
            phases.append(
                {
                    "phase": "readiness",
                    "verdict": "refused",
                    "detail": "; ".join(str(part) for part in evaluator_refusal),
                }
            )
            return _refuse(manifest, root, candidate_version, phases, admission)
    phases.append(
        {
            "phase": "readiness",
            "verdict": "ok",
            "detail": (
                f"declared evidence readable ({arms_detail}); an evaluator is "
                "available and covers every declared benchmark"
            ),
        }
    )

    ledger = CycleCostLedger(cycle_id=manifest.cycle_id)
    results: list[Mapping[str, Any]] = []
    stopped_by: str | None = None

    def charge(evidence: Mapping[str, Any]) -> None:
        """Charge one attempt to the campaign's own ledger as it happens."""
        ledger.add(
            f"{evidence.get('recipe_id')} attempt",
            "failed_attempt" if evidence.get("status") != STATUS_SUCCEEDED else "training",
            _attempt_cost(evidence),
            recipe_id=str(evidence.get("recipe_id", "")),
            notes=(
                f"status={evidence.get('status')}"
                + (
                    f" round={evidence.get('search_round')}"
                    if evidence.get("search_round") is not None
                    else ""
                )
            ),
        )

    def overrun() -> str | None:
        if not stops_on_campaign_overrun(manifest.stopping_rules):
            return None
        running = settle_campaign(manifest, total=ledger.total())
        if running.compliant:
            return None
        return "campaign ceiling reached before the remaining candidates"

    search_run = None
    if search.declared:
        # Bounded candidate search: cheap rounds first, only the final round's
        # survivors offered to selection. Every attempt is charged and recorded
        # exactly as a single-pass recipe is, so a stopped round still leaves
        # its spend in the accounting.
        search_run = run_search(
            search,
            declaration=manifest.candidate_search,
            recipes=recipes,
            project_cost=cycle.planner.project_cost,
            run_attempt=lambda recipe: executor(recipe, plan.items),
            on_attempt=lambda evidence, row: charge(evidence),  # noqa: ARG005
            should_stop=lambda _device, _wall: overrun(),  # noqa: ARG005
        )
        results = list(search_run.final_results)
        stopped_by = search_run.stopped_by
        phases.append(
            {
                "phase": "candidate_search_run",
                "verdict": "stopped" if stopped_by else "ok",
                "detail": (
                    stopped_by
                    or (
                        f"{len(search_run.rounds)} round(s) ran; "
                        f"{len(search_run.survivors)} survivor(s) from the last "
                        f"round, whose results alone selection may read"
                    )
                ),
            }
        )
    else:
        for recipe in recipes:
            evidence = dict(executor(recipe, plan.items))
            evidence["recipe_id"] = recipe.recipe_id
            results.append(evidence)
            charge(evidence)
            stopped_by = overrun()
            if stopped_by:
                break
    # Every attempt, from every round: a search's earlier, cheaper rounds are
    # real compute and real evidence, so they stay in the record even though
    # selection may only read the final round's results.
    attempted: tuple[Mapping[str, Any], ...] = (
        tuple(
            evidence
            for round_attempts in search_run.round_attempts
            for evidence in round_attempts
        )
        if search_run is not None
        else tuple(results)
    )
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
        # A run that stopped mid-flight must say so even when the stop left it
        # nothing to select: the stop is part of what happened, not a detail
        # only successful runs report.
        if stopped_by:
            phases.append(
                {"phase": "stopping", "verdict": "stopped", "detail": stopped_by}
            )
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
    #
    # A refusal at this point is *recorded*, not raised: the training compute
    # really happened, so the accounting, the attempts and the reason are
    # written out as evidence before the run stops.
    try:
        _verify_selected_artifact(selected, purpose="before it is measured")
        request, evaluation = _evaluate_candidate(
            manifest, root=root, selected=selected, eval_fn=evaluator
        )
    except CampaignRunRefusal as refusal:
        accounting_digest = ledger.write(accounting_path)
        total = ledger.total()
        if stopped_by:
            phases.append(
                {"phase": "stopping", "verdict": "stopped", "detail": stopped_by}
            )
        phases.append(
            {
                "phase": "candidate_evaluation",
                "verdict": "refused",
                "detail": str(refusal),
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
            attempts=_attempt_summary(attempted),
            selection=selected,
        )
    # ``validate_candidate_report`` has already refused an evaluation that
    # reports no cost, so this is the measured figure -- never a default zero.
    ledger.add(
        "candidate evaluation",
        "evaluation",
        validate_evaluation_cost(evaluation),
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

    # The bytes are re-read here, immediately before they are written into the
    # judged evidence set and bound into a verdict: the digest recorded at
    # training time is a claim, and this is the boundary where a recorded claim
    # would otherwise become a certified one.
    try:
        _verify_selected_artifact(selected, purpose="before the verdict is recorded")
        written = write_certification_evidence(
            manifest, root=root, selected=selected, candidate=evaluation
        )
    except CampaignRunRefusal as refusal:
        accounting_digest = ledger.write(accounting_path)
        total = ledger.total()
        phases.append(
            {
                "phase": "certification_evidence",
                "verdict": "refused",
                "detail": str(refusal),
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
            attempts=_attempt_summary(attempted),
            selection=selected,
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
        # Every attempt the run made, including a declared search's earlier,
        # cheaper rounds: they are real compute and real evidence even though
        # selection may only read the final round's results.
        attempts=_attempt_summary(attempted),
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

#: Which declared manifest field supplies each arm the run *reads*. The
#: candidate arm is absent here on purpose: it is produced by the run.








# --------------------------------------------------------------------------
# declared inputs
# --------------------------------------------------------------------------














#: The declared parent profile is not the evidence-attributed profile a
#: curriculum may be planned from. The shape a preparation pass used to write --
#: every known skill carrying the mean of whatever was measured -- cannot say
#: which skills have evidence, so planning from it trains against numbers no
#: benchmark produced.












def build_evaluator(
    manifest: CampaignManifest,
    *,
    state_root: str | Path | None = None,
    runner: Any = None,
) -> CandidateEvaluator | None:
    """Build the production candidate evaluator the campaign declared.

    The evaluator is what makes the candidate arm a *run output*: it is asked to
    measure the artifact this run selected, under the protocol the campaign
    declared, and it writes that measurement into the run root. A declared
    ``default_evaluator_factory`` overrides the production instrument (that
    seam is what the no-GPU harness supplies); otherwise this builds the real
    one from the campaign's declared evaluation material.

    ``None`` -- an instrument that cannot be built -- is a refusal
    (:data:`CANDIDATE_EVALUATION_NOT_PRODUCED`), never a licence to read a
    report from somewhere else: a campaign that cannot evaluate its own
    candidate has nothing to adjudicate on.
    """
    if default_evaluator_factory is not None:
        return default_evaluator_factory(manifest, state_root=state_root)
    if not str(manifest.evaluation_material_path).strip():
        raise CampaignRunRefusal(
            f"{CANDIDATE_EVALUATION_NOT_PRODUCED}: the campaign declares no "
            "evaluation_material_path, so the production evaluator has no "
            "datasets to measure the selected artifact on; the candidate arm is "
            "a run output and the material it is measured with is an input"
        )
    material = EvaluationMaterial.load(manifest.evaluation_material_path)
    protocol = manifest.protection.require_protocol(source=manifest.cycle_id)
    return SubprocessEvaluationFn(
        run_root=Path(state_root or manifest.state_root),
        material=material,
        protocol=protocol,
        base_model_path=manifest.base_model_path,
        base_model_digest=manifest.base_model_digest,
        # The declared execution throughput, so the candidate arm is measured
        # the same way the parent and ancestor arms were declared to be.
        batch_size=manifest.evaluation_execution.batch_size,
        # Resolved here, in this module's namespace, so one patch point covers
        # every process this campaign starts (trainer and evaluator alike).
        runner=runner or default_runner,
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
        # One attempt's process holds training *and* the in-run evaluation, so
        # the default 3600 s could not cover both: attempt-09 trained in 1766 s
        # and came within 34 s of this bound while its evaluation was still
        # running. The declared suites measure in 3706 s (the parent arm), so an
        # attempt needs ~5500 s. 7200 s is the worker timeout the arm
        # measurements already run under, with margin over that sum.
        timeout_seconds=7200.0,
    )


# --------------------------------------------------------------------------
# cycle assembly and adjudication
# --------------------------------------------------------------------------










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














def _refuse(
    manifest: CampaignManifest,
    root: Path,
    candidate_version: str,
    phases: list[Mapping[str, Any]],
    admission: list[Mapping[str, Any]],
    *,
    cost: Mapping[str, Any] | None = None,
    attempts: Sequence[Mapping[str, Any]] = (),
    selection: Mapping[str, Any] | None = None,
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
        selection=dict(selection or {}),
        promotion=None,
        record_path="",
        attempts=tuple(dict(attempt) for attempt in attempts),
    )
    return _write_record(run, root, outcome=None)


#: The artifact the run selected is not the artifact it measured.
CANDIDATE_ARTIFACT_DIGEST_STALE = "CANDIDATE_ARTIFACT_DIGEST_STALE"








# --------------------------------------------------------------------------
# readiness: one authoritative, zero-compute inspection of a declaration
# --------------------------------------------------------------------------
#
# Everything a run checks *before* it touches a GPU is knowable from the
# declaration and the files it names. ``run_campaign`` refuses at the first of
# those checks it fails; this reports *all* of them at once, so an operator (or
# CI) can see what is missing without spending compute to discover it one
# refusal per attempt. Both read the same helpers, so a ready declaration is one
# a run will not refuse before training.
#
# Stable machine identifiers, so a refusal can be consumed without parsing
# prose: ``status`` is READY only when every check is ``ok``.








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
        certification=run.certification,
        selection=run.selection,
        promotion=run.promotion,
        record_path=str(path),
        attempts=run.attempts,
    )


# --- the controller surface, re-exported ------------------------------------
#
# These live in ``campaign_controllers`` now, one module per decision the
# runner defers. They are re-exported here because this module's import path
# is part of the package's contract: ``cli``, ``growth_loop``,
# ``campaign_prepare`` and the test suite all reach for them here, and a
# refactor that moved a name without re-exporting it would break callers to
# save nothing. Import the controller directly when that is what you mean.

from .campaign_controllers.certification import (  # noqa: F401
    CERTIFICATION_EVIDENCE,
    _chosen_candidate_document,
    _measurement_artifacts,
    _preflight_arms,
    certify_before_lineage,
    write_certification_evidence,
)

from .campaign_controllers.contracts import (  # noqa: F401
    CampaignRunRefusal,
    FIELD_ENFORCEMENT,
    NON_BEHAVIORAL_FIELDS,
    assert_every_field_enforced,
)

from .campaign_controllers.declared import (  # noqa: F401
    DECLARED_INPUT_REQUIREMENTS,
    _require_path,
    require_declared_inputs,
    undeclared_inputs,
)

from .campaign_controllers.evaluation import (  # noqa: F401
    PARENT_PROFILE_NOT_ATTRIBUTED,
    _evaluate_candidate,
    _load_binder,
    _load_data_registry,
    _load_hardware_budget,
    _load_material,
    _load_profile,
    _runs_from_report,
)

from .campaign_controllers.planning import (  # noqa: F401
    CampaignPlan,
    _build_cycle,
    _select_recipes,
    envelope_for,
    plan_campaign,
    search_plan_for,
)

from .campaign_controllers.promotion import (  # noqa: F401
    _adjudicate,
    _finalize,
    _identity_detail,
    _verify_base_identity,
    _verify_digest,
    _verify_parent_identity,
    _with_resource_veto,
)

from .campaign_controllers.readiness import (  # noqa: F401
    READINESS_ANCESTOR_ARM,
    READINESS_BASE_IDENTITY,
    READINESS_CAMPAIGN_PROJECTION,
    READINESS_CANDIDATE_SEARCH,
    READINESS_CONTAMINATION,
    READINESS_DATA_REGISTRY,
    READINESS_DECLARED_INPUT,
    READINESS_EVALUATOR,
    READINESS_EVALUATOR_COVERAGE,
    READINESS_HARDWARE_BUDGET,
    READINESS_PARENT_ADAPTER_IDENTITY,
    READINESS_PARENT_ARM,
    READINESS_PARENT_PROFILE,
    READINESS_PLAN,
    READINESS_PROJECT_TEMPLATE,
    READINESS_PROTECTION_POLICY,
    READINESS_RECIPE_SET,
    READINESS_SCHEMA,
    READINESS_TRAINING_MATERIAL,
    ReadinessCheck,
    ReadinessReport,
    check_campaign_readiness,
)

from .campaign_controllers.training import (  # noqa: F401
    _attempt_cost,
    _attempt_summary,
    _verify_selected_artifact,
)


__all__ = sorted(
    [
    "CANDIDATE_ARTIFACT_DIGEST_STALE",
    "CERTIFICATION_EVIDENCE",
    "CampaignPlan",
    "CampaignRun",
    "CampaignRunRefusal",
    "DECLARED_INPUT_REQUIREMENTS",
    "FIELD_ENFORCEMENT",
    "NON_BEHAVIORAL_FIELDS",
    "PARENT_PROFILE_NOT_ATTRIBUTED",
    "READINESS_ANCESTOR_ARM",
    "READINESS_BASE_IDENTITY",
    "READINESS_CAMPAIGN_PROJECTION",
    "READINESS_CANDIDATE_SEARCH",
    "READINESS_CONTAMINATION",
    "READINESS_DATA_REGISTRY",
    "READINESS_DECLARED_INPUT",
    "READINESS_EVALUATOR",
    "READINESS_EVALUATOR_COVERAGE",
    "READINESS_HARDWARE_BUDGET",
    "READINESS_PARENT_ADAPTER_IDENTITY",
    "READINESS_PARENT_ARM",
    "READINESS_PARENT_PROFILE",
    "READINESS_PLAN",
    "READINESS_PROJECT_TEMPLATE",
    "READINESS_PROTECTION_POLICY",
    "READINESS_RECIPE_SET",
    "READINESS_SCHEMA",
    "READINESS_TRAINING_MATERIAL",
    "ReadinessCheck",
    "ReadinessReport",
    "_ceiling_enforcement",
    "_refuse",
    "_write_record",
    "assert_every_field_enforced",
    "build_evaluator",
    "build_executor",
    "certify_before_lineage",
    "check_campaign_readiness",
    "default_evaluator_factory",
    "default_runner",
    "envelope_for",
    "plan_campaign",
    "require_declared_inputs",
    "run_campaign",
    "search_plan_for",
    "undeclared_inputs",
    "write_certification_evidence",
    ]
)
