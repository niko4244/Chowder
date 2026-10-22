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
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
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
from .owner_approval import (
    authorize_promotion,
    judge_certifies_candidate,
    requires_owner_approval,
    verify_grant,
)
from .recipe_planner import HardwareBudget, RecipePlanner, TrainingRecipe
from .campaign_resume import (
    ResumeRefusal,
    commit_receipt,
    declared_inputs_digest,
    load_receipts,
    validate_evidence as validate_resume_evidence,
)
from .promotion_transaction import (
    content_digest,
    load_transaction,
    prepare_transaction,
    reconcile_transaction,
    transaction_path,
)

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
    "base_model_digest": "must equal the dense base tree's model-content digest (payload files only), else the run refuses",
    "parent_adapter_path": "hashed and compared with parent_adapter_digest when declared",
    "parent_adapter_digest": "must equal the parent adapter tree's directory digest, else the run refuses",
    "state_root": "attempts, registry, ledger, accounting, the judged evidence set and campaign-run.json live here",
    "target_benchmarks": "the cycle's target set: the improvement gate",
    "target_threshold": "the declared improvement bar (one target benchmark and a positive min_delta): validated at load against the declared target set, so a bar on an unmeasured benchmark refuses before any compute; the promotion judge reads it as the delta that must be cleared, so the standard is never a judge-side constant",
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
    "evaluation_material_path": "the dataset, fields and scoring the production evaluator measures the selected candidate with; a declared-but-missing file, or one that names no dataset for a declared benchmark, refuses before compute",
    "baseline_eval_report_path": "the trusted-ancestor (gen0) arm of the judged evidence set: branch protection is judged against it, never against an unresolved parent",
    "protection": "the declared branch-protection policy (trusted ancestor version + slice regression tolerance) the certification gate applies before any lineage record is written",
    "evaluation_execution": "how many rows one generate call decodes, for every arm and for the candidate: the candidate evaluator and both arm measurements read this one value, and each records it in its own evidence; batching changes generated tokens (measured: 14/16 rows agree with a single-row pass), so the arms and the candidate must share it and never be chosen per path",
    "candidate_search": "the declared bounded candidate search: its rounds, starting step budget, multiplier and survival rule are preregistered here, its worst-case cost is projected before any compute and must fit its own declared device/wall envelope *and* the campaign's ceilings, and the runner refuses a declared search that does not. Absent (rounds=0) means one pass over the declared recipes, which is what every manifest predating it does",
    "retention_profile": "the campaign's preregistered promotion gates: the cycle evaluates every constraint fail-closed before anything promotes (a target win over a constrained regression is REJECTED, an unmeasured constraint is a violation), and a constraint naming a benchmark the declared sets never measure refuses at load",
    "eval_tier_policy": "the declared trust classification of the campaign's benchmarks: a retention constraint measured on search-readable evidence refuses — at load when both are declared, and again at promotion",
    "training_backend": "the declared execution backend: the runner builds the campaign's executor through the provider it names (absent means local, the historical subprocess path). A declaration error — an unknown provider, an unsupported config key, a template whose trainer is not the declared one, an Unsloth knob the isolated engine refuses — stops the campaign before any compute, and an auto declaration records the provider it chose and why (with every candidate it refused) as the run's own phase",
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
        (
            "evaluation_material_path",
            "the production evaluator has no data to measure the selected candidate "
            "on, so the run would spend its training compute and refuse afterwards",
        ),
        (
            "parent_eval_report_path",
            "the promotion rule compares the candidate against its parent, so a run "
            "without that arm can only reach INCONCLUSIVE",
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
            if str(phase.get("verdict")) in ("refused", CAMPAIGN_INTERRUPTED):
                return str(phase.get("phase"))
        return ""

    @property
    def refusal_reason(self) -> str:
        """Why this run refused, in its own words, or the empty string."""
        for phase in self.phases:
            if str(phase.get("verdict")) in ("refused", CAMPAIGN_INTERRUPTED):
                return str(phase.get("detail"))
        return ""


def _restored_run(document: Mapping[str, Any], root: Path) -> CampaignRun:
    """Return an already completed campaign; never run its evaluator twice."""
    return CampaignRun(
        cycle_id=str(document["cycle_id"]),
        parent_version=str(document["parent_version"]),
        candidate_version=str(document["candidate_version"]),
        verdict=str(document["verdict"]),
        phases=tuple(document["phases"]),
        admission=tuple(document["admission"]),
        cost=document["cost"],
        settlement=document["settlement"],
        ceiling_enforcement=document["ceiling_enforcement"],
        certification=document["certification"],
        selection=document["selection"],
        promotion=document["promotion"],
        attempts=tuple(document["attempts"]),
        record_path=str(root / "campaign-run.json"),
    )


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


# --------------------------------------------------------------------------
# the campaign journal: a crash record written at every boundary
# --------------------------------------------------------------------------

#: The append-only journal the runner writes as the campaign advances: one
#: JSON line per boundary the runner computes, flushed **and fsynced** as the
#: boundary happens. An external termination of the process tree cannot be
#: intercepted by any Python exit path -- the only other durable record,
#: ``campaign-run.json``, was terminal, so a kill before it destroyed the
#: entire campaign record (phase list, admission decisions, budget spend,
#: which recipes were attempted and why). The journal closes that hole.
CAMPAIGN_JOURNAL_NAME = "campaign-journal.jsonl"

#: The entry refusal for a state root whose journal records no terminal
#: outcome: a previous run there never reached a verdict, and re-running it
#: would silently start from the top and re-spend every attempt it already
#: made.
CAMPAIGN_INTERRUPTED = "CAMPAIGN_INTERRUPTED"


def _journal_path(root: Path) -> Path:
    return root / CAMPAIGN_JOURNAL_NAME


def _journal_event(
    root: Path,
    *,
    phase: str,
    status: Any,
    recipe_id: Any = None,
    attempt: Any = None,
    gpu_hours: Any = None,
    detail: str | None = None,
    identity: str | None = None,
    index: int | None = None,
    proof_mode: str | None = None,
) -> None:
    """Append one boundary to the campaign journal, durably.

    The line is flushed **and fsynced** before the call returns: the journal
    is written for the process that does not survive long enough to close the
    file. Everything the runner has computed by that boundary is on disk
    afterwards; everything it has not computed is, honestly, absent.
    """
    record: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "attempt": attempt,
        "recipe_id": recipe_id,
        "status": status,
        "gpu_hours": gpu_hours,
    }
    if detail is not None:
        record["detail"] = detail
    if identity is not None:
        record["identity"] = identity
    if index is not None:
        record["index"] = index
    if proof_mode is not None:
        record["proof_mode"] = proof_mode
    with open(_journal_path(root), "a", encoding="utf-8") as journal:
        journal.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        journal.flush()
        os.fsync(journal.fileno())


def _journal_outcome(journal: Path) -> str | None:
    """Return only the latest run's durable terminal outcome, or None.

    A terminal from an earlier run must not mask a newer interrupted start.
    Any malformed or torn line is ambiguous and therefore refuses re-entry.
    The terminal must be the final journal event and agree with the durable
    verdict file; a journal marker alone cannot attest to a verdict.
    """
    try:
        text = journal.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    outcome: str | None = None
    last_phase = ""
    for line in text.splitlines(keepends=True):
        # Every _journal_event writes a trailing LF before fsync.
        if not line.endswith("\n"):
            return None
        try:
            record = json.loads(line)
        except (ValueError, TypeError):
            return None
        if not isinstance(record, dict) or not isinstance(record.get("phase"), str):
            return None
        phase = record["phase"]
        if phase == "campaign_start":
            outcome = None
        elif phase == "terminal":
            status = record.get("status")
            if not isinstance(status, str) or not status:
                return None
            outcome = status
        last_phase = phase

    if last_phase != "terminal" or outcome is None:
        return None
    try:
        verdict = json.loads(journal.with_name("campaign-run.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    if not isinstance(verdict, dict) or verdict.get("verdict") != outcome:
        return None
    return outcome


def _refuse_interrupted(
    manifest: CampaignManifest, root: Path, *, promotion_unresolved: bool = False,
    approval_pending: bool = False,
) -> CampaignRun:
    """Refuse entry on a state root whose journal never reached a verdict.

    A journal without a terminal record means a previous run on this state
    root was killed mid-campaign: its attempts are spent and recorded. Rather
    than silently starting from the top and re-spending them (the planner
    reserves fresh attempt directories and selection re-runs from the first
    recipe), the run refuses here and records the refusal. The refusal
    deliberately writes **no** terminal journal record: it is a refusal about
    an *open* record, not the verdict of a campaign, so the entry check below
    keeps refusing every later entry until someone resolves that state
    deliberately -- archive or delete the journal and ``campaign-run.json``.
    """
    journal = _journal_path(root)
    if approval_pending:
        detail = (
            "HUMAN_APPROVAL_PENDING: this root contains a fully measured candidate "
            "awaiting independent frozen REAP judge certification and owner N5 "
            "approval. Its compute is already spent; do not rerun training as a "
            "substitute for the missing release transaction. The original run "
            "and evaluation evidence remain unchanged."
        )
    elif promotion_unresolved:
        detail = (
            "PROMOTION_TRANSACTION_UNRESOLVED: a PREPARED or inconsistent "
            "promotion transaction exists. Its lineage is not admitted; "
            "inspect the campaign-run.json, journal and ledger together. "
            "The attempted compute is already spent and cannot be replayed."
        )
    else:
        detail = (
            f"{CAMPAIGN_INTERRUPTED}: {journal} exists and records no terminal "
            "outcome -- a previous run on this state root did not reach a "
            "verdict, so its attempts are already spent and recorded there. "
            "Resolve that state deliberately (archive or delete the journal and "
            "campaign-run.json) before starting a campaign here; refusing rather "
            "than silently starting from the top."
        )
    _journal_event(root, phase="entry_check", status=CAMPAIGN_INTERRUPTED, detail=detail)
    run = CampaignRun(
        cycle_id=manifest.cycle_id,
        parent_version=manifest.parent_version,
        candidate_version=manifest.resolved_candidate_version(),
        verdict="REFUSED",
        phases=(
            {
                "phase": "entry_check",
                "verdict": CAMPAIGN_INTERRUPTED,
                "detail": detail,
            },
        ),
        admission=(),
        cost={},
        settlement={},
        ceiling_enforcement={},
        certification={},
        selection={},
        promotion=None,
        record_path="",
    )
    # An interrupted promotion can leave its original campaign-run.json
    # published before the terminal journal marker. That original is forensic
    # evidence and must never be overwritten by a later entry refusal: it may
    # be needed to reconcile an already-persisted generation ledger entry.
    # An initial hard kill (no verdict yet) keeps the historical location.
    target = (
        "campaign-entry-refusal.json"
        if (root / "campaign-run.json").exists()
        else "campaign-run.json"
    )
    # terminal=False: this refusal cannot close the interrupted journal.
    return _write_record(run, root, outcome=None, terminal=False, record_name=target)


def _campaign_root_state(
    root: Path, *, replay_identity: str,
) -> tuple[str, tuple[dict[str, Any], ...], str]:
    """Read-only campaign admission: fresh, verified resume, or occupied.

    A terminal verdict is the *end* of a state-root campaign, not authority
    to overwrite that campaign's receipts or spend. Never infer permission
    from an editable human-approval phase: its removal must not rearm compute.
    """
    run_record = root / "campaign-run.json"
    if run_record.exists():
        # The approval phase improves the diagnostic only. Occupancy derives
        # from durable record existence, not from its editable JSON contents.
        try:
            previous = json.loads(run_record.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError):
            previous = None
        phases = previous.get("phases") if isinstance(previous, Mapping) else None
        approval_pending = isinstance(previous, Mapping) and any(
            isinstance(phase, Mapping)
            and phase.get("phase") == "human_approval"
            and phase.get("verdict") == "pending"
            for phase in (phases if isinstance(phases, (list, tuple)) else ())
        )
        return (
            "occupied", (),
            (
                "HUMAN_APPROVAL_PENDING: the qualified candidate awaits frozen "
                "judge and owner approval; do not retrain it in this root."
                if approval_pending else
                "CAMPAIGN_STATE_OCCUPIED: campaign-run.json already exists; "
                "its terminal/pending/ambiguous verdict and prior compute must "
                "remain intact. Start an intentional new campaign in a new state root."
            ),
        )
    journal = _journal_path(root)
    receipts = root / "campaign-attempt-receipts"
    refusal_record = root / "campaign-entry-refusal.json"
    if not journal.exists():
        attempts = root / "attempts"
        orphaned_attempts = attempts.is_dir() and any(attempts.iterdir())
        leftover_outputs = any(
            (root / filename).exists()
            for filename in (
                "cycle_compute_accounting.json", "candidate_evaluation.json",
                "chosen_candidate.json", "gen2_certification.json",
            )
        )
        if (
            receipts.exists() or refusal_record.exists()
            or orphaned_attempts or leftover_outputs
        ):
            return (
                "occupied", (),
                "CAMPAIGN_STATE_OCCUPIED: durable attempts, cost/evaluation "
                "artifacts, receipts or an entry refusal remain without their "
                "campaign journal; "
                "restarting could lose already-spent compute.",
            )
        return "fresh", (), "state root has no prior campaign"
    if _journal_outcome(journal) is not None:
        return (
            "occupied", (),
            "CAMPAIGN_STATE_OCCUPIED: the prior campaign has a terminal "
            "journal verdict. Its state root may not be reused for training.",
        )
    # An otherwise valid terminal record whose campaign-run.json was deleted
    # or corrupted is still evidence of concluded/spent work. Do not reinterpret
    # that history as an in-progress campaign eligible for replay.
    try:
        past_rows = [
            json.loads(line)
            for line in journal.read_text(encoding="utf-8").splitlines()
        ]
    except (OSError, UnicodeError, ValueError):
        past_rows = []
    if any(
        isinstance(row, Mapping) and row.get("phase") == "terminal"
        for row in past_rows
    ):
        return (
            "occupied", (),
            "CAMPAIGN_STATE_OCCUPIED: terminal journal evidence remains but "
            "its campaign verdict is missing or inconsistent; refuse replay.",
        )
    try:
        replay = tuple(load_receipts(root, manifest_digest=replay_identity))
    except ResumeRefusal as error:
        return "unresolved", (), str(error)
    return "resume", replay, f"{len(replay)} verified settled attempt(s) may replay"


def _refuse_occupied(
    manifest: CampaignManifest, root: Path, *, detail: str,
) -> CampaignRun:
    """Publish an entry refusal without modifying a prior terminal journal."""
    run = CampaignRun(
        cycle_id=manifest.cycle_id,
        parent_version=manifest.parent_version,
        candidate_version=manifest.resolved_candidate_version(),
        verdict="REFUSED",
        phases=({"phase": "entry_check", "verdict": "refused", "detail": detail},),
        admission=(),
        cost={},
        settlement={},
        ceiling_enforcement={},
        certification={},
        selection={},
        promotion=None,
        record_path="",
    )
    return _write_record(
        run, root, outcome=None, terminal=False,
        record_name="campaign-entry-refusal.json",
    )


@contextmanager
def _campaign_writer_lock(root: Path):
    """Hold one cross-process advisory writer lock for the entire campaign.

    The lock file persists after a crash, but the *kernel lock* does not, so
    the interrupted-journal refusal still determines recovery. A second
    process cannot overwrite the first writer's journal/record/accounting.
    """
    root.mkdir(parents=True, exist_ok=True)
    path = root / ".campaign-writer.lock"
    with path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
            os.fsync(handle.fileno())
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise CampaignRunRefusal(
                "CAMPAIGN_WRITER_BUSY: another campaign owns the state root "
                f"{root}; no execution or record write is authorized"
            ) from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class _JournalingPhases(list):
    """The run's phase record, journaled: every append is one durable line.

    A list subclass, so the run's own ``phases.append(...)`` sites carry the
    journal with them: appending a phase *is* the moment the runner computed
    that decision, and from here on that moment is also fsynced to the
    journal -- a phase decision can no longer exist only in the memory of a
    process that is about to be killed.
    """

    def __init__(self, root: Path) -> None:
        super().__init__()
        self._root = root

    def append(self, phase: Mapping[str, Any]) -> None:
        super().append(phase)
        detail = phase.get("detail")
        _journal_event(
            self._root,
            phase=str(phase.get("phase")),
            status=phase.get("verdict"),
            detail=detail if isinstance(detail, str) else None,
        )


def run_campaign(
    manifest: CampaignManifest,
    *,
    train_fn: Any = None,
    eval_fn: CandidateEvaluator | None = None,
    state_root: str | Path | None = None,
) -> CampaignRun:
    """Validate inputs, then serialize all writers against the state root."""
    assert_every_field_enforced()
    require_declared_inputs(manifest, phase="run")
    root = Path(state_root or manifest.state_root)
    with _campaign_writer_lock(root):
        return _run_campaign_locked(
            manifest, train_fn=train_fn, eval_fn=eval_fn, state_root=root
        )


def _run_campaign_locked(
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
    declaration_digest = content_digest(asdict(manifest))
    try:
        from chowder.worker_env import chowder_source_identity

        replay_identity = content_digest({
            "declaration": declaration_digest,
            "declared_files": declared_inputs_digest(manifest),
            "source_sha256": chowder_source_identity()["source_sha256"],
        })
    except ResumeRefusal as error:
        raise CampaignRunRefusal(str(error)) from error
    # A crash after the durable promotion intent may have left a committed
    # verdict, an unfinished ledger write, or only the PREPARED marker. Replay
    # only a fully verified terminal decision; otherwise preserve forensic
    # evidence and refuse re-entry before training spends anything.
    marker_path = transaction_path(
        root / "ledger", manifest.resolved_candidate_version()
    )
    if marker_path.exists():
        try:
            marker = load_transaction(marker_path)
            recovered = reconcile_transaction(
                root,
                cycle_id=manifest.cycle_id,
                candidate_version=manifest.resolved_candidate_version(),
                journal_verdict=_journal_outcome(_journal_path(root)),
                manifest_digest=declaration_digest,
            )
        except ValueError:
            recovered = None
        if recovered is None or marker is None or recovered.get("parent_version") != manifest.parent_version:
            return _refuse_interrupted(manifest, root, promotion_unresolved=True)
        return _restored_run(recovered, root)
    state, verified, detail = _campaign_root_state(root, replay_identity=replay_identity)
    if state == "occupied":
        return _refuse_occupied(manifest, root, detail=detail)
    if state == "unresolved":
        return _refuse_interrupted(manifest, root)
    interrupted = state == "resume"
    replay = list(verified)
    phases: list[Mapping[str, Any]] = _JournalingPhases(root)
    if not interrupted:
        _journal_event(
            root,
            phase="campaign_start",
            status="started",
            detail=f"campaign {manifest.cycle_id} (policy {PROMOTION_POLICY_VERSION})",
            identity=replay_identity,
        )
    else:
        _journal_event(
            root, phase="campaign_resume", status="verified",
            detail=f"resuming {len(replay)} settled attempt(s) from exact declared campaign",
            identity=replay_identity,
        )

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

    if train_fn is not None:
        executor = train_fn
        # A caller-supplied executor is recorded as such: the run still says
        # what the manifest declared, and does not pretend the supplied
        # executor was the declared provider's.
        from .training_backends import backend_declaration

        backend_record: Mapping[str, Any] = {
            "cycle_id": manifest.cycle_id,
            "declaration": backend_declaration(manifest).to_dict(),
            "executor": "caller-supplied",
            "provider": None,
            "trainer": None,
            "admitted": None,
            "selection": None,
        }
    else:
        executor, backend_record = build_executor_with_selection(
            manifest, state_root=root
        )
    # The declared backend is recorded before any attempt starts: the campaign
    # says which backend it will run on, and why, while that is still a
    # declaration rather than a reconstruction.
    phases.append(
        {
            "phase": "training-backend",
            "verdict": "ok",
            "detail": _backend_detail(backend_record),
            "backend": dict(backend_record),
        }
    )
    if not hasattr(executor, "admit"):
        raise CampaignRunRefusal(
            "the training executor exposes no admission seam (`admit(recipe)`), "
            "so projected cost could not control admission; the campaign refuses "
            "rather than running under an unenforced budget"
        )
    firewall = getattr(executor, "firewall", ContaminationFirewall())
    cycle = _build_cycle(manifest, executor=executor, firewall=firewall, root=root)
    candidate_version = manifest.resolved_candidate_version()

    def precompute_refusal(
        phases: list[Mapping[str, Any]], admission: list[Mapping[str, Any]]
    ) -> CampaignRun:
        # A recovered run already spent and durably journaled training. A
        # changed admission/readiness environment cannot publish a zero-cost
        # terminal verdict over its prior receipt history.
        if interrupted:
            return _refuse_interrupted(manifest, root)
        return _refuse(manifest, root, candidate_version, phases, admission)

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
        _journal_event(
            root,
            phase="admission",
            status="admitted" if refusal is None else "refused",
            recipe_id=recipe.recipe_id,
            gpu_hours=float(recipe.projected_wall_gpu_hours),
            detail=(entry.get("refusal_reason") if refusal is not None else None),
        )
    refused = [entry for entry in admission if not entry["admitted"]]
    detail = "; ".join(
        f"{entry['recipe_id']}: {entry['refusal_reason']}" for entry in refused
    )
    if refused and stops_on_admission_refusal(manifest.stopping_rules):
        phases.append({"phase": "admission", "verdict": "refused", "detail": detail})
        return precompute_refusal(phases, admission)
    admitted_ids = {entry["recipe_id"] for entry in admission if entry["admitted"]}
    recipes = tuple(recipe for recipe in recipes if recipe.recipe_id in admitted_ids)
    if not recipes:
        phases.append({"phase": "admission", "verdict": "refused", "detail": detail})
        return precompute_refusal(phases, admission)
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
        return precompute_refusal(phases, admission)
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
        return precompute_refusal(phases, admission)
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
        return precompute_refusal(phases, admission)

    try:
        evaluator = (
            eval_fn if eval_fn is not None else build_evaluator(manifest, state_root=root)
        )
    except CampaignRunRefusal as refusal:
        phases.append(
            {"phase": "readiness", "verdict": "refused", "detail": str(refusal)}
        )
        return precompute_refusal(phases, admission)
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
        return precompute_refusal(phases, admission)

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
            return precompute_refusal(phases, admission)
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
    replay_index = 0

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

    def run_one(recipe: Any) -> dict[str, Any]:
        """Execute one attempt, journaling its start and its end.

        The start line goes down *before* the executor is invoked: an attempt
        that dies hard -- a killed subprocess tree, a killed process, a power
        cut, anything no Python exit path can intercept -- still leaves who it
        was and that it was running on the journal.
        """
        nonlocal replay_index
        index = replay_index
        spec_digest = content_digest({
            "manifest_digest": replay_identity,
            "recipe": asdict(recipe),
            "curriculum_ids": [item.item_id for item in plan.items],
            "index": index,
        })
        if index < len(replay):
            receipt = replay[index]
            if receipt.get("spec_digest") != spec_digest:
                raise CampaignRunRefusal(
                    "CAMPAIGN_RESUME_UNVERIFIED: planned recipe/round/checkpoint "
                    "identity differs from the durable completed attempt"
                )
            evidence = dict(receipt["evidence"])
            validate_resume_evidence(root, evidence)
            if isinstance(executor, _BackendStampedTrainingFn):
                expected_backend = {
                    **executor.__dict__["_declared"],
                    "version": executor.__dict__["_version"],
                }
                if evidence.get("backend") != expected_backend:
                    raise CampaignRunRefusal(
                        "CAMPAIGN_RESUME_UNVERIFIED: recorded attempt backend "
                        "does not match the current declared backend"
                    )
            replay_index += 1
            return evidence
        if interrupted and index != len(replay):
            raise CampaignRunRefusal("CAMPAIGN_RESUME_UNVERIFIED: attempt order changed")
        if interrupted:
            # Admission projected the *original* complete campaign before
            # training. A resumed leg needs a fresh headroom check after the
            # actual spend of its replayed attempts has been restored.
            spent = ledger.total()
            if (
                spent.wall_gpu_hours + float(recipe.projected_wall_gpu_hours)
                > manifest.budget.wall_gpu_hours_ceiling_campaign + 1e-12
                or (
                    manifest.budget.device_time_measured
                    and spent.device_gpu_hours + float(recipe.projected_device_gpu_hours)
                    > manifest.budget.device_gpu_hours_ceiling_campaign + 1e-12
                )
            ):
                raise CampaignRunRefusal(
                    "CAMPAIGN_RESUME_BUDGET: insufficient unspent declared "
                    "campaign ceiling for another attempt"
                )
        # Choose the proof contract BEFORE executing any trainer. Never infer
        # native-vs-remote proof from a mutable completion event afterward.
        native_receipt = (
            train_fn is None
            and isinstance(getattr(executor, "_inner", None), SubprocessTrainingFn)
        )
        proof_mode = "receipt" if native_receipt else "completion"
        _journal_event(
            root, phase="attempt", status="started", recipe_id=recipe.recipe_id,
            identity=spec_digest, index=index, proof_mode=proof_mode,
        )
        evidence = dict(executor(recipe, plan.items))
        evidence["recipe_id"] = recipe.recipe_id
        # Caller-provided executors are an open extension seam. Their records
        # may lack the on-disk attempt evidence contract, so they continue to
        # settle normally but never become replayable. A production attempt
        # refused before training also has no GPU cost to checkpoint.
        replayable = native_receipt and isinstance(evidence.get("actual_cost"), Mapping)
        # A native executor can refuse during template/data/admission preflight
        # *before* any trainer process is started. That legitimately has no
        # paid native receipt and must remain recordable as REFUSED, not raise
        # as though paid GPU evidence disappeared. It does not become an
        # accepted native settlement proof or resumable trained attempt.
        harmless_precompute_refusal = (
            evidence.get("status") == "REFUSED"
            and evidence.get("train_started") is False
            and evidence.get("actual_cost") is None
            and evidence.get("measured_gpu_hours") is None
        )
        if native_receipt and not replayable and not harmless_precompute_refusal:
            # Local paid training cannot be relabeled as a remote completion
            # when its durable native cost/evidence receipt is missing.
            raise CampaignRunRefusal(
                "CAMPAIGN_NATIVE_PROOF_MISSING: local training has no native "
                "priced receipt; reconcile spend before any settlement"
            )
        if not replayable:
            # Remote/custom executors have no native, byte-bound local
            # checkpoint receipt. Give their terminal journal entry a distinct
            # invocation identity and a digest of the charge actually settled,
            # rather than pretending they have a replayable native receipt.
            # An unpriced started attempt still raises and leaves an open
            # journal; no fabricated zero-cost completion is recorded.
            charged = _attempt_cost(evidence)
            completion_digest = content_digest({
                "identity": spec_digest,
                "recipe_id": recipe.recipe_id,
                "attempt": evidence.get("attempt"),
                "status": evidence.get("status"),
                "cost": charged.to_dict(),
            })
            _journal_event(
                root, phase="attempt", status=evidence.get("status"),
                recipe_id=recipe.recipe_id, attempt=evidence.get("attempt"),
                gpu_hours=evidence.get("measured_gpu_hours"),
                identity=spec_digest, index=index, detail=completion_digest,
            )
            replay_index += 1
            return evidence
        # Price the finished production invocation *before* its receipt
        # becomes replayable. Unknown cost leaves it unpaired and blocks entry.
        _attempt_cost(evidence)
        try:
            evidence_digest = commit_receipt(
                root, index=index, spec_digest=spec_digest, evidence=evidence
            )
        except ResumeRefusal as error:
            raise CampaignRunRefusal(str(error)) from error
        _journal_event(
            root, phase="attempt_receipt", status=evidence.get("status"),
            recipe_id=recipe.recipe_id, attempt=evidence.get("attempt"),
            gpu_hours=evidence.get("measured_gpu_hours"), identity=spec_digest,
            index=index, detail=evidence_digest,
        )
        replay_index += 1
        return evidence

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
            run_attempt=run_one,
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
            evidence = run_one(recipe)
            results.append(evidence)
            charge(evidence)
            stopped_by = overrun()
            if stopped_by:
                break
    if replay_index < len(replay) and interrupted:
        raise CampaignRunRefusal(
            "CAMPAIGN_RESUME_UNVERIFIED: journal holds attempts outside the "
            "presently admitted plan or budget; refusing uncharged evidence"
        )
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
        _journal_event(
            root, phase="candidate_evaluation_started", status="started",
            detail="external candidate evaluator may have spent compute",
        )
        evaluation_started = time.perf_counter()
        request, evaluation = _evaluate_candidate(
            manifest, root=root, selected=selected, eval_fn=evaluator
        )
    except CampaignRunRefusal as refusal:
        if "evaluation_started" in locals():
            # The evaluator may have performed partial real work before failing
            # to emit a trustworthy cost/result. Charge the observed wall
            # interval conservatively, mark device-hours unmeasured, and refuse.
            # A missing evaluation summary is NEVER evidence of free compute.
            elapsed = max(0.0, time.perf_counter() - evaluation_started)
            declared_cost = getattr(refusal, "measured_evaluation_cost", None)
            cost = (
                declared_cost
                if isinstance(declared_cost, ComputeCost)
                else ComputeCost.from_wall_only(
                    elapsed / 3600.0,
                    source="failed-candidate-evaluation:subprocess-wall-fallback",
                )
            )
            ledger.add(
                "failed candidate evaluation (reported cost or observed wall fallback)",
                "evaluation",
                cost,
                notes=(
                    "the reported evaluation cost is independently valid"
                    if declared_cost is not None
                    else "partial evaluation elapsed wall time; device use unmeasured"
                ),
            )
            _journal_event(
                root,
                phase="candidate_evaluation_cost",
                status=(
                    "reported_cost_before_refusal"
                    if declared_cost is not None else "fallback_unmeasured_device"
                ),
                gpu_hours=cost.wall_gpu_hours,
                detail="evaluation failure charged; no promotion allowed",
            )
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

    # The promotion policy's PROMOTED is *candidate qualification*, not Nik's
    # permission to advance a generation. A denied/headless prompt preserves
    # every measured arm and the mechanical verdict in `promotion`, but no
    # promotion transaction or lineage record may be written. Never accept an
    # environment variable as an approval switch, even during tests.
    owner_grant = None
    if verdict == "PROMOTED" and requires_owner_approval(manifest.base_model_path):
        artifact_sha = str((selected or {}).get("artifact_sha256") or "")
        judge_passed = judge_certifies_candidate(root)
        if judge_passed:
            owner_grant = authorize_promotion(
                cycle_id=manifest.cycle_id,
                candidate_version=candidate_version,
                artifact_sha256=artifact_sha,
                judge_certified=True,
            )
        approved = judge_passed and verify_grant(
            owner_grant, cycle_id=manifest.cycle_id,
            candidate_version=candidate_version, artifact_sha256=artifact_sha,
        )
        phases.append({
            "phase": "human_approval",
            "verdict": "approved" if approved else "pending",
            "cycle_id": manifest.cycle_id,
            "candidate_version": candidate_version,
            "artifact_sha256": artifact_sha,
            "frozen_judge_passed": judge_passed,
            "detail": (
                f"operator explicitly approved {candidate_version} artifact sha256 {artifact_sha}"
                if approved else "HUMAN_APPROVAL_REQUIRED: qualified candidate is not authorized "
                "for generation advancement; frozen REAP judge certification and subsequent "
                "owner approval are both required; no lineage will be recorded"
            ),
        })
        if not approved:
            verdict = "INCONCLUSIVE"
            decision = replace(
                decision, verdict=verdict,
                reasons=tuple(decision.reasons) + ("HUMAN_APPROVAL_REQUIRED",),
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
    finalized_decision = _with_resource_veto(decision, settlement)
    if run.verdict == "PROMOTED":
        if finalized_decision.verdict != "PROMOTED":
            raise CampaignRunRefusal("PROMOTION_TRANSACTION_MISMATCH: resource veto not reflected in run")
        # This fsynced PREPARED intent precedes every lineage mutation. A raw
        # ledger entry written afterward stays invisible until the campaign's
        # own terminal journal and matching JSON verdict are durably published.
        prepare_transaction(
            root, cycle_id=manifest.cycle_id, candidate_version=candidate_version,
            run_document=run.to_dict(), promotion=finalized_decision.to_dict(),
            manifest_digest=declaration_digest,
        )
        _journal_event(root, phase="promotion_prepare", status="PREPARED")
    outcome = _finalize(
        manifest,
        cycle=cycle,
        decision=finalized_decision,
        selected=selected,
        root=root,
        # The report this run produced, at the path the judge reads: the lineage
        # names the measurement that decided it, not an input it was handed.
        evaluation_report_ref=str(root / CERTIFICATION_EVIDENCE["candidate"]),
        # The ledger entry itself names the backend that produced the
        # generation: a reader sees which generator produced it without
        # joining this run record.
        backend_record=backend_record,
        owner_approval=owner_grant,
    )
    saved = _write_record(run, root, outcome=outcome)
    if run.verdict == "PROMOTED":
        committed = reconcile_transaction(
            root, cycle_id=manifest.cycle_id, candidate_version=candidate_version,
            journal_verdict=_journal_outcome(_journal_path(root)),
            manifest_digest=declaration_digest,
        )
        if committed is None:
            raise CampaignRunRefusal(
                "PROMOTION_TRANSACTION_UNRESOLVED: campaign terminal and lineage "
                "could not be bound; promotion remains unavailable"
            )
    return saved


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
            # The run's own validated evaluation object rather than a rebuilt
            # one: it is the object whose measured cost the ledger charged.
            assert candidate is not None  # the only source-less arm
            write_candidate_evaluation(
                candidate, root=root, name=CERTIFICATION_EVIDENCE[arm]
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
                if source is None:
                    # The candidate arm is produced by this run, so it has no
                    # excuse for pointing outside it: a certified verdict that
                    # depends on an external directory can be destroyed by
                    # deleting that directory.
                    raise CampaignRunRefusal(
                        f"the candidate arm's {run.benchmark_qualified_id} row "
                        f"names absolute artifact {reference!r}; the run's own "
                        "measurement must live inside the run root, so the "
                        "evidence a verdict rests on cannot be moved out from "
                        "under it"
                    )
                # A declared arm may name an absolute artifact; the judge
                # hashes it there, so the dependence is recorded below.
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
    detail = f"base model-content digest matches {manifest.base_model_digest[:12]}"
    if manifest.has_parent_adapter():
        detail += (
            f"; parent adapter {manifest.parent_adapter_digest[:12]} verified "
            f"over base for {manifest.parent_version}"
        )
    else:
        detail += f"; {manifest.parent_version} is the base itself (no adapter declared)"
    return detail


def _verify_base_identity(manifest: CampaignManifest) -> str:
    """The dense base must hash, on the frozen model-content basis, as declared.

    A base is a HuggingFace directory that acquires a download cache
    (``.cache/huggingface/**``) and provenance-only files (``README.md``,
    ``.gitattributes``) after it is downloaded. A whole-tree
    :func:`directory_digest` therefore is not a stable identity for a base: it
    moves whenever the cache moves, while nothing about the model changed. The
    frozen Gen-0 identity is a *model-content* digest over the payload files
    only (:func:`chowder.local_model_manifest.model_content_digest`), and that is
    the basis a base digest must be pinned to.

    The adapter tree keeps the directory digest: it is a small byte-stable
    directory a run writes itself, with no cache in it.
    """
    path = Path(manifest.base_model_path)
    if not path.is_dir():
        raise CampaignRunRefusal(
            f"base_model_path {str(path)!r} is not an existing directory "
            "(the dense base tree)"
        )
    try:
        measured = model_content_digest(path)
    except LocalModelManifestError as error:
        raise CampaignRunRefusal(
            f"base_model_digest cannot be verified: {error}"
        ) from error
    if measured.digest != manifest.base_model_digest:
        raise CampaignRunRefusal(
            f"base_model_digest {manifest.base_model_digest[:12]} does not match "
            f"the model-content digest {measured.digest[:12]} of {path} "
            f"(basis {measured.basis}, {len(measured.files)} payload files); the "
            "campaign would train from a base other than the one it preregistered"
        )
    return (
        f"base model-content digest matches {manifest.base_model_digest[:12]} "
        f"over {len(measured.files)} payload files (basis {measured.basis})"
    )


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

    A base digest and an adapter digest identify different objects on different
    bases, so each is checked against its own tree and its own construction.
    Overloading one field to mean either is what made the gen2 manifest
    ambiguous; overloading one basis is the same defect one level down.
    """
    _verify_base_identity(manifest)
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


#: The declared parent profile is not the evidence-attributed profile a
#: curriculum may be planned from. The shape a preparation pass used to write --
#: every known skill carrying the mean of whatever was measured -- cannot say
#: which skills have evidence, so planning from it trains against numbers no
#: benchmark produced.
PARENT_PROFILE_NOT_ATTRIBUTED = "PARENT_PROFILE_NOT_ATTRIBUTED"


def _load_profile(manifest: CampaignManifest) -> CapabilityProfile:
    """The parent's attributed capability profile, as the curriculum's view.

    The authoritative document is a ``SkillProfile`` (benchmark-attributed, with
    unknown capabilities left unknown); the curriculum engine reads the flat
    ``CapabilityProfile`` shape, so the view is derived here through the explicit
    adapter in :mod:`chowder.growth.capability` rather than by a second profile
    computation. A legacy flat document is **refused by name**: accepting one
    would silently put an un-attributed mean back in front of the planner, which
    is precisely the defect this shape replaces.
    """
    from chowder.growth.target_selection import SkillProfile

    path = _require_path(
        manifest.parent_profile_path,
        "parent_profile_path",
        purpose="a curriculum cannot be planned from nothing",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise CampaignRunRefusal(f"parent profile {path} is not a JSON object")
    if not isinstance(payload.get("estimates"), list):
        raise CampaignRunRefusal(
            f"{PARENT_PROFILE_NOT_ATTRIBUTED}: the parent profile {path} carries "
            "no benchmark-attributed estimates, so which capability each number "
            "belongs to cannot be recovered; prepare the campaign again rather "
            "than planning a curriculum from an unattributed mean"
        )
    return CapabilityProfile.from_skill_profile(
        SkillProfile.from_dict(dict(payload)),
        model_version=str(manifest.parent_version),
    )


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
        model_lifecycle=_model_lifecycle(document.get("model_lifecycle"), path),
    )


def _model_lifecycle(value: Any, path: Path) -> Mapping[str, Any] | None:
    """The hardware budget's measured model lifecycle, refused if malformed."""
    if value is None:
        return None
    required = ("train_seconds_per_step", "measured_seq_len", "in_run_eval_seconds")
    if not isinstance(value, Mapping) or any(
        not isinstance(value.get(key), (int, float)) or isinstance(value.get(key), bool)
        or value.get(key) <= 0
        for key in required
    ):
        raise CampaignRunRefusal(
            f"hardware budget {path} declares a model_lifecycle without positive "
            f"{', '.join(required)}"
        )
    return dict(value)


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
        # The declared evaluation placement, or the evaluator's own historical
        # default (resident) when the campaign declares none: an MoE whose
        # experts do not fit declares ``experts-cpu`` here and the candidate is
        # measured under the same placement the declaration asked for.
        placement=str(manifest.evaluation_execution.placement).strip() or "resident",
        # The declared execution throughput, so the candidate arm is measured
        # the same way the parent and ancestor arms were declared to be.
        batch_size=manifest.evaluation_execution.batch_size,
        # The same bound an attempt gets. The binding's 3600 s default is below
        # what the declared suites cost: the parent arm itself measured 3688 s,
        # and Gen-2 run 7's candidate arm (2026-10-18) was killed at 3600 s with
        # one of three suites written, after both recipes had trained, evaluated
        # and settled.
        timeout_seconds=attempt_timeout_seconds(manifest),
        # Resolved here, in this module's namespace, so one patch point covers
        # every process this campaign starts (trainer and evaluator alike).
        runner=runner or default_runner,
    )


def build_local_training_fn(
    manifest: CampaignManifest,
    *,
    state_root: str | Path | None = None,
    runner: Any = None,
) -> SubprocessTrainingFn:
    """Build the production subprocess executor from the declared inputs.

    The single execution path behind every backend that runs training in
    process (``local`` and ``unsloth``): the declared project template decides
    the trainer engine, and this binding materializes, validates, trains and
    settles against one envelope.
    """
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
        # measurements already run under, with margin over that sum. A larger
        # declared envelope raises it: settlement refuses anything over the
        # per-recipe wall ceiling anyway, so the timeout only has to stop a
        # runaway, never an attempt that would settle (ceiling x 1.25, the
        # settlement tolerance).
        timeout_seconds=attempt_timeout_seconds(manifest),
    )


def attempt_timeout_seconds(manifest: CampaignManifest) -> float:
    """One attempt's process timeout: never below 7200 s, and never below the
    declared per-recipe wall ceiling at the settlement tolerance."""
    return max(7200.0, float(manifest.budget.wall_gpu_hours_ceiling_per_recipe) * 3600.0 * 1.25)


class _BackendStampedTrainingFn:
    """A ``TrainingFn`` that names its backend in every attempt's evidence.

    Which backend produced an attempt is a fact *about that attempt*, so it
    travels with the attempt's own evidence instead of being reconstructed from
    the run record. Everything else is delegated untouched: this wrapper adds a
    fact, it does not change what runs, what is admitted or what is measured.
    """

    def __init__(
        self, inner: Any, *, declared: Mapping[str, Any], version: str
    ) -> None:
        self.__dict__["_inner"] = inner
        self.__dict__["_declared"] = dict(declared)
        self.__dict__["_version"] = str(version)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.__dict__["_inner"], name)

    def admit(self, recipe: TrainingRecipe) -> tuple[str, str] | None:
        return self.__dict__["_inner"].admit(recipe)

    def __call__(
        self, recipe: TrainingRecipe, items: Sequence[Any]
    ) -> Mapping[str, Any]:
        evidence = self.__dict__["_inner"](recipe, items)
        if not isinstance(evidence, Mapping):
            return evidence
        stamped = dict(evidence)
        stamped["backend"] = {
            **self.__dict__["_declared"],
            "version": self.__dict__["_version"],
        }
        return stamped


def _declared_backend_version() -> str:
    """The chowder version an attempt's evidence names as its backend version."""
    from .training_backends import chowder_version

    return chowder_version()


def build_executor_with_selection(
    manifest: CampaignManifest,
    *,
    state_root: str | Path | None = None,
    runner: Any = None,
    providers: Any = None,
    probes: Any = None,
) -> tuple[Any, Mapping[str, Any]]:
    """Build the executor the manifest's declared training backend dispatches through.

    One branch point, at the backend boundary: the declared provider decides
    which executor is built and how, so nothing downstream needs to know which
    one it was. A *declaration* error -- an unknown provider, an unsupported
    config key, a template whose trainer is not the declared one, an Unsloth
    knob the isolated engine refuses, an incomplete remote wiring -- stops the
    campaign here, before any compute. A hardware fact is not decided here: the
    preflight panel reports it, and the executor's own admission plus the
    trainer's own config resolution are where it is enforced.

    Returns the executor -- wrapped so every attempt's evidence names the
    backend that produced it -- and the record of the choice: the declaration,
    the provider and trainer, the full preflight, and, when ``auto`` chose, the
    reason, the cost comparison and every candidate it refused. The run writes
    that record as its own phase before any attempt starts.

    ``providers`` and ``probes`` are injectable exactly like ``runner``: the
    no-GPU harness supplies them, production never does.
    """
    from .compute_backend import ComputeBackendRefusal
    from .training_backends import (
        STRUCTURAL_PREFLIGHT_CODES,
        TrainingBackendRefusal,
        backend_declaration,
        resolve_training_backend,
    )

    declaration = backend_declaration(manifest)
    try:
        provider, selection = resolve_training_backend(
            declaration,
            manifest,
            providers=providers,
            probes=probes,
        )
        preflight = provider.preflight(manifest)
    except TrainingBackendRefusal as refusal:
        raise CampaignRunRefusal(f"{refusal.code}: {refusal.reason}") from refusal
    if not preflight.admitted and preflight.code in STRUCTURAL_PREFLIGHT_CODES:
        raise CampaignRunRefusal(f"{preflight.code}: {preflight.reason}")
    try:
        inner = provider.build_training_fn(
            manifest, state_root=state_root, runner=runner
        )
    except (TrainingBackendRefusal, ComputeBackendRefusal) as refusal:
        # One refusal vocabulary at the runner boundary, whatever layer found
        # the problem; the machine-readable code travels in the message.
        raise CampaignRunRefusal(f"{refusal.code}: {refusal.reason}") from refusal
    record: dict[str, Any] = {
        "cycle_id": manifest.cycle_id,
        "declaration": declaration.to_dict(),
        "provider": provider.provider,
        "trainer": provider.trainer,
        "strategy": preflight.strategy,
        "admitted": preflight.admitted,
        "refused_by": None if preflight.admitted else (preflight.code or ""),
        "refusal_reason": preflight.reason,
        # A declaration error stops the run before this point; what is recorded
        # here is whether a *reported* hardware fact was the reason.
        "stops_the_run": (
            not preflight.admitted and preflight.code in STRUCTURAL_PREFLIGHT_CODES
        ),
        "panel": preflight.panel.to_dict(),
        "overrides": list(preflight.overrides),
        "selection": selection.to_dict() if selection is not None else None,
    }
    return (
        _BackendStampedTrainingFn(
            inner,
            declared={
                "provider": provider.provider,
                "trainer": provider.trainer,
                "declaration": declaration.to_dict(),
            },
            version=_declared_backend_version(),
        ),
        record,
    )


def build_executor(
    manifest: CampaignManifest,
    *,
    state_root: str | Path | None = None,
    runner: Any = None,
    providers: Any = None,
    probes: Any = None,
) -> Any:
    """Build the executor the declared training backend dispatches through."""
    executor, _selection = build_executor_with_selection(
        manifest,
        state_root=state_root,
        runner=runner,
        providers=providers,
        probes=probes,
    )
    return executor


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


def _attempt_cost(evidence: Mapping[str, Any]) -> ComputeCost:
    recorded = evidence.get("actual_cost")
    if isinstance(recorded, Mapping):
        return ComputeCost.from_dict(recorded)
    measured = evidence.get("measured_gpu_hours")
    if measured is None and evidence.get("train_started") is True:
        # An executed trainer with no valid cost has UNKNOWN spend, not zero.
        # Raising here leaves the durable campaign journal open, so re-entry
        # refuses until an operator reconciles the actual GPU time. No further
        # recipe or downstream promotion may run on unverifiable accounting.
        raise CampaignRunRefusal(
            "ACTUAL_WALL_GPU_HOURS_UNMEASURED: trainer started but neither "
            "a trustworthy summary nor subprocess wall-time measurement "
            "was available; reconcile this attempt before resuming"
        )
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
        wrapped = CampaignRunRefusal(str(refusal))
        # A report can be invalid for provenance/identity while its cost is
        # still a valid measurement. Preserve that cost rather than replacing
        # it with a second, timer-derived estimate.
        if isinstance(returned, CandidateEvaluation):
            try:
                wrapped.measured_evaluation_cost = validate_evaluation_cost(returned)
            except CandidateEvaluationRefusal:
                pass
        raise wrapped from refusal


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
        # Same rule for the device unit: the settled reading is the one the
        # declared ceiling is actually enforced against, and it is forwarded
        # only when the budget declared device time measurable. An unmeasured
        # device figure stays absent rather than reported as a measured zero.
        actual_device_gpu_hours=(
            total.device_gpu_hours if device_settleable and total.device_measured else None
        ),
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


def _backend_lineage(backend_record: Mapping[str, Any] | None) -> dict[str, Any]:
    """The backend provenance a generation's ledger entry carries.

    ``provider`` is the one whose executor actually ran; ``declared`` is the
    declaration as written (for ``auto``, which candidate was chosen and why
    is already inside the selection). A caller-supplied executor is recorded
    as exactly that -- declared, but not executed by a backend this build
    dispatched. The evidence stamp every attempt carries is the per-attempt
    version of this same fact; this is the generation-level one.
    """
    if not backend_record:
        return {"provider": "undeclared"}
    declared = backend_record.get("declaration")
    provider = backend_record.get("provider")
    entry: dict[str, Any] = {"declared": dict(declared) if isinstance(declared, Mapping) else {}}
    if provider is None:
        entry["provider"] = "caller-supplied"
        return entry
    entry["provider"] = str(provider)
    entry["trainer"] = str(backend_record.get("trainer") or "")
    entry["selection"] = backend_record.get("selection")
    return entry


def _finalize(
    manifest: CampaignManifest,
    *,
    cycle: GrowthCycle,
    decision: Any,
    selected: Mapping[str, Any] | None,
    root: Path,
    evaluation_report_ref: str,
    backend_record: Mapping[str, Any] | None = None,
    owner_approval: Any = None,
) -> CycleOutcome:
    return cycle.finalize(
        decision,
        base_model={
            **manifest.model_identity(),
            "version": manifest.parent_version,
            "lineage": (
                "reap-moe" if requires_owner_approval(manifest.base_model_path) else "primary"
            ),
        },
        dataset_manifest_ref=selected.get("evidence_path", "") if selected else "",
        curriculum_manifest_ref=str(root / "cycle_compute_accounting.json"),
        recipe={"recipe_id": selected.get("recipe_id")} if selected else {},
        training_evidence_ref=selected.get("evidence_path", "") if selected else "",
        evaluation_report_ref=evaluation_report_ref,
        notes=f"campaign {manifest.cycle_id} (policy {PROMOTION_POLICY_VERSION})",
        backend=_backend_lineage(backend_record),
        owner_approval=owner_approval,
        candidate_artifact_sha256=str((selected or {}).get("artifact_sha256") or ""),
    )


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


def _backend_detail(record: Mapping[str, Any]) -> str:
    """One line naming the backend an attempt will run on, and its standing."""
    provider = record.get("provider")
    if provider is None:
        declared = record.get("declaration")
        declared_provider = (
            declared.get("provider") if isinstance(declared, Mapping) else None
        )
        return (
            f"declared {declared_provider or '<undeclared>'}; the executor was "
            "supplied by the caller, so no provider preflight was run"
        )
    if record.get("admitted"):
        return (
            f"declared {provider} ({record.get('trainer')}), preflight admitted"
            + (f": {record.get('strategy')}" if record.get("strategy") else "")
        )
    return (
        f"declared {provider} ({record.get('trainer')}): preflight not admitted "
        f"({record.get('refused_by')}: {record.get('refusal_reason')})"
        + (
            " -- a declaration error the run will refuse"
            if record.get("stops_the_run")
            else " -- a hardware fact, reported rather than enforced"
        )
    )


def _attempt_summary(results: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    """The durable facts about what each attempt produced, refusal included."""
    return tuple(
        {
            "recipe_id": evidence.get("recipe_id"),
            # Which backend produced this attempt, from the attempt's own
            # evidence (the runner stamps it; the executor is the one that ran).
            "backend": evidence.get("backend"),
            "attempt": evidence.get("attempt"),
            "status": evidence.get("status"),
            "refused_by": evidence.get("refused_by"),
            "refusal_reason": evidence.get("refusal_reason"),
            "failure_reason": evidence.get("failure_reason"),
            "artifact_ref": evidence.get("artifact_ref"),
            "artifact_sha256": evidence.get("artifact_sha256"),
            "measured_gpu_hours": evidence.get("measured_gpu_hours"),
            # Absent for a single-pass campaign; the round an attempt belonged
            # to when a declared search ran it.
            "search_round": evidence.get("search_round"),
            "search_max_steps": evidence.get("search_max_steps"),
        }
        for evidence in results
    )


def _verify_selected_artifact(
    selected: Mapping[str, Any] | None, *, purpose: str
) -> None:
    """Re-read the selected artifact and require its bytes to be unchanged.

    A digest recorded at training time is a claim about bytes that may since
    have moved. Production re-derives it from disk at both boundaries that
    matter -- before the artifact is measured, and before a verdict is recorded
    -- so a run can never certify a digest the frozen judge would later
    recompute and reject.
    """
    if not selected:
        return
    artifact_ref = str(selected.get("artifact_ref") or "")
    recorded = str(selected.get("artifact_sha256") or "")
    if not artifact_ref or not recorded:
        return
    path = Path(artifact_ref)
    if not path.exists():
        raise CampaignRunRefusal(
            f"{CANDIDATE_ARTIFACT_DIGEST_STALE}: the selected artifact "
            f"{artifact_ref!r} no longer exists, so there is nothing to verify "
            f"{purpose}"
        )
    observed = digest_of(path)
    if observed != recorded:
        raise CampaignRunRefusal(
            f"{CANDIDATE_ARTIFACT_DIGEST_STALE}: the selected artifact "
            f"{artifact_ref!r} hashes to {observed}, but the run recorded "
            f"{recorded}; the bytes changed, so nothing measured against the "
            f"recorded digest can be trusted {purpose}"
        )


def _preflight_arms(manifest: CampaignManifest) -> str:
    """Load the declared evidence arms before any compute, or say what is wrong.

    Everything here is knowable without touching a GPU: whether the parent and
    trusted-ancestor reports exist and parse, and whether the contamination
    evidence the binder already loaded is a real declaration. A run that would
    refuse on them should refuse before it trains, not after.
    """
    checked: list[str] = []
    for arm, field_name in _EVIDENCE_ARM_SOURCES.items():
        declared = str(getattr(manifest, field_name))
        if not declared:
            raise CampaignRunRefusal(
                f"the campaign declares no {field_name}, so the {arm} arm of "
                "adjudication and branch protection cannot be read; the run "
                "refuses before compute rather than after it"
            )
        source = _require_path(
            declared,
            field_name,
            purpose=f"the {arm} arm is read before compute",
        )
        try:
            report = EvalReport.load(source)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise CampaignRunRefusal(
                f"the declared {arm} arm {source} is unreadable: {error}; a run "
                "cannot adjudicate against evidence it cannot parse"
            ) from error
        if not report.runs:
            raise CampaignRunRefusal(
                f"the declared {arm} arm {source} carries no measured rows, so it "
                "cannot be the evidence this run adjudicates against"
            )
        checked.append(f"{arm}={report.generation_version or '(unlabelled)'}/{len(report.runs)} rows")
    return ", ".join(checked)


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
READINESS_SCHEMA = "READINESS_SCHEMA"
READINESS_DECLARED_INPUT = "READINESS_DECLARED_INPUT"
READINESS_BASE_IDENTITY = "READINESS_BASE_IDENTITY"
READINESS_PARENT_ADAPTER_IDENTITY = "READINESS_PARENT_ADAPTER_IDENTITY"
READINESS_CONTAMINATION = "READINESS_CONTAMINATION"
READINESS_PROJECT_TEMPLATE = "READINESS_PROJECT_TEMPLATE"
READINESS_TRAINING_MATERIAL = "READINESS_TRAINING_MATERIAL"
READINESS_DATA_REGISTRY = "READINESS_DATA_REGISTRY"
READINESS_HARDWARE_BUDGET = "READINESS_HARDWARE_BUDGET"
READINESS_PARENT_PROFILE = "READINESS_PARENT_PROFILE"
READINESS_PARENT_ARM = "READINESS_PARENT_ARM"
READINESS_ANCESTOR_ARM = "READINESS_ANCESTOR_ARM"
READINESS_PROTECTION_POLICY = "READINESS_PROTECTION_POLICY"
READINESS_PLAN = "READINESS_PLAN"
READINESS_RECIPE_SET = "READINESS_RECIPE_SET"
READINESS_CANDIDATE_SEARCH = "READINESS_CANDIDATE_SEARCH"
READINESS_CAMPAIGN_PROJECTION = "READINESS_CAMPAIGN_PROJECTION"
READINESS_EVALUATOR = "READINESS_EVALUATOR"
READINESS_EVALUATOR_COVERAGE = "READINESS_EVALUATOR_COVERAGE"
READINESS_CAMPAIGN_STATE = "READINESS_CAMPAIGN_STATE"
READINESS_EXECUTOR_ADMISSION = "READINESS_EXECUTOR_ADMISSION"


@dataclass(frozen=True)
class ReadinessCheck:
    """One pre-compute prerequisite: ``ok``, ``refused`` or ``skipped``.

    ``skipped`` means a check it depends on did not pass, so evaluating it
    would report the same missing input a second time; it is not a pass.
    """

    check: str
    status: str
    detail: str
    reason_code: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "check": self.check,
            "status": self.status,
            "detail": self.detail,
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True)
class ReadinessReport:
    """The structured answer to "may this campaign start?", with no compute."""

    cycle_id: str
    checks: tuple[ReadinessCheck, ...]

    @property
    def ready(self) -> bool:
        """READY only when every prerequisite passed; a skip is not a pass."""
        return all(check.status == "ok" for check in self.checks)

    @property
    def status(self) -> str:
        return "READY" if self.ready else "REFUSED"

    @property
    def reason_codes(self) -> tuple[str, ...]:
        return tuple(
            check.reason_code
            for check in self.checks
            if check.status == "refused" and check.reason_code
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "status": self.status,
            "checks": [check.to_dict() for check in self.checks],
            "reason_codes": list(self.reason_codes),
        }


def check_campaign_readiness(
    manifest: CampaignManifest,
    *,
    state_root: str | Path | None = None,
    train_fn: Any = None,
    eval_fn: CandidateEvaluator | None = None,
    runner: Any = None,
) -> ReadinessReport:
    """Inspect every pre-compute prerequisite of a declared campaign.

    This starts no process and loads no model: it reads the declaration, the
    files it names and the declared instrument's own admission seam. ``train_fn``
    and ``eval_fn`` are the same seams a run would use -- supplying them lets a
    harness ask "would this campaign be ready with *these* seams?" -- and both
    default to the production ones.

    Each check is independent of the others' *results* but not of their
    prerequisites: a check whose prerequisite failed is reported ``skipped``
    rather than re-raising the same missing input. A refusal never becomes a
    default here, exactly as it never does in a run.
    """
    root = Path(state_root or manifest.state_root)
    results: dict[str, ReadinessCheck] = {}
    order: list[str] = []

    def check(name: str, code: str, requires: tuple[str, ...], probe: Any) -> Any:
        order.append(name)
        for required in requires:
            if results[required].status != "ok":
                results[name] = ReadinessCheck(
                    name,
                    "skipped",
                    f"not evaluated: {required} did not pass",
                )
                return None
        try:
            detail = probe()
        except CampaignRunRefusal as refusal:
            results[name] = ReadinessCheck(name, "refused", str(refusal), code)
            return None
        except Exception as error:  # a malformed document, a bad digest, ...
            results[name] = ReadinessCheck(
                name, "refused", f"{type(error).__name__}: {error}", code
            )
            return None
        results[name] = ReadinessCheck(name, "ok", detail)
        return detail

    def plan_of() -> CampaignPlan:
        return plan_campaign(manifest)

    def recipes_of() -> tuple[TrainingRecipe, ...]:
        return _select_recipes(manifest, plan_of().recipes)

    check("schema", READINESS_SCHEMA, (), lambda: (
        assert_every_field_enforced() or "schema matches FIELD_ENFORCEMENT"
    ))

    def declared_source_identity() -> str:
        require_declared_inputs(manifest, phase="run")
        # The run hashes declared input files *before* starting any trainer.
        # Readiness must perform that same preflight, otherwise it can report
        # READY for a nonexistent file while run_campaign correctly refuses.
        try:
            digest = declared_inputs_digest(manifest)
        except ResumeRefusal as error:
            raise CampaignRunRefusal(str(error)) from error
        return f"required inputs declared and source file SHA256 manifest {digest[:12]} verified"

    check("declared_inputs", READINESS_DECLARED_INPUT, ("schema",), declared_source_identity)

    def campaign_state_detail() -> str:
        from chowder.worker_env import chowder_source_identity

        replay_identity = content_digest({
            "declaration": content_digest(asdict(manifest)),
            "declared_files": declared_inputs_digest(manifest),
            "source_sha256": chowder_source_identity()["source_sha256"],
        })
        state, receipts, detail = _campaign_root_state(
            root, replay_identity=replay_identity,
        )
        if state not in ("fresh", "resume"):
            raise CampaignRunRefusal(detail)
        # The runner also holds a kernel lock throughout admission and
        # execution. Probe an existing lock without creating/writing one:
        # readiness is advisory, and the runner acquires it again atomically.
        lock_path = root / ".campaign-writer.lock"
        if lock_path.exists():
            try:
                with lock_path.open("r+b") as handle:
                    handle.seek(0)
                    if os.name == "nt":
                        import msvcrt
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError as error:
                raise CampaignRunRefusal(
                    f"CAMPAIGN_WRITER_BUSY: state root writer is locked: {error}"
                ) from error
        return detail

    check(
        "campaign_state", READINESS_CAMPAIGN_STATE,
        ("declared_inputs",), campaign_state_detail,
    )

    def base_identity() -> str:
        return _verify_base_identity(manifest)

    check("base_identity", READINESS_BASE_IDENTITY, ("schema",), base_identity)

    def parent_identity() -> str:
        if not manifest.has_parent_adapter():
            return f"{manifest.parent_version} is the base itself (no adapter declared)"
        _verify_digest(
            Path(manifest.parent_adapter_path),
            "parent_adapter_digest",
            manifest.parent_adapter_digest,
            of=f"the {manifest.parent_version} adapter tree",
        )
        return (
            f"parent adapter {manifest.parent_adapter_digest[:12]} verified over base "
            f"for {manifest.parent_version}"
        )

    check(
        "parent_adapter_identity",
        READINESS_PARENT_ADAPTER_IDENTITY,
        ("base_identity",),
        parent_identity,
    )

    check("contamination", READINESS_CONTAMINATION, ("declared_inputs",), lambda: (
        f"binder loaded from {manifest.contamination_manifest_path}"
        if _load_binder(manifest) is not None
        else ""
    ))

    def load_template() -> str:
        # Imported here, not at module scope: ``campaign_prepare`` calls back
        # into this module at *call* time, so the dependency is already one-way
        # at import time.
        from .campaign_prepare import CampaignPrepareRefusal, trainer_spec_refusal

        path = _require_path(
            manifest.project_template_path,
            "project_template_path",
            purpose="the executor has no project to compose",
        )
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, Mapping):
            raise CampaignRunRefusal(f"project template {path} is not a JSON object")
        # The worker's own admission, asked before compute: a declared template
        # whose MoE settings contradict each other cannot train, and readiness
        # is the phase whose job is to say so without spending the run.
        try:
            trainer_spec_refusal(document, where=f"project template {path.name}", work_dir=path.parent)
        except CampaignPrepareRefusal as refusal:
            raise CampaignRunRefusal(str(refusal)) from refusal
        return f"project template {path.name} parses"

    check("project_template", READINESS_PROJECT_TEMPLATE, ("declared_inputs",), load_template)

    def load_material_detail() -> str:
        sources, material = _load_material(manifest)
        return f"{len(material)} material items over {len(set(sources.values()))} source(s)"

    check(
        "training_material",
        READINESS_TRAINING_MATERIAL,
        ("declared_inputs",),
        load_material_detail,
    )

    check("data_registry", READINESS_DATA_REGISTRY, ("declared_inputs",), lambda: (
        _load_data_registry(manifest) is not None
        and f"admitted sources parse from {manifest.data_registry_path}"
    ))

    def hardware_detail() -> str:
        budget = _load_hardware_budget(manifest)
        return f"hardware budget for {budget.gpu_name} parses"

    check(
        "hardware_budget", READINESS_HARDWARE_BUDGET, ("declared_inputs",), hardware_detail
    )

    def profile_detail() -> str:
        profile = _load_profile(manifest)
        return f"parent profile for {profile.model_version} parses"

    check(
        "parent_profile", READINESS_PARENT_PROFILE, ("declared_inputs",), profile_detail
    )

    def arm_detail(field: str) -> str:
        runs = _runs_from_report(str(getattr(manifest, field)), field)
        if not runs:
            raise CampaignRunRefusal(
                f"the declared {field} carries no measured rows, so it cannot be "
                "the arm this run adjudicates against"
            )
        versions = sorted({str(run.generation_version) for run in runs})
        detail = f"{field}: {len(runs)} row(s) for {', '.join(versions)}"
        if field != "baseline_eval_report_path":
            return detail
        # The ancestor arm is never re-measured by the run, so the raw artifacts its
        # rows name must already resolve exactly as certification will resolve them:
        # Gen-2 run 10 trained 2.16 GPU-h before certification found the ancestor's
        # predictions missing beside its report (defect #10).
        source = Path(str(getattr(manifest, field)))
        artifacts = _measurement_artifacts(
            [(field, source, EvalReport.load(source))], root=source.parent
        )
        return f"{detail}; {len(artifacts)} raw artifact(s) resolved"

    check("parent_arm", READINESS_PARENT_ARM, ("declared_inputs",), lambda: (
        arm_detail("parent_eval_report_path")
    ))
    check("ancestor_arm", READINESS_ANCESTOR_ARM, ("declared_inputs",), lambda: (
        arm_detail("baseline_eval_report_path")
    ))

    def protection_detail() -> str:
        protection = manifest.protection
        if (
            not protection.trusted_ancestor_version
            or protection.slice_regression_max is None
        ):
            raise CampaignRunRefusal(
                "the campaign declares no protection policy "
                "(trusted_ancestor_version + slice_regression_max + protocol), so "
                "it cannot certify that the branch is protected"
            )
        protocol = protection.require_protocol(source=manifest.cycle_id)
        return (
            f"branch protection against {protection.trusted_ancestor_version} "
            f"(tolerance {protection.slice_regression_max}, {protocol.n_samples} items)"
        )

    check(
        "protection_policy",
        READINESS_PROTECTION_POLICY,
        ("schema",),
        protection_detail,
    )

    def plan_detail() -> str:
        plan = plan_of()
        if not plan.items:
            raise CampaignRunRefusal(
                "the declared parent profile produced no curriculum items, so "
                "there is nothing this campaign may train on"
            )
        return f"{len(plan.items)} curriculum items -> {len(plan.recipes)} proposed recipes"

    check("plan", READINESS_PLAN, ("parent_profile", "hardware_budget"), plan_detail)

    check("recipe_set", READINESS_RECIPE_SET, ("plan",), lambda: (
        f"{len(recipes_of())} declared recipe(s) proposed by the planner"
    ))

    def training_admission_detail() -> str:
        # Mirror the run's provider and per-recipe admission seam before it
        # journals campaign_start or touches an actual training subprocess.
        if train_fn is not None:
            executor = train_fn
        else:
            executor, _selection = build_executor_with_selection(
                manifest, state_root=root, runner=runner,
            )
        admit = getattr(executor, "admit", None)
        if not callable(admit):
            raise CampaignRunRefusal("training executor has no admit(recipe) seam")
        refusals = []
        for recipe in recipes_of():
            refusal = admit(recipe)
            if refusal is not None:
                refusals.append(f"{recipe.recipe_id}: {refusal}")
        if refusals and stops_on_admission_refusal(manifest.stopping_rules):
            raise CampaignRunRefusal("; ".join(refusals))
        if len(refusals) == len(recipes_of()):
            raise CampaignRunRefusal("every declared recipe was refused by its executor")
        return f"{len(recipes_of()) - len(refusals)} recipe(s) admitted by training backend"

    check(
        "executor_admission", READINESS_EXECUTOR_ADMISSION,
        ("recipe_set", "campaign_state"), training_admission_detail,
    )

    def search_detail() -> str:
        plan = plan_of()
        if plan.search_refusal:
            raise CampaignRunRefusal(plan.search_refusal)
        if not plan.search.declared:
            return (
                "no candidate search declared: the declared recipes run once "
                "each"
            )
        return (
            f"{len(plan.search.rounds)} declared round(s), worst case "
            f"{plan.search.total_device_gpu_hours:.6f} device / "
            f"{plan.search.total_wall_gpu_hours:.6f} wall GPU-h within the "
            "declared search envelope"
        )

    check(
        "candidate_search",
        READINESS_CANDIDATE_SEARCH,
        ("recipe_set",),
        search_detail,
    )

    def projection_detail() -> str:
        plan = plan_of()
        recipes = recipes_of()
        # A declared search spends its rounds, so the ceiling has to cover the
        # search's worst-case total rather than a single pass over the recipes.
        projected = (
            ComputeCost(
                device_gpu_hours=plan.search.total_device_gpu_hours,
                wall_gpu_hours=plan.search.total_wall_gpu_hours,
                source=(
                    f"campaign projection ({len(plan.search.rounds)} search "
                    f"rounds over {len(recipes)} recipes)"
                ),
            )
            if plan.search.declared
            else ComputeCost(
                device_gpu_hours=sum(r.projected_device_gpu_hours for r in recipes),
                wall_gpu_hours=sum(r.projected_wall_gpu_hours for r in recipes),
                source=f"campaign projection ({len(recipes)} recipes)",
            )
        )
        verdict = settle_campaign_projection(manifest, projected=projected)
        if not verdict.compliant:
            raise CampaignRunRefusal(
                "; ".join(verdict.failure_reasons)
                or "the planned campaign does not fit its declared envelope"
            )
        return (
            f"planned {projected.device_gpu_hours:.6f} device / "
            f"{projected.wall_gpu_hours:.6f} wall within the declared ceilings"
        )

    check(
        "campaign_projection",
        READINESS_CAMPAIGN_PROJECTION,
        ("recipe_set", "candidate_search"),
        projection_detail,
    )

    def evaluator_of() -> CandidateEvaluator | None:
        return eval_fn if eval_fn is not None else build_evaluator(
            manifest, state_root=root, runner=runner
        )

    def evaluator_detail() -> str:
        evaluator = evaluator_of()
        if evaluator is None:
            raise CampaignRunRefusal(
                f"{CANDIDATE_EVALUATION_NOT_PRODUCED}: no candidate evaluator is "
                "available for this campaign, so nothing could measure the "
                "artifact it is about to train"
            )
        return f"evaluator {type(evaluator).__name__} is available"

    check(
        "evaluator",
        READINESS_EVALUATOR,
        ("declared_inputs",),
        evaluator_detail,
    )

    def coverage_detail() -> str:
        evaluator = evaluator_of()
        assert evaluator is not None  # only reached after the evaluator check
        admit_evaluator = getattr(evaluator, "admit", None)
        if not callable(admit_evaluator):
            return "evaluator exposes no admission seam to check coverage"
        refusal = admit_evaluator(
            benchmarks=tuple(manifest.evaluated_benchmarks),
            protocol=manifest.protection.require_protocol(),
        )
        if refusal is not None:
            raise CampaignRunRefusal("; ".join(str(part) for part in refusal))
        return "evaluator covers every declared benchmark under the declared protocol"

    check(
        "evaluator_coverage",
        READINESS_EVALUATOR_COVERAGE,
        ("evaluator", "protection_policy"),
        coverage_detail,
    )

    return ReadinessReport(
        cycle_id=manifest.cycle_id,
        checks=tuple(results[name] for name in order),
    )


def _write_record(
    run: CampaignRun, root: Path, *, outcome: CycleOutcome | None,
    terminal: bool = True, record_name: str = "campaign-run.json"
) -> CampaignRun:
    document = run.to_dict()
    if outcome is not None:
        document["cycle_outcome"] = outcome.to_dict()
    if record_name not in ("campaign-run.json", "campaign-entry-refusal.json"):
        raise ValueError("campaign record location is not an approved filename")
    path = root / record_name
    # Atomic: a kill mid-write must never leave a truncated JSON verdict
    # behind; the terminal journal line below only goes down once the verdict
    # file is complete.
    tmp = path.with_name(path.name + ".tmp")
    # Make the verdict file's bytes durable before publishing its name. The
    # terminal journal marker must never follow an unflushed verdict file.
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(document, indent=2, sort_keys=True, default=str))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    if terminal:
        # The journal record that closes the campaign: written *after* the
        # verdict file it points at is durable, so a kill before this line
        # leaves the journal open and the state root refusing rather than a
        # verdict nobody's entry check trusts.
        selection = dict(run.selection or {})
        cost = dict(run.cost or {})
        _journal_event(
            root,
            phase="terminal",
            status=run.verdict,
            recipe_id=selection.get("recipe_id"),
            gpu_hours=cost.get("wall_gpu_hours"),
        )
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
