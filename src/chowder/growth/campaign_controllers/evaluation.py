"""Loading declared inputs into the objects that measure, and asking for the
candidate measurement.

Each ``_load_*`` resolves one declared input through the same refusal, so a
binder is never built from a partially-loaded manifest. ``_evaluate_candidate``
is the only producer of the candidate side of promotion: it asks the injected
evaluator to measure the artifact this run selected, binds the report to that
artifact's digest, and refuses a report that is not a measurement of it.

Extracted from ``campaign_runner``, which keeps the orchestration and the
two injection seams. Nothing here re-implements a decision another module owns.
"""

from __future__ import annotations

from typing import Any, Mapping

import json
from pathlib import Path
from chowder.evals.result import BenchmarkRun, EvalReport
from ..candidate_evaluation import CANDIDATE_EVALUATION_NOT_PRODUCED, CandidateEvaluation, CandidateEvaluationRefusal, CandidateEvaluator, EvaluationRequest, coerce_evaluation, evaluation_detail, validate_candidate_report, validate_evaluation_cost, write_candidate_evaluation
from ..campaign import PROMOTION_POLICY_VERSION, CampaignManifest, CampaignManifestError, settle_campaign, settle_campaign_projection, stops_on_admission_refusal, stops_on_campaign_overrun
from ..capability import CapabilityProfile
from ..data_registry import DataRegistry, DataSource, admit
from ..metric_binding import MetricBinder, PromotionAssembly
from ..recipe_planner import HardwareBudget, RecipePlanner, TrainingRecipe
from ..campaign_controllers.contracts import CampaignRunRefusal, FIELD_ENFORCEMENT, NON_BEHAVIORAL_FIELDS, assert_every_field_enforced
from ..campaign_controllers.declared import DECLARED_INPUT_REQUIREMENTS, _require_path, require_declared_inputs, undeclared_inputs

from ..candidate_search import CandidateSearchRefusal, CandidateSearchDeclaration, SearchPlan, plan_search, run_search
from ..contamination import ContaminationFirewall


def _load_binder(manifest: CampaignManifest) -> MetricBinder:
    from ..catalog import default_registry

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
    from ..campaign_runner import build_evaluator

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


def _runs_from_report(path_value: str, field: str) -> tuple[BenchmarkRun, ...]:
    path = _require_path(path_value, field, purpose="a declared report must exist")
    report = EvalReport.load(path)
    return tuple(report.runs)
