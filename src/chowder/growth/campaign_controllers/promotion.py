"""Identity, adjudication, and the vetoes that come after them.

Identity is verified before compute. ``_adjudicate`` binds the run's own
candidate measurement and lets the declared rule decide -- the run never
re-implements the verdict. ``_with_resource_veto`` applies the campaign's own
resource gate to that decision, so no code path can read a promoted generation
out of a campaign that failed its frozen envelope.

Extracted from ``campaign_runner``, which keeps the orchestration and the
two injection seams. Nothing here re-implements a decision another module owns.
"""

from __future__ import annotations

from typing import Any, Mapping

from dataclasses import dataclass, replace
from pathlib import Path
from chowder.evals.result import BenchmarkRun, EvalReport
from chowder.local_model_manifest import LocalModelManifestError, model_content_digest
from ..campaign import PROMOTION_POLICY_VERSION, CampaignManifest, CampaignManifestError, settle_campaign, settle_campaign_projection, stops_on_admission_refusal, stops_on_campaign_overrun
from ..compute_cost import ComputeCost, CycleCostLedger
from ..cycle import CycleConfig, CycleOutcome, GrowthCycle, TrainingFn
from ..metric_binding import MetricBinder, PromotionAssembly
from ..training_binding import STATUS_SUCCEEDED, GrowthEnvelope, SubprocessTrainingFn, default_runner, directory_digest
from ..campaign_controllers.contracts import CampaignRunRefusal, FIELD_ENFORCEMENT, NON_BEHAVIORAL_FIELDS, assert_every_field_enforced
from ..campaign_controllers.evaluation import PARENT_PROFILE_NOT_ATTRIBUTED, _evaluate_candidate, _load_binder, _load_data_registry, _load_hardware_budget, _load_material, _load_profile, _runs_from_report

from ..contamination import ContaminationFirewall


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
