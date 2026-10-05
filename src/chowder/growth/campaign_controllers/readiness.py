"""Whether this campaign may spend compute, answered before it spends any.

Seventeen checks over the declared inputs, identity, arms, plan, recipe set,
search, projection and evaluator coverage. The report is fail-closed by
construction: ``ready`` requires every check ``ok`` and an empty
``reason_codes``, so a new check that fails cannot be forgotten -- it lands in
the reason codes a caller prints.

Extracted from ``campaign_runner``, which keeps the orchestration and the
two injection seams. Nothing here re-implements a decision another module owns.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing import Any, Mapping

import json
from pathlib import Path
from ..candidate_evaluation import CANDIDATE_EVALUATION_NOT_PRODUCED, CandidateEvaluation, CandidateEvaluationRefusal, CandidateEvaluator, EvaluationRequest, coerce_evaluation, evaluation_detail, validate_candidate_report, validate_evaluation_cost, write_candidate_evaluation
from ..campaign import PROMOTION_POLICY_VERSION, CampaignManifest, CampaignManifestError, settle_campaign, settle_campaign_projection, stops_on_admission_refusal, stops_on_campaign_overrun
from ..compute_cost import ComputeCost, CycleCostLedger
from ..recipe_planner import HardwareBudget, RecipePlanner, TrainingRecipe
from ..campaign_controllers.contracts import CampaignRunRefusal, FIELD_ENFORCEMENT, NON_BEHAVIORAL_FIELDS, assert_every_field_enforced
from ..campaign_controllers.declared import DECLARED_INPUT_REQUIREMENTS, _require_path, require_declared_inputs, undeclared_inputs
from ..campaign_controllers.evaluation import PARENT_PROFILE_NOT_ATTRIBUTED, _evaluate_candidate, _load_binder, _load_data_registry, _load_hardware_budget, _load_material, _load_profile, _runs_from_report
from ..campaign_controllers.planning import CampaignPlan, _build_cycle, _select_recipes, envelope_for, plan_campaign, search_plan_for
from ..campaign_controllers.promotion import _adjudicate, _finalize, _identity_detail, _verify_base_identity, _verify_digest, _verify_parent_identity, _with_resource_veto

from ..contamination import ContaminationFirewall


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
    from ..campaign_runner import build_evaluator

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

    check("declared_inputs", READINESS_DECLARED_INPUT, ("schema",), lambda: (
        require_declared_inputs(manifest, phase="run")
        or "every input the run phase requires is declared"
    ))

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
        path = _require_path(
            manifest.project_template_path,
            "project_template_path",
            purpose="the executor has no project to compose",
        )
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, Mapping):
            raise CampaignRunRefusal(f"project template {path} is not a JSON object")
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
        return f"{field}: {len(runs)} row(s) for {', '.join(versions)}"

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
