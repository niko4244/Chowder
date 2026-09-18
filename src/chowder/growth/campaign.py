"""Growth campaign manifests: preregistration as configuration.

A campaign manifest is the preregistration a growth cycle executes: parent
identity, promotion sets, recipe set, budget units, stopping rules and the
input paths a run reads. Paths live in the manifest (never hardcoded in
reusable code); every budget field names its unit explicitly.

One declaration, two controls, one owner each:

- **Admission** (before compute) is owned by the *executor*: the per-recipe
  ceilings become its envelope, and it refuses a recipe whose projection
  does not fit before any subprocess starts. The campaign runner asks the
  executor (see ``campaign_runner``) rather than comparing costs itself, so
  there is exactly one implementation of "may this recipe run".
- **Settlement** (after compute) is owned here: the campaign's total actual
  cost is settled against the campaign ceilings through
  :func:`chowder.growth.compute_cost.settle_cost`, so a campaign whose
  recipes each fit can still fail as a whole, and a device ceiling declared
  over an unmeasured device figure fails closed.

The manifest refuses unknown fields, negative/non-finite numbers, unpinned
benchmark names (no ``latest``), a promotion policy this code cannot execute,
and a stopping rule the runner does not implement, so a typo in configuration
fails loudly instead of silently changing what a promotion compares.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import json

from .compute_cost import (
    PROJECTED_DEVICE_GPU_HOURS_EXCEEDED,
    PROJECTED_WALL_GPU_HOURS_EXCEEDED,
    ComputeCost,
    settle_cost,
)

#: The promotion rule this package implements. A campaign that declares any
#: other version is refused: the manifest is a preregistration, so it must not
#: name a policy the code cannot execute.
PROMOTION_POLICY_VERSION = "promotion-policy-v2-provenance-settlement"

#: The stopping rules this runner recognizes, each mapped to the behavior that
#: enforces it. A declared rule outside this table is refused at load time: the
#: runner must not accept a stopping condition nothing evaluates, because a
#: declaration that changes nothing is the same defect as an unenforced gate.
#: A rule can be enforced by a phase of the run (admission, overrun) or by the
#: code's fixed inputs (the firewall refuses tainted material; promotion
#: thresholds are frozen constants no manifest key can touch) -- either way,
#: something concrete refuses. Spellings drift across manifests already on
#: disk, so each behavior has a small alias set; aliases are one rule, not new
#: rules.
STOPPING_RULE_ON_ADMISSION_REFUSAL = "stop on admission refusal"
STOPPING_RULE_ON_CAMPAIGN_OVERRUN = "stop on campaign overrun"
STOPPING_RULE_ON_CONTAMINATION_REFUSAL = "stop on contamination refusal"
STOPPING_RULE_THRESHOLDS_FROZEN = "thresholds frozen after preregistration"

_ADMISSION_REFUSAL_SPELLINGS = frozenset(
    {
        STOPPING_RULE_ON_ADMISSION_REFUSAL,
        "stop before compute on admission refusal",
    }
)
_OVERRUN_SPELLINGS = frozenset(
    {
        STOPPING_RULE_ON_CAMPAIGN_OVERRUN,
        "stop on campaign settlement overrun",
        "stop on campaign settlement overrun (artifact preserved)",
    }
)
_CONTAMINATION_REFUSAL_SPELLINGS = frozenset(
    {
        STOPPING_RULE_ON_CONTAMINATION_REFUSAL,
        "stop on contamination refusal (KNOWN/POSSIBLE on any declared source)",
    }
)
_THRESHOLD_SPELLINGS = frozenset(
    {
        STOPPING_RULE_THRESHOLDS_FROZEN,
        "never enlarge a frozen threshold after candidate results are visible",
    }
)

#: Every recognized spelling, and what enforces it.
STOPPING_RULE_ENFORCEMENT: Mapping[str, str] = {
    **{rule: "the campaign stops before compute when a recipe is refused" for rule in _ADMISSION_REFUSAL_SPELLINGS},
    **{rule: "the campaign stops the remaining recipes once a ceiling is breached" for rule in _OVERRUN_SPELLINGS},
    **{rule: "the executor's firewall refuses KNOWN/POSSIBLE material before an attempt starts" for rule in _CONTAMINATION_REFUSAL_SPELLINGS},
    **{rule: "the promotion policy's thresholds are code constants; no manifest key can change one" for rule in _THRESHOLD_SPELLINGS},
}
STOPPING_RULES = frozenset(STOPPING_RULE_ENFORCEMENT)


def stops_on_admission_refusal(stopping_rules: Sequence[str]) -> bool:
    """Whether a refused recipe ends the campaign or is skipped."""
    return any(rule in _ADMISSION_REFUSAL_SPELLINGS for rule in stopping_rules)


def stops_on_campaign_overrun(stopping_rules: Sequence[str]) -> bool:
    """Whether exceeding a campaign ceiling stops the remaining recipes."""
    return any(rule in _OVERRUN_SPELLINGS for rule in stopping_rules)

def _require_sha256(value: Any, field: str, source: str) -> None:
    """A model digest is 64 lowercase hex characters or it is not identity."""
    if not isinstance(value, str) or len(value) != 64 or any(
        ch not in "0123456789abcdef" for ch in value
    ):
        raise CampaignManifestError(
            f"{source}: {field} must be a sha256 hex digest (64 lowercase hex "
            f"characters), got {value!r}"
        )


#: Benchmarks named in promotion sets must be pinned (``name@version``).
def _require_pinned(qualified_id: str, where: str) -> None:
    if "@" not in str(qualified_id) or str(qualified_id).split("@")[1].strip().lower() in {
        "latest",
        "current",
        "",
    }:
        raise CampaignManifestError(
            f"{where} names {qualified_id!r}, which is not a pinned benchmark@version"
        )


class CampaignManifestError(ValueError):
    """The campaign manifest is malformed or internally inconsistent."""


@dataclass(frozen=True)
class CampaignBudget:
    """All ceilings, every field named by unit.

    ``device_time_measured`` says whether this campaign's executor separates
    device time. When it does not (the default, and the truth for a trainer
    that reports wall), the device ceilings are **admission** constraints on
    the projected plan and the campaign's settlement does not claim them --
    an unsettleable ceiling is recorded as such rather than satisfied by a
    placeholder zero. Declaring ``True`` makes the device ceilings settle
    against measured device time, and an attempt that reports none refuses.
    """

    device_gpu_hours_ceiling_per_recipe: float
    wall_gpu_hours_ceiling_per_recipe: float
    device_gpu_hours_ceiling_campaign: float
    wall_gpu_hours_ceiling_campaign: float
    device_time_measured: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.device_time_measured, bool):
            raise CampaignManifestError(
                f"budget.device_time_measured must be a bool, got "
                f"{self.device_time_measured!r}"
            )
        for label, value in (
            ("device_gpu_hours_ceiling_per_recipe", self.device_gpu_hours_ceiling_per_recipe),
            ("wall_gpu_hours_ceiling_per_recipe", self.wall_gpu_hours_ceiling_per_recipe),
            ("device_gpu_hours_ceiling_campaign", self.device_gpu_hours_ceiling_campaign),
            ("wall_gpu_hours_ceiling_campaign", self.wall_gpu_hours_ceiling_campaign),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise CampaignManifestError(f"budget.{label} must be a finite number")
            if float(value) < 0:
                raise CampaignManifestError(f"budget.{label} cannot be negative")


@dataclass(frozen=True)
class CampaignManifest:
    """One preregistered campaign, and nothing this runner may invent.

    Every field here is consumed by :mod:`chowder.growth.campaign_runner`:
    a key the runner cannot act on is refused at load time rather than
    silently ignored, because a declaration that changes nothing is the same
    class of defect as an unenforced gate. ``notes`` is the single documented
    exception -- prose only, and it gates nothing.
    """

    cycle_id: str
    parent_version: str
    #: Model identity is deliberately two objects, never one overloaded field:
    #: a generation is (dense base, optional parent adapter), and a base
    #: digest must never be compared against an adapter tree (or vice versa).
    #: ``parent_adapter_*`` is empty when the parent generation *is* the base.
    base_model_path: str
    base_model_digest: str
    parent_adapter_path: str
    parent_adapter_digest: str
    state_root: str
    target_benchmarks: tuple[str, ...]
    protected_benchmarks: tuple[str, ...]
    broad_benchmarks: tuple[str, ...]
    calibration_benchmarks: tuple[str, ...]
    reliability_benchmarks: tuple[str, ...]
    budget: CampaignBudget
    recipe_ids: tuple[str, ...]
    candidate_selection_policy: str
    stopping_rules: tuple[str, ...] = ()
    promotion_policy_version: str = PROMOTION_POLICY_VERSION
    contamination_manifest_path: str = ""
    #: The generation the verdict is recorded under. Empty means "derive it
    #: from parent_version" (``gen1`` -> ``gen2``); a declared value is used
    #: verbatim.
    candidate_version: str = ""
    #: Inputs the run reads from disk. Each is optional in the schema (so a
    #: historical manifest still validates) and required by the phase that
    #: needs it, where its absence is a named refusal rather than a default.
    project_template_path: str = ""
    training_material_path: str = ""
    data_registry_path: str = ""
    hardware_budget_path: str = ""
    parent_profile_path: str = ""
    parent_eval_report_path: str = ""
    candidate_eval_report_path: str = ""
    notes: str = ""

    @classmethod
    def from_file(cls, path: Path | str) -> "CampaignManifest":
        path = Path(path)
        if not path.exists():
            raise CampaignManifestError(f"campaign manifest not found: {path}")
        document = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_mapping(document, source=str(path))

    @classmethod
    def from_mapping(cls, document: Mapping[str, Any], *, source: str = "<memory>") -> "CampaignManifest":
        allowed = {
            "cycle_id", "parent_version", "base_model_path", "base_model_digest",
            "parent_adapter_path", "parent_adapter_digest",
            "state_root", "target_benchmarks", "protected_benchmarks", "broad_benchmarks",
            "calibration_benchmarks", "reliability_benchmarks", "budget", "recipes",
            "candidate_selection_policy", "stopping_rules", "promotion_policy_version",
            "contamination_manifest_path", "notes", "candidate_version",
            "project_template_path", "training_material_path", "data_registry_path",
            "hardware_budget_path", "parent_profile_path", "parent_eval_report_path",
            "candidate_eval_report_path",
        }
        unknown = sorted(set(document) - allowed)
        if unknown:
            raise CampaignManifestError(
                f"{source}: unknown manifest fields {unknown}; a typo here would "
                "silently change what the campaign declares"
            )
        required = {
            "cycle_id", "parent_version", "base_model_path", "base_model_digest",
            "state_root", "target_benchmarks", "protected_benchmarks", "broad_benchmarks",
            "calibration_benchmarks", "reliability_benchmarks", "budget", "recipes",
            "candidate_selection_policy",
        }
        missing = sorted(required - set(document))
        if missing:
            raise CampaignManifestError(f"{source}: missing required fields {missing}")

        for set_name in (
            "target_benchmarks", "protected_benchmarks", "broad_benchmarks",
            "calibration_benchmarks", "reliability_benchmarks",
        ):
            value = document[set_name]
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                raise CampaignManifestError(f"{source}: {set_name} must be a list of benchmark ids")
            for qualified_id in value:
                _require_pinned(qualified_id, set_name)

        budget_doc = document["budget"]
        budget_required = {
            "device_gpu_hours_ceiling_per_recipe", "wall_gpu_hours_ceiling_per_recipe",
            "device_gpu_hours_ceiling_campaign", "wall_gpu_hours_ceiling_campaign",
        }
        budget_allowed = budget_required | {"device_time_measured"}
        if not isinstance(budget_doc, Mapping) or not budget_required <= set(budget_doc):
            raise CampaignManifestError(
                f"{source}: budget must declare {sorted(budget_required)} -- "
                "every ceiling names its unit"
            )
        unknown_budget = sorted(set(budget_doc) - budget_allowed)
        if unknown_budget:
            raise CampaignManifestError(
                f"{source}: unknown budget fields {unknown_budget}; a ceiling "
                "nothing reads is not a ceiling"
            )
        budget = CampaignBudget(**dict(budget_doc))

        recipes = document["recipes"]
        if not isinstance(recipes, list) or not recipes or not all(isinstance(r, str) for r in recipes):
            raise CampaignManifestError(f"{source}: recipes must be a non-empty list of recipe ids")

        policy = document["candidate_selection_policy"]
        if policy not in {"first_successful", "first_by_loss"}:
            raise CampaignManifestError(
                f"{source}: candidate_selection_policy {policy!r} is not one of "
                "first_successful / first_by_loss"
            )

        for rule in document.get("stopping_rules", ()):
            if rule not in STOPPING_RULES:
                raise CampaignManifestError(
                    f"{source}: stopping rule {rule!r} is not one this runner "
                    f"enforces (known: {sorted(STOPPING_RULES)}); a stopping rule "
                    "nobody acts on is not a stopping rule"
                )

        declared_policy = str(document.get("promotion_policy_version", PROMOTION_POLICY_VERSION))
        if declared_policy != PROMOTION_POLICY_VERSION:
            raise CampaignManifestError(
                f"{source}: promotion_policy_version {declared_policy!r} is not "
                f"the implemented policy {PROMOTION_POLICY_VERSION!r}; a "
                "preregistration that names a policy the code cannot execute is "
                "not a preregistration"
            )

        for path_field in ("base_model_path", "state_root"):
            value = document[path_field]
            if not isinstance(value, str) or not value.strip():
                raise CampaignManifestError(f"{source}: {path_field} must be a non-empty path string")

        _require_sha256(document["base_model_digest"], "base_model_digest", source)

        # The adapter pair is all-or-nothing: a path without a digest would be
        # an unverifiable parent, and a digest without a path cannot be checked.
        adapter_path = document.get("parent_adapter_path", "")
        adapter_digest = document.get("parent_adapter_digest", "")
        if bool(str(adapter_path).strip()) != bool(str(adapter_digest).strip()):
            raise CampaignManifestError(
                f"{source}: parent_adapter_path and parent_adapter_digest must be "
                "declared together or both omitted; a parent adapter is either "
                "identity-verified or not declared"
            )
        if str(adapter_digest).strip():
            _require_sha256(adapter_digest, "parent_adapter_digest", source)

        return cls(
            cycle_id=str(document["cycle_id"]),
            parent_version=str(document["parent_version"]),
            base_model_path=str(document["base_model_path"]),
            base_model_digest=str(document["base_model_digest"]),
            parent_adapter_path=str(adapter_path),
            parent_adapter_digest=str(adapter_digest),
            state_root=str(document["state_root"]),
            target_benchmarks=tuple(document["target_benchmarks"]),
            protected_benchmarks=tuple(document["protected_benchmarks"]),
            broad_benchmarks=tuple(document["broad_benchmarks"]),
            calibration_benchmarks=tuple(document["calibration_benchmarks"]),
            reliability_benchmarks=tuple(document["reliability_benchmarks"]),
            budget=budget,
            recipe_ids=tuple(recipes),
            candidate_selection_policy=str(policy),
            stopping_rules=tuple(document.get("stopping_rules", ())),
            promotion_policy_version=declared_policy,
            contamination_manifest_path=str(document.get("contamination_manifest_path", "")),
            candidate_version=str(document.get("candidate_version", "")),
            project_template_path=str(document.get("project_template_path", "")),
            training_material_path=str(document.get("training_material_path", "")),
            data_registry_path=str(document.get("data_registry_path", "")),
            hardware_budget_path=str(document.get("hardware_budget_path", "")),
            parent_profile_path=str(document.get("parent_profile_path", "")),
            parent_eval_report_path=str(document.get("parent_eval_report_path", "")),
            candidate_eval_report_path=str(document.get("candidate_eval_report_path", "")),
            notes=str(document.get("notes", "")),
        )

    def resolved_candidate_version(self) -> str:
        """The generation this campaign produces, declared or derived.

        Derivation is a convention, not a guess: ``gen<N>`` advances to
        ``gen<N+1>``, and a version that does not follow the pattern is
        refused rather than invented.
        """
        if self.candidate_version:
            return self.candidate_version
        parent = self.parent_version
        if not parent.startswith("gen") or not parent[3:].isdigit():
            raise CampaignManifestError(
                f"parent_version {parent!r} does not follow gen<N>, so the "
                "candidate version cannot be derived; declare candidate_version"
            )
        return f"gen{int(parent[3:]) + 1}"

    def has_parent_adapter(self) -> bool:
        """Whether the parent generation carries an adapter over its base."""
        return bool(self.parent_adapter_digest)

    def model_identity(self) -> dict[str, str]:
        """The parent identity as explicit (path, digest) pairs.

        Recorded into lineage so a reader can tell base from adapter, which one
        ``parent_version`` names, and which bytes were verified.
        """
        identity = {
            "base_model_path": self.base_model_path,
            "base_model_digest": self.base_model_digest,
        }
        if self.has_parent_adapter():
            identity["parent_adapter_path"] = self.parent_adapter_path
            identity["parent_adapter_digest"] = self.parent_adapter_digest
        return identity


@dataclass(frozen=True)
class CampaignProjection:
    """Whether the campaign's *planned* spend fits its declared envelope.

    Admission and settlement are different controls: this one compares what
    the campaign plans to spend against the campaign ceilings before compute,
    and :func:`settle_campaign` compares what it actually spent after. A plan
    is not a measurement, so this control makes no measurement claim -- but it
    is what gives the campaign-level ceilings (the device one included, which
    a wall-only executor can never settle) real teeth.
    """

    projected: ComputeCost
    compliant: bool
    failure_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "projected": self.projected.to_dict(),
            "compliant": self.compliant,
            "failure_reasons": list(self.failure_reasons),
        }


def settle_campaign_projection(
    manifest: CampaignManifest, *, projected: ComputeCost
) -> CampaignProjection:
    """Admission control for the campaign envelope, before any compute."""
    budget = manifest.budget
    reasons: list[str] = []
    if projected.device_gpu_hours > budget.device_gpu_hours_ceiling_campaign + 1e-12:
        reasons.append(
            f"{PROJECTED_DEVICE_GPU_HOURS_EXCEEDED}: planned device "
            f"{projected.device_gpu_hours:.6f} > campaign ceiling "
            f"{budget.device_gpu_hours_ceiling_campaign:.6f}"
        )
    if projected.wall_gpu_hours > budget.wall_gpu_hours_ceiling_campaign + 1e-12:
        reasons.append(
            f"{PROJECTED_WALL_GPU_HOURS_EXCEEDED}: planned wall "
            f"{projected.wall_gpu_hours:.6f} > campaign ceiling "
            f"{budget.wall_gpu_hours_ceiling_campaign:.6f}"
        )
    return CampaignProjection(
        projected=projected, compliant=not reasons, failure_reasons=tuple(reasons)
    )


def settle_campaign(
    manifest: CampaignManifest,
    *,
    total: ComputeCost,
):
    """Settlement control for the whole campaign: every recipe counts.

    The device ceiling is settled only when the budget declares that device
    time is measured; otherwise it stays an admission constraint (the
    executor's envelope) and settlement must not be asked to certify it.
    """
    budget = manifest.budget
    return settle_cost(
        actual=total,
        projected=ComputeCost.zero(source="campaign projection (per-recipe admission controls)"),
        device_ceiling=(
            budget.device_gpu_hours_ceiling_campaign
            if budget.device_time_measured
            else None
        ),
        wall_ceiling=budget.wall_gpu_hours_ceiling_campaign,
    )
