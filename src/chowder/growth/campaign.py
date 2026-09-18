"""Growth campaign manifests: preregistration as configuration.

A campaign manifest is the preregistration a growth cycle executes: parent
identity, promotion sets, recipe ceilings, budget units, and stopping rules
declared before compute. Paths live in the manifest (never hardcoded in
reusable code); every budget field names its unit explicitly.

Two controls, one declaration:

- **Admission** (before compute): each recipe's *projected* device and wall
  GPU-hours must fit the campaign's per-recipe ceilings and its total must
  fit the campaign total. Admission refusal happens before any attempt runs.
- **Settlement** (after compute): the campaign settles each attempt's
  *actual* cost through :func:`chowder.growth.compute_cost.settle_cost` --
  a successful train that overran its envelope is a refusal with the
  artifact preserved, and losing-recipe compute still counts toward the
  campaign total.

The manifest refuses unknown fields, negative/non-finite numbers, unpinned
benchmark names (no ``latest``), and mismatched unit declarations, so a
typo in configuration fails loudly instead of silently changing what a
promotion compares.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import json

from .compute_cost import ComputeCost, settle_cost

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
    """All ceilings, every field named by unit."""

    device_gpu_hours_ceiling_per_recipe: float
    wall_gpu_hours_ceiling_per_recipe: float
    device_gpu_hours_ceiling_campaign: float
    wall_gpu_hours_ceiling_campaign: float

    def __post_init__(self) -> None:
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
    cycle_id: str
    parent_version: str
    parent_model_path: str
    parent_model_digest: str
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
    promotion_policy_version: str = "promotion-policy-v2-provenance-settlement"
    contamination_manifest_path: str = ""
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
            "cycle_id", "parent_version", "parent_model_path", "parent_model_digest",
            "state_root", "target_benchmarks", "protected_benchmarks", "broad_benchmarks",
            "calibration_benchmarks", "reliability_benchmarks", "budget", "recipes",
            "candidate_selection_policy", "stopping_rules", "promotion_policy_version",
            "contamination_manifest_path", "notes",
        }
        unknown = sorted(set(document) - allowed)
        if unknown:
            raise CampaignManifestError(
                f"{source}: unknown manifest fields {unknown}; a typo here would "
                "silently change what the campaign declares"
            )
        required = {
            "cycle_id", "parent_version", "parent_model_path", "parent_model_digest",
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
        budget_allowed = {
            "device_gpu_hours_ceiling_per_recipe", "wall_gpu_hours_ceiling_per_recipe",
            "device_gpu_hours_ceiling_campaign", "wall_gpu_hours_ceiling_campaign",
        }
        if not isinstance(budget_doc, Mapping) or set(budget_doc) != budget_allowed:
            raise CampaignManifestError(
                f"{source}: budget must declare exactly {sorted(budget_allowed)} -- "
                "every ceiling names its unit"
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

        for path_field in ("parent_model_path", "state_root"):
            value = document[path_field]
            if not isinstance(value, str) or not value.strip():
                raise CampaignManifestError(f"{source}: {path_field} must be a non-empty path string")

        if not isinstance(document["parent_model_digest"], str) or len(document["parent_model_digest"]) != 64:
            raise CampaignManifestError(
                f"{source}: parent_model_digest must be a sha256 hex digest"
            )

        return cls(
            cycle_id=str(document["cycle_id"]),
            parent_version=str(document["parent_version"]),
            parent_model_path=str(document["parent_model_path"]),
            parent_model_digest=str(document["parent_model_digest"]),
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
            promotion_policy_version=str(document.get("promotion_policy_version", "promotion-policy-v2-provenance-settlement")),
            contamination_manifest_path=str(document.get("contamination_manifest_path", "")),
            notes=str(document.get("notes", "")),
        )


def admit_recipe(
    manifest: CampaignManifest,
    *,
    recipe_id: str,
    projected: ComputeCost,
) -> None:
    """Admission control: refuse before compute when projections exceed ceilings.

    Raises ``CampaignManifestError`` naming the breached ceiling; the refusal
    happens before any attempt directory is created.
    """
    if projected.device_gpu_hours > manifest.budget.device_gpu_hours_ceiling_per_recipe + 1e-12:
        raise CampaignManifestError(
            f"{recipe_id}: projected device {projected.device_gpu_hours:.6f} exceeds "
            f"per-recipe device ceiling "
            f"{manifest.budget.device_gpu_hours_ceiling_per_recipe:.6f} (admission refused)"
        )
    if projected.wall_gpu_hours > manifest.budget.wall_gpu_hours_ceiling_per_recipe + 1e-12:
        raise CampaignManifestError(
            f"{recipe_id}: projected wall {projected.wall_gpu_hours:.6f} exceeds "
            f"per-recipe wall ceiling "
            f"{manifest.budget.wall_gpu_hours_ceiling_per_recipe:.6f} (admission refused)"
        )


def settle_attempt(
    manifest: CampaignManifest,
    *,
    recipe_id: str,
    actual: ComputeCost,
    projected: ComputeCost,
):
    """Settlement control for one attempt against the campaign's ceilings."""
    return settle_cost(
        actual=actual,
        projected=projected,
        device_ceiling=manifest.budget.device_gpu_hours_ceiling_per_recipe,
        wall_ceiling=manifest.budget.wall_gpu_hours_ceiling_per_recipe,
    )


def settle_campaign(
    manifest: CampaignManifest,
    *,
    total: ComputeCost,
):
    """Settlement control for the whole campaign: every recipe counts."""
    return settle_cost(
        actual=total,
        projected=ComputeCost.zero(source="campaign projection (per-recipe admission controls)"),
        device_ceiling=manifest.budget.device_gpu_hours_ceiling_campaign,
        wall_ceiling=manifest.budget.wall_gpu_hours_ceiling_campaign,
    )
