"""Campaign manifest integrity: paths in config, units explicit, fail-closed.

The manifest is the preregistration-as-configuration. These tests pin:
unknown fields refuse, benchmark sets must be pinned, budgets name their
unit, admission refuses over-projection before compute, and campaign
settlement counts every recipe (a losing recipe's overrun fails the
campaign when the campaign total is hard).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chowder.growth.campaign import (
    CampaignManifest,
    CampaignManifestError,
    admit_recipe,
    settle_campaign,
)
from chowder.growth.compute_cost import (
    ACTUAL_WALL_GPU_HOURS_EXCEEDED,
    ComputeCost,
)


def _manifest_doc(**overrides) -> dict:
    doc = {
        "cycle_id": "gen2-campaign",
        "parent_version": "gen1",
        "parent_model_path": "F:/llm-models/Qwen3.8-9B-abliterated-25-bf16",
        "parent_model_digest": "a" * 64,
        "state_root": "C:/Users/nikma/Chowder-Protected/runs/gen2",
        "target_benchmarks": ["generation-diagnostics@gen2-eval-protocol-v1"],
        "protected_benchmarks": ["math500@2024-04", "mgsm@2022-11"],
        "broad_benchmarks": ["math500@2024-04"],
        "calibration_benchmarks": [],
        "reliability_benchmarks": [],
        "budget": {
            "device_gpu_hours_ceiling_per_recipe": 0.30,
            "wall_gpu_hours_ceiling_per_recipe": 0.75,
            "device_gpu_hours_ceiling_campaign": 0.60,
            "wall_gpu_hours_ceiling_campaign": 1.50,
        },
        "recipes": ["recipe-a", "recipe-b"],
        "candidate_selection_policy": "first_successful",
        "stopping_rules": ["stop on admission refusal", "stop on campaign overrun"],
    }
    doc.update(overrides)
    return doc


def test_manifest_loads_and_pins_paths_in_config(tmp_path: Path) -> None:
    path = tmp_path / "campaign.yaml"
    path.write_text(json.dumps(_manifest_doc()), encoding="utf-8")
    manifest = CampaignManifest.from_file(path)
    assert manifest.parent_model_path == "F:/llm-models/Qwen3.8-9B-abliterated-25-bf16"
    assert manifest.state_root.startswith("C:/Users/nikma")


def test_unknown_field_refuses() -> None:
    doc = _manifest_doc()
    doc["search_variants"] = True  # the known footgun
    with pytest.raises(CampaignManifestError, match="unknown manifest fields"):
        CampaignManifest.from_mapping(doc)


def test_unpinned_benchmark_name_refuses() -> None:
    doc = _manifest_doc(protected_benchmarks=["math500@latest"])
    with pytest.raises(CampaignManifestError, match="pinned"):
        CampaignManifest.from_mapping(doc)


def test_budget_must_declare_all_four_unit_named_ceilings() -> None:
    doc = _manifest_doc()
    doc["budget"] = {"gpu_hours": 1.0}
    with pytest.raises(CampaignManifestError, match="unit"):
        CampaignManifest.from_mapping(doc)


def test_bad_digest_refuses() -> None:
    with pytest.raises(CampaignManifestError, match="sha256"):
        CampaignManifest.from_mapping(_manifest_doc(parent_model_digest="abc"))


def test_admission_refuses_over_projection_before_compute() -> None:
    manifest = CampaignManifest.from_mapping(_manifest_doc())
    with pytest.raises(CampaignManifestError, match="admission refused"):
        admit_recipe(
            manifest,
            recipe_id="recipe-a",
            projected=ComputeCost(0.0, 0.90, source="recipe projection"),
        )
    # Under ceiling: admitted.
    admit_recipe(
        manifest,
        recipe_id="recipe-a",
        projected=ComputeCost(0.20, 0.70, source="recipe projection"),
    )


def test_campaign_settlement_counts_losing_recipe_overrun() -> None:
    """Winner inside budget + loser inside its own ceiling but the campaign
    total over: the hard campaign total fails."""
    manifest = CampaignManifest.from_mapping(_manifest_doc())
    verdict = settle_campaign(
        manifest,
        total=ComputeCost(0.0, 1.60, source="cycle ledger total"),
    )
    assert not verdict.compliant
    assert any(ACTUAL_WALL_GPU_HOURS_EXCEEDED in r for r in verdict.failure_reasons)


def test_campaign_settlement_passes_when_total_fits() -> None:
    manifest = CampaignManifest.from_mapping(_manifest_doc())
    verdict = settle_campaign(
        manifest,
        total=ComputeCost(0.35, 1.20, source="cycle ledger total"),
    )
    assert verdict.compliant, verdict.failure_reasons
