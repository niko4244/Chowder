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
    PROMOTION_POLICY_VERSION,
    CampaignManifest,
    CampaignManifestError,
    settle_campaign,
    stops_on_admission_refusal,
    stops_on_campaign_overrun,
)
from chowder.growth.compute_cost import (
    ACTUAL_DEVICE_GPU_HOURS_UNMEASURED,
    ACTUAL_WALL_GPU_HOURS_EXCEEDED,
    ComputeCost,
)


def _manifest_doc(**overrides) -> dict:
    doc = {
        "cycle_id": "gen2-campaign",
        "parent_version": "gen1",
        "base_model_path": "F:/llm-models/Qwen3.8-9B-abliterated-25-bf16",
        "base_model_digest": "a" * 64,
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
    assert manifest.base_model_path == "F:/llm-models/Qwen3.8-9B-abliterated-25-bf16"
    assert manifest.state_root.startswith("C:/Users/nikma")
    # No adapter declared: the parent generation is the base itself.
    assert manifest.has_parent_adapter() is False
    assert manifest.model_identity() == {
        "base_model_path": manifest.base_model_path,
        "base_model_digest": manifest.base_model_digest,
    }


def test_base_and_adapter_identity_are_separate_fields(tmp_path: Path) -> None:
    """One overloaded digest field is what made gen2's identity ambiguous."""
    doc = _manifest_doc(
        parent_adapter_path="C:/runs/gen1/attempts/attempt-10/adapter",
        parent_adapter_digest="b" * 64,
    )
    manifest = CampaignManifest.from_mapping(doc)
    assert manifest.has_parent_adapter() is True
    assert manifest.base_model_digest == "a" * 64
    assert manifest.parent_adapter_digest == "b" * 64
    # The two identify different objects and are reported as such.
    assert manifest.model_identity()["parent_adapter_digest"] == "b" * 64
    assert "parent_model_digest" not in manifest.model_identity()


def test_an_adapter_needs_both_path_and_digest() -> None:
    with pytest.raises(CampaignManifestError, match="declared together"):
        CampaignManifest.from_mapping(
            _manifest_doc(parent_adapter_digest="b" * 64)
        )
    with pytest.raises(CampaignManifestError, match="declared together"):
        CampaignManifest.from_mapping(
            _manifest_doc(parent_adapter_path="C:/runs/gen1/adapter")
        )


def test_adapter_digest_must_be_real_sha256_syntax() -> None:
    with pytest.raises(CampaignManifestError, match="sha256"):
        CampaignManifest.from_mapping(
            _manifest_doc(
                parent_adapter_path="C:/runs/gen1/adapter",
                parent_adapter_digest="NOT-A-DIGEST",
            )
        )
    with pytest.raises(CampaignManifestError, match="sha256"):
        CampaignManifest.from_mapping(_manifest_doc(base_model_digest="A" * 64))


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
        CampaignManifest.from_mapping(_manifest_doc(base_model_digest="abc"))


def test_a_stopping_rule_the_runner_cannot_act_on_refuses() -> None:
    doc = _manifest_doc(stopping_rules=["stop when the vibe changes"])
    with pytest.raises(CampaignManifestError, match="stopping rule"):
        CampaignManifest.from_mapping(doc)


def test_the_declared_stopping_rules_map_to_two_real_behaviors() -> None:
    admission, overrun = CampaignManifest.from_mapping(_manifest_doc()).stopping_rules
    assert stops_on_admission_refusal((admission,))
    assert not stops_on_admission_refusal(("stop on campaign overrun",))
    assert stops_on_campaign_overrun((overrun,))
    assert not stops_on_campaign_overrun(("stop on admission refusal",))
    # The historical spellings stay valid and mean the overrun behavior, so a
    # manifest already on disk keeps its declaration without a rewrite.
    assert stops_on_campaign_overrun(
        ("stop on campaign settlement overrun (artifact preserved)",)
    )


def test_a_promotion_policy_this_code_cannot_execute_refuses() -> None:
    doc = _manifest_doc(promotion_policy_version="promotion-policy-v9-imagined")
    with pytest.raises(CampaignManifestError, match="implemented policy"):
        CampaignManifest.from_mapping(doc)
    assert CampaignManifest.from_mapping(_manifest_doc()).promotion_policy_version == (
        PROMOTION_POLICY_VERSION
    )


def test_the_candidate_version_is_declared_or_derived_never_invented() -> None:
    assert CampaignManifest.from_mapping(_manifest_doc()).resolved_candidate_version() == "gen2"
    declared = _manifest_doc(candidate_version="gen2b-experiment")
    assert (
        CampaignManifest.from_mapping(declared).resolved_candidate_version()
        == "gen2b-experiment"
    )
    with pytest.raises(CampaignManifestError, match="declare candidate_version"):
        CampaignManifest.from_mapping(
            _manifest_doc(parent_version="qwen3.8-9b-base")
        ).resolved_candidate_version()


def test_an_unknown_budget_field_refuses() -> None:
    doc = _manifest_doc()
    doc["budget"] = dict(doc["budget"], gpu_hours_ceiling_campaign=1.0)
    with pytest.raises(CampaignManifestError, match="unknown budget fields"):
        CampaignManifest.from_mapping(doc)


def test_campaign_settlement_counts_losing_recipe_overrun() -> None:
    """Winner inside budget + loser inside its own ceiling but the campaign
    total over: the hard campaign total fails."""
    manifest = CampaignManifest.from_mapping(_manifest_doc())
    verdict = settle_campaign(
        manifest,
        total=ComputeCost.measured(
            device_gpu_hours=0.0, wall_gpu_hours=1.60, source="cycle ledger total"
        ),
    )
    assert not verdict.compliant
    assert any(ACTUAL_WALL_GPU_HOURS_EXCEEDED in r for r in verdict.failure_reasons)


def test_a_device_ceiling_is_admission_only_when_device_time_is_not_measured() -> None:
    """The default budget does not claim a device measurement it never took."""
    manifest = CampaignManifest.from_mapping(_manifest_doc())
    assert manifest.budget.device_time_measured is False
    verdict = settle_campaign(
        manifest,
        total=ComputeCost.from_wall_only(1.20, source="cycle ledger total"),
    )
    assert verdict.compliant, verdict.failure_reasons


def test_campaign_settlement_passes_when_total_fits() -> None:
    manifest = CampaignManifest.from_mapping(_manifest_doc())
    verdict = settle_campaign(
        manifest,
        total=ComputeCost.measured(
            device_gpu_hours=0.35, wall_gpu_hours=1.20, source="cycle ledger total"
        ),
    )
    assert verdict.compliant, verdict.failure_reasons


def test_campaign_settlement_refuses_a_device_ceiling_it_did_not_measure() -> None:
    """The real Gen-1 ledger is exactly this shape: wall measured, device not.

    A campaign that declares its executor separates device time must not
    settle a device ceiling against a ledger whose device figure was never
    measured.
    """
    doc = _manifest_doc()
    doc["budget"] = dict(doc["budget"], device_time_measured=True)
    manifest = CampaignManifest.from_mapping(doc)
    verdict = settle_campaign(
        manifest,
        total=ComputeCost.from_wall_only(1.20, source="cycle ledger total"),
    )
    assert not verdict.compliant
    assert any(
        ACTUAL_DEVICE_GPU_HOURS_UNMEASURED in reason
        for reason in verdict.failure_reasons
    )
