"""The campaign builder: a target becomes a frozen declaration, or a refusal.

The builder is the component that removes the human from the loop, so the tests
are mostly about what it *will not* do:

* it will not let a target move the budget, the protected set, the execution
  throughput or the trusted ancestor -- those come from the policy, and a parent
  declaration that disagrees with the policy is refused rather than merged;
* it will not optimize against a protected benchmark;
* it will not let a campaign declare an envelope its own target exceeds;
* it will not overwrite a frozen declaration: a second build into the same
  attempt refuses, because thresholds that can change after candidate results
  exist are not thresholds.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from chowder.growth.next_campaign import (
    NEXT_CAMPAIGN_ALREADY_FROZEN,
    NEXT_CAMPAIGN_PARENT_IDENTITY,
    NEXT_CAMPAIGN_POLICY_DRIFT,
    NEXT_CAMPAIGN_TARGET_TOO_EXPENSIVE,
    NEXT_CAMPAIGN_TARGET_UNUSABLE,
    NEXT_CAMPAIGN_TREATMENT_NOT_ALLOWED,
    NextCampaignBuilder,
    NextCampaignRefusal,
    LoopPolicy,
    POLICY_SCHEMA,
    TargetProposal,
)
from chowder.growth.simulator import skill_profile
from chowder.growth.target_selection import GrowthState, NextTargetSelector
from fixtures_growth_loop import parent_manifest, policy_from

START = {
    "math.reasoning": 0.62,
    "instruction.formatting": 0.31,
    "termination.control": 0.44,
}
IDENTITY = ("adapters/gen2", "c" * 64)


def _proposal(tmp_path: Path, **overrides) -> TargetProposal:
    """A real proposal from the real selector, optionally rewritten.

    Taken from the selector rather than hand-built: the builder's contract is
    with what the selector actually emits, so a hand-built stand-in could agree
    with a builder that no selector would ever feed.
    """
    proposal = NextTargetSelector().propose(
        parent_version="gen2",
        profile=skill_profile(START),
        state=GrowthState(root=tmp_path / "selector-state"),
        known_skills=tuple(START),
    )
    return dataclasses.replace(proposal, **overrides) if overrides else proposal


def _build(tmp_path: Path, **kwargs):
    parent = kwargs.pop("parent", None) or parent_manifest(tmp_path)
    policy = kwargs.pop("policy", None) or policy_from(parent)
    return NextCampaignBuilder(policy=policy).build(
        parent=parent,
        target=kwargs.pop("target", None) or _proposal(tmp_path),
        generation_root=kwargs.pop("generation_root", tmp_path / "generations"),
        **kwargs,
    )


def test_a_frozen_declaration_and_preregistration_are_written_before_any_compute(
    tmp_path: Path,
) -> None:
    frozen = _build(tmp_path)

    assert frozen.manifest_path.name == "campaign.json"
    assert frozen.preregistration_path.name == "preregistration.json"
    assert frozen.candidate_version == "gen3"
    assert frozen.directory == tmp_path / "generations" / frozen.cycle_id
    assert len(frozen.digest) == 64

    preregistration = json.loads(frozen.preregistration_path.read_text(encoding="utf-8"))
    assert preregistration["frozen_digest"] == frozen.digest
    assert preregistration["policy_digest"] == frozen.policy_digest
    assert preregistration["target"]["target_skill"] == frozen.target.target_skill
    assert "frozen before any candidate compute" in preregistration["statement"]
    # The digest covers the declaration as written, so re-reading the frozen
    # bytes and recomputing it must reproduce the digest the run will quote.
    on_disk = json.loads(frozen.manifest_path.read_text(encoding="utf-8"))
    assert on_disk == json.loads(json.dumps(preregistration["manifest"]))


def test_the_composed_declaration_is_loadable_by_the_engine(tmp_path: Path) -> None:
    """A declaration the runner cannot load is not a campaign."""
    frozen = _build(tmp_path)

    assert frozen.manifest.cycle_id == frozen.cycle_id
    # Declared the way the checked-in Gen-2 declaration declares it: derived from
    # the parent version, since the manifest schema has no candidate_version key.
    assert frozen.manifest.resolved_candidate_version() == "gen3"
    assert frozen.manifest.parent_version == "gen2"
    assert frozen.manifest.target_benchmarks == frozen.target.target_benchmarks
    assert len(frozen.manifest.recipe_ids) == len(set(frozen.manifest.recipe_ids)) >= 2


def test_the_trusted_ancestor_is_carried_from_the_parent_not_replaced(
    tmp_path: Path,
) -> None:
    """A generation that just promoted must not become its own floor."""
    parent = parent_manifest(tmp_path)
    frozen = _build(tmp_path, parent=parent)

    assert parent.baseline_eval_report_path
    assert frozen.manifest.baseline_eval_report_path == parent.baseline_eval_report_path
    assert frozen.manifest.protection.trusted_ancestor_version == (
        parent.protection.trusted_ancestor_version
    )


def test_a_second_build_into_the_same_attempt_refuses(tmp_path: Path) -> None:
    """Write-once: thresholds cannot move after candidate results exist."""
    proposal = _proposal(tmp_path)
    parent = parent_manifest(tmp_path)
    policy = policy_from(parent)
    builder = NextCampaignBuilder(policy=policy)

    builder.build(
        parent=parent, target=proposal, generation_root=tmp_path / "generations"
    )
    with pytest.raises(NextCampaignRefusal, match=NEXT_CAMPAIGN_ALREADY_FROZEN):
        builder.build(
            parent=parent, target=proposal, generation_root=tmp_path / "generations"
        )


def test_a_target_that_names_a_protected_benchmark_refuses(tmp_path: Path) -> None:
    """Protected sets are gates; a target may not optimize against one."""
    protected = parent_manifest(tmp_path).protected_benchmarks[0]
    with pytest.raises(NextCampaignRefusal, match=NEXT_CAMPAIGN_TARGET_UNUSABLE):
        _build(tmp_path, target=_proposal(tmp_path, target_benchmarks=(protected,)))


def test_a_target_whose_expected_cost_exceeds_the_campaign_envelope_refuses(
    tmp_path: Path,
) -> None:
    parent = parent_manifest(tmp_path)
    ceiling = float(parent.budget.wall_gpu_hours_ceiling_campaign)
    with pytest.raises(NextCampaignRefusal, match=NEXT_CAMPAIGN_TARGET_TOO_EXPENSIVE):
        _build(
            tmp_path,
            parent=parent,
            target=_proposal(tmp_path, expected_cost_gpu_hours=ceiling + 1.0),
        )


def test_a_treatment_the_policy_does_not_allow_refuses(tmp_path: Path) -> None:
    """A loop cannot widen its own allowed treatments."""
    parent = parent_manifest(tmp_path)
    policy = policy_from(parent, allowed_training_types=["sft"])
    with pytest.raises(NextCampaignRefusal, match=NEXT_CAMPAIGN_TREATMENT_NOT_ALLOWED):
        _build(
            tmp_path,
            parent=parent,
            policy=policy,
            target=_proposal(tmp_path, suggested_training_type="targeted_repair"),
        )


def test_a_parent_declaration_that_disagrees_with_the_policy_refuses(
    tmp_path: Path,
) -> None:
    """Merging a drifted template would let it move the ancestor or tolerance."""
    # The policy's own declaration, and a template that loosened its tolerance.
    reference = parent_manifest(tmp_path)
    loosened = json.loads(json.dumps(reference.protection.to_dict()))
    loosened["slice_regression_max"] = float(loosened.get("slice_regression_max") or 0.1) + 0.5
    drifted = parent_manifest(tmp_path, protection=loosened)

    with pytest.raises(NextCampaignRefusal, match=NEXT_CAMPAIGN_POLICY_DRIFT):
        _build(tmp_path, parent=drifted, policy=policy_from(reference))


def test_a_parent_declaration_without_a_trusted_ancestor_refuses(tmp_path: Path) -> None:
    parent = parent_manifest(tmp_path)
    drifted = parent_manifest(tmp_path, baseline_eval_report_path="")
    with pytest.raises(NextCampaignRefusal, match=NEXT_CAMPAIGN_POLICY_DRIFT):
        _build(tmp_path, parent=drifted, policy=policy_from(parent))


def test_a_parent_identity_that_is_not_a_digest_pair_refuses(tmp_path: Path) -> None:
    with pytest.raises(NextCampaignRefusal, match=NEXT_CAMPAIGN_PARENT_IDENTITY):
        _build(tmp_path, parent_identity=("adapters/gen2", "not-a-digest"))


def test_the_promoted_adapter_becomes_the_next_generation_s_parent(tmp_path: Path) -> None:
    frozen = _build(tmp_path, parent_identity=IDENTITY)

    assert frozen.manifest.parent_adapter_path == IDENTITY[0]
    assert frozen.manifest.parent_adapter_digest == IDENTITY[1]


def test_an_unknown_policy_key_refuses_rather_than_being_ignored(tmp_path: Path) -> None:
    """A limit nothing reads is not a limit."""
    parent = parent_manifest(tmp_path)
    document = {
        "maximum_generations": 2,
        "maximum_total_wall_gpu_hours": 1.0,
        "maximum_consecutive_non_promotions": 2,
        "maximum_same_target_attempts": 2,
        "maximum_candidates": len(parent.recipe_ids),
        "plateau_epsilon": 0.01,
        "allowed_training_types": ["targeted_repair"],
        "protected_benchmarks": list(parent.protected_benchmarks),
        "broad_benchmarks": list(parent.broad_benchmarks),
        "campaign_budget": json.loads(
            json.dumps(
                {
                    "device_gpu_hours_ceiling_per_recipe": parent.budget.device_gpu_hours_ceiling_per_recipe,
                    "wall_gpu_hours_ceiling_per_recipe": parent.budget.wall_gpu_hours_ceiling_per_recipe,
                    "device_gpu_hours_ceiling_campaign": parent.budget.device_gpu_hours_ceiling_campaign,
                    "wall_gpu_hours_ceiling_campaign": parent.budget.wall_gpu_hours_ceiling_campaign,
                    "device_time_measured": parent.budget.device_time_measured,
                }
            )
        ),
        "protection": parent.protection.to_dict(),
        "evaluation_execution": parent.evaluation_execution.to_dict(),
        "candidate_selection_policy": parent.candidate_selection_policy,
        "stopping_rules": list(parent.stopping_rules),
        "promotion_policy_version": parent.promotion_policy_version,
        "human_review_triggers": [],
        "maximum_unbounded_forever": 99,
    }
    with pytest.raises(NextCampaignRefusal, match=POLICY_SCHEMA):
        LoopPolicy.from_mapping(document, source="test")
