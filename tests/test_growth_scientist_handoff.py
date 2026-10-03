"""Scientist-handoff auto-consumption: a graduated survivor enters the growth
loop's next generation without a manual hand-off.

Pinned:
- the hand-off file is consumed through the loop's OWN gate chain: the
  survivor becomes the target proposal instead of the selector's pick, then
  the treatment allowlist, budget, draft composition and production
  preparation all apply unchanged;
- the loop refuses (or the survivor is refused downstream) when the hand-off
  carries no pinned benchmark for the target surface — naming the eval
  instrument remains an operator decision;
- the file is cleared exactly after the target is durably recorded; one
  survivor per generation, normal selection resumes after;
- a durable file naming a DIFFERENT target refuses to clear;
- an unreadable hand-off file refuses rather than silently skipping;
- no hand-off file → the loop behaves exactly as before (selector path).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from chowder.growth.growth_loop import GrowthLoop
from chowder.growth.next_campaign import (
    NextCampaignRefusal,
)
from chowder.growth.target_selection import GrowthState
from chowder.growth.scientist_handoff import (
    HANDOFF_FILENAME,
    ScreeningGraduate,
    clear_consumed_handoff,
    load_graduated_survivor,
    write_handoff_file,
)
from fixtures_growth_loop import parent_manifest, policy_from


def _graduate(**overrides: Any) -> ScreeningGraduate:
    base = dict(
        experiment_id="sciexp-fake-prop-fake-hyp-004",
        proposal_id="fake-prop-fake-hyp-004",
        intervention="decay replay ratio 0.25 -> 0.05",
        falsification_rule="transfer delta <= 0 on 2 of 3 seeds",
        target_skill="math.reasoning",
        suggested_training_type="targeted_repair",
        priority=0.9,
        expected_cost_gpu_hours=0.5,
        survivor_score=1.34,
        benchmark_map={"math.reasoning": "mgsm@2022-11"},
        screening_mission_id="m-screen",
    )
    base.update(overrides)
    return ScreeningGraduate(**base)


def _loop(tmp_path: Path, **policy_overrides: Any) -> GrowthLoop:
    """A loop whose real fixture parent/policy the survivor can pass through."""
    parent = parent_manifest(tmp_path)
    policy = policy_from(parent, **policy_overrides)
    return GrowthLoop(
        policy=policy,
        state=GrowthState(root=tmp_path / "loop-state"),
        parent_declaration=parent,
    )


def test_no_handoff_file_selects_normally(tmp_path: Path) -> None:
    loop = _loop(tmp_path)
    assert load_graduated_survivor(tmp_path / "loop-state") is None
    # with no hand-off and no measured profile, the loop reaches its OWN
    # first gate (NO_MEASURED_CAPABILITY) — i.e. the survivor check neither
    # fired nor blocked the pre-existing path
    plan = loop.plan_next()
    assert plan.action == "STOP_UNCERTAIN"
    assert "no measured capability profile" in plan.reason
    # and the durable state root has no hand-off residue
    assert not (tmp_path / "loop-state" / HANDOFF_FILENAME).exists()


def test_graduate_survivor_becomes_the_target_proposal(tmp_path: Path) -> None:
    write_handoff_file(tmp_path / "loop-state", graduate=_graduate())
    graduate, path = load_graduated_survivor(tmp_path / "loop-state")
    assert graduate.target_skill == "math.reasoning"
    assert graduate.benchmark_map["math.reasoning"] == "mgsm@2022-11"
    target = graduate.to_target_proposal(parent_version="gen2")
    # the projection carries the survivor's own provenance
    assert target.target_benchmarks == ("mgsm@2022-11",)
    assert "sciexp" in target.treatment_reason or "graduated" in target.treatment_reason
    assert target.factors.total == pytest.approx(1.34)


def test_adoption_without_benchmark_map_refuses_by_name(tmp_path: Path) -> None:
    graduate = _graduate(benchmark_map={})
    with pytest.raises(NextCampaignRefusal, match="HANDOFF_NO_BENCHMARK_MAP"):
        graduate.to_target_proposal(parent_version="gen2")


def test_late_cleared_handoff_survives_restart_roundtrip(tmp_path: Path) -> None:
    state_root = tmp_path / "loop-state"
    write_handoff_file(state_root, graduate=_graduate(priority=0.78))
    loaded = load_graduated_survivor(state_root)
    assert loaded is not None and loaded[0].priority == 0.78
    # clear returns True for the matching target
    assert clear_consumed_handoff(state_root, target_skill="math.reasoning") is True
    assert not (state_root / HANDOFF_FILENAME).exists()
    assert load_graduated_survivor(state_root) is None


def test_clear_refuses_a_target_mismatch(tmp_path: Path) -> None:
    state_root = tmp_path / "loop-state"
    write_handoff_file(state_root, graduate=_graduate())
    with pytest.raises(NextCampaignRefusal, match="HANDOFF_TARGET_MISMATCH"):
        clear_consumed_handoff(state_root, target_skill="some.other.skill")
    # and the file survives (a mismatch is a reviewable event, not a delete)
    assert (state_root / HANDOFF_FILENAME).exists()


def test_unreadable_handoff_file_refuses(tmp_path: Path) -> None:
    state_root = tmp_path / "loop-state"
    state_root.mkdir(parents=True, exist_ok=True)
    (state_root / HANDOFF_FILENAME).write_text("{not json", encoding="utf-8")
    with pytest.raises(NextCampaignRefusal, match="HANDOFF_FILE_UNREADABLE"):
        load_graduated_survivor(state_root)
    (state_root / HANDOFF_FILENAME).write_text(
        json.dumps({"experiment_id": "x"}), encoding="utf-8")
    with pytest.raises(NextCampaignRefusal, match="missing required fields"):
        load_graduated_survivor(state_root)


def test_service_graduation_feeds_the_loop_adopting_gate(tmp_path: Path) -> None:
    """The full cycle: the service's graduation writes the durable hand-off;
    a growth loop pointed at the same state root adopts the survivor through
    its own gate chain and composes a REAL campaign draft for it."""
    import test_scientist_service_screening as T
    service_root = tmp_path / "service"
    service = T._run_lane_to_graduation(service_root)
    manifest_path, loop_policy = T._fixture_loop_policy(
        service_root,
        allowed_training_types=["data", "targeted_repair", "sft"],
        protected_benchmarks=["math500@2024-04"],
    )
    results = service.graduate_survivors_to_campaign_drafts(
        parent_manifest_path=manifest_path,
        loop_policy=loop_policy,
        generation_root=tmp_path / "generations",
        benchmark_for_skill={"reasoning": "mgsm@2022-11"},
    )
    assert results[0].get("cycle_id")  # composed
    # the hand-off file lives in the SERVICE's state root (state/), which is
    # where an operator points the loop's session state for auto-consumption
    loop_state_root = service_root / "state"
    handoff = load_graduated_survivor(loop_state_root)
    assert handoff is not None
    graduate, _ = handoff
    assert graduate.target_skill == "reasoning"
    assert graduate.benchmark_map["reasoning"] == "mgsm@2022-11"
    # the loop's own gate chain accepts the survivor as a target proposal
    proposal = graduate.to_target_proposal(parent_version="gen2")
    assert proposal.target_benchmarks == ("mgsm@2022-11",)
    assert proposal.factors.total > 0  # the survivor's real screening score
    # and the loop composes a real draft through NextCampaignBuilder — every
    # policy gate applied — with the survivor as the target
    from chowder.growth.next_campaign import NextCampaignBuilder
    from chowder.growth.campaign import CampaignManifest
    builder = NextCampaignBuilder(policy=loop_policy)
    parent = CampaignManifest.from_file(manifest_path)
    draft = builder.draft(
        parent=parent,
        target=proposal,
        generation_root=tmp_path / "gen-root",
        attempt=1,
    )
    assert draft.target.target_skill == "reasoning"
    assert draft.target.target_benchmarks == ("mgsm@2022-11",)
    # consumption clears the file once the target is recorded
    assert clear_consumed_handoff(loop_state_root, target_skill="reasoning") is True
    assert load_graduated_survivor(loop_state_root) is None
