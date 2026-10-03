"""Scientist mode Phase 5: growth-loop integration, default-off.

Pinned:

15. human-review-required states remain gated where policy requires them
20. existing autonomous-growth behavior is unchanged when scientist mode is
    disabled (default) — and enabled only per treatment, by policy
-   plan/run parity: a dry run shows the same research diversion a run takes
-   the diversion is terminal and durable; resume adopts it, spent nothing
-   malformed scientist_policy documents fail closed
-   the two-ledger rule: a research diversion spends no growth envelope
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chowder.growth.growth_loop import (
    REQUIRES_HUMAN_REVIEW,
    RESEARCH_MISSION_CREATED,
    ScientistPolicyError,
    GrowthLoop,
    scientist_treatment_mode,
)
from chowder.growth.simulator import skill_profile
from chowder.growth.target_selection import GrowthState
from fixtures_growth_loop import (
    RecordingExecutor,
    outcome,
    parent_manifest,
    planned_recipes,
    policy_from,
)

START = {
    "math.reasoning": 0.62,
    "instruction.formatting": 0.31,
    "termination.control": 0.44,
}

#: The real mechanism that routes a target to a research-heavy treatment: the
#: policy declares the weak skill structurally out of reach for this training
#: path, so the selector classifies it architecture_research -> the
#: REVIEW_TREATMENTS gate the scientist diversion sits behind.
STRUCTURAL = {"structural_skills": ["instruction.formatting"]}


def _loop(tmp_path: Path, executor, *, policy=None):
    parent = parent_manifest(tmp_path)
    policy = policy or policy_from(parent)
    state = GrowthState(root=tmp_path / "growth-state")
    loop = GrowthLoop(
        policy=policy,
        state=state,
        executor=executor,
        parent_declaration=parent,
        parent_profile=skill_profile(START, generation="gen2"),
        prepare=planned_recipes,
        readiness=lambda frozen: True,
    )
    return loop, state


ScientistPolicyDocument = {
    "provider": "fake_deterministic",
    "provider_config": {},
    "state_treatments": {
        "evaluation_needed": "scientist_allowed",
        "architecture_research": "scientist_allowed",
        "untrainable_with_current_path": "scientist_then_review",
    },
    "mission": {
        "mission_id": "mission-from-growth",
        "objective": "research alternative mechanisms for the weak skill",
        "priorities": {"math.reasoning": 1.0},
        "budget": {
            "max_gpu_hours": 4.0,
            "max_tree_nodes": 8,
            "max_parallel_branches": 2,
            "max_cost_usd": 5.0,
        },
    },
}


def test_default_policy_keeps_human_review(tmp_path: Path) -> None:
    """Requirement 20: no scientist_policy -> historical behaviour."""
    executor = RecordingExecutor([])
    loop, _ = _loop(tmp_path, executor,
                    policy=policy_from(parent_manifest(tmp_path),
                                       **STRUCTURAL))
    decision = loop.plan_next()
    assert decision.action == REQUIRES_HUMAN_REVIEW
    assert executor.calls == []


def test_scientist_allowed_policy_diverts_to_research_mission(tmp_path: Path) -> None:
    """Requirement 15 inverted deliberately: the same state that would need a
    human ends the growth session with a research mission instead — but only
    because the policy says so, and the mission document travels with the
    decision for ModelResearchService to execute."""
    executor = RecordingExecutor([])
    policy = policy_from(parent_manifest(tmp_path),
                         **STRUCTURAL,
                         scientist_policy=ScientistPolicyDocument)
    loop, _ = _loop(tmp_path, executor, policy=policy)
    decision = loop.plan_next()
    assert decision.action == RESEARCH_MISSION_CREATED
    assert decision.terminal
    assert decision.mission_document is not None
    assert decision.mission_document["mission_id"] == "mission-from-growth"
    assert executor.calls == [], "a research diversion spends no campaign compute"


def test_scientist_then_review_mode_is_a_diversion_not_an_autorun(tmp_path: Path) -> None:
    """scientist_then_review diverts too — the research findings will still
    land in front of a human before anything trains; the growth loop itself
    must not be the component that enforces that second gate."""
    executor = RecordingExecutor([])
    document = {**ScientistPolicyDocument,
                "state_treatments": {"architecture_research": "scientist_then_review"}}
    policy = policy_from(parent_manifest(tmp_path), **STRUCTURAL,
                         scientist_policy=document)
    loop, _ = _loop(tmp_path, executor, policy=policy)
    decision = loop.plan_next()
    assert decision.action == RESEARCH_MISSION_CREATED


def test_diversion_decision_is_durable_and_adopted_on_resume(tmp_path: Path) -> None:
    executor = RecordingExecutor([])
    policy = policy_from(parent_manifest(tmp_path),
                         **STRUCTURAL,
                         scientist_policy=ScientistPolicyDocument)
    loop, state = _loop(tmp_path, executor, policy=policy)
    report = loop.run()
    assert report.decision.action == RESEARCH_MISSION_CREATED
    # resuming adopts the stored terminal verdict and launches nothing
    loop2, _ = _loop(tmp_path, RecordingExecutor([]), policy=policy, )
    loop2.state = state
    report2 = loop2.run(resume=True)
    assert report2.decision.action == RESEARCH_MISSION_CREATED
    assert report2.budget["spent_wall_gpu_hours"] == 0.0


def test_malformed_scientist_policy_fails_closed(tmp_path: Path) -> None:
    policy = policy_from(parent_manifest(tmp_path),
                         **STRUCTURAL,
                         scientist_policy={"backdoor": True})
    loop, _ = _loop(tmp_path, RecordingExecutor([]), policy=policy)
    with pytest.raises(ScientistPolicyError):
        loop.plan_next()


def test_unknown_treatment_mode_fails_closed(tmp_path: Path) -> None:
    policy = policy_from(parent_manifest(tmp_path),
                         **STRUCTURAL,
                         scientist_policy={
                             "state_treatments": {"architecture_research": "auto_promote"},
                         })
    loop, _ = _loop(tmp_path, RecordingExecutor([]), policy=policy)
    with pytest.raises(ScientistPolicyError):
        loop.plan_next()


def test_scientist_policy_survives_policy_roundtrip() -> None:
    from chowder.growth.next_campaign import LoopPolicy
    document = policy_from(parent_manifest(Path(".")), **STRUCTURAL,
                           scientist_policy=ScientistPolicyDocument)
    restored = LoopPolicy.from_mapping(document.to_dict(), source="roundtrip")
    assert restored.scientist_policy is not None
    assert restored.scientist_policy["provider"] == "fake_deterministic"
    # and the digest covers the scientist policy (a policy change is a policy change)
    without = policy_from(parent_manifest(Path(".")), **STRUCTURAL)
    assert restored.digest() != without.digest()


def test_treatment_mode_helper_defaults_to_human_review() -> None:
    from types import SimpleNamespace
    assert scientist_treatment_mode(SimpleNamespace(scientist_policy=None),
                                    "architecture_research") == "human_review"
    assert scientist_treatment_mode(SimpleNamespace(), "anything") == "human_review"
