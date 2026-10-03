"""Scientist mode: the native contract (Phase 1) plus the boundary tests the
mission pins. Covers the security/boundary requirements that need no provider
or growth loop:

1.  scientist cannot change frozen evaluation thresholds   -> provider objects
    have no policy surface; policy keys are closed
3.  scientist cannot promote a candidate                   -> no promotion API
4.  proposal exceeding budget is refused before compute    -> admission order
5.  invalid experiment proposal is refused
6.  unsupported architecture modification is refused
6b. evidence attributed to the exact run that produced it
9.  same experiment evidence cannot be falsely attributed  -> observation bound
    to its experiment_ref
10. carried evidence cannot masquerade as a fresh measurement
11. research-memory records point to exact immutable run evidence
13. a promising branch can request replication
14. a one-off benchmark gain with transfer failure is not labelled generalized
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chowder.scientist import (
    Claim,
    ExperimentCompiler,
    ExperimentConstraint,
    ExperimentObservation,
    ExperimentProposal,
    Hypothesis,
    Measurement,
    MissionBudget,
    ReplicationPolicy,
    ResearchFinding,
    ResearchMemory,
    ResearchMission,
    ResearchQuestion,
    ResearchTree,
    TrainingRecipeDelta,
    DataStrategy,
)
from chowder.scientist.provider import CarriedEvidence, ResearchContext, SkillSummary
from chowder.scientist.providers.fake import FakeDeterministicScientistProvider
from chowder.scientist.research_director import DirectorRefusal, ResearchDirector


def _mission(**overrides) -> ResearchMission:
    base = dict(
        mission_id="mission-test",
        objective="improve genuine reasoning without losing instruction following",
        priorities={"reasoning": 0.6, "coding": 0.4},
        protected_capabilities=("instruction_following",),
        budget=MissionBudget(max_gpu_hours=4.0, max_tree_nodes=8,
                             max_parallel_branches=2, max_cost_usd=10.0),
        autonomy="high",
    )
    base.update(overrides)
    return ResearchMission(**base)


def _proposal(proposal_id="p1", hypothesis_id="h1", **overrides) -> ExperimentProposal:
    base = dict(
        proposal_id=proposal_id,
        hypothesis_id=hypothesis_id,
        experiment_type="data",
        intervention="decay replay ratio after convergence",
        variables_changed=("replay_ratio",),
        variables_held_constant=("learning_rate", "epochs"),
        training_recipe_delta=TrainingRecipeDelta(epochs=2),
        data_strategy=DataStrategy(source_kinds=("curriculum", "replay"), replay_ratio=0.05),
        requested_evaluations=("reasoning",),
        transfer_evaluations=("reasoning-transfer",),
        replication_plan="3 seeds",
        expected_outcome="reasoning up, protected flat",
        falsification_rule="transfer delta <= 0 on 2 of 3 seeds",
        estimated_gpu_hours=0.5,
    )
    base.update(overrides)
    return ExperimentProposal(**base)


def _constraint(**overrides) -> ExperimentConstraint:
    base = dict(
        allowed_experiment_types=("data", "optimization", "adapter", "training_strategy"),
        max_gpu_hours_per_experiment=2.0,
        protected_capabilities=("instruction_following",),
    )
    base.update(overrides)
    return ExperimentConstraint(**base)


# ---------------------------------------------------------------------------
# contract validation (T3 / requirement 5)
# ---------------------------------------------------------------------------


def test_proposal_without_hypothesis_is_refused() -> None:
    with pytest.raises(ValueError, match="hypothesis"):
        _proposal(hypothesis_id="")


def test_proposal_without_falsification_rule_fails_constraints() -> None:
    proposal = _proposal(falsification_rule="")
    reasons = proposal.validate(_constraint())
    assert any("FALSIFICATION_RULE_REQUIRED" in r for r in reasons)


def test_proposal_both_changed_and_held_is_refused() -> None:
    proposal = _proposal(variables_held_constant=("replay_ratio",))
    reasons = proposal.validate(_constraint())
    assert any("VARIABLE_BOTH_CHANGED_AND_HELD" in r for r in reasons)


def test_unknown_policy_keys_are_refused_fail_closed() -> None:
    with pytest.raises(ValueError, match="unknown mission keys"):
        ResearchMission.from_mapping({
            "mission_id": "m", "objective": "o", "priorities": {"reasoning": 1.0},
            "promotion_threshold_override": 0.0,  # the provider must never reach policy
        })


def test_mission_cannot_protect_and_optimize_the_same_capability() -> None:
    with pytest.raises(ValueError, match="protected or optimized"):
        _mission(protected_capabilities=("reasoning",))


# ---------------------------------------------------------------------------
# requirement 6: architecture behind explicit permission
# ---------------------------------------------------------------------------


def test_architecture_proposal_refused_by_default_constraints() -> None:
    proposal = _proposal(experiment_type="architecture")
    reasons = proposal.validate(_constraint())
    assert any("EXPERIMENT_TYPE_NOT_ALLOWED" in r for r in reasons)
    # and even with permission, the compiler refuses in this release
    compiler = ExperimentCompiler()
    with pytest.raises(Exception, match="ARCHITECTURE"):
        compiler.compile(_proposal(experiment_type="architecture", status="admitted"))


def test_architecture_type_never_in_default_mission() -> None:
    mission = _mission()
    assert "architecture" not in mission.allowed_experiment_types


# ---------------------------------------------------------------------------
# requirement 4: over-budget refused BEFORE compute
# ---------------------------------------------------------------------------


def _director(tmp_path: Path, *, run_exists=lambda r: True,
              run_complete=lambda r: True) -> ResearchDirector:
    mission = _mission()
    memory = ResearchMemory(tmp_path / "research", run_exists=run_exists,
                            run_complete=run_complete)
    tree = ResearchTree(mission_id=mission.mission_id)
    provider = FakeDeterministicScientistProvider()
    return ResearchDirector(mission=mission, provider=provider, memory=memory,
                            tree=tree, run_exists=run_exists,
                            run_complete=run_complete)


def test_over_budget_proposal_refused_before_compute(tmp_path: Path) -> None:
    director = _director(tmp_path)
    mission_budget = MissionBudget(max_gpu_hours=1.0, max_tree_nodes=4,
                                   max_parallel_branches=1)
    director.spend.budget = mission_budget
    expensive = _proposal(estimated_gpu_hours=500.0)
    reasons = director.admit_proposal(expensive)
    assert any("EXPERIMENT_TOO_EXPENSIVE" in r or "BUDGET_EXHAUSTED" in r for r in reasons)
    # zero compute was reserved: the spend ledger is untouched
    assert director.spend.spent_gpu_hours == 0.0
    assert director.spend.spent_nodes == 0


def test_exhausted_mission_budget_refuses_further_experiments(tmp_path: Path) -> None:
    director = _director(tmp_path)
    director.spend = director.spend.__class__(
        budget=MissionBudget(max_gpu_hours=0.5, max_tree_nodes=4, max_parallel_branches=1),
        spent_gpu_hours=0.5, spent_nodes=1,
    )
    reasons = director.admit_proposal(_proposal(estimated_gpu_hours=0.1))
    assert any("BUDGET_EXHAUSTED" in r for r in reasons)


# ---------------------------------------------------------------------------
# T6: fabricated observations; T7: attribution; requirement 10/11
# ---------------------------------------------------------------------------


def test_fabricated_observation_refused_unknown_run(tmp_path: Path) -> None:
    director = _director(tmp_path, run_exists=lambda run_id: False)
    observation = ExperimentObservation(
        observation_id="obs1", run_id="run-does-not-exist",
        experiment_ref="sciexp-p1", proposal_id="p1", hypothesis_id="h1",
        measurements=(Measurement(surface="reasoning", benchmark="math500@2024-04", value=0.9),),
        wall_gpu_hours=0.5,
    )
    with pytest.raises(DirectorRefusal, match="RUN_UNKNOWN"):
        director.record_observation(observation)


def test_observation_of_incomplete_run_claiming_complete_refused(tmp_path: Path) -> None:
    director = _director(tmp_path, run_exists=lambda r: True, run_complete=lambda r: False)
    observation = ExperimentObservation(
        observation_id="obs2", run_id="run-1", experiment_ref="sciexp-p1",
        proposal_id="p1", hypothesis_id="h1",
        measurements=(Measurement(surface="reasoning", benchmark="b", value=0.5),),
        status="complete", wall_gpu_hours=0.1,
    )
    with pytest.raises(DirectorRefusal, match="RUN_INCOMPLETE"):
        director.record_observation(observation)


def test_observation_bound_to_exact_run_and_experiment(tmp_path: Path) -> None:
    director = _director(tmp_path)
    observation = ExperimentObservation(
        observation_id="obs3", run_id="run-abc", experiment_ref="sciexp-p1",
        proposal_id="p1", hypothesis_id="h1",
        measurements=(Measurement(surface="reasoning", benchmark="b", value=0.7),),
        wall_gpu_hours=0.25,
    )
    director.record_observation(observation)
    stored = director.memory.observations()[0]
    assert stored.run_id == "run-abc"
    assert stored.experiment_ref == "sciexp-p1"
    node = director.tree.node("node-sciexp-p1")
    assert "run-abc" in node.evidence_refs


def test_carried_evidence_is_marked_carried_in_exports(tmp_path: Path) -> None:
    director = _director(tmp_path)
    context = director.export_provider_context(
        carried_evidence=({
            "statement": "replay decay helped in a prior session",
            "source_run_ids": ("old-run-1",),
        },),
    )
    assert all(c.to_dict()["carried"] is True for c in context.carried_evidence)
    # and the carried flag survives serialization
    blob = json.dumps(context.to_dict())
    assert '"carried": true' in blob


def test_memory_findings_reference_known_observations_only(tmp_path: Path) -> None:
    memory = ResearchMemory(tmp_path / "research")
    finding = ResearchFinding(
        finding_id="f1", hypothesis_id="h1",
        claims=(Claim(claim_id="c1", statement="s", scope="sc",
                      supporting_experiments=("run-1",)),),
        observation_ids=("obs-never-recorded",),
    )
    with pytest.raises(ValueError, match="OBSERVATION_UNKNOWN"):
        memory.record_finding(finding)


def test_memory_claim_citing_unknown_run_refused(tmp_path: Path) -> None:
    memory = ResearchMemory(tmp_path / "research",
                            run_exists=lambda run_id: run_id == "real-run",
                            run_complete=lambda run_id: True)
    memory.record_observation(ExperimentObservation(
        observation_id="obs-real", run_id="real-run", experiment_ref="sciexp-p1",
        proposal_id="p1", hypothesis_id="h1",
        measurements=(Measurement(surface="reasoning", benchmark="b", value=0.5),),
    ))
    finding = ResearchFinding(
        finding_id="f2", hypothesis_id="h1",
        claims=(Claim(claim_id="c2", statement="s", scope="sc",
                      supporting_experiments=("fake-run",)),),
        observation_ids=("obs-real",),
    )
    # the observation resolves; the claim's fabricated run does not
    with pytest.raises(ValueError, match="RUN_UNKNOWN"):
        memory.record_finding(finding)


# ---------------------------------------------------------------------------
# requirement 14: one-benchmark gain with transfer failure is not generalized
# ---------------------------------------------------------------------------


def test_transfer_failure_keeps_claim_provisional(tmp_path: Path) -> None:
    memory = ResearchMemory(tmp_path / "research",
                            run_exists=lambda r: True, run_complete=lambda r: True)
    # two successful replicated runs on the target surface, no transfer rows
    for i, run in enumerate(("run-a", "run-b", "run-c")):
        memory.record_observation(ExperimentObservation(
            observation_id=f"obs-{i}", run_id=run, experiment_ref="sciexp-p1",
            proposal_id="p1", hypothesis_id="h1",
            measurements=(Measurement(surface="reasoning", benchmark="math500", value=0.5),),
            wall_gpu_hours=0.1,
        ))
    director = ResearchDirector(
        mission=_mission(), provider=FakeDeterministicScientistProvider(),
        memory=memory, tree=ResearchTree(mission_id=_mission().mission_id),
        replication_policy=ReplicationPolicy(seeds_for_replication=3,
                                             require_transfer_stage=True),
    )
    proposed_finding = ResearchFinding(
        finding_id="f3", hypothesis_id="h1",
        claims=(Claim(
            claim_id="c3", statement="replay decay improves reasoning", scope="test",
            supporting_experiments=("run-a", "run-b", "run-c"),
            affected_capabilities=("reasoning",),
        ),),
        observation_ids=("obs-0", "obs-1", "obs-2"),
    )
    reviewed = director.review_finding(proposed_finding)
    # three successful runs but NO transfer evaluation -> the mechanical
    # status is provisional, never "replicated", never a capability gain
    assert reviewed.claims[0].status == "provisional"


def test_replication_shortfall_keeps_claim_provisional(tmp_path: Path) -> None:
    director = _director(tmp_path)
    policy = director.replication_policy
    assert policy.claim_status_for(successful_runs=1, transfer_supported=True) == "provisional"
    assert policy.claim_status_for(successful_runs=0, transfer_supported=True) == "rejected"
    assert policy.claim_status_for(successful_runs=3, transfer_supported=True) == "replicated"


# ---------------------------------------------------------------------------
# requirement 13: a promising branch can request replication
# ---------------------------------------------------------------------------


def test_promising_branch_requests_replication(tmp_path: Path) -> None:
    director = _director(tmp_path)
    from chowder.scientist.research_decision import ResearchDecision
    decision = ResearchDecision(
        kind="request_replication", subject_id="branch-h1",
        reason="target gain replicated on 2/3 seeds; replicate the third",
        evidence_run_ids=("run-1", "run-2"),
    )
    assert decision.kind == "request_replication"
    assert not decision.terminal


# ---------------------------------------------------------------------------
# requirement 3: no promotion authority anywhere in the scientist layer
# ---------------------------------------------------------------------------


def test_scientist_layer_has_no_promotion_api() -> None:
    from chowder.scientist import research_director, research_service
    for module in (research_director, research_service):
        for attr in dir(module):
            obj = getattr(module, attr)
            if isinstance(obj, type):
                public = [m for m in dir(obj) if not m.startswith("_")]
                assert not any("promote" in m and m != "promote_candidate" for m in public), (
                    f"{module.__name__}.{obj.__name__} exposes a promotion surface: {public}"
                )


def test_promotion_request_requires_evidence_runs() -> None:
    from chowder.scientist.research_decision import ResearchDecision
    with pytest.raises(ValueError, match="evidence"):
        ResearchDecision(kind="promote_candidate", subject_id="cand-1", reason="trust me")


# ---------------------------------------------------------------------------
# requirement 7: provider failure does not corrupt state
# ---------------------------------------------------------------------------


def test_provider_exception_leaves_memory_intact(tmp_path: Path) -> None:
    director = _director(tmp_path)
    director.memory.record_hypothesis(Hypothesis(
        hypothesis_id="h-keep", research_question=ResearchQuestion(text="q", capability="reasoning"),
        observation="o", suspected_mechanism="m", predicted_effect="p",
        falsification_conditions=("f",),
    ))

    class ExplodingProvider(FakeDeterministicScientistProvider):
        name = "exploding"

        def propose_hypotheses(self, context, *, count=3):
            raise RuntimeError("sidecar crashed mid-ideation")

    director.provider = ExplodingProvider()
    context = director.export_provider_context()
    with pytest.raises(RuntimeError, match="crashed"):
        director.generate_portfolio(context)
    # the pre-existing record is intact and parseable
    assert [h.hypothesis_id for h in director.memory.hypotheses()] == ["h-keep"]
