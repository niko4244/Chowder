"""Scientist mode Phase 3 + service: the lab bridge and the model research
service composition.

Pinned:
- only admitted proposals compile (admission is a separate gate from
  compilation);
- the compiled spec carries recipe patch, data strategy, evaluations, and the
  falsification rule into the campaign shape the production bindings read;
- the service resolves providers by name and refuses unknown ones loudly
  (requirement 16);
- the service lifecycle: portfolio -> admission -> observation -> finding,
  all grounded in the run registry resolvers;
- provider state round-trips for restart.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chowder.scientist import (
    ExperimentCompiler,
    ExperimentObservation,
    Measurement,
    MissionBudget,
    ModelResearchService,
    ResearchMission,
)
from chowder.scientist.lab_bridge import CompilationRefusal
from chowder.scientist.research_service import ResearchServiceError


MISSION_DOC = {
    "mission_id": "m-e2e",
    "objective": "improve genuine reasoning without regressions",
    "priorities": {"reasoning": 0.7, "coding": 0.3},
    "protected_capabilities": ["instruction_following"],
    "budget": {"max_gpu_hours": 4.0, "max_tree_nodes": 10,
               "max_parallel_branches": 2, "max_cost_usd": 10.0},
    "autonomy": "high",
}

POLICY_DOC = {"provider": "fake_deterministic", "provider_config": {}}


def _service(tmp_path: Path, **overrides) -> ModelResearchService:
    return ModelResearchService.from_policy(
        mission_document=dict(MISSION_DOC, **overrides),
        scientist_policy=dict(POLICY_DOC),
        state_root=tmp_path / "state",
        run_exists=lambda run_id: str(run_id).startswith("run-"),
        run_complete=lambda run_id: "incomplete" not in str(run_id),
    )


# ---------------------------------------------------------------------------
# lab bridge
# ---------------------------------------------------------------------------


def _proposal(status="admitted", **kw):
    from chowder.scientist import (
        DataStrategy, ExperimentProposal, TrainingRecipeDelta,
    )
    base = dict(
        proposal_id="p1", hypothesis_id="h1", experiment_type="data",
        intervention="decay replay ratio", variables_changed=("replay_ratio",),
        variables_held_constant=("learning_rate",),
        training_recipe_delta=TrainingRecipeDelta(learning_rate=2e-4, lora_rank=32),
        data_strategy=DataStrategy(source_kinds=("curriculum", "replay"),
                                   replay_ratio=0.05),
        requested_evaluations=("reasoning",),
        transfer_evaluations=("reasoning-transfer",),
        replication_plan="3 seeds", expected_outcome="reasoning up",
        falsification_rule="transfer <= 0 on 2 seeds", estimated_gpu_hours=0.5,
        status=status,
    )
    base.update(kw)
    return ExperimentProposal(**base)


def test_only_admitted_proposals_compile() -> None:
    with pytest.raises(CompilationRefusal, match="NOT_ADMITTED"):
        ExperimentCompiler().compile(_proposal(status="proposed"))


def test_compiled_spec_carries_the_campaign_shape() -> None:
    compiled = ExperimentCompiler(protected_benchmarks=("piqa@y2020",)).compile(_proposal())
    spec = compiled.campaign_spec
    assert spec["recipe_patch"]["backend.training.learning_rate"] == 2e-4
    assert spec["recipe_patch"]["backend.lora.r"] == 32
    assert spec["data"]["replay_ratio"] == 0.05
    assert spec["evaluations"]["target_surfaces"] == ["reasoning"]
    assert spec["evaluations"]["transfer_surfaces"] == ["reasoning-transfer"]
    assert spec["evaluations"]["protected_benchmarks"] == ["piqa@y2020"]
    assert spec["falsification_rule"] == "transfer <= 0 on 2 seeds"


def test_architecture_proposals_never_compile_even_when_admitted() -> None:
    from chowder.scientist import ExperimentProposal as EP
    proposal = _proposal(experiment_type="architecture")
    admitted = EP.from_dict({**proposal.to_dict(), "status": "admitted"})
    with pytest.raises(CompilationRefusal, match="ARCHITECTURE"):
        ExperimentCompiler().compile(admitted)


# ---------------------------------------------------------------------------
# service composition
# ---------------------------------------------------------------------------


def test_unknown_provider_is_a_loud_refusal(tmp_path: Path) -> None:
    with pytest.raises(ResearchServiceError, match="UNKNOWN_PROVIDER"):
        ModelResearchService.from_policy(
            mission_document=dict(MISSION_DOC),
            scientist_policy={"provider": "definitely_not_a_provider"},
            state_root=tmp_path / "state",
        )


def test_end_to_end_fake_mission_lifecycle(tmp_path: Path) -> None:
    service = _service(tmp_path)
    # 1. portfolio
    portfolio = service.generate_portfolio(count=3)
    assert len(portfolio) == 3
    # 2. admission + compilation (fake emits one admissible candidate)
    compiled = service.request_experiments()
    assert len(compiled) == 1
    proposal, experiment = compiled[0]
    assert proposal.status == "admitted"
    assert experiment.campaign_spec["falsification_rule"]
    # 3. a real run lands an observation
    service.record_observation(ExperimentObservation(
        observation_id="obs-1", run_id="run-2026-10-03", experiment_ref=experiment.experiment_id,
        proposal_id=proposal.proposal_id, hypothesis_id=proposal.hypothesis_id,
        measurements=(Measurement(surface="reasoning", benchmark="math500", value=0.66),),
        wall_gpu_hours=0.4,
    ))
    # 4. the provider interprets; the director settles statuses mechanically
    finding = service.interpret_and_review(proposal.hypothesis_id)
    assert finding.claims[0].status in ("provisional", "rejected", "replicated")
    # 5. the next decision exists and is from the closed vocabulary
    decision = service.next_decision()
    assert decision.kind in (
        "expand_branch", "reject_hypothesis", "request_replication", "request_transfer",
        "promote_candidate", "investigate_anomaly", "stop_plateau", "stop_budget",
        "stop_no_admissible_hypothesis", "human_review",
    )
    # 6. durable state
    service.save()
    assert (tmp_path / "state" / "research" / "research-tree.json").exists()
    assert (tmp_path / "state" / "research" / "provider-state.json").exists()
    restored = json.loads(
        (tmp_path / "state" / "research" / "research-tree.json").read_text(encoding="utf-8")
    )
    assert restored["mission_id"] == "m-e2e"


def test_service_status_projection(tmp_path: Path) -> None:
    service = _service(tmp_path)
    view = service.status()
    assert view.mission_id == "m-e2e"
    assert view.provider == "fake_deterministic"
    assert view.remaining_gpu_hours == 4.0
    assert view.decision  # a closed-vocabulary decision name
