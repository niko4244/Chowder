"""Scientist mode: director portfolio flow, tree competition and memory
durability (Phase 1 + Phase 4 behaviors):

8.  resume reconstructs the same research tree
12. a branch that repeatedly fails loses allocation
    (also: portfolio generation is provider-mediated; refusals are journaled;
    the two-ledger rule; plateau stopping)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chowder.scientist import (
    Hypothesis,
    Measurement,
    MissionBudget,
    ResearchMemory,
    ResearchMission,
    ResearchQuestion,
    ResearchTree,
)
from chowder.scientist.providers.fake import FakeDeterministicScientistProvider
from chowder.scientist.research_director import DirectorRefusal, ResearchDirector
from chowder.scientist.research_tree import ResearchBranch, ResearchNode


def _mission(**kw) -> ResearchMission:
    base = dict(
        mission_id="m-tree", objective="improve reasoning",
        priorities={"reasoning": 1.0},
        budget=MissionBudget(max_gpu_hours=8.0, max_tree_nodes=12, max_parallel_branches=3),
    )
    base.update(kw)
    return ResearchMission(**base)


def _director(tmp_path: Path) -> ResearchDirector:
    mission = _mission()
    return ResearchDirector(
        mission=mission,
        provider=FakeDeterministicScientistProvider(),
        memory=ResearchMemory(tmp_path / "research"),
        tree=ResearchTree(mission_id=mission.mission_id),
    )


# ---------------------------------------------------------------------------
# portfolio flow with refusals journaled
# ---------------------------------------------------------------------------


def test_fake_portfolio_generates_admits_and_journals_refusals(tmp_path: Path) -> None:
    """The fake emits three candidates across its three hypotheses — one
    admissible, one over-budget, one architecture-type. Admission compiles
    exactly the admissible one and journals the two refusals with reasons."""
    director = _director(tmp_path)
    context = director.export_provider_context()
    portfolio = director.generate_portfolio(context, count=3)
    assert len(portfolio) == 3
    assert all(h.provider == "fake_deterministic" for h in portfolio)
    assert all(h.mission_id == "m-tree" for h in portfolio)

    compiled = director.request_experiments(context, tuple(portfolio))
    assert len(compiled) == 1
    assert compiled[0][0].status == "admitted"
    refusals = (tmp_path / "research" / "refusals.jsonl").read_text().splitlines()
    assert len(refusals) >= 2
    refusal_blob = "\n".join(refusals)
    assert "EXPERIMENT_TOO_EXPENSIVE" in refusal_blob
    assert "EXPERIMENT_TYPE_NOT_ALLOWED" in refusal_blob
    # refused proposals are journaled in the proposals ledger too, with reasons
    proposal_rows = (tmp_path / "research" / "experiment-proposals.jsonl").read_text().splitlines()
    import json as _json
    admitted_flags = [_json.loads(r)["admitted"] for r in proposal_rows]
    assert admitted_flags.count(False) == 2 and admitted_flags.count(True) == 1


def test_experiments_cannot_reference_unknown_hypotheses(tmp_path: Path) -> None:
    director = _director(tmp_path)
    orphan = _proposal_for("h-ghost")
    reasons = director.admit_proposal(orphan)
    assert any("HYPOTHESIS_UNKNOWN" in r for r in reasons)


def _proposal_for(hypothesis_id: str):
    from chowder.scientist import (
        DataStrategy, ExperimentProposal, TrainingRecipeDelta,
    )
    return ExperimentProposal(
        proposal_id=f"p-{hypothesis_id}", hypothesis_id=hypothesis_id,
        experiment_type="data", intervention="decay replay",
        variables_changed=("replay_ratio",),
        variables_held_constant=("learning_rate",),
        training_recipe_delta=TrainingRecipeDelta(epochs=2),
        data_strategy=DataStrategy(source_kinds=("curriculum",)),
        requested_evaluations=("reasoning",),
        expected_outcome="up", falsification_rule="delta <= 0",
        estimated_gpu_hours=0.5,
    )


# ---------------------------------------------------------------------------
# requirement 8: resume reconstructs the same tree
# ---------------------------------------------------------------------------


def test_resume_reconstructs_tree_identically(tmp_path: Path) -> None:
    tree = ResearchTree(mission_id="m-tree")
    tree.add_branch(ResearchBranch(branch_id="b1", hypothesis_id="h1", root_node_id="n1"))
    tree.add_node(ResearchNode(node_id="n1", parent_id=None, hypothesis_id="h1",
                               proposal_id="p1", capability_delta=0.1, depth=0))
    tree.add_node(ResearchNode(node_id="n2", parent_id="n1", hypothesis_id="h1",
                               proposal_id="p2", capability_delta=-0.05, depth=1))
    path = tmp_path / "tree.json"
    tree.save(path)

    restored = ResearchTree.load(path)
    assert restored.to_dict() == tree.to_dict()
    assert restored.node("n2").depth == 1
    assert restored.node("n2").parent_id == "n1"


def test_tree_roundtrip_survives_unknown_status_refusal(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="status"):
        ResearchNode(node_id="n", parent_id=None, hypothesis_id="h",
                     proposal_id="p", status="vibes")


# ---------------------------------------------------------------------------
# requirement 12: repeated failure loses allocation
# ---------------------------------------------------------------------------


def test_repeatedly_failing_branch_is_pruned(tmp_path: Path) -> None:
    tree = ResearchTree(mission_id="m-tree")
    tree.add_branch(ResearchBranch(branch_id="b-fail", hypothesis_id="h-fail",
                                   root_node_id="n-f1", failed_mechanism_repeats=2))
    tree.add_branch(ResearchBranch(branch_id="b-ok", hypothesis_id="h-ok",
                                   root_node_id="n-o1"))
    pruned = tree.prune_repeated_failures(max_repeats=2)
    assert pruned == ("b-fail",)
    assert tree.branch("b-fail").status == "pruned"
    assert tree.branch("b-ok").status == "active"


def test_budget_elimination_keeps_top_branches(tmp_path: Path) -> None:
    tree = ResearchTree(mission_id="m-tree")
    for i, delta in enumerate((0.3, 0.05, -0.1)):
        hyp = f"h{i}"
        tree.add_branch(ResearchBranch(branch_id=f"b{i}", hypothesis_id=hyp,
                                       root_node_id=f"n{i}"))
        tree.add_node(ResearchNode(node_id=f"n{i}", parent_id=None, hypothesis_id=hyp,
                                   proposal_id=f"p{i}", capability_delta=delta))
    survivors = tree.budget_survivors(keep_top=1)
    assert [b.branch_id for b in survivors] == ["b0"]
    assert tree.branch("b1").status == "pruned"
    assert tree.branch("b2").status == "pruned"
    assert tree.branch("b0").status == "leading"


def test_plateau_detection_fires_on_flat_window(tmp_path: Path) -> None:
    tree = ResearchTree(mission_id="m-tree")
    for i, delta in enumerate((0.2, 0.0, 0.0, 0.0)):
        tree.add_node(ResearchNode(node_id=f"n{i}", parent_id=None,
                                   hypothesis_id="h", proposal_id=f"p{i}",
                                   capability_delta=delta, status="observed"))
    assert tree.plateaued(window=3)
    # a rising window is not a plateau
    tree2 = ResearchTree(mission_id="m-tree")
    for i, delta in enumerate((0.0, 0.1, 0.2, 0.3)):
        tree2.add_node(ResearchNode(node_id=f"n{i}", parent_id=None,
                                    hypothesis_id="h", proposal_id=f"p{i}",
                                    capability_delta=delta, status="observed"))
    assert not tree2.plateaued(window=3)


# ---------------------------------------------------------------------------
# two-ledger rule: mission spend is charged from measured observations only
# ---------------------------------------------------------------------------


def test_mission_spend_charged_from_measured_costs(tmp_path: Path) -> None:
    director = _director(tmp_path)
    assert director.spend.spent_gpu_hours == 0.0
    from chowder.scientist import ExperimentObservation
    director.record_observation(ExperimentObservation(
        observation_id="obs-1", run_id="run-1", experiment_ref="sciexp-p1",
        proposal_id="p1", hypothesis_id="h1",
        measurements=(Measurement(surface="reasoning", benchmark="b", value=0.6),),
        wall_gpu_hours=0.75,
    ))
    assert director.spend.spent_gpu_hours == pytest.approx(0.75)
    # and the node landed on the tree with the same cost
    node = director.tree.node("node-sciexp-p1")
    assert node.measured_cost_gpu_hours == pytest.approx(0.75)


def test_next_decision_stops_when_node_budget_exhausted(tmp_path: Path) -> None:
    director = _director(tmp_path)
    director.spend.spent_nodes = director.spend.budget.max_tree_nodes
    decision = director.next_decision()
    assert decision.kind == "stop_budget"
    assert decision.terminal
