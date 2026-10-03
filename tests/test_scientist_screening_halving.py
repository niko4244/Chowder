"""Successive halving in the screening lane (docs/COMPUTE_PROVIDERS.md §4).

Pinned:
- round budgets grow by the multiplier and saturate at the screening cap;
- survivor counting mirrors the growth library (ceil of fraction, min
  survivors, capped at the candidate count);
- settlement is deterministic: score descending, experiment_id ascending as
  the tiebreak; gate vs cutoff elimination split is explicit;
- only the FINAL round's survivors graduate;
- the director's driver advances only on recorded observations (no callback →
  refusal BEFORE any compute), journals every elimination, and survives a
  candidate whose screening submission the scheduler refused (gate);
- the final survivors graduate into submit_survivor_batch's substantial runs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chowder.scientist import ExperimentClass, ExperimentScheduler
from chowder.scientist.compute import KaggleProvider, LocalCudaProvider
from chowder.scientist.lab_bridge import CompiledExperiment
from chowder.scientist.observation import ExperimentObservation, Measurement
from chowder.scientist.proposal import (
    DataStrategy,
    ExperimentProposal,
    TrainingRecipeDelta,
)
from chowder.scientist.research_memory import ResearchMemory
from chowder.scientist.research_tree import ResearchTree
from chowder.scientist.screening_halving import (
    ScreeningHalving,
    settle_screening_round,
)


# ---------------------------------------------------------------------------
# the schedule
# ---------------------------------------------------------------------------


def test_round_budgets_grow_and_saturate_at_the_cap() -> None:
    schedule = ScreeningHalving(initial_budget_gpu_hours=0.05,
                                budget_cap_gpu_hours=0.25,
                                step_multiplier=2.0, max_rounds=4)
    assert schedule.round_budget(0) == pytest.approx(0.05)
    assert schedule.round_budget(1) == pytest.approx(0.10)
    assert schedule.round_budget(2) == pytest.approx(0.20)
    assert schedule.round_budget(3) == pytest.approx(0.25)  # capped, not 0.4
    assert schedule.round_budget(9) == pytest.approx(0.25)


def test_schedule_validation_mirrors_the_growth_library() -> None:
    with pytest.raises(ValueError, match="survival_fraction"):
        ScreeningHalving(survival_fraction=0.0)
    with pytest.raises(ValueError, match="survival_fraction"):
        ScreeningHalving(survival_fraction=1.0)
    with pytest.raises(ValueError, match="min_survivors"):
        ScreeningHalving(min_survivors=0)
    with pytest.raises(ValueError, match="step_multiplier"):
        ScreeningHalving(step_multiplier=1.0)
    with pytest.raises(ValueError, match="max_rounds"):
        ScreeningHalving(max_rounds=0)
    with pytest.raises(ValueError, match="budget_cap"):
        ScreeningHalving(initial_budget_gpu_hours=0.5, budget_cap_gpu_hours=0.25)


def test_survivor_count_mirrors_the_library() -> None:
    schedule = ScreeningHalving(survival_fraction=0.5, min_survivors=1)
    assert schedule.survivor_count(0) == 0
    assert schedule.survivor_count(8) == 4
    assert schedule.survivor_count(5) == 3  # ceil(2.5)
    assert schedule.survivor_count(1) == 1
    assert schedule.survivor_count(2) == 1
    # min_survivors floors the survivor count
    wide = ScreeningHalving(survival_fraction=0.1, min_survivors=2)
    assert wide.survivor_count(5) == 2


def test_final_round_stops_on_min_survivors_or_max_rounds() -> None:
    schedule = ScreeningHalving(min_survivors=1, max_rounds=3)
    assert schedule.is_final_round(0, survivors=3) is False
    assert schedule.is_final_round(1, survivors=1) is True   # min survivors hit
    assert schedule.is_final_round(2, survivors=4) is True   # round budget out


# ---------------------------------------------------------------------------
# deterministic settlement
# ---------------------------------------------------------------------------


def test_settlement_splits_gate_from_cutoff_and_ties_break_by_id() -> None:
    schedule = ScreeningHalving(survival_fraction=0.5, min_survivors=1)
    scores = {
        "e-b": 1.2, "e-a": 1.2,   # tie: e-a wins by id ascending
        "e-c": 0.4, "e-d": 0.1,
        "e-e": None,               # never produced an observation → gate
    }
    survivors, by_gate, by_cutoff, ranked = settle_screening_round(scores, schedule)
    assert survivors == ("e-a", "e-b")  # keep ceil(4 × 0.5) = 2
    assert by_gate == ("e-e",)
    assert by_cutoff == ("e-c", "e-d")
    assert ranked[0] == ("e-a", 1.2)    # tiebreak visible in the audit trail


def test_settlement_with_only_gated_candidates_keeps_nobody() -> None:
    schedule = ScreeningHalving()
    survivors, by_gate, by_cutoff, _ = settle_screening_round(
        {"e-x": None, "e-y": None}, schedule)
    assert survivors == ()
    assert by_gate == ("e-x", "e-y")
    assert by_cutoff == ()


# ---------------------------------------------------------------------------
# the director's driver
# ---------------------------------------------------------------------------


class _FakeProvider:
    name = "fake_deterministic"

    def available(self):
        return True

    def propose_hypotheses(self, context, *, count=3):
        return ()

    def propose_experiments(self, context, hypotheses):
        return ()

    def interpret(self, context, hypothesis, observations):
        raise NotImplementedError

    def export_state(self):
        return {}


def _halving_director(tmp_path: Path):
    from chowder.scientist import MissionBudget, ResearchMission
    from chowder.scientist.research_director import ResearchDirector
    mission = ResearchMission(
        mission_id="m-halving", objective="improve reasoning",
        priorities={"reasoning": 1.0},
        budget=MissionBudget(max_gpu_hours=8.0, max_tree_nodes=32,
                             max_parallel_branches=8),
    )
    memory = ResearchMemory(tmp_path / "research")
    tree = ResearchTree(mission_id=mission.mission_id)
    return ResearchDirector(
        mission=mission, provider=_FakeProvider(), memory=memory, tree=tree,
    ), memory, tree


def _pair(i: int):
    proposal = ExperimentProposal(
        proposal_id=f"p{i}", hypothesis_id=f"h{i}", experiment_type="data",
        intervention="x", variables_changed=("replay_ratio",),
        variables_held_constant=("lr",),
        training_recipe_delta=TrainingRecipeDelta(epochs=1),
        data_strategy=DataStrategy(source_kinds=("curriculum",)),
        requested_evaluations=("reasoning",), expected_outcome="up",
        falsification_rule="delta<=0", estimated_gpu_hours=0.25,
    )
    experiment = CompiledExperiment(
        experiment_id=f"sciexp-p{i}", proposal_id=f"p{i}", hypothesis_id=f"h{i}",
        campaign_spec={"recipe_patch": {}}, estimated_gpu_hours=0.25,
    )
    return proposal, experiment


def _record(director, triple, *, quality: float, obs_id: str, run_id: str):
    proposal, experiment, submission = triple
    director.record_observation(ExperimentObservation(
        observation_id=obs_id, run_id=run_id,
        experiment_ref=experiment.experiment_id,
        proposal_id=proposal.proposal_id, hypothesis_id=proposal.hypothesis_id,
        measurements=(Measurement(surface="reasoning", benchmark="b", value=quality),),
        wall_gpu_hours=0.05, hardware_class=submission.hardware_class,
    ))


def test_driver_refuses_before_compute_without_a_results_callback(tmp_path) -> None:
    director, _, _ = _halving_director(tmp_path)
    scheduler = ExperimentScheduler([
        KaggleProvider(username="u", api_key="k", push=False),
    ])
    with pytest.raises(Exception, match="SCREENING_RESULTS_CALLBACK_REQUIRED"):
        director.run_screening_halving(scheduler, (_pair(0),))
    assert director.spend.spent_gpu_hours == 0  # refused BEFORE any compute
    assert not director.tree.nodes()


def test_driver_runs_budget_driven_elimination_rounds(tmp_path) -> None:
    director, memory, tree = _halving_director(tmp_path)
    scheduler = ExperimentScheduler([
        KaggleProvider(username="u", api_key="k", weekly_gpu_hours=50.0, push=False),
    ])
    schedule = ScreeningHalving(initial_budget_gpu_hours=0.05,
                                budget_cap_gpu_hours=0.25,
                                step_multiplier=2.0, survival_fraction=0.5,
                                min_survivors=1, max_rounds=4)

    def record_results(triples, round_index):
        for proposal, experiment, submission in triples:
            # p3 is the star; p1 is decent; p0/p2 are weak
            quality = {"p0": 0.05, "p1": 0.60, "p2": 0.10, "p3": 0.90}[proposal.proposal_id]
            _record(director, (proposal, experiment, submission), quality=quality,
                    obs_id=f"obs-r{round_index}-{proposal.proposal_id}",
                    run_id=f"run-r{round_index}-{proposal.proposal_id}")

    outcome = director.run_screening_halving(
        scheduler, (_pair(0), _pair(1), _pair(2), _pair(3)),
        halving=schedule, record_results=record_results,
    )

    # round 0: all four candidates at 0.05 → two survivors; round 1: those
    # two at 0.10 → one survivor, which IS the final round (min_survivors
    # reached — the same stop rule as the growth library's halving)
    assert outcome.total_rounds == 2
    budgets = [r.budget_gpu_hours for r in outcome.rounds]
    assert budgets == pytest.approx([0.05, 0.10])
    assert outcome.rounds[0].submitted_experiment_ids == (
        "sciexp-p0", "sciexp-p1", "sciexp-p2", "sciexp-p3")
    assert set(outcome.rounds[0].eliminated_by_cutoff_experiment_ids) == {"sciexp-p0", "sciexp-p2"}
    assert outcome.rounds[0].survivor_experiment_ids == ("sciexp-p3", "sciexp-p1")
    # round 1 resubmits the survivors best-first at the larger budget
    assert outcome.rounds[1].submitted_experiment_ids == ("sciexp-p3", "sciexp-p1")
    assert outcome.rounds[1].eliminated_by_cutoff_experiment_ids == ("sciexp-p1",)
    assert outcome.final_survivors == ("sciexp-p3",)
    # every elimination is journaled, durable, and carries its round + score
    eliminations = [row for row in memory.refusals_path.read_text(
        encoding="utf-8").splitlines() if "screening_eliminated" in row]
    assert len(eliminations) == 3  # p0, p2 (round 0) + p1 (round 1)
    assert '"score"' in eliminations[0]
    # the outcome's own cost accounting is the schedule's, not a guess
    assert outcome.total_screening_gpu_hours == pytest.approx(4 * 0.05 + 2 * 0.10)


def test_driver_gate_elimines_scheduler_refusals(tmp_path) -> None:
    director, memory, _ = _halving_director(tmp_path)
    # kaggle quota fits only ONE screening run; the rest refuse → gate
    scheduler = ExperimentScheduler([
        KaggleProvider(username="u", api_key="k", weekly_gpu_hours=0.1, push=False),
        LocalCudaProvider(accelerators=1, weekly_gpu_hours=0.0),
    ])
    schedule = ScreeningHalving(initial_budget_gpu_hours=0.05,
                                budget_cap_gpu_hours=0.05,
                                survival_fraction=0.5, min_survivors=1,
                                max_rounds=3)

    def record_results(triples, round_index):
        for triple in triples:
            _record(director, triple, quality=0.7,
                    obs_id=f"obs-{triple[1].experiment_id}",
                    run_id=f"run-{triple[1].experiment_id}")

    outcome = director.run_screening_halving(
        scheduler, (_pair(0), _pair(1), _pair(2)),
        halving=schedule, record_results=record_results,
    )
    # p0 got the kaggle slot; p1/p2 were refused everywhere (local weekly 0.0)
    gated = outcome.rounds[0].eliminated_by_gate_experiment_ids
    assert set(gated) == {"sciexp-p1", "sciexp-p2"}
    assert outcome.rounds[0].survivor_experiment_ids == ("sciexp-p0",)
    # one survivor → final round immediately; it graduates
    assert outcome.total_rounds == 1
    assert outcome.final_survivors == ("sciexp-p0",)


def test_final_survivors_graduate_to_substantial_runs(tmp_path) -> None:
    director, _, _ = _halving_director(tmp_path)
    # local first: substantial follows declaration order, screening still
    # prefers the kaggle lane regardless of order
    scheduler = ExperimentScheduler([
        LocalCudaProvider(accelerators=1),
        KaggleProvider(username="u", api_key="k", weekly_gpu_hours=50.0, push=False),
    ])
    schedule = ScreeningHalving(initial_budget_gpu_hours=0.05, max_rounds=2)

    def record_results(triples, round_index):
        for proposal, _experiment, submission in triples:
            quality = {"p0": 0.10, "p1": 0.50, "p2": 0.80}[proposal.proposal_id]
            _record(director, (proposal, _experiment_for(proposal), submission),
                    quality=quality, obs_id=f"obs-{proposal.proposal_id}",
                    run_id=f"run-{proposal.proposal_id}")

    def _experiment_for(proposal):
        return CompiledExperiment(
            experiment_id=f"sciexp-{proposal.proposal_id}",
            proposal_id=proposal.proposal_id, hypothesis_id=proposal.hypothesis_id,
            campaign_spec={}, estimated_gpu_hours=0.25,
        )

    outcome = director.run_screening_halving(
        scheduler, (_pair(0), _pair(1), _pair(2)),
        halving=schedule, record_results=record_results,
    )
    survivors = [(_pair(i)[0], _pair(i)[1]) for i in
                 (int(eid.removeprefix("sciexp-p")) for eid in outcome.final_survivors)]
    graduated = director.submit_survivor_batch(scheduler, tuple(
        (p, e, None) for p, e in survivors), keep_top=1)
    assert len(graduated) == 1
    assert graduated[0][0].proposal_id == "p2"          # the survivor, by score
    assert graduated[0][2].experiment_class == ExperimentClass.SUBSTANTIAL
    assert graduated[0][2].provider_name == "local_cuda"  # substantial: local first
