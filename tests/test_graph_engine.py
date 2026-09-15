import pytest

from chowder.engine import EvolutionEngine
from chowder.graph import ExperimentGraph, GraphInvariantError
from chowder.models import Experiment, ExperimentResult, GateDecision, Goal, Hypothesis, MetricTarget
from chowder.tournament import RankedCandidate


def exp(i: str, parent: str | None = None, hours: float = 1):
    return Experiment(i, parent, Hypothesis("obs", "cause", "fix"), {}, hours)


def test_graph_requires_known_parent():
    graph = ExperimentGraph()
    try:
        graph.add(exp("b", "missing"))
    except GraphInvariantError:
        pass
    else:
        raise AssertionError("expected GraphInvariantError")


def test_engine_enforces_parallelism_and_budget_across_proposal_waves():
    goal = Goal((MetricTarget("score", minimum=1),), gpu_hour_budget=3, max_parallel_candidates=2)
    engine = EvolutionEngine(goal, ExperimentResult("base", {"score": 1}, 0))
    first = engine.propose((exp("a"), exp("b"), exp("c")))
    second = engine.propose((exp("d"),))
    assert [e.experiment_id for e in first] == ["a", "b"]
    assert second == ()
    assert engine.reserved_gpu_hours == 2
    assert engine.remaining_budget == 1


def test_adjudication_releases_reservation_and_accounts_actual_cost():
    goal = Goal((MetricTarget("score", minimum=1),), gpu_hour_budget=3, max_parallel_candidates=2)
    engine = EvolutionEngine(goal, ExperimentResult("base", {"score": 1}, 0))
    engine.propose((exp("a", hours=1),))
    engine.adjudicate((ExperimentResult("a", {"score": 2}, 0.7),))
    assert engine.reserved_gpu_hours == 0
    assert engine.spent_gpu_hours == 0.7
    assert engine.remaining_budget == 2.3


def test_config_resolution_applies_parent_patches_root_to_child_without_mutation():
    graph = ExperimentGraph()
    root = exp("root")
    root.config_patch = {
        "model": {"id": "base", "max_length": 512},
        "training": {"lr": 1e-4, "epochs": 2},
    }
    child = exp("child", "root")
    child.config_patch = {"training": {"lr": 2e-4}, "lora": {"r": 16}}
    graph.add(root)
    graph.add(child)

    base = {"training": {"batch_size": 1}, "seed": 7}
    resolved = graph.resolve_config("child", base)

    assert resolved == {
        "training": {"batch_size": 1, "lr": 2e-4, "epochs": 2},
        "seed": 7,
        "model": {"id": "base", "max_length": 512},
        "lora": {"r": 16},
    }
    assert base == {"training": {"batch_size": 1}, "seed": 7}
    assert root.config_patch["training"]["lr"] == 1e-4


# --- deferred baselines (the amortized-eval project path) ---------------------


def test_set_baseline_completes_a_deferred_baseline_measurement():
    """A deferred baseline is a real measurement the engine does not have yet:
    the gate must never run against a placeholder, and set_baseline must be
    the only way to supply the missing numbers."""
    goal = Goal((MetricTarget("score", minimum=1),), gpu_hour_budget=3, max_parallel_candidates=2)
    engine = EvolutionEngine(goal, None, baseline_deferred=True)
    assert engine.baseline_deferred is True
    engine.set_baseline(ExperimentResult("base", {"score": 1}, 0.1))
    assert engine.baseline_deferred is False
    assert engine.baseline.metrics == {"score": 1}


def test_gate_time_methods_refuse_while_the_baseline_is_deferred():
    """Adjudication and promotion against an unmeasured baseline would rank a
    candidate against fiction; both refuse until the measurement lands."""
    engine = EvolutionEngine(
        Goal((MetricTarget("score", minimum=1),), gpu_hour_budget=3, max_parallel_candidates=2),
        None,
        baseline_deferred=True,
    )
    engine.propose((exp("a", hours=1),))
    with pytest.raises(RuntimeError, match="deferred baseline"):
        engine.adjudicate((ExperimentResult("a", {"score": 2}, 0.7),))
    assert engine.has_reservation("a"), (
        "a refused adjudication must not settle the reservation"
    )


def test_promote_refuses_a_nonempty_ranking_while_the_baseline_is_deferred():
    """A non-empty ranking against a deferred baseline would mean the gate
    ranked against fiction; an empty one promotes nothing and is the normal
    shape of a failed generation."""
    engine = EvolutionEngine(
        Goal((MetricTarget("score", minimum=1),), gpu_hour_budget=3, max_parallel_candidates=2),
        None,
        baseline_deferred=True,
    )
    fabricated = RankedCandidate(
        result=ExperimentResult("a", {"score": 2}, 0.7),
        decision=GateDecision(
            accepted=True,
            score=1.0,
            regressions={},
            unmet_targets=(),
            missing_metrics=(),
            goal_met=True,
            reason="fabricated",
        ),
        efficiency=1.0,
    )
    with pytest.raises(RuntimeError, match="deferred baseline"):
        engine.promote((fabricated,))
    assert engine.promote(()) is None
    assert engine.baseline is None


def test_set_baseline_refuses_a_second_measurement():
    engine = EvolutionEngine(
        Goal((MetricTarget("score", minimum=1),), gpu_hour_budget=3, max_parallel_candidates=2),
        None,
        baseline_deferred=True,
    )
    engine.set_baseline(ExperimentResult("base", {"score": 1.5}, 0.1))
    with pytest.raises(RuntimeError, match="already measured"):
        engine.set_baseline(ExperimentResult("base", {"score": 2.0}, 0.1))


def test_an_engine_declaring_both_modes_is_a_contradiction():
    """A deferred engine holds no baseline (there is no honest placeholder),
    and a baseline-bearing engine is not deferred."""
    goal = Goal((MetricTarget("score", minimum=1),), gpu_hour_budget=3, max_parallel_candidates=2)
    with pytest.raises(ValueError, match="baseline_deferred"):
        EvolutionEngine(goal, ExperimentResult("base", {"score": 1}, 0.1), baseline_deferred=True)
    with pytest.raises(ValueError, match="baseline is required"):
        EvolutionEngine(goal, None)


def test_set_baseline_reconciles_the_provisional_spend_with_the_measured_cost():
    """The engine starts with the provisional estimate as its spend so budget
    admission is conservative; the measured cost replaces it, not stacks."""
    goal = Goal((MetricTarget("score", minimum=1),), gpu_hour_budget=3, max_parallel_candidates=2)
    engine = EvolutionEngine(
        goal, None, spent_gpu_hours=0.01, baseline_deferred=True
    )
    engine.set_baseline(ExperimentResult("base", {"score": 1}, 0.25))
    assert engine.spent_gpu_hours == 0.25
    assert engine.remaining_budget == 2.75
