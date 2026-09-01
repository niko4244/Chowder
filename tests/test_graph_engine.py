from chowder.engine import EvolutionEngine
from chowder.graph import ExperimentGraph, GraphInvariantError
from chowder.models import Experiment, ExperimentResult, Goal, Hypothesis, MetricTarget


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
