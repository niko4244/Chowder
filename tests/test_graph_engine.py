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
