import pytest

from chowder.engine import EvolutionEngine
from chowder.graph import GraphInvariantError
from chowder.models import (
    Experiment,
    ExperimentResult,
    ExperimentStatus,
    Goal,
    Hypothesis,
    MetricTarget,
)


def _experiment(name, parent=None, hours=0.2):
    return Experiment(
        experiment_id=name,
        parent_id=parent,
        hypothesis=Hypothesis("obs", "cause", "intervention"),
        config_patch={},
        estimated_gpu_hours=hours,
    )


def _engine(max_parallel=4, budget=5.0):
    engine = EvolutionEngine(
        Goal((MetricTarget("quality", minimum=0.8),), gpu_hour_budget=budget, max_parallel_candidates=max_parallel),
        ExperimentResult("baseline", {"quality": 0.7}, 0.0),
    )
    engine.graph.add(_experiment("root", hours=1.0))
    return engine


def test_propose_is_atomic_when_later_candidate_has_unknown_parent():
    engine = _engine()
    first = _experiment("first", "root")
    broken = _experiment("broken", "missing")

    with pytest.raises(GraphInvariantError, match="unknown parent"):
        engine.propose((first, broken))

    assert set(engine.graph.nodes) == {"root"}
    assert engine.outstanding_candidates == 0
    assert engine.reserved_gpu_hours == 0


def test_propose_is_atomic_for_duplicate_ids_within_batch():
    engine = _engine()
    first = _experiment("same", "root")
    duplicate = _experiment("same", "root")

    with pytest.raises(GraphInvariantError, match="duplicate experiment id"):
        engine.propose((first, duplicate))

    assert set(engine.graph.nodes) == {"root"}
    assert engine.outstanding_candidates == 0
    assert engine.reserved_gpu_hours == 0


def test_ordered_parent_child_batch_is_still_supported():
    engine = _engine()
    parent = _experiment("parent", "root", 0.2)
    child = _experiment("child", "parent", 0.3)

    accepted = engine.propose((parent, child))

    assert accepted == (parent, child)
    assert engine.graph.ancestors("child") == ("parent", "root")
    assert engine.reserved_gpu_hours == pytest.approx(0.5)
    assert engine.outstanding_candidates == 2


def test_withdraw_proposals_restores_graph_and_reserved_budget_without_spend():
    engine = _engine()
    parent = _experiment("parent", "root", 0.2)
    child = _experiment("child", "parent", 0.3)
    engine.propose((parent, child))

    removed = engine.withdraw_proposals(("parent", "child"))

    assert {row.experiment_id for row in removed} == {"parent", "child"}
    assert set(engine.graph.nodes) == {"root"}
    assert engine.reserved_gpu_hours == 0
    assert engine.spent_gpu_hours == 0
    assert engine.outstanding_candidates == 0


def test_withdraw_refuses_running_candidate():
    engine = _engine()
    candidate = _experiment("candidate", "root")
    engine.propose((candidate,))
    engine.graph.nodes["candidate"].status = ExperimentStatus.RUNNING

    with pytest.raises(ValueError, match="cannot withdraw"):
        engine.withdraw_proposals(("candidate",))

    assert engine.has_reservation("candidate")
    assert "candidate" in engine.graph.nodes


def test_fail_requires_active_reservation():
    engine = _engine()
    with pytest.raises(ValueError, match="no active reservation"):
        engine.fail("root")
    assert engine.spent_gpu_hours == 0


def test_adjudicate_rejects_unreserved_result_before_accounting():
    engine = _engine()
    result = ExperimentResult("root", {"quality": 0.9}, 1.0)

    with pytest.raises(ValueError, match="without active reservations"):
        engine.adjudicate((result,))

    assert engine.spent_gpu_hours == 0
    assert engine.reserved_gpu_hours == 0
    assert engine.baseline.metrics["quality"] == 0.7


def test_adjudicate_duplicate_results_is_atomic():
    engine = _engine()
    candidate = _experiment("candidate", "root", 0.5)
    engine.propose((candidate,))
    result = ExperimentResult("candidate", {"quality": 0.9}, 0.4)

    with pytest.raises(ValueError, match="duplicate experiment results"):
        engine.adjudicate((result, result))

    assert engine.has_reservation("candidate")
    assert engine.reserved_gpu_hours == pytest.approx(0.5)
    assert engine.spent_gpu_hours == 0
