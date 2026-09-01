from chowder.gate import evaluate_candidate
from chowder.models import ExperimentResult, Goal, MetricTarget, OptimizationDirection


def test_gate_rejects_regression_even_when_average_improves():
    goal = Goal(
        metrics=(
            MetricTarget("reasoning", minimum=80, regression_tolerance=0.5),
            MetricTarget("tools", minimum=80, regression_tolerance=1.0),
        ),
        gpu_hour_budget=10,
    )
    base = ExperimentResult("base", {"reasoning": 80, "tools": 85}, 0)
    candidate = ExperimentResult("x", {"reasoning": 90, "tools": 82}, 1)
    decision = evaluate_candidate(goal=goal, baseline=base, candidate=candidate)
    assert not decision.accepted
    assert decision.regressions["tools"] == -3


def test_gate_allows_incremental_progress_below_final_target():
    goal = Goal((MetricTarget("reasoning", minimum=90, regression_tolerance=0.5),), 10)
    base = ExperimentResult("base", {"reasoning": 80}, 0)
    candidate = ExperimentResult("x", {"reasoning": 82}, 1)
    decision = evaluate_candidate(goal=goal, baseline=base, candidate=candidate)
    assert decision.accepted
    assert not decision.goal_met
    assert decision.unmet_targets == ("reasoning",)


def test_gate_requires_complete_evidence():
    goal = Goal((MetricTarget("reasoning", minimum=80), MetricTarget("tools", minimum=80)), 10)
    base = ExperimentResult("base", {"reasoning": 80, "tools": 80}, 0)
    candidate = ExperimentResult("x", {"reasoning": 81}, 1)
    decision = evaluate_candidate(goal=goal, baseline=base, candidate=candidate)
    assert not decision.accepted
    assert decision.missing_metrics == ("tools",)


def test_gate_rewards_lower_values_for_minimize_metrics():
    goal = Goal(
        (MetricTarget(
            "toxicity",
            maximum=0.2,
            regression_tolerance=0.01,
            direction=OptimizationDirection.MINIMIZE,
        ),),
        10,
    )
    base = ExperimentResult("base", {"toxicity": 0.18}, 0)
    candidate = ExperimentResult("x", {"toxicity": 0.12}, 1)
    decision = evaluate_candidate(goal=goal, baseline=base, candidate=candidate)
    assert decision.accepted
    assert decision.score > 0
    assert decision.goal_met


def test_gate_rejects_increase_for_minimize_metric():
    goal = Goal(
        (MetricTarget(
            "latency_ms",
            maximum=100,
            regression_tolerance=2,
            direction=OptimizationDirection.MINIMIZE,
        ),),
        10,
    )
    base = ExperimentResult("base", {"latency_ms": 90}, 0)
    candidate = ExperimentResult("x", {"latency_ms": 95}, 1)
    decision = evaluate_candidate(goal=goal, baseline=base, candidate=candidate)
    assert not decision.accepted
    assert decision.regressions["latency_ms"] == -5
