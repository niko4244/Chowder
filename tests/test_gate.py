from chowder.gate import evaluate_candidate
from chowder.models import ExperimentResult, Goal, MetricTarget


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
