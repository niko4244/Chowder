import pytest

from chowder.moe_planning import (
    ExpertImportance,
    ImportanceWeights,
    build_uniform_pruning_plan,
    score_experts,
)


def _records():
    return [
        ExpertImportance(0, 0, 10, 1.0, 1.0, 1.0),
        ExpertImportance(0, 1, 20, 2.0, 2.0, 2.0),
        ExpertImportance(0, 2, 30, 3.0, 3.0, 3.0),
        ExpertImportance(0, 3, 40, 4.0, 4.0, 4.0),
        ExpertImportance(1, 0, 40, 4.0, 4.0, 4.0),
        ExpertImportance(1, 1, 30, 3.0, 3.0, 3.0),
        ExpertImportance(1, 2, 20, 2.0, 2.0, 2.0),
        ExpertImportance(1, 3, 10, 1.0, 1.0, 1.0),
    ]


def test_uniform_half_retention_keeps_highest_scoring_experts_per_layer():
    plan = build_uniform_pruning_plan(
        _records(),
        retention_fraction=0.5,
        minimum_survivors_per_layer=1,
    )

    assert plan.layers[0].keep_experts == (2, 3)
    assert plan.layers[0].remove_experts == (0, 1)
    assert plan.layers[1].keep_experts == (0, 1)
    assert plan.layers[1].remove_experts == (2, 3)
    assert plan.actual_retention_fraction == pytest.approx(0.5)


def test_minimum_survivors_prevents_pruning_below_router_top_k_floor():
    plan = build_uniform_pruning_plan(
        _records(),
        retention_fraction=0.25,
        minimum_survivors_per_layer=2,
    )

    assert all(len(layer.keep_experts) == 2 for layer in plan.layers)


def test_score_is_deterministic_when_metrics_tie():
    records = [
        ExpertImportance(0, 1, 1, 1.0, 1.0, 1.0),
        ExpertImportance(0, 0, 1, 1.0, 1.0, 1.0),
    ]
    plan = build_uniform_pruning_plan(
        records,
        retention_fraction=0.5,
        minimum_survivors_per_layer=1,
    )

    assert plan.layers[0].keep_experts == (0,)


def test_duplicate_expert_records_are_rejected():
    record = ExpertImportance(0, 0, 1, 1.0, 1.0, 1.0)

    with pytest.raises(ValueError, match="duplicate expert importance record"):
        score_experts([record, record])


def test_importance_weights_require_at_least_one_positive_component():
    with pytest.raises(ValueError, match="at least one importance weight"):
        ImportanceWeights(0.0, 0.0, 0.0, 0.0)
