import math

import pytest

from chowder.candidate_selection import _ArmStatistics, _ucb1_score, dotted_paths, prioritize_candidates
from chowder.models import Experiment, ExperimentResult, Goal, Hypothesis, MetricTarget


def _experiment(experiment_id, config_patch):
    return Experiment(experiment_id, None, Hypothesis("o", "c", "i"), config_patch, 1.0)


def _result(experiment_id, accuracy, gpu_hours=1.0):
    return ExperimentResult(experiment_id, {"accuracy": accuracy}, gpu_hours)


@pytest.fixture
def goal():
    return Goal(metrics=(MetricTarget(name="accuracy", weight=1.0, minimum=0.0),), gpu_hour_budget=100.0)


@pytest.fixture
def baseline():
    return _result("baseline", 0.80)


def test_dotted_paths_flattens_nested_config_patch():
    paths = dotted_paths({"backend": {"training": {"learning_rate": 1e-3}, "lora": {"r": 8}}})
    assert paths == frozenset({"backend.training.learning_rate", "backend.lora.r"})


def test_dotted_paths_ignores_leaf_values():
    a = dotted_paths({"backend": {"training": {"learning_rate": 1e-3}}})
    b = dotted_paths({"backend": {"training": {"learning_rate": 5e-5}}})
    assert a == b


def test_ucb1_score_is_infinite_for_untried_arm():
    assert _ucb1_score(_ArmStatistics(), total_pulls=10) == float("inf")


def test_ucb1_score_matches_manual_formula():
    arm = _ArmStatistics(pulls=3, total_reward=1.5)
    score = _ucb1_score(arm, total_pulls=10)
    expected = 0.5 + 2.0 * math.sqrt(math.log(10) / 3)
    assert score == pytest.approx(expected)


def test_prioritize_candidates_empty_input_returns_empty_tuple(goal, baseline):
    assert prioritize_candidates([], history=(), goal=goal, baseline=baseline) == ()


def test_prioritize_candidates_cold_start_preserves_input_order(goal, baseline):
    c1 = _experiment("c1", {"backend": {"training": {"learning_rate": 1e-3}}})
    c2 = _experiment("c2", {"backend": {"lora": {"r": 16}}})
    c3 = _experiment("c3", {"backend": {"training": {"warmup_steps": 10}}})

    ordered = prioritize_candidates([c1, c2, c3], history=(), goal=goal, baseline=baseline)

    assert [e.experiment_id for e in ordered] == ["c1", "c2", "c3"]


def test_prioritize_candidates_prefers_untried_arm_over_any_tried_arm(goal, baseline):
    lr_good = _experiment("hist_lr_good", {"backend": {"training": {"learning_rate": 5e-4}}})
    history = [(lr_good, _result("hist_lr_good", 0.95))]

    cand_lr = _experiment("cand_lr", {"backend": {"training": {"learning_rate": 7e-4}}})
    cand_new = _experiment("cand_new", {"backend": {"dataset": {"max_length": 512}}})

    ordered = prioritize_candidates([cand_lr, cand_new], history=history, goal=goal, baseline=baseline)

    assert ordered[0].experiment_id == "cand_new"


def test_prioritize_candidates_ranks_historically_good_arm_above_bad_arm(goal, baseline):
    lr_good = _experiment("hist_lr_good", {"backend": {"training": {"learning_rate": 5e-4}}})
    lora_bad = _experiment("hist_lora_bad", {"backend": {"lora": {"r": 4}}})
    history = [
        (lr_good, _result("hist_lr_good", 0.95)),
        (lora_bad, _result("hist_lora_bad", 0.70)),
    ]

    cand_lr = _experiment("cand_lr", {"backend": {"training": {"learning_rate": 7e-4}}})
    cand_lora = _experiment("cand_lora", {"backend": {"lora": {"r": 32}}})

    ordered = prioritize_candidates([cand_lora, cand_lr], history=history, goal=goal, baseline=baseline)
    order_ids = [e.experiment_id for e in ordered]

    assert order_ids.index("cand_lr") < order_ids.index("cand_lora")


def test_prioritize_candidates_never_raises_when_history_has_gate_rejections(goal, baseline):
    strict_goal = Goal(metrics=(MetricTarget(name="accuracy", weight=1.0, minimum=0.9),), gpu_hour_budget=100.0)
    lr_experiment = _experiment("hist_lr", {"backend": {"training": {"learning_rate": 5e-4}}})
    regressed = _result("hist_lr", 0.10)
    history = [(lr_experiment, regressed)]

    cand_lr = _experiment("cand_lr", {"backend": {"training": {"learning_rate": 7e-4}}})
    cand_new = _experiment("cand_new", {"backend": {"dataset": {"max_length": 512}}})

    ordered = prioritize_candidates([cand_lr, cand_new], history=history, goal=strict_goal, baseline=baseline)

    assert ordered[0].experiment_id == "cand_new"


def test_prioritize_candidates_only_reorders_never_drops_or_adds(goal, baseline):
    candidates = [_experiment(f"c{i}", {"backend": {"training": {"seed": i}}}) for i in range(5)]
    ordered = prioritize_candidates(candidates, history=(), goal=goal, baseline=baseline)
    assert set(e.experiment_id for e in ordered) == {e.experiment_id for e in candidates}
    assert len(ordered) == len(candidates)
