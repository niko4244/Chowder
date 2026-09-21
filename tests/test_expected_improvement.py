import math

import pytest

from chowder.expected_improvement import (
    ArmEvidence,
    SELECTORS,
    _expected_improvement,
    backtest_selectors,
    estimate_arm_improvements,
    prioritize_by_expected_improvement,
)
from chowder.intervention_outcomes import build_intervention_outcomes
from chowder.models import (
    Experiment,
    ExperimentResult,
    ExperimentStatus,
    Goal,
    Hypothesis,
    MetricTarget,
)
from chowder.registry import RunRegistry

LR_ARM = frozenset({"backend.training.learning_rate"})
LORA_ARM = frozenset({"backend.lora.r"})
DATASET_ARM = frozenset({"backend.dataset.max_length"})

_PATCHES = {
    "lr": {"backend": {"training": {"learning_rate": 1e-3}}},
    "lora": {"backend": {"lora": {"r": 8}}},
    "dataset": {"backend": {"dataset": {"max_length": 512}}},
}


def _goal(minimum=0.0):
    return Goal((MetricTarget("quality", minimum=minimum),), gpu_hour_budget=100.0)


def _baseline(quality=0.70):
    return ExperimentResult("base", {"quality": quality}, 0.0)


def _experiment(experiment_id, config_patch):
    return Experiment(experiment_id, None, Hypothesis("o", "c", "i"), config_patch, 1.0)


def _outcomes_from_specs(tmp_path, specs, *, db_name="runs.db", baseline_quality=0.70):
    """Build genuinely registry-derived rows: record each experiment and its
    result through `RunRegistry`, then read them back through
    `build_intervention_outcomes` so the rows under test come out of the
    same join path production uses, not a hand-built dataclass.

    *specs* is `(experiment_id, arm_name, quality, gpu_hours)` in the order
    the experiments were recorded, which is the order the view returns and
    therefore the chronological order the backtest splits on.
    """
    registry = RunRegistry(tmp_path / db_name)
    for experiment_id, arm_name, quality, gpu_hours in specs:
        registry.record_experiment(_experiment(experiment_id, _PATCHES[arm_name]))
        registry.record_result(ExperimentResult(experiment_id, {"quality": quality}, gpu_hours))
    rows = build_intervention_outcomes(
        registry, goal=_goal(), baseline=_baseline(baseline_quality)
    )
    registry.close()
    return rows


# --- The estimator's statistics -------------------------------------------


def test_expected_improvement_matches_the_closed_form_gaussian_ei():
    value = _expected_improvement(mean=1.2, predictive_stdev=0.5, incumbent=1.0)
    z = (1.2 - 1.0) / 0.5
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    pdf = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    assert value == pytest.approx((1.2 - 1.0) * cdf + 0.5 * pdf)


def test_expected_improvement_of_a_zero_variance_arm_is_the_plain_gain():
    assert _expected_improvement(mean=1.5, predictive_stdev=0.0, incumbent=1.0) == pytest.approx(0.5)
    # An arm certainly worse than the incumbent offers nothing, never a negative EI.
    assert _expected_improvement(mean=0.5, predictive_stdev=0.0, incumbent=1.0) == 0.0


def test_expected_improvement_is_never_negative_even_for_a_far_worse_arm():
    assert _expected_improvement(mean=-5.0, predictive_stdev=0.5, incumbent=1.0) >= 0.0


def test_estimate_uses_sample_mean_and_predictive_stdev_of_the_arm(tmp_path):
    # quality - 0.70 over gpu_hours 1.0 gives rewards 0.20, 0.10, 0.30.
    rows = _outcomes_from_specs(
        tmp_path,
        (("a", "lr", 0.90, 1.0), ("b", "lr", 0.80, 1.0), ("c", "lr", 1.00, 1.0)),
    )
    estimate = estimate_arm_improvements(rows).estimates[LR_ARM]

    assert estimate.evidence is ArmEvidence.SUFFICIENT
    assert estimate.observations == 3
    assert estimate.mean_reward == pytest.approx(0.20)
    assert estimate.reward_stdev == pytest.approx(0.10)
    assert estimate.predictive_stdev == pytest.approx(0.10 * math.sqrt(1.0 + 1.0 / 3.0))


def test_incumbent_is_the_single_best_reward_observed_anywhere(tmp_path):
    rows = _outcomes_from_specs(
        tmp_path,
        (
            ("a", "lr", 0.90, 1.0),
            ("b", "lr", 0.80, 1.0),
            # A one-off arm holds the best reward; it still sets the incumbent.
            ("c", "lora", 0.95, 0.5),
        ),
    )
    model = estimate_arm_improvements(rows)

    assert model.incumbent_reward == pytest.approx(0.50)
    assert model.estimates[LORA_ARM].evidence is ArmEvidence.INSUFFICIENT


def test_gpu_hour_normalization_makes_a_cheap_run_worth_more(tmp_path):
    rows = _outcomes_from_specs(
        tmp_path,
        (
            ("cheap1", "lr", 0.80, 0.5),
            ("cheap2", "lr", 0.80, 0.5),
            ("dear1", "lora", 0.80, 4.0),
            ("dear2", "lora", 0.80, 4.0),
        ),
    )
    estimates = estimate_arm_improvements(rows).estimates

    assert estimates[LR_ARM].mean_reward == pytest.approx(0.20)
    assert estimates[LORA_ARM].mean_reward == pytest.approx(0.025)


# --- Explicit insufficient-evidence handling ------------------------------


def test_single_observation_arm_is_insufficient_and_has_no_point_estimate(tmp_path):
    rows = _outcomes_from_specs(tmp_path, (("only", "lr", 0.90, 1.0),))
    estimate = estimate_arm_improvements(rows).estimates[LR_ARM]

    assert estimate.evidence is ArmEvidence.INSUFFICIENT
    assert estimate.expected_improvement is None
    assert estimate.predictive_stdev is None
    # The one thing that IS a fact -- the observed value -- is still reported.
    assert estimate.observations == 1
    assert estimate.mean_reward == pytest.approx(0.20)
    assert estimate.reward_stdev is None


def test_min_observations_raises_the_bar_for_sufficiency(tmp_path):
    rows = _outcomes_from_specs(
        tmp_path, (("a", "lr", 0.90, 1.0), ("b", "lr", 0.80, 1.0))
    )
    assert estimate_arm_improvements(rows).estimates[LR_ARM].evidence is ArmEvidence.SUFFICIENT
    strict = estimate_arm_improvements(rows, min_observations=3).estimates[LR_ARM]
    assert strict.evidence is ArmEvidence.INSUFFICIENT
    assert strict.expected_improvement is None


def test_min_observations_below_two_is_rejected_rather_than_faked(tmp_path):
    rows = _outcomes_from_specs(tmp_path, (("only", "lr", 0.90, 1.0),))
    with pytest.raises(ValueError, match="min_observations"):
        estimate_arm_improvements(rows, min_observations=1)


def test_unscoreable_rows_are_excluded_and_counted_not_dropped_silently(tmp_path):
    """A goal whose metric no result carries makes the gate return -inf --
    a statement about the evaluation, not about the arm. Those rows must
    leave a visible trace rather than vanishing."""
    registry = RunRegistry(tmp_path / "runs.db")
    for experiment_id in ("a", "b"):
        registry.record_experiment(_experiment(experiment_id, _PATCHES["lr"]))
        registry.record_result(ExperimentResult(experiment_id, {"quality": 0.9}, 1.0))
    mismatched_goal = Goal((MetricTarget("throughput"),), gpu_hour_budget=100.0)
    rows = build_intervention_outcomes(
        registry, goal=mismatched_goal, baseline=ExperimentResult("base", {"throughput": 1.0}, 0.0)
    )
    registry.close()

    assert all(not math.isfinite(row.gate_score_vs_baseline) for row in rows)
    model = estimate_arm_improvements(rows)

    assert model.excluded_non_finite == 2
    estimate = model.estimates[LR_ARM]
    assert estimate.excluded_non_finite == 2
    assert estimate.observations == 0
    assert estimate.mean_reward is None
    assert estimate.evidence is ArmEvidence.INSUFFICIENT
    assert model.incumbent_reward is None


def test_a_partially_unscoreable_arm_still_estimates_from_what_scored(tmp_path):
    rows = _outcomes_from_specs(
        tmp_path,
        (("a", "lr", 0.90, 1.0), ("b", "lr", 0.80, 1.0), ("c", "lr", 1.00, 1.0)),
    )
    poisoned = rows[:2] + (
        type(rows[2])(**{**rows[2].__dict__, "gate_score_vs_baseline": float("-inf")}),
    )
    estimate = estimate_arm_improvements(poisoned).estimates[LR_ARM]

    assert estimate.observations == 2
    assert estimate.excluded_non_finite == 1
    assert estimate.evidence is ArmEvidence.SUFFICIENT


def test_expected_improvement_lookup_of_an_unseen_arm_is_none(tmp_path):
    rows = _outcomes_from_specs(
        tmp_path, (("a", "lr", 0.90, 1.0), ("b", "lr", 0.80, 1.0))
    )
    assert estimate_arm_improvements(rows).expected_improvement(DATASET_ARM) is None


# --- Determinism -----------------------------------------------------------


def test_estimator_is_deterministic_across_repeated_calls(tmp_path):
    rows = _outcomes_from_specs(
        tmp_path,
        (
            ("a", "lr", 0.90, 1.0),
            ("b", "lora", 0.75, 2.0),
            ("c", "lr", 0.82, 0.5),
            ("d", "lora", 0.71, 1.0),
        ),
    )
    first = estimate_arm_improvements(rows)
    second = estimate_arm_improvements(rows)

    assert first == second
    assert tuple(first.estimates) == tuple(second.estimates)


def test_selector_ordering_is_deterministic_across_repeated_calls(goal_and_history):
    goal, baseline, history, candidates = goal_and_history
    orders = {
        tuple(
            e.experiment_id
            for e in prioritize_by_expected_improvement(
                candidates, history=history, goal=goal, baseline=baseline
            )
        )
        for _ in range(5)
    }
    assert len(orders) == 1


@pytest.fixture
def goal_and_history():
    goal, baseline = _goal(), _baseline()
    history = [
        (_experiment("h1", _PATCHES["lr"]), ExperimentResult("h1", {"quality": 0.95}, 1.0)),
        (_experiment("h2", _PATCHES["lr"]), ExperimentResult("h2", {"quality": 0.93}, 1.0)),
        (_experiment("h3", _PATCHES["lora"]), ExperimentResult("h3", {"quality": 0.72}, 1.0)),
        (_experiment("h4", _PATCHES["lora"]), ExperimentResult("h4", {"quality": 0.71}, 1.0)),
    ]
    candidates = [
        _experiment("cand_lora", _PATCHES["lora"]),
        _experiment("cand_lr", _PATCHES["lr"]),
    ]
    return goal, baseline, history, candidates


# --- Cold start and the reorder-only contract -----------------------------


def test_empty_candidates_return_an_empty_tuple():
    assert (
        prioritize_by_expected_improvement(
            [], history=(), goal=_goal(), baseline=_baseline()
        )
        == ()
    )


def test_cold_start_preserves_input_order_exactly(goal_and_history):
    goal, baseline, _history, _candidates = goal_and_history
    pool = [
        _experiment("c1", _PATCHES["lr"]),
        _experiment("c2", _PATCHES["lora"]),
        _experiment("c3", _PATCHES["dataset"]),
    ]
    ordered = prioritize_by_expected_improvement(
        pool, history=(), goal=goal, baseline=baseline
    )
    assert [e.experiment_id for e in ordered] == ["c1", "c2", "c3"]


def test_arms_with_too_little_history_keep_their_relative_order(goal_and_history):
    """Every candidate here is unestimable -- one arm untried, one tried
    once -- so the selector must fall back to input order rather than
    inventing a preference between them."""
    goal, baseline, _history, _candidates = goal_and_history
    history = [
        (_experiment("h1", _PATCHES["lr"]), ExperimentResult("h1", {"quality": 0.95}, 1.0))
    ]
    pool = [_experiment("c_new", _PATCHES["dataset"]), _experiment("c_lr", _PATCHES["lr"])]

    ordered = prioritize_by_expected_improvement(
        pool, history=history, goal=goal, baseline=baseline
    )
    assert [e.experiment_id for e in ordered] == ["c_new", "c_lr"]


def test_an_unestimable_arm_outranks_a_well_observed_mediocre_arm(goal_and_history):
    goal, baseline, history, _candidates = goal_and_history
    pool = [
        _experiment("cand_lora", _PATCHES["lora"]),
        _experiment("cand_untried", _PATCHES["dataset"]),
    ]
    ordered = prioritize_by_expected_improvement(
        pool, history=history, goal=goal, baseline=baseline
    )
    assert ordered[0].experiment_id == "cand_untried"


def test_a_historically_better_arm_outranks_a_historically_worse_one(goal_and_history):
    goal, baseline, history, candidates = goal_and_history
    ordered = prioritize_by_expected_improvement(
        candidates, history=history, goal=goal, baseline=baseline
    )
    order = [e.experiment_id for e in ordered]
    assert order.index("cand_lr") < order.index("cand_lora")


def test_selector_only_reorders_and_never_drops_or_adds(goal_and_history):
    goal, baseline, history, _candidates = goal_and_history
    pool = [_experiment(f"c{i}", _PATCHES["lr"]) for i in range(5)]
    ordered = prioritize_by_expected_improvement(
        pool, history=history, goal=goal, baseline=baseline
    )
    assert len(ordered) == len(pool)
    assert {e.experiment_id for e in ordered} == {e.experiment_id for e in pool}


def test_selector_mutates_neither_the_candidates_nor_their_status(goal_and_history):
    """It must not promote, gate, or otherwise touch a candidate -- a
    reordered pool has to come back as untouched, still-PLANNED objects."""
    goal, baseline, history, candidates = goal_and_history
    before = [
        (e.experiment_id, e.status, dict(e.config_patch), e.estimated_gpu_hours)
        for e in candidates
    ]
    ordered = prioritize_by_expected_improvement(
        candidates, history=history, goal=goal, baseline=baseline
    )
    after = [
        (e.experiment_id, e.status, dict(e.config_patch), e.estimated_gpu_hours)
        for e in candidates
    ]

    assert before == after
    assert all(e.status is ExperimentStatus.PLANNED for e in ordered)
    # The returned tuple holds the caller's own objects, merely reordered.
    assert {id(e) for e in ordered} == {id(e) for e in candidates}


def test_selector_leaves_the_supplied_history_untouched(goal_and_history):
    goal, baseline, history, candidates = goal_and_history
    before = [(e.experiment_id, e.status) for e, _ in history]
    prioritize_by_expected_improvement(
        candidates, history=history, goal=goal, baseline=baseline
    )
    assert [(e.experiment_id, e.status) for e, _ in history] == before


# --- The backtest harness --------------------------------------------------
#
# Three deterministic regimes, fixed before any result was looked at, so the
# numbers quoted for this slice are reproducible from the repository. They
# are CONSTRUCTED, not captured from real Chowder runs -- no registry of
# real historical runs exists here yet -- so they validate the harness, not
# the selector. See the module docstring.

SEPARABLE_REGIME = tuple(
    (f"s{i}", arm, quality, 1.0)
    for i, (arm, quality) in enumerate(
        [
            ("lr", 0.90), ("lora", 0.76), ("dataset", 0.66),
            ("lr", 0.88), ("lora", 0.74), ("dataset", 0.68),
            ("lr", 0.92), ("lora", 0.78), ("dataset", 0.64),
            ("lr", 0.89), ("lora", 0.75), ("dataset", 0.67),
        ]
    )
)

NOISY_REGIME = tuple(
    (f"n{i}", arm, quality, 1.0)
    for i, (arm, quality) in enumerate(
        [
            ("lr", 0.95), ("lora", 0.85), ("dataset", 0.80),
            ("lr", 0.60), ("lora", 0.70), ("dataset", 0.75),
            ("lr", 0.92), ("lora", 0.88), ("dataset", 0.82),
            ("lr", 0.63), ("lora", 0.69), ("dataset", 0.77),
        ]
    )
)

DRIFTING_REGIME = tuple(
    (f"d{i}", arm, quality, 1.0)
    for i, (arm, quality) in enumerate(
        [
            ("lr", 0.95), ("lora", 0.68),
            ("lr", 0.90), ("lora", 0.72),
            ("lr", 0.80), ("lora", 0.78),
            ("lr", 0.72), ("lora", 0.85),
            ("lr", 0.66), ("lora", 0.90),
            ("lr", 0.62), ("lora", 0.94),
        ]
    )
)

REGIMES = {
    "separable": SEPARABLE_REGIME,
    "noisy": NOISY_REGIME,
    "drifting": DRIFTING_REGIME,
}


@pytest.mark.parametrize("regime_name", sorted(REGIMES))
def test_backtest_report_is_well_formed_on_every_regime(tmp_path, regime_name):
    rows = _outcomes_from_specs(tmp_path, REGIMES[regime_name])
    report = backtest_selectors(rows, goal=_goal(), baseline=_baseline())

    assert report.selector_names == ("expected_improvement", "ucb1", "input_order")
    assert report.splits
    for split in report.splits:
        assert split.fit_size + split.held_out_size == len(rows)
        assert split.excluded_non_finite == 0
        for name in report.selector_names:
            score = split.scores[name]
            assert score.top_1_regret >= 0.0
            assert 1 <= score.best_rank <= split.held_out_size
            # Every held-out candidate is ranked exactly once.
            assert len(set(score.ordered_experiment_ids)) == split.held_out_size


@pytest.mark.parametrize("regime_name", sorted(REGIMES))
def test_backtest_is_deterministic_across_repeated_runs(tmp_path, regime_name):
    rows = _outcomes_from_specs(tmp_path, REGIMES[regime_name])
    first = backtest_selectors(rows, goal=_goal(), baseline=_baseline())
    second = backtest_selectors(rows, goal=_goal(), baseline=_baseline())
    assert first == second


def test_backtest_splits_are_chronological_prefixes_of_growing_size(tmp_path):
    rows = _outcomes_from_specs(tmp_path, SEPARABLE_REGIME)
    report = backtest_selectors(rows, goal=_goal(), baseline=_baseline())

    fit_sizes = [split.fit_size for split in report.splits]
    assert fit_sizes == sorted(fit_sizes)
    assert fit_sizes[0] == 2
    # The last split still leaves min_held_out_size rows to rank.
    assert fit_sizes[-1] == len(rows) - 2
    for split in report.splits:
        assert split.held_out_size == len(rows) - split.fit_size


def test_backtest_never_ranks_a_row_it_fitted_on(tmp_path):
    rows = _outcomes_from_specs(tmp_path, SEPARABLE_REGIME)
    report = backtest_selectors(rows, goal=_goal(), baseline=_baseline())
    ids = [row.experiment_id for row in rows]

    for split in report.splits:
        fitted = set(ids[: split.fit_size])
        for name in report.selector_names:
            assert not fitted & set(split.scores[name].ordered_experiment_ids)


def test_backtest_input_order_control_returns_the_registry_order(tmp_path):
    rows = _outcomes_from_specs(tmp_path, SEPARABLE_REGIME)
    report = backtest_selectors(rows, goal=_goal(), baseline=_baseline())
    ids = [row.experiment_id for row in rows]

    for split in report.splits:
        assert list(split.scores["input_order"].ordered_experiment_ids) == ids[split.fit_size :]


def test_backtest_of_too_short_a_history_reports_no_splits_rather_than_faking_one(tmp_path):
    rows = _outcomes_from_specs(tmp_path, (("a", "lr", 0.9, 1.0), ("b", "lr", 0.8, 1.0)))
    report = backtest_selectors(rows, goal=_goal(), baseline=_baseline())

    assert report.splits == ()
    assert report.mean_top_1_regret("ucb1") is None
    assert report.mean_best_rank("expected_improvement") is None


def test_backtest_of_no_outcomes_at_all_reports_no_splits():
    report = backtest_selectors((), goal=_goal(), baseline=_baseline())
    assert report.splits == ()


def test_backtest_excludes_unscoreable_held_out_rows_and_counts_them(tmp_path):
    rows = _outcomes_from_specs(tmp_path, SEPARABLE_REGIME)
    poisoned = rows[:-1] + (
        type(rows[-1])(**{**rows[-1].__dict__, "gate_score_vs_baseline": float("-inf")}),
    )
    report = backtest_selectors(poisoned, goal=_goal(), baseline=_baseline())

    assert report.splits
    for split in report.splits:
        assert split.excluded_non_finite == 1
        assert split.held_out_size == len(rows) - split.fit_size - 1


def test_backtest_means_average_the_per_split_scores(tmp_path):
    rows = _outcomes_from_specs(tmp_path, NOISY_REGIME)
    report = backtest_selectors(rows, goal=_goal(), baseline=_baseline())

    for name in report.selector_names:
        assert report.mean_top_1_regret(name) == pytest.approx(
            sum(split.scores[name].top_1_regret for split in report.splits) / len(report.splits)
        )
        assert report.mean_best_rank(name) == pytest.approx(
            sum(split.scores[name].best_rank for split in report.splits) / len(report.splits)
        )


def test_backtest_accepts_a_custom_selector_set(tmp_path):
    rows = _outcomes_from_specs(tmp_path, SEPARABLE_REGIME)
    only_ucb1 = {"ucb1": SELECTORS["ucb1"]}
    report = backtest_selectors(rows, goal=_goal(), baseline=_baseline(), selectors=only_ucb1)

    assert report.selector_names == ("ucb1",)
    assert all(set(split.scores) == {"ucb1"} for split in report.splits)


def test_backtest_perfect_selector_scores_zero_regret_and_rank_one(tmp_path):
    """A sanity anchor for the metric itself: a selector that knows the
    answer must score 0.0 regret and rank 1, so a low score genuinely means
    'ranked the good experiment first'."""
    rows = _outcomes_from_specs(tmp_path, SEPARABLE_REGIME)
    truth = {
        row.experiment_id: row.gate_score_vs_baseline / max(row.gpu_hours, 1e-9) for row in rows
    }

    def oracle(candidates, *, history, goal, baseline):
        return tuple(sorted(candidates, key=lambda e: -truth[e.experiment_id]))

    report = backtest_selectors(
        rows, goal=_goal(), baseline=_baseline(), selectors={"oracle": oracle}
    )
    assert report.mean_top_1_regret("oracle") == pytest.approx(0.0)
    assert report.mean_best_rank("oracle") == pytest.approx(1.0)


def test_backtest_never_promotes_or_writes_anything_to_the_registry(tmp_path):
    """The harness reads history and reorders; it must leave the registry
    exactly as it found it."""
    registry = RunRegistry(tmp_path / "runs.db")
    for experiment_id, arm_name, quality, gpu_hours in SEPARABLE_REGIME:
        registry.record_experiment(_experiment(experiment_id, _PATCHES[arm_name]))
        registry.record_result(ExperimentResult(experiment_id, {"quality": quality}, gpu_hours))
    rows = build_intervention_outcomes(registry, goal=_goal(), baseline=_baseline())

    backtest_selectors(rows, goal=_goal(), baseline=_baseline())

    statuses = {e.experiment_id: e.status for e in registry.list_experiments()}
    result_count = len(list(registry.list_results()))
    registry.close()

    assert result_count == len(SEPARABLE_REGIME)
    assert set(statuses.values()) == {ExperimentStatus.PLANNED}
