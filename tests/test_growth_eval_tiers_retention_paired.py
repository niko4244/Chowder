"""Tests for the 0.5 evaluation-side wall:

eval isolation (a benchmark is promotion evidence unless proven otherwise,
and the search never sees tier 3), retention constraints (regressions
refuse, unmeasured refuses), and paired deltas (no pairing across a
protocol change, bootstrap CI separates effect from noise). All offline.
"""

from __future__ import annotations

import pytest

from chowder.growth.eval_isolation import (
    EvalTier,
    EvalTierPolicy,
    SearchIsolationRefusal,
    assert_search_isolation,
    classify_benchmarks,
)
from chowder.growth.paired_deltas import (
    PairedDelta,
    PairedOutcome,
    PairingRefusal,
    bootstrap_delta_interval,
    pair_outcomes,
)
from chowder.growth.retention import (
    RetentionConstraint,
    RetentionProfile,
    evaluate_retention,
)


# --- eval tiers ------------------------------------------------------------


def _policy(**declarations: str) -> EvalTierPolicy:
    return classify_benchmarks(declarations, source="test")


class TestEvalTiers:
    def test_unclassified_benchmark_defaults_to_promotion_evidence(self):
        # Fail closed: a benchmark nobody classified is tier 3, not tier 1.
        policy = _policy(dev_probe="search-evidence")
        assert policy.tier_of("dev_probe") is EvalTier.SEARCH_EVIDENCE
        assert policy.tier_of("something-nobody-declared") is (
            EvalTier.PROMOTION_EVIDENCE
        )

    def test_reserved_name_cannot_be_classified_downward(self):
        # The protected suite cannot be demoted to search evidence by config.
        with pytest.raises(SearchIsolationRefusal, match="reserved promotion"):
            _policy(protected_retained="search-evidence")

    def test_reserved_name_accepts_promotion_tier(self):
        policy = _policy(**{"gsm8k-heldout": "promotion-evidence"})
        assert policy.tier_of("gsm8k-heldout") is EvalTier.PROMOTION_EVIDENCE

    def test_unknown_tier_name_refuses(self):
        # A typo'd tier must not silently fall through to a default.
        with pytest.raises(SearchIsolationRefusal, match="not one of"):
            _policy(dev_probe="search_evidnce")

    def test_search_surface_reading_tier3_refuses(self):
        policy = _policy(
            dev_loss="search-evidence",
            dev_probe="survivor-evidence",
        )
        with pytest.raises(SearchIsolationRefusal, match="promotion evidence"):
            assert_search_isolation(
                policy=policy,
                search_readable_benchmarks=["dev_loss", "dev_probe", "final-bench"],
            )

    def test_search_surface_of_only_tier1_and_tier2_passes(self):
        policy = _policy(
            dev_loss="search-evidence",
            dev_probe="survivor-evidence",
        )
        assert_search_isolation(
            policy=policy,
            search_readable_benchmarks=["dev_loss", "dev_probe"],
        )

    def test_selection_policy_naming_protected_refuses(self):
        policy = _policy(dev_loss="search-evidence")
        with pytest.raises(SearchIsolationRefusal, match="selection policy"):
            assert_search_isolation(
                policy=policy,
                search_readable_benchmarks=["dev_loss"],
                selection_policies=["select-on-protected-suite"],
            )


# --- retention ---------------------------------------------------------------


def _profile(**constraints: tuple[str, float]) -> RetentionProfile:
    named = tuple(
        RetentionConstraint(dimension=dim, kind=kind, value=value, benchmark="final-bench")
        for dim, (kind, value) in constraints.items()
    )
    return RetentionProfile(profile_id="test-profile", constraints=named)


class TestRetention:
    def test_capability_trade_refuses(self):
        # The mandate's arithmetic: target 0.60 -> 0.78 while reasoning
        # collapses 0.71 -> 0.51. A gain is not a promotion with a footnote.
        profile = _profile(
            reasoning=("max-regression", 0.0),
            tool_validity=("max-regression", -0.05),
        )
        violations = evaluate_retention(
            profile,
            parent_values={"reasoning": 0.71, "tool_validity": 0.90},
            candidate_values={"reasoning": 0.51, "tool_validity": 0.93},
        )
        assert [v.dimension for v in violations] == ["reasoning"]
        assert violations[0].measured == pytest.approx(-0.20)

    def test_compliant_candidate_has_no_violations(self):
        profile = _profile(
            reasoning=("max-regression", -0.02),
            tool_validity=("absolute-floor", 0.85),
        )
        assert not evaluate_retention(
            profile,
            parent_values={"reasoning": 0.71, "tool_validity": 0.90},
            candidate_values={"reasoning": 0.70, "tool_validity": 0.93},
        )

    def test_small_declared_dip_within_budget_passes(self):
        profile = _profile(reasoning=("max-regression", -0.02))
        assert not evaluate_retention(
            profile,
            parent_values={"reasoning": 0.71},
            candidate_values={"reasoning": 0.695},  # -0.015, inside the budget
        )

    def test_unmeasured_candidate_dimension_is_a_violation(self):
        # Fail closed: a gate that cannot be measured was not passed.
        profile = _profile(reasoning=("max-regression", 0.0))
        violations = evaluate_retention(
            profile,
            parent_values={"reasoning": 0.71},
            candidate_values={},
        )
        assert len(violations) == 1
        assert "unmeasured is not compliance" in violations[0].detail

    def test_missing_parent_measurement_refuses_max_regression(self):
        profile = _profile(reasoning=("max-regression", 0.0))
        violations = evaluate_retention(
            profile,
            parent_values={},
            candidate_values={"reasoning": 0.71},
        )
        assert len(violations) == 1
        assert "cannot be certified" in violations[0].detail

    def test_absolute_floor_fires_below_floor(self):
        profile = _profile(termination_health=("absolute-floor", 0.95))
        violations = evaluate_retention(
            profile,
            parent_values={"termination_health": 0.97},
            candidate_values={"termination_health": 0.94},
        )
        assert violations[0].measured == pytest.approx(0.94)

    def test_empty_profile_refuses(self):
        with pytest.raises(ValueError, match="unconstrained campaign"):
            RetentionProfile(profile_id="empty", constraints=())

    def test_duplicate_dimension_refuses(self):
        with pytest.raises(ValueError, match="twice"):
            RetentionProfile(
                profile_id="dup",
                constraints=(
                    RetentionConstraint("reasoning", "max-regression", 0.0, "b"),
                    RetentionConstraint("reasoning", "max-regression", -0.1, "b"),
                ),
            )

    def test_unknown_kind_refuses(self):
        with pytest.raises(ValueError, match="unknown kind"):
            RetentionConstraint("reasoning", "at-least-parent", 0.0, "b")


# --- paired deltas -----------------------------------------------------------


def _row(task_id: str, score: float, **protocol) -> PairedOutcome:
    base = {
        "decoding": "greedy",
        "seed": 7,
        "evaluator": "unit-eval",
        "evaluator_version": "1.2.0",
        "hardware_class": "kaggle_2x_t4_16gb",
        "protocol_sha256": "abc123",
    }
    base.update(protocol)
    return PairedOutcome(task_id=task_id, score=score, protocol=base)


class TestPairedDeltas:
    def test_win_loss_tie_counting_and_means(self):
        parent = [_row(f"t{i}", s) for i, s in enumerate([0.6, 0.5, 0.5, 0.4])]
        candidate = [_row(f"t{i}", s) for i, s in enumerate([0.7, 0.5, 0.4, 0.6])]
        delta = pair_outcomes(dimension="target", parent=parent, candidate=candidate)
        assert (delta.wins, delta.losses, delta.ties) == (2, 1, 1)
        assert delta.mean_delta == pytest.approx(0.05)
        assert delta.parent_mean == pytest.approx(0.5)
        assert delta.candidate_mean == pytest.approx(0.55)
        assert delta.win_rate == pytest.approx(2 / 3)
        assert delta.task_deltas == pytest.approx((0.1, 0.0, -0.1, 0.2))

    def test_float_tie_tolerance(self):
        # A 1e-13 difference is a tie, not a loss.
        parent = [_row("t0", 0.5)]
        candidate = [_row("t0", 0.5 + 1e-13)]
        delta = pair_outcomes(dimension="d", parent=parent, candidate=candidate)
        assert delta.ties == 1 and delta.wins == 0 and delta.losses == 0

    def test_task_set_mismatch_refuses(self):
        parent = [_row("t0", 0.5), _row("t1", 0.6)]
        candidate = [_row("t0", 0.6)]
        # Pairing the intersection would hide t1's missing candidate row.
        with pytest.raises(PairingRefusal, match="different task sets"):
            pair_outcomes(dimension="d", parent=parent, candidate=candidate)

    def test_protocol_mismatch_refuses_and_names_keys(self):
        parent = [_row("t0", 0.5)]
        candidate = [_row("t0", 0.6, seed=8, protocol_sha256="def456")]
        with pytest.raises(PairingRefusal, match=r"\['seed', 'protocol_sha256'\]"):
            pair_outcomes(dimension="d", parent=parent, candidate=candidate)

    def test_bootstrap_ci_excludes_zero_for_real_effect(self):
        # Consistent +0.2 on every task: a measured effect.
        parent = [_row(f"t{i}", 0.5) for i in range(12)]
        candidate = [_row(f"t{i}", 0.7) for i in range(12)]
        delta = pair_outcomes(dimension="d", parent=parent, candidate=candidate)
        low, high = bootstrap_delta_interval(delta)
        assert low > 0.0 and high > 0.0

    def test_bootstrap_ci_includes_zero_for_noise(self):
        # Symmetric ±0.2 noise around zero mean: no measured effect.
        scores = [0.7, 0.3] * 6
        parent = [_row(f"t{i}", 0.5) for i in range(12)]
        candidate = [_row(f"t{i}", s) for i, s in enumerate(scores)]
        delta = pair_outcomes(dimension="d", parent=parent, candidate=candidate)
        low, high = bootstrap_delta_interval(delta)
        assert low <= 0.0 <= high

    def test_bootstrap_ci_is_deterministic(self):
        parent = [_row(f"t{i}", 0.5) for i in range(8)]
        candidate = [_row(f"t{i}", 0.5 + 0.1 * (i % 2)) for i in range(8)]
        delta = pair_outcomes(dimension="d", parent=parent, candidate=candidate)
        assert bootstrap_delta_interval(delta) == bootstrap_delta_interval(delta)

    def test_bootstrap_refuses_empty_deltas(self):
        empty = PairedDelta(
            dimension="d", paired_tasks=0, wins=0, losses=0, ties=0,
            parent_mean=0.0, candidate_mean=0.0, mean_delta=0.0,
            task_deltas=(),
        )
        with pytest.raises(ValueError, match="no paired deltas"):
            bootstrap_delta_interval(empty)
