"""Independent polynomial and exact-multinomial checks; research only."""

import math

import pytest

from chowder_batch.reap_amendment4_exact_null_calibration_20261029 import (
    exact_null_calibration,
)
from chowder_batch.reap_amendment4_uniform_null_enclosure_20261029 import (
    _N, _bernstein_coefficients, _split_half, _supremum_enclosure,
    uniform_boundary_null_research,
)


def _independent_bernstein_evaluation(coefficients, t):
    n = len(coefficients) - 1
    return math.fsum(math.comb(n, j) * t ** j * (1 - t) ** (n - j) * value
                     for j, value in enumerate(coefficients))


@pytest.mark.parametrize("q", [0.02, 0.03, 0.05, 0.10, 0.20, 0.50, 0.80, 1.0])
def test_endpoint_mixture_polynomial_equals_independent_exact_multinomial(q):
    """Two different summation identities must agree for every sampled q."""
    score, wald = _bernstein_coefficients()
    assert len(score) == len(wald) == _N + 1
    t = (q - 0.02) / 0.98
    exact = exact_null_calibration(q)
    assert _independent_bernstein_evaluation(score, t) == pytest.approx(
        exact["score_lower_bound_strictly_above_margin_probability"], abs=1e-11
    )
    assert _independent_bernstein_evaluation(wald, t) == pytest.approx(
        exact["section4_wald_lower_bound_strictly_above_margin_probability"], abs=1e-11
    )


def test_de_casteljau_subdivision_preserves_polynomial_and_convex_hull():
    coefficients = (0.1, 0.9, 0.2, 0.5, 0.0)
    left, right, middle = _split_half(coefficients)
    assert middle == pytest.approx(_independent_bernstein_evaluation(coefficients, 0.5))
    for t in (0.0, 0.1, 0.3, 0.7, 1.0):
        expected = _independent_bernstein_evaluation(coefficients, t)
        local = (left, 2 * t) if t <= 0.5 else (right, 2 * t - 1)
        assert _independent_bernstein_evaluation(*local) == pytest.approx(expected)
        assert min(coefficients) - 1e-12 <= expected <= max(coefficients) + 1e-12


def test_adaptive_bound_locates_non_endpoint_maximum_and_refuses_budget_failure():
    # Degree-3 polynomial 3t(1-t)^2 peaks at t=1/3 with value 4/9.
    result = _supremum_enclosure((0.0, 1.0, 0.0, 0.0),
                                 tolerance=1e-7, max_splits=20000)
    assert result["subdivisions"] > 1
    assert result["lower_bound"] <= 4 / 9 <= result["upper_bound"]
    assert result["upper_bound"] - result["lower_bound"] <= 1e-7 + 3e-10
    with pytest.raises(ArithmeticError, match="did not converge"):
        _supremum_enclosure((0.0, 1.0, 0.0, 0.0),
                             tolerance=1e-12, max_splits=1)


@pytest.mark.parametrize("tolerance", [0, -1, 0.2, float("nan"),
                                       float("inf"), True, "0.001"])
def test_invalid_tolerances_are_rejected(tolerance):
    with pytest.raises(ValueError, match="tolerance"):
        _supremum_enclosure((0.0, 1.0, 0.0),
                             tolerance=tolerance, max_splits=100)


@pytest.mark.parametrize("limit", [0, -1, 1.5, True, "5"])
def test_invalid_subdivision_limits_are_rejected(limit):
    with pytest.raises(ValueError, match="max_splits"):
        _supremum_enclosure((0.0, 1.0, 0.0),
                             tolerance=1e-5, max_splits=limit)


def test_continuum_enclosure_reports_research_only_worst_case():
    result = uniform_boundary_null_research(tolerance=1e-5)
    score = result["tango_score_supremum_enclosure"]
    wald = result["section4_wald_supremum_enclosure"]
    assert score["lower_bound"] <= 0.056557711909 <= score["upper_bound"]
    assert wald["lower_bound"] <= 0.158219474222 <= wald["upper_bound"]
    assert score["upper_bound"] - score["lower_bound"] <= 1e-5
    assert wald["upper_bound"] - wald["lower_bound"] <= 1e-5
    assert score["q_at_observed_lower_bound"] == 1.0
    assert wald["q_at_observed_lower_bound"] == 0.02
    assert result["floating_point_slack_not_formal_interval_arithmetic"] > 0
    assert result["frozen_amendment4_method_adjudicated"] is False
    assert result["eligible_for_optional_v2_replacement"] is False
    assert result["research_only"] is True
    assert result["gen0_status"] == "BLOCKED"
    assert result["chosen"] is None
    assert result["release_approved"] is False
