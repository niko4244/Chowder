"""Exact null calibration is research-only and cannot approve a model."""

import math

import pytest

from chowder_batch.reap_amendment4_exact_null_calibration_20261029 import (
    _N, _NULL_DIFFERENCE, _Z95, _decision_table,
    exact_null_calibration, fixed_research_grid,
)
from chowder_batch.reap_amendment4_paired_score_research_20261029 import (
    PairedCounts, _score_z, section4_wald_lower_diagnostic,
    tango_score_lower_diagnostic,
)


@pytest.mark.parametrize("q", [-1, 0, 0.019, 1.001, float("nan"),
                                    float("inf"), True, "0.02"])
def test_invalid_null_nuisance_is_refused(q):
    with pytest.raises(ValueError, match="discordance"):
        exact_null_calibration(q)


def test_all_possible_paired_tables_match_actual_score_interval_inversion():
    rows = _decision_table()
    assert len(rows) == 13695
    for b, c, k, score_reject, wald_reject in rows:
        cells = PairedCounts(k, b, c, 0)
        assert score_reject is (_score_z(cells, _NULL_DIFFERENCE) > _Z95)
        assert score_reject is (tango_score_lower_diagnostic(cells) > _NULL_DIFFERENCE)
        assert wald_reject is (
            section4_wald_lower_diagnostic(cells) > _NULL_DIFFERENCE
        )


@pytest.mark.parametrize("q", [0.02, 0.03, 0.05, 0.10, 0.20, 0.50, 0.80, 1.00])
def test_exact_enumeration_has_unit_probability_and_never_admits(q):
    result = exact_null_calibration(q)
    assert result["exact_probability_mass"] == pytest.approx(1, abs=1e-10)
    assert result["frozen_amendment4_method_adjudicated"] is False
    assert result["eligible_for_optional_v2_replacement"] is False
    assert result["research_only"] is True
    assert result["gen0_status"] == "BLOCKED"
    assert result["chosen"] is None
    assert result["release_approved"] is False
    assert 0 <= result["score_lower_bound_strictly_above_margin_probability"] <= 1
    assert 0 <= result["section4_wald_lower_bound_strictly_above_margin_probability"] <= 1


@pytest.mark.parametrize("q", [0.02, 1.0])
def test_boundary_nuisance_matches_independent_one_dimensional_binomial(q):
    """At q endpoints one of the three multinomial probabilities is zero."""
    p10 = (q + _NULL_DIFFERENCE) / 2
    p01 = (q - _NULL_DIFFERENCE) / 2
    expected_score = []
    expected_wald = []
    for b in range(_N + 1):
        c = _N - b if q == 1.0 else b
        candidate_only = b if q == 1.0 else 0
        reference_only = c
        concordant = 0 if q == 1.0 else _N - c
        cells = PairedCounts(concordant, candidate_only, reference_only, 0)
        if q == 1.0:
            mass = math.comb(_N, b) * p10 ** b * p01 ** c
        else:
            mass = math.comb(_N, c) * p01 ** c * (1 - p01) ** concordant
        if tango_score_lower_diagnostic(cells) > _NULL_DIFFERENCE:
            expected_score.append(mass)
        if section4_wald_lower_diagnostic(cells) > _NULL_DIFFERENCE:
            expected_wald.append(mass)
    result = exact_null_calibration(q)
    assert result["score_lower_bound_strictly_above_margin_probability"] == pytest.approx(
        math.fsum(expected_score), abs=1e-11
    )
    assert result["section4_wald_lower_bound_strictly_above_margin_probability"] == pytest.approx(
        math.fsum(expected_wald), abs=1e-11
    )


def test_interior_nuisance_matches_independent_conditional_binomial():
    """B~Bin(n,p10), C|B~Bin(n-B,p01/(1-p10)), not log-multinomial."""
    q = 0.10
    p10 = (q + _NULL_DIFFERENCE) / 2
    p01 = (q - _NULL_DIFFERENCE) / 2
    conditional_p = p01 / (1 - p10)
    score_masses = []
    wald_masses = []
    all_masses = []
    for b in range(_N + 1):
        mass_b = math.comb(_N, b) * p10 ** b * (1 - p10) ** (_N - b)
        for c in range(_N - b + 1):
            k = _N - b - c
            mass_c = (math.comb(_N - b, c) * conditional_p ** c
                      * (1 - conditional_p) ** k)
            mass = mass_b * mass_c
            cells = PairedCounts(k, b, c, 0)
            all_masses.append(mass)
            if tango_score_lower_diagnostic(cells) > _NULL_DIFFERENCE:
                score_masses.append(mass)
            if section4_wald_lower_diagnostic(cells) > _NULL_DIFFERENCE:
                wald_masses.append(mass)
    result = exact_null_calibration(q)
    assert math.fsum(all_masses) == pytest.approx(1, abs=1e-11)
    assert result["score_lower_bound_strictly_above_margin_probability"] == pytest.approx(
        math.fsum(score_masses), abs=1e-11
    )
    assert result["section4_wald_lower_bound_strictly_above_margin_probability"] == pytest.approx(
        math.fsum(wald_masses), abs=1e-11
    )


def test_material_null_size_difference_and_score_not_exact_level():
    near_zero = exact_null_calibration(0.02)
    assert near_zero["score_lower_bound_strictly_above_margin_probability"] == pytest.approx(
        0.036397907215, abs=1e-10
    )
    assert near_zero["section4_wald_lower_bound_strictly_above_margin_probability"] == pytest.approx(
        0.158219474222, abs=1e-10
    )
    high_discordance = exact_null_calibration(1.0)
    # The asymptotic score interval is not an exact uniform 5% test either.
    assert high_discordance["score_lower_bound_strictly_above_margin_probability"] == pytest.approx(
        0.056557711909, abs=1e-10
    )
    assert high_discordance["score_lower_bound_strictly_above_margin_probability"] > 0.05


def test_grid_is_prospectively_fixed_and_never_reads_model_evidence():
    grid = fixed_research_grid()
    assert tuple(row["null_discordance_probability"] for row in grid) == (
        0.02, 0.03, 0.05, 0.10, 0.20, 0.50, 0.80, 1.00
    )
    assert all(row["release_approved"] is False for row in grid)
