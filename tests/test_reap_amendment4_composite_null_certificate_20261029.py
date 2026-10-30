"""Exact composite-null proof tests; no original models or evidence accessed."""

from fractions import Fraction
from math import comb

import pytest

from chowder_batch import reap_amendment4_composite_null_certificate_20261029 as cert


def _binomial_upper_tail(m: int, threshold: int, probability: Fraction) -> Fraction:
    return sum((comb(m, b) * probability**b * (1 - probability)**(m - b)
                for b in range(threshold, m + 1)), Fraction(0))


@pytest.mark.parametrize("q,lower_delta", [
    (Fraction(1, 5), Fraction(-1, 10)),
    (Fraction(3, 5), Fraction(-1, 2)),
    (Fraction(1), Fraction(-1)),
])
def test_exact_conditional_stochastic_domination(q, lower_delta):
    boundary_delta = Fraction(-1, 50)
    assert abs(lower_delta) <= q and lower_delta < boundary_delta
    lower_p = (q + lower_delta) / (2 * q)
    boundary_p = (q + boundary_delta) / (2 * q)
    assert lower_p < boundary_p
    for m in range(7):
        for threshold in range(m + 2):
            assert _binomial_upper_tail(m, threshold, lower_p) <= (
                _binomial_upper_tail(m, threshold, boundary_p)
            )


def test_upper_tail_helper_accepts_no_rejection_and_all_rejection():
    assert cert._upper_tail_cutoffs(lambda b, c: False, n=4) == (1, 2, 3, 4, 5)
    assert cert._upper_tail_cutoffs(lambda b, c: True, n=4) == (0, 0, 0, 0, 0)


def test_upper_tail_helper_rejects_nonmonotone_forged_predicate():
    with pytest.raises(ArithmeticError, match="nonmonotone paired rejection"):
        cert._upper_tail_cutoffs(lambda b, c: b == 1, n=3)


@pytest.mark.parametrize("n", [0, -1, 1.5, True])
def test_invalid_population_rejected(n):
    with pytest.raises(ValueError, match="positive integer"):
        cert._upper_tail_cutoffs(lambda b, c: False, n=n)


def test_nonboolean_rejection_is_refused():
    with pytest.raises(ValueError, match="Boolean"):
        cert._upper_tail_cutoffs(lambda b, c: 1, n=2)


def test_exact_composite_null_certificate_extends_both_boundary_maxima():
    result = cert.composite_null_certificate_research()
    assert result["n"] == 164
    assert result["paired_tables_exactly_certified"] == 13695
    assert result["discordance_counts_with_proven_upper_tail"] == 165
    assert result["null_hypothesis"] == "p10-p01 <= -1/50"
    assert result["rational_score_decision_sha256"] == (
        "0eb2615268c596f3cb74ecfb3def9beab467dbe4f5101eb7fb04d00378ef990c"
    )
    for method, q in (("tango_score_composite_null", "1"),
                      ("section4_wald_composite_null", "0.02")):
        endpoint = result[method]
        assert endpoint["attained_boundary_discordance_q"] == q
        assert endpoint["maximum_exceeds_nominal_one_sided_alpha"] is True
        assert Fraction(int(endpoint["exact_maximum_numerator"]),
                        int(endpoint["exact_maximum_denominator"])) > Fraction(1, 20)
    assert result["fixed_benchmark_not_a_random_sample_certificate"] is True
    assert result["frozen_amendment4_method_adjudicated"] is False
    assert result["eligible_for_optional_v2_replacement"] is False
    assert result["research_only"] is True
    assert result["gen0_status"] == "BLOCKED"
    assert result["chosen"] is None
    assert result["release_approved"] is False


def test_tampered_boundary_coefficient_dominance_refused():
    with pytest.raises(ArithmeticError, match="Bernstein dominance"):
        cert._exact_composite_maximum({
            "all_integer_coefficients_at_most_attained_endpoint": False,
            "coefficient_count": 165,
        })


def test_tampered_boundary_denominator_refused():
    with pytest.raises(ArithmeticError, match="invalid exact boundary"):
        cert._exact_composite_maximum({
            "all_integer_coefficients_at_most_attained_endpoint": True,
            "coefficient_count": 165,
            "exact_maximum_numerator": "1",
            "exact_common_denominator": "1",
        })


def test_forged_original_decision_cell_refused(monkeypatch):
    original = cert._decision_table()
    b, c, k, score, wald = original[0]
    monkeypatch.setattr(cert, "_decision_table", lambda: (
        (b, c, k, not score, wald), *original[1:]
    ))
    with pytest.raises(ArithmeticError, match="uncertified paired decision"):
        cert.composite_null_certificate_research()
