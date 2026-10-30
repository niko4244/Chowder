"""Independent CPU-only and adversarial checks of exact paired CI enclosure."""

from fractions import Fraction
from math import comb

import pytest

from chowder_batch import reap_amendment4_exact_ci_enclosure_20261029 as ci
from chowder_batch.reap_amendment4_exact_conditional_ni_20261029 import (
    PairedCounts,
)
from chowder_batch import reap_amendment4_exact_conditional_ni_20261029 as ni


def independent_tail(m: int, b: int, p: Fraction) -> Fraction:
    return sum(
        (Fraction(comb(m, k)) * p**k * (1 - p)**(m - k)
         for k in range(b, m + 1)),
        Fraction(0),
    )


@pytest.mark.parametrize("m", [0, 1, 2, 3, 9, 20, 164, 1319])
@pytest.mark.parametrize("p", [Fraction(0), Fraction(1, 5),
                               Fraction(49, 100), Fraction(1, 2),
                               Fraction(1)])
def test_exact_rational_tail_matches_independent_comb(m, p):
    for b in sorted({0, m // 2, m}):
        num, den = ci.binomial_upper_tail_rational(m, b, p.numerator,
                                                    p.denominator)
        assert Fraction(num, den) == independent_tail(m, b, p)


@pytest.mark.parametrize("m,b", [(0, 0), (1, 0), (1, 1), (9, 5),
                                  (164, 82), (164, 164),
                                  (1319, 1), (1319, 650), (1319, 1319)])
def test_cp_endpoint_has_certified_dyadic_bracket(m, b):
    lower, upper = ci.clopper_pearson_lower_enclosure(m, b, bits=32)
    assert 0 <= lower <= upper <= 1
    if b == 0:
        assert lower == upper == 0
    else:
        assert upper - lower == Fraction(1, 1 << 32)
        assert independent_tail(m, b, lower) < Fraction(1, 20)
        assert independent_tail(m, b, upper) >= Fraction(1, 20)


def test_complete_research_pair_remains_blocked_and_margin_not_qualified():
    report = ci.paired_lower_confidence_research(
        PairedCounts(147, 5, 4, 8), suite="humaneval",
    )
    assert report["discordant"] == 9
    assert report["exact_tail_strictly_below_five_percent"] is False
    assert report["upper_enclosure_at_or_below_minus_two_percent"] is True
    assert report["margin_enclosure_resolved"] is True
    assert report["research_only"] is True
    assert report["frozen_amendment4_method_adjudicated"] is False
    assert report["eligible_for_optional_v2_replacement"] is False
    assert report["gen0_status"] == "BLOCKED"
    assert report["chosen"] is None
    assert report["release_approved"] is False


def test_zero_discordance_and_all_candidate_only_are_conservative():
    zero = ci.paired_lower_confidence_research(
        PairedCounts(164, 0, 0, 0), suite="humaneval",
    )
    assert zero["risk_difference_lower_enclosure"] == "-1"
    assert zero["exact_tail_strictly_below_five_percent"] is False
    all_candidate = ci.paired_lower_confidence_research(
        PairedCounts(155, 9, 0, 0), suite="humaneval",
    )
    assert all_candidate["exact_tail_strictly_below_five_percent"] is True
    assert all_candidate["lower_enclosure_strictly_above_minus_two_percent"] is True
    assert all_candidate["release_approved"] is False


@pytest.mark.parametrize("n", [1, 2, 4, 6])
@pytest.mark.parametrize("q,p", [
    (Fraction(0), Fraction(0)),
    (Fraction(1, 5), Fraction(1, 5)),
    (Fraction(1, 2), Fraction(49, 100)),
    (Fraction(3, 4), Fraction(3, 4)),
    (Fraction(1), Fraction(1)),
])
def test_exhaustive_small_multinomial_coverage_at_least_95_percent(n, q, p):
    # Independently enumerate M~Bin(n,q), B|M~Bin(M,p).
    # This verifies the lower risk-difference bound, not just the p-value.
    delta = q * (2 * p - 1)
    coverage = Fraction(0)
    for m in range(n + 1):
        m_probability = Fraction(comb(n, m)) * q**m * (1-q)**(n-m)
        for b in range(m + 1):
            b_probability = (Fraction(comb(m, b)) * p**b
                             * (1-p)**(m-b))
            lower, _ = ci.clopper_pearson_lower_enclosure(m, b, bits=24)
            lower_delta = min(Fraction(0), 2*lower-1)
            if lower_delta <= delta:
                coverage += m_probability * b_probability
    assert coverage >= Fraction(19, 20)


@pytest.mark.parametrize("m", [0, 1, 9, 164, 1319])
def test_exact_ni_decision_matches_original_independent_tail(m):
    for b in sorted({0, 1 if m else 0, m // 2, m}):
        numerator, denominator = ci.binomial_upper_tail_rational(m, b, 49, 100)
        assert (20 * numerator < denominator) == (
            independent_tail(m, b, Fraction(49, 100)) < Fraction(1, 20)
        )


@pytest.mark.parametrize("args", [
    (True, 0, 1, 2), (2.0, 1, 1, 2), (-1, 0, 1, 2),
    (3, 4, 1, 2), (1, 1, -1, 2), (1, 1, 3, 2), (1, 1, 1, 0),
])
def test_invalid_binomial_inputs_fail_closed(args):
    with pytest.raises(ValueError):
        ci.binomial_upper_tail_rational(*args)


@pytest.mark.parametrize("bits", [True, 23, 65, 48.0, "48"])
def test_invalid_precision_fails_closed(bits):
    with pytest.raises(ValueError, match="bits"):
        ci.clopper_pearson_lower_enclosure(9, 5, bits=bits)


def test_invalid_population_and_type_fail_closed():
    with pytest.raises(ValueError, match="PairedCounts"):
        ci.paired_lower_confidence_research((147, 5, 4, 8), suite="humaneval")
    with pytest.raises(ValueError, match="frozen canonical"):
        ci.paired_lower_confidence_research(
            PairedCounts(147, 5, 4, 8), suite="gsm8k",
        )
    for malformed_suite in ([], {}, 1, None, True):
        with pytest.raises(ValueError, match="frozen canonical"):
            ci.paired_lower_confidence_research(
                PairedCounts(147, 5, 4, 8), suite=malformed_suite,
            )


def test_mutable_upstream_population_registry_cannot_admit_wrong_size(monkeypatch):
    monkeypatch.setattr(ni, "FROZEN_COUNTS", {"humaneval": 9, "gsm8k": 9})
    with pytest.raises(ValueError, match="frozen canonical"):
        ci.paired_lower_confidence_research(
            PairedCounts(3, 2, 2, 2), suite="humaneval",
        )


@pytest.mark.parametrize("suite,cells", [
    ("humaneval", PairedCounts(72, 92, 0, 0)),
    ("gsm8k", PairedCounts(641, 678, 0, 0)),
])
def test_positive_research_bound_still_cannot_qualify_frozen_gate(suite, cells):
    report = ci.paired_lower_confidence_research(cells, suite=suite)
    assert report["risk_difference_lower_enclosure"] == "0"
    assert report["exact_tail_strictly_below_five_percent"] is True
    assert report["margin_enclosure_resolved"] is True
    assert report["eligible_for_optional_v2_replacement"] is False
    assert report["release_approved"] is False


@pytest.mark.parametrize("suite,m,below,at", [
    ("humaneval", 164, 91, 92),
    ("gsm8k", 1319, 676, 677),
])
@pytest.mark.parametrize("bits", [24, 48])
def test_exact_margin_transition_matches_certified_confidence_interval(
    suite, m, below, at, bits,
):
    for b, expected in ((below, False), (at, True)):
        report = ci.paired_lower_confidence_research(
            PairedCounts(0, b, m-b, 0), suite=suite, bits=bits,
        )
        assert report["exact_tail_strictly_below_five_percent"] is expected
        assert report["margin_enclosure_resolved"] is True
        assert report["lower_enclosure_strictly_above_minus_two_percent"] is expected
        assert report["upper_enclosure_at_or_below_minus_two_percent"] is not expected
