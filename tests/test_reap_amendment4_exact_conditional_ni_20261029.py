"""CPU-only adversarial checks of exact conditional noninferiority research."""

from fractions import Fraction
from math import comb

import pytest

from chowder_batch import reap_amendment4_exact_conditional_ni_20261029 as ni


def independent_tail(m: int, b: int) -> Fraction:
    return sum((Fraction(comb(m, k) * 49**k * 51**(m-k), 100**m)
                for k in range(b, m + 1)), Fraction(0))


@pytest.mark.parametrize("m", [0, 1, 2, 3, 4, 5, 9, 20, 164, 1319])
def test_exact_integer_tail_matches_independent_binomial_enumeration(m):
    for b in sorted({0, 1 if m else 0, m // 2, m}):
        numerator, denominator = ni.exact_upper_tail(m, b)
        assert Fraction(numerator, denominator) == independent_tail(m, b)
        assert denominator == 100**m


def test_all_humaneval_discordance_counts_have_monotone_upper_tail():
    for m in range(165):
        previous = Fraction(1)
        for b in range(m + 1):
            numerator, denominator = ni.exact_upper_tail(m, b)
            current = Fraction(numerator, denominator)
            assert current <= previous
            assert current == independent_tail(m, b)
            previous = current


def test_composite_null_discordant_success_probability_is_at_most_49_percent():
    # The proof applies for every q in [1/50,1], not only these witnesses.
    for q in (Fraction(1, 50), Fraction(1, 25), Fraction(1, 10),
              Fraction(1, 2), Fraction(1)):
        for delta in (-q, Fraction(-1, 50)):
            if delta > -Fraction(1, 50):
                continue
            p = (q + delta) / (2 * q)
            assert 0 <= p <= Fraction(49, 100)


@pytest.mark.parametrize("m", [0, 1, 2, 5, 9, 20, 164])
@pytest.mark.parametrize("p", [Fraction(0), Fraction(1, 100),
                               Fraction(1, 4), Fraction(49, 100)])
def test_exact_conditional_size_is_at_most_five_percent(m, p):
    # Exhaust every b at each chosen m, including the least favorable p=.49.
    reject_probability = sum(
        (Fraction(comb(m, b)) * p**b * (1-p)**(m-b)
         for b in range(m + 1)
         if (lambda v: 20 * v[0] < v[1])(ni.exact_upper_tail(m, b))),
        Fraction(0),
    )
    assert reject_probability <= Fraction(1, 20)


def test_completed_research_humaneval_pair_is_not_an_original_frozen_gate():
    # Previously completed Q8-vs-Q4 research pair; NOT Q8-vs-dense Gate 2.
    cells = ni.PairedCounts(147, 5, 4, 8)
    report = ni.exact_conditional_ni_research(cells, suite="humaneval")
    assert report["n"] == 164
    assert report["discordant"] == 9
    assert report["candidate_minus_reference_numerator"] == 1
    assert report["strict_p_below_one_twentieth"] is False
    assert Fraction(int(report["exact_one_sided_p_numerator"]),
                    int(report["exact_one_sided_p_denominator"])) == independent_tail(9, 5)
    assert report["research_only"] is True
    assert report["frozen_amendment4_method_adjudicated"] is False
    assert report["eligible_for_optional_v2_replacement"] is False
    assert report["gen0_status"] == "BLOCKED"
    assert report["chosen"] is None
    assert report["release_approved"] is False


def test_known_wald_endpoint_false_approval_does_not_approve_exact_conditional():
    # Under the genuine boundary null p10=0,p01=.02, b=0,c=18 is one
    # table that the frozen Wald diagnostic falsely approves at n=1319.
    cells = ni.PairedCounts(1301, 0, 18, 0)
    report = ni.exact_conditional_ni_research(cells, suite="gsm8k")
    assert report["exact_one_sided_p_numerator"] == report["exact_one_sided_p_denominator"]
    assert report["strict_p_below_one_twentieth"] is False


def test_all_candidate_only_discordances_can_pass_research_test_but_not_gate():
    report = ni.exact_conditional_ni_research(
        ni.PairedCounts(155, 9, 0, 0), suite="humaneval",
    )
    assert report["strict_p_below_one_twentieth"] is True
    assert report["eligible_for_optional_v2_replacement"] is False
    assert report["release_approved"] is False


@pytest.mark.parametrize("bad", [(-1, 1, 0, 164), (0, 0, 0, 0),
                                 (True, 1, 1, 161), (1.0, 1, 1, 161),
                                 ("1", 1, 1, 161)])
def test_invalid_cells_fail_closed(bad):
    with pytest.raises(ValueError, match="paired"):
        ni.PairedCounts(*bad)


@pytest.mark.parametrize("m,b", [(True, 0), (9, False), (1.0, 0),
                                  (9, -1), (-1, 0), (9, 10), (1320, 1)])
def test_invalid_discordances_fail_closed(m, b):
    with pytest.raises(ValueError, match="invalid exact conditional"):
        ni.exact_upper_tail(m, b)


@pytest.mark.parametrize("suite", ["gsm8k", "humaneval", "made-up", ""])
def test_wrong_population_or_suite_refused(suite):
    with pytest.raises(ValueError, match="frozen canonical"):
        ni.exact_conditional_ni_research(
            ni.PairedCounts(4, 1, 1, 3), suite=suite,
        )


@pytest.mark.parametrize("numerator,denominator,expected", [
    (1, 20, False), (499, 10000, True), (501, 10000, False),
])
def test_exact_strict_threshold_not_rounded_display(monkeypatch, numerator,
                                                    denominator, expected):
    # Synthetic p-values probe the strict threshold without pretending to be
    # real binomial tails. The integer comparison, not decimal formatting,
    # determines the diagnostic Boolean.
    monkeypatch.setattr(ni, "exact_upper_tail", lambda _m, _b: (
        numerator, denominator,
    ))
    report = ni.exact_conditional_ni_research(
        ni.PairedCounts(155, 9, 0, 0), suite="humaneval",
    )
    assert report["strict_p_below_one_twentieth"] is expected
