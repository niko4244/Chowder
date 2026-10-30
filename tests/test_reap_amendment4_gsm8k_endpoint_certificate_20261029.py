"""Adversarial CPU-only tests of the GSM8K endpoint mathematical certificate."""

from decimal import Decimal, localcontext
from fractions import Fraction
from math import comb
from statistics import NormalDist

import pytest

from chowder_batch import reap_amendment4_gsm8k_endpoint_certificate_20261029 as cert
from chowder_batch.reap_amendment4_paired_score_research_20261029 import (
    PairedCounts, _score_z, section4_wald_lower_diagnostic,
)


def test_all_1320_wald_decisions_match_independent_frozen_sample_sd():
    for c in range(cert._N + 1):
        cells = PairedCounts(cert._N - c, 0, c, 0)
        assert cert._exact_wald_reject(c) is (
            section4_wald_lower_diagnostic(cells) > -0.02), c
        assert cert._exact_wald_reject(c) is (c <= 19), c


def test_all_1320_score_decisions_match_independent_profile_z():
    cert.rational._certify_cutoff()
    cutoff = NormalDist().inv_cdf(0.95)
    for c in range(cert._N + 1):
        cells = PairedCounts(cert._N - c, 0, c, 0)
        assert cert._rational_score_reject(c) is (
            _score_z(cells, -0.02) > cutoff), c
        assert cert._rational_score_reject(c) is (c <= 18), c


def test_exact_wald_strict_margin_switches_between_19_and_20():
    n = cert._N
    shifted19 = n - 50 * 19
    shifted20 = n - 50 * 20
    assert 16 * shifted19**2 * (n - 1) - 329**2 * (n * 19 - 19**2) == 197810468
    assert 16 * shifted20**2 * (n - 1) - 329**2 * (n * 20 - 20**2) == -666165212
    assert cert._exact_wald_reject(19) is True
    assert cert._exact_wald_reject(20) is False


def test_score_boundary_is_bracketed_by_independently_proven_normal_cutoff():
    cert.rational._certify_cutoff()
    n = cert._N
    assert 51 * 26 <= 2 * n  # constrained q-hat = 1/50 when z can be positive
    assert n - 50 * 26 > 0
    assert n - 50 * 27 < 0
    squared18 = Fraction((n - 50 * 18)**2, 49 * n)
    squared19 = Fraction((n - 50 * 19)**2, 49 * n)
    assert squared18 > cert.rational._HIGH_CUTOFF**2
    assert squared19 < cert.rational._LOW_CUTOFF**2
    assert cert._rational_score_reject(18) is True
    assert cert._rational_score_reject(19) is False


@pytest.mark.parametrize("cutoff", [18, 19])
def test_independently_recursive_integer_binomial_weights_match_direct_tail(cutoff):
    n = cert._N
    weight = 98**n
    total = 0
    tail = 0
    for c in range(n + 1):
        total += weight
        if c <= cutoff:
            tail += weight
        if c < n:
            next_num, next_den = weight * (n - c) * 2, (c + 1) * 98
            assert next_num % next_den == 0
            weight = next_num // next_den
    numerator, denominator = cert._exact_binomial_tail(cutoff)
    assert weight == 2**n
    assert total == denominator == 100**n
    assert tail == numerator
    assert numerator == sum(comb(n, k) * 2**k * 98**(n-k)
                            for k in range(cutoff + 1))
    assert 20 * numerator > denominator


def test_full_certificate_exposes_exact_rationals_and_hard_blocks_selection():
    report = cert.gsm8k_endpoint_certificate_research()
    assert report["n"] == 1319
    assert report["suite"] == "gsm8k"
    assert report["null_difference"] == "-1/50"
    assert report["endpoint_discordance_q"] == "1/50"
    assert report["normal_cutoff_exact_rational_bracket"] == ["1028/625", "16449/10000"]
    for name, cutoff, decimal_prefix in (
        ("section4_wald", 19, "0.083051931494331184038794159351695848413006240468465"),
        ("tango_score", 18, "0.054445993835112092221679282980029797093531633582306"),
    ):
        result = report[name]
        assert result["last_rejecting_reference_only_count"] == cutoff
        assert result["rejecting_endpoint_tables"] == cutoff + 1
        assert result["all_endpoint_tables_checked"] == 1320
        numerator = int(result["exact_probability_numerator"])
        denominator = int(result["exact_probability_denominator"])
        assert (numerator, denominator) == cert._exact_binomial_tail(cutoff)
        assert Fraction(numerator, denominator) > Fraction(1, 20)
        assert result["decimal_false_approval_probability"].startswith(decimal_prefix)
        with localcontext() as context:
            context.prec = 70
            assert Decimal(result["decimal_false_approval_probability"]) == (
                Decimal(numerator) / Decimal(denominator))
        assert len(result["decision_vector_sha256"]) == 64
        assert result["strictly_exceeds_nominal_one_sided_alpha"] is True
    assert report["boundary_endpoint_only_not_global_null_supremum"] is True
    assert report["fixed_benchmark_not_a_random_sample_certificate"] is True
    assert report["frozen_amendment4_method_adjudicated"] is False
    assert report["eligible_for_optional_v2_replacement"] is False
    assert report["research_only"] is True
    assert report["gen0_status"] == "BLOCKED"
    assert report["chosen"] is None
    assert report["release_approved"] is False


@pytest.mark.parametrize("bad", [-1, 1320, True, 1.0, "1", None])
def test_invalid_endpoint_counts_fail_closed(bad):
    with pytest.raises(ValueError, match="invalid endpoint"):
        cert._exact_wald_reject(bad)
    with pytest.raises(ValueError, match="invalid endpoint"):
        cert._rational_score_reject(bad)


@pytest.mark.parametrize("bad", [-2, 1320, False, 18.0, "18"])
def test_invalid_binomial_cutoff_fails_closed(bad):
    with pytest.raises(ValueError, match="invalid endpoint cutoff"):
        cert._exact_binomial_tail(bad)


def test_empty_prefix_has_zero_exact_probability():
    assert cert._exact_binomial_tail(-1) == (0, 100**cert._N)


def test_reject_nonmonotone_endpoint_predicate():
    with pytest.raises(ArithmeticError, match="not a prefix"):
        cert._prefix_rejection_cutoff(lambda c: c == 1)


def test_reject_nonboolean_endpoint_predicate():
    with pytest.raises(ArithmeticError, match="non-Boolean"):
        cert._prefix_rejection_cutoff(lambda c: 1 if c == 0 else False)


def test_ambiguous_rational_normal_threshold_fails_closed(monkeypatch):
    monkeypatch.setattr(cert.rational, "_LOW_CUTOFF", Fraction(164, 100))
    monkeypatch.setattr(cert.rational, "_HIGH_CUTOFF", Fraction(165, 100))
    with pytest.raises(ArithmeticError, match="cannot be certified"):
        cert._rational_score_reject(18)


def test_tampered_normal_cutoff_certificate_fails_closed(monkeypatch):
    monkeypatch.setattr(cert.rational, "_NORMAL_DENSITY_LO", Fraction(1, 2))
    with pytest.raises(ArithmeticError, match="normal density"):
        cert.gsm8k_endpoint_certificate_research()


def test_changed_but_valid_normal_quantile_bracket_is_rejected(monkeypatch):
    monkeypatch.setattr(cert.rational, "_LOW_CUTOFF", Fraction(16447, 10000))
    with pytest.raises(ArithmeticError, match="normal cutoff bracket changed"):
        cert.gsm8k_endpoint_certificate_research()


def test_tampered_monotone_cutoff_is_rejected(monkeypatch):
    original = cert._exact_wald_reject
    monkeypatch.setattr(cert, "_exact_wald_reject", lambda c: original(c) and c != 19)
    with pytest.raises(ArithmeticError, match="unexpected certified rejection cutoff"):
        cert.gsm8k_endpoint_certificate_research()


def test_tampered_exact_binomial_tail_is_rejected(monkeypatch):
    monkeypatch.setattr(cert, "_exact_binomial_tail", lambda cutoff: (0, 100**cert._N))
    with pytest.raises(ArithmeticError, match="invalid exact endpoint probability"):
        cert.gsm8k_endpoint_certificate_research()
