"""Adversarial rational-cutoff certificate checks; no original inference."""

from decimal import Decimal, localcontext
from fractions import Fraction

import pytest

from chowder_batch import reap_amendment4_rational_score_certificate_20261029 as cert


def test_machin_pi_and_normal_density_brackets_are_strict():
    pi_lo, pi_hi = cert._pi_interval()
    assert pi_lo < pi_hi
    with localcontext() as context:
        context.prec = 75
        reference_pi = Decimal("3.141592653589793238462643383279502884197169399375105820974944592307816")
        assert Decimal(pi_lo.numerator) / Decimal(pi_lo.denominator) < reference_pi
        assert Decimal(pi_hi.numerator) / Decimal(pi_hi.denominator) > reference_pi
    assert cert._NORMAL_DENSITY_LO**2 * 2 * pi_hi < 1
    assert cert._NORMAL_DENSITY_HI**2 * 2 * pi_lo > 1


def test_normal_cutoff_is_enclosed_without_floating_point():
    cert._certify_cutoff()
    _, lower_cdf_upper = cert._normal_cdf_interval(cert._LOW_CUTOFF)
    upper_cdf_lower, _ = cert._normal_cdf_interval(cert._HIGH_CUTOFF)
    assert lower_cdf_upper < Fraction(19, 20) < upper_cdf_lower
    assert cert._LOW_CUTOFF == Fraction(16448, 10000)
    assert cert._HIGH_CUTOFF == Fraction(16449, 10000)


@pytest.mark.parametrize("b,c", [(0, 0), (0, 164), (164, 0), (82, 82),
                                   (0, 1), (1, 0), (80, 0), (0, 80),
                                   (30, 20), (20, 30)])
def test_exact_rational_decisions_agree_with_independent_decimal(b, c):
    with localcontext() as context:
        context.prec = 85
        n = Decimal(164)
        d = -Decimal(1) / 50
        diff = Decimal(b - c)
        m = Decimal(b + c)
        k = n - m
        a = m + diff * d
        discriminant = a * a - 4 * n * (diff * d - k * d * d)
        q = min(Decimal(1), max(abs(d), (a + discriminant.sqrt()) / (2 * n)))
        z = (diff / n - d) / ((q - d * d) / n).sqrt()
        independent = z > Decimal("1.6448536269514722")
    assert cert._certified_score_reject(b, c) is independent


def test_all_13695_cells_and_frozen_integer_proof_are_linked():
    result = cert.rational_score_decision_certificate()
    assert result["paired_tables_certified"] == 13695
    assert result["score_rejection_cells"] == 5955
    assert result["rational_score_decision_sha256"] == (
        "0eb2615268c596f3cb74ecfb3def9beab467dbe4f5101eb7fb04d00378ef990c"
    )
    assert result["normal_quantile_bracket_exact_rational_series"] is True
    assert result["score_decisions_exact_rational_interval_certified"] is True
    assert result["frozen_amendment4_method_adjudicated"] is False
    assert result["eligible_for_optional_v2_replacement"] is False
    assert result["gen0_status"] == "BLOCKED"
    assert result["chosen"] is None
    assert result["release_approved"] is False


def test_tampered_normal_density_refuses_certification(monkeypatch):
    monkeypatch.setattr(cert, "_NORMAL_DENSITY_LO", Fraction(1, 2))
    with pytest.raises(ArithmeticError, match="normal density"):
        cert.rational_score_decision_certificate()


def test_tampered_original_research_decision_refuses_certification(monkeypatch):
    from chowder_batch.reap_amendment4_exact_null_calibration_20261029 import _decision_table

    original = _decision_table()
    b, c, k, score, wald = original[0]
    monkeypatch.setattr(cert, "_decision_table", lambda: (
        (b, c, k, not score, wald), *original[1:]
    ))
    with pytest.raises(ArithmeticError, match="differs from rational certificate"):
        cert.rational_score_decision_certificate()


def test_ambiguous_cutoff_interval_fails_closed(monkeypatch):
    monkeypatch.setattr(cert, "_LOW_CUTOFF", Fraction(0))
    monkeypatch.setattr(cert, "_HIGH_CUTOFF", Fraction(10))
    with pytest.raises(ArithmeticError, match="not certified"):
        cert._certified_score_reject(30, 20)


@pytest.mark.parametrize("b,c", [(True, 0), (0, False), (-1, 1), (1, -1),
                                   (165, 0), (0, 165), (100, 100)])
def test_invalid_paired_cells_refused(b, c):
    with pytest.raises(ValueError, match="invalid canonical"):
        cert._certified_score_reject(b, c)


def test_normal_integral_outside_proven_domain_refused():
    for x in (Fraction(0), Fraction(2), Fraction(-1)):
        with pytest.raises(ValueError, match="certified only"):
            cert._normal_integral_interval(x)
