"""Independent exact-arithmetic and adversarial checks, never a release gate."""

from decimal import Decimal, localcontext
from math import comb

import pytest

from chowder_batch.reap_amendment4_exact_null_calibration_20261029 import (
    _decision_table,
)
from chowder_batch.reap_amendment4_integer_null_certificate_20261029 import (
    _DENOMINATOR,
    _N,
    _endpoint_certificate,
    _exact_wald_reject,
    _integer_binomial_weights,
    exact_integer_boundary_null_research,
    exact_integer_coefficients,
)


def _independent_decimal_score_reject(b: int, c: int) -> tuple[bool, Decimal]:
    """Recompute score statistic at 80-digit precision, not using _score_z."""
    with localcontext() as context:
        context.prec = 80
        n = Decimal(_N)
        d = -Decimal(1) / 50
        m, k = Decimal(b + c), Decimal(_N - b - c)
        a = m + Decimal(b - c) * d
        disc = a * a - 4 * n * (Decimal(b - c) * d - k * d * d)
        q = min(Decimal(1), max(abs(d), (a + disc.sqrt()) / (2 * n)))
        variance = (q - d * d) / n
        residual = Decimal(b - c) / n - d
        if variance == 0:
            return residual > 0, Decimal("Infinity")
        z = residual / variance.sqrt()
        # Independent 17-digit normal quantile. The minimum observed gap
        # is >8e-4, vastly exceeding uncertainty in this cutoff spelling.
        cutoff = Decimal("1.6448536269514722")
        return z > cutoff, abs(z - cutoff)


def test_every_score_decision_matches_independent_high_precision_computation():
    minimum_gap = Decimal("Infinity")
    for b, c, _k, score, wald in _decision_table():
        independent, gap = _independent_decimal_score_reject(b, c)
        assert independent is score, (b, c)
        assert _exact_wald_reject(b, c) is wald, (b, c)
        minimum_gap = min(minimum_gap, gap)
    assert minimum_gap > Decimal("0.0008")


def test_integer_binomial_weights_are_normalized_even_at_endpoints():
    for n, success, failure in ((0, 0, 100), (164, 0, 100),
                                (164, 100, 0), (83, 49, 51), (81, 2, 98)):
        weights = _integer_binomial_weights(n, success, failure)
        assert sum(weights) == (success + failure) ** n
    with pytest.raises(ValueError):
        _integer_binomial_weights(-1, 49, 51)


@pytest.mark.parametrize("t_numerator,t_denominator,probabilities", [
    (1, 2, (49, 53, 98)),
    (1, 4, (49, 57, 294)),
])
def test_exact_bernstein_identity_equals_independent_multinomial_integer_sum(
        t_numerator, t_denominator, probabilities):
    """Same exact numerator from two independently factored distributions."""
    score, wald = exact_integer_coefficients()
    bernstein_score = sum(
        comb(_N, j) * t_numerator ** j
        * (t_denominator - t_numerator) ** (_N - j) * score[j]
        for j in range(_N + 1)
    )
    bernstein_wald = sum(
        comb(_N, j) * t_numerator ** j
        * (t_denominator - t_numerator) ** (_N - j) * wald[j]
        for j in range(_N + 1)
    )
    p10, p01, pk = probabilities
    assert p10 + p01 + pk == t_denominator * 100
    direct_score = direct_wald = 0
    for b, c, k, score_reject, wald_reject in _decision_table():
        mass = (comb(_N, b) * comb(_N - b, c)
                * p10 ** b * p01 ** c * pk ** k)
        if score_reject:
            direct_score += mass
        if wald_reject:
            direct_wald += mass
    assert bernstein_score == direct_score
    assert bernstein_wald == direct_wald
    assert 0 <= direct_score <= (t_denominator * 100) ** _N
    assert 0 <= direct_wald <= (t_denominator * 100) ** _N


def test_exact_integer_endpoint_dominance_proves_global_maximum():
    score, wald = exact_integer_coefficients()
    assert len(score) == len(wald) == _N + 1
    assert score[-1] == max(score)
    assert wald[0] == max(wald)
    assert score[-1] == wald[-1]
    assert _DENOMINATOR == 100 ** 164
    assert _endpoint_certificate(score)["endpoint_q"] == "1"
    assert _endpoint_certificate(wald)["endpoint_q"] == "0.02"
    assert 0 < score[-1] < wald[0] < _DENOMINATOR


def test_interior_exceedance_is_refused_instead_of_fabricating_proof():
    score, _ = exact_integer_coefficients()
    corrupt = list(score)
    corrupt[1] = score[-1] + 1
    with pytest.raises(ArithmeticError, match="interior Bernstein"):
        _endpoint_certificate(tuple(corrupt))


def test_tampered_float_wald_decision_is_detected(monkeypatch):
    import chowder_batch.reap_amendment4_integer_null_certificate_20261029 as module

    original = _decision_table()
    first = original[0]
    altered = ((first[0], first[1], first[2], first[3], not first[4]),
               *original[1:])
    monkeypatch.setattr(module, "_decision_table", lambda: altered)
    module.exact_integer_coefficients.cache_clear()
    try:
        with pytest.raises(ArithmeticError, match="floating Wald decision"):
            module.exact_integer_coefficients()
    finally:
        module.exact_integer_coefficients.cache_clear()


def test_certificate_is_research_only_and_not_gate_authority():
    result = exact_integer_boundary_null_research()
    assert result["tango_score"]["all_integer_coefficients_at_most_attained_endpoint"]
    assert result["section4_wald"]["all_integer_coefficients_at_most_attained_endpoint"]
    assert result["tango_score"]["integer_coefficients_sha256"] == (
        "81cc0ffa828ffc16a2781c7b2b1e7d73518650dc9aaff3e206f7cd9887fa857b"
    )
    assert result["section4_wald"]["integer_coefficients_sha256"] == (
        "079c9c99a3f4813d7d4979c925a93805cb71491e590421e62bd81acce29e8958"
    )
    assert result["conditional_on_research_decision_table"] is True
    assert result["normal_quantile_predicate_formally_certified"] is False
    assert result["frozen_amendment4_method_adjudicated"] is False
    assert result["eligible_for_optional_v2_replacement"] is False
    assert result["research_only"] is True
    assert result["gen0_status"] == "BLOCKED"
    assert result["chosen"] is None
    assert result["release_approved"] is False
