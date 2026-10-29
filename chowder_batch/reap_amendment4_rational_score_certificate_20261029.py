"""Research-only rational certificate for the Amendment 4 score decision table.

Unlike the preceding exact-integer Bernstein certificate, this module also
certifies the *normal cutoff* and all 13,695 score decisions without binary
floating point, scipy, a normal inverse-CDF implementation, or original model
evidence. It does not adjudicate Amendment 4 or authorize any frozen gate.
"""

from __future__ import annotations

from fractions import Fraction
from hashlib import sha256
from math import factorial, isqrt

from chowder_batch.reap_amendment4_exact_null_calibration_20261029 import (
    _N, _decision_table,
)

_SCHEMA = "chowder.reap-amendment4-rational-score-certificate-research.v1"
_LOW_CUTOFF = Fraction(16448, 10000)
_HIGH_CUTOFF = Fraction(16449, 10000)
_NORMAL_DENSITY_LO = Fraction(3989422804014, 10**13)
_NORMAL_DENSITY_HI = Fraction(3989422804015, 10**13)
_SQRT_SCALE = 10**20


def _alternating_arctan_interval(x: Fraction, terms: int) -> tuple[Fraction, Fraction]:
    """Alternating-series enclosure for atan(x), with 0 < x < 1."""
    if not 0 < x < 1 or terms < 1:
        raise ValueError("arctan interval requires 0 < x < 1 and terms >= 1")
    partial = sum(((-1) ** k * x ** (2 * k + 1) / (2 * k + 1)
                   for k in range(terms)), Fraction(0))
    next_partial = partial + (-1) ** terms * x ** (2 * terms + 1) / (2 * terms + 1)
    return min(partial, next_partial), max(partial, next_partial)


def _pi_interval() -> tuple[Fraction, Fraction]:
    """Machin's identity with exact rational alternating-series bounds."""
    a_lo, a_hi = _alternating_arctan_interval(Fraction(1, 5), 25)
    b_lo, b_hi = _alternating_arctan_interval(Fraction(1, 239), 10)
    return 16 * a_lo - 4 * b_hi, 16 * a_hi - 4 * b_lo


def _normal_integral_interval(x: Fraction) -> tuple[Fraction, Fraction]:
    """Enclose integral_0^x exp(-t*t/2) dt by alternating rational sums."""
    if not 0 < x < 2:
        raise ValueError("normal integral series certified only for 0 < x < 2")
    partial = sum(((-1) ** k * x ** (2 * k + 1)
                   / ((2 * k + 1) * 2**k * factorial(k))
                   for k in range(25)), Fraction(0))
    next_partial = (partial - x**51 / (51 * 2**25 * factorial(25)))
    return min(partial, next_partial), max(partial, next_partial)


def _normal_cdf_interval(x: Fraction) -> tuple[Fraction, Fraction]:
    """Exact rational enclosure, using independently bracketed 1/sqrt(2*pi)."""
    pi_lo, pi_hi = _pi_interval()
    if not (_NORMAL_DENSITY_LO**2 * 2 * pi_hi < 1
            < _NORMAL_DENSITY_HI**2 * 2 * pi_lo):
        raise ArithmeticError("normal density rational bracket not certified")
    integral_lo, integral_hi = _normal_integral_interval(x)
    if integral_lo <= 0:
        raise ArithmeticError("positive normal integral not certified")
    return (Fraction(1, 2) + _NORMAL_DENSITY_LO * integral_lo,
            Fraction(1, 2) + _NORMAL_DENSITY_HI * integral_hi)


def _certify_cutoff() -> None:
    """Prove 1.6448 < Phi^-1(0.95) < 1.6449 by rational inequalities."""
    _, low_cdf_upper = _normal_cdf_interval(_LOW_CUTOFF)
    high_cdf_lower, _ = _normal_cdf_interval(_HIGH_CUTOFF)
    if not low_cdf_upper < Fraction(19, 20) < high_cdf_lower:
        raise ArithmeticError("one-sided 95% normal cutoff bracket not proven")


def _certified_score_reject(b: int, c: int) -> bool:
    """Decide the constrained-score rejection using only integer/rational bounds.

    At null d=-1/50, the constrained discordance MLE is
      q = clamp((A+sqrt(D))/(100*n), 1/50, 1),
      A=50*(b+c)-(b-c), D=A*A+4*n*(50*(b-c)+k).
    Integer sqrt bounds enclose q; compare the squared positive score to
    BOTH rational endpoints of the independently certified normal cutoff.
    Any undecidable cell fails closed rather than guessing a float result.
    """
    if type(b) is not int or type(c) is not int or min(b, c) < 0 or b + c > _N:
        raise ValueError("invalid canonical discordant cell")
    difference, k = b - c, _N - b - c
    residual_numerator = 50 * difference + _N
    if residual_numerator <= 0:
        return False
    a = 50 * (b + c) - difference
    discriminant = a * a + 4 * _N * (50 * difference + k)
    if discriminant < 0:
        raise ArithmeticError("negative constrained-MLE discriminant")
    sqrt_floor = isqrt(discriminant * _SQRT_SCALE**2)
    q_lo = Fraction(a * _SQRT_SCALE + sqrt_floor,
                    100 * _N * _SQRT_SCALE)
    q_hi = Fraction(a * _SQRT_SCALE + sqrt_floor + 1,
                    100 * _N * _SQRT_SCALE)
    q_lo = min(Fraction(1), max(Fraction(1, 50), q_lo))
    q_hi = min(Fraction(1), max(Fraction(1, 50), q_hi))
    if not Fraction(1, 50) <= q_lo <= q_hi <= 1:
        raise ArithmeticError("constrained-MLE interval invalid")
    numerator = Fraction(residual_numerator**2, 2500 * _N)
    denominator_lo = q_lo - Fraction(1, 2500)
    denominator_hi = q_hi - Fraction(1, 2500)
    if numerator > _HIGH_CUTOFF**2 * denominator_hi:
        return True
    if numerator < _LOW_CUTOFF**2 * denominator_lo:
        return False
    raise ArithmeticError(f"score decision not certified at b={b}, c={c}")


def rational_score_decision_certificate() -> dict:
    """Certify the complete research table, never a production selection."""
    _certify_cutoff()
    certified = []
    for b in range(_N + 1):
        for c in range(_N - b + 1):
            certified.append((b, c, _certified_score_reject(b, c)))
    original = tuple((b, c, score) for b, c, _k, score, _wald in _decision_table())
    if tuple(certified) != original:
        raise ArithmeticError("research score decision table differs from rational certificate")
    digest = sha256(";".join(f"{b},{c},{int(value)}" for b, c, value in certified)
                    .encode("ascii")).hexdigest()
    return {
        "schema": _SCHEMA,
        "suite": "humaneval",
        "n": _N,
        "paired_tables_certified": len(certified),
        "score_rejection_cells": sum(value for _, _, value in certified),
        "rational_score_decision_sha256": digest,
        "normal_quantile_bracket": [str(_LOW_CUTOFF), str(_HIGH_CUTOFF)],
        "normal_quantile_bracket_exact_rational_series": True,
        "score_decisions_exact_rational_interval_certified": True,
        "frozen_amendment4_method_adjudicated": False,
        "eligible_for_optional_v2_replacement": False,
        "research_only": True,
        "gen0_status": "BLOCKED",
        "chosen": None,
        "release_approved": False,
    }
