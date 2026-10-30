"""Research-only exact-integer continuum check for Amendment 4 null size.

The earlier Bernstein enclosure uses floating-point coefficients and an
explicit numerical slack. Here all 165 Bernstein coefficients are calculated
as integer numerators with the COMMON denominator 100**164. If every
coefficient is at most an endpoint coefficient, the Bernstein convex-hull
identity proves that endpoint is the global maximum over q in [.02, 1].

This is exact arithmetic CONDITIONAL ON the Boolean research decision table.
The score predicate itself uses a floating-point normal quantile and is not
formally interval-certified here. Wald decisions are separately checked by
integer inequalities. Neither procedure is approved as the frozen gate.
No original evidence, models, F: files, or native processes are accessed.
"""

from __future__ import annotations

from decimal import Decimal, localcontext
from functools import lru_cache
from hashlib import sha256
from math import comb

from chowder_batch.reap_amendment4_exact_null_calibration_20261029 import (
    _N, _decision_table,
)

_SCHEMA = "chowder.reap-amendment4-integer-null-certificate-research.v1"
_DENOMINATOR = 100 ** _N


def _integer_binomial_weights(n: int, success: int, failure: int) -> tuple[int, ...]:
    """Binomial numerator weights, with denominator (success+failure)**n."""
    if type(n) is not int or n < 0 or success < 0 or failure < 0:
        raise ValueError("nonnegative binomial integer parameters required")
    weights = tuple(comb(n, k) * success ** k * failure ** (n - k)
                    for k in range(n + 1))
    if sum(weights) != (success + failure) ** n:
        raise ArithmeticError("integer binomial normalization failed")
    return weights


def _exact_wald_reject(b: int, c: int) -> bool:
    """Exact §4 Wald decision for z=1.645=329/200 and margin -1/50.

    The strict lower-bound test is equivalent to
      16 * (50*(b-c)+n)**2 * (n-1) > 329**2 * (n*(b+c)-(b-c)**2)
    PROVIDED (b-c)/n + 1/50 > 0. The latter is checked first.
    """
    difference = b - c
    shifted = 50 * difference + _N
    if shifted <= 0:
        return False
    return (16 * shifted * shifted * (_N - 1)
            > 329 ** 2 * (_N * (b + c) - difference * difference))


@lru_cache(maxsize=1)
def exact_integer_coefficients() -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Compute all coefficients without any floating-point probability sums."""
    decisions = [[None for _ in range(_N - b + 1)]
                 for b in range(_N + 1)]
    for b, c, _k, score, wald in _decision_table():
        if decisions[b][c] is not None:
            raise ArithmeticError("duplicate research decision table cell")
        if wald != _exact_wald_reject(b, c):
            raise ArithmeticError("floating Wald decision differs from exact integer test")
        decisions[b][c] = (score, wald)
    if any(cell is None for row in decisions for cell in row):
        raise ArithmeticError("incomplete research decision table")

    score_coefficients, wald_coefficients = [], []
    for j in range(_N + 1):
        # Endpoint A: (p10,p01,pconcordant)=(0,2,98)/100.
        # Endpoint B: (49,51,0)/100. Conditional on j B trials,
        # b~Bin(j,49/100), c0~Bin(n-j,2/100), and c=j-b+c0.
        high = _integer_binomial_weights(j, 49, 51)
        low = _integer_binomial_weights(_N - j, 2, 98)
        score_numerator = 0
        wald_numerator = 0
        for b, high_weight in enumerate(high):
            for c0, low_weight in enumerate(low):
                score, wald = decisions[b][j - b + c0]
                mass_numerator = high_weight * low_weight
                if score:
                    score_numerator += mass_numerator
                if wald:
                    wald_numerator += mass_numerator
        if not (0 <= score_numerator <= _DENOMINATOR
                and 0 <= wald_numerator <= _DENOMINATOR):
            raise ArithmeticError("invalid integer Bernstein coefficient")
        score_coefficients.append(score_numerator)
        wald_coefficients.append(wald_numerator)
    return tuple(score_coefficients), tuple(wald_coefficients)


def _endpoint_certificate(coefficients: tuple[int, ...]) -> dict:
    if len(coefficients) != _N + 1:
        raise ValueError("expected all 165 Bernstein coefficients")
    endpoint_index = 0 if coefficients[0] >= coefficients[-1] else _N
    endpoint = coefficients[endpoint_index]
    if any(value > endpoint or value < 0 for value in coefficients):
        raise ArithmeticError("interior Bernstein coefficient exceeds endpoints")
    with localcontext() as context:
        context.prec = 50
        decimal_probability = str(Decimal(endpoint) / Decimal(_DENOMINATOR))
    digest = sha256(",".join(map(str, coefficients)).encode("ascii")).hexdigest()
    return {
        "endpoint_q": "0.02" if endpoint_index == 0 else "1",
        "exact_maximum_numerator": str(endpoint),
        "exact_common_denominator": str(_DENOMINATOR),
        "decimal_maximum": decimal_probability,
        "coefficient_count": len(coefficients),
        "all_integer_coefficients_at_most_attained_endpoint": True,
        "integer_coefficients_sha256": digest,
    }


def exact_integer_boundary_null_research() -> dict:
    """Exact coefficient proof, not a formally certified test cutoff or gate."""
    score, wald = exact_integer_coefficients()
    return {
        "schema": _SCHEMA,
        "suite": "humaneval",
        "n": _N,
        "boundary_null_difference": "-0.02",
        "boundary_null_discordance_interval": ["0.02", "1"],
        "proof": "exact integer Bernstein convex-hull endpoint dominance",
        "conditional_on_research_decision_table": True,
        "normal_quantile_predicate_formally_certified": False,
        "tango_score": _endpoint_certificate(score),
        "section4_wald": _endpoint_certificate(wald),
        "frozen_amendment4_method_adjudicated": False,
        "eligible_for_optional_v2_replacement": False,
        "research_only": True,
        "gen0_status": "BLOCKED",
        "chosen": None,
        "release_approved": False,
    }
