"""Research-only worst-case Amendment 4 boundary-null size enclosure.

The earlier exact-null audit evaluates a *fixed grid* of nuisance discordance
probabilities. A grid cannot establish the maximum false-approval probability
over the entire null boundary q in [0.02, 1]. Here the multinomial rejection
probability is represented as a degree-164 Bernstein polynomial in q, and
adaptive de Casteljau subdivision encloses its supremum on that continuum.

The bounds are numerical (with explicit floating-point slack), not a formal
directed-rounding proof, not a prospective choice of statistical method, and
NEVER qualification or release authority. No model data or F: evidence is read.
"""

from __future__ import annotations

from functools import lru_cache
import heapq
import math

from chowder_batch.reap_amendment4_exact_null_calibration_20261029 import (
    _N, _NULL_DIFFERENCE, _decision_table,
)

_SCHEMA = "chowder.reap-amendment4-uniform-null-enclosure-research.v1"
_SLACK = 1e-10
_LOW_Q = abs(_NULL_DIFFERENCE)
_HIGH_Q = 1.0


def _binomial_pmf(n: int, p: float) -> tuple[float, ...]:
    return tuple(math.comb(n, k) * p ** k * (1 - p) ** (n - k)
                 for k in range(n + 1))


@lru_cache(maxsize=1)
def _bernstein_coefficients() -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Exact-polynomial construction from endpoint-mixture trial identities.

    Let t=(q-0.02)/0.98. Each trial is independently drawn from endpoint A
    (p10,p01,pconcordant)=(0,.02,.98) with probability 1-t, or endpoint B
    (.49,.51,0) with probability t. Conditional on exactly j B trials,
    candidate-only B~Bin(j,.49) and extra reference-only C~Bin(n-j,.02).
    Therefore E[reject | j] is the degree-n Bernstein coefficient. Its
    convex hull bounds rejection probability on every subinterval.
    """
    decisions = [[(False, False) for _ in range(_N - b + 1)]
                 for b in range(_N + 1)]
    for b, c, _k, score, wald in _decision_table():
        decisions[b][c] = (score, wald)
    score_coefficients, wald_coefficients = [], []
    for j in range(_N + 1):
        high = _binomial_pmf(j, 0.49)
        low = _binomial_pmf(_N - j, 0.02)
        if (not math.isclose(math.fsum(high), 1, abs_tol=1e-12)
                or not math.isclose(math.fsum(low), 1, abs_tol=1e-12)):
            raise ArithmeticError("endpoint binomial mass drift")
        score_terms, wald_terms = [], []
        for b, p_b in enumerate(high):
            score_conditional, wald_conditional = [], []
            for c0, p_c0 in enumerate(low):
                score, wald = decisions[b][j - b + c0]
                if score:
                    score_conditional.append(p_c0)
                if wald:
                    wald_conditional.append(p_c0)
            score_terms.append(p_b * math.fsum(score_conditional))
            wald_terms.append(p_b * math.fsum(wald_conditional))
        score_coefficients.append(math.fsum(score_terms))
        wald_coefficients.append(math.fsum(wald_terms))
    for coefficient in (*score_coefficients, *wald_coefficients):
        if not -_SLACK <= coefficient <= 1 + _SLACK:
            raise ArithmeticError("invalid Bernstein rejection coefficient")
    return tuple(score_coefficients), tuple(wald_coefficients)


def _split_half(coefficients: tuple[float, ...]) -> tuple[
        tuple[float, ...], tuple[float, ...], float]:
    """Stable de Casteljau subdivision, preserving Bernstein convex hulls."""
    levels = [coefficients]
    while len(levels[-1]) > 1:
        previous = levels[-1]
        levels.append(tuple((a + b) / 2 for a, b in zip(previous, previous[1:])))
    n = len(coefficients) - 1
    left = tuple(levels[j][0] for j in range(n + 1))
    right = tuple(levels[n - j][j] for j in range(n + 1))
    return left, right, levels[-1][0]


def _supremum_enclosure(coefficients: tuple[float, ...], *,
                        tolerance: float, max_splits: int) -> dict:
    if (type(tolerance) not in (float, int) or not math.isfinite(tolerance)
            or not 0 < tolerance <= 0.1):
        raise ValueError("finite positive enclosure tolerance <= 0.1 required")
    if type(max_splits) is not int or max_splits < 1:
        raise ValueError("positive integer max_splits required")
    best = max(coefficients[0], coefficients[-1])
    argmax_t = 0.0 if coefficients[0] >= coefficients[-1] else 1.0
    heap = [(-max(coefficients), 0.0, 1.0, coefficients)]
    splits = 0
    while -heap[0][0] - best > tolerance:
        if splits >= max_splits:
            raise ArithmeticError("uniform null enclosure did not converge")
        _, lo, hi, current = heapq.heappop(heap)
        left, right, midpoint_value = _split_half(current)
        midpoint = (lo + hi) / 2
        if midpoint_value > best:
            best, argmax_t = midpoint_value, midpoint
        heapq.heappush(heap, (-max(left), lo, midpoint, left))
        heapq.heappush(heap, (-max(right), midpoint, hi, right))
        splits += 1
    return {
        "lower_bound": max(0.0, best - _SLACK),
        "upper_bound": min(1.0, -heap[0][0] + _SLACK),
        "q_at_observed_lower_bound": _LOW_Q + (_HIGH_Q - _LOW_Q) * argmax_t,
        "subdivisions": splits,
    }


def uniform_boundary_null_research(*, tolerance: float = 0.0001,
                                   max_splits: int = 20000) -> dict:
    """Numerically enclose maximum boundary-null false approvals, not a gate."""
    score, wald = _bernstein_coefficients()
    score_bounds = _supremum_enclosure(score, tolerance=tolerance,
                                      max_splits=max_splits)
    wald_bounds = _supremum_enclosure(wald, tolerance=tolerance,
                                     max_splits=max_splits)
    return {
        "schema": _SCHEMA,
        "suite": "humaneval",
        "n": _N,
        "null_candidate_minus_reference": _NULL_DIFFERENCE,
        "null_discordance_interval": [_LOW_Q, _HIGH_Q],
        "numerical_tolerance": tolerance,
        "floating_point_slack_not_formal_interval_arithmetic": _SLACK,
        "tango_score_supremum_enclosure": score_bounds,
        "section4_wald_supremum_enclosure": wald_bounds,
        "frozen_amendment4_method_adjudicated": False,
        "eligible_for_optional_v2_replacement": False,
        "research_only": True,
        "gen0_status": "BLOCKED",
        "chosen": None,
        "release_approved": False,
    }
