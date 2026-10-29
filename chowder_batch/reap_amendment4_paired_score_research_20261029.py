"""Research-only paired risk-difference interval for Amendment 4 method review.

The frozen preregistration says "paired McNemar" and a one-sided 95% lower
confidence bound for a *nonzero* (-0.02) risk-difference margin. Classical
McNemar tests equality of paired marginal proportions, not that bound.

Tango (1998, Statistics in Medicine 17:891-908) derives an efficient-score
noninferiority test and interval, of which McNemar's equality test is a special
case. Fagerland, Lydersen & Laake (2014, Statistics in Medicine 33:2850-2875)
recommend paired score intervals among several alternatives. This module
implements score inversion for **method comparison**, not a retrospective
choice of the frozen gate's interpretation.

Never import this module into the frozen selector or use its result as a
release/qualification decision without a separately approved, prospective
statistical-method adjudication. It consumes only aggregate paired counts;
no model inference, benchmark scoring, or original evidence mutation occurs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import NormalDist


_Z95 = NormalDist().inv_cdf(0.95)
_MARGIN = -0.02
_FROZEN_N = {"gsm8k": 1319, "humaneval": 164}
_SCHEMA = "chowder.reap-amendment4-paired-score-method-research.v1"


@dataclass(frozen=True)
class PairedCounts:
    both_correct: int
    candidate_only: int
    reference_only: int
    both_incorrect: int

    def __post_init__(self) -> None:
        cells = (self.both_correct, self.candidate_only,
                 self.reference_only, self.both_incorrect)
        if any(type(x) is not int or x < 0 for x in cells) or sum(cells) < 2:
            raise ValueError("paired cells must be nonnegative integers with n >= 2")

    @property
    def n(self) -> int:
        return (self.both_correct + self.candidate_only
                + self.reference_only + self.both_incorrect)

    @property
    def difference(self) -> float:
        return (self.candidate_only - self.reference_only) / self.n

    def reversed(self) -> PairedCounts:
        return PairedCounts(self.both_correct, self.reference_only,
                            self.candidate_only, self.both_incorrect)


def section4_wald_lower_diagnostic(cells: PairedCounts) -> float:
    """Exactly the frozen §4 sample-SD statistic, not a McNemar NI interval."""
    delta = cells.difference
    discordant = cells.candidate_only + cells.reference_only
    variance = (discordant - cells.n * delta * delta) / (cells.n - 1)
    return delta - 1.645 * math.sqrt(max(0.0, variance / cells.n))


def exact_conditional_mcnemar_equality_p(cells: PairedCounts) -> float:
    """Two-sided conditional equality test; NOT noninferiority at -0.02."""
    b, c = cells.candidate_only, cells.reference_only
    discordant = b + c
    if not discordant:
        return 1.0
    k = min(b, c)
    coefficient = 1
    cumulative = 1
    for i in range(1, k + 1):
        coefficient = coefficient * (discordant - i + 1) // i
        cumulative += coefficient
    return min(1.0, (2 * cumulative) / (1 << discordant))


def _constrained_discordance(cells: PairedCounts, null_difference: float) -> float:
    """Constrained multinomial MLE of p10+p01 at fixed p10-p01.

    Maximize b*log((q+d)/2) + c*log((q-d)/2) +
    (n-b-c)*log(1-q) over |d| <= q <= 1. The nonnegative quadratic
    root solves the profile score equation; it also handles zero cells.
    """
    d = null_difference
    if not -1.0 <= d <= 1.0 or not math.isfinite(d):
        raise ValueError("null paired difference must be finite in [-1, 1]")
    b, c, n = cells.candidate_only, cells.reference_only, cells.n
    m, k = b + c, n - b - c
    a = m + (b - c) * d
    discriminant = a * a - 4 * n * ((b - c) * d - k * d * d)
    q = (a + math.sqrt(max(0.0, discriminant))) / (2 * n)
    return min(1.0, max(abs(d), q))


def _score_z(cells: PairedCounts, null_difference: float) -> float:
    """Efficient-score z for the paired marginal difference under the null."""
    q = _constrained_discordance(cells, null_difference)
    variance = max(0.0, (q - null_difference * null_difference) / cells.n)
    residual = cells.difference - null_difference
    if variance == 0.0:
        return math.copysign(math.inf, residual) if residual else 0.0
    return residual / math.sqrt(variance)


def tango_score_lower_diagnostic(cells: PairedCounts) -> float:
    """Invert the one-sided 95% paired score test (Tango/Nam formulation).

    A lower bound L solves Z(L)=Phi^-1(0.95). This is an independent
    *candidate* interpretation for prospective review, never gate authority.
    """
    if cells.difference == -1.0:
        return -1.0
    lo, hi = -1.0, cells.difference
    for _ in range(90):
        mid = (lo + hi) / 2
        if _score_z(cells, mid) > _Z95:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def tango_score_upper_diagnostic(cells: PairedCounts) -> float:
    return -tango_score_lower_diagnostic(cells.reversed())


def research_comparison(cells: PairedCounts, *, suite: str) -> dict:
    """Return transparent statistics while hard-coding *no* eligibility."""
    if suite not in _FROZEN_N or cells.n != _FROZEN_N[suite]:
        raise ValueError("paired sample differs from frozen canonical suite population")
    wald = section4_wald_lower_diagnostic(cells)
    score = tango_score_lower_diagnostic(cells)
    return {
        "schema": _SCHEMA,
        "suite": suite,
        "n": cells.n,
        "paired_cells": {
            "both_correct": cells.both_correct,
            "candidate_only": cells.candidate_only,
            "reference_only": cells.reference_only,
            "both_incorrect": cells.both_incorrect,
        },
        "candidate_minus_reference": cells.difference,
        "frozen_margin": _MARGIN,
        "section4_wald_one_sided_95_lower_diagnostic": wald,
        "tango_paired_score_one_sided_95_lower_diagnostic": score,
        "exact_conditional_mcnemar_equality_p_not_NI":
            exact_conditional_mcnemar_equality_p(cells),
        "diagnostic_methods_disagree_on_margin": (wald > _MARGIN) != (score > _MARGIN),
        "frozen_amendment4_method_adjudicated": False,
        "eligible_for_optional_v2_replacement": False,
        "research_only": True,
        "gen0_status": "BLOCKED",
        "chosen": None,
        "release_approved": False,
    }
