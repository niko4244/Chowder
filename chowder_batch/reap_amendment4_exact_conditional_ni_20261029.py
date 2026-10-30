"""Research-only exact conditional paired noninferiority test for Amendment 4.

This is a *prospective-method candidate*, not the frozen Amendment 4 method.
It never reads original model evidence, scores benchmarks, or authorizes Gen-0.

For iid paired Bernoulli outcomes let B be candidate-only correct, C be
reference-only correct, M=B+C, q=P(discordance), and delta=P(B)-P(C).
Under H0: delta <= -1/50, for q>0,

    P(B | discordance) = (q+delta)/(2*q) <= 49/100,

because q<=1. Conditional on M=m, B is Binomial(m,p), p<=49/100.
Consequently the exact upper-tail Binomial(m,49/100) p-value is superuniform
*conditional on every m*, and therefore unconditionally, for the entire
composite null. This gives finite-sample size at most 1/20 without relying on
Wald or score asymptotics. It is conservative when q<1, potentially severely.

This is NOT a classical McNemar equality test, nor is it the frozen §4 Wald
bound or an independently approved interpretation of "paired McNemar" at a
nonzero margin. A fixed benchmark is not an iid sampling certificate.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext


SCHEMA = "chowder.reap-amendment4-exact-conditional-ni-research.v1"
FROZEN_COUNTS = {"gsm8k": 1319, "humaneval": 164}
MARGIN_NUMERATOR = -1
MARGIN_DENOMINATOR = 50
NULL_DISCORDANT_SUCCESS_NUMERATOR = 49
NULL_DISCORDANT_SUCCESS_DENOMINATOR = 100
ALPHA_DENOMINATOR = 20


@dataclass(frozen=True)
class PairedCounts:
    both_correct: int
    candidate_only: int
    reference_only: int
    both_incorrect: int

    def __post_init__(self) -> None:
        values = (self.both_correct, self.candidate_only,
                  self.reference_only, self.both_incorrect)
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("paired counts must be nonnegative plain integers")
        if sum(values) < 1:
            raise ValueError("paired population must be nonempty")

    @property
    def n(self) -> int:
        return (self.both_correct + self.candidate_only
                + self.reference_only + self.both_incorrect)

    @property
    def discordant(self) -> int:
        return self.candidate_only + self.reference_only


def exact_upper_tail(discordant: int, candidate_only: int) -> tuple[int, int]:
    """Return exact P[Bin(discordant,49/100)>=candidate_only] as integers.

    The returned denominator is always 100**discordant. No floating-point
    arithmetic, approximation, observed score, or mutable external scorer is
    involved. A zero-discordance population has p-value exactly one.
    """
    if (type(discordant) is not int or type(candidate_only) is not int
            or not 0 <= candidate_only <= discordant <= 1319):
        raise ValueError("invalid exact conditional discordance counts")

    denominator = 100 ** discordant
    weight = 51 ** discordant
    numerator = 0
    for b in range(discordant + 1):
        if b >= candidate_only:
            numerator += weight
        if b < discordant:
            next_numerator = weight * (discordant - b) * 49
            next_denominator = (b + 1) * 51
            if next_numerator % next_denominator:
                raise ArithmeticError("exact binomial recurrence not integral")
            weight = next_numerator // next_denominator
    if not 0 <= numerator <= denominator:
        raise ArithmeticError("exact conditional tail outside probability range")
    return numerator, denominator


def exact_conditional_ni_research(cells: PairedCounts, *, suite: str) -> dict:
    """Exact null test, never an original frozen scientific gate decision."""
    if suite not in FROZEN_COUNTS or cells.n != FROZEN_COUNTS[suite]:
        raise ValueError("paired sample differs from frozen canonical suite")
    numerator, denominator = exact_upper_tail(
        cells.discordant, cells.candidate_only,
    )
    with localcontext() as context:
        context.prec = 45
        decimal_p = str(Decimal(numerator) / Decimal(denominator))
    return {
        "schema": SCHEMA,
        "suite": suite,
        "n": cells.n,
        "paired_cells": {
            "both_correct": cells.both_correct,
            "candidate_only": cells.candidate_only,
            "reference_only": cells.reference_only,
            "both_incorrect": cells.both_incorrect,
        },
        "discordant": cells.discordant,
        "candidate_minus_reference_numerator": (
            cells.candidate_only - cells.reference_only
        ),
        "candidate_minus_reference_denominator": cells.n,
        "null_difference": "-1/50",
        "null_discordant_success_upper_bound": "49/100",
        "exact_one_sided_p_numerator": str(numerator),
        "exact_one_sided_p_denominator": str(denominator),
        "exact_one_sided_p_decimal": decimal_p,
        "strict_p_below_one_twentieth": 20 * numerator < denominator,
        "iid_composite_null_type_i_upper_bound": "1/20",
        "fixed_canonical_benchmark_not_iid_certificate": True,
        "not_frozen_mcnemar_method_or_wald_bound": True,
        "independent_prospective_method_approval_required": True,
        "frozen_amendment4_method_adjudicated": False,
        "eligible_for_optional_v2_replacement": False,
        "research_only": True,
        "gen0_status": "BLOCKED",
        "chosen": None,
        "release_approved": False,
    }
