"""Certified dyadic enclosure of a conservative paired NI lower confidence bound.

Research only: this does not reinterpret frozen Amendment 4 or approve Gen-0.
The coverage argument requires iid paired Bernoulli observations, which a fixed
benchmark does not establish. No model outputs, F: evidence or GPU are read.
"""

from __future__ import annotations

from fractions import Fraction

from chowder_batch.reap_amendment4_exact_conditional_ni_20261029 import PairedCounts


SCHEMA = "chowder.reap-amendment4-exact-ci-enclosure-research.v1"
ALPHA_NUMERATOR = 1
ALPHA_DENOMINATOR = 20
MAX_DISCORDANT = 1319
CANONICAL_POPULATIONS = (("humaneval", 164), ("gsm8k", 1319))


def binomial_upper_tail_rational(
    m: int, b: int, success_numerator: int, success_denominator: int,
) -> tuple[int, int]:
    """Exact P[Bin(m,success_numerator/success_denominator)>=b]."""
    values = (m, b, success_numerator, success_denominator)
    if any(type(value) is not int for value in values):
        raise ValueError("binomial arguments must be plain integers")
    if (not 0 <= b <= m <= MAX_DISCORDANT
            or not 0 <= success_numerator <= success_denominator
            or success_denominator <= 0):
        raise ValueError("invalid binomial probability or counts")
    denominator = success_denominator**m
    if b == 0:
        return denominator, denominator
    if success_numerator == 0:
        return 0, denominator
    if success_numerator == success_denominator:
        return denominator, denominator

    failure = success_denominator - success_numerator
    weight = failure**m
    numerator = 0
    for k in range(m + 1):
        if k >= b:
            numerator += weight
        if k < m:
            top = weight * (m - k) * success_numerator
            bottom = (k + 1) * failure
            if top % bottom:
                raise ArithmeticError("nonintegral binomial recurrence")
            weight = top // bottom
    if not 0 <= numerator <= denominator:
        raise ArithmeticError("binomial tail outside [0,1]")
    return numerator, denominator


def clopper_pearson_lower_enclosure(
    m: int, b: int, *, bits: int = 48,
) -> tuple[Fraction, Fraction]:
    """Bound the CP lower p endpoint by adjacent dyadic rationals.

    For b>0, the unique root of P[Bin(m,p)>=b]=1/20 lies in
    (lower,upper]. For b=0 or m=0 the endpoint is exactly zero.
    """
    if (type(m) is not int or type(b) is not int
            or not 0 <= b <= m <= MAX_DISCORDANT):
        raise ValueError("invalid discordance counts")
    if type(bits) is not int or not 24 <= bits <= 64:
        raise ValueError("bits must be a plain integer in [24,64]")
    if b == 0:
        return Fraction(0), Fraction(0)

    scale = 1 << bits
    lo, hi = 0, scale
    while hi - lo > 1:
        mid = (lo + hi) // 2
        numerator, denominator = binomial_upper_tail_rational(
            m, b, mid, scale,
        )
        if ALPHA_DENOMINATOR * numerator >= ALPHA_NUMERATOR * denominator:
            hi = mid
        else:
            lo = mid
    return Fraction(lo, scale), Fraction(hi, scale)


def paired_lower_confidence_research(
    cells: PairedCounts, *, suite: str, bits: int = 48,
) -> dict:
    """A conservative one-sided >=95% iid lower risk-difference enclosure.

    Let q=P(discordance), p=P(candidate-only|discordance). Conditional on
    M=m, B~Bin(m,p), so the one-sided CP bound p_L covers p with >=95%
    probability. Since delta=q(2p-1), q in [0,1],
    L=min(0,2*p_L-1) is a lower confidence bound for delta. A dyadic lower
    enclosure for p_L produces an even more conservative reported endpoint.
    """
    if not isinstance(cells, PairedCounts):
        raise ValueError("paired cells must be PairedCounts")
    expected = (next((n for name, n in CANONICAL_POPULATIONS if suite == name), None)
                if type(suite) is str else None)
    if expected is None or cells.n != expected:
        raise ValueError("paired population differs from frozen canonical suite")

    m, b = cells.discordant, cells.candidate_only
    p_lo, p_hi = clopper_pearson_lower_enclosure(m, b, bits=bits)
    delta_lo = min(Fraction(0), 2 * p_lo - 1)
    delta_hi = min(Fraction(0), 2 * p_hi - 1)
    tail_num, tail_den = binomial_upper_tail_rational(m, b, 49, 100)
    strict = 20 * tail_num < tail_den
    margin = Fraction(-1, 50)
    return {
        "schema": SCHEMA,
        "suite": suite,
        "n": cells.n,
        "discordant": m,
        "candidate_only": b,
        "reference_only": cells.reference_only,
        "precision_bits": bits,
        "cp_p_lower_enclosure": str(p_lo),
        "cp_p_upper_enclosure": str(p_hi),
        "risk_difference_lower_enclosure": str(delta_lo),
        "risk_difference_upper_enclosure": str(delta_hi),
        "risk_difference_enclosure_width_at_most": str(Fraction(2, 1 << bits)),
        "lower_enclosure_strictly_above_minus_two_percent": delta_lo > margin,
        "upper_enclosure_at_or_below_minus_two_percent": delta_hi <= margin,
        "exact_tail_strictly_below_five_percent": strict,
        "margin_enclosure_resolved": (delta_lo > margin or delta_hi <= margin),
        "iid_one_sided_coverage_at_least": "19/20",
        "fixed_benchmark_not_iid_certificate": True,
        "frozen_amendment4_method_adjudicated": False,
        "eligible_for_optional_v2_replacement": False,
        "research_only": True,
        "gen0_status": "BLOCKED",
        "chosen": None,
        "release_approved": False,
    }
