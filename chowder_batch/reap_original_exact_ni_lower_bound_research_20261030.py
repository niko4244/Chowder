"""Source-locked one-sided exact paired-NI lower-bound research, CPU only.

Invert the already reviewed 2.5% CP discordance bound + 2.5% exact
conditional test. This produces a conservative one-sided 95% lower
confidence bound for the *marginal paired accuracy difference* under iid
paired sampling, not an Amendment 4 CI or a frozen scientific gate.

The inversion is confined to nonpositive null differences. For positive
differences the CP *upper* discordance bound cannot be substituted into the
same null success-probability inequality. Results are clipped at zero.
No model inference, completed journal, F: evidence, or approval is mutated.
"""

from __future__ import annotations

from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import math
from pathlib import Path
import sys

from chowder_batch import reap_original_exact_ni_composite_p_witness_20261030 as witness
from chowder_batch import reap_original_gsm8k_crosshost_exact_ni_20261030 as gsm


SCHEMA = "chowder.reap-original-exact-ni-lower-bound-research.v1"
WITNESS_SHA256 = "deeb5c3503e0ca47929d6887cb3baa44f48261a4e66f31dc51d8cd0066563904"
GRID_BITS = 24
TEST_ALPHA = Fraction(1, 40)
TOTAL_ALPHA = Fraction(1, 20)
MARGIN = Fraction(1, 50)


class LowerBoundRefused(ValueError):
    """The requested research inversion is not source- or input-admissible."""


def require_source() -> None:
    if sys.flags.optimize or not __debug__:
        raise LowerBoundRefused("optimized Python cannot certify research sources")
    path = Path(__file__).resolve().parent / "reap_original_exact_ni_composite_p_witness_20261030.py"
    if (not path.is_file() or path.is_symlink()
            or hashlib.sha256(path.read_bytes()).hexdigest() != WITNESS_SHA256
            or Path(witness.__file__).resolve() != path):
        raise LowerBoundRefused("independent composite-null witness source drift")
    witness.require_source()


def exact_tail(*, discordant: int, candidate_only: int,
               null_success_probability: Fraction) -> Fraction:
    """Independent finite binomial sum; no production-tail recurrence."""
    if (type(discordant) is not int or not 0 <= discordant <= 1319
            or type(candidate_only) is not int
            or not 0 <= candidate_only <= discordant
            or type(null_success_probability) is not Fraction
            or not 0 <= null_success_probability <= 1):
        raise LowerBoundRefused("invalid exact conditional tail arguments")
    p = null_success_probability
    return sum((Fraction(math.comb(discordant, j)) * p**j
                * (1 - p)**(discordant - j)
                for j in range(candidate_only, discordant + 1)), Fraction(0))


def lower_bound(*, n: int, candidate_only: int, reference_only: int) -> dict:
    """Return a conservative dyadic lower bound and its adjacent bracket.

    For delta0<=0, the composite H0: delta<=delta0 implies q>=-delta0
    and p<=max(0,(q+delta0)/(2q)). On the CP event q<=U(M), this is
    bounded by p0=max(0,(U(M)+delta0)/(2U(M))). The exact binomial tail
    below 1/40 then rejects H0 with unconditional size <=1/20 by union
    bound. Rejection is monotone in delta0, so the largest rejected grid
    point is a valid conservative one-sided lower confidence bound.
    """
    if (type(n) is not int or n not in (164, 1319)
            or type(candidate_only) is not int or type(reference_only) is not int
            or min(candidate_only, reference_only) < 0
            or candidate_only + reference_only > n):
        raise LowerBoundRefused("invalid paired original research population")
    discordant = candidate_only + reference_only
    q_upper = gsm.discordance_q_upper(n, discordant)[1]
    scale = 1 << GRID_BITS

    def reject(k: int) -> bool:
        delta0 = -Fraction(k, scale)
        p0 = max(Fraction(0), (q_upper + delta0) / (2 * q_upper))
        return exact_tail(discordant=discordant, candidate_only=candidate_only,
                          null_success_probability=p0) < TEST_ALPHA

    if reject(0):
        k, clipped, any_rejected = 0, True, True
    elif not reject(scale):
        # No discordant candidate-only success: no negative null is rejected.
        k, clipped, any_rejected = scale, False, False
    else:
        left, right = 0, scale  # nonrejected at left; rejected at right
        while right - left > 1:
            mid = (left + right) // 2
            if reject(mid):
                right = mid
            else:
                left = mid
        k, clipped, any_rejected = right, False, True
    lower = -Fraction(k, scale)
    adjacent_nonrejected = (-Fraction(k - 1, scale)
                            if any_rejected and k > 0 else None)
    at_margin = exact_tail(
        discordant=discordant, candidate_only=candidate_only,
        null_success_probability=max(Fraction(0), (q_upper - MARGIN) / (2 * q_upper)),
    )
    composite_p = min(Fraction(1), Fraction(1, 40) + at_margin)
    with localcontext() as ctx:
        ctx.prec = 48
        composite_decimal = str(Decimal(composite_p.numerator)
                                / Decimal(composite_p.denominator))
    return {
        "schema": SCHEMA, "n": n, "candidate_only": candidate_only,
        "reference_only": reference_only, "discordant": discordant,
        "observed_difference": str(Fraction(candidate_only - reference_only, n)),
        "cp_q_upper": str(q_upper), "dyadic_bits": GRID_BITS,
        "conservative_lower_bound": str(lower),
        "adjacent_nonrejected_null": (str(adjacent_nonrejected)
                                      if adjacent_nonrejected is not None else None),
        "clipped_at_zero": clipped,
        "any_nonpositive_null_rejected": any_rejected,
        "minus_two_percent_null_rejected_under_iid": at_margin < TEST_ALPHA,
        "composite_null_p_at_minus_two_percent": str(composite_p),
        "composite_null_p_decimal": composite_decimal,
        "iid_assumption_not_certified_for_fixed_benchmark": True,
        "frozen_amendment4_method_adjudicated": False,
        "frozen_q8_vs_dense_gate2_qualified": False,
        "native_http_lineage_authenticated": False,
        "research_only": True, "status": "BLOCKED", "chosen": None,
        "release_approved": False,
    }


def verify_authentic_research() -> dict:
    """Bind two already completed original research pairs, without reruns."""
    require_source()
    original = witness.verify_original_research()
    if (original.get("status") != "BLOCKED"
            or original.get("chosen", 1) is not None
            or original.get("release_approved") is not False):
        raise LowerBoundRefused("original paired research was not fail closed")
    gsm_result = lower_bound(n=1319, candidate_only=25, reference_only=21)
    human_result = lower_bound(n=164, candidate_only=5, reference_only=4)
    for name, result in (("gsm8k_crosshost_q4_vs_q8", gsm_result),
                         ("humaneval_q8_vs_q4", human_result)):
        record = original[name]
        if (record.get("n") != result["n"]
                or record.get("candidate_only") != result["candidate_only"]
                or record.get("reference_only") != result["reference_only"]
                or record.get("conservative_composite_null_p_rational")
                != result["composite_null_p_at_minus_two_percent"]
                or record.get("research_only") is not True):
            raise LowerBoundRefused("source-certified original exact research disagrees")
    return {
        "schema": SCHEMA, "gsm8k_crosshost_q4_vs_q8": gsm_result,
        "humaneval_q8_vs_q4": human_result,
        "statistical_scope": "iid_paired_research_only_not_frozen_amendment4",
        "status": "BLOCKED", "chosen": None, "release_approved": False,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(verify_authentic_research(), indent=2, sort_keys=True))
