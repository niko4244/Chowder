"""Research-only full composite-null extension of the Amendment 4 audit.

The exact-integer Bernstein certificate bounds rejection on the boundary
delta = p10-p01 = -1/50, for every discordance q in [1/50, 1].  This module
proves that the boundary is least favorable throughout the *composite* null
delta <= -1/50 under the iid paired-multinomial sampling model.

For fixed q, M=B+C ~ Bin(n,q), independent of delta. Conditional on M=m,
B ~ Bin(m, (q+delta)/(2*q)). Every certified rejection set must be an upper
tail in B at each m. Binomial stochastic ordering then bounds every smaller
delta by the boundary delta=-1/50 at the same q. Since q >= abs(delta), the
boundary point is admissible at that q. This proof does not select a CI method,
authenticate model responses, or authorize any frozen gate.
"""

from __future__ import annotations

from fractions import Fraction
from hashlib import sha256

from chowder_batch import reap_amendment4_integer_null_certificate_20261029 as integer
from chowder_batch import reap_amendment4_rational_score_certificate_20261029 as rational
from chowder_batch.reap_amendment4_exact_null_calibration_20261029 import (
    _N,
    _decision_table,
)

_SCHEMA = "chowder.reap-amendment4-composite-null-certificate-research.v1"
_MARGIN = Fraction(-1, 50)
_ALPHA = Fraction(1, 20)


def _upper_tail_cutoffs(decision, *, n: int = _N) -> tuple[int, ...]:
    """Prove a binary rejection decision is increasing in B at fixed M.

    A cutoff m+1 means no rejection at that discordance count.  This is an
    exhaustive, finite check rather than an assumption of score monotonicity.
    """
    if type(n) is not int or n < 1:
        raise ValueError("positive integer paired population required")
    cutoffs = []
    for m in range(n + 1):
        first_reject = m + 1
        seen_reject = False
        for b in range(m + 1):
            value = decision(b, m - b)
            if type(value) is not bool:
                raise ValueError("paired rejection decision must be Boolean")
            if value:
                if not seen_reject:
                    first_reject = b
                seen_reject = True
            elif seen_reject:
                raise ArithmeticError(
                    f"nonmonotone paired rejection at m={m}, b={b}"
                )
        cutoffs.append(first_reject)
    return tuple(cutoffs)


def _exact_composite_maximum(endpoint: dict) -> dict:
    if (endpoint.get("all_integer_coefficients_at_most_attained_endpoint")
            is not True or endpoint.get("coefficient_count") != _N + 1):
        raise ArithmeticError("missing exact boundary Bernstein dominance")
    numerator = int(endpoint["exact_maximum_numerator"])
    denominator = int(endpoint["exact_common_denominator"])
    if denominator != 100**_N or not 0 <= numerator <= denominator:
        raise ArithmeticError("invalid exact boundary-null probability")
    maximum = Fraction(numerator, denominator)
    return {
        "attained_boundary_discordance_q": endpoint["endpoint_q"],
        "exact_maximum_numerator": str(numerator),
        "exact_maximum_denominator": str(denominator),
        "maximum_exceeds_nominal_one_sided_alpha": maximum > _ALPHA,
        "bernstein_coefficient_sha256": endpoint["integer_coefficients_sha256"],
    }


def composite_null_certificate_research() -> dict:
    """Certify the full composite-null maximum, without gate authority."""
    # First certify the normal quantile, every score decision, and the
    # integer Bernstein endpoint maximum. These functions fail closed.
    rational_report = rational.rational_score_decision_certificate()
    boundary_report = integer.exact_integer_boundary_null_research()
    if (rational_report["paired_tables_certified"] != (_N + 1) * (_N + 2) // 2
            or rational_report["score_decisions_exact_rational_interval_certified"]
            is not True):
        raise ArithmeticError("rational score decision certificate incomplete")

    decisions = {
        (b, c): (score, wald)
        for b, c, _k, score, wald in _decision_table()
    }
    if len(decisions) != (_N + 1) * (_N + 2) // 2:
        raise ArithmeticError("incomplete or duplicate paired decision table")
    for (b, c), (score, wald) in decisions.items():
        if (score != rational._certified_score_reject(b, c)
                or wald != integer._exact_wald_reject(b, c)):
            raise ArithmeticError("uncertified paired decision table cell")

    score_cutoffs = _upper_tail_cutoffs(lambda b, c: decisions[b, c][0])
    wald_cutoffs = _upper_tail_cutoffs(lambda b, c: decisions[b, c][1])
    score_digest = sha256(
        ";".join(f"{b},{c},{int(decisions[b, c][0])}"
                 for b in range(_N + 1) for c in range(_N + 1 - b))
        .encode("ascii")
    ).hexdigest()
    if score_digest != rational_report["rational_score_decision_sha256"]:
        raise ArithmeticError("score decision digest changed")

    for method in ("tango_score", "section4_wald"):
        if method not in boundary_report:
            raise ArithmeticError("missing boundary-null method certificate")
    return {
        "schema": _SCHEMA,
        "suite": "humaneval",
        "n": _N,
        "paired_tables_exactly_certified": len(decisions),
        "discordance_counts_with_proven_upper_tail": _N + 1,
        "null_hypothesis": "p10-p01 <= -1/50",
        "sampling_assumption": "iid paired multinomial observations",
        "proof": (
            "M=B+C~Bin(n,q); B|M=m~Bin(m,(q+delta)/(2q)); "
            "upper-tail rejection is stochastically nondecreasing in delta "
            "at fixed q; boundary delta=-1/50 is admissible for q>=abs(delta); "
            "exact integer Bernstein coefficients bound all boundary q"
        ),
        "score_upper_tail_cutoffs_sha256": sha256(
            ",".join(map(str, score_cutoffs)).encode("ascii")
        ).hexdigest(),
        "wald_upper_tail_cutoffs_sha256": sha256(
            ",".join(map(str, wald_cutoffs)).encode("ascii")
        ).hexdigest(),
        "rational_score_decision_sha256": score_digest,
        "tango_score_composite_null": _exact_composite_maximum(
            boundary_report["tango_score"]
        ),
        "section4_wald_composite_null": _exact_composite_maximum(
            boundary_report["section4_wald"]
        ),
        "fixed_benchmark_not_a_random_sample_certificate": True,
        "frozen_amendment4_method_adjudicated": False,
        "eligible_for_optional_v2_replacement": False,
        "research_only": True,
        "gen0_status": "BLOCKED",
        "chosen": None,
        "release_approved": False,
    }
