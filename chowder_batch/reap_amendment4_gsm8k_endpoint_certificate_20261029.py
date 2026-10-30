"""Research-only exact GSM8K Amendment 4 boundary-null endpoint certificate.

This analyzes a *hypothetical iid paired-multinomial population*, not model
outputs or the fixed canonical benchmark. At delta=p10-p01=-1/50 and the
minimum feasible discordance q=p10+p01=1/50, the candidate-only count is
zero and the reference-only count C is Binomial(1319, 1/50).

Certify the Section 4 Wald and illustrative Tango/Nam score rejection sets
using integer/rational comparisons, then sum exact binomial masses. This
endpoint counterexample proves a lower bound on each method's worst-case
null rejection probability; it does not certify a nuisance-continuum maximum,
select the unresolved Amendment 4 method, or approve a frozen gate.
"""

from __future__ import annotations

from decimal import Decimal, localcontext
from fractions import Fraction
from hashlib import sha256
from math import comb

from chowder_batch import reap_amendment4_rational_score_certificate_20261029 as rational

_N = 1319
_MARGIN = Fraction(-1, 50)
_ALPHA = Fraction(1, 20)
_SCHEMA = "chowder.reap-amendment4-gsm8k-endpoint-certificate-research.v1"
_EXPECTED_CUTOFFS = {"section4_wald": 19, "tango_score": 18}


def _exact_wald_reject(c: int) -> bool:
    """Decide frozen Section 4's z=329/200 sample-SD Wald bound exactly."""
    if type(c) is not int or not 0 <= c <= _N:
        raise ValueError("invalid endpoint reference-only count")
    shifted = _N - 50 * c
    return (shifted > 0 and
            16 * shifted**2 * (_N - 1) > 329**2 * (_N * c - c**2))


def _rational_score_reject(c: int) -> bool:
    """Score decision using the certified two-sided bracket on z_0.95.

    For B=0 and a positive observed-minus-null difference, C<=26 and the
    constrained likelihood's derivative at q=1/50 is nonpositive:
    51*C<=2*n. Its concavity therefore fixes q_hat=1/50, yielding
    Z=(n-50*C)/(7*sqrt(n)). If C>=27, Z cannot be positive. The caller
    must first establish the normal-cutoff bracket via rational._certify_cutoff.
    """
    if type(c) is not int or not 0 <= c <= _N:
        raise ValueError("invalid endpoint reference-only count")
    shifted = _N - 50 * c
    if shifted <= 0:
        return False
    if 51 * c > 2 * _N:
        raise ArithmeticError("positive score outside proven nuisance-MLE boundary")
    squared_z = Fraction(shifted**2, 49 * _N)
    if squared_z > rational._HIGH_CUTOFF**2:
        return True
    if squared_z < rational._LOW_CUTOFF**2:
        return False
    raise ArithmeticError(f"score decision cannot be certified at c={c}")


def _prefix_rejection_cutoff(decision) -> tuple[int, tuple[bool, ...]]:
    """Exhaustively prove the endpoint rejects exactly a low-C prefix."""
    decisions = tuple(decision(c) for c in range(_N + 1))
    if any(type(value) is not bool for value in decisions):
        raise ArithmeticError("non-Boolean endpoint decision")
    first_accept = next((i for i, reject in enumerate(decisions) if not reject), _N + 1)
    if any(decisions[first_accept:]):
        raise ArithmeticError("endpoint rejection predicate is not a prefix")
    return first_accept - 1, decisions


def _exact_binomial_tail(cutoff: int) -> tuple[int, int]:
    if type(cutoff) is not int or not -1 <= cutoff <= _N:
        raise ValueError("invalid endpoint cutoff")
    return (sum(comb(_N, c) * 2**c * 98**(_N - c)
                for c in range(cutoff + 1)), 100**_N)


def gsm8k_endpoint_certificate_research() -> dict:
    """Prove exact endpoint rejection probabilities; keep Gen-0 blocked."""
    # This established rational-series certificate proves the normal cutoff
    # without relying on a floating inverse-CDF or a fixed decimal constant.
    rational._certify_cutoff()
    if not (rational._LOW_CUTOFF == Fraction(16448, 10000)
            and rational._HIGH_CUTOFF == Fraction(16449, 10000)):
        raise ArithmeticError("normal cutoff bracket changed")

    # Exact binomial normalization independently checks every endpoint mass.
    denominator = 100**_N
    if sum(comb(_N, c) * 2**c * 98**(_N - c)
           for c in range(_N + 1)) != denominator:
        raise ArithmeticError("endpoint binomial weights failed normalization")

    methods = {}
    for name, decision in (("section4_wald", _exact_wald_reject),
                           ("tango_score", _rational_score_reject)):
        cutoff, decisions = _prefix_rejection_cutoff(decision)
        if cutoff != _EXPECTED_CUTOFFS[name]:
            raise ArithmeticError(f"{name}: unexpected certified rejection cutoff")
        numerator, actual_denominator = _exact_binomial_tail(cutoff)
        if actual_denominator != denominator or not 0 < numerator < denominator:
            raise ArithmeticError("invalid exact endpoint probability")
        probability = Fraction(numerator, denominator)
        with localcontext() as context:
            context.prec = 70
            decimal_probability = str(Decimal(numerator) / Decimal(denominator))
        methods[name] = {
            "last_rejecting_reference_only_count": cutoff,
            "rejecting_endpoint_tables": sum(decisions),
            "all_endpoint_tables_checked": _N + 1,
            "decision_vector_sha256": sha256(
                "".join("1" if value else "0" for value in decisions).encode("ascii")
            ).hexdigest(),
            "exact_probability_numerator": str(numerator),
            "exact_probability_denominator": str(denominator),
            "decimal_false_approval_probability": decimal_probability,
            "strictly_exceeds_nominal_one_sided_alpha": probability > _ALPHA,
        }
        if probability <= _ALPHA:
            raise ArithmeticError(f"{name}: claimed endpoint counterexample not proven")

    return {
        "schema": _SCHEMA,
        "suite": "gsm8k",
        "n": _N,
        "paired_model": "hypothetical iid paired multinomial observations",
        "null_difference": "-1/50",
        "endpoint_discordance_q": "1/50",
        "endpoint_p10_candidate_only": "0",
        "endpoint_p01_reference_only": "1/50",
        "endpoint_p_concordant": "49/50",
        "endpoint_count_distribution": "C~Binomial(1319,1/50), B=0",
        "nominal_one_sided_alpha": "1/20",
        "normal_cutoff_exact_rational_bracket": [
            str(rational._LOW_CUTOFF), str(rational._HIGH_CUTOFF)],
        "boundary_endpoint_only_not_global_null_supremum": True,
        "fixed_benchmark_not_a_random_sample_certificate": True,
        "section4_wald": methods["section4_wald"],
        "tango_score": methods["tango_score"],
        "frozen_amendment4_method_adjudicated": False,
        "eligible_for_optional_v2_replacement": False,
        "research_only": True,
        "gen0_status": "BLOCKED",
        "chosen": None,
        "release_approved": False,
    }
