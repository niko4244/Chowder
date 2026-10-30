"""Source-bound correction of conditional-tail versus composite-null p-values.

The existing exact paired NI research splits 5% between a one-sided 97.5%
Clopper--Pearson upper bound on discordance and a 2.5% conditional test.
The reported *conditional* tail is NOT by itself a valid p-value for the
composite marginal-risk-difference null. With the predeclared 1/40 bound
error, ``min(1, 1/40 + conditional_tail)`` is a conservative, superuniform
composite-null p-value under the same iid paired-observation assumption.

Proof: with probability at least 39/40, true discordance q is below its
random CP upper bound U(M). On this event and under delta <= -1/50,
the conditional candidate-only success probability is <= p_max(U(M)).
For each M satisfying that event, the exact upper-tail test is superuniform.
For t < 1/40, P(1/40 + T <= t) = 0. For 1/40 <= t < 1,
P(1/40 + T <= t) <= 1/40 + (t-1/40) = t; at t=1 the
probability is at most one. Thus the capped adjusted p-value is valid.
No claim of iid for a fixed benchmark, frozen Amendment 4 adoption, original
Q8-vs-dense qualification, or authentic native HTTP lineage is made here.
No original inference, F: evidence, source, or certificate is modified.
"""

from __future__ import annotations

from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import math
from pathlib import Path
import sys

from chowder_batch import reap_original_gsm8k_crosshost_exact_ni_20261030 as gsm
from chowder_batch import reap_original_human164_crossquant_exact_ni_20261030 as human


SCHEMA = "chowder.reap-original-exact-ni-composite-p-witness.v1"
ALPHA_BOUND = Fraction(1, 40)
ALPHA_TEST = Fraction(1, 40)
MARGIN = Fraction(1, 50)
SOURCE_SHA256 = {
    "reap_original_gsm8k_crosshost_exact_ni_20261030.py":
        "0b20f89fc4fa1ce1eac7d4a9a20ca8ca31d34c9190634b182913d69f3c0cdef4",
    "reap_original_human164_crossquant_exact_ni_20261030.py":
        "e0ffe96fba5472e96490798116a0b883615487ca826ed93853cff7533d13bcba",
}


class CompositePRefused(ValueError):
    """Source or exact research statistics are not authentic."""


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise CompositePRefused(reason)


def require_source() -> None:
    _require(sys.flags.optimize == 0 and __debug__, "optimized Python not admitted")
    directory = Path(__file__).resolve().parent
    for name, digest in SOURCE_SHA256.items():
        path = directory / name
        _require(path.is_file() and not path.is_symlink()
                 and hashlib.sha256(path.read_bytes()).hexdigest() == digest,
                 "source-pinned exact NI research method changed: " + name)
    _require(Path(gsm.__file__).resolve() == directory /
             "reap_original_gsm8k_crosshost_exact_ni_20261030.py"
             and Path(human.__file__).resolve() == directory /
             "reap_original_human164_crossquant_exact_ni_20261030.py",
             "exact NI method imported outside approved worktree")
    gsm._source_lock()
    human._source_lock()


def independently_recount_tail(*, n: int, candidate_only: int,
                               reference_only: int) -> tuple[Fraction, Fraction]:
    """Independent binomial-coefficient sum; no production tail recurrence.

    Return the conditional tail and the conservative composite-null p-value.
    The CP endpoint still comes from the separately source-locked exact method.
    """
    _require(type(n) is int and n in (164, 1319)
             and type(candidate_only) is int and type(reference_only) is int
             and candidate_only >= 0 and reference_only >= 0
             and candidate_only + reference_only <= n,
             "invalid paired outcome counts")
    m = candidate_only + reference_only
    q_lo, q_hi = gsm.discordance_q_upper(n, m)
    _require(Fraction(0) <= q_lo <= q_hi <= Fraction(1)
             and q_hi > 0, "invalid exact CP upper endpoint")
    p_null = max(Fraction(0), (q_hi - MARGIN) / (2 * q_hi))
    _require(Fraction(0) <= p_null <= Fraction(1), "invalid null p bound")
    tail = sum((Fraction(math.comb(m, k)) * p_null**k
                * (1 - p_null)**(m - k)
                for k in range(candidate_only, m + 1)), Fraction(0))
    _require(Fraction(0) <= tail <= Fraction(1), "invalid exact conditional tail")
    return tail, min(Fraction(1), ALPHA_BOUND + tail)


def _decimal(value: Fraction) -> str:
    with localcontext() as ctx:
        ctx.prec = 48
        return str(Decimal(value.numerator) / Decimal(value.denominator))


def _witness(*, arm: str, counts: tuple[int, int, int, int],
             original: dict) -> dict:
    _require(type(counts) is tuple and len(counts) == 4
             and all(type(x) is int and x >= 0 for x in counts),
             "invalid authenticated paired cells")
    n = sum(counts)
    conditional, adjusted = independently_recount_tail(
        n=n, candidate_only=counts[1], reference_only=counts[2])
    original_hash = hashlib.sha256(
        f"{conditional.numerator}/{conditional.denominator}".encode("ascii")
    ).hexdigest()
    _require(original.get("exact_conditional_p_rational_sha256") == original_hash
             and original.get("n", original.get("population")) == n
             and original.get("status") == "BLOCKED"
             and original.get("chosen") is None
             and original.get("release_approved") is False
             and original.get("research_only") is True,
             "source-bound original conditional research disagrees")
    # The original strict 2.5% conditional decision is equivalent to the
    # conservative composite p-value being strictly below the full 5% budget.
    _require((conditional < ALPHA_TEST) == (adjusted < 2 * ALPHA_TEST),
             "composite null significance accounting inconsistent")
    return {
        "schema": SCHEMA, "arm": arm, "n": n,
        "both_correct": counts[0], "candidate_only": counts[1],
        "reference_only": counts[2], "neither_correct": counts[3],
        "conditional_tail_exact_rational_sha256": original_hash,
        "conditional_tail_decimal": _decimal(conditional),
        "cp_discordance_error_budget": str(ALPHA_BOUND),
        "conditional_test_error_budget": str(ALPHA_TEST),
        "conservative_composite_null_p_decimal": _decimal(adjusted),
        "conservative_composite_null_p_rational": str(adjusted),
        "reject_composite_null_at_five_percent_under_iid":
            adjusted < Fraction(1, 20),
        "conditional_tail_is_not_composite_null_p_value": True,
        "iid_fixed_benchmark_not_established": True,
        "original_native_http_lineage_authenticated": False,
        "frozen_amendment4_method_adjudicated": False,
        "frozen_q8_vs_dense_gate2_qualified": False,
        "research_only": True, "status": "BLOCKED", "chosen": None,
        "release_approved": False,
    }


def verify_original_research() -> dict:
    """Only certified, previously completed original research; no new runs."""
    require_source()
    g = gsm.verify_authentic_crosshost_research()
    h = human.verify_authentic_pair()
    _require(g.get("paired_cells") == {
        "both_correct": 1199, "candidate_only": 25,
        "reference_only": 21, "both_incorrect": 74,
    } and h.get("paired_cells") == {
        "both_correct": 147, "q8_only": 5, "q4_only": 4,
        "neither_correct": 8,
    }, "certified original paired counts differ")
    return {
        "schema": SCHEMA,
        "gsm8k_crosshost_q4_vs_q8": _witness(
            arm="original_crosshost_gsm8k_q4_vs_q8",
            counts=(1199, 25, 21, 74), original=g),
        "humaneval_q8_vs_q4": _witness(
            arm="original_research_humaneval_q8_vs_q4",
            counts=(147, 5, 4, 8), original=h),
        "scientific_status": "RESEARCH_ONLY_NOT_FROZEN_GATE",
        "status": "BLOCKED", "chosen": None, "release_approved": False,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(verify_original_research(), sort_keys=True, indent=2))
