"""Prospective exact paired-NI research on the *already completed* Q4/Q8 pair.

This is NOT Amendment 4's frozen method, an iid certificate for a fixed test
set, an original same-host Q8-vs-dense comparison, or Gen-0 admission.
No model inference, F: writes, benchmark reruns or new approval pins occur.

Under iid paired outcomes let q=P(discordant), p=P(candidate-only|discordant)
and delta=q(2*p-1). Under H0 delta<=-1/50, q>=1/50 and
p <= (q-1/50)/(2*q). Obtain a one-sided (1-1/40) Clopper-Pearson
*upper* bound U(M) for q from M~Bin(n,q). Conditional on M=m and
q<=U(m), p <= max(0,(U(m)-1/50)/(2*U(m))). A one-sided exact binomial
tail at that upper p is a level-1/40 conditional test. Union bounding the
CP upper-bound failure and the conditional test gives size <=1/20 under
the entire composite null. Dyadic rounding of U is UPWARD, never liberal.
"""

from __future__ import annotations

from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import math
from pathlib import Path
import sys

from chowder_batch import reap_amendment4_exact_ci_enclosure_20261029 as exact
from chowder_batch import reap_original_gsm8k_windows_q4_linux_q8_pair_20261030 as original


SCHEMA = "chowder.reap-original-gsm8k-crosshost-exact-ni-research.v1"
ALPHA_PART_NUM = 1
ALPHA_PART_DEN = 40
MAX_N = 1319
BITS = 40
PINNED_SOURCES = {
    "reap_original_gsm8k_windows_q4_linux_q8_pair_20261030.py":
        "3f808ca865d624cccbc33268703ccef6375925c9368dba638b763623b6750c70",
    "reap_amendment4_exact_ci_enclosure_20261029.py":
        "f01499a6c57df709e2a92c3a5e3d54d0b2a2a5023a833d81d1b1f58672d047f4",
}
EXPECTED_COUNTS = (1199, 25, 21, 74)


class ExactNIRefused(ValueError):
    """Input, original evidence or exact arithmetic is not admissible."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExactNIRefused(message)


def _source_lock() -> None:
    _require(sys.flags.optimize == 0 and __debug__, "optimized Python is not source-approved")
    parent = Path(__file__).resolve().parent
    for name, digest in PINNED_SOURCES.items():
        path = parent / name
        _require(path.is_file() and not path.is_symlink()
                 and hashlib.sha256(path.read_bytes()).hexdigest() == digest,
                 "original statistical/source verifier drift: " + name)
    _require(Path(original.__file__).resolve() == parent / next(iter(PINNED_SOURCES))
             and Path(exact.__file__).resolve()
             == parent / "reap_amendment4_exact_ci_enclosure_20261029.py",
             "source-bound statistical modules imported outside authorized worktree")


def binomial_lower_tail(n: int, m: int, num: int, den: int) -> tuple[int, int]:
    """Exact P[Bin(n,num/den)<=m], with a short-side integer recurrence."""
    if (any(type(v) is not int for v in (n, m, num, den))
            or not 1 <= n <= MAX_N or not 0 <= m <= n
            or not 0 <= num <= den or den <= 0):
        raise ExactNIRefused("invalid exact binomial lower-tail arguments")
    denominator = den**n
    if m == n or num == 0:
        return denominator, denominator
    if num == den:
        return 0, denominator
    fail = den - num
    if m <= n // 2:
        weight = fail**n
        total = weight
        for k in range(m):
            top = weight * (n - k) * num
            divisor = (k + 1) * fail
            _require(top % divisor == 0, "nonintegral lower-tail recurrence")
            weight = top // divisor
            total += weight
    else:
        weight = num**n
        upper = weight
        for k in range(n, m + 1, -1):
            top = weight * k * fail
            divisor = (n - k + 1) * num
            _require(top % divisor == 0, "nonintegral upper-tail recurrence")
            weight = top // divisor
            upper += weight
        total = denominator - upper
    _require(0 <= total <= denominator, "binomial lower tail out of bounds")
    return total, denominator


def discordance_q_upper(n: int, m: int, *, bits: int = BITS) -> tuple[Fraction, Fraction]:
    """Adjacent dyadics enclosing the exact one-sided 97.5% CP q upper bound."""
    if (type(n) is not int or type(m) is not int or type(bits) is not int
            or not 1 <= n <= MAX_N or not 0 <= m <= n
            or not 24 <= bits <= 48):
        raise ExactNIRefused("invalid CP upper-bound arguments")
    if m == n:
        return Fraction(1), Fraction(1)
    scale = 1 << bits
    lo, hi = 0, scale
    while hi - lo > 1:
        mid = (lo + hi) // 2
        tail, denominator = binomial_lower_tail(n, m, mid, scale)
        if ALPHA_PART_DEN * tail <= ALPHA_PART_NUM * denominator:
            hi = mid
        else:
            lo = mid
    return Fraction(lo, scale), Fraction(hi, scale)


def exact_research(cells: tuple[int, int, int, int], *, n: int = MAX_N) -> dict:
    """Pure exact method on paired counts; callers authenticate real source."""
    if (type(cells) is not tuple or len(cells) != 4
            or any(type(v) is not int or v < 0 for v in cells)
            or type(n) is not int or n != MAX_N or sum(cells) != n):
        raise ExactNIRefused("paired original GSM8K1319 counts invalid")
    both, candidate, reference, neither = cells
    m = candidate + reference
    q_lo, q_hi = discordance_q_upper(n, m)
    p_upper = max(Fraction(0), (q_hi - Fraction(1, 50)) / (2 * q_hi))
    numerator, denominator = exact.binomial_upper_tail_rational(
        m, candidate, p_upper.numerator, p_upper.denominator,
    )
    _require(0 <= numerator <= denominator and denominator > 0,
             "conditional exact tail invalid")
    # Hash the *canonical reduced rational*, not an implementation-dependent
    # binomial denominator that a second exact implementation may simplify.
    canonical_tail = Fraction(numerator, denominator)
    with localcontext() as context:
        context.prec = 48
        p_decimal = str(Decimal(canonical_tail.numerator)
                        / Decimal(canonical_tail.denominator))
    return {
        "schema": SCHEMA,
        "paired_cells": {"both_correct": both, "candidate_only": candidate,
                         "reference_only": reference, "both_incorrect": neither},
        "n": n, "discordant": m,
        "observed_candidate_minus_reference": str(Fraction(candidate - reference, n)),
        "cp_q_lower_dyadic": str(q_lo), "cp_q_upper_dyadic": str(q_hi),
        "cp_dyadic_bits": BITS, "null_conditional_p_upper": str(p_upper),
        "alpha_for_discordance_bound": "1/40",
        "alpha_for_conditional_tail": "1/40",
        "exact_conditional_p_decimal": p_decimal,
        "exact_conditional_p_rational_sha256": hashlib.sha256(
            (str(canonical_tail.numerator) + "/"
             + str(canonical_tail.denominator)).encode("ascii")
        ).hexdigest(),
        "reject_minus_two_percent_composite_null_under_iid":
            ALPHA_PART_DEN * numerator < ALPHA_PART_NUM * denominator,
        "iid_composite_null_type_i_bound": "1/20",
        "fixed_benchmark_not_iid_certificate": True,
        "frozen_amendment4_method_adjudicated": False,
        "same_host_frozen_comparison": False,
        "native_http_lineage_authenticated": False,
        "research_only": True,
        "status": "BLOCKED", "chosen": None, "release_approved": False,
    }


def verify_authentic_crosshost_research() -> dict:
    """Reopen SHA-pinned genuine original files; never adopt synthetic counts."""
    _source_lock()
    source = original.verify_original_research()
    _require(source.get("schema") == original.SCHEMA
             and source.get("population") == MAX_N
             and source.get("canonical_sha256") == original.grade.FROZEN_GSM8K_SHA256
             and source.get("q4_rows_sha256") == original.Q4_ROWS_SHA256
             and source.get("q8_rows_sha256") == original.Q8_ROWS_SHA256
             and source.get("q4_correct") == 1224
             and source.get("q8_correct") == 1220
             and source.get("native_http_lineage_authenticated") is False
             and source.get("same_host_frozen_comparison") is False
             and source.get("research_only") is True
             and source.get("status") == "BLOCKED"
             and source.get("chosen") is None
             and source.get("release_approved") is False,
             "original source-bound research pairing not admissible")
    counts = source.get("paired_counts")
    _require(type(counts) is dict and (
        counts.get("both_correct"), counts.get("windows_q4_only"),
        counts.get("linux_q8_only"), counts.get("neither_correct")
    ) == EXPECTED_COUNTS, "authentic paired research cells changed")
    result = exact_research(EXPECTED_COUNTS)
    result["candidate"] = "original_windows_q4"
    result["reference"] = "original_linux_q8"
    result["original_q4_rows_sha256"] = original.Q4_ROWS_SHA256
    result["original_q8_rows_sha256"] = original.Q8_ROWS_SHA256
    result["original_paired_per_item_sha256"] = source["paired_per_item_sha256"]
    return result


if __name__ == "__main__":
    import json

    print(json.dumps(verify_authentic_crosshost_research(), sort_keys=True, indent=2))
