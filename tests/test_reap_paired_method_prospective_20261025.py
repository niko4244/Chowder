"""Research-only paired-CI counterexamples; never a release/selection gate.

No datasets, model files, or real result rows are read. A constrained-nuisance
score calculation illustrates one *possible* McNemar-family alternative.
Its use for qualification requires separate prospective statistical approval.
"""

import math

import pytest

from chowder_batch.reap_gates import lower_bound
from chowder_batch.reap_select import MARGIN, PAIRED_METHOD_DISCREPANCY, REQUIRED, select


Z_ONE_SIDED_95 = 1.6448536269514722


def wald_lower_bound(n: int, wins: int, losses: int) -> float:
    """Exact frozen Section 4 implementation, sample stdev and z=1.645."""
    return lower_bound([1.0] * wins + [-1.0] * losses
                       + [0.0] * (n - wins - losses))


def nuisance_mle_discordance(n: int, wins: int, losses: int, delta: float) -> float:
    """Constrained multinomial MLE of q=P(discordant) at fixed paired Δ.

    Cell probabilities (v2-only correct, v1-only correct, concordant) are
    ((q+Δ)/2, (q-Δ)/2, 1-q). Concavity permits deterministic derivative
    bisection. This is a demonstration, NOT an approved CI implementation.
    """
    assert 0 <= wins + losses <= n and -1 < delta < 1
    concordant = n - wins - losses
    boundary = abs(delta)

    def slope(q: float) -> float:
        return (wins / (q + delta) if wins else 0.0) + (
            losses / (q - delta) if losses else 0.0
        ) - (concordant / (1 - q) if concordant else 0.0)

    epsilon = 1e-12
    lo, hi = boundary + epsilon, 1.0 - epsilon
    if slope(lo) <= 0:
        return boundary
    if slope(hi) >= 0:
        return 1.0
    for _ in range(90):
        mid = (lo + hi) / 2
        if slope(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def illustrative_score_z(n: int, wins: int, losses: int, delta: float) -> float:
    """One-sided constrained-nuisance paired score at the *fixed* margin."""
    q = nuisance_mle_discordance(n, wins, losses, delta)
    variance = (q - delta * delta) / n
    assert variance > 0
    return ((wins - losses) / n - delta) / math.sqrt(variance)


@pytest.mark.parametrize(
    ("n", "wins", "losses", "wald_admits", "score_admits"),
    [
        (164, 0, 1, True, False),
        (1319, 0, 19, True, False),
        (164, 0, 0, True, True),
        (164, 20, 21, False, False),
        (164, 5, 0, True, True),
    ],
)
def test_paired_methods_can_disagree_at_frozen_minus_two_percent(
    n, wins, losses, wald_admits, score_admits,
):
    assert (wald_lower_bound(n, wins, losses) > MARGIN) is wald_admits
    assert (illustrative_score_z(n, wins, losses, MARGIN)
            > Z_ONE_SIDED_95) is score_admits


def test_nonzero_noninferiority_margin_is_not_ordinary_exact_mcnemar_null():
    # McNemar's conditional equality test addresses p10=p01 (Δ=0),
    # whereas the frozen NI null boundary is Δ=-0.02.
    assert MARGIN == -0.02
    assert MARGIN != 0.0


def test_optional_v2_replacement_remains_blocked_regardless_of_wald_result():
    passing = {"gates": {key: {"status": "PASS"} for key in REQUIRED}}
    for diagnostic in (-0.019, -0.021, 0.0):
        decision = select({"reap128": passing, "reap128_v2": passing},
                          {"gsm8k": diagnostic, "humaneval": diagnostic})
        assert decision["status"] == "BLOCKED"
        assert decision["chosen"] is None
        assert decision["reason"] == PAIRED_METHOD_DISCREPANCY
