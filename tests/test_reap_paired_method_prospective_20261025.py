"""Research-only paired-CI counterexamples; never a release/selection gate.

No datasets, model files, or real result rows are read. A constrained-nuisance
score calculation illustrates one *possible* McNemar-family alternative.
Its use for qualification requires separate prospective statistical approval.
"""

import math
from pathlib import Path

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


def test_original_primary_prereg_contains_both_incompatible_method_descriptions():
    primary = (Path(__file__).resolve().parents[1] / "docs" /
               "REAP_QWEN36_PREREG.md").read_text(encoding="utf-8")
    original_gates, amendment = primary.split("## Amendment 4 ", 1)
    assert "LB = mean(d) − 1.645 · sd(d) / √n" in original_gates
    assert "LB(v2 − v1) > −0.02" in amendment
    assert "paired McNemar, one-sided 95%" in amendment


def test_identical_mcnemar_discordants_do_not_identify_risk_difference_ci():
    # The ordinary conditional exact McNemar equality test knows only b,c.
    # Same b=0,c=2 in both populations => identical equality-test p-value;
    # but the marginal difference (b-c)/n and its Section 4 lower bound
    # change with the full number of canonical matched items.
    wins, losses = 0, 2
    discordant = wins + losses
    exact_one_sided_equality_p = sum(
        math.comb(discordant, k) for k in range(wins, discordant + 1)
    ) / 2**discordant
    assert exact_one_sided_equality_p == 1.0
    assert (wins - losses) / 164 == pytest.approx(-0.012195121951219513)
    assert (wins - losses) / 1319 == pytest.approx(-0.001516300227445034)
    assert wald_lower_bound(164, wins, losses) < MARGIN
    assert wald_lower_bound(1319, wins, losses) > MARGIN


def test_optional_v2_replacement_remains_blocked_regardless_of_wald_result():
    passing = {"gates": {key: {"status": "PASS"} for key in REQUIRED}}
    for diagnostic in (-0.019, -0.021, 0.0):
        decision = select({"reap128": passing, "reap128_v2": passing},
                          {"gsm8k": diagnostic, "humaneval": diagnostic})
        assert decision["status"] == "BLOCKED"
        assert decision["chosen"] is None
        assert decision["reason"] == PAIRED_METHOD_DISCREPANCY
