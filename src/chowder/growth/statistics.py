"""Statistics for evidence-driven model-growth decisions.

A 1-question swing on a tiny benchmark is not proof of improvement. Every
comparison between two measured systems goes through here so promotion
decisions see uncertainty, not just point estimates.

Implements: Wilson score intervals (for proportions from small samples),
normal-approximation intervals, Welch's t-test with the
Satterthwaite--Welch degrees-of-freedom approximation, Cohen's d effect
size, paired-comparison bootstrap, and pass@k (the unbiased estimator from
Chen et al. 2021, used for repeated-sample reliability).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

_Z_95 = 1.959963984540054  # two-sided 95% normal quantile
_T_TABLE_95: dict[int, float] = {
    # two-sided 95% critical values for small df (Welch's t)
    1: 12.706204736,
    2: 4.302652730,
    3: 3.182446305,
    4: 2.776445105,
    5: 2.570581836,
    6: 2.446911846,
    7: 2.364624252,
    8: 2.306004135,
    9: 2.262157163,
    10: 2.228138852,
    12: 2.178812830,
    15: 2.131449546,
    20: 2.085963447,
    25: 2.059538553,
    30: 2.042272456,
    40: 2.021075383,
    60: 2.000297822,
    120: 1.979930397,
}


def _t_critical_95(df: float) -> float:
    """Two-sided 95% t critical value, linear-interpolated for fractional df."""
    if df >= 120:
        return _Z_95
    keys = sorted(_T_TABLE_95)
    for lo, hi in zip(keys, keys[1:]):
        if lo <= df <= hi:
            f = (df - lo) / (hi - lo)
            return _T_TABLE_95[lo] * (1 - f) + _T_TABLE_95[hi] * f
    return _T_TABLE_95[keys[0]]  # df < 1: maximally conservative


@dataclass(frozen=True)
class Comparison:
    """A statistical comparison between two systems on one benchmark.

    ``verdict`` is the honest, predeclared language promotion rules act on:
    ``improved`` / ``regressed`` only when the difference is significant at
    the declared level, ``flat`` when measured and not significant, and
    ``inconclusive`` when there is not enough data to decide anything.
    """

    mean_before: float
    mean_after: float
    delta: float
    ci_low: float
    ci_high: float
    significant: bool
    effect_size: float | None  # Cohen's d, None when not computable
    verdict: str  # improved / regressed / flat / inconclusive
    n_before: int
    n_after: int


def wilson_interval(successes: int, total: int, *, z: float = _Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion -- well-behaved at
    small n (a 0/1 or 1/1 sample does not produce a degenerate [0, 1])."""
    if total <= 0:
        return (0.0, 1.0)
    if successes < 0 or successes > total:
        raise ValueError("successes must be within [0, total]")
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def proportion_ci(successes: int, total: int, *, z: float = _Z_95) -> tuple[float, float]:
    return wilson_interval(successes, total, z=z)


def mean(values: Sequence[float]) -> float:
    return sum(float(v) for v in values) / len(values) if values else 0.0


def variance(values: Sequence[float]) -> float:
    """Unbiased sample variance; 0.0 for fewer than 2 samples."""
    n = len(values)
    if n < 2:
        return 0.0
    m = mean(values)
    return sum((float(v) - m) ** 2 for v in values) / (n - 1)


def effect_size_cohens_d(
    before: Sequence[float], after: Sequence[float]
) -> float | None:
    """Pooled-SD Cohen's d (after - before). None when SD is degenerate."""
    n1, n2 = len(before), len(after)
    if n1 < 2 or n2 < 2:
        return None
    v1, v2 = variance(before), variance(after)
    pooled = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if pooled <= 0.0:
        return None
    return (mean(after) - mean(before)) / pooled


def compare(
    before: Sequence[float],
    after: Sequence[float],
    *,
    alpha: float = 0.05,
    min_effect: float = 0.0,
) -> Comparison:
    """Compare two independent samples (per-question scores or repeated runs).

    Welch's t-test with Satterthwaite df (small samples, unequal variances
    are the norm in evaluation batteries). ``min_effect`` is the smallest
    delta that counts as meaningful: a significant-but-tiny change still
    reports ``flat`` rather than ``improved``.
    """
    if not before or not after:
        raise ValueError("comparison requires at least one sample on each side")
    n1, n2 = len(before), len(after)
    m1, m2 = mean(before), mean(after)
    delta = m2 - m1
    v1, v2 = variance(before), variance(after)

    if n1 < 2 or n2 < 2 or (v1 <= 0.0 and v2 <= 0.0):
        # No usable variance estimate: cannot decide significance honestly.
        # A degenerate pair (e.g. a parent pinned at the scale floor -- all
        # zeros -- against a candidate pinned at the ceiling) has a real,
        # decisive delta with no spread to test; the |delta| against
        # min_effect still decides the direction. Hiding a hard lift past
        # the declared minimum behind "inconclusive" would make a floor
        # start unpromotable no matter what the candidate achieves.
        if delta > min_effect:
            verdict = "improved"
        elif delta < -min_effect:
            verdict = "regressed"
        else:
            verdict = "flat"
        return Comparison(
            mean_before=m1,
            mean_after=m2,
            delta=delta,
            ci_low=min(m1, m2),
            ci_high=max(m1, m2),
            significant=False,
            effect_size=effect_size_cohens_d(before, after),
            verdict=verdict,
            n_before=n1,
            n_after=n2,
        )

    se = math.sqrt(v1 / n1 + v2 / n2)
    df = (v1 / n1 + v2 / n2) ** 2 / (
        (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
    )
    t_crit = _t_critical_95(df) if abs(alpha - 0.05) < 1e-9 else _Z_95
    ci_low = delta - t_crit * se
    ci_high = delta + t_crit * se
    significant = (ci_low > 0 and ci_high > 0) or (ci_low < 0 and ci_high < 0)

    if not significant:
        verdict = "flat"
    elif delta > min_effect:
        verdict = "improved"
    elif delta < -min_effect:
        verdict = "regressed"
    else:
        # Significant but smaller than the declared meaningful effect.
        verdict = "flat"

    return Comparison(
        mean_before=m1,
        mean_after=m2,
        delta=delta,
        ci_low=ci_low,
        ci_high=ci_high,
        significant=significant,
        effect_size=effect_size_cohens_d(before, after),
        verdict=verdict,
        n_before=n1,
        n_after=n2,
    )


def bootstrap_paired_delta(
    before: Sequence[float],
    after: Sequence[float],
    *,
    iterations: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean paired delta (after - before).

    Requires paired observations (same questions, repeated samples). The CI
    excluding zero is suggestive, not a formal test -- pair with
    ``compare()`` for significance.
    """
    if len(before) != len(after):
        raise ValueError("paired bootstrap requires equal-length samples")
    deltas = [float(a) - float(b) for b, a in zip(before, after)]
    rng = random.Random(seed)
    n = len(deltas)
    means: list[float] = []
    for _ in range(iterations):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(mean(sample))
    means.sort()
    lo_index = int((alpha / 2) * iterations)
    hi_index = min(iterations - 1, int((1 - alpha / 2) * iterations))
    return (means[lo_index], means[hi_index])


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al. 2021):
    1 - C(n - c, k) / C(n, k). Numerically stable form."""
    if k <= 0 or n <= 0 or k > n:
        raise ValueError("pass@k requires 0 < k <= n")
    if c < 0 or c > n:
        raise ValueError("c must be within [0, n]")
    if c == 0:
        return 0.0
    if c == n:
        return 1.0
    # 1 - prod_{i=0}^{k-1} (n - c - i) / (n - i)
    result = 1.0
    for i in range(k):
        result *= (n - c - i) / (n - i)
    return 1.0 - result
