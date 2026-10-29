"""Research-only *exact* null-size calibration of Amendment 4 paired methods.

For HumanEval's frozen n=164, enumerate every possible discordant paired
table under the noninferiority boundary p10-p01=-0.02. The nuisance parameter
q=p10+p01 is varied prospectively; neither model predictions nor original
evidence are read. This is NOT a gate, an approved CI method, or a replacement
for source-bound canonical paired evaluation.

The score rejection predicate Z(-0.02)>Phi^-1(.95) is equivalent to inverting
the one-sided score lower limit and checking L>-0.02; the equivalence is
independently exhaustively tested over all 13,695 tables.
"""

from __future__ import annotations

from functools import lru_cache
import math
from statistics import NormalDist

from chowder_batch.reap_amendment4_paired_score_research_20261029 import (
    PairedCounts,
    _score_z,
    section4_wald_lower_diagnostic,
)


_N = 164
_NULL_DIFFERENCE = -0.02
_Z95 = NormalDist().inv_cdf(0.95)
_SCHEMA = "chowder.reap-amendment4-exact-null-calibration-research.v1"
_GRID = (0.02, 0.03, 0.05, 0.10, 0.20, 0.50, 0.80, 1.00)


@lru_cache(maxsize=1)
def _decision_table() -> tuple[tuple[int, int, int, bool, bool], ...]:
    """Enumerate all (candidate-only, reference-only, concordant) tables."""
    rows = []
    for b in range(_N + 1):
        for c in range(_N + 1 - b):
            k = _N - b - c
            # The two concordant cells are interchangeable for both methods.
            cells = PairedCounts(k, b, c, 0)
            rows.append((
                b, c, k,
                _score_z(cells, _NULL_DIFFERENCE) > _Z95,
                section4_wald_lower_diagnostic(cells) > _NULL_DIFFERENCE,
            ))
    assert len(rows) == (_N + 1) * (_N + 2) // 2
    return tuple(rows)


def exact_null_calibration(discordance_probability: float) -> dict:
    """Exact finite-sample boundary-null rejection probabilities, not a gate.

    At fixed q, (B,C,K) is multinomial with probabilities
    ((q-0.02)/2, (q+0.02)/2, 1-q). The joint concordant category K can be
    aggregated because both paired tests depend only on B and C. Probabilities
    are evaluated in log space and summed with math.fsum, including q endpoints.
    """
    q = discordance_probability
    if (type(q) not in (int, float) or not math.isfinite(q)
            or not abs(_NULL_DIFFERENCE) <= q <= 1.0):
        raise ValueError("null discordance must be finite in [0.02, 1.0]")
    p10 = (q + _NULL_DIFFERENCE) / 2
    p01 = (q - _NULL_DIFFERENCE) / 2
    p_concordant = 1 - q
    log_factorials = tuple(math.lgamma(i + 1) for i in range(_N + 1))
    normalizer = log_factorials[_N]
    score_masses = []
    wald_masses = []
    all_masses = []
    for b, c, k, score_reject, wald_reject in _decision_table():
        if ((b and p10 == 0) or (c and p01 == 0)
                or (k and p_concordant == 0)):
            continue
        log_probability = (normalizer - log_factorials[b]
                           - log_factorials[c] - log_factorials[k])
        for count, probability in ((b, p10), (c, p01), (k, p_concordant)):
            if count:
                log_probability += count * math.log(probability)
        mass = math.exp(log_probability)
        all_masses.append(mass)
        if score_reject:
            score_masses.append(mass)
        if wald_reject:
            wald_masses.append(mass)

    total = math.fsum(all_masses)
    if not math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-10):
        raise ArithmeticError("multinomial null enumeration lost probability mass")
    return {
        "schema": _SCHEMA,
        "suite": "humaneval",
        "n": _N,
        "null_candidate_minus_reference": _NULL_DIFFERENCE,
        "null_discordance_probability": q,
        "exact_probability_mass": total,
        "score_lower_bound_strictly_above_margin_probability": math.fsum(score_masses),
        "section4_wald_lower_bound_strictly_above_margin_probability": math.fsum(wald_masses),
        "nominal_one_sided_alpha": 0.05,
        "null_boundary_only_not_uniform_size_proof": True,
        "frozen_amendment4_method_adjudicated": False,
        "eligible_for_optional_v2_replacement": False,
        "research_only": True,
        "gen0_status": "BLOCKED",
        "chosen": None,
        "release_approved": False,
    }


def fixed_research_grid() -> tuple[dict, ...]:
    """Prespecified nuisance values; not optimized on any actual outcomes."""
    return tuple(exact_null_calibration(q) for q in _GRID)
