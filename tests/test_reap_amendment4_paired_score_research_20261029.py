"""Synthetic paired-table tests; never substitute for original scored arms."""

import math

import pytest

from chowder_batch.reap_amendment4_paired_score_research_20261029 import (
    PairedCounts, _constrained_discordance, _score_z,
    exact_conditional_mcnemar_equality_p, research_comparison,
    section4_wald_lower_diagnostic, tango_score_lower_diagnostic,
    tango_score_upper_diagnostic,
)


@pytest.mark.parametrize("counts", [(-1, 0, 0, 165), (True, 0, 0, 163),
                                    (1.0, 0, 0, 163), (0, 0, 0, 1)])
def test_invalid_cells_fail_closed(counts):
    with pytest.raises(ValueError):
        PairedCounts(*counts)


@pytest.mark.parametrize("counts,d", [((147, 5, 4, 8), -0.02),
                                     ((163, 0, 1, 0), -0.01),
                                     ((164, 0, 0, 0), -0.01),
                                     ((0, 82, 82, 0), 0.2),
                                     ((0, 164, 0, 0), 0.5)])
def test_constrained_mle_maximizes_actual_multinomial_loglikelihood(counts, d):
    cells = PairedCounts(*counts)
    q = _constrained_discordance(cells, d)

    def likelihood(x):
        p10, p01, concordant = (x + d) / 2, (x - d) / 2, 1 - x
        if (cells.candidate_only and p10 <= 0
                or cells.reference_only and p01 <= 0
                or (cells.both_correct + cells.both_incorrect) and concordant <= 0):
            return -math.inf
        return sum(count * math.log(prob) for count, prob in (
            (cells.candidate_only, p10), (cells.reference_only, p01),
            (cells.both_correct + cells.both_incorrect, concordant),
        ) if count)

    assert abs(d) <= q <= 1
    best = likelihood(q)
    for i in range(1001):
        trial = abs(d) + (1 - abs(d)) * i / 1000
        assert best >= likelihood(trial) - 1e-8


def test_zero_discordance_score_interval_is_not_zero_width_wald():
    cells = PairedCounts(164, 0, 0, 0)
    z = 1.6448536269514722
    expected = -z * z / (164 + z * z)
    assert section4_wald_lower_diagnostic(cells) == 0
    assert tango_score_lower_diagnostic(cells) == pytest.approx(expected, abs=1e-12)
    assert exact_conditional_mcnemar_equality_p(cells) == 1


@pytest.mark.parametrize("cells", [PairedCounts(147, 5, 4, 8),
                                    PairedCounts(163, 0, 1, 0),
                                    PairedCounts(164, 0, 0, 0),
                                    PairedCounts(0, 82, 82, 0),
                                    PairedCounts(0, 164, 0, 0)])
def test_inverted_score_root_and_symmetry(cells):
    lower = tango_score_lower_diagnostic(cells)
    upper = tango_score_upper_diagnostic(cells)
    assert -1 <= lower <= cells.difference <= upper <= 1
    assert lower == pytest.approx(-tango_score_upper_diagnostic(cells.reversed()))
    if lower > -1:
        assert _score_z(cells, lower) == pytest.approx(1.6448536269514722, abs=1e-7)


def test_mcnemar_equality_is_not_noninferiority_at_nonzero_margin():
    assert exact_conditional_mcnemar_equality_p(PairedCounts(159, 0, 5, 0)) == 0.0625
    cells = PairedCounts(163, 0, 1, 0)
    result = research_comparison(cells, suite="humaneval")
    assert result["exact_conditional_mcnemar_equality_p_not_NI"] == 1.0
    assert result["frozen_amendment4_method_adjudicated"] is False
    assert result["eligible_for_optional_v2_replacement"] is False
    assert result["gen0_status"] == "BLOCKED"
    assert result["release_approved"] is False


@pytest.mark.parametrize("suite,cells", [
    ("humaneval", PairedCounts(163, 0, 1, 0)),
    ("gsm8k", PairedCounts(1300, 0, 19, 0)),
])
def test_original_wald_and_paired_score_disagree_on_frozen_margin(suite, cells):
    """Synthetic counterexamples establish the ambiguity is decision-relevant."""
    result = research_comparison(cells, suite=suite)
    assert result["section4_wald_one_sided_95_lower_diagnostic"] > -0.02
    assert result["tango_paired_score_one_sided_95_lower_diagnostic"] < -0.02
    assert result["diagnostic_methods_disagree_on_margin"] is True
    assert result["eligible_for_optional_v2_replacement"] is False


def test_research_uses_only_registered_populations():
    with pytest.raises(ValueError, match="population"):
        research_comparison(PairedCounts(163, 0, 0, 0), suite="humaneval")
    with pytest.raises(ValueError, match="population"):
        research_comparison(PairedCounts(164, 0, 0, 0), suite="unknown")


def test_mcnemar_exact_tail_uses_discordant_cells_only():
    a = PairedCounts(100, 4, 1, 59)
    b = PairedCounts(0, 4, 1, 159)
    assert exact_conditional_mcnemar_equality_p(a) == 0.375
    assert exact_conditional_mcnemar_equality_p(b) == 0.375
    assert exact_conditional_mcnemar_equality_p(a.reversed()) == 0.375
