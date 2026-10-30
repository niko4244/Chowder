"""CPU-only independent inversion tests; no Gen-0 authority or model execution."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from chowder_batch import reap_original_exact_ni_lower_bound_research_20261030 as lb


def test_original_gsm8k_conservative_lower_bound_is_between_adjacent_dyadics():
    result = lb.lower_bound(n=1319, candidate_only=25, reference_only=21)
    lower = Fraction(result["conservative_lower_bound"])
    next_null = Fraction(result["adjacent_nonrejected_null"])
    assert Fraction("-0.010162") < lower < Fraction("-0.010161")
    assert next_null - lower == Fraction(1, 1 << lb.GRID_BITS)
    assert result["minus_two_percent_null_rejected_under_iid"] is True
    assert result["composite_null_p_decimal"].startswith("0.025191533192786468")
    assert result["clipped_at_zero"] is False
    assert result["status"] == "BLOCKED"
    assert result["chosen"] is None
    assert result["release_approved"] is False


def test_original_humaneval_conservative_lower_bound_does_not_reject_margin():
    result = lb.lower_bound(n=164, candidate_only=5, reference_only=4)
    lower = Fraction(result["conservative_lower_bound"])
    assert Fraction("-0.058531") < lower < Fraction("-0.058529")
    assert result["minus_two_percent_null_rejected_under_iid"] is False
    assert result["composite_null_p_decimal"].startswith("0.294903122650484542")
    assert result["frozen_q8_vs_dense_gate2_qualified"] is False


@pytest.mark.parametrize("n,b,c", [(164, 5, 4), (1319, 25, 21), (164, 3, 7)])
def test_exact_tail_monotone_and_brackets_rejection(n, b, c):
    result = lb.lower_bound(n=n, candidate_only=b, reference_only=c)
    q = Fraction(result["cp_q_upper"])
    m = b + c

    def reject(delta):
        p = max(Fraction(0), (q + delta) / (2 * q))
        return lb.exact_tail(discordant=m, candidate_only=b,
                             null_success_probability=p) < lb.TEST_ALPHA

    lower = Fraction(result["conservative_lower_bound"])
    next_null = Fraction(result["adjacent_nonrejected_null"])
    assert reject(lower)
    assert not reject(next_null)
    assert not reject(Fraction(0))


def test_no_discordant_success_gives_only_trivial_bound():
    result = lb.lower_bound(n=164, candidate_only=0, reference_only=0)
    assert result["conservative_lower_bound"] == "-1"
    assert result["any_nonpositive_null_rejected"] is False
    assert result["adjacent_nonrejected_null"] is None
    assert result["minus_two_percent_null_rejected_under_iid"] is False
    assert lb.exact_tail(discordant=0, candidate_only=0,
                         null_success_probability=Fraction(0)) >= lb.TEST_ALPHA


def test_extreme_candidate_advantage_is_clipped_at_zero():
    result = lb.lower_bound(n=164, candidate_only=8, reference_only=0)
    assert result["conservative_lower_bound"] == "0"
    assert result["clipped_at_zero"] is True
    assert result["any_nonpositive_null_rejected"] is True
    assert result["adjacent_nonrejected_null"] is None
    assert lb.exact_tail(discordant=8, candidate_only=8,
                         null_success_probability=Fraction(1, 2)) < lb.TEST_ALPHA
    assert result["research_only"] is True


@pytest.mark.parametrize("args", [
    {"n": 165, "candidate_only": 5, "reference_only": 4},
    {"n": True, "candidate_only": 5, "reference_only": 4},
    {"n": 164, "candidate_only": True, "reference_only": 4},
    {"n": 164, "candidate_only": -1, "reference_only": 4},
    {"n": 164, "candidate_only": 5, "reference_only": 160},
])
def test_invalid_populations_refused(args):
    with pytest.raises(lb.LowerBoundRefused):
        lb.lower_bound(**args)


@pytest.mark.parametrize("m,b,p", [
    (-1, 0, Fraction(1, 2)),
    (3, 4, Fraction(1, 2)),
    (3, True, Fraction(1, 2)),
    (3, 1, 0.5),
    (3, 1, Fraction(-1, 2)),
])
def test_invalid_exact_tails_refused(m, b, p):
    with pytest.raises(lb.LowerBoundRefused):
        lb.exact_tail(discordant=m, candidate_only=b, null_success_probability=p)


def test_exact_binomial_sum_known_small_values():
    assert lb.exact_tail(discordant=4, candidate_only=3,
                         null_success_probability=Fraction(1, 2)) == Fraction(5, 16)
    assert lb.exact_tail(discordant=0, candidate_only=0,
                         null_success_probability=Fraction(0)) == 1


def test_upstream_source_drift_refused(monkeypatch):
    monkeypatch.setattr(lb, "WITNESS_SHA256", "0" * 64)
    with pytest.raises(lb.LowerBoundRefused, match="source drift"):
        lb.require_source()


def test_foreign_import_refused(monkeypatch):
    monkeypatch.setattr(lb.witness, "__file__", str(Path("C:/foreign/witness.py")))
    with pytest.raises(lb.LowerBoundRefused, match="source drift"):
        lb.require_source()


def test_certified_original_research_only_and_consistent():
    if not lb.witness.human.PAIR_PATH.is_file():
        pytest.skip("immutable original paired HumanEval certificate unavailable")
    result = lb.verify_authentic_research()
    assert result["status"] == "BLOCKED"
    assert result["chosen"] is None
    assert result["release_approved"] is False
    assert result["gsm8k_crosshost_q4_vs_q8"]["n"] == 1319
    assert result["humaneval_q8_vs_q4"]["n"] == 164
    assert result["gsm8k_crosshost_q4_vs_q8"]["research_only"] is True
