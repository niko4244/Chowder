"""Adversarial CPU-only checks for prospective composite-null p-value accounting."""

from __future__ import annotations

from fractions import Fraction
import hashlib
import math
from pathlib import Path

import pytest

from chowder_batch import reap_original_exact_ni_composite_p_witness_20261030 as witness
from chowder_batch import reap_original_gsm8k_crosshost_exact_ni_20261030 as gsm
from chowder_batch import reap_original_human164_crossquant_exact_ni_20261030 as human


def _original_research(cells: tuple[int, int, int, int]) -> dict:
    n = sum(cells)
    if n == 1319:
        return gsm.exact_research(cells)
    assert n == 164
    return human.exact_research(cells)


@pytest.mark.parametrize("cells", [
    (1199, 25, 21, 74),
    (147, 5, 4, 8),
    (164, 0, 0, 0),
    (0, 164, 0, 0),
    (0, 0, 164, 0),
    (1319, 0, 0, 0),
    (0, 1319, 0, 0),
    (0, 0, 1319, 0),
])
def test_independent_exact_tail_and_split_alpha_agree(cells):
    original = _original_research(cells)
    n = sum(cells)
    conditional, adjusted = witness.independently_recount_tail(
        n=n, candidate_only=cells[1], reference_only=cells[2])
    assert Fraction(0) <= conditional <= 1
    assert adjusted == min(Fraction(1), Fraction(1, 40) + conditional)
    assert Fraction(1, 40) <= adjusted <= 1
    assert hashlib.sha256(
        f"{conditional.numerator}/{conditional.denominator}".encode()
    ).hexdigest() == original["exact_conditional_p_rational_sha256"]
    assert (conditional < Fraction(1, 40)) == (adjusted < Fraction(1, 20))


def test_gsm8k_conditional_tail_not_unconditional_p_value():
    original = gsm.exact_research((1199, 25, 21, 74))
    result = witness._witness(
        arm="original_crosshost_gsm8k_q4_vs_q8",
        counts=(1199, 25, 21, 74), original=original)
    assert result["conditional_tail_decimal"].startswith("0.000191533192786468")
    assert result["conservative_composite_null_p_decimal"].startswith(
        "0.025191533192786468")
    assert result["reject_composite_null_at_five_percent_under_iid"] is True
    assert result["iid_fixed_benchmark_not_established"] is True
    assert result["status"] == "BLOCKED"
    assert result["chosen"] is None
    assert result["release_approved"] is False


def test_humaneval_conditional_tail_not_unconditional_p_value():
    original = human.exact_research((147, 5, 4, 8))
    result = witness._witness(
        arm="original_research_humaneval_q8_vs_q4",
        counts=(147, 5, 4, 8), original=original)
    assert result["conditional_tail_decimal"].startswith("0.269903122650484542")
    assert result["conservative_composite_null_p_decimal"].startswith(
        "0.294903122650484542")
    assert result["reject_composite_null_at_five_percent_under_iid"] is False
    assert result["frozen_q8_vs_dense_gate2_qualified"] is False


@pytest.mark.parametrize("n,candidate,reference", [
    (0, 0, 0), (163, 1, 0), (1320, 1, 0),
    (164, True, 0), (164, -1, 0), (164, 1, -1),
    (164, 165, 0), (1319, 1320, 0),
])
def test_invalid_paired_population_refused(n, candidate, reference):
    with pytest.raises(witness.CompositePRefused):
        witness.independently_recount_tail(
            n=n, candidate_only=candidate, reference_only=reference)


def test_forged_original_conditional_tail_refused():
    original = gsm.exact_research((1199, 25, 21, 74))
    original["exact_conditional_p_rational_sha256"] = "0" * 64
    with pytest.raises(witness.CompositePRefused, match="disagrees"):
        witness._witness(
            arm="research", counts=(1199, 25, 21, 74), original=original)


@pytest.mark.parametrize("field, forged", [
    ("status", "PASS"), ("chosen", "q4"),
    ("release_approved", True), ("research_only", False),
])
def test_forged_original_admission_refused(field, forged):
    original = human.exact_research((147, 5, 4, 8))
    original[field] = forged
    with pytest.raises(witness.CompositePRefused, match="disagrees"):
        witness._witness(
            arm="research", counts=(147, 5, 4, 8), original=original)


def test_source_hash_drift_refused(monkeypatch):
    monkeypatch.setitem(witness.SOURCE_SHA256,
                        "reap_original_gsm8k_crosshost_exact_ni_20261030.py",
                        "0" * 64)
    with pytest.raises(witness.CompositePRefused, match="changed"):
        witness.require_source()


def test_import_outside_authorized_worktree_refused(monkeypatch):
    monkeypatch.setattr(witness.gsm, "__file__", str(Path("C:/other/scorer.py")))
    with pytest.raises(witness.CompositePRefused, match="outside"):
        witness.require_source()


def test_authentic_certificates_research_only():
    if not human.PAIR_PATH.is_file():
        pytest.skip("immutable F: original research certificate unavailable")
    result = witness.verify_original_research()
    assert result["status"] == "BLOCKED"
    assert result["chosen"] is None
    assert result["release_approved"] is False
    assert result["gsm8k_crosshost_q4_vs_q8"]["candidate_only"] == 25
    assert result["humaneval_q8_vs_q4"]["candidate_only"] == 5
    assert all(result[name]["research_only"] is True for name in (
        "gsm8k_crosshost_q4_vs_q8", "humaneval_q8_vs_q4"))


def test_independent_tail_against_binomial_coefficient_at_small_m():
    n, b, c = 164, 3, 4
    q = gsm.discordance_q_upper(n, b + c)[1]
    p = max(Fraction(0), (q - Fraction(1, 50)) / (2 * q))
    expected = sum((Fraction(math.comb(b + c, j)) * p**j
                    * (1 - p)**(b + c - j)
                    for j in range(b, b + c + 1)), Fraction(0))
    actual, adjusted = witness.independently_recount_tail(
        n=n, candidate_only=b, reference_only=c)
    assert actual == expected
    assert adjusted == Fraction(1, 40) + expected
