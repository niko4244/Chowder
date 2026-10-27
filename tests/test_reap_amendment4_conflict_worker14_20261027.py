"""Adversarial synthetic tables: never score measured model predictions."""

import math

import pytest

from chowder_batch.reap_amendment4_conflict_worker14_20261027 import (
    PairedCells, prospective_conflict_audit, original_section4_wald_diagnostic,
    DISPOSITION,
)


@pytest.mark.parametrize("values", [(-1, 0, 0, 165), (True, 0, 0, 163),
                                    (1.0, 0, 0, 163), (0, 0, 0, 1)])
def test_adversarial_invalid_cells(values):
    with pytest.raises(ValueError):
        PairedCells(*values)


def test_sample_standard_deviation_and_direction():
    c = PairedCells(163, 0, 1, 0)
    assert math.isclose(original_section4_wald_diagnostic(c), -0.016128, abs_tol=1e-6)
    reverse = PairedCells(163, 1, 0, 0)
    assert original_section4_wald_diagnostic(reverse) > original_section4_wald_diagnostic(c)


def test_hypothetical_wald_pass_does_not_unblock_mcnemar():
    result = prospective_conflict_audit(PairedCells(163, 0, 1, 0), suite="humaneval")
    assert result["wald_diagnostic_strictly_above_margin"] is True
    assert result["eligible_for_optional_v2_replacement"] is False
    assert result["decision"] == DISPOSITION
    assert result["amendment4_mcnemar_nonzero_margin_ci_defined"] is False


def test_large_cohort_adversarial_disagreement_remains_blocked():
    r = prospective_conflict_audit(PairedCells(1300, 0, 19, 0), suite="gsm8k")
    assert -0.02 < r["frozen_section4_wald_lower_bound_diagnostic"] < 0
    assert r["decision"] == DISPOSITION


def test_zero_discordance_never_implies_method_approval():
    r = prospective_conflict_audit(PairedCells(164, 0, 0, 0), suite="humaneval")
    assert r["frozen_section4_wald_lower_bound_diagnostic"] == 0
    assert r["decision"] == DISPOSITION


def test_wrong_cohort_and_wrong_original_gate_rejected():
    with pytest.raises(ValueError, match="population"):
        prospective_conflict_audit(PairedCells(163, 0, 0, 0), suite="humaneval")
    with pytest.raises(ValueError, match="Gate 1"):
        prospective_conflict_audit(PairedCells(164, 0, 0, 0), suite="humaneval", original_gate1=True)


def test_original_gsm8k_gate1_margin_remains_separate():
    r = prospective_conflict_audit(PairedCells(1318, 0, 1, 0), suite="gsm8k", original_gate1=True)
    assert r["frozen_margin"] == -0.03
    assert r["original_gate1_formula"] is True
    assert r["decision"] == DISPOSITION
