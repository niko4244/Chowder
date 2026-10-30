"""Pure synthetic paired-table audit. Never decides Amendment 4 eligibility."""

from __future__ import annotations

from dataclasses import dataclass
import math

Z_FROZEN = 1.645
ORIGINAL_G1_MARGIN = -0.03
AMENDMENT4_OPTIONAL_MARGIN = -0.02
COHORTS = {"gsm8k": 1319, "humaneval": 164}
DISPOSITION = "BLOCKED_PENDING_INDEPENDENT_PREREG_METHOD_RECONCILIATION"


@dataclass(frozen=True)
class PairedCells:
    both_correct: int
    candidate_only: int
    reference_only: int
    both_incorrect: int

    def __post_init__(self) -> None:
        values = (self.both_correct, self.candidate_only, self.reference_only,
                  self.both_incorrect)
        if any(type(v) is not int or v < 0 for v in values) or sum(values) < 2:
            raise ValueError("paired cells require nonnegative integers and n >= 2")

    @property
    def n(self) -> int:
        return (self.both_correct + self.candidate_only +
                self.reference_only + self.both_incorrect)


def original_section4_wald_diagnostic(cells: PairedCells) -> float:
    """Frozen sample-SD formula, explicitly diagnostic outside original gates."""
    n = cells.n
    difference = (cells.candidate_only - cells.reference_only) / n
    squared_sum = cells.candidate_only + cells.reference_only
    sample_variance = (squared_sum - n * difference * difference) / (n - 1)
    return difference - Z_FROZEN * math.sqrt(max(0.0, sample_variance / n))


def prospective_conflict_audit(
    cells: PairedCells, *, suite: str, original_gate1: bool = False
) -> dict:
    if suite not in COHORTS:
        raise ValueError("suite is outside frozen paired populations")
    if cells.n != COHORTS[suite]:
        raise ValueError("paired sample size differs from frozen canonical population")
    if original_gate1 and suite != "gsm8k":
        raise ValueError("original Gate 1 is GSM8K only")
    bound = original_section4_wald_diagnostic(cells)
    margin = ORIGINAL_G1_MARGIN if original_gate1 else AMENDMENT4_OPTIONAL_MARGIN
    return {
        "suite": suite, "n": cells.n,
        "estimand_candidate_minus_reference":
            (cells.candidate_only - cells.reference_only) / cells.n,
        "frozen_section4_wald_lower_bound_diagnostic": bound,
        "frozen_margin": margin,
        "wald_diagnostic_strictly_above_margin": bound > margin,
        "original_gate1_formula": original_gate1,
        "amendment4_mcnemar_nonzero_margin_ci_defined": False,
        "eligible_for_optional_v2_replacement": False,
        "decision": DISPOSITION,
    }
