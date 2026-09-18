"""Promotion-level settlement: a frozen budget can veto after execution.

The two integrity rules pinned here:
1. An actual cost overrun of the preregistered ceiling vetoes promotion
   mechanically -- target improvement cannot buy back a blown envelope.
2. A real target repair with incomplete full-promotion evidence is exactly
   INCONCLUSIVE plus ``target_repair_validated`` -- not a new verdict class,
   not a silent PROMOTED.
"""

from __future__ import annotations

from chowder.evals.result import MEASURED_PARENT, MEASURED_THIS_GENERATION
from chowder.growth.promotion import BenchmarkResult, PromotionInput, evaluate_promotion


def _result(qid: str, score: float, samples: tuple[float, ...], *, origin: str) -> BenchmarkResult:
    return BenchmarkResult(
        benchmark_qualified_id=qid,
        score=score,
        samples=samples,
        contamination="CLEAN",
        measurement_origin=origin,
    )


TARGET = "diagnostics@v1"
PROTECTED = "math500@2024-04"


def _input(*, actual_wall: float | None, wall_ceiling: float | None) -> PromotionInput:
    return PromotionInput(
        candidate_version="gen1",
        parent_version="gen0",
        target_benchmarks=(TARGET,),
        candidate_results={
            TARGET: _result(TARGET, 0.9, (0.9, 0.95, 0.85, 0.9, 0.9, 0.9), origin=MEASURED_THIS_GENERATION),
            PROTECTED: _result(PROTECTED, 0.0, (0.0,) * 24, origin=MEASURED_THIS_GENERATION),
        },
        parent_results={
            TARGET: _result(TARGET, 0.0, (0.0,) * 6, origin=MEASURED_PARENT),
            PROTECTED: _result(PROTECTED, 0.0, (0.0,) * 24, origin=MEASURED_PARENT),
        },
        protected_benchmarks=(PROTECTED,),
        broad_battery_benchmarks=(PROTECTED,),
        actual_wall_gpu_hours=actual_wall,
        wall_gpu_hours_ceiling=wall_ceiling,
    )


def test_budget_compliant_improvement_promotes() -> None:
    decision = evaluate_promotion(_input(actual_wall=0.70, wall_ceiling=1.00))
    assert decision.checks["resource_envelope"] == "ok"
    assert decision.verdict == "PROMOTED", decision.reasons


def test_actual_cost_overrun_vetoes_promotion() -> None:
    """Train succeeded, target improved -- the blown ceiling still vetoes."""
    decision = evaluate_promotion(_input(actual_wall=1.30, wall_ceiling=1.00))
    assert decision.checks["resource_envelope"] == "violated"
    assert decision.verdict == "REJECTED"
    assert any("1.30" in r or "wall" in r for r in decision.reasons)


def test_missing_settlement_numbers_do_not_block_but_are_visible() -> None:
    decision = evaluate_promotion(_input(actual_wall=None, wall_ceiling=None))
    assert decision.checks["resource_envelope"] == "unmeasured"


def test_real_target_repair_with_incomplete_protected_evidence_is_inconclusive_not_promoted() -> None:
    """The gen1 re-adjudication shape: real candidate-measured target lift,
    but a protected benchmark never measured on the candidate."""
    data = PromotionInput(
        candidate_version="gen1",
        parent_version="gen0",
        target_benchmarks=(TARGET,),
        candidate_results={
            TARGET: _result(TARGET, 1.0, (1.0,) * 16, origin=MEASURED_THIS_GENERATION),
        },
        parent_results={
            TARGET: _result(TARGET, 0.0, (0.0,) * 16, origin=MEASURED_PARENT),
        },
        protected_benchmarks=(PROTECTED,),
        broad_battery_benchmarks=(PROTECTED,),
        min_target_improvement=0.90,
    )
    decision = evaluate_promotion(data)
    assert decision.checks["target:diagnostics@v1"] == "improved"
    assert decision.checks["target_improvement"] == "met"
    assert decision.checks["protected:math500@2024-04"] == "inconclusive"
    assert decision.verdict == "INCONCLUSIVE"
    # The scoped-repair design: the decision payload itself records that the
    # target repair is real, so a downstream reader never has to re-derive it.
    assert decision.target_repair_validated is True
    assert decision.target_repair_metrics["diagnostics@v1"] == 1.0
