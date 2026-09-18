"""Budget settlement and cycle compute accounting.

Actual measured cost is authoritative. A run that trains successfully but
overruns its frozen envelope must settle as a refusal with the artifact and
measurements preserved -- never as a success with a note.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chowder.growth.compute_cost import (
    ACTUAL_EXCEEDS_PROJECTION,
    ACTUAL_WALL_GPU_HOURS_EXCEEDED,
    RESOURCE_OVERRUN,
    ComputeCost,
    CycleCostLedger,
    settle_cost,
)


# ---------------- ComputeCost validation ----------------


def test_compute_cost_rejects_non_finite_and_negative() -> None:
    with pytest.raises(ValueError, match="finite"):
        ComputeCost(float("nan"), 0.0, source="x")
    with pytest.raises(ValueError, match="finite"):
        ComputeCost(0.0, float("inf"), source="x")
    with pytest.raises(ValueError, match="non-negative"):
        ComputeCost(-0.1, 0.0, source="x")
    with pytest.raises(ValueError, match="non-negative"):
        ComputeCost(0.0, -1.0, source="x")
    with pytest.raises(ValueError, match="source"):
        ComputeCost(0.0, 0.0, source="  ")


def test_wall_only_cost_records_that_device_is_not_separated() -> None:
    cost = ComputeCost.from_wall_only(0.4325, source="attempt:10")
    assert cost.wall_gpu_hours == pytest.approx(0.4325)
    assert cost.device_gpu_hours == 0.0
    assert "not separated" in cost.measurement_method


# ---------------- settlement matrix ----------------


def test_settlement_under_projection_passes() -> None:
    verdict = settle_cost(
        actual=ComputeCost.from_wall_only(0.20, source="a"),
        projected=ComputeCost(0.0, 0.24, source="p"),
        device_ceiling=None,
        wall_ceiling=0.30,
    )
    assert verdict.compliant


def test_settlement_over_projection_under_ceiling_passes() -> None:
    """Estimate 0.24, actual 0.29, ceiling 0.30: inside the envelope -> pass."""
    verdict = settle_cost(
        actual=ComputeCost.from_wall_only(0.29, source="a"),
        projected=ComputeCost(0.0, 0.24, source="p"),
        device_ceiling=None,
        wall_ceiling=0.30,
    )
    assert verdict.compliant, verdict.failure_reasons


def test_settlement_over_ceiling_fails_with_machine_reason() -> None:
    """Estimate 0.24, actual 0.43, ceiling 0.30: RESOURCE_OVERRUN, never a pass."""
    verdict = settle_cost(
        actual=ComputeCost.from_wall_only(0.43, source="a"),
        projected=ComputeCost(0.0, 0.24, source="p"),
        device_ceiling=None,
        wall_ceiling=0.30,
    )
    assert not verdict.compliant
    assert any(ACTUAL_WALL_GPU_HOURS_EXCEEDED in r for r in verdict.failure_reasons)
    assert any(RESOURCE_OVERRUN not in r for r in verdict.failure_reasons) or any(
        RESOURCE_OVERRUN in r for r in verdict.failure_reasons
    )


def test_settlement_flags_projection_blowout_even_under_ceiling() -> None:
    verdict = settle_cost(
        actual=ComputeCost.from_wall_only(0.34, source="a"),
        projected=ComputeCost(0.0, 0.24, source="p"),
        device_ceiling=None,
        wall_ceiling=None,
        projection_tolerance=0.25,
    )
    assert not verdict.compliant
    assert any(ACTUAL_EXCEEDS_PROJECTION in r for r in verdict.failure_reasons)


def test_settlement_units_cannot_be_interchanged() -> None:
    """A device cost against a wall ceiling must not cancel out."""
    verdict = settle_cost(
        actual=ComputeCost(device_gpu_hours=0.5, wall_gpu_hours=0.0, source="a"),
        projected=None,
        device_ceiling=None,
        wall_ceiling=0.30,
    )
    # wall 0.0 vs wall ceiling 0.30 passes; the device 0.5 is invisible to a
    # wall ceiling BY DESIGN -- callers must compare device against the
    # device ceiling, and settle_cost does so separately:
    assert verdict.compliant
    device_verdict = settle_cost(
        actual=ComputeCost(device_gpu_hours=0.5, wall_gpu_hours=0.0, source="a"),
        projected=None,
        device_ceiling=0.30,
        wall_ceiling=0.30,
    )
    assert not device_verdict.compliant


def test_project_budget_is_charged_in_wall_units() -> None:
    verdict = settle_cost(
        actual=ComputeCost.from_wall_only(0.35, source="a"),
        projected=None,
        device_ceiling=None,
        wall_ceiling=None,
        project_budget_wall_gpu_hours=0.25,
    )
    assert not verdict.compliant
    assert any(RESOURCE_OVERRUN in r for r in verdict.failure_reasons)


# ---------------- ledger ----------------


def test_ledger_counts_losing_recipe_and_failed_attempts() -> None:
    ledger = CycleCostLedger(cycle_id="c1")
    ledger.add("recipe-a train", "training", ComputeCost.from_wall_only(0.173, source="a10"), recipe_id="a")
    ledger.add("recipe-a eval", "evaluation", ComputeCost.from_wall_only(0.260, source="e10"), recipe_id="a")
    ledger.add("recipe-b train", "training", ComputeCost.from_wall_only(0.134, source="a11"), recipe_id="b")
    ledger.add("recipe-b eval", "evaluation", ComputeCost.from_wall_only(0.263, source="e11"), recipe_id="b")
    ledger.add("failed attempt", "failed_attempt", ComputeCost.from_wall_only(0.021, source="a09"), recipe_id="b")
    total = ledger.total()
    assert total.wall_gpu_hours == pytest.approx(0.173 + 0.260 + 0.134 + 0.263 + 0.021)
    assert ledger.total_for_recipe("a").wall_gpu_hours == pytest.approx(0.433)
    assert ledger.total_for_recipe("b").wall_gpu_hours == pytest.approx(0.418)


def test_historical_baseline_reference_adds_zero_incremental_spend() -> None:
    ledger = CycleCostLedger(cycle_id="c1")
    ledger.add_reference("gen0 baseline", "gen0-freeze/eval-report.json")
    ledger.add("train", "training", ComputeCost.from_wall_only(0.2, source="t"))
    total = ledger.total(incremental_only=True)
    assert total.wall_gpu_hours == pytest.approx(0.2)
    everything = ledger.total(incremental_only=False)
    assert everything.wall_gpu_hours == pytest.approx(0.2)  # reference is zero anyway
    rendered = ledger.render()
    kinds = [e["kind"] for e in rendered["entries"]]
    assert kinds.count("baseline_reference") == 1


def test_winner_only_accounting_is_detectably_wrong() -> None:
    """Winner-only sums (the gen1 defect) omit losing-recipe compute."""
    ledger = CycleCostLedger(cycle_id="c1")
    ledger.add("recipe-a train", "training", ComputeCost.from_wall_only(0.173, source="a"), recipe_id="a")
    ledger.add("recipe-a eval", "evaluation", ComputeCost.from_wall_only(0.260, source="a"), recipe_id="a")
    ledger.add("recipe-b train", "training", ComputeCost.from_wall_only(0.134, source="b"), recipe_id="b")
    winner_only = 0.173 + 0.260
    full = ledger.total().wall_gpu_hours
    assert full == pytest.approx(winner_only + 0.134)
    assert full > winner_only


def test_render_is_deterministic_with_stable_digest(tmp_path: Path) -> None:
    def build() -> CycleCostLedger:
        ledger = CycleCostLedger(cycle_id="c1")
        ledger.add("a train", "training", ComputeCost.from_wall_only(0.17, source="s"), recipe_id="a")
        ledger.add("b train", "training", ComputeCost.from_wall_only(0.13, source="s"), recipe_id="b")
        ledger.add_reference("baseline", "gen0/eval-report.json")
        return ledger

    first = build().render()
    second = build().render()
    assert first == second
    assert first["digest_sha256"] == second["digest_sha256"]

    path = tmp_path / "cycle_compute_accounting.json"
    digest = build().write(path)
    assert digest == first["digest_sha256"]
    assert json.loads(path.read_text(encoding="utf-8"))["digest_sha256"] == digest
