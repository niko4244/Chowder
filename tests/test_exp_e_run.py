"""Driver wiring for the Phase-4 quantized routing lane (``exp_e_run``).

The lane is gated by two independently measured quantities: the margin shift
and the fraction of the reference arm's greens the quantized arm loses. These
tests pin the contract the driver relies on: supplying either measurement
alone leaves the lane closed for every query, only both together (with a
validated held-out gate) can route, and a measured Experiment F report is the
sanctioned source for those measurements -- the driver derives them from
per-task data instead of accepting hand-typed scalars.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch", reason="the exp_e runner imports torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "chowder_batch"))
sys.path.insert(0, str(ROOT / "src"))

import exp_e_run  # noqa: E402
from exp_e_confidence import (  # noqa: E402
    HELDOUT_GATE_STATUS,
    calibrate_margin_threshold,
    heldout_transfer_gate,
    quantized_arm_admission,
)
from exp_e_run import quantized_guard_state, quantized_route_decision  # noqa: E402
from exp_f_ptq_margin import load_quantized_evidence  # noqa: E402

MEASURED_REPORT = ROOT / "evidence" / "exp_f_ptq_margin_qwen25_1p5b_int8sq_guided20_20260926.json"


def _dev_rows():
    return [
        {"task": f"dev_{index}", "margin": margin, "correct": correct}
        for index, (margin, correct) in enumerate([
            (0.1, False), (0.2, True), (0.3, True), (0.4, True), (0.5, False), (0.6, True),
        ])
    ]


def _heldout_rows():
    return [
        {"task": f"held_{index}", "margin": margin, "correct": True}
        for index, margin in enumerate([0.25, 0.35, 0.45, 0.55, 0.65, 0.75])
    ]


def _state(**overrides):
    inputs = {
        "quantized_margin_shift": None,
        "max_quantized_margin_shift": 0.2,
        "quantized_green_loss_fraction": None,
        "max_quantized_green_loss_fraction": 0.0,
    }
    inputs.update(overrides)
    return quantized_guard_state(**inputs)


def _serving_calibration(state):
    calibration = calibrate_margin_threshold(_dev_rows(), min_precision=0.8, min_samples=4)
    calibration["precision_arm"] = "int8_smoothquant"
    calibration.update(state)
    return calibration


def _route(state, *, margin=100.0, arm_admission=None):
    """Mirror what main() does for a query on a serving arm."""
    calibration = _serving_calibration(state)
    gate = heldout_transfer_gate(calibration, _heldout_rows(), min_precision=0.8, min_samples=4)
    assert gate["status"] == HELDOUT_GATE_STATUS
    if arm_admission is None:
        arm_admission = quantized_arm_admission(
            calibration,
            quantized_margin_shift=state["quantized_margin_shift"],
            max_quantized_margin_shift=calibration["max_quantized_margin_shift"],
            green_loss_fraction=state["quantized_green_loss_fraction"],
            max_green_loss_fraction=calibration["max_quantized_green_loss_fraction"],
            heldout_gate=gate,
        )
    return quantized_route_decision(
        margin,
        calibration,
        guard_state=state,
        arm_admission=arm_admission,
        heldout_gate=gate,
    )


def test_one_measurement_alone_leaves_the_quantized_lane_closed():
    # Shift measured, green retention unmeasured: the active lane blocks.
    shift_only = _state(quantized_margin_shift=0.05)
    assert shift_only["quantized_guard_active"] is True
    assert shift_only["quantized_margin_shift_fails_closed"] is False
    assert shift_only["quantized_green_loss_fails_closed"] is True
    assert _route(shift_only) is False

    # Green retention measured, shift unmeasured: the shift guard blocks.
    green_only = _state(quantized_green_loss_fraction=0.0)
    assert green_only["quantized_guard_active"] is True
    assert green_only["quantized_margin_shift_fails_closed"] is True
    assert green_only["quantized_green_loss_fails_closed"] is False
    assert _route(green_only) is False

    # Both measured: a high margin routes small, but only above the threshold.
    both = _state(quantized_margin_shift=0.05, quantized_green_loss_fraction=0.0)
    assert _route(both) is True
    assert _route(both, margin=0.0) is False

    # Neither: the pure-BF16 lane is untouched (no quantized guard is active).
    bf16 = _state()
    assert bf16["quantized_guard_active"] is False
    assert bf16["quantized_margin_shift_fails_closed"] is False
    assert bf16["quantized_green_loss_fails_closed"] is False
    assert _route(bf16, arm_admission={}) is True


def test_routing_follows_the_aggregate_admission_on_every_guard_combination():
    for overrides in (
        {"quantized_margin_shift": 0.05, "quantized_green_loss_fraction": 0.0},
        {"quantized_margin_shift": 0.05, "quantized_green_loss_fraction": 1.0},
        {"quantized_margin_shift": 0.5, "quantized_green_loss_fraction": 0.0},
        {"quantized_margin_shift": None, "quantized_green_loss_fraction": 0.0},
        {"quantized_margin_shift": 0.05},
    ):
        state = _state(**overrides)
        calibration = _serving_calibration(state)
        gate = heldout_transfer_gate(calibration, _heldout_rows(), min_precision=0.8, min_samples=4)
        assert gate["status"] == HELDOUT_GATE_STATUS
        admission = quantized_arm_admission(
            calibration,
            quantized_margin_shift=state["quantized_margin_shift"],
            max_quantized_margin_shift=calibration["max_quantized_margin_shift"],
            green_loss_fraction=state["quantized_green_loss_fraction"],
            max_green_loss_fraction=calibration["max_quantized_green_loss_fraction"],
            heldout_gate=gate,
        )
        route = quantized_route_decision(
            100.0, calibration, guard_state=state, arm_admission=admission, heldout_gate=gate
        )
        assert route is admission["admitted"]
        if not admission["admitted"]:
            assert admission["refusals"]  # every refusal names its guard


def _synthetic_report() -> dict:
    """A paired report shaped like the live one: margins rise with index, the
    held-out half is a genuine transfer test, and the quantized arm loses one
    reference green (task_6) while keeping every other."""

    def bf16_correct(index: int) -> bool:
        return index >= 4

    def quant_correct(index: int) -> bool:
        return index >= 4 and index != 6

    per_task = [
        {
            "task": f"task_{index}",
            "bf16_margin": 1.0 + index * 0.1,
            "quant_margin": 1.0 + index * 0.1 - 0.1,
            "margin_shift": -0.1,
            "bf16_correct": bf16_correct(index),
            "quant_correct": quant_correct(index),
        }
        for index in range(16)
    ]
    return {
        "model_path": "synthetic/model",
        "ptq_config": "int8_weight_only",
        "margin_comparison": {
            "n_tasks": 16,
            "bf16_mean_margin": sum(row["bf16_margin"] for row in per_task) / 16,
            "quant_mean_margin": sum(row["quant_margin"] for row in per_task) / 16,
            "per_task": per_task,
        },
    }


def test_evidence_file_derives_the_measurements_and_refuses_conflicts(tmp_path, monkeypatch):
    evidence_path = tmp_path / "exp_f_report.json"
    evidence_path.write_text(json.dumps(_synthetic_report()), encoding="utf-8")

    evidence = load_quantized_evidence(evidence_path)
    assert evidence["precision_arm"] == "int8_weight_only"
    assert evidence["margin_shift"] == pytest.approx(-0.1)
    assert evidence["green_loss_fraction"] == pytest.approx(1 / 12)
    # The shift passes while the lost green refuses admission -- on its own.
    assert evidence["admission"]["guards"]["margin_shift"]["passed"] is True
    assert evidence["admission"]["admitted"] is False
    assert [refusal["guard"] for refusal in evidence["admission"]["refusals"]] == ["green_retention"]
    assert evidence["admission_verification"]["valid"] is True
    # The report's held-out half travels with the evidence.
    heldout_ids = set(evidence["report_heldout_gate"]["heldout_task_ids"])
    assert {row["task"] for row in evidence["heldout_rows"]} == heldout_ids
    assert evidence["report_heldout_gate"]["status"] == HELDOUT_GATE_STATUS

    # A caller that declares the loss acceptable gets a measured pass instead.
    tolerant = load_quantized_evidence(evidence_path, max_green_loss_fraction=0.1)
    assert tolerant["admission"]["admitted"] is True

    # --quantized-evidence refuses to mix sources: derive or declare, never both.
    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text(json.dumps({"tasks": []}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "exp_e_run.py", "--tasks", str(tasks_path), "--out", str(tmp_path / "out.json"),
        "--quantized-evidence", str(evidence_path), "--quantized-margin-shift", "0.1",
    ])
    with pytest.raises(SystemExit):
        exp_e_run.main()

    # A file that is not an Experiment F report is refused, never guessed at.
    not_a_report = tmp_path / "not_a_report.json"
    not_a_report.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="report object"):
        load_quantized_evidence(not_a_report)
    missing_arm = tmp_path / "missing_arm.json"
    missing_arm.write_text(json.dumps({"margin_comparison": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="ptq_config"):
        load_quantized_evidence(missing_arm)


@pytest.mark.skipif(not MEASURED_REPORT.exists(), reason="measured n=20 report is not in this checkout")
def test_measured_report_evidence_refuses_on_retention_with_the_shift_passing():
    evidence = load_quantized_evidence(MEASURED_REPORT)
    assert evidence["precision_arm"] == "int8_smoothquant"
    assert evidence["margin_shift"] == pytest.approx(0.007589349319392369)
    assert evidence["green_loss_fraction"] == 1.0
    assert evidence["green_retention"]["reference_greens"] == 6
    assert evidence["green_retention"]["retained_greens"] == 0
    assert evidence["admission"]["admitted"] is False
    # The bound that used to be the only quantized guard: it passed here.
    assert evidence["admission"]["guards"]["margin_shift"]["passed"] is True
    assert {refusal["guard"] for refusal in evidence["admission"]["refusals"]} == {
        "calibration", "heldout_transfer_gate", "green_retention"
    }
    assert evidence["admission_verification"]["valid"] is True
    assert len(evidence["heldout_rows"]) == 10
