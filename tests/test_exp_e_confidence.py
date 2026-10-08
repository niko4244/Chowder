from __future__ import annotations

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chowder_batch"))

from exp_e_confidence import (
    ARM_ADMISSION_VERSION,
    HELDOUT_GATE_STATUS,
    aggregate_margin,
    calibrate_margin_threshold,
    calibrate_margin_threshold_per_precision,
    chosen_token_margins,
    green_loss_fails_closed,
    heldout_transfer_gate,
    heldout_transfer_gate_allows,
    margin_shift_fails_closed,
    quant_route_allowed,
    quantized_arm_admission,
    small_route_allowed,
    validate_quantized_green_retention,
    validate_quantized_margin_shift,
    verify_arm_admission,
)


def _response() -> dict:
    return {
        "choices": [{
            "logprobs": {
                "content": [
                    {"token": "A", "logprob": -0.1, "top_logprobs": [
                        {"token": "A", "logprob": -0.1}, {"token": "B", "logprob": -0.8},
                    ]},
                    {"token": "C", "logprob": -0.05, "top_logprobs": [
                        {"token": "C", "logprob": -0.05}, {"token": "D", "logprob": -1.0},
                    ]},
                ]
            }
        }]
    }


def test_selected_token_margin_uses_best_distinct_alternative():
    assert chosen_token_margins(_response()) == pytest.approx([0.7, 0.95])
    assert aggregate_margin(_response()) == pytest.approx(0.825)


def test_missing_or_non_alternative_logprobs_are_unavailable_not_heuristic():
    assert chosen_token_margins({"choices": [{"message": {"content": "answer"}}]}) is None
    only_self = {"choices": [{"logprobs": {"content": [
        {"token": "A", "logprob": -0.1, "top_logprobs": [{"token": "A", "logprob": -0.1}]}
    ]}}]}
    assert aggregate_margin(only_self) is None
    assert aggregate_margin({"choices": [{"logprobs": {"content": []}}]}) is None


def test_dev_calibration_selects_highest_coverage_threshold_meeting_precision():
    dev_rows = [
        {"margin": 0.1, "correct": False},
        {"margin": 0.2, "correct": True},
        {"margin": 0.3, "correct": True},
        {"margin": 0.4, "correct": True},
        {"margin": 0.5, "correct": False},
        {"margin": 0.6, "correct": True},
    ]
    calibration = calibrate_margin_threshold(dev_rows, min_precision=0.8, min_samples=4)
    assert calibration["status"] == "calibrated"
    assert calibration["threshold"] == 0.2
    assert calibration["selected"] == 5
    assert calibration["precision"] == 0.8
    assert small_route_allowed(0.2, calibration)
    assert not small_route_allowed(0.19, calibration)
    assert not small_route_allowed(None, calibration)


def test_calibration_fails_closed_when_logprobs_are_missing_or_precision_unreachable():
    unavailable = calibrate_margin_threshold(
        [{"margin": None, "correct": True}] * 6, min_samples=4
    )
    assert unavailable["status"] == "insufficient_dev_logprobs"
    assert not small_route_allowed(100.0, unavailable)

    wrong = calibrate_margin_threshold(
        [{"margin": float(i), "correct": False} for i in range(6)],
        min_precision=0.8,
        min_samples=4,
    )
    assert wrong["status"] == "no_threshold_meets_precision"
    assert not small_route_allowed(100.0, wrong)


def _dev_rows():
    return [
        {"task": f"dev_{index}", "margin": margin, "correct": correct}
        for index, (margin, correct) in enumerate([
            (0.1, False), (0.2, True), (0.3, True), (0.4, True), (0.5, False), (0.6, True),
        ])
    ]


def _quant_rows():
    # Distinct row set from the BF16 fixture: all-correct rows after 0.1 yield
    # a different winning threshold (0.1 vs 0.2), proving per-arm calibration.
    return [
        {"task": f"q_{index}", "margin": margin, "correct": correct}
        for index, (margin, correct) in enumerate([
            (0.1, False), (0.2, True), (0.3, True), (0.4, True), (0.5, True), (0.6, True),
        ])
    ]


def _heldout_rows():
    # A genuinely separate split: same separating structure as the dev rows at
    # a different scale, which is what a threshold that transfers looks like.
    return [
        {"task": f"held_{index}", "margin": margin, "correct": correct}
        for index, (margin, correct) in enumerate([
            (0.15, False), (0.25, True), (0.35, True), (0.45, True), (0.55, True), (0.65, True),
        ])
    ]


def test_margin_shift_guard_fails_closed_on_missing_or_invalid_inputs():
    # No quantized arm served: guard inactive regardless of tolerance.
    assert margin_shift_fails_closed(None, None) is False
    # Quantized arm served (any valid tolerance): an unmeasured shift blocks,
    # even under a loose tolerance -- you cannot verify a bound you never measured.
    assert margin_shift_fails_closed(None, 5.0) is True
    assert margin_shift_fails_closed(None, 0.2) is True
    assert margin_shift_fails_closed(float("nan"), 0.2) is True
    assert margin_shift_fails_closed(float("inf"), 0.2) is True
    assert margin_shift_fails_closed(True, 0.2) is True
    # Invalid tolerance values fail closed (a negative or non-finite tolerance
    # cannot express "the declared acceptable drift").
    assert margin_shift_fails_closed(0.1, -1.0) is True
    assert margin_shift_fails_closed(0.1, float("nan")) is True
    assert margin_shift_fails_closed(0.1, True) is True
    # Within tolerance passes; beyond tolerance (either direction) blocks.
    assert margin_shift_fails_closed(0.19, 0.2) is False
    assert margin_shift_fails_closed(-0.19, 0.2) is False
    assert margin_shift_fails_closed(0.21, 0.2) is True
    assert margin_shift_fails_closed(-0.21, 0.2) is True


def test_validate_quantized_margin_shift_computes_or_refuses_shift():
    result = validate_quantized_margin_shift(2.0, 1.5, max_quantized_margin_shift=0.2)
    assert result["margin_shift"] == pytest.approx(-0.5)
    assert result["quantized_margin_shift_fails_closed"] is True
    ok = validate_quantized_margin_shift(2.0, 2.1, max_quantized_margin_shift=0.2)
    assert ok["margin_shift"] == pytest.approx(0.1)
    assert ok["quantized_margin_shift_fails_closed"] is False
    unmeasured = validate_quantized_margin_shift(None, 1.5, max_quantized_margin_shift=0.2)
    assert unmeasured["margin_shift"] is None
    assert unmeasured["quantized_margin_shift_fails_closed"] is True
    with pytest.raises(ValueError, match="bf16_mean_margin"):
        validate_quantized_margin_shift(float("nan"), 1.5, max_quantized_margin_shift=0.2)
    with pytest.raises(ValueError, match="quantized_mean_margin"):
        validate_quantized_margin_shift(2.0, "high", max_quantized_margin_shift=0.2)  # type: ignore[arg-type]


def test_quantized_guard_blocks_routing_even_when_margin_clears_threshold():
    calibration = calibrate_margin_threshold(_dev_rows(), min_precision=0.8, min_samples=4)
    assert calibration["status"] == "calibrated"
    # Margin clears the threshold, but serving a quantized arm with an
    # unmeasured shift must still route to the teacher.
    assert not small_route_allowed(0.6, calibration, max_quantized_margin_shift=0.2)
    assert small_route_allowed(0.6, calibration)  # pure-BF16 lane unaffected
    assert small_route_allowed(0.6, calibration, quantized_margin_shift=0.1, max_quantized_margin_shift=0.2)
    assert not small_route_allowed(0.6, calibration, quantized_margin_shift=-0.3, max_quantized_margin_shift=0.2)


def _outcome_rows(green_tasks, tasks=("task_a", "task_b", "task_c", "task_d")):
    return [{"task": task, "correct": task in green_tasks} for task in tasks]


def test_green_loss_guard_fails_closed_on_missing_or_invalid_inputs():
    # No quantized arm served: guard inactive regardless of tolerance.
    assert green_loss_fails_closed(None, None) is False
    assert green_loss_fails_closed(0.0, None) is False
    # Quantized arm served: an unmeasured loss fraction blocks even under a
    # permissive tolerance -- a bound you never measured cannot be verified.
    assert green_loss_fails_closed(None, 1.0) is True
    assert green_loss_fails_closed(None, 0.0) is True
    assert green_loss_fails_closed(float("nan"), 0.5) is True
    assert green_loss_fails_closed(float("inf"), 0.5) is True
    assert green_loss_fails_closed(True, 0.5) is True
    # A loss fraction is a ratio in [0, 1]; anything else is invalid evidence.
    assert green_loss_fails_closed(-0.01, 0.5) is True
    assert green_loss_fails_closed(1.01, 0.5) is True
    # Invalid tolerances fail closed too.
    assert green_loss_fails_closed(0.1, -0.5) is True
    assert green_loss_fails_closed(0.1, 1.5) is True
    assert green_loss_fails_closed(0.1, float("nan")) is True
    assert green_loss_fails_closed(0.1, True) is True
    # Loss at or below the configured fraction passes; above it blocks.
    assert green_loss_fails_closed(0.0, 0.0) is False
    assert green_loss_fails_closed(0.25, 0.25) is False
    assert green_loss_fails_closed(0.26, 0.25) is True
    assert green_loss_fails_closed(1.0, 1.0) is False


def test_green_retention_is_measured_per_task_not_by_matching_totals():
    # Same green count in both arms, but on different tasks: a totals-only
    # guard would call this intact, so retention is paired per task.
    reference = _outcome_rows({"task_a", "task_b"})
    swapped = _outcome_rows({"task_c", "task_d"})
    record = validate_quantized_green_retention(reference, swapped, max_green_loss_fraction=0.5)
    assert record["reference_greens"] == 2 and record["quantized_greens"] == 2
    assert record["retained_greens"] == 0 and record["lost_greens"] == 2
    assert record["gained_greens"] == 2  # gains are recorded, never credited
    assert record["green_loss_fraction"] == pytest.approx(1.0)
    assert record["quantized_green_loss_fails_closed"] is True

    kept = validate_quantized_green_retention(
        reference, _outcome_rows({"task_a"}), max_green_loss_fraction=0.5
    )
    assert (kept["retained_greens"], kept["lost_greens"], kept["gained_greens"]) == (1, 1, 0)
    assert kept["green_loss_fraction"] == pytest.approx(0.5)
    assert kept["quantized_green_loss_fails_closed"] is False  # at the bound passes
    strict = validate_quantized_green_retention(
        reference, _outcome_rows({"task_a"}), max_green_loss_fraction=0.0
    )
    assert strict["quantized_green_loss_fails_closed"] is True


def test_green_retention_fails_closed_on_a_vacuous_or_broken_measurement():
    # The reference arm never greens a task: nothing to retain, no evidence.
    vacuous = validate_quantized_green_retention(
        _outcome_rows(set()), _outcome_rows({"task_a"}), max_green_loss_fraction=1.0
    )
    assert vacuous["green_loss_fraction"] is None
    assert vacuous["quantized_green_loss_fails_closed"] is True
    assert "no green tasks" in vacuous["reason"]
    with pytest.raises(ValueError, match="same tasks"):
        validate_quantized_green_retention(
            _outcome_rows({"task_a"}), _outcome_rows({"task_a"}, tasks=("task_a", "task_b")),
            max_green_loss_fraction=0.0,
        )
    with pytest.raises(ValueError, match="boolean"):
        validate_quantized_green_retention(
            [{"task": "task_a", "correct": True}], [{"task": "task_a", "correct": 1}],
            max_green_loss_fraction=0.0,
        )
    with pytest.raises(ValueError, match="no task name"):
        validate_quantized_green_retention(
            [{"correct": True}], [{"correct": True}], max_green_loss_fraction=0.0
        )
    with pytest.raises(ValueError, match="no green rows"):
        validate_quantized_green_retention(
            [], [{"task": "task_a", "correct": True}], max_green_loss_fraction=0.0
        )
    with pytest.raises(ValueError, match="duplicate"):
        validate_quantized_green_retention(
            [{"task": "task_a", "correct": True}] * 2, [{"task": "task_a", "correct": True}],
            max_green_loss_fraction=0.0,
        )


def test_a_passing_shift_and_gate_cannot_license_a_green_losing_arm():
    # Margin-side evidence is fully satisfied: the live margin clears the
    # calibrated threshold, the shift is inside tolerance, and the held-out
    # gate was measured for this same arm and threshold.
    calibration = calibrate_margin_threshold(_dev_rows(), min_precision=0.8, min_samples=4)
    calibration["precision_arm"] = "int8_smoothquant"
    gate = heldout_transfer_gate(calibration, _heldout_rows(), min_precision=0.8, min_samples=4)
    assert gate["status"] == HELDOUT_GATE_STATUS
    margin_side = {"quantized_margin_shift": 0.1, "max_quantized_margin_shift": 0.2, "heldout_gate": gate}
    assert quant_route_allowed(
        100.0, calibration, green_loss_fraction=0.0, max_green_loss_fraction=0.0, **margin_side
    )
    # The quantized arm loses the reference arm's greens: the margin evidence
    # cannot license it, at any tolerance below the loss.
    assert not quant_route_allowed(
        100.0, calibration, green_loss_fraction=1.0, max_green_loss_fraction=0.0, **margin_side
    )
    # An unmeasured loss blocks even when the declared tolerance is permissive.
    assert not quant_route_allowed(
        100.0, calibration, green_loss_fraction=None, max_green_loss_fraction=0.5, **margin_side
    )
    # A declared tolerance is honoured, up to the declared fraction.
    assert quant_route_allowed(
        100.0, calibration, green_loss_fraction=0.5, max_green_loss_fraction=0.5, **margin_side
    )
    assert not quant_route_allowed(
        100.0, calibration, green_loss_fraction=0.5001, max_green_loss_fraction=0.5, **margin_side
    )


def test_per_precision_calibration_is_independent_and_never_borrows_thresholds():
    calibrations = calibrate_margin_threshold_per_precision(
        {"bf16": _dev_rows(), "int8_smoothquant": _quant_rows()}
    )
    assert set(calibrations) == {"bf16", "int8_smoothquant"}
    for arm, record in calibrations.items():
        assert record["precision_arm"] == arm
    assert calibrations["bf16"]["threshold"] == 0.2
    # The quantized arm's calibration is computed from its own rows only and
    # lands on a different threshold than the BF16 arm's.
    assert calibrations["int8_smoothquant"]["status"] == "calibrated"
    assert calibrations["int8_smoothquant"]["threshold"] == 0.1
    fail = calibrate_margin_threshold_per_precision(
        {"bf16": _dev_rows(), "int4_awq": [{"margin": None, "correct": True}] * 6}
    )
    assert fail["int4_awq"]["status"] == "insufficient_dev_logprobs"
    assert fail["int4_awq"]["threshold"] is None
    assert not quant_route_allowed(
        100.0, fail["int4_awq"], quantized_margin_shift=0.0, max_quantized_margin_shift=1.0,
        green_loss_fraction=0.0, max_green_loss_fraction=0.0,
    )
    with pytest.raises(ValueError, match="at least one precision arm"):
        calibrate_margin_threshold_per_precision({})


def test_quant_route_requires_precision_tagged_calibration_and_passes_guard():
    tagged = calibrate_margin_threshold(_dev_rows(), min_precision=0.8, min_samples=4)
    tagged["precision_arm"] = "int8_smoothquant"
    gate = heldout_transfer_gate(tagged, _heldout_rows(), min_precision=0.8, min_samples=4)
    assert gate["status"] == HELDOUT_GATE_STATUS
    assert quant_route_allowed(
        0.6, tagged, quantized_margin_shift=0.1, max_quantized_margin_shift=0.2,
        green_loss_fraction=0.0, max_green_loss_fraction=0.0, heldout_gate=gate,
    )
    assert not quant_route_allowed(
        0.6, tagged, quantized_margin_shift=-0.5, max_quantized_margin_shift=0.2,
        green_loss_fraction=0.0, max_green_loss_fraction=0.0, heldout_gate=gate,
    )
    # A calibration record not tagged for the quantized precision routes closed.
    untagged = calibrate_margin_threshold(_dev_rows(), min_precision=0.8, min_samples=4)
    assert not quant_route_allowed(
        0.6, untagged, quantized_margin_shift=0.0, max_quantized_margin_shift=0.2,
        green_loss_fraction=0.0, max_green_loss_fraction=0.0, heldout_gate=gate,
    )
    assert not quant_route_allowed(
        0.6, {"status": "calibrated", "threshold": 0.2},
        quantized_margin_shift=0.0, max_quantized_margin_shift=0.2,
        green_loss_fraction=0.0, max_green_loss_fraction=0.0, heldout_gate=gate,
    )


def test_calibration_records_which_tasks_it_was_fit_on():
    calibration = calibrate_margin_threshold(_dev_rows(), min_precision=0.8, min_samples=4)
    assert calibration["fit_task_ids"] == [f"dev_{index}" for index in range(6)]
    assert calibration["n_fit_tasks"] == 6
    # Anonymous rows cannot be audited for disjointness, so nothing is claimed.
    anonymous = calibrate_margin_threshold(
        [{"margin": 0.3, "correct": True}] * 6, min_precision=0.8, min_samples=4
    )
    assert anonymous["fit_task_ids"] == [] and anonymous["n_fit_tasks"] == 0


def test_heldout_gate_passes_when_the_threshold_transfers():
    calibration = calibrate_margin_threshold(_dev_rows(), min_precision=0.8, min_samples=4)
    calibration["precision_arm"] = "int8_smoothquant"
    gate = heldout_transfer_gate(calibration, _heldout_rows(), min_precision=0.8, min_samples=4)
    assert gate["status"] == HELDOUT_GATE_STATUS
    assert gate["threshold"] == calibration["threshold"]
    assert gate["precision_arm"] == "int8_smoothquant"
    assert gate["n_heldout"] == 6 and gate["n_heldout_with_margin"] == 6
    assert gate["selected"] == 5 and gate["heldout_precision"] == pytest.approx(1.0)
    assert gate["heldout_accuracy"] == pytest.approx(5 / 6)


def test_heldout_gate_fails_closed_when_the_threshold_does_not_transfer():
    # Same calibration, but on held-out tasks the "confident" rows are wrong:
    # the dev-split separation was noise, so the arm must not serve.
    calibration = calibrate_margin_threshold(_dev_rows(), min_precision=0.8, min_samples=4)
    calibration["precision_arm"] = "int8_smoothquant"
    poisoned = [
        {"task": f"held_{index}", "margin": margin, "correct": correct}
        for index, (margin, correct) in enumerate([
            (0.15, False), (0.25, False), (0.35, False), (0.45, False), (0.55, True), (0.65, True),
        ])
    ]
    gate = heldout_transfer_gate(calibration, poisoned, min_precision=0.8, min_samples=4)
    assert gate["status"] == "heldout_rejected"
    assert "does not transfer" in gate["reason"]
    assert not heldout_transfer_gate_allows(calibration, gate)


def test_heldout_gate_refuses_contaminated_and_unprovable_splits():
    calibration = calibrate_margin_threshold(_dev_rows(), min_precision=0.8, min_samples=4)
    calibration["precision_arm"] = "int8_smoothquant"

    contaminated = heldout_transfer_gate(calibration, _dev_rows(), min_precision=0.8, min_samples=4)
    assert contaminated["status"] == "heldout_contaminated"
    assert contaminated["n_overlapping_tasks"] == 6
    assert contaminated["overlapping_tasks"] == [f"dev_{index}" for index in range(5)]

    anonymous = heldout_transfer_gate(
        calibration, [{"margin": 0.5, "correct": True}] * 6, min_precision=0.8, min_samples=4
    )
    assert anonymous["status"] == "heldout_rejected"
    assert "no task id" in anonymous["reason"]

    no_margins = heldout_transfer_gate(
        calibration,
        [{"task": f"held_{i}", "margin": None, "correct": True} for i in range(6)],
        min_precision=0.8, min_samples=4,
    )
    assert no_margins["status"] == "heldout_rejected"
    assert "carry a margin" in no_margins["reason"]

    too_few = heldout_transfer_gate(calibration, _heldout_rows()[:2], min_precision=0.8, min_samples=4)
    assert too_few["status"] == "heldout_rejected"

    # A calibration with no recorded fit set cannot prove disjointness at all.
    untraceable = calibrate_margin_threshold(
        [{"margin": m, "correct": c} for m, c in [(0.2, True), (0.3, True), (0.4, True), (0.5, True)]],
        min_precision=0.8, min_samples=4,
    )
    blind = heldout_transfer_gate(untraceable, _heldout_rows(), min_precision=0.8, min_samples=4)
    assert blind["status"] == "heldout_rejected"
    assert "fit on" in blind["reason"]


def test_a_gate_cannot_be_spent_on_another_arm_or_another_threshold():
    calibration = calibrate_margin_threshold(_dev_rows(), min_precision=0.8, min_samples=4)
    calibration["precision_arm"] = "int8_smoothquant"
    gate = heldout_transfer_gate(calibration, _heldout_rows(), min_precision=0.8, min_samples=4)
    assert heldout_transfer_gate_allows(calibration, gate)

    # The BF16 arm's transfer evidence does not authorize the INT8 arm.
    bf16 = calibrate_margin_threshold(_dev_rows(), min_precision=0.8, min_samples=4)
    bf16["precision_arm"] = "bf16"
    assert not heldout_transfer_gate_allows(bf16, gate)
    assert not quant_route_allowed(
        0.6, bf16, quantized_margin_shift=0.0, max_quantized_margin_shift=0.2,
        green_loss_fraction=0.0, max_green_loss_fraction=0.0, heldout_gate=gate,
    )

    # Nor does a gate measured for a different threshold.
    retuned = {**calibration, "threshold": 0.35}
    assert not heldout_transfer_gate_allows(retuned, gate)
    assert not quant_route_allowed(
        0.6, retuned, quantized_margin_shift=0.0, max_quantized_margin_shift=0.2,
        green_loss_fraction=0.0, max_green_loss_fraction=0.0, heldout_gate=gate,
    )

    # Missing, unvalidated, or malformed gates all block.
    assert not quant_route_allowed(
        0.6, calibration, quantized_margin_shift=0.0, max_quantized_margin_shift=0.2,
        green_loss_fraction=0.0, max_green_loss_fraction=0.0,
    )
    for bad in (None, {}, {"status": "heldout_rejected"}, {**gate, "status": "ok"},
                {**gate, "threshold": None}, {**gate, "threshold": True}):
        assert not quant_route_allowed(
            0.6, calibration, quantized_margin_shift=0.0, max_quantized_margin_shift=0.2,
            green_loss_fraction=0.0, max_green_loss_fraction=0.0, heldout_gate=bad,
        )


def _admission_inputs():
    calibration = calibrate_margin_threshold(_dev_rows(), min_precision=0.8, min_samples=4)
    calibration["precision_arm"] = "int8_smoothquant"
    gate = heldout_transfer_gate(calibration, _heldout_rows(), min_precision=0.8, min_samples=4)
    assert gate["status"] == HELDOUT_GATE_STATUS
    return calibration, gate


def test_arm_admission_aggregates_every_guard_and_refuses_when_one_is_missing():
    calibration, gate = _admission_inputs()
    admitted = quantized_arm_admission(
        calibration,
        quantized_margin_shift=0.1,
        max_quantized_margin_shift=0.2,
        green_loss_fraction=0.0,
        max_green_loss_fraction=0.0,
        heldout_gate=gate,
    )
    assert admitted["artifact"] == ARM_ADMISSION_VERSION
    assert admitted["admitted"] is True and admitted["refusals"] == []
    assert set(admitted["guards"]) == {
        "calibration", "heldout_transfer_gate", "margin_shift", "green_retention"
    }
    assert all(record["passed"] for record in admitted["guards"].values())
    assert verify_arm_admission(admitted) == {"valid": True, "reason": None, "recomputed_admitted": True}

    # One missing measurement refuses admission even when every other guard passes.
    missing = quantized_arm_admission(
        calibration,
        quantized_margin_shift=0.1,
        max_quantized_margin_shift=0.2,
        green_loss_fraction=None,
        max_green_loss_fraction=0.0,
        heldout_gate=gate,
    )
    assert missing["admitted"] is False
    assert [refusal["guard"] for refusal in missing["refusals"]] == ["green_retention"]
    assert verify_arm_admission(missing)["valid"] is True

    # A missing tolerance is a refusal here, never the BF16-lane "inactive" state.
    undeclared = quantized_arm_admission(
        calibration,
        quantized_margin_shift=0.1,
        max_quantized_margin_shift=None,
        green_loss_fraction=0.0,
        max_green_loss_fraction=None,
        heldout_gate=gate,
    )
    assert undeclared["admitted"] is False
    assert {refusal["guard"] for refusal in undeclared["refusals"]} == {"margin_shift", "green_retention"}

    # An untagged (BF16) calibration with no gate refuses on both counts.
    untagged = calibrate_margin_threshold(_dev_rows(), min_precision=0.8, min_samples=4)
    refused = quantized_arm_admission(
        untagged,
        quantized_margin_shift=0.0,
        max_quantized_margin_shift=0.2,
        green_loss_fraction=0.0,
        max_green_loss_fraction=0.0,
        heldout_gate=None,
    )
    assert refused["admitted"] is False
    assert {refusal["guard"] for refusal in refused["refusals"]} == {"calibration", "heldout_transfer_gate"}


def test_admission_verification_catches_fabricated_verdicts_and_edits():
    calibration, gate = _admission_inputs()
    admission = quantized_arm_admission(
        calibration,
        quantized_margin_shift=0.1,
        max_quantized_margin_shift=0.2,
        green_loss_fraction=0.0,
        max_green_loss_fraction=0.0,
        heldout_gate=gate,
    )
    fabricated = {
        **admission,
        "guards": {
            **admission["guards"],
            "green_retention": {**admission["guards"]["green_retention"], "passed": False},
        },
    }
    result = verify_arm_admission(fabricated)
    assert result["valid"] is False and "do not follow" in result["reason"]

    edited = {**admission, "measurements": {**admission["measurements"], "green_loss_fraction": 1.0}}
    result = verify_arm_admission(edited)
    assert result["valid"] is False and "digest mismatch" in result["reason"]

    for bad in (
        {},
        {"artifact": "quantized_arm_admission/v2"},
        {**admission, "artifact": "something/else"},
        {**admission, "measurements": None},
    ):
        assert verify_arm_admission(bad)["valid"] is False


def test_quantized_arm_is_blocked_without_heldout_evidence_even_on_a_clean_calibration():
    # A dev set that calibrates beautifully is still not held-out evidence.
    calibrations = calibrate_margin_threshold_per_precision(
        {"bf16": _dev_rows(), "int8_smoothquant": _quant_rows()}
    )
    quant = calibrations["int8_smoothquant"]
    assert quant["status"] == "calibrated" and quant["threshold"] == 0.1
    assert not quant_route_allowed(
        100.0, quant, quantized_margin_shift=0.0, max_quantized_margin_shift=1.0,
        green_loss_fraction=0.0, max_green_loss_fraction=0.0,
    )
    gate = heldout_transfer_gate(quant, _heldout_rows(), min_precision=0.8, min_samples=4)
    assert gate["status"] == HELDOUT_GATE_STATUS  # held-out rows clear the 0.1 threshold
    assert quant_route_allowed(
        100.0, quant, quantized_margin_shift=0.0, max_quantized_margin_shift=1.0,
        green_loss_fraction=0.0, max_green_loss_fraction=0.0, heldout_gate=gate,
    )
