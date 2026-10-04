"""Strict token log-probability confidence and dev-only router calibration.

A margin is available only when the response contains the selected token's
logprob and at least one distinct alternative token's logprob at every output
position. Missing fields are unavailable evidence, never replaced by a lexical
or self-review heuristic.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


def chosen_token_margins(response: Mapping[str, Any]) -> list[float] | None:
    """Return selected-token vs best-alternative margins, or ``None``.

    Expected shape is the OpenAI-compatible chat completion response:
    ``choices[0].logprobs.content[*]`` with ``token``, ``logprob`` and a
    ``top_logprobs`` list. Every token must have an actual distinct alternative.
    """
    try:
        choice = response["choices"][0]
        entries = choice["logprobs"]["content"]
    except (KeyError, IndexError, TypeError):
        return None
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)) or not entries:
        return None

    margins: list[float] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            return None
        token = entry.get("token")
        selected = entry.get("logprob")
        alternatives = entry.get("top_logprobs")
        if not isinstance(token, str) or not isinstance(selected, (int, float)) or isinstance(selected, bool):
            return None
        if not isinstance(alternatives, Sequence) or isinstance(alternatives, (str, bytes)):
            return None
        other_values = [
            float(item["logprob"])
            for item in alternatives
            if isinstance(item, Mapping)
            and isinstance(item.get("token"), str)
            and item["token"] != token
            and isinstance(item.get("logprob"), (int, float))
            and not isinstance(item.get("logprob"), bool)
            and math.isfinite(float(item["logprob"]))
        ]
        selected_value = float(selected)
        if not other_values or not math.isfinite(selected_value):
            return None
        alternative = max(other_values)
        if not math.isfinite(alternative):
            return None
        margins.append(selected_value - alternative)
    return margins


def aggregate_margin(response: Mapping[str, Any]) -> float | None:
    """Use the mean selected-token margin as the completion confidence score."""
    margins = chosen_token_margins(response)
    if margins is None:
        return None
    return sum(margins) / len(margins)


def calibrate_margin_threshold(
    rows: Sequence[Mapping[str, Any]], *, min_precision: float = 0.80,
    min_samples: int = 4,
) -> dict[str, Any]:
    """Choose the highest-coverage small-route cutoff using dev rows only.

    ``rows`` must contain ``margin`` and ground-truth ``correct``. The returned
    record includes all candidate statistics needed to audit the selection. If
    the dev signal is missing, undersized, or cannot meet the precision target,
    the result fails closed and routes all evaluation queries to the teacher.
    """
    if not 0.0 < min_precision <= 1.0:
        raise ValueError("min_precision must be in (0, 1]")
    if min_samples < 1:
        raise ValueError("min_samples must be positive")
    usable = [
        {"margin": float(row["margin"]), "correct": bool(row["correct"])}
        for row in rows
        if isinstance(row.get("margin"), (int, float))
        and math.isfinite(float(row["margin"]))
        and isinstance(row.get("correct"), bool)
    ]
    # Which tasks the threshold was fit on. A held-out transfer gate cannot
    # prove disjointness without this, so it is recorded even when the caller
    # supplied anonymous rows (the set is then empty and the gate fails closed).
    fit_ids = sorted({
        str(row.get("task") or row.get("id") or "").strip() for row in rows
    } - {""})
    base = {
        "n_dev": len(rows),
        "n_with_margin": len(usable),
        "min_precision": min_precision,
        "fit_task_ids": fit_ids,
        "n_fit_tasks": len(fit_ids),
    }
    if len(usable) < min_samples:
        return {**base, "status": "insufficient_dev_logprobs", "threshold": None, "selected": 0}

    candidates: list[dict[str, Any]] = []
    for threshold in sorted({row["margin"] for row in usable}):
        selected = [row for row in usable if row["margin"] >= threshold]
        precision = sum(row["correct"] for row in selected) / len(selected)
        candidates.append({
            "threshold": threshold,
            "selected": len(selected),
            "coverage": len(selected) / len(usable),
            "precision": precision,
            "meets_min_samples": len(selected) >= min_samples,
        })
    eligible = [
        row for row in candidates
        if row["precision"] >= min_precision and row["selected"] >= min_samples
    ]
    if not eligible:
        return {
            **base, "status": "no_threshold_meets_precision", "threshold": None,
            "selected": 0, "candidates": candidates,
        }
    winner = max(eligible, key=lambda row: (row["selected"], row["precision"], -row["threshold"]))
    return {**base, "status": "calibrated", **winner, "candidates": candidates}


DEFAULT_MARGIN_SHIFT_TOLERANCE = 0.2


def margin_shift_fails_closed(
    quantized_margin_shift: float | None, max_quantized_margin_shift: float | None
) -> bool:
    """Decide whether an unmeasured or over-tolerance margin shift blocks the small route.

    ``max_quantized_margin_shift=None`` means no quantized arm is being served
    (the BF16 lane): no guard is active and this returns ``False``. In every
    other case the guard fails closed: an unmeasured shift (``None``), a
    non-finite or boolean shift, a missing/invalid tolerance, or a shift whose
    magnitude exceeds the tolerance all block the small route.
    """
    if max_quantized_margin_shift is None:
        return False
    if (
        isinstance(max_quantized_margin_shift, bool)
        or not isinstance(max_quantized_margin_shift, (int, float))
        or not math.isfinite(float(max_quantized_margin_shift))
        or float(max_quantized_margin_shift) < 0
    ):
        return True
    if (
        quantized_margin_shift is None
        or isinstance(quantized_margin_shift, bool)
        or not isinstance(quantized_margin_shift, (int, float))
        or not math.isfinite(float(quantized_margin_shift))
    ):
        return True
    return abs(float(quantized_margin_shift)) > float(max_quantized_margin_shift)


def validate_quantized_margin_shift(
    bf16_mean_margin: float | None,
    quantized_mean_margin: float | None,
    *,
    max_quantized_margin_shift: float,
) -> dict[str, Any]:
    """Compute the measured BF16→quantized mean-margin shift and its routing verdict.

    Inputs must be finite numbers or ``None`` (booleans rejected); ``None`` on
    either side yields ``margin_shift: None`` rather than a guessed value, and
    the verdict comes from ``margin_shift_fails_closed`` (so an unmeasured
    shift blocks the small route whenever a quantized arm is being served).
    """
    means: list[float | None] = []
    for name, value in (("bf16_mean_margin", bf16_mean_margin), ("quantized_mean_margin", quantized_mean_margin)):
        if value is None or (
            not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))
        ):
            means.append(None if value is None else float(value))
        else:
            raise ValueError(f"{name} must be a finite number or None")
    shift = None
    if means[0] is not None and means[1] is not None:
        shift = means[1] - means[0]
    return {
        "margin_shift": shift,
        "quantized_margin_shift_fails_closed": margin_shift_fails_closed(shift, max_quantized_margin_shift),
    }


DEFAULT_MAX_GREEN_LOSS_FRACTION = 0.0


def green_loss_fails_closed(
    quantized_green_loss_fraction: float | None, max_green_loss_fraction: float | None
) -> bool:
    """Decide whether greens lost by a quantized arm block the small route.

    ``max_green_loss_fraction=None`` means no quantized arm is being served
    (the BF16 lane): no guard is active and this returns ``False``. In every
    other case the guard fails closed: an unmeasured loss fraction (``None``),
    a boolean/non-finite fraction outside ``[0, 1]``, a missing/invalid
    tolerance outside ``[0, 1]``, or a measured loss above the tolerance all
    block the small route. A margin bound is evidence about the margin scale,
    never about whether the arm still solves tasks, so this guard is evaluated
    independently of ``margin_shift_fails_closed``.
    """
    if max_green_loss_fraction is None:
        return False
    if (
        isinstance(max_green_loss_fraction, bool)
        or not isinstance(max_green_loss_fraction, (int, float))
        or not math.isfinite(float(max_green_loss_fraction))
        or not 0.0 <= float(max_green_loss_fraction) <= 1.0
    ):
        return True
    if (
        quantized_green_loss_fraction is None
        or isinstance(quantized_green_loss_fraction, bool)
        or not isinstance(quantized_green_loss_fraction, (int, float))
        or not math.isfinite(float(quantized_green_loss_fraction))
    ):
        return True
    fraction = float(quantized_green_loss_fraction)
    if not 0.0 <= fraction <= 1.0:
        return True
    return fraction > float(max_green_loss_fraction)


def validate_quantized_green_retention(
    reference_rows: Sequence[Mapping[str, Any]],
    quantized_rows: Sequence[Mapping[str, Any]],
    *,
    max_green_loss_fraction: float,
) -> dict[str, Any]:
    """Measure the fraction of the reference arm's greens the quantized arm loses.

    Rows are per-task outcomes, ``{"task": str, "correct": bool}``, and the two
    arms must cover exactly the same task set (paired design, like
    ``compare_margin_signal``). Retention is counted *per task*, so an arm that
    swaps one green for another cannot hide the loss behind equal totals.
    Malformed, duplicated, anonymous, or unpaired rows raise ``ValueError`` —
    that is a broken measurement, not a routing decision.

    The record carries the measured counts, ``green_loss_fraction`` over the
    reference arm's greens, and the verdict from ``green_loss_fails_closed``.
    A reference arm with no greens on the suite fails closed with the fraction
    left ``None``: with no demonstrated capability there is nothing whose
    retention could be evidenced.
    """
    by_arm: dict[str, dict[str, bool]] = {}
    for label, rows in (("reference", reference_rows), ("quantized", quantized_rows)):
        if not rows:
            raise ValueError(f"{label} arm has no green rows; retention is unmeasured")
        table: dict[str, bool] = {}
        for row in rows:
            task = str(row.get("task") or row.get("id") or "").strip()
            if not task:
                raise ValueError(f"{label} green row has no task name")
            if task in table:
                raise ValueError(f"{label} arm has duplicate rows for task {task}")
            correct = row.get("correct")
            if not isinstance(correct, bool):
                raise ValueError(f"{label} row for task {task} must carry a boolean correct flag")
            table[task] = correct
        by_arm[label] = table
    reference_tasks, quantized_tasks = set(by_arm["reference"]), set(by_arm["quantized"])
    if reference_tasks != quantized_tasks:
        unpaired = sorted(reference_tasks ^ quantized_tasks)[:3]
        raise ValueError(f"arms are not paired over the same tasks; first mismatches: {unpaired}")

    tasks = sorted(reference_tasks)
    reference_greens = sum(by_arm["reference"].values())
    quantized_greens = sum(by_arm["quantized"].values())
    retained_greens = sum(by_arm["reference"][task] and by_arm["quantized"][task] for task in tasks)
    gained_greens = sum(by_arm["quantized"][task] and not by_arm["reference"][task] for task in tasks)
    lost_greens = reference_greens - retained_greens
    fraction = lost_greens / reference_greens if reference_greens else None
    record: dict[str, Any] = {
        "n_tasks": len(tasks),
        "reference_greens": reference_greens,
        "quantized_greens": quantized_greens,
        "retained_greens": retained_greens,
        "lost_greens": lost_greens,
        "gained_greens": gained_greens,
        "green_loss_fraction": fraction,
        "max_green_loss_fraction": max_green_loss_fraction,
        "quantized_green_loss_fails_closed": green_loss_fails_closed(fraction, max_green_loss_fraction),
    }
    if reference_greens == 0:
        record["reason"] = (
            "the reference arm has no green tasks on this suite, so no retention can be "
            "evidenced; refusing to license the quantized arm on a vacuous denominator"
        )
    return record


def small_route_allowed(
    margin: float | None,
    calibration: Mapping[str, Any],
    *,
    quantized_margin_shift: float | None = None,
    max_quantized_margin_shift: float | None = None,
) -> bool:
    """Fail closed to the teacher unless valid dev calibration and margin exist.

    When the serving model is a quantized arm, pass ``quantized_margin_shift``
    (mean quantized margin minus mean BF16 margin, from
    ``validate_quantized_margin_shift``) and ``max_quantized_margin_shift``:
    an unmeasured or over-tolerance shift then blocks the small route even if
    the margin itself clears the threshold. That shift alone never licenses a
    quantized arm: the full decision must go through ``quant_route_allowed``,
    which additionally requires the held-out transfer gate and the measured
    green retention. Strict parsing excludes booleans from both the margin and
    the stored threshold.
    """
    threshold = calibration.get("threshold")
    allowed = (
        calibration.get("status") == "calibrated"
        and isinstance(margin, (int, float))
        and not isinstance(margin, bool)
        and math.isfinite(float(margin))
        and isinstance(threshold, (int, float))
        and not isinstance(threshold, bool)
        and math.isfinite(float(threshold))
        and float(margin) >= float(threshold)
    )
    if not allowed:
        return False
    return not margin_shift_fails_closed(quantized_margin_shift, max_quantized_margin_shift)


def calibrate_margin_threshold_per_precision(
    rows_by_precision: Mapping[str, Sequence[Mapping[str, Any]]], *,
    min_precision: float = 0.80,
    min_samples: int = 4,
) -> dict[str, dict[str, Any]]:
    """Calibrate one threshold per serving precision (e.g. ``bf16`` vs a PTQ arm).

    Quantization can flatten the selected-token margin distribution, so a
    threshold calibrated on BF16 logprobs must never be applied to a quantized
    arm. Each precision key gets its own independent calibration record, tagged
    with ``precision_arm``; a precision whose rows fail (missing logprobs,
    unreachable precision) yields its fail-closed status verbatim, never a
    borrowed threshold.
    """
    if not rows_by_precision:
        raise ValueError("per-precision calibration requires at least one precision arm")
    calibrations: dict[str, dict[str, Any]] = {}
    for precision, rows in rows_by_precision.items():
        key = str(precision).strip()
        if not key:
            raise ValueError("precision keys must be nonempty")
        calibration = calibrate_margin_threshold(
            list(rows), min_precision=min_precision, min_samples=min_samples
        )
        # Tagged separately from the record's numeric `precision` (the winner's
        # selected-precision fraction), which must stay untouched for audits.
        calibration["precision_arm"] = key
        calibrations[key] = calibration
    return calibrations


def quant_route_allowed(
    margin: float | None,
    quantized_calibration: Mapping[str, Any],
    *,
    quantized_margin_shift: float | None,
    max_quantized_margin_shift: float | None,
    green_loss_fraction: float | None,
    max_green_loss_fraction: float | None,
    heldout_gate: Mapping[str, Any] | None = None,
) -> bool:
    """Small-route decision for a quantized serving arm.

    Requires a ``calibrated`` record that was actually produced for the
    quantized precision (tagged via ``calibrate_margin_threshold_per_precision``
    — a record without a ``precision_arm`` tag is treated as uncalibrated), and
    enforces the margin-shift guard. All keyword arguments are required so a
    caller cannot silently skip measuring the shift.

    ``heldout_gate`` is the held-out transfer gate
    (``heldout_transfer_gate``). It is required, not optional: a quantized arm
    may not serve the small route on dev-set evidence alone. The gate must have
    been measured for this same precision arm and this same threshold, so a
    BF16 transfer result cannot authorize the INT8 arm or vice versa.

    ``green_loss_fraction`` (measured by ``validate_quantized_green_retention``)
    and ``max_green_loss_fraction`` are a second mandatory, independent guard.
    A shift inside tolerance says the margin *scale* moved no more than
    declared, not that the arm still solves the tasks the reference arm solved.
    Losing more than the declared fraction of the reference arm's greens — or
    leaving that loss unmeasured — blocks, so a passing margin-shift bound can
    never license routing by itself.
    """
    precision_tag = quantized_calibration.get("precision_arm")
    if not isinstance(precision_tag, str) or not precision_tag.strip():
        return False
    if not heldout_transfer_gate_allows(quantized_calibration, heldout_gate):
        return False
    if green_loss_fails_closed(green_loss_fraction, max_green_loss_fraction):
        return False
    return small_route_allowed(
        margin,
        quantized_calibration,
        quantized_margin_shift=quantized_margin_shift,
        max_quantized_margin_shift=max_quantized_margin_shift,
    )


HELDOUT_GATE_STATUS = "heldout_validated"


def heldout_transfer_gate(
    calibration: Mapping[str, Any],
    heldout_rows: Sequence[Mapping[str, Any]],
    *,
    min_precision: float = 0.80,
    min_samples: int = 4,
    min_tasks: int = 4,
) -> dict[str, Any]:
    """Test whether a calibrated threshold still holds up on unseen tasks.

    The dev split picks a threshold; nothing about that threshold guarantees it
    separates correct from incorrect answers on tasks the dev split never saw.
    This gate re-evaluates the *already chosen* threshold on a held-out set and
    fails closed unless the separation survives at the same precision target.

    Fails closed on every ambiguity rather than guessing:

    * the calibration is not ``calibrated`` or has no finite threshold;
    * the calibration did not record which tasks it was fit on, or any held-out
      task was already in that fit set (a contaminated "held-out" split);
    * a held-out row has no task id, so disjointness cannot be proven;
    * too few held-out rows carry a margin, too few clear the threshold, or the
      held-out precision is below ``min_precision``.

    The record echoes the ``precision_arm`` and ``threshold`` it was measured
    for, so the result can only be spent on the arm it actually describes.
    """
    if not 0.0 < min_precision <= 1.0:
        raise ValueError("min_precision must be in (0, 1]")
    if min_samples < 1 or min_tasks < 1:
        raise ValueError("min_samples and min_tasks must be positive")
    threshold = calibration.get("threshold")
    record: dict[str, Any] = {
        "precision_arm": calibration.get("precision_arm"),
        "threshold": threshold,
        "min_precision": min_precision,
        "min_samples": min_samples,
        "min_tasks": min_tasks,
        "n_heldout": len(heldout_rows),
        "status": "heldout_rejected",
    }
    if calibration.get("status") != "calibrated":
        record["reason"] = "calibration is not calibrated"
        return record
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
    ):
        record["reason"] = "calibration has no finite threshold"
        return record
    fit_ids = calibration.get("fit_task_ids")
    if not isinstance(fit_ids, Sequence) or isinstance(fit_ids, (str, bytes)) or not fit_ids:
        record["reason"] = "calibration does not record the tasks it was fit on"
        return record
    fit_set = {str(task) for task in fit_ids}
    record["n_fit_tasks"] = len(fit_set)

    heldout_ids: list[str] = []
    usable: list[dict[str, Any]] = []
    for index, row in enumerate(heldout_rows):
        task_id = str(row.get("task") or row.get("id") or "").strip()
        if not task_id:
            record["reason"] = f"held-out row {index} has no task id; disjointness unproven"
            return record
        heldout_ids.append(task_id)
        if (
            isinstance(row.get("margin"), (int, float))
            and not isinstance(row.get("margin"), bool)
            and math.isfinite(float(row["margin"]))
            and isinstance(row.get("correct"), bool)
        ):
            usable.append({"task": task_id, "margin": float(row["margin"]), "correct": bool(row["correct"])})
    overlap = sorted(fit_set & set(heldout_ids))
    record["heldout_task_ids"] = sorted(set(heldout_ids))
    record["n_heldout_with_margin"] = len(usable)
    if overlap:
        record["status"] = "heldout_contaminated"
        record["n_overlapping_tasks"] = len(overlap)
        record["overlapping_tasks"] = overlap[:5]
        record["reason"] = "held-out tasks overlap the calibration fit set"
        return record
    if len(usable) < min_tasks:
        record["reason"] = f"only {len(usable)} held-out rows carry a margin; {min_tasks} required"
        return record
    selected = [row for row in usable if row["margin"] >= float(threshold)]
    precision = sum(row["correct"] for row in selected) / len(selected) if selected else 0.0
    record.update({
        "selected": len(selected),
        "coverage": len(selected) / len(usable),
        "heldout_precision": precision,
        "heldout_accuracy": sum(row["correct"] for row in usable) / len(usable),
    })
    if len(selected) < min_samples:
        record["reason"] = f"only {len(selected)} held-out rows clear the threshold; {min_samples} required"
        return record
    if precision < min_precision:
        record["reason"] = (
            f"held-out precision {precision:.3f} is below the {min_precision:.3f} target; "
            "the dev-set threshold does not transfer"
        )
        return record
    record["status"] = HELDOUT_GATE_STATUS
    return record


def heldout_transfer_gate_allows(
    quantized_calibration: Mapping[str, Any], heldout_gate: Mapping[str, Any] | None
) -> bool:
    """True only for a validated gate measured for this arm and this threshold."""
    if not isinstance(heldout_gate, Mapping):
        return False
    if heldout_gate.get("status") != HELDOUT_GATE_STATUS:
        return False
    arm = quantized_calibration.get("precision_arm")
    if not isinstance(arm, str) or not arm.strip():
        return False
    if heldout_gate.get("precision_arm") != arm:
        return False
    threshold = quantized_calibration.get("threshold")
    gate_threshold = heldout_gate.get("threshold")
    if isinstance(threshold, bool) or isinstance(gate_threshold, bool):
        return False
    if not isinstance(threshold, (int, float)) or not isinstance(gate_threshold, (int, float)):
        return False
    return math.isfinite(float(threshold)) and float(threshold) == float(gate_threshold)


ARM_ADMISSION_VERSION = "quantized_arm_admission/v1"

# Gate fields carried into the artifact's recorded measurements, so a verifier
# can re-derive the gate verdict without the full gate record.
_GATE_AUDIT_FIELDS = (
    "status", "reason", "precision_arm", "threshold", "n_heldout", "n_heldout_with_margin",
    "n_fit_tasks", "selected", "coverage", "heldout_precision", "heldout_accuracy",
    "n_overlapping_tasks", "overlapping_tasks",
)


def _recorded_number(value: Any) -> float | None:
    """Normalize a measurement for recording: finite floats only, booleans rejected."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    return float(value)


def _admission_digest(measurements: Mapping[str, Any]) -> str:
    """Deterministic content hash over the recorded measurements (tamper-evidence)."""
    import hashlib

    payload = json.dumps(measurements, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def quantized_arm_admission(
    calibration: Mapping[str, Any],
    *,
    quantized_margin_shift: float | None,
    max_quantized_margin_shift: float | None,
    green_loss_fraction: float | None,
    max_green_loss_fraction: float | None,
    heldout_gate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate every router guard into one auditable quantized-arm admission.

    Admission is arm-level, not per query: it says whether this precision may
    serve the small route at all. Four guards are recorded with their measured
    inputs and individual verdicts:

    * ``calibration`` -- a ``calibrated`` record with a finite threshold and a
      ``precision_arm`` tag (per-precision calibration);
    * ``heldout_transfer_gate`` -- the validated held-out gate, bound to this
      arm and this threshold (contaminated or non-transferring splits are its
      refusal statuses);
    * ``margin_shift`` -- the measured quantized-vs-reference mean-margin shift
      against a declared tolerance;
    * ``green_retention`` -- the measured fraction of the reference arm's greens
      the quantized arm loses against a declared tolerance.

    Unlike the per-guard helpers, a missing tolerance here is a refusal, not
    "no quantized arm is served": this artifact only ever describes a
    quantized arm. Admission is granted only when all four guards pass, so a
    missing measurement can never be silently omitted. The recorded
    measurements are digested so ``verify_arm_admission`` can re-derive every
    verdict later: an artifact that misreports its own verdict, or whose inputs
    were edited, fails verification.
    """
    arm = calibration.get("precision_arm")
    arm = arm.strip() if isinstance(arm, str) else ""
    calibration_status = calibration.get("status")
    if not isinstance(calibration_status, str):
        calibration_status = None
    threshold = _recorded_number(calibration.get("threshold"))
    calibration_passed = calibration_status == "calibrated" and threshold is not None and bool(arm)

    gate_passed = heldout_transfer_gate_allows(calibration, heldout_gate)
    gate_reason = heldout_gate.get("reason") if isinstance(heldout_gate, Mapping) else None
    if not gate_passed and not isinstance(gate_reason, str):
        gate_reason = "no held-out gate is validated for this arm and threshold"

    shift = _recorded_number(quantized_margin_shift)
    shift_tolerance = _recorded_number(max_quantized_margin_shift)
    if shift_tolerance is not None and shift_tolerance < 0.0:
        shift_tolerance = None
    shift_passed = (
        shift is not None
        and shift_tolerance is not None
        and not margin_shift_fails_closed(quantized_margin_shift, max_quantized_margin_shift)
    )
    if shift is None:
        shift_reason = "the quantized margin shift is unmeasured"
    elif shift_tolerance is None:
        shift_reason = "no valid margin-shift tolerance is declared"
    else:
        shift_reason = f"measured shift {shift:+.4f} exceeds the declared tolerance {shift_tolerance:.4f}"

    loss_fraction = _recorded_number(green_loss_fraction)
    if loss_fraction is not None and not 0.0 <= loss_fraction <= 1.0:
        loss_fraction = None
    loss_tolerance = _recorded_number(max_green_loss_fraction)
    if loss_tolerance is not None and not 0.0 <= loss_tolerance <= 1.0:
        loss_tolerance = None
    retention_passed = (
        loss_fraction is not None
        and loss_tolerance is not None
        and not green_loss_fails_closed(green_loss_fraction, max_green_loss_fraction)
    )
    if loss_fraction is None:
        retention_reason = "the green-loss fraction is unmeasured"
    elif loss_tolerance is None:
        retention_reason = "no valid green-loss tolerance is declared"
    else:
        retention_reason = (
            f"the quantized arm loses {loss_fraction:.4f} of the reference arm's greens, "
            f"above the declared {loss_tolerance:.4f}"
        )

    refusals: list[dict[str, str]] = []
    if not calibration_passed:
        refusals.append({"guard": "calibration", "reason": "no calibrated, precision-tagged threshold"})
    if not gate_passed:
        refusals.append({"guard": "heldout_transfer_gate", "reason": str(gate_reason)})
    if not shift_passed:
        refusals.append({"guard": "margin_shift", "reason": shift_reason})
    if not retention_passed:
        refusals.append({"guard": "green_retention", "reason": retention_reason})

    recorded_gate = (
        {key: heldout_gate[key] for key in _GATE_AUDIT_FIELDS if key in heldout_gate}
        if isinstance(heldout_gate, Mapping)
        else None
    )
    measurements = {
        "precision_arm": arm or None,
        "calibration_status": calibration_status,
        "threshold": threshold,
        "quantized_margin_shift": shift,
        "max_quantized_margin_shift": shift_tolerance,
        "green_loss_fraction": loss_fraction,
        "max_green_loss_fraction": loss_tolerance,
        "heldout_gate": recorded_gate,
    }
    return {
        "artifact": ARM_ADMISSION_VERSION,
        "precision_arm": arm or None,
        "guards": {
            "calibration": {"passed": calibration_passed, "status": calibration_status, "threshold": threshold},
            "heldout_transfer_gate": {
                "passed": gate_passed,
                "status": recorded_gate.get("status") if recorded_gate else None,
                "reason": gate_reason,
            },
            "margin_shift": {"passed": shift_passed, "measured": shift, "tolerance": shift_tolerance},
            "green_retention": {
                "passed": retention_passed,
                "measured": loss_fraction,
                "tolerance": loss_tolerance,
            },
        },
        "refusals": refusals,
        "admitted": not refusals,
        "measurements": measurements,
        "measurements_sha256": _admission_digest(measurements),
    }


def verify_arm_admission(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Re-derive an admission artifact's verdicts from its own measurements.

    Fails closed: an unknown artifact version, a missing measurements block, a
    digest mismatch (inputs edited after writing), or guard verdicts that do
    not follow from a recomputation of the recorded measurements all report
    ``valid: False`` with the reason. A verified artifact says nothing about
    the *truthfulness* of the measurements themselves -- that is what the
    per-guard measurement paths and their own digests are for -- only that the
    recorded verdict follows from the recorded numbers and that they were not
    edited afterwards.
    """
    if not isinstance(artifact, Mapping):
        return {"valid": False, "reason": "admission artifact is not a mapping", "recomputed_admitted": None}
    if artifact.get("artifact") != ARM_ADMISSION_VERSION:
        return {
            "valid": False,
            "reason": f"unknown admission artifact version: {artifact.get('artifact')!r}",
            "recomputed_admitted": None,
        }
    measurements = artifact.get("measurements")
    if not isinstance(measurements, Mapping):
        return {"valid": False, "reason": "admission artifact has no measurements block", "recomputed_admitted": None}
    if artifact.get("measurements_sha256") != _admission_digest(measurements):
        return {
            "valid": False,
            "reason": "measurements digest mismatch: the artifact was edited after it was written",
            "recomputed_admitted": None,
        }
    recomputed = quantized_arm_admission(
        {
            "status": measurements.get("calibration_status"),
            "precision_arm": measurements.get("precision_arm"),
            "threshold": measurements.get("threshold"),
        },
        quantized_margin_shift=measurements.get("quantized_margin_shift"),
        max_quantized_margin_shift=measurements.get("max_quantized_margin_shift"),
        green_loss_fraction=measurements.get("green_loss_fraction"),
        max_green_loss_fraction=measurements.get("max_green_loss_fraction"),
        heldout_gate=measurements.get("heldout_gate"),
    )
    if recomputed["guards"] != artifact.get("guards") or recomputed["refusals"] != artifact.get("refusals"):
        return {
            "valid": False,
            "reason": "recorded guard verdicts do not follow from the recorded measurements",
            "recomputed_admitted": recomputed["admitted"],
        }
    return {"valid": True, "reason": None, "recomputed_admitted": recomputed["admitted"]}
