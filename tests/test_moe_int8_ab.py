"""``moe_int8_ab`` summarises phases from measurements, so its arithmetic is
checkable without a card: the same four-phase shape the real run writes."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "chowder_batch"))

import moe_int8_ab as ab  # noqa: E402


def _phase(name: str, seconds: list[float], *, alloc: float, reserved: float) -> dict:
    return {
        "name": name,
        "step_seconds": seconds,
        "losses": [0.5] * len(seconds),
        "peak_alloc_gb": alloc,
        "peak_reserved_gb": reserved,
    }


def test_the_first_step_of_each_phase_is_kept_but_not_averaged():
    phases = [
        _phase("loop", [99.0, 10.0, 12.0], alloc=10.0, reserved=13.0),
        _phase("grouped", [40.0, 6.0, 8.0], alloc=10.5, reserved=13.4),
    ]
    summary = ab.summarize_phases(phases)

    assert summary["phases"]["loop"]["s_per_step"] == 11.0
    assert summary["phases"]["grouped"]["s_per_step"] == 7.0
    assert summary["phases"]["loop"]["steps_measured"] == 2


def test_a_single_step_phase_has_no_warm_mean_rather_than_reporting_it():
    summary = ab.summarize_phases([_phase("loop", [42.0], alloc=1.0, reserved=2.0)])

    assert summary["phases"]["loop"]["s_per_step"] is None
    assert summary["phases"]["loop"]["steps_measured"] == 0
    assert summary["grouped_speedup"] is None


def test_speedups_and_memory_deltas_are_ratios_of_the_measured_phases():
    phases = [
        _phase("loop", [99.0, 20.0, 20.0], alloc=10.0, reserved=13.0),
        _phase("grouped", [40.0, 5.0, 5.0], alloc=10.2, reserved=13.1),
        _phase("grouped+retain", [30.0, 4.0, 4.0], alloc=11.2, reserved=14.0),
        _phase("loop (repeat)", [20.0, 22.0, 22.0], alloc=10.0, reserved=13.0),
    ]
    summary = ab.summarize_phases(phases)

    assert summary["grouped_speedup"] == 4.0
    assert summary["retain_speedup"] == 1.25
    assert summary["drift"] == pytest.approx(20.0 / 22.0, abs=1e-3)
    assert summary["grouped_peak_reserved_delta_gb"] == 0.1
    assert summary["retain_peak_alloc_delta_gb"] == 1.0


def test_a_phase_that_never_ran_reports_none_and_does_not_crash():
    summary = ab.summarize_phases([_phase("loop", [10.0, 10.0], alloc=1.0, reserved=2.0)])

    assert summary["retain_speedup"] is None
    assert summary["retain_peak_reserved_delta_gb"] is None
