"""The hardware calibration report's units, and the statistic it reports.

Every figure ``chowder hardware-calibrate`` prints is GiB-based (bytes / 2**30)
and every rate is the *median* of the timed passes. The fields used to be named
``_gbps``/``_gb``, the decimal unit the repo's own convention reserves for 10**9,
so a reader mixing a model card's decimal GB with these readings overstated
capacity and throughput by ~7.4%. These tests pin the arithmetic (GiB, at the
median) and the names, because the payload an operator reads *is* these field
names -- ``to_dict`` is ``asdict`` (reported-metric audit R7).
"""

from __future__ import annotations

import dataclasses

import pytest

from chowder.calibration import (
    CudaTransferCalibration,
    HostMemoryCalibration,
    StorageCalibration,
    _median_gib_per_s,
    calibrate_hardware,
    calibrate_host_memory,
    calibrate_storage,
)


def test_the_median_helper_reports_gib_per_s_at_the_median_pass():
    assert _median_gib_per_s(1024**3, [2.0, 1.0, 3.0]) == 0.5


def test_a_rate_is_gib_and_not_decimal_gb():
    """10**9 bytes in one second is 0.93 GiB/s: the unit is pinned, not implied."""
    assert _median_gib_per_s(1000**3, [1.0]) == pytest.approx(0.9313225746154785)
    # No positive pass means no rate -- never a fabricated one.
    assert _median_gib_per_s(1024**3, [0.0]) == 0.0


def test_storage_calibration_uses_temp_file_and_cleans_up(tmp_path):
    result = calibrate_storage(tmp_path, sample_mib=1, passes=1)
    assert result.read_gib_per_s_median > 0
    assert result.write_gib_per_s_median > 0
    assert result.read_cache_sensitive
    assert result.durable_write
    assert not list(tmp_path.glob(".chowder-cal-*"))


def test_host_memory_calibration_reports_effective_copy_bandwidth():
    result = calibrate_host_memory(sample_mib=1, passes=2)
    assert result.copy_gib_per_s_median > 0


def test_combined_calibration_can_skip_cuda(tmp_path):
    result = calibrate_hardware(tmp_path, sample_mib=1, passes=1, include_cuda=False)
    assert result.storage is not None
    assert result.host_memory is not None
    assert result.cuda is None
    assert any("page-cache" in note for note in result.notes)


def test_the_reported_payload_names_gib_and_the_statistic(tmp_path):
    """The keys an operator reads are the field names, so they are pinned here."""
    payload = calibrate_hardware(
        tmp_path, sample_mib=1, passes=1, include_cuda=False
    ).to_dict()
    assert "read_gib_per_s_median" in payload["storage"]
    assert "copy_gib_per_s_median" in payload["host_memory"]
    for section in ("storage", "host_memory"):
        decimal = [
            key
            for key in payload[section]
            if key.endswith("_gbps") or key.endswith("_gb")
        ]
        assert decimal == [], decimal


def test_no_calibration_field_is_named_with_a_decimal_unit():
    """A ``_gb``/``_gbps`` field would be a second vocabulary to keep honest."""
    for cls in (StorageCalibration, HostMemoryCalibration, CudaTransferCalibration):
        names = [field.name for field in dataclasses.fields(cls)]
        decimal = [name for name in names if name.endswith("_gb") or name.endswith("_gbps")]
        assert decimal == [], (cls.__name__, decimal)
        assert [
            name
            for name in names
            if name.endswith("_gib") or name.endswith("_gib_per_s_median")
        ], names
