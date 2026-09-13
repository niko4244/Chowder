"""P6: headroom evidence, with the sampling caveat attached.

The evaluation leg's VRAM verdict was once decided from ``nvidia-smi`` -- which
measures the whole machine -- while the run's own footprint was never recorded.
The other half of that lesson is subtler: a *sampled* minimum free-memory figure
is not proof of an unobserved instantaneous minimum. These tests pin both the
measurement and its stated limits:

* an unavailable field is ``None``, never ``0``;
* a measured zero *is* a measurement (free memory of 0 bytes is the OOM case);
* a summary says how often it sampled and refuses to imply it saw everything.
"""

from __future__ import annotations

import pytest

from chowder.evaluators.vram import (
    MemorySampler,
    device_memory,
    host_memory,
    summarize_samples,
)


# ---------------------------------------------------------------------------
# probes
# ---------------------------------------------------------------------------


def test_host_memory_never_raises_and_never_fabricates_a_zero():
    report = host_memory()
    assert set(report) >= {"rss_bytes", "commit_bytes", "total_system_bytes", "source"}
    for field in ("rss_bytes", "commit_bytes", "total_system_bytes"):
        value = report[field]
        assert value is None or (isinstance(value, int) and value >= 0)


def test_device_memory_on_a_non_cuda_device_is_unknown_not_zero():
    report = device_memory("cpu")
    assert report["free_bytes"] is None
    assert report["total_bytes"] is None
    assert report["used_bytes"] is None


def test_device_memory_reports_a_real_cuda_reading_when_available(monkeypatch):
    import sys
    import types

    fake = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            mem_get_info=lambda index: (3 * 1024**3, 8 * 1024**3),
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake)

    report = device_memory("cuda:0")
    assert report["free_bytes"] == 3 * 1024**3
    assert report["total_bytes"] == 8 * 1024**3
    assert report["used_bytes"] == 5 * 1024**3


def test_a_failing_device_query_is_unknown_not_zero(monkeypatch):
    import sys
    import types

    def explode(index):
        raise RuntimeError("NVML is having a day")

    fake = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: True, mem_get_info=explode)
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    report = device_memory("cuda:0")
    assert report["free_bytes"] is None
    assert report["used_bytes"] is None


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------


def test_summary_takes_the_extremes_across_samples():
    report = summarize_samples(
        [
            {"free_bytes": 100, "used_bytes": 10, "rss_bytes": 50, "commit_bytes": 60},
            {"free_bytes": 40, "used_bytes": 90, "rss_bytes": 30, "commit_bytes": 80},
            {"free_bytes": 70, "used_bytes": 40, "rss_bytes": 70, "commit_bytes": 20},
        ],
        cadence_seconds=0.5,
        span_seconds=1.5,
    )

    assert report["samples"] == 3
    assert report["min_free_device_bytes"] == 40
    assert report["max_used_device_bytes"] == 90
    assert report["min_host_rss_bytes"] == 30
    assert report["max_host_commit_bytes"] == 80
    assert report["cadence_seconds"] == 0.5
    assert report["unavailable_fields"] == []
    # The caveat is part of the evidence, not folklore.
    assert "sampl" in report["note"].lower()


def test_a_measured_zero_free_is_a_reading_not_a_missing_value():
    report = summarize_samples(
        [{"free_bytes": 0, "used_bytes": 100}], cadence_seconds=0.5, span_seconds=0.5
    )
    assert report["min_free_device_bytes"] == 0
    assert "free_device_bytes" not in report["unavailable_fields"]


def test_fields_never_observed_are_listed_as_unavailable():
    report = summarize_samples(
        [{"free_bytes": 100}], cadence_seconds=0.5, span_seconds=0.5
    )
    assert report["min_free_device_bytes"] == 100
    assert report["min_host_rss_bytes"] is None
    assert set(report["unavailable_fields"]) >= {
        "used_device_bytes",
        "host_rss_bytes",
        "host_commit_bytes",
    }


def test_no_samples_reports_unknown_rather_than_zero():
    report = summarize_samples([], cadence_seconds=0.5, span_seconds=0.0)
    assert report["samples"] == 0
    assert report["min_free_device_bytes"] is None
    assert report["max_used_device_bytes"] is None
    assert len(report["unavailable_fields"]) == 4


# ---------------------------------------------------------------------------
# the sampler
# ---------------------------------------------------------------------------


def test_sampler_records_the_minimum_free_bytes_it_actually_saw():
    readings = iter(
        [
            {"free_bytes": 900, "used_bytes": 100},
            {"free_bytes": 300, "used_bytes": 700},
            {"free_bytes": 600, "used_bytes": 400},
        ]
    )
    sampler = MemorySampler(
        device_name="cuda:0",
        interval_seconds=0.01,
        device_probe=lambda name: next(readings, {"free_bytes": None, "used_bytes": None}),
        host_probe=lambda: {"rss_bytes": 1234, "commit_bytes": None, "total_system_bytes": None},
    )
    sampler.sample_now()
    sampler.start()
    report = sampler.stop()

    assert report["samples"] >= 1
    assert report["min_free_device_bytes"] is not None
    assert report["min_free_device_bytes"] <= 900
    assert report["cadence_seconds"] == 0.01
    assert report["device"] == "cuda:0"
    # A host field the probe never produced stays unknown.
    assert "host_commit_bytes" in report["unavailable_fields"]


def test_sampler_stop_without_start_is_an_empty_report_not_a_crash():
    sampler = MemorySampler(device_name="cuda:0", device_probe=lambda name: None)
    report = sampler.stop()
    assert report["samples"] == 0
    assert report["min_free_device_bytes"] is None


def test_a_failing_probe_never_kills_the_run():
    def explode(name):
        raise RuntimeError("driver error")

    sampler = MemorySampler(
        device_name="cuda:0",
        interval_seconds=0.01,
        device_probe=explode,
        host_probe=lambda: {"rss_bytes": None, "commit_bytes": None, "total_system_bytes": None},
    )
    sampler.start()
    report = sampler.stop()
    assert report["min_free_device_bytes"] is None
    assert "free_device_bytes" in report["unavailable_fields"]


def test_sampler_is_usable_as_a_context_manager():
    sampler = MemorySampler(
        device_name="cpu",
        interval_seconds=0.01,
        device_probe=lambda name: {"free_bytes": None, "used_bytes": None},
        host_probe=host_memory,
    )
    with sampler as handle:
        assert handle is sampler
    assert sampler.report is not None
    assert sampler.report["samples"] >= 1


@pytest.mark.parametrize("interval", [0.0, -1.0])
def test_a_nonpositive_interval_is_refused(interval):
    with pytest.raises(ValueError):
        MemorySampler(device_name="cpu", interval_seconds=interval)
