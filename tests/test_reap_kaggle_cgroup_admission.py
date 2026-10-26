"""Pure synthetic adversarial external Q4 memory admission tests, CPU only."""

import math

import pytest

from chowder_batch.reap_kaggle_cgroup_admission import GIB, analyze_cgroup


def _sample(**changes):
    values = {
        "limit": str(40 * GIB),
        "current": str(10 * GIB),
        "peak": str(11 * GIB),
        "memory_stat": f"file {7 * GIB}\nanon {2 * GIB}\n",
        "memory_events": "oom 0\noom_kill 0\n",
        "physical_free_gib": 30.0,
        "gpu_free_mib": [14912, 14912],
        "disk_free_gib": 19.0,
    }
    values.update(changes)
    return analyze_cgroup(**values)


def test_no_false_official_qualification_on_adequate_provisional_host():
    result = _sample()
    assert result["status"] == "MAY_ATTEMPT_PROVISIONAL_SMOKE_ONLY"
    assert result["registered_windows_commit_attested"] is False
    assert result["scientific_selection"] == "UNKNOWN"
    assert result["no_process_launched"] is True


def test_clean_cache_is_not_falsely_added_to_free_commit():
    result = _sample(current=str(22 * GIB), peak=str(23 * GIB),
                     memory_stat=f"file {15 * GIB}\nanon {6 * GIB}\n")
    assert result["status"] == "REFUSED"
    assert result["cgroup_unused_gib"] == pytest.approx(18.0)
    assert result["file_cache_gib_descriptive_not_guaranteed_reclaimable"] == pytest.approx(15.0)


@pytest.mark.parametrize("mutation", [
    {"limit": "max"}, {"limit": "0"}, {"current": "-2"},
    {"current": str(41 * GIB)}, {"peak": "1"},
    {"memory_stat": "file 1\nfile 1\nanon 1\n"},
    {"memory_stat": "file -1\nanon 1\n"},
    {"memory_stat": "anon 3\n"},
    {"memory_events": "oom 0\n"},
    {"physical_free_gib": math.nan}, {"physical_free_gib": True},
    {"gpu_free_mib": [14912, math.nan]},
    {"disk_free_gib": math.inf},
])
def test_missing_corrupt_inconsistent_or_nonfinite_resource_evidence_refused(mutation):
    with pytest.raises(ValueError):
        _sample(**mutation)


@pytest.mark.parametrize("change", [
    {"physical_free_gib": 6.9}, {"gpu_free_mib": [14912, 12000]},
    {"disk_free_gib": 7.9},
    {"memory_events": "oom 1\noom_kill 1\n"},
])
def test_each_protected_runtime_floor_and_prior_oom_fail_closed(change):
    result = _sample(**change)
    assert result["status"] == "REFUSED"


def test_less_than_eight_gib_cgroup_unused_explicitly_refused():
    result = _sample(current=str(33 * GIB), peak=str(33 * GIB),
                     memory_stat=f"file {20 * GIB}\nanon {12 * GIB}\n")
    assert result["status"] == "REFUSED"
    assert any("8 GiB" in reason for reason in result["reasons"])
