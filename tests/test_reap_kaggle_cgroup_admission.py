"""Pure synthetic adversarial external Q4 memory admission tests, CPU only."""

import math
import hashlib
import json

import pytest

from chowder_batch import reap_kaggle_cgroup_admission as admission
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


def _dense_evidence():
    return {
        "schema": "reap-v2-dense-q4-kaggle-first200-provisional.v1",
        "status": "REFUSED_OR_ABORTED",
        "error": "conservative 10 GiB cgroup stop before original 8 GiB runtime floor",
        "flags": ["--fit", "on", "-np", "4", "-c", "16384", "--jinja"],
        "data": {"dense_q4_sha256": "2c6d9466129e8802839566b35728e2506aaf1e8705cd137009e2f08feb9dd049",
                 "requested_cohort_n": 200},
        "initial": {"cgroup_unused_gib": 29.444},
        "samples": [{"cgroup_unused_gib": 29.444}, {"cgroup_unused_gib": 9.992}],
    }


def _run_dense(monkeypatch, **kwargs):
    payload=json.dumps(_dense_evidence(),sort_keys=True).encode()
    monkeypatch.setattr(admission,"ORIGINAL_DENSE_ABORT_RECEIPT_SHA",hashlib.sha256(payload).hexdigest())
    values={"current_cgroup_unused_gib":29.444,"physical_free_gib":35.0,
            "gpu_free_mib":[14912,14912],"disk_free_gib":50.0}
    values.update(kwargs)
    return admission.analyze_original_dense_replay(payload,**values)


def test_proven_dense_29_gib_cgroup_initial_load_must_be_refused(monkeypatch):
    result=_run_dense(monkeypatch)
    assert result["status"]=="REFUSED"
    assert result["original_dense_200_speed_certified"] is False
    assert result["observed_incomplete_load_consumed_gib"]==pytest.approx(19.452)
    assert result["minimum_provisional_next_initial_headroom_gib"]==pytest.approx(33.452)


def test_extra_resources_admit_only_fresh_provisional_attempt_not_gate_pass(monkeypatch):
    result=_run_dense(monkeypatch,current_cgroup_unused_gib=45.0)
    assert result["status"]=="MAY_ATTEMPT_NEW_PROVISIONAL_DENSE_LOAD_ONLY"
    assert result["original_dense_200_speed_certified"] is False
    assert result["original_windows_commit_attested"] is False


@pytest.mark.parametrize("change",[
    {"current_cgroup_unused_gib":float("nan")},
    {"current_cgroup_unused_gib":33.0},
    {"physical_free_gib":6.9},
    {"gpu_free_mib":[14912,9000]},
    {"disk_free_gib":7.9},
])
def test_dense_new_provisional_admission_rejects_unsupported_or_insufficient_host(monkeypatch,change):
    if math.isnan(change.get("current_cgroup_unused_gib",0)):
        with pytest.raises(ValueError):_run_dense(monkeypatch,**change)
    else:
        assert _run_dense(monkeypatch,**change)["status"]=="REFUSED"


def test_original_dense_provenance_must_be_exact_bytes(monkeypatch):
    payload=json.dumps(_dense_evidence(),sort_keys=True).encode()
    monkeypatch.setattr(admission,"ORIGINAL_DENSE_ABORT_RECEIPT_SHA",hashlib.sha256(payload).hexdigest())
    with pytest.raises(ValueError):
        admission.analyze_original_dense_replay(payload+b" ",current_cgroup_unused_gib=50.0,
            physical_free_gib=60.0,gpu_free_mib=[14912,14912],disk_free_gib=50.0)
