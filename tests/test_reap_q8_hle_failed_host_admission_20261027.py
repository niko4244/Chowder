"""No model execution, Kaggle API calls or GPU resource changes."""

import json
from pathlib import Path

import pytest

from chowder_batch import reap_q8_hle_failed_host_admission_20261027 as admission
from chowder_batch import reap_q8_hle_missing29_resume_20261027 as frozen


def adequate():
    return {"cgroup_unused_gib": 48.0, "host_physical_available_gib": 48.0,
            "gpu_free_mib": [12288, 12288], "disk_free_gib": 8.0}


def test_original_failed_host_never_passes_same_resource_capacity():
    r = admission.admit_measured_snapshot({
        "cgroup_unused_gib": 28.806,
        "host_physical_available_gib": 29.77,
        "gpu_free_mib": [14912, 14912], "disk_free_gib": 50.0})
    assert r["status"] == "REFUSED_KNOWN_INSUFFICIENT_HLE_RUNNER"
    assert not r["prospective_gates"]["cgroup_unused_48_gib"]
    assert r["hles_scored"] == 0
    assert r["gen0_status"] == "BLOCKED"


def test_even_larger_host_is_only_prestart_candidate():
    record = admission.admit_measured_snapshot(adequate())
    assert record["status"] == "MEASURED_PRESTART_FLOORS_SATISFIED_NOT_MODEL_ADMITTED"
    assert record["full_model_postload_resource_pass"] is False
    assert record["forty_one_cuda_layers_proven"] is False
    assert record["new_job_submitted"] is False
    assert record["release_approved"] is False


@pytest.mark.parametrize("field,unavailable", [
    ("cgroup_unused_gib", 47.999),
    ("host_physical_available_gib", 47.999),
    ("gpu_free_mib", [12287, 14912]),
    ("gpu_free_mib", [14912, 12287]),
    ("disk_free_gib", 7.999),
])
def test_every_resource_floor_is_independently_mandatory(field, unavailable):
    record = adequate()
    record[field] = unavailable
    assert admission.admit_measured_snapshot(record)["status"].startswith("REFUSED")


@pytest.mark.parametrize("field,bad", [
    ("cgroup_unused_gib", float("nan")),
    ("cgroup_unused_gib", float("inf")),
    ("host_physical_available_gib", True),
    ("gpu_free_mib", [12300]),
    ("gpu_free_mib", [12300, "12300"]),
    ("gpu_free_mib", [12300, True]),
    ("disk_free_gib", -1),
])
def test_missing_forged_or_nonfinite_resource_input_is_refused(field, bad):
    record = adequate()
    record[field] = bad
    with pytest.raises(ValueError):
        admission.admit_measured_snapshot(record)


def test_original_private_cgroup_abort_is_exactly_source_bound_if_available():
    if not admission.ROOT.is_dir():
        pytest.skip("private original Kaggle output is not present on this host")
    result = admission.failed_original_evidence()
    assert result["original_prior_q8_rows"] == 99
    assert result["new_q8_valid_rows"] == 0
    assert result["initial_cgroup_unused_gib"] == 28.806
    assert result["original_failed_receipt_sha256"] == admission.RECEIPT_SHA
    assert result["original_native_log_sha256"] == admission.LOG_SHA
    assert result["release_approved"] is False


def test_original_receipt_hash_not_replaced_by_synthetic_pass(tmp_path, monkeypatch):
    p = tmp_path / frozen.NEW_RECEIPT
    p.write_text(json.dumps({"status": "COMPLETE", "new_rows": 29}))
    (tmp_path / frozen.NEW_LOG).write_bytes(b"fake log")
    with pytest.raises(ValueError):
        admission.failed_original_evidence(tmp_path)


def test_native_resource_census_does_not_pretend_to_run_on_windows(monkeypatch):
    monkeypatch.setattr(admission.sys, "platform", "win32")
    with pytest.raises(ValueError, match="Linux"):
        admission.measure_original_linux_host()


def test_no_generated_code_submission_or_model_execution():
    text = Path(admission.__file__).read_text(encoding="utf-8")
    for banned in ("kaggle kernels push", "kaggle kernels output",
                   "subprocess.Popen", "torch.cuda", "shell=True",
                   "requests.post(", "exec("):
        assert banned not in text
