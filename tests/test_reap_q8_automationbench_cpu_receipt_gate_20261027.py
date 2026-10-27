"""Synthetic-only adversarial checks for the original Q8 CPU receipt gate."""

import json
from pathlib import Path

import pytest

from chowder_batch import reap_q8_automationbench_cpu_receipt_gate_20261027 as gate


def valid_receipt() -> dict:
    common = {
        "package_count": 108,
        "locked_download_bytes": 143_537_351,
        "official_commit": gate.EXPECTED_SOURCE_SHA,
        "gen0_status": "BLOCKED",
        "weights_loaded": False,
        "gpu_requested": False,
    }
    return {
        "schema": "reap-q8-automationbench-real-cpu600-qual.v1",
        "status": "OFFICIAL_SOURCE_LOCKED_600_CPU_BOOTSTRAP_ADMITTED_NO_MODEL_SCORE",
        "gpu_enabled": False,
        "internet_enabled": False,
        "weights_loaded": False,
        "model_score_claimed": False,
        "gen0_status": "BLOCKED",
        "inspect": {**common, "status": "CPU_OFFLINE_SOURCES_AND_WHEELS_VERIFIED_NOT_INSTALLED"},
        "install": {**common,
                    "status": "CPU_OFFLINE_600_SOURCE_LOADER_PASS_NOT_A_MODEL_SCORE",
                    "scorer_source_mutated": False,
                    "model_inference_started": False,
                    "site_packages": "/kaggle/working/reap-q8-official600-cpu-private-target/packages",
                    "official_source": "/kaggle/working/reap-q8-official600-cpu-private-target/official-source"},
    }


def test_actual_private_frozen_sources_and_both_distinct_q4_pilots():
    census = gate.audit_source_and_pilots()
    assert census["source_tar_sha256"] == "0d9b2b3ea8d54250bf3f14a0ce1d0b7d766d445047cdd730c88575d19b57f278"
    assert census["wheel_tar_sha256"] == gate.WHEEL_TAR_SHA
    arms = census["matched_profiles"]
    assert arms["six"]["exact_task_count"] == 6
    assert arms["guard2"]["exact_task_count"] == 2
    assert set(arms["six"]["exact_task_names"]).isdisjoint(arms["guard2"]["exact_task_names"])
    assert arms["six"]["q4_public_original_passes"] == 0
    assert arms["guard2"]["q4_public_original_passes"] == 0
    assert arms["six"]["serving_flags"] != arms["guard2"]["serving_flags"]
    assert "BLOCKED" in arms["six"]["status"]
    assert "BLOCKED" in arms["guard2"]["status"]


def test_no_receipt_is_not_cpu_admission():
    decision = gate.admission(None)
    assert decision["status"] == "BLOCKED_ORIGINAL_CPU_RECEIPT_NOT_PROVIDED"
    assert decision["cpu_proof"] is None
    assert decision["model_or_gpu_launched"] is False


def test_synthetic_receipt_has_only_provisional_cpu_admission():
    result = gate._validate_cpu_receipt(valid_receipt())
    assert result["status"] == "ORIGINAL_CPU_RECEIPT_SOURCE_LOCKED_PROVISIONAL_ADMISSION"
    assert result["original_600_source_verified_by_cpu_runner"] is True
    assert result["independent_600_model_score"] is None
    assert result["actual_gpu_evidence"] is False


@pytest.mark.parametrize("key,bad", [
    ("status", "BLOCKED_CPU_BOOTSTRAP_NOT_ADMITTED"),
    ("schema", "another-authorization"),
    ("gpu_enabled", True),
    ("internet_enabled", True),
    ("weights_loaded", True),
    ("model_score_claimed", True),
    ("gen0_status", "PASS"),
])
def test_not_original_cpu_success_rejected(key, bad):
    r = valid_receipt()
    r[key] = bad
    with pytest.raises(ValueError, match="CPU receipt unsuccessful or changed"):
        gate._validate_cpu_receipt(r)


@pytest.mark.parametrize("part,key,bad", [
    ("inspect", "status", "SKIPPED"),
    ("install", "status", "INSTALLED_NOT_SOURCE_SCORER_PROVEN"),
    ("install", "package_count", 107),
    ("inspect", "locked_download_bytes", 0),
    ("install", "official_commit", "0" * 40),
    ("inspect", "gpu_requested", True),
    ("install", "weights_loaded", True),
])
def test_receipt_source_and_wheel_omissions_are_fail_closed(part, key, bad):
    r = valid_receipt()
    r[part][key] = bad
    with pytest.raises(ValueError):
        gate._validate_cpu_receipt(r)


def test_cpu_private_isolation_and_scorer_immutability_required():
    for field, value in (("scorer_source_mutated", True),
                         ("model_inference_started", True),
                         ("site_packages", "/usr/local/lib/python3.13/site-packages"),
                         ("official_source", "/tmp/not-original-source")):
        r = valid_receipt()
        r["install"][field] = value
        with pytest.raises(ValueError):
            gate._validate_cpu_receipt(r)


def test_forged_success_metric_is_never_an_original_cpu_score():
    r = valid_receipt()
    r["model_score"] = 1.0
    with pytest.raises(ValueError, match="cannot assert original model quality"):
        gate._validate_cpu_receipt(r)


def test_actual_filename_and_private_root_required(tmp_path: Path):
    private = tmp_path / "original-private"
    private.mkdir()
    bad = private / "fake.json"
    bad.write_text(json.dumps(valid_receipt()), encoding="utf-8")
    with pytest.raises(ValueError, match="authentic named"):
        gate._private_receipt(bad, private)
    named = private / gate.EXPECTED_CPU_FILE_NAME
    named.write_text(json.dumps(valid_receipt()), encoding="utf-8")
    evidence = gate._private_receipt(named, private)
    assert len(evidence["sha256"]) == 64
    (tmp_path / "wrong-private").mkdir()
    with pytest.raises(ValueError, match="within private evidence root"):
        gate._private_receipt(named, tmp_path / "wrong-private")


def test_no_kaggle_api_gpu_execution_or_frozen_file_modifications():
    raw = Path(gate.__file__).read_text(encoding="utf-8")
    assert "kaggle kernels output" not in raw
    assert "kaggle kernels push" not in raw
    assert "nvidia-smi" not in raw
    assert "subprocess.Popen" not in raw
    assert "shell=True" not in raw
    assert "--no-index" not in raw
