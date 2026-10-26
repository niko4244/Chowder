"""Synthetic, CPU-only challenge of evidence-derived 200-item Q4 admission."""

import hashlib
import json
import math

import pytest

from chowder_batch import reap_q4_resource_admission as admission


def _source(tmp_path):
    record = {
        "schema": "reap128-v2-q4-registered-200-supervisor.v1",
        "status": "ABORT_UNCHANGED_RUNTIME_MEMORY_FLOOR",
        "command": ["python", "-B", "reap_serve.py", "--gguf",
                    "G:/reap128-v2-q4_k_m-20261025.gguf", "--arm",
                    "reap128_v2_q4", "--out", "C:/isolated", "--",
                    "--port", "8790", "--arm", "reap128_v2_q4",
                    "--suites", "gsm8k", "--limit", "200", "--parallel", "4"],
        "initial": {"gpu_free_mib": 13993, "ram_available_gib": 21.41,
                    "commit_available_gib": 28.84},
        "memory_floors_gib": {"ram": 7, "commit": 8},
        "samples": [
            {"s": 0, "ram_available_gib": 21.39, "commit_available_gib": 28.84},
            {"s": 98.4, "ram_available_gib": 6.98, "commit_available_gib": 10.41},
        ],
    }
    p = tmp_path / "abort.json"
    p.write_text(json.dumps(record))
    return p


def test_red_historical_abort_refuses_same_host_resource_level(tmp_path):
    path = _source(tmp_path)
    history = admission.validate_aborted_speed_history(
        path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    )
    assert math.isclose(history["measured_ram_drop_gib"], 14.43)
    assert math.isclose(history["measured_commit_drop_gib"], 18.43)
    gate = admission.admit_host(
        history, ram_available_gib=21.41,
        commit_available_gib=28.84, gpu_free_mib=13993,
        disk_free_gib=37, ram_headroom_gib=2, commit_headroom_gib=2,
    )
    assert gate["status"] == "REFUSED"
    assert gate["minimum_ram_gib"] == pytest.approx(23.43)
    assert gate["minimum_commit_gib"] == pytest.approx(28.43)
    assert any("RAM" in issue for issue in gate["reasons"])


def test_trend_calculation_does_not_confuse_last_sample_with_minimum(tmp_path):
    p = _source(tmp_path)
    d = json.loads(p.read_text())
    d["samples"].insert(1, {"s": 42, "ram_available_gib": 5.5,
                            "commit_available_gib": 9.5})
    p.write_text(json.dumps(d))
    history = admission.validate_aborted_speed_history(
        p, expected_sha256=hashlib.sha256(p.read_bytes()).hexdigest()
    )
    assert history["measured_ram_drop_gib"] == pytest.approx(15.91)
    assert history["measured_commit_drop_gib"] == pytest.approx(19.34)


@pytest.mark.parametrize("mutation", ["sha", "floor", "status", "samples", "command", "gpu", "nan"])
def test_red_mutated_or_untrustworthy_abort_evidence_fails_closed(tmp_path, mutation):
    p = _source(tmp_path)
    d = json.loads(p.read_text())
    old_sha = hashlib.sha256(p.read_bytes()).hexdigest()
    if mutation == "sha":
        expected = "0" * 64
    else:
        expected = None
        if mutation == "floor":
            d["memory_floors_gib"]["ram"] = 0
        elif mutation == "status":
            d["status"] = "CHILD_EXITED"
        elif mutation == "samples":
            d["samples"] = []
        elif mutation == "command":
            d["command"] = ["python", "reap_eval.py", "--limit", "200"]
        elif mutation == "gpu":
            d["initial"]["gpu_free_mib"] = 50
        elif mutation == "nan":
            d["samples"][0]["ram_available_gib"] = math.nan
        p.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        admission.validate_aborted_speed_history(
            p, expected_sha256=expected or hashlib.sha256(p.read_bytes()).hexdigest()
        )
    assert old_sha


def test_red_fresh_resource_headroom_does_not_imply_scientific_pass(tmp_path):
    p = _source(tmp_path)
    d = admission.validate_aborted_speed_history(
        p, expected_sha256=hashlib.sha256(p.read_bytes()).hexdigest()
    )
    decision = admission.admit_host(
        d, ram_available_gib=30, commit_available_gib=35,
        gpu_free_mib=15000, disk_free_gib=37,
    )
    assert decision["status"] == "ADMITTED_TO_ATTEMPT_NOT_SCIENTIFICALLY_QUALIFIED"
    assert decision["quality_gate"] == "UNKNOWN"
    assert decision["speed_gate"] == "UNKNOWN"


@pytest.mark.parametrize("ram,commit,gpu,disk", [
    (30, 35, 12000, 37), (30, 35, 15000, 7),
    (30, 27, 15000, 37), (6.9, 35, 15000, 37),
    (math.nan, 35, 15000, 37), (True, 35, 15000, 37),
])
def test_red_fail_closed_resource_dimensions(tmp_path, ram, commit, gpu, disk):
    p = _source(tmp_path)
    d = admission.validate_aborted_speed_history(
        p, expected_sha256=hashlib.sha256(p.read_bytes()).hexdigest()
    )
    result = admission.admit_host(
        d, ram_available_gib=ram, commit_available_gib=commit,
        gpu_free_mib=gpu, disk_free_gib=disk,
    )
    assert result["status"] == "REFUSED"
