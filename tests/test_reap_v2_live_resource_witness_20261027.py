"""Bounded Windows resource witness adversarial tests; no simulated scores."""

from copy import deepcopy

import pytest

from chowder_batch import reap_v2_live_resource_witness_20261027 as witness


def sample(ram=10_000, commit=20_000, used=800, free=1200, ws=40000):
    return {"physical_and_commit": {"ram_available_bytes": ram,
                                    "commit_free_bytes": commit},
            "gpus": [{"index": 0, "used_mib": used, "free_mib": free}],
            "processes": [{"pid": 4, "role": "native_llama_server",
                           "working_set_bytes": ws}]}


def test_bounded_observations_are_not_actual_experiment_peak():
    a, b = sample(), sample(ram=8000, commit=17000, used=1400, free=600, ws=60000)
    result = witness.summarize([a, b])
    assert result["sampled_max_host_gpu_used_mib"] == 1400
    assert result["sampled_min_host_gpu_free_mib"] == 600
    assert result["sampled_max_native_llama_working_set_bytes"] == 60000
    assert result["min_available"] == {
        "ram_available_bytes": 8000, "commit_free_bytes": 17000,
    }
    assert result["isolated_model_peak_ram_or_vram_known"] is False
    assert result["full_frozen_serving_stability_proven"] is False
    assert result["gen0"]["status"] == "BLOCKED"


@pytest.mark.parametrize("count", [0, 17])
def test_empty_or_unbounded_sampling_is_refused(count):
    with pytest.raises(ValueError, match="bounded"):
        witness.summarize([sample()] * count)


def test_missing_gpu_is_refused():
    damaged = deepcopy(sample())
    damaged["gpus"] = []
    with pytest.raises(ValueError, match="GPU"):
        witness.summarize([damaged])


def test_readonly_real_windows_commit_counters():
    counters = witness.windows_commit()
    assert counters["ram_total_bytes"] > 40 * 1024**3
    assert counters["commit_limit_bytes"] > counters["commit_total_bytes"]
    assert counters["commit_free_bytes"] > 0


def test_readonly_real_native_gpu_census():
    gpu = witness.gpu_snapshot()
    assert gpu and gpu[0]["total_mib"] >= 16000
    assert gpu[0]["free_mib"] >= 0
    assert gpu[0]["wddm_unaccounted_mib"] >= 0


def test_wddm_reserved_gpu_memory_must_not_be_included_in_free_capacity(monkeypatch):
    from subprocess import CompletedProcess
    monkeypatch.setattr(witness.subprocess, "run", lambda *a, **k: CompletedProcess(
        [], 0, "0, RTX 5060 Ti, 16311, 12324, 3727, 89\n", ""))
    one = witness.gpu_snapshot()[0]
    assert one["wddm_unaccounted_mib"] == 260
    assert one["free_mib"] == 3727


def test_impossible_gpu_accounting_fails_closed(monkeypatch):
    from subprocess import CompletedProcess
    monkeypatch.setattr(witness.subprocess, "run", lambda *a, **k: CompletedProcess(
        [], 0, "0, RTX 5060 Ti, 16311, 15000, 2300, 89\n", ""))
    with pytest.raises(ValueError, match="impossible"):
        witness.gpu_snapshot()


def test_genuine_snapshot_does_not_launch_model():
    result = witness.summarize([witness.snapshot()])
    assert result["model_inference_launched"] is False
    assert result["sample_count"] == 1
