"""No-GPU regression for independently sampled Windows WDDM memory counters."""

import copy

import pytest

from chowder_batch.wddm_process_memory import WddmWitnessError, summarize_samples


def _sample(pid=26096, shared=1024, dedicated=2048):
    base = f"\\\\nikolas\\gpu process memory(pid_{pid}_luid_0x0000_phys_0)"
    return {"timestamp": "2026-10-23T04:00:00.000-05:00", "counters": [
        {"path": base + "\\shared usage", "bytes": shared},
        {"path": base + "\\dedicated usage", "bytes": dedicated},
    ]}


def test_passive_counter_summary_preserves_pid_and_disclaims_training_gate():
    witness = summarize_samples([_sample(), _sample(shared=8192, dedicated=4096)],
                                pid=26096, requested_samples=2)
    assert witness["maximum_shared_usage_bytes"] == 8192
    assert witness["maximum_dedicated_usage_bytes"] == 4096
    assert witness["observed_samples"] == 2
    assert witness["qualification_status"] == "OBSERVATION_ONLY_NOT_A_WDDM_TRAINING_GATE"
    assert "does not certify" in witness["limitations"]
    assert "underrepresent CUDA" in witness["limitations"]


@pytest.mark.parametrize("damage", ["partial", "foreign", "negative", "nan", "duplicate",
                                    "missing_shared", "bad_path", "not_sample", "fractional"])
def test_passive_counter_refuses_invalid_or_incomplete_samples(damage):
    sample = _sample()
    raw = [sample]
    if damage == "partial":
        raw = []
    elif damage == "foreign":
        sample["counters"][0]["path"] = _sample(222)["counters"][0]["path"]
    elif damage == "negative":
        sample["counters"][0]["bytes"] = -1
    elif damage == "nan":
        sample["counters"][0]["bytes"] = float("nan")
    elif damage == "fractional":
        sample["counters"][0]["bytes"] = 7.3
    elif damage == "duplicate":
        sample["counters"].append(copy.deepcopy(sample["counters"][0]))
    elif damage == "missing_shared":
        sample["counters"].pop(0)
    elif damage == "bad_path":
        sample["counters"][0]["path"] = "unknown"
    elif damage == "not_sample":
        raw = ["no reading"]
    with pytest.raises(WddmWitnessError):
        summarize_samples(raw, pid=26096, requested_samples=1)


def test_no_phantom_wddm_spill_verdict_even_when_shared_zero():
    result = summarize_samples([_sample(shared=0)], pid=26096, requested_samples=1)
    assert result["maximum_shared_usage_bytes"] == 0
    assert "NOT_A_WDDM_TRAINING_GATE" in result["qualification_status"]
