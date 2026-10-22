"""Frozen amendment-4 branch-order regression (all CPU, no real model)."""

import hashlib
import json

import pytest

from chowder_batch.reap_gates import GateError
from chowder_batch.reap_select import REQUIRED, evaluate, select


def arm(**statuses):
    return {
        "gates": {k: {"status": statuses.get(k, "PASS")} for k in REQUIRED},
    }


def test_v1_not_assumed_pass_if_speed_unknown():
    d = select({"reap128": arm(speed="UNKNOWN")})
    assert d["status"] == "BLOCKED" and d["chosen"] is None


def test_v1_passes_and_v2_not_declared():
    d = select({"reap128": arm()})
    assert d["status"] == "SELECTED" and d["chosen"] == "reap128"


def test_v1_fail_gate1_needs_both_fallbacks_even_if_160_passes():
    d = select({"reap128": arm(gate1="FAIL", speed="NOT_EVALUATED"),
                "reap160": arm()})
    assert d["status"] == "BLOCKED"
    assert "both REAP128_v2 and REAP160" in d["reason"]


def test_v1_fail_gate1_needs_160_scored_even_if_v2_passes():
    d = select({"reap128": arm(gate1="FAIL", speed="NOT_EVALUATED"),
                "reap128_v2": arm()})
    assert d["status"] == "BLOCKED"


def test_fallback_v2_precedes_160_with_full_gate_sets():
    d = select({"reap128": arm(gate1="FAIL", speed="NOT_EVALUATED"),
                "reap128_v2": arm(), "reap160": arm()})
    assert d["status"] == "SELECTED" and d["chosen"] == "reap128_v2"


def test_fallback_160_used_only_after_v2_failed():
    d = select({"reap128": arm(gate1="FAIL", speed="NOT_EVALUATED"),
                "reap128_v2": arm(gate2="FAIL"), "reap160": arm()})
    assert d["status"] == "SELECTED" and d["chosen"] == "reap160"


def test_third_fallback_is_conditional():
    current = {"reap128": arm(gate1="FAIL", speed="NOT_EVALUATED"),
               "reap128_v2": arm(gate1="FAIL"),
               "reap160": arm(speed="FAIL")}
    assert select(current)["status"] == "BLOCKED"
    assert select({**current, "reap160_v2": arm()})["chosen"] == "reap160_v2"
    assert select({**current, "reap160_v2": arm(gate1="FAIL")})["status"] == "FAILED"


def test_v2_normal_replacement_requires_both_strict_noninferiority_bounds():
    current = {"reap128": arm(), "reap128_v2": arm()}
    assert select(current)["status"] == "BLOCKED"
    assert select(current, {"gsm8k": -0.019, "humaneval": -0.019})["chosen"] == "reap128_v2"
    assert select(current, {"gsm8k": -0.02, "humaneval": -0.019})["chosen"] == "reap128"
    assert select(current, {"gsm8k": -0.019, "humaneval": -0.02})["chosen"] == "reap128"


def test_v1_coding_regression_does_not_authorize_fallback():
    d = select({"reap128": arm(gate2="FAIL"), "reap128_v2": arm(), "reap160": arm()})
    assert d["status"] == "FAILED" and d["chosen"] is None


def test_d0_fail_denies_every_arm():
    d = select({"reap128": arm(d0="FAIL", gate1="FAIL"),
                "reap128_v2": arm(), "reap160": arm()})
    assert d["status"] == "FAILED" and d["chosen"] is None


def _attest(root, arm, suite, n, *, fail=0, speed=False):
    directory = root / arm
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{suite}.jsonl"
    rows = [
        {"id": f"{suite}-{i}", "correct": i >= fail, "terminated": True,
         "trigram_ratio": 0.9, "predicted_per_second": 42.0 if speed else None}
        for i in range(n)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    entry = {"n": n, "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    summary = directory / "summary.json"
    values = json.loads(summary.read_text(encoding="utf-8")) if summary.exists() else {}
    values[suite] = entry
    summary.write_text(json.dumps(values), encoding="utf-8")
    if speed:
        (directory / "serve.json").write_text(json.dumps({"experts_offloaded": 0}))
    return path


def _evidence(root):
    for arm, failures in (("dense", 0), ("reap128", 90), ("random128", 1000)):
        _attest(root, arm, "gsm8k", 1319, fail=failures)
    for arm, failures in (("dense", 0), ("reap128", 0)):
        _attest(root, arm, "humaneval", 164, fail=failures)
    _attest(root, "dense_q4", "gsm8k", 200, speed=True)
    _attest(root, "reap128_q4", "gsm8k", 200, speed=True)
    observation = root / "observation.json"
    observation.write_text(json.dumps({"layers": [{"layer": i} for i in range(40)],
                                       "calibration_ppl": 10.646096}), encoding="utf-8")
    return observation


def test_full_synthetic_raw_evidence_blocks_missing_mandatory_v2(tmp_path):
    obs = _evidence(tmp_path)
    result = evaluate(tmp_path, obs)
    assert result["gates"]["reap128"]["gates"]["gate1"]["status"] == "FAIL"
    assert result["decision"]["status"] == "BLOCKED"
    assert "both REAP128_v2 and REAP160" in result["decision"]["reason"]
    assert result["input_sha256"][str(tmp_path / "dense" / "gsm8k.jsonl")]
    assert result["release_approved"] is False


def test_sha_tampered_raw_baseline_refuses_before_selection(tmp_path):
    obs = _evidence(tmp_path)
    with (tmp_path / "dense" / "gsm8k.jsonl").open("a", encoding="utf-8") as file:
        file.write("{}\n")
    with pytest.raises(GateError, match="SHA256"):
        evaluate(tmp_path, obs)


def test_missing_quality_arm_is_not_silently_equivalent(tmp_path):
    obs = _evidence(tmp_path)
    _attest(tmp_path, "reap128_v2", "gsm8k", 1319)
    with pytest.raises(GateError, match="missing full SHA-attested"):
        evaluate(tmp_path, obs)
