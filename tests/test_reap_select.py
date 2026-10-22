"""Frozen amendment-4 branch-order regression (all CPU, no real model)."""

from chowder_batch.reap_select import REQUIRED, select


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
