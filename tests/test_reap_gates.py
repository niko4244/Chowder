import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "chowder_batch"))
import reap_gates as rg  # noqa: E402


def write_arm(base, arm, suite, correct, terminated=None, trigram=None, pps=None, serve=None):
    d = Path(base) / arm
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, c in enumerate(correct):
        row = {"id": f"{suite}-{i}", "correct": bool(c),
               "terminated": True if terminated is None else bool(terminated[i]),
               "trigram_ratio": 0.9 if trigram is None else trigram[i],
               "completion_tokens": 10, "seconds": 1.0, "text": "x"}
        if pps is not None:
            row["predicted_per_second"] = pps[i]
        rows.append(row)
    (d / f"{suite}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    if serve is not None:
        (d / "serve.json").write_text(json.dumps(serve))


def build(tmp, n=1319, he_n=164, arm_he=None, arm_gsm=None, ctrl=None, **kw):
    ones = [1] * n
    write_arm(tmp, "dense", "gsm8k", ones)
    write_arm(tmp, "reap128", "gsm8k", arm_gsm if arm_gsm is not None else ones, **kw)
    write_arm(tmp, "random128", "gsm8k", ctrl if ctrl is not None else [0] * n)
    write_arm(tmp, "dense", "humaneval", [1] * he_n)
    write_arm(tmp, "reap128", "humaneval", arm_he if arm_he is not None else [1] * he_n)
    (tmp / "observation.json").write_text(json.dumps({
        "layers": [{"layer": i} for i in range(40)], "calibration_ppl": 10.6461,
    }), encoding="utf-8")


def run(tmp, **kw):
    return rg.compute(tmp, "reap128", "dense", "random128", observation=tmp / "observation.json", **kw)


def test_lb_hand_computed():
    # d = [1,0,0,-1,0,0,0,0,0,0]: mean 0, var = 2/9, sd = sqrt(2/9)
    d = [1, 0, 0, -1, 0, 0, 0, 0, 0, 0]
    expected = 0 - 1.645 * math.sqrt(2 / 9) / math.sqrt(10)
    assert rg.lower_bound(d) == pytest.approx(expected)
    assert rg.lower_bound(d) == pytest.approx(-0.24522, abs=1e-4)


def test_all_pass(tmp_path):
    build(tmp_path)
    write_arm(tmp_path, "reap128_q4", "gsm8k", [1] * 200, pps=[40] * 200,
              serve={"experts_offloaded": 0, "vram_used_mib": 15000, "n_cpu_moe": None})
    write_arm(tmp_path, "dense_q4", "gsm8k", [1] * 200, pps=[20] * 200,
              serve={"experts_offloaded": 30, "vram_used_mib": 15000, "n_cpu_moe": 20})
    r = run(tmp_path, speed_arm="reap128_q4", speed_dense="dense_q4")
    g = r["gates"]
    assert [g[k]["status"] for k in ("gate0", "gate1", "gate2", "control", "speed")] == ["PASS"] * 5
    assert g["gate1"]["n"] == 1319
    assert g["d0"]["status"] == "PASS"
    assert r["decision"] == "reap128 is the custom MoE"


def test_gate0_fail_terminated_marks_rest_not_evaluated(tmp_path):
    n = 1319
    build(tmp_path, terminated=[0] * 40 + [1] * (n - 40))  # >2% fail
    r = run(tmp_path)
    g = r["gates"]
    assert g["gate0"]["status"] == "FAIL"
    assert g["gate1"]["status"] == "NOT_EVALUATED" and g["gate1"]["after_failure"] is True
    assert g["gate1"]["statistic"] is not None
    assert g["gate2"]["after_failure"] is True
    assert "both reap128_v2 and reap160" in r["decision"]


def test_gate0_fail_degenerate(tmp_path):
    n = 1319
    build(tmp_path, trigram=[0.1] * 40 + [0.9] * (n - 40))
    assert run(tmp_path)["gates"]["gate0"]["status"] == "FAIL"


def test_gate1_fail(tmp_path):
    n = 1319
    build(tmp_path, arm_gsm=[0] * 60 + [1] * (n - 60))  # mean d = -0.15
    r = run(tmp_path)
    assert r["gates"]["gate0"]["status"] == "PASS"
    assert r["gates"]["gate1"]["status"] == "FAIL"
    assert r["gates"]["gate2"]["after_failure"] is True
    assert "both reap128_v2 and reap160" in r["decision"]


def test_gate2_fail_is_coding_regression(tmp_path):
    build(tmp_path, arm_he=[0] * 60 + [1] * 104)
    r = run(tmp_path)
    assert r["gates"]["gate1"]["status"] == "PASS"
    assert r["gates"]["gate2"]["status"] == "FAIL"
    assert r["decision"] == "coding regression — do not promote"


def test_control_fail(tmp_path):
    build(tmp_path, ctrl=[1] * 1319)
    r = run(tmp_path)
    assert r["gates"]["control"]["status"] == "FAIL"
    assert "not a REAP result" in r["decision"]


def test_id_mismatch_refuses(tmp_path):
    build(tmp_path)
    write_arm(tmp_path, "random128", "gsm8k", [0] * 1318)
    with pytest.raises(rg.GateError, match="id"):
        run(tmp_path)


def test_missing_speed_data_is_unknown(tmp_path):
    build(tmp_path)
    r = run(tmp_path)
    assert r["gates"]["speed"]["status"] == "UNKNOWN"
    assert r["decision"].startswith("undecided")
    write_arm(tmp_path, "reap128_q4", "gsm8k", [1] * 200)  # no pps, no serve.json
    write_arm(tmp_path, "dense_q4", "gsm8k", [1] * 200)
    r = run(tmp_path, speed_arm="reap128_q4", speed_dense="dense_q4")
    assert r["gates"]["speed"]["status"] == "UNKNOWN"


def test_speed_fail_offloaded_and_ratio(tmp_path):
    build(tmp_path)
    write_arm(tmp_path, "dense_q4", "gsm8k", [1] * 200, pps=[20] * 200)
    write_arm(tmp_path, "reap128_q4", "gsm8k", [1] * 200, pps=[40] * 200,
              serve={"experts_offloaded": 2, "vram_used_mib": 1, "n_cpu_moe": 1})
    assert run(tmp_path, speed_arm="reap128_q4", speed_dense="dense_q4")["gates"]["speed"]["status"] == "FAIL"
    write_arm(tmp_path, "reap128_q4", "gsm8k", [1] * 200, pps=[25] * 200,
              serve={"experts_offloaded": 0, "vram_used_mib": 1, "n_cpu_moe": None})
    s = run(tmp_path, speed_arm="reap128_q4", speed_dense="dense_q4")["gates"]["speed"]
    assert s["status"] == "FAIL" and s["statistic"]["ratio"] == pytest.approx(1.25)


def test_cli_writes_json(tmp_path):
    build(tmp_path)
    out = tmp_path / "o.json"
    done = subprocess.run([sys.executable, str(ROOT / "chowder_batch" / "reap_gates.py"), "--eval-dir", str(tmp_path),
                           "--arm", "reap128", "--dense", "dense", "--control", "random128",
                           "--observation", str(tmp_path / "observation.json"), "--out", str(out)],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert json.loads(out.read_text())["gates"]["gate0"]["status"] == "PASS"


def test_missing_observation_cannot_select_a_model(tmp_path):
    build(tmp_path)
    result = rg.compute(tmp_path, "reap128", "dense", "random128")
    assert result["gates"]["d0"]["status"] == "UNKNOWN"
    assert result["decision"] == "undecided — UNKNOWN: d0"


def test_invalid_observer_refuses_even_if_benchmark_gates_pass(tmp_path):
    build(tmp_path)
    (tmp_path / "observation.json").write_text(json.dumps({
        "layers": [{"layer": i} for i in range(39)], "calibration_ppl": 10,
    }))
    result = run(tmp_path)
    assert result["gates"]["d0"]["status"] == "FAIL"
    assert "D0 failed" in result["decision"]


def test_short_suite_or_duplicate_ids_refuses_instead_of_passing(tmp_path):
    build(tmp_path)
    write_arm(tmp_path, "reap128", "gsm8k", [1] * 1318)
    with pytest.raises(rg.GateError, match="frozen gate requires 1319"):
        run(tmp_path)
    write_arm(tmp_path, "reap128", "gsm8k", [1] * 1319)
    file = tmp_path / "reap128" / "gsm8k.jsonl"
    with file.open("a", encoding="utf-8") as out:
        out.write(json.dumps({"id": "gsm8k-0", "correct": True}) + "\n")
    with pytest.raises(rg.GateError, match="duplicate item id"):
        run(tmp_path)


def test_reap160_fallback_allows_partial_offload_only_when_fast_enough(tmp_path):
    build(tmp_path)
    write_arm(tmp_path, "reap160", "gsm8k", [1] * 1319)
    write_arm(tmp_path, "reap160", "humaneval", [1] * 164)
    write_arm(tmp_path, "reap160_q4", "gsm8k", [1] * 200, pps=[40] * 200,
              serve={"experts_offloaded": 10})
    write_arm(tmp_path, "dense_q4", "gsm8k", [1] * 200, pps=[20] * 200,
              serve={"experts_offloaded": 80})
    result = rg.compute(tmp_path, "reap160", "dense", "random128",
                        "reap160_q4", "dense_q4", tmp_path / "observation.json")
    assert result["gates"]["speed"]["status"] == "PASS"
    assert "selection still requires amendment 4" in result["decision"]


@pytest.mark.parametrize("forged_correct", [2, -1, 1.0, "1", None])
def test_frozen_quality_rejects_nonbinary_correct_values(tmp_path, forged_correct):
    # The evaluator writes bool; numerical/string truthiness must not be
    # interpreted as a fractional or arbitrarily large benchmark score.
    build(tmp_path, arm_gsm=[0] * 90 + [1] * (1319 - 90))
    path = tmp_path / "reap128" / "gsm8k.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["correct"] = forged_correct
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(rg.GateError, match="correct.*boolean"):
        run(tmp_path)


@pytest.mark.parametrize("forged_terminated", ["false", "true", 0, 1, None])
def test_gate0_rejects_nonboolean_termination_even_when_quality_is_binary(
    tmp_path, forged_terminated,
):
    # 27/1319 missed terminations breach the frozen 2% Gate0 tolerance;
    # the GSM8K paired-quality lower bound still passes at about -0.0269.
    build(tmp_path, arm_gsm=[0] * 27 + [1] * (1319 - 27),
          terminated=[0] * 27 + [1] * (1319 - 27))
    original = run(tmp_path)
    assert original["gates"]["gate0"]["status"] == "FAIL"
    assert original["gates"]["gate1"]["computed_status"] == "PASS"
    path = tmp_path / "reap128" / "gsm8k.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for row in rows[:27]:
        row["terminated"] = forged_terminated
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(rg.GateError, match="terminated.*boolean"):
        run(tmp_path)


@pytest.mark.parametrize("forged_trigram", [
    float("nan"), float("inf"), -0.01, 1.01, "0.9", True, None,
])
def test_gate0_rejects_nonfinite_or_out_of_range_trigram_ratio(tmp_path, forged_trigram):
    # Production computes unique trigrams / all trigrams: always finite [0, 1].
    build(tmp_path, trigram=[0.1] * 40 + [0.9] * (1319 - 40))
    assert run(tmp_path)["gates"]["gate0"]["status"] == "FAIL"
    path = tmp_path / "reap128" / "gsm8k.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for row in rows[:40]:
        row["trigram_ratio"] = forged_trigram
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(rg.GateError, match="trigram_ratio.*finite.*\[0, 1\]"):
        run(tmp_path)


@pytest.mark.parametrize(("incomplete", "bad_value"), [
    ("arm", None),
    ("dense", None),
    ("both", None),
    ("arm", float("nan")),
    ("arm", float("inf")),
    ("dense", True),
    ("arm", -3.0),
])
def test_speed_gate_refuses_one_sample_or_invalid_samples(tmp_path, incomplete, bad_value):
    build(tmp_path)
    arm_pps = [100.0] * 200
    dense_pps = [10.0] * 200
    for side, samples in (("arm", arm_pps), ("dense", dense_pps)):
        if incomplete in (side, "both"):
            if bad_value is None:
                samples[1:] = [None] * 199
            else:
                samples[0] = bad_value
    write_arm(tmp_path, "reap128_q4", "gsm8k", [1] * 200, pps=arm_pps,
              serve={"experts_offloaded": 0})
    write_arm(tmp_path, "dense_q4", "gsm8k", [1] * 200, pps=dense_pps)
    speed = run(tmp_path, speed_arm="reap128_q4", speed_dense="dense_q4")["gates"]["speed"]
    assert speed["status"] == "UNKNOWN"
    assert "speed samples" in speed["reason"]
    if incomplete in ("arm", "both"):
        assert speed["n"] < 200
    else:
        assert speed["n"] == 200


def test_speed_gate_requires_and_accepts_all_200_valid_samples(tmp_path):
    build(tmp_path)
    write_arm(tmp_path, "reap128_q4", "gsm8k", [1] * 200, pps=[25.0] * 200,
              serve={"experts_offloaded": 0})
    write_arm(tmp_path, "dense_q4", "gsm8k", [1] * 200, pps=[10.0] * 200)
    speed = run(tmp_path, speed_arm="reap128_q4", speed_dense="dense_q4")["gates"]["speed"]
    assert speed["status"] == "PASS"
    assert speed["n"] == 200
    assert speed["statistic"]["ratio"] == pytest.approx(2.5)


@pytest.mark.parametrize("changed_arm", ["reap128_q4", "dense_q4", "both"])
def test_speed_gate_refuses_200_fast_but_unpaired_gsm8k_items(tmp_path, changed_arm):
    build(tmp_path)
    write_arm(tmp_path, "reap128_q4", "gsm8k", [1] * 200, pps=[25.0] * 200,
              serve={"experts_offloaded": 0})
    write_arm(tmp_path, "dense_q4", "gsm8k", [1] * 200, pps=[10.0] * 200)
    targets = ("reap128_q4", "dense_q4") if changed_arm == "both" else (changed_arm,)
    for target in targets:
        path = tmp_path / target / "gsm8k.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows[0]["id"] = "gsm8k-200"  # still 200 distinct, finite speed observations
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(rg.GateError, match="first 200 GSM8K"):
        run(tmp_path, speed_arm="reap128_q4", speed_dense="dense_q4")


@pytest.mark.parametrize(("resumed_arm", "marker"), [
    ("reap128_q4", "serve.resume-rescored.json"),
    ("dense_q4", "serve.resume-rescored.json"),
    ("reap128_q4", "server.resume-interrupted.log"),
    ("reap128_q4", ".serve.lock"),
])
def test_speed_gate_cannot_reuse_original_fit_for_resumed_speed_rows(
    tmp_path, resumed_arm, marker,
):
    build(tmp_path)
    write_arm(tmp_path, "reap128_q4", "gsm8k", [1] * 200, pps=[25.0] * 200,
              serve={"experts_offloaded": 0})
    write_arm(tmp_path, "dense_q4", "gsm8k", [1] * 200, pps=[10.0] * 200)
    # Per-item rows have no session ID; a resume can contribute fast samples
    # while serve.json still describes the original GPU placement.
    (tmp_path / resumed_arm / marker).write_text(
        json.dumps({"experts_offloaded": 12}) if marker.endswith(".json") else "resumed session",
        encoding="utf-8",
    )
    speed = run(tmp_path, speed_arm="reap128_q4", speed_dense="dense_q4")["gates"]["speed"]
    assert speed["status"] == "UNKNOWN"
    assert "session" in speed["reason"]


def test_historical_reap160_null_placement_remains_unknown(tmp_path):
    build(tmp_path)
    write_arm(tmp_path, "reap160", "gsm8k", [1] * 1319)
    write_arm(tmp_path, "reap160", "humaneval", [1] * 164)
    write_arm(tmp_path, "reap160_q4", "gsm8k", [1] * 200, pps=[40.0] * 200,
              serve={"experts_offloaded": None})
    write_arm(tmp_path, "dense_q4", "gsm8k", [1] * 200, pps=[10.0] * 200)
    result = rg.compute(tmp_path, "reap160", "dense", "random128",
                        "reap160_q4", "dense_q4", tmp_path / "observation.json")
    speed = result["gates"]["speed"]
    assert speed["status"] == "UNKNOWN"
    assert speed["statistic"]["ratio"] == pytest.approx(4.0)
