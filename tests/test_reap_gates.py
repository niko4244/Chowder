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


def build(tmp, n=400, he_n=164, arm_he=None, arm_gsm=None, ctrl=None, **kw):
    ones = [1] * n
    write_arm(tmp, "dense", "gsm8k", ones)
    write_arm(tmp, "reap128", "gsm8k", arm_gsm if arm_gsm is not None else ones, **kw)
    write_arm(tmp, "random128", "gsm8k", ctrl if ctrl is not None else [0] * n)
    write_arm(tmp, "dense", "humaneval", [1] * he_n)
    write_arm(tmp, "reap128", "humaneval", arm_he if arm_he is not None else [1] * he_n)


def run(tmp, **kw):
    return rg.compute(tmp, "reap128", "dense", "random128", **kw)


def test_lb_hand_computed():
    # d = [1,0,0,-1,0,0,0,0,0,0]: mean 0, var = 2/9, sd = sqrt(2/9)
    d = [1, 0, 0, -1, 0, 0, 0, 0, 0, 0]
    expected = 0 - 1.645 * math.sqrt(2 / 9) / math.sqrt(10)
    assert rg.lower_bound(d) == pytest.approx(expected)
    assert rg.lower_bound(d) == pytest.approx(-0.24522, abs=1e-4)


def test_all_pass(tmp_path):
    build(tmp_path)
    write_arm(tmp_path, "reap128_q4", "gsm8k", [1] * 5, pps=[40] * 5,
              serve={"experts_offloaded": 0, "vram_used_mib": 15000, "n_cpu_moe": None})
    write_arm(tmp_path, "dense_q4", "gsm8k", [1] * 5, pps=[20] * 5,
              serve={"experts_offloaded": 30, "vram_used_mib": 15000, "n_cpu_moe": 20})
    r = run(tmp_path, speed_arm="reap128_q4", speed_dense="dense_q4")
    g = r["gates"]
    assert [g[k]["status"] for k in ("gate0", "gate1", "gate2", "control", "speed")] == ["PASS"] * 5
    assert g["gate1"]["n"] == 400
    assert r["decision"] == "reap128 is the custom MoE"


def test_gate0_fail_terminated_marks_rest_not_evaluated(tmp_path):
    n = 400
    build(tmp_path, terminated=[0] * 20 + [1] * (n - 20))  # 5% < dense-2%
    r = run(tmp_path)
    g = r["gates"]
    assert g["gate0"]["status"] == "FAIL"
    assert g["gate1"]["status"] == "NOT_EVALUATED" and g["gate1"]["after_failure"] is True
    assert g["gate1"]["statistic"] is not None
    assert g["gate2"]["after_failure"] is True
    assert r["decision"] == "build reap160"


def test_gate0_fail_degenerate(tmp_path):
    n = 400
    build(tmp_path, trigram=[0.1] * 20 + [0.9] * (n - 20))
    assert run(tmp_path)["gates"]["gate0"]["status"] == "FAIL"


def test_gate1_fail(tmp_path):
    n = 400
    build(tmp_path, arm_gsm=[0] * 60 + [1] * (n - 60))  # mean d = -0.15
    r = run(tmp_path)
    assert r["gates"]["gate0"]["status"] == "PASS"
    assert r["gates"]["gate1"]["status"] == "FAIL"
    assert r["gates"]["gate2"]["after_failure"] is True
    assert r["decision"] == "build reap160"


def test_gate2_fail_is_coding_regression(tmp_path):
    build(tmp_path, arm_he=[0] * 60 + [1] * 104)
    r = run(tmp_path)
    assert r["gates"]["gate1"]["status"] == "PASS"
    assert r["gates"]["gate2"]["status"] == "FAIL"
    assert r["decision"] == "coding regression — do not promote"


def test_control_fail(tmp_path):
    build(tmp_path, ctrl=[1] * 400)
    r = run(tmp_path)
    assert r["gates"]["control"]["status"] == "FAIL"
    assert "not a REAP result" in r["decision"]


def test_id_mismatch_refuses(tmp_path):
    build(tmp_path)
    write_arm(tmp_path, "random128", "gsm8k", [0] * 399)
    with pytest.raises(rg.GateError, match="id"):
        run(tmp_path)


def test_missing_speed_data_is_unknown(tmp_path):
    build(tmp_path)
    r = run(tmp_path)
    assert r["gates"]["speed"]["status"] == "UNKNOWN"
    assert r["decision"].startswith("undecided")
    write_arm(tmp_path, "reap128_q4", "gsm8k", [1] * 3)  # no pps, no serve.json
    write_arm(tmp_path, "dense_q4", "gsm8k", [1] * 3)
    r = run(tmp_path, speed_arm="reap128_q4", speed_dense="dense_q4")
    assert r["gates"]["speed"]["status"] == "UNKNOWN"


def test_speed_fail_offloaded_and_ratio(tmp_path):
    build(tmp_path)
    write_arm(tmp_path, "dense_q4", "gsm8k", [1] * 3, pps=[20] * 3)
    write_arm(tmp_path, "reap128_q4", "gsm8k", [1] * 3, pps=[40] * 3,
              serve={"experts_offloaded": 2, "vram_used_mib": 1, "n_cpu_moe": 1})
    assert run(tmp_path, speed_arm="reap128_q4", speed_dense="dense_q4")["gates"]["speed"]["status"] == "FAIL"
    write_arm(tmp_path, "reap128_q4", "gsm8k", [1] * 3, pps=[25] * 3,
              serve={"experts_offloaded": 0, "vram_used_mib": 1, "n_cpu_moe": None})
    s = run(tmp_path, speed_arm="reap128_q4", speed_dense="dense_q4")["gates"]["speed"]
    assert s["status"] == "FAIL" and s["statistic"]["ratio"] == pytest.approx(1.25)


def test_cli_writes_json(tmp_path):
    build(tmp_path)
    out = tmp_path / "o.json"
    done = subprocess.run([sys.executable, str(ROOT / "chowder_batch" / "reap_gates.py"), "--eval-dir", str(tmp_path),
                           "--arm", "reap128", "--dense", "dense", "--control", "random128", "--out", str(out)],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert json.loads(out.read_text())["gates"]["gate0"]["status"] == "PASS"
