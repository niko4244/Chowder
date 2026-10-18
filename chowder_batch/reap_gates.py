"""Compute the frozen REAP pre-registration gates (docs/REAP_QWEN36_PREREG.md §4-5)
from per-arm reap_eval.py output.

  python chowder_batch/reap_gates.py --eval-dir D --arm reap128 --dense dense \
      --control random128 [--speed-arm reap128_q4 --speed-dense dense_q4] --out gates.json

Thresholds are the pre-registered ones; nothing here is tunable. Absent
speed/fit data is UNKNOWN, never PASS. Mismatched ids between arms refuse.
Gate D0 (observer perplexity) is not computable from eval output and is not
checked here.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

Z = 1.645
EPS = 1e-12
G0_TOL = 0.02
G1_THRESH = -0.03
G2_THRESH = -0.10
SPEEDUP = 1.3
DEGENERATE_TRIGRAM = 0.5


class GateError(Exception):
    pass


def lower_bound(d: list[float]) -> float | None:
    n = len(d)
    if n < 2:
        return None
    return statistics.fmean(d) - Z * statistics.stdev(d) / math.sqrt(n)


def _load(eval_dir, arm: str, suite: str):
    path = Path(eval_dir) / arm / f"{suite}.jsonl"
    if not path.exists():
        return None
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["id"]] = row
    return rows


def _pair(a: dict, b: dict, names: str):
    if set(a) != set(b):
        only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
        raise GateError(f"id mismatch between {names}: {len(only_a)} only in first (e.g. {only_a[:3]}), "
                        f"{len(only_b)} only in second (e.g. {only_b[:3]})")
    ids = sorted(a)
    return ids


def _g(status, statistic=None, threshold=None, n=None, **extra):
    return {"status": status, "statistic": statistic, "threshold": threshold, "n": n, **extra}


def _lb_gate(arm_rows, base_rows, threshold, names, strict_gt=True):
    if arm_rows is None or base_rows is None:
        return _g("UNKNOWN", None, threshold, None, reason="eval file missing")
    ids = _pair(arm_rows, base_rows, names)
    d = [float(arm_rows[i]["correct"]) - float(base_rows[i]["correct"]) for i in ids]
    lb = lower_bound(d)
    if lb is None:
        return _g("UNKNOWN", None, threshold, len(ids), reason="n < 2")
    return _g("PASS" if lb > threshold else "FAIL", lb, threshold, len(ids), mean_diff=statistics.fmean(d))


def _rates(rows):
    n = len(rows)
    return (sum(bool(r["terminated"]) for r in rows.values()) / n,
            sum(r["trigram_ratio"] < DEGENERATE_TRIGRAM for r in rows.values()) / n)


def _gate0(arm_rows, base_rows):
    if arm_rows is None or base_rows is None:
        return _g("UNKNOWN", None, {"terminated_delta_min": -G0_TOL, "degenerate_delta_max": G0_TOL}, None,
                  reason="eval file missing")
    _pair(arm_rows, base_rows, "gsm8k arm/dense")
    at, ad = _rates(arm_rows)
    dt, dd = _rates(base_rows)
    ok = at >= dt - G0_TOL - EPS and ad <= dd + G0_TOL + EPS
    return _g("PASS" if ok else "FAIL",
              {"terminated_arm": at, "terminated_dense": dt, "degenerate_arm": ad, "degenerate_dense": dd},
              {"terminated_delta_min": -G0_TOL, "degenerate_delta_max": G0_TOL}, len(arm_rows))


def _mean_pps(eval_dir, arm):
    rows = _load(eval_dir, arm, "gsm8k")
    vals = [r["predicted_per_second"] for r in (rows or {}).values() if r.get("predicted_per_second") is not None]
    return (statistics.fmean(vals), len(vals)) if vals else (None, 0)


def _speed(eval_dir, speed_arm, speed_dense):
    thr = {"experts_offloaded": 0, "speed_ratio_min": SPEEDUP}
    if not speed_arm or not speed_dense:
        return _g("UNKNOWN", None, thr, None, reason="speed arms not given")
    serve_path = Path(eval_dir) / speed_arm / "serve.json"
    offloaded = None
    if serve_path.exists():
        offloaded = json.loads(serve_path.read_text(encoding="utf-8")).get("experts_offloaded")
    a, na = _mean_pps(eval_dir, speed_arm)
    b, nb = _mean_pps(eval_dir, speed_dense)
    ratio = a / b if a is not None and b else None
    stat = {"experts_offloaded": offloaded, "pps_arm": a, "pps_dense": b, "ratio": ratio}
    if offloaded is not None and offloaded != 0:
        return _g("FAIL", stat, thr, na)  # offload alone fails the gate whatever the speed
    if offloaded is None or ratio is None:
        return _g("UNKNOWN", stat, thr, na, reason="missing serve.json or predicted_per_second")
    return _g("PASS" if ratio >= SPEEDUP - EPS else "FAIL", stat, thr, na)


def _decide(g):
    st = {k: v["status"] for k, v in g.items()}
    if "FAIL" in (st["gate0"], st["gate1"]):
        return "build reap160"
    if st["gate2"] == "FAIL":
        return "coding regression — do not promote"
    if st["control"] == "FAIL":
        return "reap128 is a 50% expert drop, not a REAP result (control gate failed)"
    unknown = [k for k, v in st.items() if v in ("UNKNOWN", "NOT_EVALUATED")]
    if unknown:
        return "undecided — UNKNOWN: " + ", ".join(unknown)
    if st["speed"] == "FAIL":
        return "speed/fit gate failed — do not promote"
    return "reap128 is the custom MoE"


def compute(eval_dir, arm, dense, control, speed_arm=None, speed_dense=None) -> dict:
    gsm = {a: _load(eval_dir, a, "gsm8k") for a in (arm, dense, control)}
    he = {a: _load(eval_dir, a, "humaneval") for a in (arm, dense)}
    computed = {
        "gate0": _gate0(gsm[arm], gsm[dense]),
        "gate1": _lb_gate(gsm[arm], gsm[dense], G1_THRESH, "gsm8k arm/dense"),
        "gate2": _lb_gate(he[arm], he[dense], G2_THRESH, "humaneval arm/dense"),
        "control": _lb_gate(gsm[arm], gsm[control], 0.0, "gsm8k arm/control"),
        "speed": _speed(eval_dir, speed_arm, speed_dense),
    }
    gates, failed = {}, False
    for name in ("gate0", "gate1", "gate2", "control", "speed"):
        res = computed[name]
        if failed:
            res = {**res, "status": "NOT_EVALUATED", "after_failure": True, "computed_status": res["status"]}
        gates[name] = res
        failed = failed or computed[name]["status"] == "FAIL"
    return {"arm": arm, "dense": dense, "control": control, "gates": gates,
            "decision": _decide(gates)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--dense", required=True)
    p.add_argument("--control", required=True)
    p.add_argument("--speed-arm")
    p.add_argument("--speed-dense")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    try:
        result = compute(a.eval_dir, a.arm, a.dense, a.control, a.speed_arm, a.speed_dense)
    except GateError as e:
        sys.exit(f"reap_gates: refusing: {e}")
    Path(a.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v["status"] for k, v in result["gates"].items()}), result["decision"])


if __name__ == "__main__":
    main()
