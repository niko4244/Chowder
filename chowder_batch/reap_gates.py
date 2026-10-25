"""Compute the frozen REAP pre-registration gates (docs/REAP_QWEN36_PREREG.md §4-5)
from per-arm reap_eval.py output.

  python chowder_batch/reap_gates.py --eval-dir D --arm reap128 --dense dense \
      --control random128 [--speed-arm reap128_q4 --speed-dense dense_q4] --out gates.json

Thresholds are the pre-registered ones; nothing here is tunable. Absent
speed/fit data is UNKNOWN, never PASS. Mismatched ids between arms refuse.
Gate D0 uses the separately captured 40-layer observation. If it is absent,
the whole result remains undecided (never an implicit PASS).
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
FROZEN_SUITE_SIZES = {"gsm8k": 1319, "humaneval": 164}
FROZEN_SOURCE_SHA256 = {
    "gsm8k": "913ca81f724cba16a07099608047e7522b00910c1c7fe13499c501ac9e4bad89",
    "humaneval": "49fb688c6788b0531803dfd1cb85a69926ec5a7179e74cdfffc53ba52a92b420",
}


class GateError(Exception):
    pass


def lower_bound(d: list[float]) -> float | None:
    n = len(d)
    if n < 2:
        return None
    return statistics.fmean(d) - Z * statistics.stdev(d) / math.sqrt(n)


def _load(eval_dir, arm: str, suite: str, *, expected_n: int | None = None):
    path = Path(eval_dir) / arm / f"{suite}.jsonl"
    if not path.exists():
        return None
    rows = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            row = json.loads(line)
            # The frozen per-item quality outcome is binary, as reap_eval emits.
            # float() in the downstream paired gate must never turn a forged
            # value such as 2 into a score beyond one correct answer.
            if not isinstance(row, dict) or not isinstance(row.get("correct"), bool):
                raise GateError(f"{path}:{line_number}: correct must be boolean")
            ident = row["id"]
            if ident in rows:
                raise GateError(f"{path}:{line_number}: duplicate item id {ident!r}")
            # reap_eval emits a boolean termination and the unique-trigram
            # count divided by the number of trigrams, bounded to [0, 1].
            # bool("false") is True, while NaN < 0.5 is False; either value
            # could hide an actual Gate0 termination/degeneracy regression.
            if not isinstance(row.get("terminated"), bool):
                raise GateError(f"{path}:{line_number}: terminated must be boolean")
            ratio = row.get("trigram_ratio")
            if (isinstance(ratio, bool) or not isinstance(ratio, (int, float))
                    or not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0):
                raise GateError(
                    f"{path}:{line_number}: trigram_ratio must be finite within [0, 1]"
                )
            rows[ident] = row
    required = expected_n if expected_n is not None else FROZEN_SUITE_SIZES.get(suite)
    if required is not None and len(rows) != required:
        raise GateError(f"{path}: {suite} has {len(rows)} rows, but frozen gate requires {required}")
    return rows


def _d0(observation: str | Path | None):
    """Frozen observation sanity gate, which must precede quality evidence."""
    if not observation:
        return _g("UNKNOWN", None, 20.0, None, reason="full observer record not supplied")
    path = Path(observation)
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        return _g("UNKNOWN", None, 20.0, None, reason=f"observer unreadable: {error}")
    if not isinstance(report, dict) or not isinstance(report.get("layers"), list):
        return _g("FAIL", None, 20.0, None, reason="observer has no layer census")
    layers = report["layers"]
    indexes = [row.get("layer") if isinstance(row, dict) else None for row in layers]
    ppl = report.get("calibration_ppl")
    valid_ppl = isinstance(ppl, (int, float)) and not isinstance(ppl, bool) and math.isfinite(ppl)
    valid_indexes = all(isinstance(i, int) and not isinstance(i, bool) for i in indexes)
    passed = (len(layers) == 40 and valid_indexes and set(indexes) == set(range(40))
              and valid_ppl and 0.0 < ppl < 20.0)
    return _g("PASS" if passed else "FAIL", ppl if valid_ppl else None, 20.0, len(layers),
              reason=None if passed else "observer needs 40 distinct layers and 0 < perplexity < 20")


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


def _mean_pps(eval_dir, arm, *, frozen_first_200=False):
    # The scientific gate must use the SAME first 200 GSM8K items in both
    # Q4 arms. Leave read-only historical speed reconciliation descriptive.
    rows = _load(eval_dir, arm, "gsm8k", expected_n=200)
    if frozen_first_200 and rows is not None and set(rows) != {f"gsm8k-{i}" for i in range(200)}:
        raise GateError(f"{arm}: speed gate needs the frozen first 200 GSM8K item ids")
    # Filtering absent/invalid values must not turn one fast item into a PASS.
    vals = [r.get("predicted_per_second") for r in (rows or {}).values()]
    measured = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)
                and math.isfinite(v) and v >= 0]
    return (statistics.fmean(measured), 200) if len(measured) == 200 else (None, len(measured))


def _verify_new_q4_speed_session(eval_dir, arm, receipt):
    """Admit the serving wrapper's *sealed* original 200-item Q4 run only."""
    import hashlib
    import re

    directory = Path(eval_dir) / arm
    proof = receipt.get("speed_session")
    if proof is None:
        return False  # no post-hoc certification of old or interrupted runs
    if not isinstance(proof, dict):
        raise GateError(f"{arm}: malformed Q4 speed session receipt")
    if (proof.get("schema") != "chowder.reap-q4-same-session-speed.v1"
            or not isinstance(proof.get("session_id"), str)
            or not re.fullmatch(r"[0-9a-f]{32}", proof["session_id"])
            or type(proof.get("server_pid")) is not int or proof["server_pid"] <= 0
            or type(proof.get("port")) is not int or not 1 <= proof["port"] <= 65535
            or proof.get("suite") != "gsm8k"
            or type(proof.get("n")) is not int or proof["n"] != 200):
        raise GateError(f"{arm}: invalid Q4 same-session speed receipt")
    paths = {name: directory / name for name in
             ("server.log", "gsm8k.jsonl", "summary.json", "gsm8k.inputs.json")}
    if any(not p.is_file() or p.is_symlink() for p in paths.values()):
        raise GateError(f"{arm}: incomplete Q4 same-session speed evidence")
    digest = lambda raw: hashlib.sha256(raw).hexdigest()
    try:
        inputs = json.loads(paths["gsm8k.inputs.json"].read_text(encoding="utf-8"))
        summary = json.loads(paths["summary.json"].read_text(encoding="utf-8"))
        source = inputs.get("source_items_sha256") if isinstance(inputs, dict) else None
        model = {key: receipt.get(key) for key in ("gguf", "gguf_sha256", "flags")}
        entry = summary.get("gsm8k") if isinstance(summary, dict) else None
        rows_sha = digest(paths["gsm8k.jsonl"].read_bytes())
        frozen_decode = {"system": "You are a helpful assistant.",
                         "temperature": 0.0, "top_k": 1, "seed": 7,
                         "max_tokens": 1024, "enable_thinking": False}
        decoded = inputs.get("decode") if isinstance(inputs, dict) else None
        if (not isinstance(inputs, dict)
                or inputs.get("schema") != "chowder.reap-eval-input-identity.v1"
                or inputs.get("suite") != "gsm8k" or type(inputs.get("source_items_n")) is not int
                or inputs["source_items_n"] != 1319
                or source != FROZEN_SOURCE_SHA256["gsm8k"]
                or not isinstance(decoded, dict) or decoded != frozen_decode
                or any(type(decoded[key]) is not type(value)
                       for key, value in frozen_decode.items())
                or inputs.get("scoring") != "frozen reap_eval production scoring: final_number_match/boxed/HumanEval"
                or inputs.get("model") != model
                or not isinstance(model["gguf"], str) or not model["gguf"].strip()
                or model["flags"] != ["--fit", "on", "-np", "4", "-c", "16384", "--jinja"]
                or not isinstance(model["gguf_sha256"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", model["gguf_sha256"])
                or proof.get("source_items_sha256") != source
                or proof.get("gguf_sha256") != model["gguf_sha256"]
                or proof.get("inputs_sha256") != digest(paths["gsm8k.inputs.json"].read_bytes())
                or proof.get("rows_sha256") != rows_sha
                or not isinstance(entry, dict) or type(entry.get("n")) is not int
                or entry["n"] != 200 or entry.get("file_sha256") != rows_sha
                or proof.get("summary_entry_sha256") != digest(json.dumps(
                    entry, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))):
            raise GateError(f"{arm}: Q4 same-session speed source/model/rows/summary receipt mismatch")
        prefix_size = receipt.get("server_log_startup_bytes")
        if type(prefix_size) is not int or not 1 <= prefix_size <= 64 * 1024 * 1024:
            raise GateError(f"{arm}: invalid Q4 startup log length")
        with paths["server.log"].open("rb") as stream:
            raw = stream.read(prefix_size)
        if (len(raw) != prefix_size or digest(raw) != receipt.get("server_log_startup_sha256")
                or proof.get("server_log_startup_sha256") != digest(raw)):
            raise GateError(f"{arm}: Q4 startup log SHA256 receipt mismatch")
        # Verify fit from actual startup lines, not a forged zero in serve.json.
        try:
            from chowder_batch import reap_serve
        except ImportError:
            import reap_serve
        reap_serve._verify_offload_receipt(receipt, paths["server.log"])
        if receipt.get("experts_offloaded") != receipt.get("measured_experts_offloaded"):
            raise GateError(f"{arm}: Q4 original serving offload receipt mismatch")
    except (OSError, UnicodeError, ValueError, TypeError, KeyError) as exc:
        raise GateError(f"{arm}: invalid Q4 same-session speed receipt: {exc}") from exc
    return True


def _speed(eval_dir, speed_arm, speed_dense, *, arm):
    # Amendment 2 + §5 permit partial offload for the REAP160 fallback;
    # REAP128 must still fit with zero routed experts offloaded.
    partial_allowed = arm in {"reap160", "reap160_v2"}
    thr = {"experts_offloaded": "partial permitted" if partial_allowed else 0,
           "speed_ratio_min": SPEEDUP}
    if not speed_arm or not speed_dense:
        return _g("UNKNOWN", None, thr, None, reason="speed arms not given")
    serve_path = Path(eval_dir) / speed_arm / "serve.json"
    offloaded = None
    if serve_path.exists():
        offloaded = json.loads(serve_path.read_text(encoding="utf-8")).get("experts_offloaded")
    if offloaded is not None and (
        isinstance(offloaded, bool) or not isinstance(offloaded, int) or offloaded < 0
    ):
        return _g("FAIL", {"experts_offloaded": offloaded}, thr, None,
                  reason="invalid negative or noninteger expert offload count")
    a, na = _mean_pps(eval_dir, speed_arm, frozen_first_200=True)
    b, nb = _mean_pps(eval_dir, speed_dense, frozen_first_200=True)
    ratio = a / b if a is not None and b else None
    stat = {"experts_offloaded": offloaded, "pps_arm": a, "pps_dense": b, "ratio": ratio}
    # ponytail: treat all resumed speed sessions as UNKNOWN without per-row
    # attribution. Ceiling: harmless resumes cannot PASS. Upgrade: seal each
    # row's session ID against that session's model/fit serving receipt.
    for speed_dir in (Path(eval_dir) / speed_arm, Path(eval_dir) / speed_dense):
        if ((speed_dir / ".serve.lock").exists()
                or any(speed_dir.glob("serve.resume-*.json"))
                or any(speed_dir.glob("server.resume-*.log"))
                or any(speed_dir.glob(".serve-pending-*.json"))):
            return _g("UNKNOWN", stat, thr, na,
                      reason=f"{speed_dir.name}: speed samples lack session-bound fit evidence "
                             "(resumed or active session)")
    if speed_arm in {"reap128_v2_q4", "reap160_v2_q4"}:
        if not serve_path.is_file():
            return _g("UNKNOWN", stat, thr, na, reason="new v2 Q4 speed session proof missing")
        if not _verify_new_q4_speed_session(eval_dir, speed_arm,
                                            json.loads(serve_path.read_text(encoding="utf-8"))):
            return _g("UNKNOWN", stat, thr, na, reason="new v2 Q4 speed session proof missing")
    if offloaded is not None and offloaded != 0 and not partial_allowed:
        return _g("FAIL", stat, thr, na)  # offload alone fails the gate whatever the speed
    if offloaded is None or ratio is None:
        return _g("UNKNOWN", stat, thr, na,
                  reason=f"missing serve.json or complete finite speed samples "
                         f"(arm {na}/200, dense {nb}/200)")
    return _g("PASS" if ratio >= SPEEDUP - EPS else "FAIL", stat, thr, na)


def _decide(g, arm):
    st = {k: v["status"] for k, v in g.items()}
    if st["d0"] == "FAIL":
        return "observer D0 failed — no model may be selected"
    if st["d0"] != "PASS":
        return "undecided — UNKNOWN: d0"
    if "FAIL" in (st["gate0"], st["gate1"]):
        if arm == "reap128":
            return "v1 failed: score both reap128_v2 and reap160 (amendment 4)"
        return f"{arm} failed a quality gate — do not select it"
    if st["gate2"] == "FAIL":
        return "coding regression — do not promote"
    if st["control"] == "FAIL":
        return "reap128 is a 50% expert drop, not a REAP result (control gate failed)"
    unknown = [k for k, v in st.items() if v in ("UNKNOWN", "NOT_EVALUATED")]
    if unknown:
        return "undecided — UNKNOWN: " + ", ".join(unknown)
    if st["speed"] == "FAIL":
        return "speed/fit gate failed — do not promote"
    if arm == "reap128_v2":
        return "reap128_v2 passes gates; still requires paired noninferiority versus v1"
    if arm in {"reap160", "reap160_v2"}:
        return f"{arm} passes gates; selection still requires amendment 4's ordered arms"
    return "reap128 is the custom MoE"


def compute(eval_dir, arm, dense, control, speed_arm=None, speed_dense=None, observation=None) -> dict:
    gsm = {a: _load(eval_dir, a, "gsm8k") for a in (arm, dense, control)}
    he = {a: _load(eval_dir, a, "humaneval") for a in (arm, dense)}
    computed = {
        "d0": _d0(observation),
        "gate0": _gate0(gsm[arm], gsm[dense]),
        "gate1": _lb_gate(gsm[arm], gsm[dense], G1_THRESH, "gsm8k arm/dense"),
        "gate2": _lb_gate(he[arm], he[dense], G2_THRESH, "humaneval arm/dense"),
        "control": _lb_gate(gsm[arm], gsm[control], 0.0, "gsm8k arm/control"),
        "speed": _speed(eval_dir, speed_arm, speed_dense, arm=arm),
    }
    gates, failed = {}, False
    for name in ("d0", "gate0", "gate1", "gate2", "control", "speed"):
        res = computed[name]
        if failed:
            res = {**res, "status": "NOT_EVALUATED", "after_failure": True, "computed_status": res["status"]}
        gates[name] = res
        failed = failed or computed[name]["status"] == "FAIL"
    return {"arm": arm, "dense": dense, "control": control, "gates": gates,
            "decision": _decide(gates, arm)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--dense", required=True)
    p.add_argument("--control", required=True)
    p.add_argument("--speed-arm")
    p.add_argument("--speed-dense")
    p.add_argument("--observation", help="full 40-layer observer JSON; absent means D0 UNKNOWN")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    try:
        result = compute(a.eval_dir, a.arm, a.dense, a.control, a.speed_arm, a.speed_dense, a.observation)
    except GateError as e:
        sys.exit(f"reap_gates: refusing: {e}")
    Path(a.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v["status"] for k, v in result["gates"].items()}), result["decision"])


if __name__ == "__main__":
    main()
