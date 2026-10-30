"""Read-only historical REAP160 placement-evidence reconciliation.

This tool never changes frozen serve.json, placement.json, eval rows, or gate
thresholds. It demonstrates which claims a separate load-only probe supports
and which claims it cannot bind to the original 200-item quality/speed run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

from chowder_batch.reap_gates import GateError, _mean_pps

FULL_OFFLOAD = re.compile(
    r"load_tensors:\s+offloaded\s+(\d+)/(\d+)\s+layers to GPU"
)
FIT_SUCCESS = "common_fit_params: successfully fit params to free device memory"
FROZEN_FLAGS = ["--fit", "on", "-np", "4", "-c", "16384", "--jinja"]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reconcile(eval_dir: Path, *, speed_arm: str = "reap160_q4") -> dict:
    """Return grounded evidence without promoting a secondary source to gate."""
    arm_dir = eval_dir / speed_arm
    serving = arm_dir / "serve.json"
    placement = arm_dir / "placement.json"
    log = arm_dir / "probe.log"
    for file in (serving, placement, log):
        if not file.is_file() or file.stat().st_size > 5 * 1024 * 1024:
            raise ValueError(f"missing/oversized immutable speed evidence: {file}")
    first = json.loads(serving.read_text(encoding="utf-8"))
    second = json.loads(placement.read_text(encoding="utf-8"))
    raw = log.read_text(encoding="utf-8", errors="replace")
    if (first.get("flags") != FROZEN_FLAGS or second.get("flags") != FROZEN_FLAGS
            or first.get("gguf") != second.get("gguf")
            or first.get("gguf_sha256") is None
            or second.get("log_verbosity") != "4"):
        raise ValueError("original serving and separate load-only probe parameters disagree")
    parsed = [(int(a), int(b)) for a, b in FULL_OFFLOAD.findall(raw)]
    if len(parsed) != 1 or parsed[0] != (41, 41) or FIT_SUCCESS not in raw:
        raise ValueError("unambiguous full GPU load/fit evidence missing from original probe.log")
    lines = second.get("offload_log_lines")
    if (not isinstance(lines, list) or
            not all(isinstance(line, str) and line in raw for line in lines)
            or second.get("experts_offloaded") != 0):
        raise ValueError("placement summary not supported by unchanged raw load-only log")
    speed, n_arm = _mean_pps(eval_dir, speed_arm)
    baseline, n_dense = _mean_pps(eval_dir, "dense_q4")
    ratio = speed / baseline if baseline else None
    if (n_arm != 200 or n_dense != 200 or ratio is None or not math.isfinite(ratio)):
        raise ValueError("preregistered complete 200-item speed measurements unavailable")
    return {
        "schema": "chowder.reap160-historical-placement-reconciliation.v1",
        "status": "CORROBORATED_SEPARATE_LOAD_PROBE_NOT_CANONICAL_GATE",
        "arm": speed_arm,
        "original_serve_offload_status": first.get("experts_offloaded"),
        "separate_probe_all_gpu_layers": parsed[0][0],
        "separate_probe_total_layers": parsed[0][1],
        "separate_probe_fit": True,
        "speed_arm_pps": speed,
        "speed_dense_pps": baseline,
        "speed_ratio": ratio,
        "frozen_speed_ratio_required": 1.3,
        "speed_ratio_passes": ratio >= 1.3,
        "original_gguf": first["gguf"],
        "original_gguf_recorded_sha256": first["gguf_sha256"],
        "input_sha256": {str(p.resolve()): _digest(p) for p in
                         (serving, placement, log, arm_dir / "gsm8k.jsonl",
                          eval_dir / "dense_q4" / "gsm8k.jsonl")},
        "what_is_not_proven": (
            "The separately generated load-only placement probe has no complete "
            "cryptographic, live-process or artifact/measurement-session binding "
            "to the earlier serve.json and 200-item evaluation. The original "
            "canonical gate remains UNKNOWN until provenance adjudication; "
            "this report does not override reap_gates or thaw thresholds."
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="new immutable evidence file")
    args = parser.parse_args(argv)
    try:
        result = reconcile(args.eval_dir)
        payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        with args.out.open("x", encoding="utf-8", newline="\n") as file:
            file.write(payload)
        print(json.dumps({"status": result["status"], "out": str(args.out),
                          "speed_ratio": result["speed_ratio"]}))
        return 0
    except (ValueError, OSError, KeyError, TypeError, GateError) as exc:
        print(f"REAP160 historical placement refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
