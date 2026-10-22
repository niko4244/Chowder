"""Fail-closed Gen-0 selection under frozen REAP preregistration amendment 4.

Individual arm gates are not selection approval: when REAP128 v1 fails its
quality gate, both REAP128_v2 and REAP160 must be *scored* before choosing an
arm. Never silently promote a later arm because an earlier arm is missing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

try:
    from chowder_batch.reap_gates import GateError, _load, compute, lower_bound
except ImportError:  # direct script mode outside the repository package path
    from reap_gates import GateError, _load, compute, lower_bound

REQUIRED = ("d0", "gate0", "gate1", "gate2", "control", "speed")
MARGIN = -0.02  # frozen strict lower bound for paired v2-v1 comparisons


def _status(result: dict, key: str) -> str:
    gate = result["gates"][key]
    return gate.get("computed_status", gate["status"])


def _complete(result: dict) -> bool:
    return all(_status(result, key) in {"PASS", "FAIL"} for key in REQUIRED)


def _pass(result: dict) -> bool:
    return all(_status(result, key) == "PASS" for key in REQUIRED)


def select(gates: dict[str, dict | None], paired: dict | None = None) -> dict:
    """Enforce the specified precedence and evidence prerequisites, no overrides."""
    v1 = gates.get("reap128")
    if not v1 or any(_status(v1, key) not in {"PASS", "FAIL"} for key in
                     ("d0", "gate0", "gate1")):
        return {"status": "BLOCKED", "chosen": None, "reason": "REAP128 v1 D0/Gate0/Gate1 incomplete"}
    if _status(v1, "d0") != "PASS":
        return {"status": "FAILED", "chosen": None, "reason": "frozen D0 observer failed"}
    failed_fallback = any(_status(v1, key) == "FAIL" for key in ("gate0", "gate1"))
    v2 = gates.get("reap128_v2")
    reap160 = gates.get("reap160")
    if failed_fallback:
        # Amendment 4 explicitly says BOTH arms must be built and scored.
        if not v2 or not _complete(v2) or not reap160 or not _complete(reap160):
            return {"status": "BLOCKED", "chosen": None,
                    "reason": "amendment 4 requires both REAP128_v2 and REAP160 full gates"}
        if _pass(v2):
            return {"status": "SELECTED", "chosen": "reap128_v2", "reason": "first passing frozen fallback"}
        if _pass(reap160):
            return {"status": "SELECTED", "chosen": "reap160", "reason": "second passing frozen fallback"}
        reap160_v2 = gates.get("reap160_v2")
        if not reap160_v2 or not _complete(reap160_v2):
            return {"status": "BLOCKED", "chosen": None,
                    "reason": "both first fallbacks failed; REAP160_v2 is now required"}
        if _pass(reap160_v2):
            return {"status": "SELECTED", "chosen": "reap160_v2", "reason": "third passing frozen fallback"}
        return {"status": "FAILED", "chosen": None, "reason": "all frozen fallbacks failed"}
    if not _complete(v1):
        return {"status": "BLOCKED", "chosen": None, "reason": "REAP128 v1 remaining gates incomplete"}
    if not _pass(v1):
        return {"status": "FAILED", "chosen": None,
                "reason": "v1 failed a non-fallback gate; promotion prohibited"}
    if v2 is None:
        return {"status": "SELECTED", "chosen": "reap128",
                "reason": "v1 passes; optional v2 replacement has not been evaluated"}
    if not _complete(v2):
        return {"status": "BLOCKED", "chosen": None, "reason": "declared v2 evaluation incomplete"}
    if _pass(v2):
        if paired is None or any(paired.get(suite) is None for suite in ("gsm8k", "humaneval")):
            return {"status": "BLOCKED", "chosen": None,
                    "reason": "v2-v1 paired GSM8K and HumanEval bounds missing"}
        if all(paired[suite] > MARGIN for suite in ("gsm8k", "humaneval")):
            return {"status": "SELECTED", "chosen": "reap128_v2",
                    "reason": "v2 passes every gate and both strict paired bounds"}
    return {"status": "SELECTED", "chosen": "reap128", "reason": "v1 retained under amendment 4"}


def _attested_rows(eval_dir: Path, arm: str, suite: str, expected: int) -> dict | None:
    path = eval_dir / arm / f"{suite}.jsonl"
    summary_path = eval_dir / arm / "summary.json"
    if not path.is_file() or not summary_path.is_file():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    entry = summary.get(suite)
    if not isinstance(entry, dict):
        raise GateError(f"missing {arm}/{suite} summary entry")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if entry.get("file_sha256") != actual:
        raise GateError(f"{arm}/{suite} SHA256 does not match its saved summary")
    rows = _load(eval_dir, arm, suite, expected_n=expected)
    if entry.get("n") != len(rows):
        raise GateError(f"{arm}/{suite} summary item count mismatch")
    return rows


def _paired(eval_dir: Path) -> dict | None:
    pairs = {}
    for suite, n in (("gsm8k", 1319), ("humaneval", 164)):
        arm = _attested_rows(eval_dir, "reap128_v2", suite, n)
        v1 = _attested_rows(eval_dir, "reap128", suite, n)
        if arm is None or v1 is None:
            return None
        if set(arm) != set(v1):
            raise GateError(f"v2-v1 paired {suite} item identifiers mismatch")
        pairs[suite] = lower_bound([float(arm[i]["correct"]) - float(v1[i]["correct"])
                                    for i in sorted(v1)])
    return pairs


def _input_digests(
    eval_dir: Path, observation: Path, observation_v2: Path | None,
    calibration_v2: Path | None,
) -> dict[str, str]:
    """Pin every evaluation input actually present; absent files remain absent."""
    paths = [observation]
    if observation_v2 is not None:
        paths.append(observation_v2)
    if calibration_v2 is not None:
        paths.extend((calibration_v2 / "calibration.jsonl",
                      calibration_v2 / "calibration-manifest.json"))
    for arm in ("dense", "dense_q4", "random128", "reap128",
                "reap128_q4", "reap128_v2", "reap128_v2_q4",
                "reap160", "reap160_q4", "reap160_v2", "reap160_v2_q4"):
        directory = eval_dir / arm
        for name in ("summary.json", "gsm8k.jsonl", "humaneval.jsonl", "serve.json"):
            path = directory / name
            if path.is_file():
                paths.append(path)
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths if path.is_file()}


def evaluate(
    eval_dir: str | Path, observation: str | Path,
    observation_v2: str | Path | None = None,
    calibration_v2: str | Path | None = None,
) -> dict:
    """Compute gates from raw, SHA-attested full-size evaluation rows."""
    root = Path(eval_dir)
    observation_path = Path(observation)
    v2_path = Path(observation_v2) if observation_v2 else None
    v2_calib = Path(calibration_v2) if calibration_v2 else None
    input_sha256 = _input_digests(root, observation_path, v2_path, v2_calib)
    # The shared dense/control inputs must be authenticated *before* any arm
    # computation reads them. Otherwise an altered baseline could trigger a
    # parser exception or influence a gate before its summary SHA was checked.
    for arm, suite, n in (("dense", "gsm8k", 1319), ("dense", "humaneval", 164),
                          ("random128", "gsm8k", 1319), ("dense_q4", "gsm8k", 200)):
        if _attested_rows(root, arm, suite, n) is None:
            raise GateError(f"missing frozen baseline {arm}/{suite}")
    present = {}
    for arm in ("reap128", "reap128_v2", "reap160", "reap160_v2"):
        # A partially present arm is explicitly unresolved, not silently absent.
        entries = [_attested_rows(root, arm, suite, n) for suite, n in
                   (("gsm8k", 1319), ("humaneval", 164))]
        if all(row is None for row in entries):
            continue
        if any(row is None for row in entries):
            raise GateError(f"{arm}: missing full SHA-attested quality evidence")
        speed_arm = f"{arm}_q4"
        _attested_rows(root, speed_arm, "gsm8k", 200)
        # v2 is a separately observed saliency/calibration arm. Never borrow
        # the v1 observer's successful D0 as evidence that v2 passed D0.
        observer = observation_path
        if arm.endswith("_v2"):
            observer = None  # no v1 observer substitution
            if v2_path is not None and v2_calib is not None:
                manifest_path = v2_calib / "calibration-manifest.json"
                rows_path = v2_calib / "calibration.jsonl"
                if not manifest_path.is_file() or not rows_path.is_file():
                    raise GateError("v2 calibration evidence missing manifest or input rows")
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                actual_calib_sha = hashlib.sha256(rows_path.read_bytes()).hexdigest()
                rows_count = sum(1 for line in rows_path.read_text(encoding="utf-8").splitlines()
                                 if line.strip())
                observation_record = json.loads(v2_path.read_text(encoding="utf-8"))
                if (manifest.get("mix") != "v2"
                        or not isinstance(manifest.get("rows"), int)
                        or isinstance(manifest.get("rows"), bool)
                        or manifest["rows"] != rows_count
                        or rows_count == 0
                        or manifest.get("calibration_sha256") != actual_calib_sha
                        or observation_record.get("calibration_sha256") != actual_calib_sha):
                    raise GateError("v2 D0 observer calibration SHA disagrees with the v2 manifest")
                observer = v2_path
        present[arm] = compute(root, arm, "dense", "random128",
                               speed_arm, "dense_q4", observer)
    paired = _paired(root) if "reap128_v2" in present else None
    decision = select(present, paired)
    if _input_digests(root, observation_path, v2_path, v2_calib) != input_sha256:
        raise GateError("REAP evaluation inputs changed while frozen gates were computed")
    return {"schema": "reap-frozen-amendment4-selection-v1", "gates": present,
            "paired_v2_minus_v1": paired, "decision": decision,
            "input_sha256": input_sha256, "release_approved": False,
            "scope": "frozen selection score only; human approval and hardware gates separate"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--observation", required=True)
    parser.add_argument("--observation-v2",
                        help="distinct full v2 observer record; mandatory to pass any v2 D0")
    parser.add_argument("--calibration-v2",
                        help="v2 calibration folder with calibration.jsonl and SHA-pinned manifest")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        result = evaluate(args.eval_dir, args.observation, args.observation_v2,
                          args.calibration_v2)
    except (GateError, ValueError, OSError) as error:
        print(f"REAP frozen selection refused: {error}", file=sys.stderr)
        return 2
    destination = Path(args.out)
    if destination.exists():
        print("REAP frozen selection refused: output already exists", file=sys.stderr)
        return 2
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result["decision"]))
    return 0 if result["decision"]["status"] == "SELECTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
