"""Fail-closed Gen-0 selection under frozen REAP preregistration amendment 4.

Individual arm gates are not selection approval: when REAP128 v1 fails its
quality gate, both REAP128_v2 and REAP160 must be *scored* before choosing an
arm. Never silently promote a later arm because an earlier arm is missing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

try:
    from chowder_batch.reap_gates import GateError, FROZEN_SOURCE_SHA256, _load, compute, lower_bound
    from chowder_batch.reap_v2_source_identity import (
        V2SourceIdentityError, frozen_v2_source_identity,
    )
except ImportError:  # direct script mode outside the repository package path
    from reap_gates import GateError, FROZEN_SOURCE_SHA256, _load, compute, lower_bound
    from reap_v2_source_identity import V2SourceIdentityError, frozen_v2_source_identity

REQUIRED = ("d0", "gate0", "gate1", "gate2", "control", "speed")
MARGIN = -0.02  # frozen strict lower bound for paired v2-v1 comparisons
FROZEN_SERVE_FLAGS = ["--fit", "on", "-np", "4", "-c", "16384", "--jinja"]
FROZEN_SCORING = "frozen reap_eval production scoring: final_number_match/boxed/HumanEval"
FROZEN_INPUT_SCHEMA = "chowder.reap-eval-input-identity.v1"
FROZEN_SOURCE_COUNTS = {"gsm8k": 1319, "humaneval": 164}
FROZEN_MAX_TOKENS = {"gsm8k": 1024, "humaneval": 768}
# §4 explicitly defines LB = mean(d) - 1.645 * sd(d)/sqrt(n), whereas
# Amendment 4 §2 calls for "paired McNemar" one-sided 95% noninferiority at
# -0.02. McNemar's equality test does not uniquely determine a lower CI for
# the paired marginal risk difference at a nonzero margin. In particular,
# Wald, Tango score, Newcombe and exact-compatible limits can disagree.
# Keep the originally coded §4 Wald output as a diagnostic, NOT evidence
# sufficient to certify optional v2 replacement before method adjudication.
PAIRED_METHOD_DISCREPANCY = (
    "frozen Amendment 4 paired McNemar one-sided noninferiority CI "
    "is not uniquely specified against the frozen Section 4 Wald LB; "
    "prospective documented statistical-method resolution required"
)


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
        # These paired Wald bounds remain observable, but cannot by
        # themselves certify the separately named McNemar noninferiority
        # method. This affects only OPTIONAL replacement of passing v1;
        # the frozen Gate0/1-failure fallback order above is a distinct rule.
        return {"status": "BLOCKED", "chosen": None,
                "reason": PAIRED_METHOD_DISCREPANCY}
    return {"status": "SELECTED", "chosen": "reap128", "reason": "v1 retained under amendment 4"}


def _is_new_v2(arm: str) -> bool:
    return arm in {"reap128_v2", "reap128_v2_q4", "reap160_v2", "reap160_v2_q4"}


def _v2_input_witness(eval_dir: Path, arm: str, suite: str) -> dict:
    """Validate the frozen evaluator's new-arm source/model/decode claim."""
    directory = eval_dir / arm
    path = directory / f"{suite}.inputs.json"
    serving = directory / "serve.json"
    if not path.is_file() or not serving.is_file() or path.is_symlink() or serving.is_symlink():
        raise GateError(f"{arm}/{suite}: new v2 evaluation witness or serve.json missing")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        serve = json.loads(serving.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError) as exc:
        raise GateError(f"{arm}/{suite}: v2 input witness/serving receipt is unreadable") from exc
    if not isinstance(record, dict) or record.get("schema") != FROZEN_INPUT_SCHEMA:
        raise GateError(f"{arm}/{suite}: frozen input witness schema mismatch")
    if set(record) != {"schema", "model", "suite", "source_items_sha256",
                       "source_items_n", "decode", "scoring"}:
        raise GateError(f"{arm}/{suite}: frozen input witness schema fields mismatch")
    if record.get("suite") != suite:
        raise GateError(f"{arm}/{suite}: input witness suite mismatch")
    source_sha = record.get("source_items_sha256")
    if (not isinstance(source_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", source_sha)
            or type(record.get("source_items_n")) is not int
            or record["source_items_n"] != FROZEN_SOURCE_COUNTS[suite]):
        raise GateError(f"{arm}/{suite}: frozen source item digest/count invalid")
    if source_sha != FROZEN_SOURCE_SHA256[suite]:
        raise GateError(f"{arm}/{suite}: new v2 source fingerprint differs from frozen official {suite}")
    decode = {"system": "You are a helpful assistant.", "temperature": 0.0,
              "top_k": 1, "seed": 7, "max_tokens": FROZEN_MAX_TOKENS[suite],
              "enable_thinking": False}
    if (record.get("decode") != decode or not isinstance(record["decode"], dict)
            or any(type(record["decode"][key]) is not type(value)
                   for key, value in decode.items())):
        raise GateError(f"{arm}/{suite}: frozen decode policy mismatch")
    if record.get("scoring") != FROZEN_SCORING:
        raise GateError(f"{arm}/{suite}: frozen scoring policy mismatch")
    if not isinstance(serve, dict) or serve.get("flags") != FROZEN_SERVE_FLAGS:
        raise GateError(f"{arm}/{suite}: frozen serving flags mismatch")
    model = record.get("model")
    if (not isinstance(model, dict) or set(model) != {"gguf", "gguf_sha256", "flags"}
            or not isinstance(model.get("gguf"), str) or not model["gguf"].strip()
            or not isinstance(model.get("gguf_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", model["gguf_sha256"])
            or any(serve.get(key) != model[key] for key in model)):
        raise GateError(f"{arm}/{suite}: input witness model identity disagrees with serving receipt")
    return record


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
    if _is_new_v2(arm):
        # The official source-population fingerprint in .inputs.json binds
        # the *questions*, but alone does not prove that the saved answers
        # correspond to the canonical item IDs. Full cohorts and the first
        # 200 speed items must use the actual frozen dataset ordering.
        if suite == "gsm8k" and expected in (1319, 200):
            frozen_ids = {f"gsm8k-{i}" for i in range(expected)}
        elif suite == "humaneval" and expected == 164:
            frozen_ids = {f"HumanEval/{i}" for i in range(164)}
        else:
            raise GateError(f"{arm}/{suite}: unregistered v2 population requested")
        if set(rows) != frozen_ids:
            raise GateError(f"{arm}/{suite}: item IDs do not match the frozen official cohort")
        _v2_input_witness(eval_dir, arm, suite)
    return rows


def _paired(eval_dir: Path) -> dict | None:
    """Diagnostic §4 paired-difference Wald bounds; NOT a McNemar NI CI."""
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
        for name in ("summary.json", "gsm8k.jsonl", "humaneval.jsonl", "serve.json",
                     "gsm8k.inputs.json", "humaneval.inputs.json"):
            path = directory / name
            if path.is_file():
                paths.append(path)
        if _is_new_v2(arm) and arm.endswith("_q4"):
            log = directory / "server.log"
            if log.exists() and (log.is_symlink() or not log.is_file()):
                raise GateError(f"{arm}: Q4 original startup log is not a regular file")
            if log.is_file():
                paths.append(log)
    # Snapshot serving log bytes as well as the receipt; otherwise a log that
    # changes after the speed gate check could escape the final drift check.
    digests = {}
    for path in paths:
        if path.is_file():
            h = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1 << 20), b""):
                    h.update(chunk)
            digests[str(path)] = h.hexdigest()
    return digests


def _validate_v2_observation(calibration: Path, observation: Path) -> None:
    """Bind v2 D0 to the complete declared calibration, not a limited observer run."""
    manifest_path = calibration / "calibration-manifest.json"
    rows_path = calibration / "calibration.jsonl"
    if not manifest_path.is_file() or not rows_path.is_file():
        raise GateError("v2 calibration evidence missing manifest or input rows")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    observed = json.loads(observation.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(observed, dict):
        raise GateError("v2 calibration manifest or observer is not an object")
    calibration_sha = hashlib.sha256(rows_path.read_bytes()).hexdigest()
    if (manifest.get("mix") != "v2"
            or manifest.get("calibration_sha256") != calibration_sha
            or observed.get("calibration_sha256") != calibration_sha):
        raise GateError("v2 D0 observer calibration SHA disagrees with the v2 manifest")

    # Amendment 4 names reap_calibration.py --mix v2, whose preregistered
    # Ornith rendering is "bare". An explicit "worked" override changes the
    # calibration population and needs a separate preregistered variant.
    ornith = manifest.get("ornith")
    if isinstance(ornith, dict):
        if ornith.get("render") != "bare":
            raise GateError(
                "v2 calibration Ornith render is not preregistered: expected 'bare'; "
                "the '--ornith-render worked' variant needs a dated preregistration"
            )
    elif ornith != "absent: no admitted Ornith curriculum supplied":
        raise GateError("v2 calibration Ornith rendering/absence declaration is missing or invalid")

    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    def integer(value: object) -> bool:
        return isinstance(value, int) and not isinstance(value, bool)
    declared_rows = manifest.get("rows")
    declared_tokens = manifest.get("tokens")
    max_tokens = manifest.get("max_tokens")
    sources = manifest.get("sources")
    if (not integer(declared_rows) or declared_rows < 1 or len(rows) != declared_rows
            or not integer(declared_tokens) or declared_tokens < 1
            or not integer(max_tokens) or max_tokens < 256
            or not isinstance(sources, dict) or not sources):
        raise GateError("v2 calibration manifest has invalid row/token/source census")
    if not isinstance(ornith, dict) and "ornith_curriculum" in sources:
        raise GateError("v2 calibration declares Ornith rows despite absent admitted curriculum")
    source_counts: dict[str, int] = {}
    tokens = 0
    for row in rows:
        if not isinstance(row, dict):
            raise GateError("v2 calibration contains a non-object row")
        source, count = row.get("source"), row.get("n_tokens")
        if (not isinstance(source, str) or source not in sources
                or not integer(count) or not 256 <= count <= max_tokens
                or not isinstance(row.get("text"), str) or not row["text"]):
            raise GateError("v2 calibration row has invalid source, text or token count")
        source_counts[source] = source_counts.get(source, 0) + 1
        tokens += count
    if tokens != declared_tokens:
        raise GateError("v2 calibration token total disagrees with manifest")
    for name, stats in sources.items():
        if (not isinstance(stats, dict) or not integer(stats.get("kept"))
                or stats["kept"] != source_counts.get(name, 0)):
            raise GateError(f"v2 calibration source count disagrees for {name!r}")

    # The observer may retokenize decoded calibration text; its token count
    # need not equal the packed IDs, but it must prove all sequences were read.
    if (not integer(observed.get("n_seqs")) or observed["n_seqs"] != declared_rows
            or not integer(observed.get("n_tokens"))
            or observed["n_tokens"] < declared_rows):
        raise GateError("v2 D0 observer did not process the full calibration sequence census")
    # SHA-consistency alone cannot establish preregistration: otherwise a
    # rehashed 3-row corpus or invented 13-row bare Ornith suffix could pass.
    try:
        frozen_v2_source_identity(calibration, manifest)
    except V2SourceIdentityError as exc:
        raise GateError(str(exc)) from exc
    # The actual preregistered v2 source contains exactly 704 sequences.
    # No completed 704-row legacy in-RAM D0 is qualified: the historical r2
    # in-place slab was interrupted and the current registered observer uses
    # two committed BF16 buffers. For that exact registered population,
    # stripping BOTH receipt-identifying fields cannot silently downgrade
    # its gate to an unsealed legacy observer.
    if declared_rows == 704 and not isinstance(observed.get("checkpoint_receipt"), dict):
        raise GateError("registered 704-row v2 D0 requires sealed BF16 checkpoint receipt")
    if "hidden_storage" in observed or "checkpoint_receipt" in observed:
        from chowder_batch.reap_hidden_checkpoint import validate_observation_receipt
        try:
            validate_observation_receipt(
                observation.parent, observed,
                require_registered_original=(declared_rows == 704),
            )
        except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
            raise GateError(f"v2 BF16 checkpoint provenance invalid: {exc}") from exc


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
        speed_rows = _attested_rows(root, speed_arm, "gsm8k", 200)
        if _is_new_v2(arm) and speed_rows is not None:
            quality_source = _v2_input_witness(root, arm, "gsm8k")
            speed_source = _v2_input_witness(root, speed_arm, "gsm8k")
            if (quality_source["source_items_sha256"] != speed_source["source_items_sha256"]
                    or quality_source["source_items_n"] != speed_source["source_items_n"]
                    or not set(speed_rows).issubset(entries[0])):
                raise GateError(f"{arm}: Q8/Q4 source item fingerprints or paired GSM8K IDs mismatch")
        # v2 is a separately observed saliency/calibration arm. Never borrow
        # the v1 observer's successful D0 as evidence that v2 passed D0.
        observer = observation_path
        if arm.endswith("_v2"):
            observer = None  # no v1 observer substitution
            if v2_path is not None and v2_calib is not None:
                _validate_v2_observation(v2_calib, v2_path)
                observer = v2_path
        present[arm] = compute(root, arm, "dense", "random128",
                               speed_arm, "dense_q4", observer)
    paired = _paired(root) if "reap128_v2" in present else None
    decision = select(present, paired)
    if _input_digests(root, observation_path, v2_path, v2_calib) != input_sha256:
        raise GateError("REAP evaluation inputs changed while frozen gates were computed")
    return {"schema": "reap-frozen-amendment4-selection-v1", "gates": present,
            "paired_v2_minus_v1": paired, "decision": decision,
            "paired_method": {
                "status": "UNRESOLVED" if "reap128_v2" in present else "NOT_EVALUATED",
                "scope": "optional v2 replacement of passing v1; frozen mandatory fallback unchanged",
                "reported_statistic": "Section 4 paired-difference Wald one-sided 95% LB",
                "frozen_margin": MARGIN,
                "amendment_4_method": "paired McNemar one-sided 95% LB",
                "qualification": "NOT_CERTIFIED: " + PAIRED_METHOD_DISCREPANCY,
            },
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
