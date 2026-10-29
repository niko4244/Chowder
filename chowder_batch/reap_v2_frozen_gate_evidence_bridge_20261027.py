"""Read-only admission bridge for the *original* frozen v2 selector.

Source-controlled approval SHA pins are intentionally EMPTY. An ordinary
receipt, its self-reported PASS, a caller-provided hash, or the completed
research-only Q4/Q8 HumanEval runs cannot authorize a frozen Gen-0 gate.
Only independently reviewed, source-pinned native original evidence can be
passed to the unchanged ``reap_select.evaluate`` and its deterministic selector.

This module performs no model inference, generated-code execution, networking,
container launch, file mutation, or release authorization. Tests replace pins
with explicitly SYNTHETIC adversarial specimens in isolated test processes.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

from chowder_batch import reap_gates, reap_select, reap_serve

SCHEMA = "chowder.reap-v2-frozen-gate-evidence-bridge.v1"
REVIEW_SCHEMA = "chowder.reap-v2-independent-original-frozen-review.v1"
SECURITY_SCHEMA = "chowder.reap-v2-independent-frozen-humaneval-security.v1"
HISTORICAL_REAP160_SCHEMA = "chowder.reap-v2-independent-historical-reap160-prerequisite.v1"

Q8_V2_SHA256 = "89dcaf33b61f59c83a31b718ba98b73c33437d0320ae0cc56737a5d7e43519c6"
Q4_DENSE_SHA256 = "2c6d9466129e8802839566b35728e2506aaf1e8705cd137009e2f08feb9dd049"

COHORTS = {
    "reap128_v2": {"gsm8k": 1319, "humaneval": 164},
    "dense_v2": {"gsm8k": 1319, "humaneval": 164},
    "random128_v2": {"gsm8k": 1319},
    "reap128_v2_q4": {"gsm8k": 200},
    "dense_v2_q4": {"gsm8k": 200},
}
# Historical rows predate the v2 .inputs.json witness. They still must use
# the original registered task IDs and scoring semantics. Validate them at
# this independent admission boundary without modifying the frozen selector.
HISTORICAL_COHORTS = {
    "dense": {"gsm8k": 1319, "humaneval": 164},
    "random128": {"gsm8k": 1319},
    "reap128": {"gsm8k": 1319, "humaneval": 164},
    "reap160": {"gsm8k": 1319, "humaneval": 164},
    "dense_q4": {"gsm8k": 200},
    "reap128_q4": {"gsm8k": 200},
    "reap160_q4": {"gsm8k": 200},
}
MODEL_SHA256 = {
    "reap128_v2": Q8_V2_SHA256,
    "dense_v2": reap_select.FROZEN_REFERENCE_MODEL_SHA256["dense_v2"],
    "random128_v2": reap_select.FROZEN_REFERENCE_MODEL_SHA256["random128_v2"],
    "reap128_v2_q4": reap_gates.PUBLISHED_REAP128_V2_Q4_SHA256,
    "dense_v2_q4": Q4_DENSE_SHA256,
}

# Source-owned pins must be entered only after independent review of *actual*
# original frozen model, scorer, native serving and benchmark records. The map
# keys are absolute artifact paths; changing it at runtime is test-only.
APPROVED_ORIGINAL_ARTIFACT_SHA256: dict[str, str] = {}
APPROVED_INDEPENDENT_REVIEW_SHA256: str | None = None
APPROVED_HUMANEVAL_SECURITY_SHA256: dict[str, str] = {}
# Each per-arm input manifest is separately source-approved by an independent
# reviewer. The certifier also demands its OWN source-controlled independent
# external trace reviewer pins. A JSON's self-reported digest or success label
# can never install either authority; both production maps remain EMPTY.
APPROVED_HUMANEVAL_CERTIFICATE_INPUT_SHA256: dict[str, str] = {}
# REAP160 is a mandatory *scored prerequisite* after historical v1 Gate0/1
# failure, even if the frozen selector ultimately chooses REAP128_v2 first.
# Neither an archived summary nor a selected-v2 decision authenticates its
# original model, native serving, per-item scoring, or first200 speed. No
# production historical REAP160 authority has yet been independently admitted.
APPROVED_HISTORICAL_REAP160_REVIEW_SHA256: str | None = None
APPROVED_HISTORICAL_REAP160_MODEL_SHA256: dict[str, str] = {}
APPROVED_HISTORICAL_SPEED_SERVER_LOG_SHA256: dict[str, str] = {}
# Historical reap_eval._score never wrote eval_session_id into original JSONL.
# Independently observed per-item native lineage must therefore be separately
# source-approved; NEVER retrofit or reserialize the immutable scored rows.
APPROVED_HISTORICAL_ITEM_LINEAGE_SHA256: dict[str, str] = {}
CERTIFICATE_INPUT_FIELDS = frozenset({
    "capture_path", "witness_path", "capability_path", "broker_path",
    "review_path", "trace_dir", "serve_path", "owner_review_path",
})


class FrozenEvidenceError(ValueError):
    """An original scientific or security prerequisite is unverified."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise FrozenEvidenceError(reason)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _hex(value: object, width: int = 64) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{" + str(width) + r"}", value) is not None


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict:
    """Refuse ambiguous independent receipts, including nested duplicate keys."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate independent receipt JSON key: {key}")
        result[key] = value
    return result


def _json_bytes(raw: bytes, *, max_bytes: int = 512 * 1024) -> dict:
    """Parse the exact bytes independently hashed, never reopen after approval."""
    _require(0 < len(raw) <= max_bytes, "independent receipt size invalid")
    try:
        result = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite")),
        )
    except (ValueError, UnicodeError) as exc:
        raise FrozenEvidenceError("independent receipt JSON invalid") from exc
    _require(isinstance(result, dict), "independent receipt must be a JSON object")
    return result


def _json_file(path: Path, *, max_bytes: int = 512 * 1024) -> dict:
    _require(path.is_file() and not path.is_symlink(), f"missing independent receipt: {path.name}")
    _require(0 < path.stat().st_size <= max_bytes, "independent receipt size invalid")
    return _json_bytes(path.read_bytes(), max_bytes=max_bytes)


def _verify_pins(
    root: Path, observation: Path, observation_v2: Path | None,
    calibration_v2: Path | None,
) -> dict[str, str]:
    _require(bool(APPROVED_ORIGINAL_ARTIFACT_SHA256), "independently approved artifact pins absent")
    _require(_hex(APPROVED_INDEPENDENT_REVIEW_SHA256), "independent protocol review pin absent")
    _require(set(APPROVED_HUMANEVAL_SECURITY_SHA256) == {"reap128_v2", "dense_v2"}
             and all(_hex(v) for v in APPROVED_HUMANEVAL_SECURITY_SHA256.values()),
             "independent original HumanEval security approvals absent")
    _require(root.is_dir() and not root.is_symlink(), "original frozen evaluation root invalid")

    # Snapshot *every* source file that the unchanged selector will consume.
    # A new unpinned candidate/control/score is a refusal, never a new PASS.
    observed = reap_select._input_digests(root, observation, observation_v2, calibration_v2)
    required = {observation}
    _require((observation_v2 is None) == (calibration_v2 is None),
             "v2 D0 observer and frozen calibration must be supplied together")
    if observation_v2 is not None and calibration_v2 is not None:
        required.update({observation_v2, calibration_v2 / "calibration.jsonl",
                         calibration_v2 / "calibration-manifest.json"})
    for arm, suites in COHORTS.items():
        directory = root / arm
        _require(not (directory / ".serve.lock").exists()
                 and not any(directory.glob("serve.resume-*.json"))
                 and not any(directory.glob("server.resume-*.log"))
                 and not any(directory.glob(".serve-pending-*.json")),
                 f"{arm}: resumed or incomplete native serving session cannot be certified")
        required.add(directory / "serve.json")
        required.add(directory / "summary.json")
        for suite in suites:
            required.add(directory / f"{suite}.jsonl")
            required.add(directory / f"{suite}.inputs.json")
        if arm.endswith("_q4"):
            required.add(directory / "server.log")
    _require(all(path.is_file() and not path.is_symlink() for path in required),
             "original Q8/dense/control/Q4 source or serving cohort incomplete")
    expected = APPROVED_ORIGINAL_ARTIFACT_SHA256
    _require(all(_hex(v) for v in expected.values()), "invalid source-owned artifact SHA pin")
    _require(set(expected) == set(observed) and required.issubset({Path(p) for p in observed}),
             "original selector artifacts not fully pinned or an unreviewed artifact appeared")
    for name, digest in observed.items():
        path = Path(name)
        _require(not path.is_symlink() and path.is_file() and digest == expected[name],
                 f"independently approved source SHA mismatch: {path.name}")
    return observed


def _verify_historical_canonical_rows(root: Path) -> None:
    """Reject re-sealed legacy rows with forged IDs or impossible correctness.

    Original frozen ``reap_select`` checks canonical IDs for new v2 arms,
    not historical arms. This independent bridge must not allow two legacy
    arms with matching *noncanonical* IDs to create a valid-looking pairing.
    """
    for arm, suites in HISTORICAL_COHORTS.items():
        for suite, count in suites.items():
            path = root / arm / f"{suite}.jsonl"
            if not path.exists():
                continue
            _require(path.is_file() and not path.is_symlink(),
                     f"{arm}/{suite}: historical row path invalid")
            rows = reap_select._attested_rows(root, arm, suite, count)
            _require(rows is not None, f"{arm}/{suite}: historical source summary absent")
            canonical = ({f"gsm8k-{i}" for i in range(count)} if suite == "gsm8k"
                         else {f"HumanEval/{i}" for i in range(count)})
            _require(set(rows) == canonical,
                     f"{arm}/{suite}: historical task IDs are not canonical")
            _require(all(type(row.get("correct")) is bool
                         and type(row.get("terminated")) is bool
                         and not (row["correct"] and not row["terminated"])
                         for row in rows.values()),
                     f"{arm}/{suite}: historical scorer semantics invalid")


def _verify_historical_item_lineage(
    root: Path, arm: str, suite: str, rows: dict, serve: dict, record: dict,
) -> None:
    """Join independently source-approved native traces to untouched score rows."""
    key = f"{arm}/{suite}"
    pinned = APPROVED_HISTORICAL_ITEM_LINEAGE_SHA256.get(key)
    declared = record.get("item_lineage_sha256")
    _require(_hex(pinned) and isinstance(declared, dict)
             and declared.get(suite) == pinned,
             f"{key}: independent original native item-lineage approval absent")
    path = root / arm / f"{suite}.native-lineage.jsonl"
    _require(path.is_file() and not path.is_symlink(),
             f"{key}: original native item-lineage trace missing")
    raw = path.read_bytes()
    _require(0 < len(raw) <= 4 * 1024 * 1024 and _sha(raw) == pinned,
             f"{key}: original native item-lineage SHA mismatch")
    lines = raw.splitlines(keepends=True)
    _require(len(lines) == len(rows), f"{key}: original native item-lineage incomplete")
    fields = {"id", "eval_session_id", "gguf_sha256", "server_pid", "server_port",
              "native_response_sha256", "scored_row_sha256", "completion_tokens",
              "seconds", "predicted_per_second"}
    for (task_id, row), line in zip(rows.items(), lines, strict=True):
        _require(line.endswith(b"\n") and not line.startswith(b"\xef\xbb\xbf")
                 and b"\r" not in line.rstrip(b"\r\n"),
                 f"{key}: original native item-lineage framing invalid")
        try:
            trace = json.loads(line.decode("utf-8"),
                               object_pairs_hook=_reject_duplicate_json_keys,
                               parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite")))
        except (ValueError, UnicodeError) as exc:
            raise FrozenEvidenceError(f"{key}: original native item-lineage JSON invalid") from exc
        scored = json.dumps(row, sort_keys=True, separators=(",", ":"),
                            ensure_ascii=False, allow_nan=False).encode("utf-8")
        _require(isinstance(trace, dict) and set(trace) == fields
                 and trace["id"] == task_id
                 and trace["eval_session_id"] == serve["eval_session_id"]
                 and trace["gguf_sha256"] == serve["gguf_sha256"]
                 and type(trace["server_pid"]) is int
                 and trace["server_pid"] == serve["server_pid"]
                 and type(trace["server_port"]) is int
                 and trace["server_port"] == serve["server_port"]
                 and isinstance(row.get("text"), str)
                 and _hex(trace["native_response_sha256"])
                 and trace["native_response_sha256"] == _sha(row["text"].encode("utf-8"))
                 and _hex(trace["scored_row_sha256"])
                 and trace["scored_row_sha256"] == _sha(scored)
                 and type(trace["completion_tokens"]) is int
                 and trace["completion_tokens"] == row.get("completion_tokens")
                 and type(trace["seconds"]) in (float, int)
                 and math.isfinite(trace["seconds"])
                 and trace["seconds"] == row.get("seconds")
                 and (trace["predicted_per_second"] is None
                      or (type(trace["predicted_per_second"]) in (float, int)
                          and math.isfinite(trace["predicted_per_second"])))
                 and trace["predicted_per_second"] == row.get("predicted_per_second"),
                 f"{key}/{task_id}: original native item-lineage disagrees with score")


def _verify_historical_reap160_prerequisite(
    root: Path, *, forbidden_sessions: frozenset[str],
) -> None:
    """Require separately pinned native REAP160 evidence before v2 fallback selection.

    The original frozen selector legitimately computes from historical rows;
    this independent admission boundary must additionally authenticate the
    historical *prerequisite* rather than only the arm ultimately chosen.
    This never changes the frozen selector or its fallback order.
    """
    # The historical fallback is a *paired* quality/control comparison, not
    # an isolated REAP160 score. Its dense and random128 Q8 reference rows
    # must carry independently approved model/session/host provenance too.
    arms = {"reap160", "dense", "random128", "reap160_q4", "dense_q4"}
    quality_arms = {"reap160", "dense", "random128"}
    speed_arms = {"reap160_q4", "dense_q4"}
    expected_lineage = {f"{arm}/{suite}" for arm, suites in (
        ("reap160", ("gsm8k", "humaneval")),
        ("dense", ("gsm8k", "humaneval")),
        ("random128", ("gsm8k",)),
        ("reap160_q4", ("gsm8k",)),
        ("dense_q4", ("gsm8k",)),
    ) for suite in suites}
    _require(_hex(APPROVED_HISTORICAL_REAP160_REVIEW_SHA256)
             and set(APPROVED_HISTORICAL_ITEM_LINEAGE_SHA256) == expected_lineage
             and all(_hex(value) for value in APPROVED_HISTORICAL_ITEM_LINEAGE_SHA256.values())
             and set(APPROVED_HISTORICAL_SPEED_SERVER_LOG_SHA256) == speed_arms
             and all(_hex(value) for value in APPROVED_HISTORICAL_SPEED_SERVER_LOG_SHA256.values())
             and set(APPROVED_HISTORICAL_REAP160_MODEL_SHA256) == arms
             and APPROVED_HISTORICAL_REAP160_MODEL_SHA256.get("dense_q4") == Q4_DENSE_SHA256
             and all(_hex(value) for value in APPROVED_HISTORICAL_REAP160_MODEL_SHA256.values()),
             "historical REAP160 prerequisite has no independent source-owned approval")
    path = root / "reap160.frozen-prerequisite-review.json"
    _require(path.is_file() and not path.is_symlink(),
             "historical REAP160 independent native review missing")
    raw = path.read_bytes()
    _require(_sha(raw) == APPROVED_HISTORICAL_REAP160_REVIEW_SHA256,
             "historical REAP160 independent review SHA mismatch")
    review = _json_bytes(raw)
    _require(review.get("schema") == HISTORICAL_REAP160_SCHEMA
             and review.get("status") == "INDEPENDENTLY_APPROVED_ORIGINAL_REAP160_PREREQUISITE"
             and review.get("native_windows_server_owner_verified") is True
             and review.get("independent_model_bytes_verified") is True
             and review.get("per_item_session_binding_verified") is True
             and review.get("original_executable_scorer_verified") is True
             and review.get("same_hardware_speed_pair_verified") is True
             and review.get("original_q4_offload_fit_verified") is True
             and review.get("native_first200_speed_verified") is True
             and review.get("canonical_source_sha256") == reap_gates.FROZEN_SOURCE_SHA256
             and review.get("original_scored_counts") == {
                 "gsm8k": 1319, "humaneval": 164, "first200_speed": 200,
             }
             and set(review.get("arms", {})) == arms,
             "historical REAP160 native prerequisite review incomplete")
    for arm, suites in (("reap160", ("gsm8k", "humaneval")),
                        ("dense", ("gsm8k", "humaneval")),
                        ("random128", ("gsm8k",)),
                        ("reap160_q4", ("gsm8k",)),
                        ("dense_q4", ("gsm8k",))):
        record = review["arms"][arm]
        _require(isinstance(record, dict), "historical REAP160 arm review malformed")
        serve_path = root / arm / "serve.json"
        serve = _json_file(serve_path)
        session = serve.get("eval_session_id")
        directory = root / arm
        _require(not (directory / ".serve.lock").exists()
                 and not any(directory.glob("serve.resume-*.json"))
                 and not any(directory.glob("server.resume-*.log"))
                 and not any(directory.glob(".serve-pending-*.json")),
                 f"{arm}: historical serving session resumed or incomplete")
        _require(_hex(session, 32)
                 and serve.get("gguf_sha256") == APPROVED_HISTORICAL_REAP160_MODEL_SHA256[arm]
                 and serve.get("flags") == reap_select.FROZEN_SERVE_FLAGS
                 and type(serve.get("server_pid")) is int and serve["server_pid"] > 0
                 and type(serve.get("server_port")) is int
                 and 0 < serve["server_port"] < 65536
                 and APPROVED_ORIGINAL_ARTIFACT_SHA256.get(str(serve_path))
                 == _sha(serve_path.read_bytes())
                 and record.get("model_sha256") == serve["gguf_sha256"]
                 and record.get("session_id") == session
                 and _hex(record.get("physical_host_sha256"))
                 and _hex(record.get("hardware_profile_sha256"))
                 and isinstance(record.get("rows_sha256"), dict)
                 and set(record["rows_sha256"]) == set(suites)
                 and isinstance(record.get("item_lineage_sha256"), dict)
                 and set(record["item_lineage_sha256"]) == set(suites),
                 f"{arm}: historical native model/session/rows provenance incomplete")
        source_rows = {}
        for suite in suites:
            rows_path = root / arm / f"{suite}.jsonl"
            _require(rows_path.is_file() and not rows_path.is_symlink()
                     and record["rows_sha256"][suite] == _sha(rows_path.read_bytes())
                     and APPROVED_ORIGINAL_ARTIFACT_SHA256.get(str(rows_path))
                     == record["rows_sha256"][suite],
                     f"{arm}/{suite}: original historical REAP160 rows not independently bound")
            expected_n = 200 if arm in speed_arms else (1319 if suite == "gsm8k" else 164)
            rows = reap_select._attested_rows(root, arm, suite, expected_n)
            _require(isinstance(rows, dict)
                     and all(("eval_session_id" not in row
                              or row["eval_session_id"] == session)
                             for row in rows.values()),
                     f"{arm}/{suite}: historical original per-item serving session mismatch")
            _verify_historical_item_lineage(root, arm, suite, rows, serve, record)
            source_rows[suite] = rows
        if arm in speed_arms:
            offloaded = serve.get("experts_offloaded")
            _require(type(offloaded) is int and offloaded >= 0
                     and (arm != "dense_q4" or offloaded == 0)
                     and serve.get("measured_experts_offloaded") == offloaded,
                     f"{arm}: historical original fit/offload receipt invalid")
            native_log = directory / "server.log"
            _require(native_log.is_file() and not native_log.is_symlink(),
                     f"{arm}: historical Q4 native server log missing")
            native_log_raw = native_log.read_bytes()
            _require(_sha(native_log_raw)
                     == APPROVED_HISTORICAL_SPEED_SERVER_LOG_SHA256[arm]
                     and review.get("native_q4_server_log_sha256")
                     == APPROVED_HISTORICAL_SPEED_SERVER_LOG_SHA256,
                     f"{arm}: historical Q4 native server log not independently pinned")
            # Parse the very bytes authenticated above. A review's claim of
            # successful --fit, or two matching JSON offload counts, cannot
            # substitute for the original native placement/fit log.
            startup = native_log_raw
            if "server_log_startup_sha256" in serve:
                size = serve.get("server_log_startup_bytes")
                _require(type(size) is int
                         and 1 <= size <= min(len(native_log_raw), 64 * 1024 * 1024)
                         and _sha(native_log_raw[:size]) == serve["server_log_startup_sha256"],
                         f"{arm}: historical native startup log prefix mismatch")
                startup = native_log_raw[:size]
            lines, measured = reap_serve.parse_offload(
                startup.decode("utf-8", errors="replace")
            )
            # parse_offload may infer a count from n_cpu_moe before the server
            # has finished fitting or loading even one model layer. A count
            # alone is not evidence that a native model was successfully fit.
            fit_complete = any("successfully fit" in line.lower() for line in lines)
            loaded_layers = any(reap_serve._ALL_LAYERS.search(line) for line in lines)
            _require(fit_complete and loaded_layers
                     and measured == offloaded and serve.get("offload_log_lines") == lines,
                     f"{arm}: historical native fit/offload log disagrees with serving receipt")
            rows = source_rows["gsm8k"]
            _require(isinstance(rows, dict) and len(rows) == 200
                     and all(type(row.get("completion_tokens")) is int
                             and row["completion_tokens"] > 0
                             and type(row.get("seconds")) in (int, float)
                             and math.isfinite(row["seconds"]) and row["seconds"] > 0
                             and type(row.get("predicted_per_second")) in (int, float)
                             and math.isfinite(row["predicted_per_second"])
                             and row["predicted_per_second"] > 0
                             for row in rows.values()),
                     f"{arm}: historical original first200 native timing rows incomplete")
    _require(all(
        review["arms"][arm]["physical_host_sha256"]
        == review["arms"]["reap160"]["physical_host_sha256"]
        and review["arms"][arm]["hardware_profile_sha256"]
        == review["arms"]["reap160"]["hardware_profile_sha256"]
        for arm in quality_arms
    ), "historical REAP160/dense/random128 Q8 quality comparator host or profile differs")
    _require(review["arms"]["reap160_q4"]["physical_host_sha256"]
             == review["arms"]["dense_q4"]["physical_host_sha256"]
             and review["arms"]["reap160_q4"]["hardware_profile_sha256"]
             == review["arms"]["dense_q4"]["hardware_profile_sha256"]
             and len({review["arms"][arm]["session_id"] for arm in arms}) == len(arms),
             "historical REAP160/dense Q4 speed comparator host or sessions differ")
    # Separate historical and v2 reviewers each prove internal uniqueness,
    # but neither alone proves that a purported *fresh* v2 measurement did
    # not replay an older REAP160/dense/random serving session. In particular,
    # dense_q4 and dense_v2_q4 may share model bytes but must not share the
    # same evaluation session for the required fresh speed comparison.
    historical_sessions = {review["arms"][arm]["session_id"] for arm in arms}
    _require(not (historical_sessions & forbidden_sessions),
             "historical REAP160 and fresh v2 serving sessions overlap")


def _verify_original_sessions(root: Path, review: dict) -> dict[str, dict]:
    _require(review.get("schema") == REVIEW_SCHEMA
             and review.get("status") == "INDEPENDENTLY_APPROVED_ORIGINAL_FROZEN_EVIDENCE"
             and review.get("native_windows_server_owner_verified") is True
             and review.get("same_hardware_speed_pair_verified") is True
             and review.get("no_cross_session_row_pooling") is True
             and set(review.get("arms", {})) == set(COHORTS),
             "original independent Windows serving review not approved")
    approvals = review["arms"]
    serving = {}
    for arm, suites in COHORTS.items():
        record = approvals[arm]
        _require(isinstance(record, dict), "original native arm approval invalid")
        serve = _json_file(root / arm / "serve.json")
        source_model = MODEL_SHA256[arm]
        session = serve.get("eval_session_id")
        _require(_hex(session, 32)
                 and serve.get("flags") == reap_select.FROZEN_SERVE_FLAGS
                 and serve.get("gguf_sha256") == source_model
                 and type(serve.get("server_pid")) is int and serve["server_pid"] > 0
                 and type(serve.get("server_port")) is int and 0 < serve["server_port"] < 65536,
                 f"{arm}: native original model, serving geometry or session invalid")
        _require(record.get("model_sha256") == source_model
                 and record.get("session_id") == session
                 and _hex(record.get("hardware_profile_sha256"))
                 and _hex(record.get("physical_host_sha256"))
                 and record.get("independent_model_bytes_verified") is True
                 and record.get("native_windows_server_owner_verified") is True
                 and record.get("per_item_session_binding_verified") is True
                 and record.get("nonresumed_complete_cohort") is True
                 and isinstance(record.get("rows_sha256"), dict)
                 and set(record.get("rows_sha256", {})) == set(suites),
                 f"{arm}: independent original model and item-session provenance incomplete")
        for suite, n in suites.items():
            # The existing source witness validates official item fingerprint,
            # frozen prompt/decode, scoring and independently pinned model SHAs.
            witness = reap_select._v2_input_witness(root, arm, suite)
            rows = reap_select._attested_rows(root, arm, suite, n)
            _require(rows is not None, f"{arm}/{suite}: original rows absent")
            expected_ids = ({f"gsm8k-{i}" for i in range(n)} if suite == "gsm8k"
                            else {f"HumanEval/{i}" for i in range(n)})
            _require(set(rows) == expected_ids
                     and witness["model"]["gguf_sha256"] == source_model
                     and record["rows_sha256"].get(suite)
                     == _sha((root / arm / f"{suite}.jsonl").read_bytes()),
                     f"{arm}/{suite}: independent original per-item receipt mismatch")
            _require(all(type(row.get("correct")) is bool
                         and type(row.get("terminated")) is bool
                         and not (row["correct"] and not row["terminated"])
                         and ("eval_session_id" not in row or row["eval_session_id"] == session)
                         for row in rows.values()),
                     f"{arm}/{suite}: scorer semantics or recorded item session mismatch")
        serving[arm] = {"session_id": session,
                        "hardware_profile_sha256": record["hardware_profile_sha256"],
                        "physical_host_sha256": record["physical_host_sha256"]}
    _require(serving["reap128_v2_q4"]["hardware_profile_sha256"]
             == serving["dense_v2_q4"]["hardware_profile_sha256"]
             and serving["reap128_v2_q4"]["physical_host_sha256"]
             == serving["dense_v2_q4"]["physical_host_sha256"],
             "original first200 Q4 speed runs do not share an approved physical host/profile")
    _require(len({item["session_id"] for item in serving.values()}) == len(serving),
             "two incompatible original model weights share one claimed serving session")
    _require(all(
        serving[arm]["physical_host_sha256"] == serving["reap128_v2"]["physical_host_sha256"]
        and serving[arm]["hardware_profile_sha256"]
        == serving["reap128_v2"]["hardware_profile_sha256"]
        for arm in ("dense_v2", "random128_v2")
    ), "original full1319 Q8/dense/random comparisons do not share physical host/profile")
    return serving


def _verify_humaneval_security(root: Path, serving: dict[str, dict]) -> None:
    for arm in ("reap128_v2", "dense_v2"):
        path = root / f"{arm}.frozen-humaneval-security.json"
        _require(path.is_file() and not path.is_symlink(), "original HumanEval security review absent")
        raw = path.read_bytes()
        _require(_sha(raw) == APPROVED_HUMANEVAL_SECURITY_SHA256[arm],
                 "original independently approved HumanEval security SHA mismatch")
        proof = _json_bytes(raw)
        controls = proof.get("controls")
        _require(proof.get("schema") == SECURITY_SCHEMA
                 and proof.get("status") == "INDEPENDENTLY_APPROVED_FROZEN_GATE2_EXECUTION"
                 and proof.get("arm") == arm
                 and proof.get("model_sha256") == MODEL_SHA256[arm]
                 and proof.get("session_id") == serving[arm]["session_id"]
                 and proof.get("canonical_source_sha256") == reap_gates.FROZEN_SOURCE_SHA256["humaneval"]
                 and proof.get("original_rows_sha256")
                 == _sha((root / arm / "humaneval.jsonl").read_bytes())
                 and type(proof.get("source_bound_cases")) is int
                 and proof["source_bound_cases"] == 164
                 and _hex(proof.get("independent_observer_implementation_sha256"))
                 and isinstance(controls, dict)
                 and set(controls) == {
                     "original_python_I_ten_second_timeout", "per_case_original_program_sha_verified",
                     "isolated_nonprivileged_process", "network_egress_denied",
                     "host_filesystem_and_secrets_denied", "memory_and_pid_limits_enforced",
                     "independent_process_tree_extinction", "no_unreviewed_generated_code_execution",
                 }
                 and all(value is True for value in controls.values()),
                 f"{arm}: original independent execution sandbox not approved")


def _verify_humaneval_certificates(root: Path, serving: dict[str, dict]) -> None:
    """Reopen 164 original external traces before the unchanged frozen selector.

    The approved path manifests are not certificates. The independent verifier
    reopens and hashes actual per-case traces and recomputes the score, while
    this bridge *independently* matches each source-bound scored row and its
    item termination against the full original frozen serving session.
    """
    from chowder_batch import reap_v2_humaneval_certificate_verifier_20261027 as cert

    arms = ("reap128_v2", "dense_v2")
    _require(set(APPROVED_HUMANEVAL_CERTIFICATE_INPUT_SHA256) == set(arms)
             and all(_hex(value) for value in APPROVED_HUMANEVAL_CERTIFICATE_INPUT_SHA256.values()),
             "original independent HumanEval certificate input approvals absent")
    _require(not cert.approval_gaps(),
             "original outside-host HumanEval independent trace approvals absent")
    _require(
        serving["reap128_v2"]["physical_host_sha256"]
        == serving["dense_v2"]["physical_host_sha256"]
        and serving["reap128_v2"]["hardware_profile_sha256"]
        == serving["dense_v2"]["hardware_profile_sha256"]
        and serving["reap128_v2"]["session_id"]
        != serving["dense_v2"]["session_id"],
        "original Q8/dense full164 must use distinct sessions on same physical host/profile",
    )
    verified_certificates = {}
    for arm in arms:
        manifest = root / f"{arm}.frozen-humaneval-certificate-inputs.json"
        _require(manifest.is_file() and not manifest.is_symlink(),
                 f"{arm}: independently reviewed certificate input manifest absent")
        raw = manifest.read_bytes()
        _require(0 < len(raw) <= 64 * 1024
                 and _sha(raw) == APPROVED_HUMANEVAL_CERTIFICATE_INPUT_SHA256[arm],
                 f"{arm}: independently reviewed certificate manifest SHA mismatch")
        inputs = _json_bytes(raw, max_bytes=64 * 1024)
        _require(set(inputs) == CERTIFICATE_INPUT_FIELDS
                 and all(type(value) is str and value for value in inputs.values()),
                 f"{arm}: original certificate input path set invalid")
        paths = {key: Path(value) for key, value in inputs.items()}
        _require(all(p.is_absolute() for p in paths.values()),
                 f"{arm}: original certificate path not absolute")
        _require(paths["serve_path"] == root / arm / "serve.json",
                 f"{arm}: independent execution certificate serving source differs from original selector")
        # Never forward an unreviewed `canonical_path` override. The verifier
        # uses its own exact originally archived canonical source/hash.
        certificate = cert.verify_arm(arm=arm, **paths)
        verified_certificates[arm] = certificate
        # Certificate exporter uses a separate source-reviewed JSONL writer,
        # while the unchanged original reap_eval writer uses its own key
        # order, spacing and seconds precision. NEVER require those distinct
        # serializations to share a byte SHA: verify the original bytes from
        # independently approved source pins, and join all original per-item
        # scored/terminated semantics to external trace outcomes instead.
        original_rows_path = root / arm / "humaneval.jsonl"
        _require(original_rows_path.is_file() and not original_rows_path.is_symlink()
                 and APPROVED_ORIGINAL_ARTIFACT_SHA256.get(str(original_rows_path))
                 == _sha(original_rows_path.read_bytes()),
                 f"{arm}: original evaluator row bytes not independently pinned")
        _require(certificate.get("status") == "INDEPENDENT_164_CERTIFICATE_RECOMPUTED_NOT_RELEASE"
                 and certificate.get("model_sha256") == MODEL_SHA256[arm]
                 and certificate.get("session_id") == serving[arm]["session_id"]
                 and certificate.get("physical_host_sha256") == serving[arm]["physical_host_sha256"]
                 and certificate.get("population") == 164
                 and certificate.get("canonical_source_sha256") == cert.CANONICAL_RAW_SHA256
                 and certificate.get("frozen_source_items_sha256")
                 == reap_gates.FROZEN_SOURCE_SHA256["humaneval"]
                 and _hex(certificate.get("selector_rows_sha256"))
                 and certificate.get("chosen") is None
                 and certificate.get("release_approved") is False,
                 f"{arm}: independently verified original certificate identity mismatch")
        rows = reap_select._attested_rows(root, arm, "humaneval", 164)
        correct = certificate.get("outcomes")
        terminated = certificate.get("termination_outcomes")
        response_hashes = certificate.get("native_response_sha256_outcomes")
        health = certificate.get("native_health_outcomes")
        _require(isinstance(rows, dict)
                 and type(correct) in (list, tuple) and len(correct) == 164
                 and type(terminated) in (list, tuple) and len(terminated) == 164
                 and type(response_hashes) in (list, tuple) and len(response_hashes) == 164
                 and type(health) in (list, tuple) and len(health) == 164,
                 f"{arm}: complete independent per-item certificate absent")
        for i in range(164):
            key = f"HumanEval/{i}"
            expected = rows.get(key)
            item = correct[i]
            finish = terminated[i]
            response = response_hashes[i]
            metrics = health[i]
            _require(isinstance(expected, dict)
                     and type(item) in (tuple, list) and len(item) == 2
                     and type(finish) in (tuple, list) and len(finish) == 2
                     and type(response) in (tuple, list) and len(response) == 2
                     and type(metrics) in (tuple, list) and len(metrics) == 5
                     and item[0] == finish[0] == response[0] == metrics[0] == key
                     and type(item[1]) is bool and type(finish[1]) is bool
                     and _hex(response[1])
                     and isinstance(expected.get("text"), str)
                     and _sha(expected["text"].encode("utf-8")) == response[1]
                     and type(expected.get("trigram_ratio")) in (float, int)
                     and expected["trigram_ratio"] == metrics[1]
                     and type(expected.get("completion_tokens")) is int
                     and expected["completion_tokens"] == metrics[2]
                     and type(expected.get("seconds")) in (float, int)
                     and expected["seconds"] == metrics[3]
                     and expected.get("predicted_per_second") == metrics[4]
                     and expected.get("correct") is item[1]
                     and expected.get("terminated") is finish[1]
                     and not (item[1] and not finish[1]),
                     f"{arm}/{key}: frozen official row differs from verified external trace")
    # Individual trace reviews establish per-arm uniqueness, not that the
    # candidate and dense cohorts used distinct real Python executions.
    cert.verify_disjoint_execution_lineage(
        verified_certificates["reap128_v2"], verified_certificates["dense_v2"],
    )


def _blocked(reason: str) -> dict:
    return {
        "schema": SCHEMA, "status": "BLOCKED_UNVERIFIED_FROZEN_EVIDENCE",
        "reason": reason, "gate0": "UNKNOWN", "gate1": "UNKNOWN", "gate2": "UNKNOWN",
        "control": "UNKNOWN", "speed": "UNKNOWN",
        "selection": reap_select.select({}),
        "amendment4": "BLOCKED_PENDING_INDEPENDENT_PREREG_METHOD_RECONCILIATION",
        "chosen": None, "release_approved": False,
    }


def bridge_frozen_evidence(
    eval_dir: str | Path, observation: str | Path,
    observation_v2: str | Path | None = None,
    calibration_v2: str | Path | None = None,
) -> dict:
    """Read and admit pinned *original* evidence into the unmodified selector.

    Approval SHA pins live only in this reviewed source module, never in the
    supplied evaluation files or call arguments. Returns BLOCKED/UNKNOWN for
    missing or invalid evidence. ``release_approved`` is always False, even if
    the frozen deterministic selector eventually returns SELECTED.
    """
    try:
        root = Path(eval_dir)
        observer = Path(observation)
        observer_v2 = Path(observation_v2) if observation_v2 is not None else None
        calibration = Path(calibration_v2) if calibration_v2 is not None else None
        first = _verify_pins(root, observer, observer_v2, calibration)
        _verify_historical_canonical_rows(root)
        review_path = root / "frozen-independent-review.json"
        review_raw = review_path.read_bytes() if review_path.is_file() and not review_path.is_symlink() else b""
        _require(_sha(review_raw) == APPROVED_INDEPENDENT_REVIEW_SHA256,
                 "independent reviewed original native-serving manifest SHA mismatch")
        serving = _verify_original_sessions(root, _json_bytes(review_raw))
        _verify_humaneval_security(root, serving)
        _verify_humaneval_certificates(root, serving)
        # Only the existing original frozen gate engine calculates scientific
        # outcomes. Its paired Wald is statistics.stdev (sample SD), not a
        # discordance-only McNemar substitution; it preserves Amendment4 block.
        report = reap_select.evaluate(root, observer, observer_v2, calibration)
        _require(reap_select._input_digests(root, observer, observer_v2, calibration) == first,
                 "frozen selector inputs changed during independent validation")
        # Original selector input digests do not contain the separately
        # independently reviewed execution traces. Re-read/re-hash all
        # original 164-case external traces and approved manifest bytes
        # AFTER selector grading too, rejecting concurrent certificate edits
        # or trace replacement rather than trusting a one-time precheck.
        _verify_humaneval_certificates(root, serving)
        # The selector's input digest set excludes independently approved
        # review/security manifests. Recheck them after grading as well.
        review_files = {review_path: APPROVED_INDEPENDENT_REVIEW_SHA256}
        review_files.update({
            root / f"{arm}.frozen-humaneval-security.json": digest
            for arm, digest in APPROVED_HUMANEVAL_SECURITY_SHA256.items()
        })
        for path, digest in review_files.items():
            _require(path.is_file() and not path.is_symlink()
                     and _sha(path.read_bytes()) == digest,
                     "independently approved review/security changed during grading")
        gates = report["gates"].get("reap128_v2", {}).get("gates", {})
        _require(set(reap_select.REQUIRED).issubset(gates),
                 "original v2 gate set was not computed by frozen selector")
        _require(report["paired_method"]["status"] == "UNRESOLVED"
                 and report["release_approved"] is False,
                 "Amendment4 method conflict or frozen release guard was lost")
        # The frozen selector may legitimately choose a later fallback, but
        # this bridge has no independent native/model/security approvals for
        # reap160, reap160_v2 or historical reap128. A selection is not an
        # admission certificate: refuse rather than return an unreviewed arm.
        _require(report["decision"].get("chosen") in (None, "reap128_v2"),
                 "frozen selector chose fallback without independent original admission")
        # A selected v2 must have every mandatory frozen gate genuinely PASS.
        # Do NOT impose the unresolved optional paired-McNemar replacement
        # criterion on Amendment4's distinct ordered fallback after v1 failure.
        # The unchanged frozen selector owns that scientific precedence.
        if report["decision"].get("chosen") == "reap128_v2":
            _require(report["decision"].get("status") == "SELECTED"
                     and all(isinstance(gates[name], dict)
                             and reap_select._status(report["gates"]["reap128_v2"], name)
                             == "PASS"
                             for name in reap_select.REQUIRED),
                     "frozen v2 selected without all mandatory original gates PASS")
            historical_v1 = report["gates"].get("reap128")
            _require(isinstance(historical_v1, dict),
                     "historical v1 selection prerequisite absent")
            if any(reap_select._status(historical_v1, name) == "FAIL"
                   for name in ("gate0", "gate1")):
                # The frozen mandatory fallback requires BOTH v2 and REAP160
                # to be genuinely scored, not just the winning v2 arm. A
                # synthetic or unreviewed REAP160 row set cannot satisfy it.
                _verify_historical_reap160_prerequisite(
                    root, forbidden_sessions=frozenset(
                        item["session_id"] for item in serving.values()
                    ),
                )
        return {
            "schema": SCHEMA, "status": "SOURCE_APPROVED_ORIGINAL_EVIDENCE_EVALUATED",
            "gate0": gates["gate0"]["status"],
            "gate1": gates["gate1"].get("computed_status", gates["gate1"]["status"]),
            "gate2": gates["gate2"].get("computed_status", gates["gate2"]["status"]),
            "control": gates["control"].get("computed_status", gates["control"]["status"]),
            "speed": gates["speed"].get("computed_status", gates["speed"]["status"]),
            "frozen_selector": report, "selection": report["decision"],
            "amendment4": "BLOCKED_PENDING_INDEPENDENT_PREREG_METHOD_RECONCILIATION",
            "chosen": report["decision"].get("chosen"), "release_approved": False,
        }
    except (FrozenEvidenceError, reap_gates.GateError, ValueError, TypeError,
            KeyError, OSError, UnicodeError, OverflowError, ZeroDivisionError) as exc:
        return _blocked(str(exc)[:500])
