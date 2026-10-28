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
from pathlib import Path
import re

from chowder_batch import reap_gates, reap_select

SCHEMA = "chowder.reap-v2-frozen-gate-evidence-bridge.v1"
REVIEW_SCHEMA = "chowder.reap-v2-independent-original-frozen-review.v1"
SECURITY_SCHEMA = "chowder.reap-v2-independent-frozen-humaneval-security.v1"

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
        _require(isinstance(rows, dict)
                 and type(correct) in (list, tuple) and len(correct) == 164
                 and type(terminated) in (list, tuple) and len(terminated) == 164,
                 f"{arm}: complete independent per-item certificate absent")
        for i in range(164):
            key = f"HumanEval/{i}"
            expected = rows.get(key)
            item = correct[i]
            finish = terminated[i]
            _require(isinstance(expected, dict)
                     and type(item) in (tuple, list) and len(item) == 2
                     and type(finish) in (tuple, list) and len(finish) == 2
                     and item[0] == finish[0] == key
                     and type(item[1]) is bool and type(finish[1]) is bool
                     and expected.get("correct") is item[1]
                     and expected.get("terminated") is finish[1]
                     and not (item[1] and not finish[1]),
                     f"{arm}/{key}: frozen official row differs from verified external trace")


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
        # The unresolved preregistered paired-method ambiguity cannot be
        # silently promoted into a Gen-0 choice even if the unchanged frozen
        # selector reports a nominally passing candidate. Preserve the frozen
        # selector untouched; this independent admission boundary fails closed.
        _require(report["decision"].get("status") == "BLOCKED"
                 and report["decision"].get("chosen") is None,
                 "Amendment4 unresolved: frozen selection cannot be promoted")
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
