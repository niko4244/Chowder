"""SYNTHETIC ADVERSARIAL FIXTURES ONLY; NO original model score or approval.

Every source-owned approval SHA is monkeypatched inside this test process.
None of these fabricated rows or attestations may be published as Gen-0 proof.
"""

import hashlib
import json
import math
from pathlib import Path
import statistics

import pytest

from chowder_batch import reap_gates, reap_select
from chowder_batch import reap_v2_frozen_gate_evidence_bridge_20261027 as bridge


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("payload", [
    '{"schema":"approved","schema":"denied"}',
    '{"schema":"approved","nested":{"status":"approved","status":"denied"}}',
    '{"schema":"approved","arms":[{"owner":1,"owner":2}]}',
    '{"schema":"approved","count":NaN}',
    '{"schema":"approved","count":Infinity}',
])
def test_independent_receipt_json_rejects_ambiguous_keys_and_nonfinite(
    tmp_path: Path, payload: str,
):
    path = tmp_path / "independent-review.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(bridge.FrozenEvidenceError, match="independent receipt JSON invalid"):
        bridge._json_file(path)


def test_independent_receipt_json_preserves_unambiguous_nested_receipts(
    tmp_path: Path,
):
    path = tmp_path / "independent-review.json"
    path.write_text('{"arms":{"q8":{"approved":true}},"count":164}', encoding="utf-8")
    assert bridge._json_file(path) == {"arms": {"q8": {"approved": True}}, "count": 164}


def _rows(root, arm, suite, n, failures, *, source_witness, speed=0.0):
    folder = root / arm
    folder.mkdir(exist_ok=True)
    output = folder / f"{suite}.jsonl"
    rows = []
    for i in range(n):
        rows.append({"id": f"gsm8k-{i}" if suite == "gsm8k" else f"HumanEval/{i}",
                     "correct": i >= failures, "terminated": True,
                     "trigram_ratio": 0.9, "completion_tokens": 12,
                     "seconds": 1.0, "predicted_per_second": speed,
                     "text": "SYNTHETIC TEST ONLY"})
    output.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    summary_path = folder / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary[suite] = {"n": n, "file_sha256": _sha(output)}
    _write(summary_path, summary)
    if source_witness:
        serve = json.loads((folder / "serve.json").read_text())
        _write(folder / f"{suite}.inputs.json", {
            "schema": reap_select.FROZEN_INPUT_SCHEMA,
            "model": {key: serve[key] for key in ("gguf", "gguf_sha256", "flags")},
            "suite": suite,
            "source_items_sha256": reap_gates.FROZEN_SOURCE_SHA256[suite],
            "source_items_n": 1319 if suite == "gsm8k" else 164,
            "decode": {"system": "You are a helpful assistant.", "temperature": 0.0,
                       "top_k": 1, "seed": 7, "max_tokens": 1024 if suite == "gsm8k" else 768,
                       "enable_thinking": False},
            "scoring": reap_select.FROZEN_SCORING,
        })


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    """Untrusted test-only evidence, with fake review hashes installed in memory."""
    root = tmp_path / "original-frozen-fixtures"
    root.mkdir()
    obs = root / "observer-v1.json"
    _write(obs, {"layers": [{"layer": i} for i in range(40)], "calibration_ppl": 10})
    legacy = {
        "dense": {"gsm8k": (1319, 0), "humaneval": (164, 0)},
        "random128": {"gsm8k": (1319, 1000)},
        "reap128": {"gsm8k": (1319, 90), "humaneval": (164, 0)},
        "dense_q4": {"gsm8k": (200, 0)},
        "reap128_q4": {"gsm8k": (200, 0)},
    }
    for arm, suites in legacy.items():
        _write(root / arm / "serve.json", {"experts_offloaded": 0})
        for suite, (n, fail) in suites.items():
            _rows(root, arm, suite, n, fail, source_witness=False,
                  speed=20.0 if arm == "dense_q4" else 30.0)

    synthetic_failures = {
        "reap128_v2": (0, 0), "dense_v2": (5, 4),
        "random128_v2": (700, 0), "reap128_v2_q4": (0, 0),
        "dense_v2_q4": (0, 0),
    }
    review_arms = {}
    for index, (arm, suites) in enumerate(bridge.COHORTS.items(), 1):
        session = f"{index:x}" * 32
        model = bridge.MODEL_SHA256[arm]
        _write(root / arm / "serve.json", {
            "gguf": f"C:/synthetic-test-only/{arm}.gguf", "gguf_sha256": model,
            "flags": reap_select.FROZEN_SERVE_FLAGS,
            "eval_session_id": session,
            "server_pid": 100 + index, "server_port": 8800 + index,
            "experts_offloaded": 0, "measured_experts_offloaded": 0,
        })
        for suite, n in suites.items():
            failure = synthetic_failures[arm][suite == "humaneval"]
            _rows(root, arm, suite, n, failure, source_witness=True,
                  speed=40.0 if arm == "reap128_v2_q4" else 20.0)
        if arm.endswith("_q4"):
            (root / arm / "server.log").write_text("SYNTHETIC NO ORIGINAL NATIVE SERVING\n")
        review_arms[arm] = {
            "model_sha256": model, "session_id": session,
            "hardware_profile_sha256": "a" * 64,
            "physical_host_sha256": "b" * 64,
            "independent_model_bytes_verified": True,
            "native_windows_server_owner_verified": True,
            "per_item_session_binding_verified": True,
            "nonresumed_complete_cohort": True,
            "rows_sha256": {suite: _sha(root / arm / f"{suite}.jsonl") for suite in suites},
        }
    review = {
        "schema": bridge.REVIEW_SCHEMA,
        "status": "INDEPENDENTLY_APPROVED_ORIGINAL_FROZEN_EVIDENCE",
        "native_windows_server_owner_verified": True,
        "same_hardware_speed_pair_verified": True,
        "no_cross_session_row_pooling": True,
        "arms": review_arms,
    }
    review_path = root / "frozen-independent-review.json"
    _write(review_path, review)
    security_pins = {}
    for arm in ("reap128_v2", "dense_v2"):
        path = root / f"{arm}.frozen-humaneval-security.json"
        _write(path, {
            "schema": bridge.SECURITY_SCHEMA,
            "status": "INDEPENDENTLY_APPROVED_FROZEN_GATE2_EXECUTION",
            "arm": arm, "model_sha256": bridge.MODEL_SHA256[arm],
            "session_id": review_arms[arm]["session_id"],
            "canonical_source_sha256": reap_gates.FROZEN_SOURCE_SHA256["humaneval"],
            "original_rows_sha256": _sha(root / arm / "humaneval.jsonl"),
            "source_bound_cases": 164,
            "independent_observer_implementation_sha256": "c" * 64,
            "controls": {key: True for key in (
                "original_python_I_ten_second_timeout", "per_case_original_program_sha_verified",
                "isolated_nonprivileged_process", "network_egress_denied",
                "host_filesystem_and_secrets_denied", "memory_and_pid_limits_enforced",
                "independent_process_tree_extinction", "no_unreviewed_generated_code_execution")},
        })
        security_pins[arm] = _sha(path)
    pins = reap_select._input_digests(root, obs, None, None)
    monkeypatch.setattr(bridge, "APPROVED_ORIGINAL_ARTIFACT_SHA256", pins)
    monkeypatch.setattr(bridge, "APPROVED_INDEPENDENT_REVIEW_SHA256", _sha(review_path))
    monkeypatch.setattr(bridge, "APPROVED_HUMANEVAL_SECURITY_SHA256", security_pins)
    # Only this synthetic fixture bypasses the otherwise mandatory externally
    # reviewed trace-file intake; real production approval maps remain empty.
    # Independent end-to-end certificate path refusal/row joins are tested in
    # test_reap_v2_phase49_humaneval_intake_20261028.py.
    monkeypatch.setattr(bridge, "_verify_humaneval_certificates", lambda root, serving: None)
    return root, obs


def test_live_production_has_no_security_or_source_approval():
    assert not bridge.APPROVED_ORIGINAL_ARTIFACT_SHA256
    assert bridge.APPROVED_INDEPENDENT_REVIEW_SHA256 is None
    assert not bridge.APPROVED_HUMANEVAL_SECURITY_SHA256
    assert not bridge.APPROVED_HUMANEVAL_CERTIFICATE_INPUT_SHA256
    blocked = bridge.bridge_frozen_evidence("Z:/nonexistent", "Z:/observer.json")
    assert blocked["status"] == "BLOCKED_UNVERIFIED_FROZEN_EVIDENCE"
    assert blocked["chosen"] is None and blocked["release_approved"] is False
    assert all(blocked[g] == "UNKNOWN" for g in ("gate0", "gate1", "gate2", "control", "speed"))


def test_synthetic_complete_reviewer_pins_delegate_exact_science_to_original_selector(synthetic):
    root, obs = synthetic
    result = bridge.bridge_frozen_evidence(root, obs)
    assert result["status"] == "SOURCE_APPROVED_ORIGINAL_EVIDENCE_EVALUATED", result["reason"] if "reason" in result else result
    assert result["gate0"] == result["gate1"] == result["gate2"] == result["control"] == "PASS"
    assert result["speed"] == "UNKNOWN"  # no genuine native same-session source proof
    assert result["frozen_selector"]["gates"]["reap128_v2"]["gates"]["d0"]["status"] == "UNKNOWN"
    assert result["selection"]["status"] == "BLOCKED" and result["release_approved"] is False
    assert result["frozen_selector"]["paired_method"]["status"] == "UNRESOLVED"
    q8 = [1.0] * 5 + [0.0] * (1319 - 5)
    expected = statistics.fmean(q8) - 1.645 * statistics.stdev(q8) / math.sqrt(1319)
    gate = result["frozen_selector"]["gates"]["reap128_v2"]["gates"]["gate1"]
    assert gate["statistic"] == pytest.approx(expected, abs=1e-14)


def _reseal_legacy_for_adversarial_test(root, obs, monkeypatch, arm, suite, mutate):
    """Test-only resealing simulates an attacker controlling legacy summaries."""
    path = root / arm / f"{suite}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    mutate(rows)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    summary_path = root / arm / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary[suite]["file_sha256"] = _sha(path)
    _write(summary_path, summary)
    monkeypatch.setattr(bridge, "APPROVED_ORIGINAL_ARTIFACT_SHA256",
                        reap_select._input_digests(root, obs, None, None))


def test_resealed_matching_noncanonical_legacy_pair_cannot_enter_frozen_bridge(
    synthetic, monkeypatch,
):
    root, obs = synthetic
    def forge(rows):
        for index, item in enumerate(rows):
            item["id"] = f"fabricated-gsm8k-{index}"

    for arm in ("dense", "reap128", "random128"):
        _reseal_legacy_for_adversarial_test(root, obs, monkeypatch, arm, "gsm8k", forge)
    # Legacy pairs have identical forged IDs and internally consistent SHAs;
    # the frozen selector alone would not establish the official task cohort.
    result = bridge.bridge_frozen_evidence(root, obs)
    assert result["status"] == "BLOCKED_UNVERIFIED_FROZEN_EVIDENCE"
    assert "historical task IDs are not canonical" in result["reason"]
    assert result["chosen"] is None and result["release_approved"] is False


@pytest.mark.parametrize("arm,suite", [
    ("dense", "gsm8k"), ("reap128", "humaneval"),
    ("random128", "gsm8k"), ("dense_q4", "gsm8k"),
])
def test_resealed_legacy_correct_nontermination_refused_by_independent_bridge(
    synthetic, monkeypatch, arm, suite,
):
    root, obs = synthetic
    def forge(rows):
        rows[0]["correct"] = True
        rows[0]["terminated"] = False

    _reseal_legacy_for_adversarial_test(root, obs, monkeypatch, arm, suite, forge)
    result = bridge.bridge_frozen_evidence(root, obs)
    assert result["status"] == "BLOCKED_UNVERIFIED_FROZEN_EVIDENCE"
    assert "historical scorer semantics invalid" in result["reason"]
    assert result["chosen"] is None and result["release_approved"] is False


@pytest.mark.parametrize("unreviewed", ["reap160", "reap160_v2", "reap128"])
def test_unreviewed_frozen_fallback_cannot_be_promoted_by_independent_bridge(
    synthetic, monkeypatch, unreviewed,
):
    root, obs = synthetic
    original_evaluate = reap_select.evaluate

    def select_unreviewed(*args, **kwargs):
        report = original_evaluate(*args, **kwargs)
        report["decision"] = {"status": "SELECTED", "chosen": unreviewed,
                              "reason": "synthetic unreviewed fallback"}
        return report

    monkeypatch.setattr(bridge.reap_select, "evaluate", select_unreviewed)
    result = bridge.bridge_frozen_evidence(root, obs)
    assert result["status"] == "BLOCKED_UNVERIFIED_FROZEN_EVIDENCE"
    assert "fallback without independent original admission" in result["reason"]
    assert result["chosen"] is None and result["release_approved"] is False


def test_incomplete_mandatory_gate_cannot_expose_selected_v2_as_gen0_choice(
    synthetic, monkeypatch,
):
    """Synthetic SELECTED cannot override missing original speed admission."""
    root, obs = synthetic
    original_evaluate = reap_select.evaluate

    def select_while_unresolved(*args, **kwargs):
        report = original_evaluate(*args, **kwargs)
        assert report["paired_method"]["status"] == "UNRESOLVED"
        report["decision"] = {
            "status": "SELECTED", "chosen": "reap128_v2",
            "reason": "synthetic selected result despite unresolved method",
        }
        return report

    monkeypatch.setattr(bridge.reap_select, "evaluate", select_while_unresolved)
    result = bridge.bridge_frozen_evidence(root, obs)
    assert result["status"] == "BLOCKED_UNVERIFIED_FROZEN_EVIDENCE"
    assert "mandatory original gates PASS" in result["reason"]
    assert result["chosen"] is None and result["release_approved"] is False


def test_optional_mcnemar_ambiguity_does_not_veto_mandatory_v1_fail_fallback(
    synthetic, monkeypatch,
):
    """Test-only: mandatory ordered fallback is distinct from optional replacement."""
    root, obs = synthetic
    original_evaluate = reap_select.evaluate

    def simulate_complete_frozen_fallback(*args, **kwargs):
        report = original_evaluate(*args, **kwargs)
        gates = report["gates"]["reap128_v2"]["gates"]
        for name in reap_select.REQUIRED:
            gates[name]["status"] = "PASS"
        report["decision"] = {
            "status": "SELECTED", "chosen": "reap128_v2",
            "reason": "synthetic mandatory fallback after historical v1 failure",
        }
        return report

    monkeypatch.setattr(bridge.reap_select, "evaluate", simulate_complete_frozen_fallback)
    result = bridge.bridge_frozen_evidence(root, obs)
    assert result["status"] == "SOURCE_APPROVED_ORIGINAL_EVIDENCE_EVALUATED"
    assert result["chosen"] == "reap128_v2"
    assert result["amendment4"].startswith("BLOCKED_PENDING")
    assert result["release_approved"] is False


def test_malformed_independent_rows_digest_map_fails_closed(synthetic, monkeypatch):
    """Malformed independently pinned reviewer data must return BLOCKED, not crash."""
    root, obs = synthetic
    review_path = root / "frozen-independent-review.json"
    review = json.loads(review_path.read_text())
    arm = "reap128_v2"
    review["arms"][arm]["rows_sha256"] = list(bridge.COHORTS[arm])
    _write(review_path, review)
    monkeypatch.setattr(bridge, "APPROVED_INDEPENDENT_REVIEW_SHA256", _sha(review_path))
    result = bridge.bridge_frozen_evidence(root, obs)
    assert result["status"] == "BLOCKED_UNVERIFIED_FROZEN_EVIDENCE"
    assert result["chosen"] is None and result["release_approved"] is False


@pytest.mark.parametrize("which", ["review", "security"])
def test_independently_pinned_authority_changed_during_grading_is_refused(
    synthetic, monkeypatch, which,
):
    """Selector input digests do not cover external review/security manifests."""
    root, obs = synthetic
    path = (root / "frozen-independent-review.json" if which == "review"
            else root / "reap128_v2.frozen-humaneval-security.json")
    original_evaluate = reap_select.evaluate

    def change_review_during_grading(*args, **kwargs):
        report = original_evaluate(*args, **kwargs)
        evidence = json.loads(path.read_text())
        evidence["status"] = "REVOKED_DURING_SYNTHETIC_TEST"
        _write(path, evidence)
        return report

    monkeypatch.setattr(bridge.reap_select, "evaluate", change_review_during_grading)
    result = bridge.bridge_frozen_evidence(root, obs)
    assert result["status"] == "BLOCKED_UNVERIFIED_FROZEN_EVIDENCE"
    assert "review/security changed during grading" in result["reason"]
    assert result["chosen"] is None and result["release_approved"] is False


@pytest.mark.parametrize("kind", [
    "unpin", "new_unapproved_file", "alter_scored_row", "wrong_review_sha",
    "foreign_q8_model", "wrong_slot_count", "cross_session", "wrong_host_speed",
    "research_only_sandbox", "missing_security", "wrong_security_sha",
    "nonboolean_quality", "source_alias", "missing_dense", "resumed_owner",
])
def test_synthetic_adversarial_mutations_cannot_cross_original_security_boundary(
    synthetic, monkeypatch, kind,
):
    root, obs = synthetic
    model_path = root / "reap128_v2" / "serve.json"
    review_path = root / "frozen-independent-review.json"
    security_path = root / "reap128_v2.frozen-humaneval-security.json"
    if kind == "unpin":
        monkeypatch.setattr(bridge, "APPROVED_ORIGINAL_ARTIFACT_SHA256", {})
    elif kind == "new_unapproved_file":
        _write(root / "reap160" / "summary.json", {"gsm8k": {"n": 1319}})
    elif kind == "alter_scored_row":
        file = root / "reap128_v2" / "gsm8k.jsonl"
        file.write_bytes(file.read_bytes() + b"\n")
    elif kind == "wrong_review_sha":
        _write(review_path, {"status": "unreviewed"})
    elif kind == "foreign_q8_model":
        data = json.loads(model_path.read_text())
        data["gguf_sha256"] = "f" * 64
        _write(model_path, data)
    elif kind == "wrong_slot_count":
        data = json.loads(model_path.read_text())
        data["flags"][3] = "1"
        _write(model_path, data)
    elif kind == "cross_session":
        review = json.loads(review_path.read_text())
        review["arms"]["reap128_v2"]["session_id"] = "f" * 32
        _write(review_path, review)
        monkeypatch.setattr(bridge, "APPROVED_INDEPENDENT_REVIEW_SHA256", _sha(review_path))
    elif kind == "wrong_host_speed":
        review = json.loads(review_path.read_text())
        review["arms"]["dense_v2_q4"]["physical_host_sha256"] = "f" * 64
        _write(review_path, review)
        monkeypatch.setattr(bridge, "APPROVED_INDEPENDENT_REVIEW_SHA256", _sha(review_path))
    elif kind == "research_only_sandbox":
        evidence = json.loads(security_path.read_text())
        evidence["status"] = "RESEARCH_ONLY_SOURCE_BOUND_NOT_FROZEN_GATE2"
        _write(security_path, evidence)
        monkeypatch.setitem(bridge.APPROVED_HUMANEVAL_SECURITY_SHA256,
                            "reap128_v2", _sha(security_path))
    elif kind == "missing_security":
        security_path.unlink()
    elif kind == "wrong_security_sha":
        monkeypatch.setitem(bridge.APPROVED_HUMANEVAL_SECURITY_SHA256,
                            "reap128_v2", "f" * 64)
    elif kind == "nonboolean_quality":
        file = root / "reap128_v2" / "gsm8k.jsonl"
        text = file.read_text()
        file.write_text(text.replace('"correct": true', '"correct": 2', 1))
    elif kind == "source_alias":
        witness = root / "reap128_v2" / "gsm8k.inputs.json"
        data = json.loads(witness.read_text())
        data["source_items_sha256"] = "f" * 64
        _write(witness, data)
    elif kind == "missing_dense":
        (root / "dense_v2" / "humaneval.jsonl").unlink()
    elif kind == "resumed_owner":
        (root / "reap128_v2_q4" / ("serve.resume-" + "f" * 32 + ".json")).write_text("{}")
    result = bridge.bridge_frozen_evidence(root, obs)
    assert result["status"] == "BLOCKED_UNVERIFIED_FROZEN_EVIDENCE", kind
    assert result["chosen"] is None and result["release_approved"] is False
    assert all(result[g] == "UNKNOWN" for g in ("gate0", "gate1", "gate2", "control", "speed"))


def test_optional_mcnemar_ci_cannot_be_replaced_by_good_paired_wald(synthetic):
    root, obs = synthetic
    result = bridge.bridge_frozen_evidence(root, obs)
    assert result["frozen_selector"]["paired_v2_minus_v1"]["gsm8k"] is not None
    assert result["amendment4"].startswith("BLOCKED_PENDING")
    assert result["release_approved"] is False


def _synthetic_reapprove_quality(root, obs, monkeypatch, arm, suite, wrong):
    """TEST-ONLY: simulate freshly reviewed fabricated rows, never real approval."""
    file = root / arm / f"{suite}.jsonl"
    rows = [json.loads(line) for line in file.read_text().splitlines()]
    for row in rows[:wrong]:
        row["correct"] = False
    if arm == "random128_v2" and wrong == 0:
        for row in rows:
            row["correct"] = True
    file.write_text("".join(json.dumps(row) + "\n" for row in rows))
    summary_path = root / arm / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary[suite]["file_sha256"] = _sha(file)
    _write(summary_path, summary)
    review_path = root / "frozen-independent-review.json"
    review = json.loads(review_path.read_text())
    review["arms"][arm]["rows_sha256"][suite] = _sha(file)
    _write(review_path, review)
    monkeypatch.setattr(bridge, "APPROVED_INDEPENDENT_REVIEW_SHA256", _sha(review_path))
    if suite == "humaneval":
        security_path = root / f"{arm}.frozen-humaneval-security.json"
        proof = json.loads(security_path.read_text())
        proof["original_rows_sha256"] = _sha(file)
        _write(security_path, proof)
        monkeypatch.setitem(bridge.APPROVED_HUMANEVAL_SECURITY_SHA256,
                            arm, _sha(security_path))
    monkeypatch.setattr(bridge, "APPROVED_ORIGINAL_ARTIFACT_SHA256",
                        reap_select._input_digests(root, obs, None, None))


@pytest.mark.parametrize(("suite", "arm", "wrong", "expected_gate"), [
    ("gsm8k", "reap128_v2", 110, "gate1"),
    ("humaneval", "reap128_v2", 40, "gate2"),
    ("gsm8k", "random128_v2", 0, "control"),
])
def test_synthetic_resealed_complete_negative_cohorts_keep_strict_original_gates(
    synthetic, monkeypatch, suite, arm, wrong, expected_gate,
):
    root, obs = synthetic
    _synthetic_reapprove_quality(root, obs, monkeypatch, arm, suite, wrong)
    result = bridge.bridge_frozen_evidence(root, obs)
    assert result["status"] == "SOURCE_APPROVED_ORIGINAL_EVIDENCE_EVALUATED", result
    assert result[expected_gate] == "FAIL"
    assert result["selection"]["status"] == "BLOCKED"
    assert result["release_approved"] is False


def test_synthetic_gate0_termination_regression_cannot_be_papered_over(
    synthetic, monkeypatch,
):
    root, obs = synthetic
    _synthetic_reapprove_quality(root, obs, monkeypatch, "reap128_v2", "gsm8k", 40)
    file = root / "reap128_v2" / "gsm8k.jsonl"
    rows = [json.loads(line) for line in file.read_text().splitlines()]
    for row in rows[:40]:
        row["terminated"] = False
    file.write_text("".join(json.dumps(row) + "\n" for row in rows))
    _synthetic_reapprove_quality(root, obs, monkeypatch, "reap128_v2", "gsm8k", 40)
    result = bridge.bridge_frozen_evidence(root, obs)
    assert result["status"] == "SOURCE_APPROVED_ORIGINAL_EVIDENCE_EVALUATED", result
    assert result["gate0"] == "FAIL"
    assert result["release_approved"] is False
