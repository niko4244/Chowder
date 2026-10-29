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


def _rows(root, arm, suite, n, failures, *, source_witness, speed=0.0,
          session_id=None):
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
        if session_id is not None:
            rows[-1]["eval_session_id"] = session_id
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
    assert bridge.APPROVED_HISTORICAL_REAP160_REVIEW_SHA256 is None
    assert not bridge.APPROVED_HISTORICAL_REAP160_MODEL_SHA256
    assert not bridge.APPROVED_HISTORICAL_SPEED_SERVER_LOG_SHA256
    assert not bridge.APPROVED_HISTORICAL_ITEM_LINEAGE_SHA256
    assert bridge.APPROVED_HISTORICAL_V1_REVIEW_SHA256 is None
    assert bridge.APPROVED_HISTORICAL_V1_MODEL_SHA256 is None
    assert not bridge.APPROVED_HISTORICAL_V1_ITEM_LINEAGE_SHA256
    blocked = bridge.bridge_frozen_evidence("Z:/nonexistent", "Z:/observer.json")
    assert blocked["status"] == "BLOCKED_UNVERIFIED_FROZEN_EVIDENCE"
    assert blocked["chosen"] is None and blocked["release_approved"] is False
    assert all(blocked[g] == "UNKNOWN" for g in ("gate0", "gate1", "gate2", "control", "speed"))


@pytest.fixture
def synthetic_historical_v1(synthetic, monkeypatch):
    """Separate synthetic v1 native reviewer; never installs production pins."""
    from chowder_batch.reap_v2_authentic_amendment4_replay_20261027 import replay

    root, observer = synthetic
    model = "1" * 64
    session = "9" * 32
    serve = {
        "gguf_sha256": model, "flags": reap_select.FROZEN_SERVE_FLAGS,
        "eval_session_id": session, "server_pid": 191,
        "server_port": 8991,
    }
    _write(root / "reap128" / "serve.json", serve)
    lineage = {}
    for suite in ("gsm8k", "humaneval"):
        rows = [json.loads(line) for line in
                (root / "reap128" / f"{suite}.jsonl").read_text().splitlines()]
        traces = []
        for row in rows:
            canonical = json.dumps(row, sort_keys=True, separators=(",", ":"),
                                   ensure_ascii=False, allow_nan=False).encode()
            traces.append({
                "id": row["id"], "eval_session_id": session,
                "gguf_sha256": model, "server_pid": 191, "server_port": 8991,
                "native_response_sha256": hashlib.sha256(row["text"].encode()).hexdigest(),
                "scored_row_sha256": hashlib.sha256(canonical).hexdigest(),
                "completion_tokens": row["completion_tokens"],
                "seconds": row["seconds"],
                "predicted_per_second": row["predicted_per_second"],
            })
        path = root / "reap128" / f"{suite}.native-lineage.jsonl"
        path.write_text("".join(json.dumps(item) + "\n" for item in traces),
                        encoding="utf-8")
        lineage[suite] = _sha(path)
    snapshot = replay()
    historical = json.loads(
        (Path(__file__).resolve().parent.parent /
         "docs/quals/reap128-v2-frozen-gate-snapshot-prequality-20261025.json")
        .read_text(encoding="utf-8")
    )["gates"]["reap128"]
    review = {
        "schema": bridge.HISTORICAL_V1_SCHEMA,
        "status": "INDEPENDENTLY_APPROVED_ORIGINAL_REAP128_V1_PREREQUISITE",
        "native_windows_server_owner_verified": True,
        "independent_model_bytes_verified": True,
        "per_item_session_binding_verified": True,
        "original_executable_scorer_verified": True,
        "original_40_layer_bf16_observer_verified": True,
        "canonical_source_sha256": reap_gates.FROZEN_SOURCE_SHA256,
        "frozen_snapshot_sha256": snapshot["source_sha256"],
        "historical_gate1_lower_bound": snapshot["historical_v1_gate1"]["lower_bound"],
        "observer_sha256": _sha(observer),
        "original_scored_counts": {"gsm8k": 1319, "humaneval": 164},
        "arms": {"reap128": {
            "model_sha256": model, "session_id": session,
            "physical_host_sha256": "a" * 64,
            "hardware_profile_sha256": "b" * 64,
            "rows_sha256": {
                suite: _sha(root / "reap128" / f"{suite}.jsonl")
                for suite in ("gsm8k", "humaneval")
            },
            "item_lineage_sha256": lineage,
        }},
    }
    path = root / "reap128-v1.frozen-prerequisite-review.json"
    _write(path, review)
    monkeypatch.setattr(bridge, "APPROVED_ORIGINAL_ARTIFACT_SHA256",
                        reap_select._input_digests(root, observer, None, None))
    monkeypatch.setattr(bridge, "APPROVED_HISTORICAL_V1_REVIEW_SHA256", _sha(path))
    monkeypatch.setattr(bridge, "APPROVED_HISTORICAL_V1_MODEL_SHA256", model)
    monkeypatch.setattr(bridge, "APPROVED_HISTORICAL_V1_ITEM_LINEAGE_SHA256",
                        {f"reap128/{suite}": sha for suite, sha in lineage.items()})
    baseline = {
        arm: {
            "session_id": str(i) * 32,
            "physical_host_sha256": "a" * 64,
            "hardware_profile_sha256": "b" * 64,
        }
        for i, arm in enumerate(
            ("reap160", "dense", "random128", "reap160_q4", "dense_q4"), 2
        )
    }
    return root, observer, historical, baseline, session, review, path


def test_historical_v1_independent_lineage_valid_synthetic(synthetic_historical_v1):
    root, observer, historical, baseline, session, _, _ = synthetic_historical_v1
    bridge._verify_historical_v1_prerequisite(
        root, observer, historical, baseline, forbidden_sessions=frozenset(),
    )
    with pytest.raises(bridge.FrozenEvidenceError, match="incompatible hosts or sessions"):
        bridge._verify_historical_v1_prerequisite(
            root, observer, historical, baseline,
            forbidden_sessions=frozenset({session}),
        )


@pytest.mark.parametrize("attack", [
    "model_substitution", "review_sha_drift", "lineage_missing",
    "response_substitution", "baseline_session_replay", "baseline_host_drift",
    "snapshot_gate1_substitution", "observer_substitution",
])
def test_historical_v1_adversarial_refusal(
    synthetic_historical_v1, monkeypatch, attack,
):
    root, observer, historical, baseline, session, review, path = synthetic_historical_v1
    if attack == "model_substitution":
        serve_path = root / "reap128" / "serve.json"
        serve = json.loads(serve_path.read_text())
        serve["gguf_sha256"] = "f" * 64
        _write(serve_path, serve)
        monkeypatch.setattr(bridge, "APPROVED_ORIGINAL_ARTIFACT_SHA256",
                            reap_select._input_digests(root, observer, None, None))
    elif attack == "review_sha_drift":
        review["native_windows_server_owner_verified"] = False
        _write(path, review)
    elif attack == "lineage_missing":
        (root / "reap128" / "gsm8k.native-lineage.jsonl").unlink()
    elif attack == "response_substitution":
        trace_path = root / "reap128" / "humaneval.native-lineage.jsonl"
        traces = trace_path.read_text().splitlines()
        first = json.loads(traces[0])
        first["native_response_sha256"] = "0" * 64
        traces[0] = json.dumps(first)
        trace_path.write_text("\n".join(traces) + "\n", encoding="utf-8")
        new_sha = _sha(trace_path)
        review["arms"]["reap128"]["item_lineage_sha256"]["humaneval"] = new_sha
        _write(path, review)
        monkeypatch.setattr(bridge, "APPROVED_HISTORICAL_V1_REVIEW_SHA256", _sha(path))
        pins = dict(bridge.APPROVED_HISTORICAL_V1_ITEM_LINEAGE_SHA256)
        pins["reap128/humaneval"] = new_sha
        monkeypatch.setattr(bridge, "APPROVED_HISTORICAL_V1_ITEM_LINEAGE_SHA256", pins)
    elif attack == "baseline_session_replay":
        baseline["dense"]["session_id"] = session
    elif attack == "baseline_host_drift":
        baseline["random128"]["physical_host_sha256"] = "f" * 64
    elif attack == "snapshot_gate1_substitution":
        historical["gates"]["gate1"]["statistic"] = -0.031
    elif attack == "observer_substitution":
        observer.write_text('{"fake":true}\n', encoding="utf-8")
    with pytest.raises(bridge.FrozenEvidenceError):
        bridge._verify_historical_v1_prerequisite(
            root, observer, historical, baseline, forbidden_sessions=frozenset(),
        )


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


@pytest.mark.parametrize("approve_v1", [False, True])
def test_optional_mcnemar_ambiguity_does_not_veto_mandatory_v1_fail_fallback(
    synthetic, monkeypatch, approve_v1,
):
    """Test-only: mandatory ordered fallback is distinct from optional replacement."""
    root, obs = synthetic
    original_evaluate = reap_select.evaluate

    def simulate_complete_frozen_fallback(*args, **kwargs):
        report = original_evaluate(*args, **kwargs)
        gates = report["gates"]["reap128_v2"]["gates"]
        for name in reap_select.REQUIRED:
            gates[name]["status"] = "PASS"
            if "computed_status" in gates[name]:
                gates[name]["computed_status"] = "PASS"
        report["decision"] = {
            "status": "SELECTED", "chosen": "reap128_v2",
            "reason": "synthetic mandatory fallback after historical v1 failure",
        }
        return report

    monkeypatch.setattr(bridge.reap_select, "evaluate", simulate_complete_frozen_fallback)
    # This test isolates the optional-statistics branch. The separate
    # historical REAP160 native prerequisite has its own adversarial tests.
    def independently_approved_historical_prerequisite(root_arg, *, forbidden_sessions):
        assert root_arg == root
        assert forbidden_sessions == frozenset(
            json.loads((root / arm / "serve.json").read_text())["eval_session_id"]
            for arm in bridge.COHORTS
        )

    monkeypatch.setattr(bridge, "_verify_historical_reap160_prerequisite",
                        independently_approved_historical_prerequisite)
    # This test isolates the selector's optional McNemar distinction; the
    # historical v1 antecedent has separate adversarial source-lineage tests.
    if approve_v1:
        monkeypatch.setattr(bridge, "_verify_historical_v1_prerequisite",
                            lambda *args, **kwargs: None)
    result = bridge.bridge_frozen_evidence(root, obs)
    if not approve_v1:
        assert result["status"] == "BLOCKED_UNVERIFIED_FROZEN_EVIDENCE"
        assert "historical REAP128-v1 failure antecedent" in result["reason"]
        assert result["chosen"] is None and result["release_approved"] is False
        return
    assert result["status"] == "SOURCE_APPROVED_ORIGINAL_EVIDENCE_EVALUATED"
    assert result["chosen"] == "reap128_v2"
    assert result["amendment4"].startswith("BLOCKED_PENDING")
    assert result["release_approved"] is False


def test_forged_reap160_speed_fail_cannot_satisfy_v2_fallback_prerequisite(
    synthetic, monkeypatch,
):
    """A bogus negative offload must not count as a genuinely scored REAP160."""
    root, obs = synthetic
    _write(root / "reap160" / "serve.json", {"experts_offloaded": 0})
    _rows(root, "reap160", "gsm8k", 1319, 0, source_witness=False)
    _rows(root, "reap160", "humaneval", 164, 0, source_witness=False)
    _write(root / "reap160_q4" / "serve.json", {"experts_offloaded": -1})
    assert not (root / "reap160_q4" / "gsm8k.jsonl").exists()
    monkeypatch.setattr(bridge, "APPROVED_ORIGINAL_ARTIFACT_SHA256",
                        reap_select._input_digests(root, obs, None, None))
    original_evaluate = reap_select.evaluate

    def synthetic_all_v2_gates_pass(*args, **kwargs):
        report = original_evaluate(*args, **kwargs)
        reap160 = report["gates"]["reap160"]
        assert reap160["gates"]["speed"]["status"] == "FAIL"
        assert reap_select._complete(reap160)
        assert not (root / "reap160_q4" / "gsm8k.jsonl").exists()
        v2 = report["gates"]["reap128_v2"]
        for gate in reap_select.REQUIRED:
            v2["gates"][gate]["status"] = "PASS"
            if "computed_status" in v2["gates"][gate]:
                v2["gates"][gate]["computed_status"] = "PASS"
        # Exercise the actual frozen fallback selector, not a forged decision.
        report["decision"] = reap_select.select(report["gates"])
        assert report["decision"]["chosen"] == "reap128_v2"
        return report

    monkeypatch.setattr(bridge.reap_select, "evaluate", synthetic_all_v2_gates_pass)
    result = bridge.bridge_frozen_evidence(root, obs)
    assert result["status"] == "BLOCKED_UNVERIFIED_FROZEN_EVIDENCE"
    assert "historical REAP160 prerequisite" in result["reason"]
    assert result["chosen"] is None and result["release_approved"] is False


@pytest.mark.parametrize("substituted_arm", ["dense", "random128"])
def test_historical_reference_model_sha_cannot_be_reapproved_as_an_alternate(
    tmp_path, monkeypatch, substituted_arm,
):
    """Source approval must not redefine the frozen original comparator."""
    models = {
        "reap160": "d" * 64,
        "dense": reap_select.FROZEN_REFERENCE_MODEL_SHA256["dense_v2"],
        "random128": reap_select.FROZEN_REFERENCE_MODEL_SHA256["random128_v2"],
        "reap160_q4": "e" * 64, "dense_q4": bridge.Q4_DENSE_SHA256,
    }
    models[substituted_arm] = "f" * 64
    monkeypatch.setattr(bridge, "APPROVED_HISTORICAL_REAP160_REVIEW_SHA256", "a" * 64)
    monkeypatch.setattr(bridge, "APPROVED_HISTORICAL_REAP160_MODEL_SHA256", models)
    monkeypatch.setattr(bridge, "APPROVED_HISTORICAL_SPEED_SERVER_LOG_SHA256", {
        "reap160_q4": "a" * 64, "dense_q4": "b" * 64,
    })
    monkeypatch.setattr(bridge, "APPROVED_HISTORICAL_ITEM_LINEAGE_SHA256", {
        f"{arm}/{suite}": "c" * 64
        for arm, suites in (
            ("reap160", ("gsm8k", "humaneval")),
            ("dense", ("gsm8k", "humaneval")),
            ("random128", ("gsm8k",)),
            ("reap160_q4", ("gsm8k",)),
            ("dense_q4", ("gsm8k",)),
        )
        for suite in suites
    })
    with pytest.raises(bridge.FrozenEvidenceError, match="source-owned approval"):
        bridge._verify_historical_reap160_prerequisite(
            tmp_path, forbidden_sessions=frozenset(),
        )


def test_historical_reap160_requires_separate_source_pinned_native_review(
    synthetic, monkeypatch,
):
    """Synthetic-only positive schema plus original-byte tamper rejection."""
    root, obs = synthetic
    models = {"reap160": "d" * 64,
              "dense": reap_select.FROZEN_REFERENCE_MODEL_SHA256["dense_v2"],
              "random128": reap_select.FROZEN_REFERENCE_MODEL_SHA256["random128_v2"],
              "reap160_q4": "e" * 64,
              "dense_q4": bridge.Q4_DENSE_SHA256}
    sessions = {"reap160": "c" * 32, "dense": "3" * 32,
                "random128": "4" * 32, "reap160_q4": "f" * 32,
                "dense_q4": "9" * 32}
    for arm, suites in (("reap160", ("gsm8k", "humaneval")),
                        ("dense", ("gsm8k", "humaneval")),
                        ("random128", ("gsm8k",)),
                        ("reap160_q4", ("gsm8k",)),
                        ("dense_q4", ("gsm8k",))):
        _write(root / arm / "serve.json", {
            "gguf_sha256": models[arm], "eval_session_id": sessions[arm],
            "flags": reap_select.FROZEN_SERVE_FLAGS, "experts_offloaded": 0,
            "measured_experts_offloaded": 0,
            "server_pid": {"reap160": 101, "dense": 104, "random128": 105,
                           "reap160_q4": 102, "dense_q4": 103}[arm],
            "server_port": {"reap160": 8901, "dense": 8904, "random128": 8905,
                            "reap160_q4": 8902, "dense_q4": 8903}[arm],
        })
        for suite in suites:
            _rows(root, arm, suite, 164 if suite == "humaneval" else
                  (200 if arm.endswith("_q4") else 1319), 0,
                  source_witness=False, speed=25.0)
    native_logs = {}
    for arm in ("reap160_q4", "dense_q4"):
        native_log = root / arm / "server.log"
        fit_lines = ["llama_params_fit: successfully fit",
                     "llama_model_load: offloaded 40/40 layers to GPU"]
        native_log.write_text("\n".join(fit_lines) + "\n", encoding="utf-8")
        serve_path = root / arm / "serve.json"
        serve = json.loads(serve_path.read_text(encoding="utf-8"))
        serve["offload_log_lines"] = fit_lines
        _write(serve_path, serve)
        native_logs[arm] = _sha(native_log)
    # Match the immutable original reap_eval._score writer: scored rows do NOT
    # contain eval_session_id. Independent external native traces, not edited
    # historical JSONL, must authenticate their serving-session lineage.
    lineage_pins = {}
    lineage_by_arm = {}
    for arm in models:
        lineage_by_arm[arm] = {}
        for suite in (("gsm8k", "humaneval") if arm in ("reap160", "dense")
                      else ("gsm8k",)):
            rows = [json.loads(line) for line in
                    (root / arm / f"{suite}.jsonl").read_text(encoding="utf-8").splitlines()]
            assert all("eval_session_id" not in row for row in rows)
            traces = []
            for row in rows:
                canonical = json.dumps(row, sort_keys=True, separators=(",", ":"),
                                       ensure_ascii=False, allow_nan=False).encode("utf-8")
                traces.append({
                    "id": row["id"], "eval_session_id": sessions[arm],
                    "gguf_sha256": models[arm],
                    "server_pid": {"reap160": 101, "dense": 104, "random128": 105,
                                   "reap160_q4": 102, "dense_q4": 103}[arm],
                    "server_port": {"reap160": 8901, "dense": 8904, "random128": 8905,
                                    "reap160_q4": 8902, "dense_q4": 8903}[arm],
                    "native_response_sha256": hashlib.sha256(row["text"].encode()).hexdigest(),
                    "scored_row_sha256": hashlib.sha256(canonical).hexdigest(),
                    "completion_tokens": row["completion_tokens"],
                    "seconds": row["seconds"],
                    "predicted_per_second": row["predicted_per_second"],
                })
            trace_path = root / arm / f"{suite}.native-lineage.jsonl"
            trace_path.write_text("".join(json.dumps(item) + "\n" for item in traces),
                                  encoding="utf-8")
            lineage_by_arm[arm][suite] = _sha(trace_path)
            lineage_pins[f"{arm}/{suite}"] = _sha(trace_path)
    monkeypatch.setattr(bridge, "APPROVED_ORIGINAL_ARTIFACT_SHA256",
                        reap_select._input_digests(root, obs, None, None))
    review = {
        "schema": bridge.HISTORICAL_REAP160_SCHEMA,
        "status": "INDEPENDENTLY_APPROVED_ORIGINAL_REAP160_PREREQUISITE",
        "native_windows_server_owner_verified": True,
        "independent_model_bytes_verified": True,
        "per_item_session_binding_verified": True,
        "original_executable_scorer_verified": True,
        "same_hardware_speed_pair_verified": True,
        "original_q4_offload_fit_verified": True,
        "native_first200_speed_verified": True,
        "native_q4_server_log_sha256": native_logs,
        "canonical_source_sha256": reap_gates.FROZEN_SOURCE_SHA256,
        "original_scored_counts": {"gsm8k": 1319, "humaneval": 164,
                                   "first200_speed": 200},
        "arms": {
            arm: {
                "model_sha256": models[arm], "session_id": sessions[arm],
                "physical_host_sha256": "a" * 64,
                "hardware_profile_sha256": "b" * 64,
                "rows_sha256": {
                    suite: _sha(root / arm / f"{suite}.jsonl")
                    for suite in (("gsm8k", "humaneval") if arm in ("reap160", "dense")
                                  else ("gsm8k",))
                },
                "item_lineage_sha256": lineage_by_arm[arm],
            }
            for arm in models
        },
    }
    path = root / "reap160.frozen-prerequisite-review.json"
    _write(path, review)
    monkeypatch.setattr(bridge, "APPROVED_HISTORICAL_REAP160_REVIEW_SHA256", _sha(path))
    monkeypatch.setattr(bridge, "APPROVED_HISTORICAL_REAP160_MODEL_SHA256", models)
    monkeypatch.setattr(bridge, "APPROVED_HISTORICAL_SPEED_SERVER_LOG_SHA256",
                        native_logs)
    monkeypatch.setattr(bridge, "APPROVED_HISTORICAL_ITEM_LINEAGE_SHA256",
                        lineage_pins)
    bridge._verify_historical_reap160_prerequisite(root, forbidden_sessions=frozenset())
    # A historical session cannot double as a purported fresh v2 serving
    # session, even if all five historical arms are mutually distinct and
    # their original per-item lineage and review hashes were reapproved.
    bridge._verify_historical_reap160_prerequisite(
        root, forbidden_sessions=frozenset({"7" * 32}),
    )
    with pytest.raises(bridge.FrozenEvidenceError, match="sessions overlap"):
        bridge._verify_historical_reap160_prerequisite(
            root, forbidden_sessions=frozenset({sessions["dense_q4"]}),
        )
    # Even after source reapproval, a forged native response cannot be joined
    # to the original scored row by session and task ID alone.
    trace_path = root / "random128" / "gsm8k.native-lineage.jsonl"
    original_trace = trace_path.read_bytes()
    trace_path.unlink()
    with pytest.raises(bridge.FrozenEvidenceError, match="native item-lineage trace missing"):
        bridge._verify_historical_reap160_prerequisite(root, forbidden_sessions=frozenset())
    trace_path.write_bytes(original_trace)
    traces = [json.loads(line) for line in original_trace.splitlines()]
    traces[0]["native_response_sha256"] = "0" * 64
    trace_path.write_text("".join(json.dumps(item) + "\n" for item in traces),
                          encoding="utf-8")
    lineage_pins["random128/gsm8k"] = _sha(trace_path)
    review["arms"]["random128"]["item_lineage_sha256"]["gsm8k"] = _sha(trace_path)
    _write(path, review)
    monkeypatch.setattr(bridge, "APPROVED_HISTORICAL_REAP160_REVIEW_SHA256", _sha(path))
    with pytest.raises(bridge.FrozenEvidenceError, match="native item-lineage disagrees"):
        bridge._verify_historical_reap160_prerequisite(root, forbidden_sessions=frozenset())
    trace_path.write_bytes(original_trace)
    lineage_pins["random128/gsm8k"] = _sha(trace_path)
    review["arms"]["random128"]["item_lineage_sha256"]["gsm8k"] = _sha(trace_path)
    _write(path, review)
    monkeypatch.setattr(bridge, "APPROVED_HISTORICAL_REAP160_REVIEW_SHA256", _sha(path))
    bridge._verify_historical_reap160_prerequisite(root, forbidden_sessions=frozenset())
    # Historical Q8 reference/control arms are not interchangeable with a
    # self-reported REAP160 success. Every comparator needs its own source pin.
    monkeypatch.setattr(bridge, "APPROVED_HISTORICAL_REAP160_MODEL_SHA256",
                        {k: v for k, v in models.items() if k != "random128"})
    with pytest.raises(bridge.FrozenEvidenceError, match="source-owned approval"):
        bridge._verify_historical_reap160_prerequisite(root, forbidden_sessions=frozenset())
    monkeypatch.setattr(bridge, "APPROVED_HISTORICAL_REAP160_MODEL_SHA256", models)
    review["arms"]["random128"]["physical_host_sha256"] = "0" * 64
    _write(path, review)
    monkeypatch.setattr(bridge, "APPROVED_HISTORICAL_REAP160_REVIEW_SHA256", _sha(path))
    with pytest.raises(bridge.FrozenEvidenceError, match="Q8 quality comparator host"):
        bridge._verify_historical_reap160_prerequisite(root, forbidden_sessions=frozenset())
    review["arms"]["random128"]["physical_host_sha256"] = "a" * 64
    _write(path, review)
    monkeypatch.setattr(bridge, "APPROVED_HISTORICAL_REAP160_REVIEW_SHA256", _sha(path))
    bridge._verify_historical_reap160_prerequisite(root, forbidden_sessions=frozenset())
    # A separately SHA-approved log cannot contradict the declared offload
    # count; the actual original startup text must be parsed, not merely pinned.
    native_log = root / "reap160_q4" / "server.log"
    # Even a source-reapproved, count-consistent zero is not a completed
    # original --fit. A lone n_cpu_moe line previously passed this boundary.
    native_log.write_text("llama_params_fit: n_cpu_moe = 0\n", encoding="utf-8")
    native_logs["reap160_q4"] = _sha(native_log)
    review["native_q4_server_log_sha256"] = dict(native_logs)
    _write(path, review)
    monkeypatch.setattr(bridge, "APPROVED_HISTORICAL_REAP160_REVIEW_SHA256", _sha(path))
    monkeypatch.setattr(bridge, "APPROVED_HISTORICAL_SPEED_SERVER_LOG_SHA256",
                        dict(native_logs))
    with pytest.raises(bridge.FrozenEvidenceError, match="native fit/offload log disagrees"):
        bridge._verify_historical_reap160_prerequisite(root, forbidden_sessions=frozenset())
    native_log.write_text("llama_params_fit: successfully fit\n"
                          "llama_params_fit: n_cpu_moe = 0\n", encoding="utf-8")
    native_logs["reap160_q4"] = _sha(native_log)
    review["native_q4_server_log_sha256"] = dict(native_logs)
    _write(path, review)
    monkeypatch.setattr(bridge, "APPROVED_HISTORICAL_REAP160_REVIEW_SHA256", _sha(path))
    monkeypatch.setattr(bridge, "APPROVED_HISTORICAL_SPEED_SERVER_LOG_SHA256",
                        dict(native_logs))
    with pytest.raises(bridge.FrozenEvidenceError, match="native fit/offload log disagrees"):
        bridge._verify_historical_reap160_prerequisite(root, forbidden_sessions=frozenset())
    native_log.write_text("llama_params_fit: successfully fit\n"
                          "llama_model_load: offloaded 40/40 layers to GPU\n", encoding="utf-8")
    native_logs["reap160_q4"] = _sha(native_log)
    review["native_q4_server_log_sha256"] = dict(native_logs)
    _write(path, review)
    monkeypatch.setattr(bridge, "APPROVED_HISTORICAL_REAP160_REVIEW_SHA256", _sha(path))
    monkeypatch.setattr(bridge, "APPROVED_HISTORICAL_SPEED_SERVER_LOG_SHA256",
                        dict(native_logs))
    # Resealed rows must not override the actual serving-session identity.
    rows_path = root / "reap160_q4" / "gsm8k.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["eval_session_id"] = "0" * 32
    rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    summary_path = root / "reap160_q4" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["gsm8k"]["file_sha256"] = _sha(rows_path)
    _write(summary_path, summary)
    review["arms"]["reap160_q4"]["rows_sha256"]["gsm8k"] = _sha(rows_path)
    _write(path, review)
    monkeypatch.setattr(bridge, "APPROVED_HISTORICAL_REAP160_REVIEW_SHA256", _sha(path))
    monkeypatch.setattr(bridge, "APPROVED_ORIGINAL_ARTIFACT_SHA256",
                        reap_select._input_digests(root, obs, None, None))
    with pytest.raises(bridge.FrozenEvidenceError, match="per-item serving session mismatch"):
        bridge._verify_historical_reap160_prerequisite(root, forbidden_sessions=frozenset())
    rows[0].pop("eval_session_id")
    rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    summary["gsm8k"]["file_sha256"] = _sha(rows_path)
    _write(summary_path, summary)
    review["arms"]["reap160_q4"]["rows_sha256"]["gsm8k"] = _sha(rows_path)
    _write(path, review)
    monkeypatch.setattr(bridge, "APPROVED_HISTORICAL_REAP160_REVIEW_SHA256", _sha(path))
    monkeypatch.setattr(bridge, "APPROVED_ORIGINAL_ARTIFACT_SHA256",
                        reap_select._input_digests(root, obs, None, None))
    review["arms"]["dense_q4"]["physical_host_sha256"] = "0" * 64
    _write(path, review)
    monkeypatch.setattr(bridge, "APPROVED_HISTORICAL_REAP160_REVIEW_SHA256", _sha(path))
    with pytest.raises(bridge.FrozenEvidenceError, match="speed comparator host"):
        bridge._verify_historical_reap160_prerequisite(root, forbidden_sessions=frozenset())
    review["arms"]["dense_q4"]["physical_host_sha256"] = "a" * 64
    _write(path, review)
    monkeypatch.setattr(bridge, "APPROVED_HISTORICAL_REAP160_REVIEW_SHA256", _sha(path))
    # Even a separately pinned review cannot turn an invalid negative offload
    # into a legitimate completed original first200 speed measurement.
    serve_path = root / "reap160_q4" / "serve.json"
    bad_serve = json.loads(serve_path.read_text(encoding="utf-8"))
    bad_serve["experts_offloaded"] = -1
    bad_serve["measured_experts_offloaded"] = -1
    _write(serve_path, bad_serve)
    monkeypatch.setattr(bridge, "APPROVED_ORIGINAL_ARTIFACT_SHA256",
                        reap_select._input_digests(root, obs, None, None))
    with pytest.raises(bridge.FrozenEvidenceError, match="fit/offload receipt invalid"):
        bridge._verify_historical_reap160_prerequisite(root, forbidden_sessions=frozenset())
    bad_serve["experts_offloaded"] = bad_serve["measured_experts_offloaded"] = 0
    _write(serve_path, bad_serve)
    monkeypatch.setattr(bridge, "APPROVED_ORIGINAL_ARTIFACT_SHA256",
                        reap_select._input_digests(root, obs, None, None))
    with (root / "reap160_q4" / "gsm8k.jsonl").open("a", encoding="utf-8") as stream:
        stream.write('{"id":"gsm8k-0","correct":true}\n')
    with pytest.raises(bridge.FrozenEvidenceError, match="rows not independently bound"):
        bridge._verify_historical_reap160_prerequisite(root, forbidden_sessions=frozenset())


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
