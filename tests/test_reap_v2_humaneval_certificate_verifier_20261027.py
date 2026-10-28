"""All fabricated model/process receipt fixtures are ADVERSARIAL TEST DATA ONLY.

No model-generated Python is executed and no independent original sandbox
authorization is conferred by test-time monkeypatching of approval SHA pins.
"""

import json
from pathlib import Path

import pytest

from chowder_batch import reap_eval
from chowder_batch import reap_v2_original_humaneval_frozen_executor_20261027 as previous
from chowder_batch import reap_v2_humaneval_certificate_verifier_20261027 as certificate


def _write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _json(path: Path, value: object) -> bytes:
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
           + "\n").encode("utf-8")
    _write(path, raw)
    return raw


@pytest.fixture
def specimens(tmp_path, monkeypatch):
    """Entirely fake complete 164-item candidate with independent trace format."""
    arm = "reap128_v2"
    tasks = [
        {"id": f"HumanEval/{i}", "prompt": f"SYNTHE-TIC prompt {i}",
         "header": f"def synthetic_{i}():\n    pass\n", "entry_point": f"synthetic_{i}",
         "test": f"def check(fn):\n    assert fn() == {i}\n"}
        for i in range(164)
    ]
    canonical = tmp_path / "source-original-fixture.json"
    source_raw = _json(canonical, tasks)
    sha_source = reap_eval._digest(tasks)
    monkeypatch.setattr(previous, "FROZEN_CANONICAL_SHA256", certificate._sha(source_raw))
    monkeypatch.setattr(previous, "FROZEN_SOURCE_ITEMS_SHA256", sha_source)
    monkeypatch.setitem(reap_eval.FROZEN_V2_QUALITY_SOURCES, "humaneval", (164, sha_source))
    monkeypatch.setattr(certificate, "CANONICAL_RAW_SHA256", certificate._sha(source_raw))
    monkeypatch.setattr(certificate, "SOURCE_HUMANEVAL164_SHA256", sha_source)
    monkeypatch.setattr(previous.security, "execution_admission_gaps", lambda: [])
    # This fixture represents only one fake model; production checks all arms.
    monkeypatch.setattr(previous, "execution_admission_gaps", lambda: [])
    session = "a" * 32
    host = "b" * 64
    owner_sha = "c" * 64
    cap_sha = "d" * 64
    witness_arm_sha = previous.MODELS[arm]
    monkeypatch.setattr(previous, "APPROVED_NATIVE_OWNER_REVIEW_SHA256", {arm: owner_sha})
    monkeypatch.setattr(previous, "APPROVED_SANDBOX_CAPABILITY_SHA256", cap_sha)
    monkeypatch.setattr(previous, "APPROVED_SANDBOX_BROKER_SHA256", "e" * 64)
    cap = {"sandbox_profile_sha256": "f" * 64, "runtime_host_sha256": "1" * 64,
           "python_executable_sha256": "2" * 64}
    monkeypatch.setattr(previous, "_capability", lambda *args: cap)
    monkeypatch.setattr(previous, "_owner_review", lambda arm, serve, review: {
        "arm": arm, "model_sha256": previous.MODELS[arm], "session_id": session,
        "native_owner_review_sha256": owner_sha, "physical_host_sha256": host,
    })
    cases = []
    for i, task in enumerate(tasks):
        terminated = i not in (10, 150)
        response = f"```python\ndef synthetic_{i}():\n    return {i}\n```"
        source = previous._original_program(task, response) if terminated else None
        cases.append({
            "schema": previous.CAPTURE_SCHEMA, "ordinal": i, "id": task["id"],
            "arm": arm, "model_sha256": witness_arm_sha,
            "session_id": session, "native_owner_review_sha256": owner_sha,
            "physical_host_sha256": host,
            "source_items_sha256": sha_source, "canonical_sha256": certificate._sha(source_raw),
            "evaluator_sha256": previous.EVALUATOR_SHA,
            "scorer_sha256": previous.SCORER_SHA,
            "finished": terminated, "completion_tokens": 60,
            "seconds": 1.0, "predicted_per_second": 20.0,
            "trigram_ratio": round(reap_eval._trigram_ratio(response), 4),
            "native_generation_text": response,
            "native_response_sha256": certificate._sha(response.encode()),
            "candidate_program_sha256": certificate._sha(source) if source else None,
            "frozen_correct": None, "generated_code_executed": False,
        })
    capture_path = tmp_path / "captured-human164.jsonl"
    capture_raw = previous._serialized_capture(cases)
    _write(capture_path, capture_raw)
    monkeypatch.setattr(previous, "APPROVED_NATIVE_CAPTURE_COHORT_SHA256",
                        {arm: certificate._sha(capture_raw)})
    witness_rows = []
    trace_dir = tmp_path / "independent-traces"
    trace_dir.mkdir()
    trace_sha = []
    for i, case in enumerate(cases):
        proof = {
            "ordinal": i, "id": case["id"], "session_id": session,
            "candidate_program_sha256": case["candidate_program_sha256"],
        }
        if not case["finished"]:
            proof.update(status="NOT_EXECUTED_UNTERMINATED", correct=False,
                         child_pid=None, sandbox_owner=None)
            trace_sha.append(None)
        else:
            proof.update({
                "status": "PYTHON_EXIT", "correct": True, "exit_code": 0,
                "elapsed_milliseconds": 100, "timeout_kill_observed": False,
                "sandbox_owner": f"original-case-owner-{i:03d}", "child_pid": 10000 + i,
                "independently_observed_preexecution_source_sha256": case["candidate_program_sha256"],
                "sealed_memfd_source_matches": True,
                "python_isolated_mode": True,
                "observer_outside_workload_namespaces": True,
                "network_egress_denied": True, "host_secrets_denied": True,
                "cgroup_limits_enforced": True, "descendants_extinct_after": True,
                "observed_cgroup_empty_after": True,
            })
            trace = {
                "schema": certificate.TRACE_SCHEMA, "arm": arm,
                "model_sha256": witness_arm_sha, "session_id": session,
                "ordinal": i, "id": case["id"],
                "capture_sha256": certificate._sha(capture_raw),
                "program_sha256": case["candidate_program_sha256"],
                "sandbox_capability_sha256": cap_sha,
                "sandbox_profile_sha256": cap["sandbox_profile_sha256"],
                "runtime_host_sha256": cap["runtime_host_sha256"],
                "python_executable_sha256": cap["python_executable_sha256"],
                "reviewed_observer_source_sha256": "3" * 64,
                "sandbox_owner": proof["sandbox_owner"], "child_pid": proof["child_pid"],
                "execution_status": "PYTHON_EXIT", "exit_code": 0, "correct": True,
                "elapsed_milliseconds": 100, "timeout_kill_observed": False,
                "python_version": "3.13", "python_argv_shape": ["python", "-I", "candidate.py"],
                "isolated_nonroot_uid": True,
                "isolated_user_pid_mount_net_namespaces": True, "no_network_egress": True,
                "no_host_mounts_or_secrets": True, "source_staged_before_executable_release": True,
                "source_file_sha256": case["candidate_program_sha256"],
                "sealed_memfd_source_sha256": case["candidate_program_sha256"],
                "sealed_memfd_immutable": True, "memory_and_pid_cgroup_caps_enforced": True,
                "observed_cgroup_empty_after": True, "all_descendants_extinct_after": True,
                "outside_host_process_observer": True,
                "child_start_ticks": 1000 + i,
                "preexecution_source_observed_ns": 100_000_000_000,
                "execution_release_ns": 100_001_000_000,
                "execution_finished_ns": 100_101_000_000,
                "actual_cgroup_path": f"/original-frozen-case-{i:03d}",
            }
            blob = _json(trace_dir / f"{i:03d}.json", trace)
            trace_sha.append(certificate._sha(blob))
            proof["independent_trace_sha256"] = certificate._sha(blob)
        witness_rows.append(proof)
    witness = {
        "schema": previous.WITNESS_SCHEMA,
        "status": "INDEPENDENTLY_VERIFIED_ORIGINAL_FROZEN_164", "arm": arm,
        "model_sha256": witness_arm_sha, "session_id": session,
        "source_items_sha256": sha_source,
        "canonical_sha256": certificate._sha(source_raw),
        "capability_sha256": cap_sha,
        "sandbox_profile_sha256": cap["sandbox_profile_sha256"],
        "runtime_host_sha256": cap["runtime_host_sha256"],
        "python_executable_sha256": cap["python_executable_sha256"],
        "independent_correct": 162, "release_approved": False, "per_case": witness_rows,
    }
    witness_path = tmp_path / "outside-host-witness.json"
    witness_raw = _json(witness_path, witness)
    monkeypatch.setattr(previous, "APPROVED_OUTSIDE_HOST_WITNESS_SHA256",
                        {arm: certificate._sha(witness_raw)})
    review = {
        "schema": certificate.REVIEW_SCHEMA,
        "status": "INDEPENDENTLY_SOURCE_REVIEWED_FROZEN_164_TRACES",
        "arm": arm, "model_sha256": witness_arm_sha, "session_id": session,
        "canonical_source_sha256": certificate._sha(source_raw),
        "frozen_source_items_sha256": sha_source,
        "capture_sha256": certificate._sha(capture_raw),
        "witness_sha256": certificate._sha(witness_raw),
        "sandbox_capability_sha256": cap_sha,
        "sandbox_profile_sha256": cap["sandbox_profile_sha256"],
        "runtime_host_sha256": cap["runtime_host_sha256"],
        "python_executable_sha256": cap["python_executable_sha256"],
        "reviewed_observer_source_sha256": "3" * 64,
        "observer_outside_workload_namespaces": True,
        "observer_independently_reviewed": True,
        "case_trace_sha256": trace_sha,
    }
    review_path = tmp_path / "independent-source-review.json"
    review_raw = _json(review_path, review)
    monkeypatch.setattr(certificate, "APPROVED_INDEPENDENT_TRACE_REVIEW_SHA256",
                        {arm: certificate._sha(review_raw), "dense_v2": "f" * 64})
    arguments = {
        "arm": arm, "capture_path": capture_path, "witness_path": witness_path,
        "capability_path": tmp_path / "synth-capability",
        "broker_path": tmp_path / "synth-broker", "review_path": review_path,
        "trace_dir": trace_dir, "serve_path": tmp_path / "synth-serve",
        "owner_review_path": tmp_path / "synth-owner", "canonical_path": canonical,
    }
    return {"args": arguments, "cases": cases, "witness": witness,
            "review": review, "cap": cap, "monkeypatch": monkeypatch}


def _repin_receipts(specimens):
    """TEST-ONLY resealing demonstrates digest checks alone are not approval."""
    args = specimens["args"]
    monkeypatch = specimens["monkeypatch"]
    witness_raw = _json(args["witness_path"], specimens["witness"])
    monkeypatch.setattr(previous, "APPROVED_OUTSIDE_HOST_WITNESS_SHA256",
                        {"reap128_v2": certificate._sha(witness_raw)})
    specimens["review"]["witness_sha256"] = certificate._sha(witness_raw)
    review_raw = _json(args["review_path"], specimens["review"])
    monkeypatch.setattr(certificate, "APPROVED_INDEPENDENT_TRACE_REVIEW_SHA256",
                        {"reap128_v2": certificate._sha(review_raw), "dense_v2": "f" * 64})


def test_real_frozen_source_hashes_and_security_gaps():
    assert certificate.SOURCE_HUMANEVAL164_SHA256 == reap_eval.FROZEN_V2_QUALITY_SOURCES["humaneval"][1]
    assert certificate.MODEL_SHA256["reap128_v2"] == previous.MODELS["reap128_v2"]
    assert certificate.MODEL_SHA256["dense_v2"] == previous.MODELS["dense_v2"]
    assert not certificate.APPROVED_INDEPENDENT_TRACE_REVIEW_SHA256
    gaps = certificate.approval_gaps()
    assert any("APPROVED_SANDBOX_CAPABILITY_SHA256" in s for s in gaps)
    assert any("164_RUNTIME_TRACE_REVIEW_SHA" in s for s in gaps)


def test_genuine_original_archived_sources_remain_research_only():
    try:
        q8 = previous.research_archive_inventory("q8")
        q4 = previous.research_archive_inventory("q4")
    except (OSError, ValueError):
        pytest.skip("private original archive unavailable on host")
    assert q8["cases"] == q4["cases"] == 164
    assert q8["terminated"] == q4["terminated"] == 162
    assert q8["original_rows_sha256"] == "cd3c9d1577611a032b36e69a3f96bd272d118819074352503f143bcd68cac301"
    assert q4["original_rows_sha256"] == "92ec6439be77a7dbeb1a80448a0bb664e3cb6113a9cc0d356f7dbe300db69c43"
    assert q8["frozen_gate2"] == q4["frozen_gate2"] == "UNKNOWN"


def test_synthetic_full_review_produces_diagnostic_only(specimens):
    result = certificate.verify_arm(**specimens["args"])
    assert result["correct"] == result["executed_with_verified_trace"] == 162
    assert result["nonterminated_original_nonpasses"] == 2
    assert result["population"] == 164
    assert result["gate2"].startswith("UNKNOWN")
    assert result["release_approved"] is False and result["chosen"] is None


@pytest.mark.parametrize(("mutation", "error"), [
    ("unpin_review", "security approval"),
    ("missing_trace", "missing, linked or oversized"),
    ("replaced_trace", "per-case trace fields disagree"),
    ("forged_source_sha", "per-case trace fields disagree"),
    ("reordered_original_ids", "witness does not pair"),
    ("reused_owner", "replayed or reused"),
    ("reused_cgroup", "replayed or reused"),
    ("wrong_start_ticks", "start-time/ordering"),
    ("trace_says_no_memfd", "isolation/immutable"),
    ("egress_open", "isolation/immutable"),
    ("no_descendant_extinction", "isolation/immutable"),
    ("wrong_capture_source", "native capture SHA"),
    ("changed_164_witness", "outside-host witness SHA"),
    ("unterminated_executed", "unterminated original item"),
])
def test_falsified_original_security_and_case_receipts_refuse(specimens, mutation, error):
    args = specimens["args"]
    witness = specimens["witness"]
    review = specimens["review"]
    if mutation == "unpin_review":
        specimens["monkeypatch"].setattr(certificate, "APPROVED_INDEPENDENT_TRACE_REVIEW_SHA256", {})
    elif mutation == "missing_trace":
        (args["trace_dir"] / "000.json").unlink()
    elif mutation in ("replaced_trace", "forged_source_sha", "reused_owner",
                      "reused_cgroup", "wrong_start_ticks", "trace_says_no_memfd",
                      "egress_open", "no_descendant_extinction"):
        path = args["trace_dir"] / ("001.json" if mutation in ("reused_owner", "reused_cgroup") else "000.json")
        trace = json.loads(path.read_text())
        if mutation == "replaced_trace":
            trace["correct"] = False
        elif mutation == "forged_source_sha":
            trace["program_sha256"] = "f" * 64
        elif mutation == "reused_owner":
            trace["sandbox_owner"] = "original-case-owner-000"
            witness["per_case"][1]["sandbox_owner"] = "original-case-owner-000"
        elif mutation == "reused_cgroup":
            trace["actual_cgroup_path"] = "/original-frozen-case-000"
        elif mutation == "wrong_start_ticks":
            trace["child_start_ticks"] = 0
        elif mutation == "trace_says_no_memfd":
            trace["sealed_memfd_immutable"] = False
        elif mutation == "egress_open":
            trace["no_network_egress"] = False
        elif mutation == "no_descendant_extinction":
            trace["all_descendants_extinct_after"] = False
        raw = _json(path, trace)
        i = 1 if mutation in ("reused_owner", "reused_cgroup") else 0
        review["case_trace_sha256"][i] = certificate._sha(raw)
        witness["per_case"][i]["independent_trace_sha256"] = certificate._sha(raw)
        _repin_receipts(specimens)
    elif mutation == "reordered_original_ids":
        witness["per_case"][0]["id"] = "HumanEval/1"
        _repin_receipts(specimens)
    elif mutation == "wrong_capture_source":
        args["capture_path"].write_bytes(args["capture_path"].read_bytes() + b"\n")
    elif mutation == "changed_164_witness":
        witness["independent_correct"] = 164
        _json(args["witness_path"], witness)
    elif mutation == "unterminated_executed":
        witness["per_case"][10]["child_pid"] = 99
        _repin_receipts(specimens)
    with pytest.raises(certificate.CertificateRefusal, match=error):
        certificate.verify_arm(**args)


def test_existing_grade_export_accepts_fake_in_memory_outcomes_but_new_layer_refuses_without_traces(specimens):
    # This reproduces a real trust-boundary gap in the prior additive helper:
    # a caller-created grade with a copied pinned witness hash can change the
    # accepted pass@1 booleans. It must NOT be used as independent certificate.
    cases = specimens["cases"]
    forged = {
        "status": "INDEPENDENTLY_SOURCE_PINNED_FROZEN_164_CASE_RESULTS",
        "arm": "reap128_v2", "model_sha256": previous.MODELS["reap128_v2"],
        "source_items_sha256": previous.FROZEN_SOURCE_ITEMS_SHA256,
        "outside_host_witness_sha256": certificate._sha(
            specimens["args"]["witness_path"].read_bytes()),
        "release_approved": False, "session_id": cases[0]["session_id"],
        "outcomes": [{"id": c["id"], "correct": c["finished"],
                      "terminated": c["finished"]} for c in cases],
        "correct": 162,
    }
    forged["outcomes"][0]["correct"] = False
    forged["correct"] = 161
    # Current older export accepts this forged in-memory "grade".
    old = previous.build_selector_rows(cases, forged, arm="reap128_v2",
                                       canonical=specimens["args"]["canonical_path"])
    assert old["summary_entry"]["accuracy"] == 161 / 164
    # The new verifier REFUSES if external trace certificate unavailable.
    (specimens["args"]["trace_dir"] / "000.json").unlink()
    with pytest.raises(certificate.CertificateRefusal, match="missing, linked or oversized"):
        certificate.verify_arm(**specimens["args"])


def test_paired_original_gate2_statistic_is_exact_sample_sd_wald_on_item_ids(monkeypatch):
    def synthetic_arm(arm):
        n = 164
        if arm == "reap128_v2":
            pass_ids = set(range(120))
        else:
            pass_ids = set(range(110))
        return {"physical_host_sha256": "a" * 64,
                "session_id": "1" * 32 if arm == "reap128_v2" else "2" * 32,
                "frozen_source_items_sha256": certificate.SOURCE_HUMANEVAL164_SHA256,
                "outcomes": tuple((f"HumanEval/{i}", i in pass_ids) for i in range(n)),
                "correct": len(pass_ids)}
    monkeypatch.setattr(certificate, "verify_arm", lambda **kwargs: synthetic_arm(kwargs["arm"]))
    outcome = certificate.verify_paired_gate2(candidate={}, dense={})
    expected = reap_eval  # frozen method is in reap_gates, not an alternate McNemar CI
    assert expected is reap_eval
    from chowder_batch.reap_gates import lower_bound
    d = [1] * 10 + [0] * 154
    assert outcome["one_sided_sample_sd_wald_95_lower_bound"] == pytest.approx(lower_bound(d))
    assert outcome["paired_count"] == 164
    assert outcome["strict_frozen_threshold_met"] is True
    assert outcome["release_approved"] is False


@pytest.mark.parametrize("mode", ["same_session", "different_host", "bad_id", "nonbinary"])
def test_paired_gate2_refuses_incomparable_runs(monkeypatch, mode):
    def fake_arm(arm):
        rows = [(f"HumanEval/{i}", True) for i in range(164)]
        if mode == "bad_id" and arm == "dense_v2":
            rows[9] = ("HumanEval/999", True)
        if mode == "nonbinary" and arm == "dense_v2":
            rows[9] = ("HumanEval/9", 1)
        return {"physical_host_sha256": ("a" if arm == "reap128_v2" else (
            "b" if mode == "different_host" else "a")) * 64,
            "session_id": ("1" if arm == "reap128_v2" else (
                "1" if mode == "same_session" else "2")) * 32,
            "frozen_source_items_sha256": certificate.SOURCE_HUMANEVAL164_SHA256,
            "outcomes": tuple(rows), "correct": 164}
    monkeypatch.setattr(certificate, "verify_arm", lambda **kwargs: fake_arm(kwargs["arm"]))
    with pytest.raises(certificate.CertificateRefusal):
        certificate.verify_paired_gate2(candidate={}, dense={})
