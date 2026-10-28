"""SYNTHETIC ADVERSARIAL FIXTURES ONLY; never model results or sandbox approval."""

import hashlib
import json
from pathlib import Path

import pytest

from chowder_batch import reap_eval, reap_select
from chowder_batch import reap_v2_original_humaneval_frozen_executor_20261027 as frozen


def _write(path: Path, value) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    path.write_bytes(raw)
    return raw


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def synthetic_original(tmp_path, monkeypatch):
    """Explicitly fabricated 164-source cohort, with only test-time SHA pins."""
    tasks = []
    for i in range(164):
        header = f"def synthetic_function_{i}():\n    pass\n"
        tasks.append({"id": f"HumanEval/{i}", "prompt": (
            "Complete this Python function. Reply with the full function in one ```python "
            f"code block.\n\n```python\n{header}```"),
            "header": header, "entry_point": f"synthetic_function_{i}",
            "test": f"\ndef check(candidate):\n    assert candidate() == {i}\n"})
    canonical = tmp_path / "synthetic-canonical-164.json"
    raw = _write(canonical, tasks)
    monkeypatch.setattr(frozen, "FROZEN_CANONICAL_SHA256", _sha(raw))
    source_sha = reap_eval._digest(tasks)
    monkeypatch.setattr(frozen, "FROZEN_SOURCE_ITEMS_SHA256", source_sha)
    monkeypatch.setitem(reap_eval.FROZEN_V2_QUALITY_SOURCES, "humaneval", (164, source_sha))
    arm = "reap128_v2"
    session = "a" * 32
    serve_path = tmp_path / "serve.json"
    _write(serve_path, {"gguf": "C:/SYNTHETIC_DO_NOT_LOAD.gguf",
                        "gguf_sha256": frozen.MODELS[arm],
                        "flags": reap_select.FROZEN_SERVE_FLAGS,
                        "eval_session_id": session, "server_pid": 9999, "server_port": 8811})
    review_path = tmp_path / "independent-review-SYNTHETIC.json"
    raw = _write(review_path, {
        "schema": frozen.SESSION_SCHEMA,
        "status": "INDEPENDENTLY_VERIFIED_NATIVE_WINDOWS_FROZEN_OWNER",
        "arm": arm, "model_sha256": frozen.MODELS[arm], "session_id": session,
        "serve_sha256": _sha(serve_path.read_bytes()),
        "physical_windows_owner_verified": True,
        "real_original_model_bytes_verified": True,
        "per_case_process_binding_verified": True,
        "serving_geometry_verified": True,
        "uninterrupted_original_164_run": True,
        "physical_host_sha256": "c" * 64,
        "native_server_source_sha256": "d" * 64,
    })
    monkeypatch.setattr(frozen, "APPROVED_NATIVE_OWNER_REVIEW_SHA256", {arm: _sha(raw)})
    return canonical, serve_path, review_path, tasks


def _response(i, *, finish="stop", text=None):
    if text is None:
        text = f"```python\ndef synthetic_function_{i}():\n    return {i}\n```"
    return {"text": text, "finish_reason": finish, "completion_tokens": 25,
            "seconds": 1.5, "timings": {"predicted_per_second": 19.3}}


def _capture(i, fixture, *, finish="stop"):
    canonical, serve, review, _ = fixture
    return frozen.capture_case("reap128_v2", i, _response(i, finish=finish),
                               serve_path=serve, owner_review_path=review,
                               canonical=canonical)


def test_production_source_owned_execution_admission_is_unset():
    assert frozen.APPROVED_NATIVE_OWNER_REVIEW_SHA256 == {}
    assert frozen.APPROVED_NATIVE_CAPTURE_COHORT_SHA256 == {}
    assert frozen.APPROVED_SANDBOX_CAPABILITY_SHA256 is None
    assert frozen.APPROVED_SANDBOX_BROKER_SHA256 is None
    assert frozen.APPROVED_OUTSIDE_HOST_WITNESS_SHA256 == {}
    assert frozen.execution_admission_gaps()


def test_real_archived_source_only_receipts_have_exact_original_164_ids_if_available():
    from chowder_batch import reap_humaneval_v38_actual_execution_20261027 as original
    if not original.CANONICAL.is_file() or not original.Q8_ROWS.is_file():
        pytest.skip("actual original archived source not mounted on this test machine")
    items = frozen.canonical_tasks()
    assert len(items) == 164 and items[0]["id"] == "HumanEval/0"
    assert items[-1]["id"] == "HumanEval/163"
    result = frozen.research_archive_inventory("q8")
    assert result["original_rows_sha256"] == original.Q8_ROW_SHA256
    assert result["session_id"] == original.Q8_SESSION
    assert result["terminated"] == 162
    assert result["frozen_gate2"] == "UNKNOWN"
    assert result["generated_code_executed_by_this_module"] is False


def test_exact_original_prompt_and_decode_controls(synthetic_original):
    canonical, _, _, tasks = synthetic_original
    result = frozen.frozen_request(0, canonical=canonical)
    r = result["request"]
    assert r["messages"] == [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": tasks[0]["prompt"]},
    ]
    assert (r["temperature"], r["top_k"], r["seed"], r["max_tokens"]) == (0, 1, 7, 768)
    assert r["chat_template_kwargs"] == {"enable_thinking": False}
    with pytest.raises(frozen.FrozenExecutionRefusal, match="unregistered"):
        frozen.frozen_request(164, canonical=canonical)


def test_source_only_capture_builds_original_python_without_running_it(synthetic_original):
    case = _capture(17, synthetic_original)
    assert case["candidate_program_sha256"]
    assert case["frozen_correct"] is None
    assert case["generated_code_executed"] is False
    tasks = frozen.canonical_tasks(synthetic_original[0])
    _, source = frozen._candidate_packet(case, tasks)
    assert source.endswith(b"check(synthetic_function_17)\n")
    assert b"return 17" in source


@pytest.mark.parametrize("invalid_rate", [-1.0, float("nan"), True, "20"])
def test_reopened_native_capture_rejects_invalid_token_rate(synthetic_original, invalid_rate):
    case = _capture(17, synthetic_original)
    case["predicted_per_second"] = invalid_rate
    tasks = frozen.canonical_tasks(synthetic_original[0])
    with pytest.raises(frozen.FrozenExecutionRefusal, match="native token-rate"):
        frozen._candidate_packet(case, tasks)


def test_unterminated_generation_is_never_sent_for_execution(synthetic_original):
    case = _capture(2, synthetic_original, finish="length")
    assert case["candidate_program_sha256"] is None
    _, raw = frozen._candidate_packet(case, frozen.canonical_tasks(synthetic_original[0]))
    assert raw is None


def test_source_locked_original_164_capture_exclusive_and_no_overwrite(tmp_path, synthetic_original):
    cases = [_capture(i, synthetic_original, finish="length" if i in (11, 91) else "stop")
             for i in range(164)]
    dest = tmp_path / "frozen-capture-only.jsonl"
    result = frozen.seal_capture(cases, dest, arm="reap128_v2", canonical=synthetic_original[0])
    assert result["status"] == "CAPTURED_164_UNSCORED_ONLY"
    assert result["captured_rows"] == 164 and result["original_frozen_gate2"] == "UNKNOWN"
    assert result["captured_rows_sha256"] == _sha(dest.read_bytes())
    assert len(dest.read_bytes().splitlines()) == 164
    with pytest.raises(frozen.FrozenExecutionRefusal, match="new exclusive"):
        frozen.seal_capture(cases, dest, arm="reap128_v2", canonical=synthetic_original[0])
    cases[7]["candidate_program_sha256"] = "f" * 64
    with pytest.raises(frozen.FrozenExecutionRefusal, match="program SHA mismatch"):
        frozen.seal_capture(cases, tmp_path / "not-created.jsonl",
                            arm="reap128_v2", canonical=synthetic_original[0])


@pytest.mark.parametrize(("mutation", "error"), [
    ("wrong_model", "serving owner SHA mismatch|model/Windows owner"),
    ("wrong_slot", "serving owner SHA mismatch|four-slot"),
    ("wrong_owner_review", "owner SHA mismatch"),
    ("wrong_source", "raw SHA mismatch"),
    ("missing_model_pin", "owner approval missing"),
    ("wrong_case_ordinal", "ordinal"),
    ("wrong_finish", "generation"),
    ("out_of_range_tokens", "generation"),
])
def test_original_capture_rejects_adversarial_inputs(
    synthetic_original, monkeypatch, mutation, error,
):
    canonical, serve, review, tasks = synthetic_original
    ordinal = 1
    response = _response(ordinal)
    if mutation in ("wrong_model", "wrong_slot"):
        data = json.loads(serve.read_text())
        if mutation == "wrong_model":
            data["gguf_sha256"] = "b" * 64
        else:
            data["flags"] = ["--fit", "on", "-np", "1", "-c", "16384", "--jinja"]
        _write(serve, data)
    elif mutation == "wrong_owner_review":
        data = json.loads(review.read_text())
        data["status"] = "UNREVIEWED"
        _write(review, data)
    elif mutation == "wrong_source":
        tasks[1]["prompt"] += "post-hoc edited prompt"
        _write(canonical, tasks)
    elif mutation == "missing_model_pin":
        monkeypatch.setattr(frozen, "APPROVED_NATIVE_OWNER_REVIEW_SHA256", {})
    elif mutation == "wrong_case_ordinal":
        ordinal = 164
    elif mutation == "wrong_finish":
        response["finish_reason"] = "untrusted"
    elif mutation == "out_of_range_tokens":
        response["completion_tokens"] = 769
    with pytest.raises(frozen.FrozenExecutionRefusal, match=error):
        frozen.capture_case("reap128_v2", ordinal, response, serve_path=serve,
                            owner_review_path=review, canonical=canonical)


def test_q4_frozen_gate2_cannot_be_sourced_from_research_outputs(synthetic_original):
    canonical, serve, review, _ = synthetic_original
    with pytest.raises(frozen.FrozenExecutionRefusal, match="Q8 candidate or original dense"):
        frozen.capture_case("reap128_v2_q4", 0, _response(0), serve_path=serve,
                            owner_review_path=review, canonical=canonical)


def test_unapproved_sandbox_never_releases_model_code(synthetic_original, monkeypatch):
    case = _capture(0, synthetic_original)
    def forbidden(*args, **kwargs):
        raise AssertionError("must not start a broker or execute generated Python")
    monkeypatch.setattr(frozen.subprocess, "run", forbidden)
    for operation in (frozen.prepare_isolation_handoff, frozen.execute_approved_broker):
        with pytest.raises(frozen.FrozenExecutionRefusal, match="sandbox capability/witness approval"):
            operation(case, broker_path=Path("C:/unapproved-broker.exe"),
                      capture_path=Path("C:/unapproved-capture.jsonl"),
                      capability_path=Path("C:/unapproved-capability.json"),
                      canonical=synthetic_original[0])


def test_original_safe_handoff_protocol_remains_stdin_only_with_synthetic_review(
    synthetic_original, tmp_path, monkeypatch,
):
    case = _capture(0, synthetic_original)
    broker = tmp_path / "SYNTHETIC_NOT_EXECUTABLE.exe"
    broker.write_bytes(b"NONEXECUTABLE-SYNTHETIC-BROKER")
    monkeypatch.setattr(frozen, "APPROVED_SANDBOX_BROKER_SHA256", _sha(broker.read_bytes()))
    capability = tmp_path / "SYNTHETIC-CAPABILITY.json"
    fields = ("rootless_isolated_user_pid_mount_net_namespaces", "no_network_egress",
              "readonly_host_and_no_secrets", "per_task_private_tmpfs", "no_new_privileges",
              "actual_cgroup_v2_memory_and_pids_enforced", "python_I_10_second_watchdog",
              "exact_original_python313_interpreter_pinned", "immutable_sealed_code_stdin_only",
              "outside_host_pre_execution_sha_witness", "all_descendants_extinct_after",
              "separate_original_task_cgroup", "independent_process_observer_outside_sandbox")
    data = {"schema": frozen.CAPABILITY_SCHEMA,
            "status": "INDEPENDENTLY_APPROVED_FROZEN_ORIGINAL_EXECUTION",
            "broker_sha256": frozen.APPROVED_SANDBOX_BROKER_SHA256,
            "sandbox_profile_sha256": "b" * 64,
            "runtime_host_sha256": "c" * 64,
            "sandbox_image_sha256": "d" * 64,
            "python_executable_sha256": "e" * 64,
            "python_version": "3.13", "entrypoint": ["python", "-I", "candidate.py"],
            "timeout_seconds": 10, "address_space_bytes": 2 * 1024**3,
            "output_limit_bytes": 1024**2,
            "controls": {key: True for key in fields}}
    monkeypatch.setattr(frozen.security, "execution_admission_gaps", lambda: [])
    monkeypatch.setattr(frozen, "APPROVED_SANDBOX_CAPABILITY_SHA256",
                        _sha(_write(capability, data)))
    cohort = tmp_path / "synthetic-reviewed-capture.jsonl"
    original_cases = [_capture(i, synthetic_original) for i in range(164)]
    approved = frozen.seal_capture(original_cases, cohort,
                                   arm="reap128_v2", canonical=synthetic_original[0])
    monkeypatch.setattr(frozen, "APPROVED_NATIVE_CAPTURE_COHORT_SHA256",
                        {"reap128_v2": approved["captured_rows_sha256"]})
    meta, packet = frozen.prepare_isolation_handoff(
        case, capture_path=cohort, capability_path=capability,
        broker_path=broker, canonical=synthetic_original[0])
    serialized = json.loads(packet)
    assert serialized["request"] == meta
    assert "return 0" in serialized["candidate_source_utf8"]
    assert meta["candidate_program_sha256"] == _sha(serialized["candidate_source_utf8"].encode())
    assert "BROKER_STDIN_ONLY" in meta["status"]
    case["candidate_program_sha256"] = "0" * 64
    with pytest.raises(frozen.FrozenExecutionRefusal, match="program SHA mismatch"):
        frozen.prepare_isolation_handoff(case, capability_path=capability,
                                          capture_path=cohort, broker_path=broker,
                                          canonical=synthetic_original[0])


def test_unapproved_original_full_grade_fails_closed_even_with_complete_rows(
    synthetic_original, tmp_path,
):
    cases = [_capture(i, synthetic_original) for i in range(164)]
    fake_witness = tmp_path / "unapproved-witness.json"
    _write(fake_witness, {"status": "PASS", "correct": 164})
    with pytest.raises(frozen.FrozenExecutionRefusal, match="native 164-response cohort SHA absent"):
        frozen.score_frozen_164(cases, fake_witness,
                                arm="reap128_v2", canonical=synthetic_original[0])


def test_original_program_builder_matches_archived_source_only_programs_if_available():
    from chowder_batch import reap_humaneval_v38_actual_execution_20261027 as original
    from chowder_batch.reap_humaneval_podman_replay import exact_program
    if not original.CANONICAL.is_file() or not original.Q8_ROWS.is_file():
        pytest.skip("original source-backed archived rows not mounted on this test runner")
    tasks = frozen.canonical_tasks()
    raw = original._bound_source(original.Q8_ROWS, original.Q8_ROW_SHA256, 64 * 1024 * 1024)
    rows = {r["id"]: r for r in (json.loads(x) for x in raw.splitlines())}
    verified = 0
    for task in tasks:
        row = rows[task["id"]]
        if row["terminated"]:
            assert frozen._original_program(task, row["text"]) == exact_program(task, row).encode("utf-8")
            verified += 1
    assert verified == 162


@pytest.fixture
def synthetic_approved_replay(synthetic_original, tmp_path, monkeypatch):
    """Fabricated permission and outside-host observations; test memory ONLY."""
    cases = [_capture(i, synthetic_original, finish="length" if i in (5, 77) else "stop")
             for i in range(164)]
    fake_cap = {"sandbox_profile_sha256": "b" * 64,
                "runtime_host_sha256": "c" * 64,
                "python_executable_sha256": "d" * 64}
    monkeypatch.setattr(frozen.security, "execution_admission_gaps", lambda: [])
    monkeypatch.setattr(frozen, "_capability", lambda *args: fake_cap)
    monkeypatch.setattr(frozen, "APPROVED_SANDBOX_CAPABILITY_SHA256", "a" * 64)
    rows = []
    for i, case in enumerate(cases):
        if not case["finished"]:
            rows.append({"ordinal": i, "id": case["id"], "session_id": case["session_id"],
                         "candidate_program_sha256": None,
                         "status": "NOT_EXECUTED_UNTERMINATED", "correct": False,
                         "child_pid": None, "sandbox_owner": None})
        else:
            rows.append({"ordinal": i, "id": case["id"], "session_id": case["session_id"],
                         "candidate_program_sha256": case["candidate_program_sha256"],
                         "status": "PYTHON_EXIT", "correct": True,
                         "independent_trace_sha256": f"{i+1:064x}",
                         "sandbox_owner": f"SYNTHETIC_OWNER_{i:03d}",
                         "child_pid": 10000 + i,
                         "independently_observed_preexecution_source_sha256":
                             case["candidate_program_sha256"],
                         "sealed_memfd_source_matches": True,
                         "python_isolated_mode": True,
                         "observer_outside_workload_namespaces": True,
                         "network_egress_denied": True, "host_secrets_denied": True,
                         "cgroup_limits_enforced": True,
                         "descendants_extinct_after": True,
                         "observed_cgroup_empty_after": True,
                         "elapsed_milliseconds": 100,
                         "exit_code": 0, "timeout_kill_observed": False})
    witness = {"schema": frozen.WITNESS_SCHEMA,
               "status": "INDEPENDENTLY_VERIFIED_ORIGINAL_FROZEN_164",
               "arm": "reap128_v2", "model_sha256": frozen.MODELS["reap128_v2"],
               "source_items_sha256": frozen.FROZEN_SOURCE_ITEMS_SHA256,
               "canonical_sha256": frozen.FROZEN_CANONICAL_SHA256,
               "capability_sha256": frozen.APPROVED_SANDBOX_CAPABILITY_SHA256,
               "sandbox_profile_sha256": fake_cap["sandbox_profile_sha256"],
               "runtime_host_sha256": fake_cap["runtime_host_sha256"],
               "python_executable_sha256": fake_cap["python_executable_sha256"],
               "session_id": cases[0]["session_id"],
               "independent_correct": 162, "release_approved": False,
               "per_case": rows}
    evidence = tmp_path / "SYNTHETIC-EXTERNAL-164-WITNESS.json"
    raw = _write(evidence, witness)
    monkeypatch.setattr(frozen, "APPROVED_OUTSIDE_HOST_WITNESS_SHA256",
                        {"reap128_v2": _sha(raw)})
    monkeypatch.setattr(frozen, "APPROVED_NATIVE_CAPTURE_COHORT_SHA256",
                        {"reap128_v2": _sha(frozen._serialized_capture(cases))})
    return cases, witness, evidence, synthetic_original[0], monkeypatch


def test_only_independently_pinned_synthetic_164_case_witness_can_build_selector_rows(
    synthetic_approved_replay,
):
    cases, _, path, canonical, _ = synthetic_approved_replay
    grade = frozen.score_frozen_164(cases, path, arm="reap128_v2",
                                    capability_path=Path("C:/SYNTHETIC-CAPABILITY"),
                                    broker_path=Path("C:/SYNTHETIC-BROKER"), canonical=canonical)
    assert grade["correct"] == 162
    assert len(grade["outcomes"]) == 164
    assert grade["release_approved"] is False
    assert "UNKNOWN" in grade["gate2"]
    prepared = frozen.build_selector_rows(cases, grade, arm="reap128_v2", canonical=canonical)
    assert prepared["status"] == "SELECTOR_COMPATIBLE_AUTHENTICATED_ROWS_NOT_PUBLISHED"
    assert prepared["summary_entry"]["n"] == 164
    assert prepared["summary_entry"]["accuracy"] == 162 / 164
    assert prepared["summary_entry"]["file_sha256"] == _sha(prepared["jsonl_bytes"])
    rows = [json.loads(line) for line in prepared["jsonl_bytes"].splitlines()]
    assert [r["id"] for r in rows] == [f"HumanEval/{i}" for i in range(164)]
    assert rows[5]["correct"] is False and rows[5]["terminated"] is False
    assert rows[0]["correct"] is True and rows[0]["terminated"] is True
    assert prepared["release_approved"] is False


def test_original_timeout_after_ten_seconds_is_strict_nonpass(synthetic_approved_replay):
    cases, witness, path, canonical, monkeypatch = synthetic_approved_replay
    witness["per_case"][0].update(status="PYTHON_TIMEOUT_10S", correct=False,
                                  exit_code=None, timeout_kill_observed=True,
                                  elapsed_milliseconds=10_500)
    witness["independent_correct"] = 161
    monkeypatch.setattr(frozen, "APPROVED_OUTSIDE_HOST_WITNESS_SHA256",
                        {"reap128_v2": _sha(_write(path, witness))})
    grade = frozen.score_frozen_164(
        cases, path, arm="reap128_v2",
        capability_path=Path("C:/SYNTHETIC-CAPABILITY"),
        broker_path=Path("C:/SYNTHETIC-BROKER"), canonical=canonical,
    )
    assert grade["correct"] == 161
    assert grade["outcomes"][0]["correct"] is False
    assert grade["release_approved"] is False


@pytest.mark.parametrize(("mutation", "error"), [
    ("wrong_original_program", "candidate SHA mismatch"),
    ("reused_owner", "isolation and independent process"),
    ("missing_host_observer", "isolation and independent process"),
    ("false_egress", "isolation and independent process"),
    ("missing_teardown", "isolation and independent process"),
    ("falsified_pass_exit", "exit and original pass"),
    ("late_success", "original 10-second timeout"),
    ("premature_timeout", "original 10-second timeout"),
    ("unterminated_executed", "unterminated response was executed"),
    ("wrong_session", "case/order/candidate SHA mismatch"),
    ("wrong_aggregate", "aggregate"),
    ("wrong_runtime_profile", "population/profile invalid"),
])
def test_fabricated_source_pinned_external_witness_still_fails_semantic_attestation(
    synthetic_approved_replay, mutation, error,
):
    cases, witness, path, canonical, monkeypatch = synthetic_approved_replay
    rows = witness["per_case"]
    if mutation == "wrong_original_program":
        rows[0]["candidate_program_sha256"] = "f" * 64
    elif mutation == "reused_owner":
        rows[1]["sandbox_owner"] = rows[0]["sandbox_owner"]
    elif mutation == "missing_host_observer":
        rows[0]["observer_outside_workload_namespaces"] = False
    elif mutation == "false_egress":
        rows[0]["network_egress_denied"] = False
    elif mutation == "missing_teardown":
        rows[0]["descendants_extinct_after"] = False
    elif mutation == "falsified_pass_exit":
        rows[0]["correct"] = True
        rows[0]["exit_code"] = 1
    elif mutation == "late_success":
        rows[0]["elapsed_milliseconds"] = 11_500
    elif mutation == "premature_timeout":
        rows[0].update(status="PYTHON_TIMEOUT_10S", correct=False,
                       exit_code=None, timeout_kill_observed=True)
        witness["independent_correct"] = 161
    elif mutation == "unterminated_executed":
        rows[5]["child_pid"] = 1234
    elif mutation == "wrong_session":
        rows[0]["session_id"] = "f" * 32
    elif mutation == "wrong_aggregate":
        witness["independent_correct"] = 164
    elif mutation == "wrong_runtime_profile":
        witness["sandbox_profile_sha256"] = "f" * 64
    # Re-pinning this FAKE source review in a test must not bypass semantics.
    monkeypatch.setattr(frozen, "APPROVED_OUTSIDE_HOST_WITNESS_SHA256",
                        {"reap128_v2": _sha(_write(path, witness))})
    with pytest.raises(frozen.FrozenExecutionRefusal, match=error):
        frozen.score_frozen_164(cases, path, arm="reap128_v2",
                                capability_path=Path("C:/SYNTHETIC-CAPABILITY"),
                                broker_path=Path("C:/SYNTHETIC-BROKER"), canonical=canonical)
