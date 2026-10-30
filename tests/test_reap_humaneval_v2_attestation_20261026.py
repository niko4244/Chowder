"""Synthetic CPU-only adversarial v2 replay provenance checks.

Never executes any generated programs or reads private scoring responses.
"""

import hashlib
import json

import pytest

from chowder_batch import reap_humaneval_v2_attestation as v2


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def failed_capability():
    return {
        "schema": "reap-v2-private-human164-harmless-kaggle-sandbox-probe.v1",
        "status": "CAPABILITY_PROBE_ONLY_NOT_AN_EXECUTION_APPROVAL",
        "actual_cpu_kernel_euid": 0,
        "available_namespaced_launchers": [],
        "unshare_probe": {"status": "UNAVAILABLE", "exit_code": 1},
        "bubblewrap_present": False,
        "cgroup_delegation_proven": False,
        "new_user_pid_mount_network_namespaces_proven": False,
        "verified_complete_humaneval_execution_sandbox": False,
        "no_generated_python_executed": True,
        "no_model_loaded": True,
        "no_network_request_issued": True,
    }


def test_actual_negative_kaggle_capability_is_sha_bound_and_ineligible(monkeypatch):
    raw = encoded(failed_capability())
    monkeypatch.setattr(v2, "FAILED_KAGGLE_CAPABILITY_SHA256", hashlib.sha256(raw).hexdigest())
    result = v2.audit_failed_kaggle_capability(raw)
    assert result["status"] == "REFUSED_SANDBOX_UNAVAILABLE"
    assert result["generated_programs_executed"] == 0
    assert result["execution_approved"] is False


@pytest.mark.parametrize("field,value", [
    ("verified_complete_humaneval_execution_sandbox", True),
    ("cgroup_delegation_proven", True),
    ("no_generated_python_executed", False),
    ("available_namespaced_launchers", ["bwrap"]),
    ("actual_cpu_kernel_euid", 1000),
    ("status", "PASS"),
])
def test_kaggle_refusal_cannot_be_converted_to_approval(monkeypatch, field, value):
    fake = failed_capability()
    fake[field] = value
    raw = encoded(fake)
    # Even a deliberately replaced hash must not convert failure to approval.
    monkeypatch.setattr(v2, "FAILED_KAGGLE_CAPABILITY_SHA256", hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError, match="capability|refusal|isolation"):
        v2.audit_failed_kaggle_capability(raw)


def test_kaggle_negative_receipt_must_match_original_file_hash():
    with pytest.raises(ValueError, match="SHA"):
        v2.audit_failed_kaggle_capability(encoded(failed_capability()))


def test_six_field_replay_and_self_asserted_status_never_qualify():
    fake = {"status": "PASS_INDEPENDENT_ISOLATED_HUMANEVAL_FULL_164_PROVISIONAL_LINUX",
            "independent_full_rows": 164, "matched_recorded_scores": 164,
            "correct": 152, "session_id": "95876063255e45159f57e29306e0e885",
            "original_row_sha256": "cd3c9d1577611a032b36e69a3f96bd272d118819074352503f143bcd68cac301"}
    with pytest.raises(ValueError, match="independent|approved|witness|SHA"):
        v2.verify_original_q8_v2_attestation(
            replay_raw=encoded(fake), capability_raw=encoded(fake),
            external_witness_raw=encoded(fake), canonical_raw=b"[]",
            original_rows_raw=b"", preflight_raw=b"{}")


def test_no_execution_gate_or_synthetic_claim_can_approve_replay():
    assert "APPROVED_SANDBOX_CAPABILITY_SHA256_REQUIRED" in v2.execution_admission_gaps()
    assert "APPROVED_INDEPENDENT_WITNESS_SHA256_REQUIRED" in v2.execution_admission_gaps()
    with pytest.raises(ValueError, match="independent|approved|witness"):
        v2.verify_original_q8_v2_attestation(
            replay_raw=b"{}", capability_raw=b"{}", external_witness_raw=b"{}",
            canonical_raw=b"[]", original_rows_raw=b"", preflight_raw=b"{}")


def test_json_duplicate_keys_and_nonfinite_numbers_are_refused(monkeypatch):
    # A trusted file hash is only useful if its JSON has a unique meaning.
    for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}'):
        monkeypatch.setattr(v2, "FAILED_KAGGLE_CAPABILITY_SHA256", hashlib.sha256(raw).hexdigest())
        with pytest.raises(ValueError):
            v2.audit_failed_kaggle_capability(raw)


def synthetic_full_164(monkeypatch, corrupt=None):
    """Build fully artificial evidence; never read or execute real code."""
    canonical = [{"id": f"HumanEval/{i}", "header": "def example():\n",
                  "entry_point": "example", "test": "def check(f): pass\n"}
                 for i in range(164)]
    original = [{"id": f"HumanEval/{i}", "eval_session_id": v2.SESSION,
                 "text": "def example():\n    return 1\n", "terminated": i < 162,
                 "correct": i < 152} for i in range(164)]
    canonical_raw = encoded(canonical)
    rows_raw = b"".join(encoded(row) + b"\n" for row in original)
    monkeypatch.setattr(v2, "CANONICAL_SHA", hashlib.sha256(canonical_raw).hexdigest())
    monkeypatch.setattr(v2, "ORIGINAL_ROW_SHA", hashlib.sha256(rows_raw).hexdigest())
    preflight = {"arm": "reap128_v2_q8", "session_id": v2.SESSION,
                 "original_humaneval_row_sha256": v2.ORIGINAL_ROW_SHA,
                 "canonical_humaneval_raw_sha256": v2.CANONICAL_SHA,
                 "sandbox_verified": False, "candidate_code_executed": False}
    preflight_raw = encoded(preflight)
    monkeypatch.setattr(v2, "PREFLIGHT_SHA", hashlib.sha256(preflight_raw).hexdigest())
    capability = {
        "schema": "chowder.reap-v2-external-sandbox-capability.v2",
        "status": "INDEPENDENTLY_VERIFIED_SANDBOX_V2",
        "session_id": v2.SESSION,
        "runtime_host_id": "d" * 64,
        "runtime_image_sha256": "e" * 64,
        "independent_benign_probe_sha256": "f" * 64,
        "sandbox_profile_sha256": "1" * 64,
        "python_major_minor": "3.13",
        "python_executable_sha256": "c" * 64,
        "python_argv_shape": ["python", "-I", "candidate.py"],
        "candidate_working_directory": "temporary_cwd",
        "python_timeout_seconds": 10, "rlimit_as_bytes": 2 * 1024**3,
        "rlimit_fsize_bytes": 1024**2, "cgroup_memory_max_bytes": 805306368,
        "cgroup_pids_max": 16,
    }
    capability.update({name: True for name in (
        "independent_user_namespace", "independent_pid_namespace",
        "independent_mount_namespace", "independent_network_namespace",
        "network_egress_denied", "independent_process_tree_teardown",
        "cgroup_memory_limit_enforced", "cgroup_pids_limit_enforced",
        "read_only_host_mounts", "no_host_secrets", "no_new_privileges",
        "unprivileged_worker", "fixed_benign_probes_only",
        "independent_vm_process_observer")})
    cap_raw = encoded(capability)
    monkeypatch.setattr(v2, "APPROVED_SANDBOX_CAPABILITY_SHA256", hashlib.sha256(cap_raw).hexdigest())
    replay = {"schema": "chowder.reap-v2-human164-independent-replay.v2",
              "status": "COMPLETE_ORIGINAL_Q8_REPLAY_V2_RESEARCH_ONLY",
              "arm": "reap128_v2_q8", "model_sha256": v2.Q8_SHA,
              "session_id": v2.SESSION,
              "original_machine_receipt_sha256": v2.MACHINE_SHA,
              "original_independent_full_audit_sha256": v2.TRUSTED_AUDIT_SHA,
              "original_preexecution_preflight_sha256": v2.PREFLIGHT_SHA,
              "original_humaneval_row_sha256": v2.ORIGINAL_ROW_SHA,
              "canonical_humaneval_raw_sha256": v2.CANONICAL_SHA,
              "original_evaluator_sha256": v2.EVALUATOR_SHA,
              "original_scorer_sha256": v2.SCORER_SHA,
              "sandbox_capability_sha256": hashlib.sha256(cap_raw).hexdigest(),
              "source_rows": 164, "gen0_status": "BLOCKED", "chosen": None,
              "release_approved": False, "original_windows_gate_eligible": False,
              "actually_executed_programs": 162, "matched_recorded_scores": 164,
              "correct": 152, "per_task": []}
    witness = {"schema": "chowder.reap-v2-external-process-witness.v2",
               "status": "COMPLETE_INDEPENDENT_VM_TASK_WITNESS_V2",
               "session_id": v2.SESSION, "model_sha256": v2.Q8_SHA,
               "runtime_host_id": "d" * 64,
               "runtime_image_sha256": "e" * 64,
               "sandbox_profile_sha256": "1" * 64,
               "sandbox_capability_sha256": hashlib.sha256(cap_raw).hexdigest(),
               "per_task": []}
    traces = []
    for index, (item, orig) in enumerate(zip(canonical, original)):
        task = {"index": index, "id": item["id"],
                "original_correct": orig["correct"],
                "original_terminated": orig["terminated"],
                "independent_correct": orig["correct"]}
        obs = {"index": index, "id": item["id"]}
        if index >= 162:
            task.update(execution_status="NOT_EXECUTED_UNTERMINATED",
                        program_sha256=None, exit_code=None)
            obs.update(status="NOT_EXECUTED_UNTERMINATED", child_seen=False)
            traces.append(None)
        else:
            code = 0 if orig["correct"] else 1
            sha = hashlib.sha256(v2.exact_program(item, orig).encode()).hexdigest()
            task.update(execution_status="PYTHON_EXIT", exit_code=code, program_sha256=sha)
            obs.update(status="INDEPENDENTLY_OBSERVED_CHILD_EXTINCT",
                       program_sha256=sha, execution_status="PYTHON_EXIT",
                       observed_python_exit_code=code,
                       python_argv_shape=["python", "-I", "candidate.py"],
                       python_major_minor="3.13",
                       python_executable_sha256="c" * 64,
                       python_isolated_mode=True, candidate_file_basename="candidate.py",
                       runtime_host_id="d" * 64,
                       sandbox_profile_sha256="1" * 64,
                       python_timeout_seconds=10,
                       candidate_working_directory="temporary_cwd",
                       child_seen=True, cgroup_empty_after=True,
                       descendants_extinct_after=True,
                       independent_child_starttime_ticks=index + 1,
                       exclusive_cgroup_owner=f"synthetic-owned-cgroup-{index}",
                       process_trace_sha256=None)
            trace = {"schema": "chowder.reap-v2-independent-process-trace.v2",
                     "index": index, "id": item["id"],
                     "session_id": v2.SESSION, "model_sha256": v2.Q8_SHA,
                     "program_sha256": sha, "runtime_host_id": "d" * 64,
                     "runtime_image_sha256": "e" * 64,
                     "sandbox_profile_sha256": "1" * 64,
                     "exclusive_cgroup_owner": f"synthetic-owned-cgroup-{index}",
                     "independent_child_starttime_ticks": index + 1,
                     "child_pid": index + 2000,
                     "start_monotonic_ns": (index + 1) * 1000000000,
                     "end_monotonic_ns": (index + 1) * 1000000000 + 1000000,
                     "elapsed_milliseconds": 1,
                     "execution_status": "PYTHON_EXIT",
                     "observed_python_exit_code": code,
                     "observer_outside_workload_namespaces": True,
                     "verified_no_egress": True,
                     "observed_cgroup_empty_after": True,
                     "observed_descendants_extinct_after": True,
                     "timeout_kill_observed": False}
            raw_trace = encoded(trace)
            traces.append(raw_trace)
            obs["process_trace_sha256"] = hashlib.sha256(raw_trace).hexdigest()
        replay["per_task"].append(task)
        witness["per_task"].append(obs)
    if corrupt:
        corrupt(capability, replay, witness)
        cap_raw = encoded(capability)
        monkeypatch.setattr(v2, "APPROVED_SANDBOX_CAPABILITY_SHA256", hashlib.sha256(cap_raw).hexdigest())
        replay["sandbox_capability_sha256"] = hashlib.sha256(cap_raw).hexdigest()
        witness["sandbox_capability_sha256"] = hashlib.sha256(cap_raw).hexdigest()
    replay_raw = encoded(replay)
    witness["replay_sha256"] = hashlib.sha256(replay_raw).hexdigest()
    witness_raw = encoded(witness)
    monkeypatch.setattr(v2, "APPROVED_INDEPENDENT_WITNESS_SHA256", hashlib.sha256(witness_raw).hexdigest())
    return {"replay_raw": replay_raw, "capability_raw": cap_raw,
            "external_witness_raw": witness_raw, "canonical_raw": canonical_raw,
            "original_rows_raw": rows_raw, "preflight_raw": preflight_raw,
            "external_trace_raws": traces}


def test_synthetic_complete_164_only_reports_provisional_research(monkeypatch):
    result = v2.verify_original_q8_v2_attestation(**synthetic_full_164(monkeypatch))
    assert result == {"status": "RESEARCH_ONLY_ORIGINAL_Q8_REPLAY_MATCHED",
                      "source_rows": 164, "actually_executed_programs": 162,
                      "matched_recorded_scores": 164, "correct": 152,
                      "original_windows_gate_eligible": False,
                      "gen0_status": "BLOCKED", "chosen": None,
                      "release_approved": False}


@pytest.mark.parametrize("corrupt", [
    lambda cap, replay, witness: cap.update(network_egress_denied=False),
    lambda cap, replay, witness: cap.update(cgroup_memory_max_bytes=0),
    lambda cap, replay, witness: cap.update(cgroup_pids_max=3000),
    lambda cap, replay, witness: cap.update(runtime_host_id="5" * 64),
    lambda cap, replay, witness: cap.update(python_executable_sha256="bad"),
    lambda cap, replay, witness: witness["per_task"][0].update(
        python_executable_sha256="d" * 64),
    lambda cap, replay, witness: replay["per_task"].pop(),
    lambda cap, replay, witness: replay["per_task"][0].update(exit_code=125),
    lambda cap, replay, witness: replay["per_task"][0].update(program_sha256="b" * 64),
    lambda cap, replay, witness: witness["per_task"][1].update(cgroup_empty_after=False),
    lambda cap, replay, witness: witness["per_task"][1].update(
        exclusive_cgroup_owner="synthetic-owned-cgroup-0"),
    lambda cap, replay, witness: witness["per_task"][1].update(
        process_trace_sha256="0" * 63 + "1"),
    lambda cap, replay, witness: witness["per_task"][162].update(
        execution_status="PYTHON_EXIT", observed_python_exit_code=0),
    lambda cap, replay, witness: witness["per_task"][162].update(
        process_trace_sha256="f" * 64),
    lambda cap, replay, witness: witness["per_task"][0].update(observed_python_exit_code=1),
    lambda cap, replay, witness: replay["per_task"][162].update(independent_correct=True),
    lambda cap, replay, witness: replay.update(session_id="f" * 32),
    lambda cap, replay, witness: witness["per_task"][0].update(
        python_argv_shape=["python", "-I", "-"]),
])
def test_even_source_pinned_synthetic_mutants_are_refused(monkeypatch, corrupt):
    sources = synthetic_full_164(monkeypatch, corrupt)
    with pytest.raises(ValueError):
        v2.verify_original_q8_v2_attestation(**sources)


@pytest.mark.parametrize("mutate", [
    lambda trace: trace.update(end_monotonic_ns=trace["start_monotonic_ns"] + 10_000_000_000),
    lambda trace: trace.update(elapsed_milliseconds=10_000),
    lambda trace: trace.update(host_writable_mounts_visible=True),
    lambda trace: trace.update(observer_outside_workload_namespaces=False),
    lambda trace: trace.update(timeout_kill_observed=True),
])
def test_even_resealed_external_trace_claims_require_coherent_timing_and_contract(
    monkeypatch, mutate,
):
    sources = synthetic_full_164(monkeypatch)
    trace = json.loads(sources["external_trace_raws"][0])
    mutate(trace)
    sources["external_trace_raws"][0] = encoded(trace)
    # Simulate a compromised/incorrectly approving independent witness source.
    # Re-pin both the trace digest and whole witness to ensure the verification
    # actually checks trace semantics, rather than merely recognizing its hash.
    witness = json.loads(sources["external_witness_raw"])
    witness["per_task"][0]["process_trace_sha256"] = hashlib.sha256(
        sources["external_trace_raws"][0]
    ).hexdigest()
    sources["external_witness_raw"] = encoded(witness)
    monkeypatch.setattr(v2, "APPROVED_INDEPENDENT_WITNESS_SHA256", hashlib.sha256(
        sources["external_witness_raw"]
    ).hexdigest())
    with pytest.raises(ValueError):
        v2.verify_original_q8_v2_attestation(**sources)
