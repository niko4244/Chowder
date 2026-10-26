"""Source-only independent HumanEval replay-v2 attestation gate.

No subprocess, model generation, sandbox construction or generated Python is
executed here. A public JSON PASS is not a trust root. Only two separately
reviewed, *source-pinned* private evidence hashes could admit a future replay.
Both approval pins deliberately remain unset pending actual runtime proof.

Even a fully verified original Q8 replay is provisional Linux research; this
module NEVER certifies original Windows serving, speed, or Gen-0 selection.
"""

from __future__ import annotations

import hashlib
import re

from chowder_batch.reap_external_full_audit import (
    EVALUATOR_SHA, Q8_SHA, SCORER_SHA, _strict_json,
)
from chowder_batch.reap_humaneval_podman_replay import (
    CANONICAL_SHA, MACHINE_SHA, ORIGINAL_ROW_SHA, PREFLIGHT_SHA, SESSION,
    TRUSTED_AUDIT_SHA, exact_program,
)

# Observed private, harmless, separate Kaggle CPU probe (NOT execution-proof).
FAILED_KAGGLE_CAPABILITY_SHA256 = (
    "66c47b94f23c05a739242357bfa4f4628c51ec9f1ebe75e5eb4597e3b89e0991"
)

# An independent reviewer must verify actual namespace/cgroup/egress/runtime
# evidence on an entirely DIFFERENT eligible host BEFORE pinning these in a
# new reviewed source commit. Never accept hashes or flags from CLI / receipt.
APPROVED_SANDBOX_CAPABILITY_SHA256: str | None = None
APPROVED_INDEPENDENT_WITNESS_SHA256: str | None = None


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _object(raw: bytes, limit: int) -> dict:
    if not isinstance(raw, bytes) or not 0 < len(raw) <= limit:
        raise ValueError("missing or unbounded private attestation bytes")
    try:
        value = _strict_json(raw)
    except (UnicodeError, ValueError) as exc:
        raise ValueError("private attestation JSON is invalid") from exc
    if not isinstance(value, dict):
        raise ValueError("private attestation must be an object")
    return value


def _bound(raw: bytes, expected: str, limit: int) -> dict:
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("source-approved independent evidence SHA missing")
    if not isinstance(raw, bytes) or not 0 < len(raw) <= limit or _sha(raw) != expected:
        raise ValueError("private attestation SHA mismatch")
    return _object(raw, limit)


def audit_failed_kaggle_capability(raw: bytes) -> dict:
    """Recheck exact original private observation; never turn it into PASS."""
    observed = _bound(raw, FAILED_KAGGLE_CAPABILITY_SHA256, 64 * 1024)
    if (observed.get("schema") != "reap-v2-private-human164-harmless-kaggle-sandbox-probe.v1"
            or observed.get("status") != "CAPABILITY_PROBE_ONLY_NOT_AN_EXECUTION_APPROVAL"
            or type(observed.get("actual_cpu_kernel_euid")) is not int
            or observed["actual_cpu_kernel_euid"] != 0
            or observed.get("available_namespaced_launchers") != []
            or observed.get("bubblewrap_present") is not False
            or observed.get("cgroup_delegation_proven") is not False
            or observed.get("new_user_pid_mount_network_namespaces_proven") is not False
            or observed.get("verified_complete_humaneval_execution_sandbox") is not False
            or observed.get("no_generated_python_executed") is not True
            or observed.get("no_model_loaded") is not True
            or observed.get("no_network_request_issued") is not True
            or not isinstance(observed.get("unshare_probe"), dict)
            or observed["unshare_probe"].get("status") != "UNAVAILABLE"
            or type(observed["unshare_probe"].get("exit_code")) is not int
            or observed["unshare_probe"]["exit_code"] != 1):
        raise ValueError("actual negative Kaggle capability/refusal evidence inconsistent")
    return {"status": "REFUSED_SANDBOX_UNAVAILABLE",
            "capability_sha256": _sha(raw),
            "generated_programs_executed": 0,
            "execution_approved": False, "gen0_status": "BLOCKED",
            "release_approved": False}


def execution_admission_gaps() -> list[str]:
    """Neither historical Kaggle Q4 nor a six-field JSON can populate pins."""
    gaps = []
    if APPROVED_SANDBOX_CAPABILITY_SHA256 is None:
        gaps.append("APPROVED_SANDBOX_CAPABILITY_SHA256_REQUIRED")
    if APPROVED_INDEPENDENT_WITNESS_SHA256 is None:
        gaps.append("APPROVED_INDEPENDENT_WITNESS_SHA256_REQUIRED")
    return gaps


def _verify_provenance(source: dict, *, cap_sha: str) -> None:
    expected = {
        "arm": "reap128_v2_q8", "model_sha256": Q8_SHA,
        "session_id": SESSION, "original_machine_receipt_sha256": MACHINE_SHA,
        "original_independent_full_audit_sha256": TRUSTED_AUDIT_SHA,
        "original_preexecution_preflight_sha256": PREFLIGHT_SHA,
        "original_humaneval_row_sha256": ORIGINAL_ROW_SHA,
        "canonical_humaneval_raw_sha256": CANONICAL_SHA,
        "original_evaluator_sha256": EVALUATOR_SHA,
        "original_scorer_sha256": SCORER_SHA,
        "sandbox_capability_sha256": cap_sha,
        "source_rows": 164,
        "gen0_status": "BLOCKED", "chosen": None,
        "release_approved": False,
        "original_windows_gate_eligible": False,
    }
    for name, value in expected.items():
        if type(source.get(name)) is not type(value) or source[name] != value:
            raise ValueError(f"original Q8 source/session provenance mismatch: {name}")


def verify_original_q8_v2_attestation(
    *, replay_raw: bytes, capability_raw: bytes, external_witness_raw: bytes,
    canonical_raw: bytes, original_rows_raw: bytes, preflight_raw: bytes,
    external_trace_raws: list[bytes | None] | None = None,
) -> dict:
    """Verify 164 Q8 outcomes only against independently approved evidence.

    Positive proof requires actual reviewed source-pinned *two* private hashes,
    164 individual matching runtime witnesses, and original Q8 source hashes.
    This has no ability to change execution qualification or scientific gates.
    """
    gaps = execution_admission_gaps()
    if gaps:
        raise ValueError("independent externally approved replay witness unavailable: "
                         + ", ".join(gaps))
    capability = _bound(capability_raw, APPROVED_SANDBOX_CAPABILITY_SHA256, 256 * 1024)
    witness = _bound(external_witness_raw, APPROVED_INDEPENDENT_WITNESS_SHA256,
                     4 * 1024 * 1024)
    cap_sha, witness_sha = _sha(capability_raw), _sha(external_witness_raw)
    if cap_sha == FAILED_KAGGLE_CAPABILITY_SHA256:
        raise ValueError("refused Kaggle capability cannot authorize execution")
    host_id = capability.get("runtime_host_id")
    image_sha = capability.get("runtime_image_sha256")
    probe_sha = capability.get("independent_benign_probe_sha256")
    profile_sha = capability.get("sandbox_profile_sha256")
    if any(not isinstance(x, str) or not re.fullmatch(r"[0-9a-f]{64}", x)
           for x in (host_id, image_sha, probe_sha, profile_sha)):
        raise ValueError("approved sandbox requires pinned host, image, profile and probe identities")
    if (capability.get("schema") != "chowder.reap-v2-external-sandbox-capability.v2"
            or capability.get("status") != "INDEPENDENTLY_VERIFIED_SANDBOX_V2"
            or capability.get("session_id") != SESSION
            or capability.get("python_major_minor") != "3.13"
            or not isinstance(capability.get("python_executable_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", capability["python_executable_sha256"])
            or capability.get("python_argv_shape") != ["python", "-I", "candidate.py"]
            or capability.get("candidate_working_directory") != "temporary_cwd"
            or type(capability.get("python_timeout_seconds")) is not int
            or capability["python_timeout_seconds"] != 10
            or type(capability.get("rlimit_as_bytes")) is not int
            or capability["rlimit_as_bytes"] != 2 * 1024 ** 3
            or type(capability.get("rlimit_fsize_bytes")) is not int
            or capability["rlimit_fsize_bytes"] != 1024 ** 2
            or type(capability.get("cgroup_memory_max_bytes")) is not int
            or not 0 < capability["cgroup_memory_max_bytes"] <= 2 * 1024 ** 3
            or type(capability.get("cgroup_pids_max")) is not int
            or not 0 < capability["cgroup_pids_max"] <= 16
            or any(capability.get(name) is not True for name in (
                "independent_user_namespace", "independent_pid_namespace",
                "independent_mount_namespace", "independent_network_namespace",
                "network_egress_denied", "independent_process_tree_teardown",
                "cgroup_memory_limit_enforced", "cgroup_pids_limit_enforced",
                "read_only_host_mounts", "no_host_secrets", "no_new_privileges",
                "unprivileged_worker", "fixed_benign_probes_only",
                "independent_vm_process_observer",
            ))):
        raise ValueError("independently verified v2 runtime isolation/semantic capability missing")

    replay = _object(replay_raw, 4 * 1024 * 1024)
    if (replay.get("schema") != "chowder.reap-v2-human164-independent-replay.v2"
            or replay.get("status") != "COMPLETE_ORIGINAL_Q8_REPLAY_V2_RESEARCH_ONLY"):
        raise ValueError("legacy or self-declared replay PASS cannot be attested")
    # The witness pins replay SHA and the source pins witness SHA. The replay
    # MUST NOT also pin witness SHA: that would make an impossible circular
    # digest dependency or encourage an unchecked replacement pointer.
    _verify_provenance(replay, cap_sha=cap_sha)
    if (witness.get("schema") != "chowder.reap-v2-external-process-witness.v2"
            or witness.get("status") != "COMPLETE_INDEPENDENT_VM_TASK_WITNESS_V2"
            or witness.get("replay_sha256") != _sha(replay_raw)
            or witness.get("sandbox_capability_sha256") != cap_sha
            or witness.get("session_id") != SESSION
            or witness.get("model_sha256") != Q8_SHA
            or witness.get("runtime_host_id") != host_id
            or witness.get("runtime_image_sha256") != image_sha
            or witness.get("sandbox_profile_sha256") != profile_sha):
        raise ValueError("external witness is not tied to this exact replay/source/runtime")

    preflight = _bound(preflight_raw, PREFLIGHT_SHA, 2 * 1024 * 1024)
    if (preflight.get("arm") != "reap128_v2_q8"
            or preflight.get("session_id") != SESSION
            or preflight.get("original_humaneval_row_sha256") != ORIGINAL_ROW_SHA
            or preflight.get("canonical_humaneval_raw_sha256") != CANONICAL_SHA
            or preflight.get("sandbox_verified") is not False
            or preflight.get("candidate_code_executed") is not False):
        raise ValueError("original nonexecuting source preflight provenance missing")
    if (_sha(canonical_raw) != CANONICAL_SHA
            or _sha(original_rows_raw) != ORIGINAL_ROW_SHA):
        raise ValueError("original Q8 frozen canonical/rows SHA mismatch")
    canonical = _object(b'{"rows":' + canonical_raw + b'}', 2 * 1024 * 1024).get("rows")
    original_rows = [_object(line, 2 * 1024 * 1024) for line in original_rows_raw.splitlines()]
    tasks = replay.get("per_task")
    attestations = witness.get("per_task")
    if (not isinstance(canonical, list) or len(canonical) != 164
            or len(original_rows) != 164 or not isinstance(tasks, list)
            or len(tasks) != 164 or not isinstance(attestations, list)
            or len(attestations) != 164
            or not isinstance(external_trace_raws, list)
            or len(external_trace_raws) != 164
            or {item.get("id") for item in canonical if isinstance(item, dict)}
                != {f"HumanEval/{i}" for i in range(164)}):
        raise ValueError("individual original 164 source/runtime attestation missing")
    original_by_id = {row.get("id"): row for row in original_rows}
    if len(original_by_id) != 164:
        raise ValueError("original Q8 row identities are duplicated")
    matched = correct = executed = 0
    owners: set[str] = set()
    traces: set[str] = set()
    for index, (item, task, observed) in enumerate(zip(canonical, tasks, attestations)):
        if not all(isinstance(x, dict) for x in (item, task, observed)):
            raise ValueError("malformed individual runtime/witness source")
        ident = item["id"]
        original = original_by_id.get(ident)
        if (not isinstance(original, dict)
                or original.get("eval_session_id") != SESSION
                or type(original.get("correct")) is not bool
                or type(original.get("terminated")) is not bool
                or (original["correct"] and not original["terminated"])):
            raise ValueError("original independently scored Q8 item mismatch")
        for record in (task, observed):
            if (type(record.get("index")) is not int or record["index"] != index
                    or record.get("id") != ident):
                raise ValueError("individual v2 replay item order/identity mismatch")
        if (task.get("original_correct") is not original["correct"]
                or task.get("original_terminated") is not original["terminated"]):
            raise ValueError("individual v2 outcome not bound to original source")
        if not original["terminated"]:
            if (task.get("execution_status") != "NOT_EXECUTED_UNTERMINATED"
                    or task.get("independent_correct") is not False
                    or task.get("exit_code") is not None
                    or task.get("program_sha256") is not None
                    or observed.get("status") != "NOT_EXECUTED_UNTERMINATED"
                    or observed.get("child_seen") is not False
                    or set(task) != {"index", "id", "original_correct",
                                        "original_terminated", "independent_correct",
                                        "execution_status", "exit_code", "program_sha256"}
                    or set(observed) != {"index", "id", "status", "child_seen"}
                    or external_trace_raws[index] is not None):
                raise ValueError("original unterminated code was executed or attested as a pass")
        else:
            program_sha = _sha(exact_program(item, original).encode("utf8"))
            owner = observed.get("exclusive_cgroup_owner")
            child_start = observed.get("independent_child_starttime_ticks")
            trace = observed.get("process_trace_sha256")
            if (task.get("program_sha256") != program_sha
                    or observed.get("program_sha256") != program_sha
                    or task.get("execution_status") not in ("PYTHON_EXIT", "PYTHON_TIMEOUT_10S")
                    or observed.get("execution_status") != task.get("execution_status")
                    or observed.get("observed_python_exit_code") != task.get("exit_code")
                    or observed.get("status") != "INDEPENDENTLY_OBSERVED_CHILD_EXTINCT"
                    or observed.get("python_argv_shape") != ["python", "-I", "candidate.py"]
                    or observed.get("python_major_minor") != "3.13"
                    or observed.get("python_executable_sha256") != capability["python_executable_sha256"]
                    or observed.get("python_isolated_mode") is not True
                    or observed.get("candidate_file_basename") != "candidate.py"
                    or observed.get("runtime_host_id") != host_id
                    or observed.get("sandbox_profile_sha256") != profile_sha
                    or observed.get("python_timeout_seconds") != 10
                    or observed.get("candidate_working_directory") != "temporary_cwd"
                    or observed.get("child_seen") is not True
                    or observed.get("cgroup_empty_after") is not True
                    or observed.get("descendants_extinct_after") is not True
                    or type(child_start) is not int or child_start <= 0
                    or not isinstance(owner, str) or not 8 <= len(owner) <= 256
                    or owner in owners
                    or not isinstance(trace, str)
                    or not re.fullmatch(r"[0-9a-f]{64}", trace)
                    or trace in traces):
                raise ValueError("independent per-task execution, source or descendant witness missing")
            trace_record = _bound(external_trace_raws[index], trace, 256 * 1024)
            if (trace_record.get("schema") != "chowder.reap-v2-independent-process-trace.v2"
                    or trace_record.get("index") != index
                    or trace_record.get("id") != ident
                    or trace_record.get("session_id") != SESSION
                    or trace_record.get("model_sha256") != Q8_SHA
                    or trace_record.get("program_sha256") != program_sha
                    or trace_record.get("runtime_host_id") != host_id
                    or trace_record.get("runtime_image_sha256") != image_sha
                    or trace_record.get("sandbox_profile_sha256") != profile_sha
                    or trace_record.get("exclusive_cgroup_owner") != owner
                    or trace_record.get("independent_child_starttime_ticks") != child_start
                    or trace_record.get("execution_status") != task["execution_status"]
                    or trace_record.get("observed_python_exit_code") != task.get("exit_code")
                    or trace_record.get("observer_outside_workload_namespaces") is not True
                    or trace_record.get("verified_no_egress") is not True
                    or trace_record.get("observed_cgroup_empty_after") is not True
                    or trace_record.get("observed_descendants_extinct_after") is not True
                    or type(trace_record.get("child_pid")) is not int
                    or trace_record["child_pid"] <= 0
                    or type(trace_record.get("start_monotonic_ns")) is not int
                    or trace_record["start_monotonic_ns"] <= 0
                    or type(trace_record.get("end_monotonic_ns")) is not int
                    or trace_record["end_monotonic_ns"] <= trace_record["start_monotonic_ns"]
                    or type(trace_record.get("elapsed_milliseconds")) is not int
                    or trace_record["elapsed_milliseconds"] <= 0
                    or trace_record["elapsed_milliseconds"] > 12000
                    or trace_record.get("timeout_kill_observed") is not (
                        task["execution_status"] == "PYTHON_TIMEOUT_10S")):
                raise ValueError("source-bound external runtime trace and descendant evidence missing")
            owners.add(owner)
            traces.add(trace)
            code = task.get("exit_code")
            if task["execution_status"] == "PYTHON_TIMEOUT_10S":
                if code is not None or task.get("independent_correct") is not False:
                    raise ValueError("timed-out Python was marked as passed")
            elif (type(code) is not int or not 0 <= code <= 124
                  or task.get("independent_correct") is not (code == 0)):
                raise ValueError("Python exit code was spoofed or infrastructure failure hidden")
            executed += 1
        matched += task["independent_correct"] == original["correct"]
        correct += task["independent_correct"] is True
    if (executed != 162 or replay.get("actually_executed_programs") != 162
            or replay.get("matched_recorded_scores") != matched
            or replay.get("correct") != correct):
        raise ValueError("full original Q8 population/witness aggregates missing")
    return {"status": ("RESEARCH_ONLY_ORIGINAL_Q8_REPLAY_MATCHED" if matched == 164
                       and correct == 152 else "RESEARCH_ONLY_REPLAY_DISAGREEMENT"),
            "source_rows": 164, "actually_executed_programs": executed,
            "matched_recorded_scores": matched, "correct": correct,
            "original_windows_gate_eligible": False, "gen0_status": "BLOCKED",
            "chosen": None, "release_approved": False}
