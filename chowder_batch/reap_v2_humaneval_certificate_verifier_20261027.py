"""Independent, CPU-only frozen HumanEval164 certificate/trace verifier.

This is a *second*, fail-closed verification layer after the native capture,
approved isolation broker and outside-host witness. Its approved trace-review
SHA pins are intentionally NOT SET: historical Q4/Q8 research cannot become
frozen Gate2 by relabeling a JSON file, and the verifier cannot approve its
own external security/runtime. No subprocess, generated Python, GPU, Podman,
model loading, artifact modification or network calls occur in this module.

Important distinctions:
* A filename, an outcome dict, or a SHA printed inside the outcome is NOT an
  independent execution receipt; the entire review and every trace must be
  hash-bound to source-owned independent approval.
* The existing build_selector_rows() trusts an in-memory grade. This layer
  obtains its grade by RE-READING full captured source, real independent
  witness, and signed-off external traces, not accepting caller grade values.
* Original §4 Gate2 uses Q8-vs-dense on SAME HumanEval/0..163 items,
  sample-SD paired Wald lower bound > -0.10, NOT Q8-vs-Q4 research, not a
  McNemar equality p-value or Amendment4's unresolved -0.02 methodology.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from chowder_batch import reap_gates
from chowder_batch import reap_v2_original_humaneval_frozen_executor_20261027 as original

SCHEMA = "chowder.reap-v2-human164-independent-certificate-audit.v1"
REVIEW_SCHEMA = "chowder.reap-v2-independent-full164-trace-review.v1"
TRACE_SCHEMA = "chowder.reap-v2-original-human164-outside-sandbox-process-trace.v1"
SOURCE_HUMANEVAL164_SHA256 = "49fb688c6788b0531803dfd1cb85a69926ec5a7179e74cdfffc53ba52a92b420"
CANONICAL_RAW_SHA256 = "5144d31cf337b25ce262b780ed103b128b1e9d289d3c0d15f4fe34933bb705d5"
MODEL_SHA256 = {
    "reap128_v2": "89dcaf33b61f59c83a31b718ba98b73c33437d0320ae0cc56737a5d7e43519c6",
    "dense_v2": "2ff57a74a36939c5baabf40a0d3eb06d4424ecc94cf7d4ef04b3ac02059097c2",
}
EXPECTED_COUNT = 164

# Prospective independently reviewed original runtime trace-manifest SHA256s
# MUST be peer-reviewed and source-pinned BEFORE use. Do not accept CLI pins,
# environment toggles, caller-supplied verification status or research scores.
APPROVED_INDEPENDENT_TRACE_REVIEW_SHA256: dict[str, str] = {}


class CertificateRefusal(ValueError):
    """Frozen case trace, source, sandbox or statistical identity is not proven."""


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise CertificateRefusal(reason)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hex(value: object, size: int = 64) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{%d}" % size, value) is not None


def _unique(pairs: list[tuple[str, object]]) -> dict:
    out = {}
    for key, value in pairs:
        if key in out:
            raise CertificateRefusal("duplicate independent certificate JSON field")
        out[key] = value
    return out


def _json(raw: bytes, *, max_bytes: int = 4 * 1024 * 1024) -> dict:
    _require(isinstance(raw, bytes) and 0 < len(raw) <= max_bytes,
             "missing or unbounded certificate bytes")
    try:
        result = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique,
                            parse_constant=lambda _: (_ for _ in ()).throw(
                                CertificateRefusal("nonfinite independent receipt JSON")))
    except (UnicodeError, ValueError) as error:
        raise CertificateRefusal("invalid UTF-8 original evidence JSON") from error
    _require(isinstance(result, dict), "certificate must contain a JSON object")
    return result


def _file(path: Path, *, limit: int) -> bytes:
    _require(isinstance(path, Path) and path.is_absolute()
             and path.is_file() and not path.is_symlink()
             and 0 < path.stat().st_size <= limit,
             "independent original evidence file missing, linked or oversized")
    raw = path.read_bytes()
    _require(0 < len(raw) <= limit, "original evidence changed or exceeded size cap")
    return raw


def approval_gaps() -> list[str]:
    """Read-only: existing original security pins AND new external reviewer pins."""
    gaps = list(original.execution_admission_gaps())
    for arm in MODEL_SHA256:
        if not _hex(APPROVED_INDEPENDENT_TRACE_REVIEW_SHA256.get(arm)):
            gaps.append(f"INDEPENDENT_{arm.upper()}_164_RUNTIME_TRACE_REVIEW_SHA_REQUIRED")
    return gaps


def _capture(path: Path, arm: str, tasks: list[dict]) -> tuple[list[dict], bytes]:
    raw = _file(path, limit=32 * 1024 * 1024)
    _require(raw.endswith(b"\n"), "incomplete original native capture journal")
    _require(_hex(original.APPROVED_NATIVE_CAPTURE_COHORT_SHA256.get(arm))
             and _sha(raw) == original.APPROVED_NATIVE_CAPTURE_COHORT_SHA256[arm],
             "original full164 native capture SHA not independently source-approved")
    lines = raw.splitlines()
    _require(len(lines) == EXPECTED_COUNT, "native captured cohort not exactly 164")
    rows = [_json(line, max_bytes=original.MAX_BYTES + 1024) for line in lines]
    # Original writer requires exact canonical separators, order and UTF-8.
    _require(original._serialized_capture(rows) == raw,
             "native capture bytes are not exact original sealed writer format")
    _require({r.get("arm") for r in rows} == {arm}
             and {r.get("session_id") for r in rows}
             == {rows[0].get("session_id")}
             and {r.get("physical_host_sha256") for r in rows}
             == {rows[0].get("physical_host_sha256")},
             "cross-model, cross-session or cross-host original captures pooled")
    _require(_hex(rows[0]["session_id"], 32)
             and _hex(rows[0]["physical_host_sha256"])
             and {r.get("native_owner_review_sha256") for r in rows}
             == {original.APPROVED_NATIVE_OWNER_REVIEW_SHA256.get(arm)},
             "native Windows owner review was not independently source-pinned")
    for index, row in enumerate(rows):
        original._candidate_packet(row, tasks)
        _require(row.get("ordinal") == index and row.get("id") == f"HumanEval/{index}"
                 and row.get("model_sha256") == MODEL_SHA256[arm],
                 "original capture's registered case/model identity invalid")
    return rows, raw


def _review(review_path: Path, arm: str, capture_sha: str,
            witness_sha: str, capability: dict,
            session: str) -> dict:
    _require(_hex(APPROVED_INDEPENDENT_TRACE_REVIEW_SHA256.get(arm)),
             "outside-host original 164 trace review not independently approved")
    raw = _file(review_path, limit=256 * 1024)
    _require(_sha(raw) == APPROVED_INDEPENDENT_TRACE_REVIEW_SHA256[arm],
             "outside-host original 164 trace-review approval SHA mismatch")
    review = _json(raw, max_bytes=256 * 1024)
    _require(review.get("schema") == REVIEW_SCHEMA
             and review.get("status") == "INDEPENDENTLY_SOURCE_REVIEWED_FROZEN_164_TRACES"
             and review.get("arm") == arm
             and review.get("model_sha256") == MODEL_SHA256[arm]
             and review.get("session_id") == session
             and review.get("canonical_source_sha256") == CANONICAL_RAW_SHA256
             and review.get("frozen_source_items_sha256") == SOURCE_HUMANEVAL164_SHA256
             and review.get("capture_sha256") == capture_sha
             and review.get("witness_sha256") == witness_sha
             and review.get("sandbox_capability_sha256")
                 == original.APPROVED_SANDBOX_CAPABILITY_SHA256
             and review.get("sandbox_profile_sha256") == capability["sandbox_profile_sha256"]
             and review.get("runtime_host_sha256") == capability["runtime_host_sha256"]
             and review.get("python_executable_sha256") == capability["python_executable_sha256"]
             and review.get("observer_outside_workload_namespaces") is True
             and review.get("observer_independently_reviewed") is True
             and _hex(review.get("reviewed_observer_source_sha256"))
             and isinstance(review.get("case_trace_sha256"), list)
             and len(review["case_trace_sha256"]) == EXPECTED_COUNT,
             "independent original runtime-trace review source, session or population invalid")
    return review


def _verify_trace(raw: bytes, case: dict, proof: dict, cap: dict,
                  index: int, *, arm: str, reviewed_observer_sha: str,
                  capture_sha: str, seen_owners: set[str],
                  seen_processes: set[tuple[str, int, int]],
                  seen_trace_hashes: set[str],
                  seen_cgroups: set[str]) -> None:
    trace = _json(raw, max_bytes=256 * 1024)
    source_sha = case["candidate_program_sha256"]
    expected = {
        "schema": TRACE_SCHEMA,
        "arm": arm,
        "model_sha256": MODEL_SHA256[arm],
        "session_id": case["session_id"],
        "ordinal": index,
        "id": case["id"],
        "capture_sha256": capture_sha,
        "program_sha256": source_sha,
        "sandbox_capability_sha256": original.APPROVED_SANDBOX_CAPABILITY_SHA256,
        "sandbox_profile_sha256": cap["sandbox_profile_sha256"],
        "runtime_host_sha256": cap["runtime_host_sha256"],
        "python_executable_sha256": cap["python_executable_sha256"],
        "reviewed_observer_source_sha256": reviewed_observer_sha,
        "sandbox_owner": proof["sandbox_owner"],
        "child_pid": proof["child_pid"],
        "execution_status": proof["status"],
        "exit_code": proof["exit_code"],
        "correct": proof["correct"],
        "elapsed_milliseconds": proof["elapsed_milliseconds"],
        "timeout_kill_observed": proof["timeout_kill_observed"],
    }
    _require(all(type(trace.get(k)) is type(v) and trace[k] == v
                 for k, v in expected.items()), "independent per-case trace fields disagree")
    _require(trace.get("python_version") == "3.13"
             and trace.get("python_argv_shape") == ["python", "-I", "candidate.py"]
             and trace.get("isolated_nonroot_uid") is True
             and trace.get("isolated_user_pid_mount_net_namespaces") is True
             and trace.get("no_network_egress") is True
             and trace.get("no_host_mounts_or_secrets") is True
             and trace.get("source_staged_before_executable_release") is True
             and trace.get("source_file_sha256") == source_sha
             and trace.get("sealed_memfd_source_sha256") == source_sha
             and trace.get("sealed_memfd_immutable") is True
             and trace.get("memory_and_pid_cgroup_caps_enforced") is True
             and trace.get("observed_cgroup_empty_after") is True
             and trace.get("all_descendants_extinct_after") is True
             and trace.get("outside_host_process_observer") is True,
             "independent original isolation/immutable source evidence absent")
    start = trace.get("child_start_ticks")
    first = trace.get("preexecution_source_observed_ns")
    released = trace.get("execution_release_ns")
    end = trace.get("execution_finished_ns")
    _require(all(type(t) is int and t > 0 for t in (start, first, released, end))
             and first <= released < end
             and abs((end - released) / 1_000_000 - trace["elapsed_milliseconds"]) <= 2
             and type(trace.get("actual_cgroup_path")) is str
             and 8 <= len(trace["actual_cgroup_path"]) <= 512
             and trace["actual_cgroup_path"].startswith("/"),
             "process start-time/ordering or exclusive cgroup witness missing")
    process = (cap["runtime_host_sha256"], proof["child_pid"], start)
    owner = trace["sandbox_owner"]
    digest = _sha(raw)
    cgroup = trace["actual_cgroup_path"]
    _require(process not in seen_processes and owner not in seen_owners
             and digest not in seen_trace_hashes and cgroup not in seen_cgroups,
             "original process/source witness replayed or reused")
    seen_processes.add(process)
    seen_owners.add(owner)
    seen_trace_hashes.add(digest)
    seen_cgroups.add(cgroup)
    _require(trace["elapsed_milliseconds"] <= 12000
             and (trace["execution_status"] != "PYTHON_TIMEOUT_10S"
                  or (trace["exit_code"] is None and trace["correct"] is False)),
             "original 10-second timeout or nonpass rule broken")


def verify_arm(
    *, arm: str, capture_path: Path, witness_path: Path, capability_path: Path,
    broker_path: Path, review_path: Path, trace_dir: Path,
    serve_path: Path, owner_review_path: Path,
    canonical_path: Path | None = None,
) -> dict:
    """Independently re-audit every original case/trace before accepting grade.

    This API NEVER accepts an externally passed in-memory grade. Only the
    actual captured cohort, reviewed security capability, independently pinned
    outside-host witness AND 164-item source-approved trace review can enter.
    Only terminated responses require a real executed trace; unterminated
    original responses are strict nonpasses with no process.
    """
    _require(arm in MODEL_SHA256, "frozen Gate2 admits Q8 candidate or Q8 dense only")
    _require(not approval_gaps(), "original HumanEval sandbox/owner/trace security approval missing")
    _require(SOURCE_HUMANEVAL164_SHA256 == original.FROZEN_SOURCE_ITEMS_SHA256
             and CANONICAL_RAW_SHA256 == original.FROZEN_CANONICAL_SHA256
             and MODEL_SHA256 == original.MODELS,
             "original source/model hardcoded preregistration drift")
    cap = original._capability(Path(capability_path), Path(broker_path))
    tasks = original.canonical_tasks(
        Path(canonical_path) if canonical_path is not None else original.archived.CANONICAL)
    cases, capture_raw = _capture(Path(capture_path), arm, tasks)
    serving = original._owner_review(arm, Path(serve_path), Path(owner_review_path))
    _require(serving["session_id"] == cases[0]["session_id"]
             and serving["model_sha256"] == MODEL_SHA256[arm]
             and serving["physical_host_sha256"] == cases[0]["physical_host_sha256"]
             and serving["native_owner_review_sha256"] == cases[0]["native_owner_review_sha256"],
             "original native four-slot serving receipt or Windows owner differs from capture")
    witness_raw = _file(Path(witness_path), limit=4 * 1024 * 1024)
    _require(_sha(witness_raw) == original.APPROVED_OUTSIDE_HOST_WITNESS_SHA256.get(arm),
             "independent original full164 outside-host witness SHA mismatch")
    witness = _json(witness_raw)
    _require(witness.get("schema") == original.WITNESS_SCHEMA
             and witness.get("arm") == arm
             and witness.get("model_sha256") == MODEL_SHA256[arm]
             and witness.get("session_id") == cases[0]["session_id"]
             and isinstance(witness.get("per_case"), list)
             and len(witness["per_case"]) == EXPECTED_COUNT,
             "original 164-case witness source/model/session mismatch")
    review = _review(Path(review_path), arm, _sha(capture_raw), _sha(witness_raw),
                     cap, cases[0]["session_id"])
    tdir = Path(trace_dir)
    _require(tdir.is_absolute() and tdir.is_dir() and not tdir.is_symlink(),
             "independent per-case trace directory absent")
    seen_owners: set[str] = set()
    seen_processes: set[tuple[str, int, int]] = set()
    seen_traces: set[str] = set()
    seen_cgroups: set[str] = set()
    for i, (case, proof, expected_sha) in enumerate(zip(
        cases, witness["per_case"], review["case_trace_sha256"],
    )):
        _require(type(proof) is dict
                 and proof.get("ordinal") == i and proof.get("id") == case["id"]
                 and proof.get("session_id") == case["session_id"]
                 and proof.get("candidate_program_sha256") == case["candidate_program_sha256"],
                 "independent original witness does not pair with item/program")
        trace_path = tdir / f"{i:03d}.json"
        if not case["finished"]:
            _require(expected_sha is None and not trace_path.exists()
                     and proof.get("status") == "NOT_EXECUTED_UNTERMINATED"
                     and proof.get("correct") is False
                     and proof.get("child_pid") is None
                     and proof.get("sandbox_owner") is None,
                     "unterminated original item cannot execute or have a process trace")
        else:
            _require(_hex(expected_sha), "independently pinned per-case trace SHA missing")
            raw = _file(trace_path, limit=256 * 1024)
            _require(_sha(raw) == expected_sha
                     and proof.get("independent_trace_sha256") == expected_sha,
                     "full164 original review/witness per-case trace SHA mismatch")
            _verify_trace(raw, case, proof, cap, i, arm=arm,
                          reviewed_observer_sha=review["reviewed_observer_source_sha256"],
                          capture_sha=_sha(capture_raw),
                          seen_owners=seen_owners, seen_processes=seen_processes,
                          seen_trace_hashes=seen_traces, seen_cgroups=seen_cgroups)
    # At this point the second independent provenance audit is complete; use
    # the pre-existing scoring verifier and then check its output independently.
    grade = original.score_frozen_164(
        cases, Path(witness_path), arm=arm,
        capability_path=Path(capability_path), broker_path=Path(broker_path),
        canonical=Path(canonical_path) if canonical_path is not None
        else original.archived.CANONICAL,
    )
    expected_correct = sum(proof["correct"] for proof in witness["per_case"])
    _require(grade.get("correct") == expected_correct
             and grade.get("evaluated_original_rows") == EXPECTED_COUNT
             and grade.get("release_approved") is False
             and [r.get("correct") for r in grade.get("outcomes", [])]
                 == [p["correct"] for p in witness["per_case"]],
             "previous frozen scorer outcomes disagree with independent original traces")
    prepared = original.build_selector_rows(
        cases, grade, arm=arm,
        canonical=Path(canonical_path) if canonical_path is not None
        else original.archived.CANONICAL,
    )
    _require(prepared.get("row_count") == EXPECTED_COUNT
             and _sha(prepared["jsonl_bytes"]) == prepared["summary_entry"]["file_sha256"],
             "original selector export differs from independently audited full164 results")
    return {
        "schema": SCHEMA, "status": "INDEPENDENT_164_CERTIFICATE_RECOMPUTED_NOT_RELEASE",
        "arm": arm, "model_sha256": MODEL_SHA256[arm],
        "session_id": cases[0]["session_id"],
        "physical_host_sha256": cases[0]["physical_host_sha256"],
        "canonical_source_sha256": CANONICAL_RAW_SHA256,
        "frozen_source_items_sha256": SOURCE_HUMANEVAL164_SHA256,
        "native_capture_sha256": _sha(capture_raw),
        "outside_host_witness_sha256": _sha(witness_raw),
        "independent_trace_review_sha256": _sha(_file(Path(review_path), limit=256 * 1024)),
        "sandbox_profile_sha256": cap["sandbox_profile_sha256"],
        "runtime_host_sha256": cap["runtime_host_sha256"],
        "python_executable_sha256": cap["python_executable_sha256"],
        "executed_with_verified_trace": len(seen_traces),
        # Internal independently observed process identities. Two distinct
        # model-serving arms must not credit the same Python execution twice.
        "verified_process_identities": tuple(sorted(seen_processes)),
        "nonterminated_original_nonpasses": EXPECTED_COUNT - len(seen_traces),
        "correct": expected_correct, "population": EXPECTED_COUNT,
        "outcomes": tuple((row["id"], row["correct"]) for row in grade["outcomes"]),
        # Keep the independently source-bound original generation termination
        # semantics available to the official frozen intake. Never equate a
        # forged selector row's `terminated` field with independent capture.
        "termination_outcomes": tuple((case["id"], case["finished"]) for case in cases),
        # An equal correctness/termination bit does not prove the original
        # evaluator scored THIS native response. The admission bridge must
        # independently bind every original row's raw text to captured bytes.
        "native_response_sha256_outcomes": tuple(
            (case["id"], case["native_response_sha256"]) for case in cases
        ),
        "native_health_outcomes": tuple(
            (case["id"], case["trigram_ratio"], case["completion_tokens"],
             round(case["seconds"], 2), case["predicted_per_second"])
            for case in cases
        ),
        "selector_rows_sha256": prepared["summary_entry"]["file_sha256"],
        "gate2": "UNKNOWN_UNTIL_PAIRED_SOURCE_BOUND_DENSE_AND_FROZEN_SELECTOR",
        "chosen": None, "release_approved": False,
    }


def verify_disjoint_execution_lineage(candidate: dict, dense: dict) -> None:
    """Reject reusing one actual sandbox process across distinct original arms.

    PID alone can legitimately recur after exit; the host, PID and kernel
    process-start identity together identify an execution. Cgroup paths may
    also be reused sequentially, so they are not used as a cross-arm key.
    """
    identities = []
    for name, certificate in (("candidate", candidate), ("dense", dense)):
        _require(isinstance(certificate, dict), f"{name}: original certificate absent")
        count = certificate.get("executed_with_verified_trace")
        processes = certificate.get("verified_process_identities")
        _require(type(count) is int and 0 <= count <= EXPECTED_COUNT
                 and type(processes) in (tuple, list) and len(processes) == count,
                 f"{name}: original verified process inventory incomplete")
        normalized = []
        for process in processes:
            _require(type(process) in (tuple, list) and len(process) == 3
                     and _hex(process[0]) and type(process[1]) is int
                     and process[1] > 0 and type(process[2]) is int
                     and process[2] > 0,
                     f"{name}: original process identity invalid")
            normalized.append(tuple(process))
        _require(len(set(normalized)) == count,
                 f"{name}: original process identity repeated within arm")
        identities.append(set(normalized))
    _require(not identities[0].intersection(identities[1]),
             "original Q8 and dense HumanEval reuse one executed Python process")


def verify_paired_gate2(*, candidate: dict[str, Any], dense: dict[str, Any]) -> dict:
    """Reverify both full original arms from disk and compute §4 paired Wald.

    The keyword dicts are paths/kwargs for verify_arm(), NOT certificates or
    caller-supplied correctness values. Synthetic or self-claimed grades cannot
    drive this comparison. Returns diagnostic PASS/FAIL but NEVER selection.
    """
    _require(isinstance(candidate, dict) and isinstance(dense, dict),
             "paired original evidence paths not supplied")
    arm_result = verify_arm(arm="reap128_v2", **candidate)
    dense_result = verify_arm(arm="dense_v2", **dense)
    verify_disjoint_execution_lineage(arm_result, dense_result)
    _require(arm_result["physical_host_sha256"] == dense_result["physical_host_sha256"],
             "original Q8 and dense comparisons used different physical Windows hosts")
    _require(arm_result["session_id"] != dense_result["session_id"],
             "two different original Q8 model weights cannot share one serving session")
    _require(arm_result["frozen_source_items_sha256"]
             == dense_result["frozen_source_items_sha256"]
             == SOURCE_HUMANEVAL164_SHA256,
             "paired Gate2 original source population mismatch")
    a = arm_result["outcomes"]
    b = dense_result["outcomes"]
    _require(len(a) == len(b) == EXPECTED_COUNT
             and all(a[i][0] == b[i][0] == f"HumanEval/{i}"
                     and type(a[i][1]) is bool and type(b[i][1]) is bool
                     for i in range(EXPECTED_COUNT)),
             "paired HumanEval item keys or binary correctness mismatch")
    differences = [int(a[i][1]) - int(b[i][1]) for i in range(EXPECTED_COUNT)]
    bound = reap_gates.lower_bound(differences)  # original sample-SD 1.645 Wald
    _require(bound is not None and math.isfinite(bound),
             "frozen paired sample-SD Wald lower bound invalid")
    return {
        "schema": SCHEMA, "status": "ORIGINAL_GATE2_PAIRED_DIAGNOSTIC_NOT_SELECTION",
        "suite": "humaneval", "paired_count": EXPECTED_COUNT,
        "candidate_model_sha256": MODEL_SHA256["reap128_v2"],
        "dense_model_sha256": MODEL_SHA256["dense_v2"],
        "source_sha256": SOURCE_HUMANEVAL164_SHA256,
        "candidate_correct": arm_result["correct"], "dense_correct": dense_result["correct"],
        "paired_mean_difference": sum(differences) / EXPECTED_COUNT,
        "one_sided_sample_sd_wald_95_lower_bound": bound,
        "preregistered_margin": -0.10, "strict_frozen_threshold_met": bound > -0.10,
        "gate2": "PASS" if bound > -0.10 else "FAIL",
        "frozen_selector_called": False, "chosen": None, "release_approved": False,
        "amendment4_optional_comparator": "UNRESOLVED_DISTINCT_PAIRED_MC_NEMAR_METHOD",
    }
