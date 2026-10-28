"""Original 164-item native HumanEval capture and independently isolated handoff.

The frozen evaluator's ``_humaneval_pass`` executes arbitrary generated Python
locally. Never invoke it here. Native generation is captured as unscored data;
the exact original candidate program is sent over stdin ONLY to an independently
source-approved sandbox broker, whose outside-host witness is required before
the resulting 164 outcomes may enter the frozen selector.

No trusted broker, execution-capability or outside-host witness source is
approved yet. All production execution/scoring therefore refuses. Existing
Q4/Q8 164-case Podman/memfd research is expressly NOT frozen Gate2 evidence.
This module does not start model servers, Podman, WSL, a shell or generated code.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess

from chowder_batch import reap_eval, reap_gates, reap_select
from chowder_batch import reap_humaneval_v2_attestation as security
from chowder_batch import reap_humaneval_v38_actual_execution_20261027 as archived
from chowder_batch.reap_external_full_audit import EVALUATOR_SHA, SCORER_SHA
from chowder.evaluators.scoring import final_answer

SCHEMA = "chowder.reap-v2-original-native-human164-handoff.v1"
CAPTURE_SCHEMA = "chowder.reap-v2-frozen-human164-unscored-capture.v1"
SESSION_SCHEMA = "chowder.reap-v2-independent-windows-human164-native-owner.v1"
CAPABILITY_SCHEMA = "chowder.reap-v2-independent-human164-sandbox-capability.v1"
RESULT_SCHEMA = "chowder.reap-v2-independent-human164-case-result.v1"
WITNESS_SCHEMA = "chowder.reap-v2-independent-human164-outside-host-witness.v1"

POPULATION = 164
MAX_BYTES = 2 * 1024 * 1024
MAX_JSON = 256 * 1024
FROZEN_CANONICAL_SHA256 = archived.CANONICAL_SHA256
FROZEN_SOURCE_ITEMS_SHA256 = reap_gates.FROZEN_SOURCE_SHA256["humaneval"]
MODELS = {
    "reap128_v2": "89dcaf33b61f59c83a31b718ba98b73c33437d0320ae0cc56737a5d7e43519c6",
    "dense_v2": reap_select.FROZEN_REFERENCE_MODEL_SHA256["dense_v2"],
}

# Independently reviewed code-owned SHA pins, added prospectively in an
# authorized review. The caller CANNOT supply or override these at runtime.
APPROVED_NATIVE_OWNER_REVIEW_SHA256: dict[str, str] = {}
APPROVED_NATIVE_CAPTURE_COHORT_SHA256: dict[str, str] = {}
APPROVED_SANDBOX_CAPABILITY_SHA256: str | None = None
APPROVED_SANDBOX_BROKER_SHA256: str | None = None
APPROVED_OUTSIDE_HOST_WITNESS_SHA256: dict[str, str] = {}


class FrozenExecutionRefusal(ValueError):
    """An original source, security, runtime or attestation boundary failed."""


def _need(ok: bool, message: str) -> None:
    if not ok:
        raise FrozenExecutionRefusal(message)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash(value: object, length: int = 64) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{" + str(length) + "}", value) is not None


def _read(path: Path, limit: int = MAX_BYTES) -> bytes:
    _need(path.is_absolute() and path.is_file() and not path.is_symlink(),
          "original input must be an absolute ordinary file")
    _need(0 < path.stat().st_size <= limit, "original input byte limit violated")
    raw = path.read_bytes()
    _need(0 < len(raw) <= limit, "original input changed or exceeded bound")
    return raw


def _json(raw: bytes) -> dict:
    _need(len(raw) <= MAX_JSON, "unbounded evidence JSON")
    try:
        def unique(pairs: list[tuple[str, object]]) -> dict:
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate JSON key")
                result[key] = value
            return result

        value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError("NaN/Infinity")))
    except (ValueError, UnicodeError) as exc:
        raise FrozenExecutionRefusal("invalid or ambiguous independent evidence JSON") from exc
    _need(isinstance(value, dict), "independent evidence must be an object")
    return value


def _source_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    _need(_sha(_read(root / "chowder_batch/reap_eval.py")) == EVALUATOR_SHA
          and _sha(_read(root / "src/chowder/evaluators/scoring.py")) == SCORER_SHA,
          "original frozen evaluator or production scorer source changed")
    _need(FROZEN_SOURCE_ITEMS_SHA256 == reap_eval.FROZEN_V2_QUALITY_SOURCES["humaneval"][1]
          and reap_eval.MAX_TOKENS["humaneval"] == 768
          and reap_eval.SYSTEM == "You are a helpful assistant.",
          "original HumanEval scoring/decoding source drift")


def canonical_tasks(path: Path = archived.CANONICAL) -> list[dict]:
    """Read the exact original ordered prompts, tests and entrypoints, no code execution."""
    _source_contract()
    raw = _read(Path(path))
    _need(_sha(raw) == FROZEN_CANONICAL_SHA256, "official HumanEval canonical raw SHA mismatch")
    try:
        tasks = json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite")))
    except (ValueError, UnicodeError) as exc:
        raise FrozenExecutionRefusal("canonical original HumanEval JSON invalid") from exc
    _need(isinstance(tasks, list) and len(tasks) == POPULATION, "original HumanEval 164 missing")
    for i, task in enumerate(tasks):
        _need(isinstance(task, dict) and set(task) == {"id", "prompt", "entry_point", "test", "header"}
              and task.get("id") == f"HumanEval/{i}"
              and all(isinstance(task[key], str) for key in ("prompt", "entry_point", "test", "header"))
              and 0 < len(task["test"]) <= 64000
              and len(task["prompt"]) <= 64000,
              f"canonical original HumanEval item {i} invalid")
    _need(reap_eval._digest(tasks) == FROZEN_SOURCE_ITEMS_SHA256,
          "original 164 prompted items do not match registered frozen source SHA")
    return tasks


def frozen_request(ordinal: int, *, canonical: Path = archived.CANONICAL) -> dict:
    """Exact original native chat request for one predeclared HumanEval item."""
    _need(type(ordinal) is int and 0 <= ordinal < POPULATION, "unregistered HumanEval ordinal")
    item = canonical_tasks(canonical)[ordinal]
    return {"schema": SCHEMA, "id": item["id"], "source_items_sha256": FROZEN_SOURCE_ITEMS_SHA256,
            "request": {
                "messages": [{"role": "system", "content": reap_eval.SYSTEM},
                             {"role": "user", "content": item["prompt"]}],
                "temperature": 0.0, "top_k": 1, "seed": 7, "max_tokens": 768,
                "chat_template_kwargs": {"enable_thinking": False},
            }}


def _owner_review(arm: str, serve_path: Path, review_path: Path) -> dict:
    """Source-approved ORIGINAL Windows native owner/session; caller claims don't approve."""
    _need(arm in MODELS, "frozen Gate2 requires Q8 candidate or original dense, never Q4")
    _need(_hash(APPROVED_NATIVE_OWNER_REVIEW_SHA256.get(arm)),
          "independent native original serving owner approval missing")
    review_raw = _read(review_path, MAX_JSON)
    _need(_sha(review_raw) == APPROVED_NATIVE_OWNER_REVIEW_SHA256[arm],
          "original independently reviewed Windows owner SHA mismatch")
    review = _json(review_raw)
    serve_raw = _read(serve_path, MAX_JSON)
    serve = _json(serve_raw)
    session = serve.get("eval_session_id")
    _need(review.get("schema") == SESSION_SCHEMA
          and review.get("status") == "INDEPENDENTLY_VERIFIED_NATIVE_WINDOWS_FROZEN_OWNER"
          and review.get("arm") == arm and review.get("model_sha256") == MODELS[arm]
          and review.get("session_id") == session and _hash(session, 32)
          and review.get("serve_sha256") == _sha(serve_raw)
          and review.get("physical_windows_owner_verified") is True
          and review.get("real_original_model_bytes_verified") is True
          and review.get("per_case_process_binding_verified") is True
          and review.get("serving_geometry_verified") is True
          and review.get("uninterrupted_original_164_run") is True
          and _hash(review.get("physical_host_sha256"))
          and _hash(review.get("native_server_source_sha256"))
          and serve.get("gguf_sha256") == MODELS[arm]
          and serve.get("flags") == reap_select.FROZEN_SERVE_FLAGS
          and type(serve.get("server_pid")) is int and serve["server_pid"] > 0
          and type(serve.get("server_port")) is int and 1 <= serve["server_port"] <= 65535,
          "original Q8 model/Windows owner/four-slot session unverified")
    return {"arm": arm, "model_sha256": MODELS[arm], "session_id": session,
            "native_owner_review_sha256": _sha(review_raw),
            "physical_host_sha256": review["physical_host_sha256"]}


def _original_program(item: dict, text: str) -> bytes:
    """Byte-identical source composition from original _humaneval_pass, NO exec."""
    reply = final_answer(text)
    blocks = re.findall(r"```(?:python)?\n(.*?)```", reply, flags=re.S)
    code = blocks[-1] if blocks else reply
    if f"def {item['entry_point']}" not in code:
        code = item["header"] + code
    return (code + "\n\n" + item["test"] + f"\n\ncheck({item['entry_point']})\n").encode("utf-8")


def capture_case(arm: str, ordinal: int, response: dict, *, serve_path: Path,
                 owner_review_path: Path, canonical: Path = archived.CANONICAL) -> dict:
    """Capture original native generation as unscored data. Never execute generated code."""
    owner = _owner_review(arm, Path(serve_path), Path(owner_review_path))
    item = canonical_tasks(canonical)[ordinal] if type(ordinal) is int and 0 <= ordinal < POPULATION else None
    _need(item is not None, "unregistered HumanEval ordinal")
    _need(isinstance(response, dict) and set(response) >= {
        "text", "finish_reason", "completion_tokens", "seconds", "timings"},
        "original native generation response incomplete")
    text = response["text"]
    completion = response["completion_tokens"]
    duration = response["seconds"]
    finished = response["finish_reason"]
    _need(isinstance(text, str) and len(text.encode("utf-8")) <= MAX_BYTES
          and finished in ("stop", "length")
          and type(completion) is int and 0 <= completion <= 768
          and type(duration) in (int, float) and math.isfinite(duration) and duration >= 0,
          "invalid original native generation or maximum-token status")
    _need(isinstance(response["timings"], (dict, type(None))), "invalid native timing values")
    pps = (response["timings"] or {}).get("predicted_per_second")
    _need(pps is None or (type(pps) in (int, float) and math.isfinite(pps) and pps >= 0),
          "invalid original native token-rate witness")
    program_raw = _original_program(item, text) if finished == "stop" else None
    _need(program_raw is None or 0 < len(program_raw) <= MAX_BYTES,
          "rendered original HumanEval candidate exceeds source cap")
    return {"schema": CAPTURE_SCHEMA, "ordinal": ordinal, "id": item["id"],
            **owner, "source_items_sha256": FROZEN_SOURCE_ITEMS_SHA256,
            "canonical_sha256": FROZEN_CANONICAL_SHA256,
            "evaluator_sha256": EVALUATOR_SHA, "scorer_sha256": SCORER_SHA,
            "finished": finished == "stop", "completion_tokens": completion,
            "seconds": duration, "predicted_per_second": pps,
            "trigram_ratio": round(reap_eval._trigram_ratio(text), 4),
            "native_generation_text": text,
            "native_response_sha256": _sha(text.encode("utf-8")),
            "candidate_program_sha256": _sha(program_raw) if program_raw else None,
            "frozen_correct": None, "generated_code_executed": False}


def seal_capture(cases: list[dict], destination: Path, *, arm: str,
                 canonical: Path = archived.CANONICAL) -> dict:
    """Exclusive, fsynced source-owned capture; no model Python is run or scored."""
    _need(arm in MODELS and isinstance(cases, list) and len(cases) == POPULATION,
          "original sealed capture must have all 164 Q8/dense cases")
    _need([c.get("id") for c in cases if isinstance(c, dict)]
          == [f"HumanEval/{i}" for i in range(POPULATION)],
          "missing, reordered or duplicate frozen HumanEval case")
    first = cases[0]
    tasks = canonical_tasks(Path(canonical))
    lineage = ("model_sha256", "session_id", "native_owner_review_sha256",
               "source_items_sha256", "physical_host_sha256")
    for index, case in enumerate(cases):
        _candidate_packet(case, tasks)
        _need(case.get("schema") == CAPTURE_SCHEMA and case.get("ordinal") == index
              and case.get("arm") == arm and case.get("frozen_correct") is None
              and case.get("generated_code_executed") is False
              and all(case.get(k) == first.get(k) for k in lineage)
              and case.get("model_sha256") == MODELS[arm]
              and case.get("source_items_sha256") == FROZEN_SOURCE_ITEMS_SHA256
              and case.get("canonical_sha256") == FROZEN_CANONICAL_SHA256
              and case.get("scorer_sha256") == SCORER_SHA
              and case.get("evaluator_sha256") == EVALUATOR_SHA
              and _sha(case.get("native_generation_text", "").encode("utf-8"))
                  == case.get("native_response_sha256"),
              "frozen capture source/model/session/case invariant invalid")
    path = Path(destination)
    _need(path.is_absolute() and path.parent.is_dir() and not path.exists()
          and not path.is_symlink(), "capture output requires a new exclusive file")
    payload = _serialized_capture(cases)
    _need(len(payload) <= 32 * 1024 * 1024, "unbounded captured native cohort payload")
    # No overwrite or "repair" of concurrent original receipts; partial
    # failures retain a visibly incomplete exclusive file for independent review.
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return {"schema": CAPTURE_SCHEMA, "status": "CAPTURED_164_UNSCORED_ONLY",
            "model_sha256": MODELS[arm], "session_id": first["session_id"],
            "source_items_sha256": FROZEN_SOURCE_ITEMS_SHA256,
            "captured_rows_sha256": _sha(payload), "captured_rows": POPULATION,
            "original_frozen_gate2": "UNKNOWN", "release_approved": False}


def _serialized_capture(cases: list[dict]) -> bytes:
    return b"".join((json.dumps(c, sort_keys=True, ensure_ascii=False,
                                   allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
                    for c in cases)


def _admit_capture_cohort(arm: str, cases: list[dict]) -> str:
    _need(_hash(APPROVED_NATIVE_CAPTURE_COHORT_SHA256.get(arm)),
          "independently approved original native 164-response cohort SHA absent")
    _need(len(cases) == POPULATION and all(isinstance(x, dict) for x in cases),
          "original native capture population missing")
    digest = _sha(_serialized_capture(cases))
    _need(digest == APPROVED_NATIVE_CAPTURE_COHORT_SHA256[arm],
          "original independently reviewed native capture cohort SHA mismatch")
    return digest


def research_archive_inventory(arm: str = "q8") -> dict:
    """Genuine read-only archived Q4/Q8 source identities, explicitly not a frozen grade."""
    _need(arm in archived.ARMS, "only the original source-bound research cohorts may be inventoried")
    witness = archived.original_program_manifest(arm)
    _need(witness["status"] == "COMPLETE_ORIGINAL_SOURCE_BINDING_ZERO_GENERATED_PROGRAMS_EXECUTED"
          and witness["source_rows"] == POPULATION
          and witness["v2_approved_execution"] is False,
          "original archived source completeness or research boundary invalid")
    return {"schema": SCHEMA, "status": "ORIGINAL_164_SOURCE_RESEARCH_ONLY",
            "arm": witness["arm"], "model_sha256": witness["original_model_sha256"],
            "session_id": witness["original_serving_session"],
            "canonical_sha256": witness["canonical_raw_sha256"],
            "original_rows_sha256": witness["original_rows_raw_sha256"],
            "ordered_program_sha256": witness["ordered_program_manifest_sha256"],
            "cases": POPULATION, "terminated": witness["terminated_original_rows"],
            "frozen_gate2": "UNKNOWN", "generated_code_executed_by_this_module": False,
            "chosen": None, "release_approved": False}


def execution_admission_gaps() -> list[str]:
    """Code-owned security prerequisites; none can be set by an input receipt."""
    gaps = security.execution_admission_gaps()
    if not all(_hash(APPROVED_NATIVE_OWNER_REVIEW_SHA256.get(arm)) for arm in MODELS):
        gaps.append("INDEPENDENT_WINDOWS_Q8_AND_DENSE_SERVING_OWNER_APPROVAL_REQUIRED")
    if not all(_hash(APPROVED_NATIVE_CAPTURE_COHORT_SHA256.get(arm)) for arm in MODELS):
        gaps.append("INDEPENDENT_FULL164_Q8_AND_DENSE_NATIVE_CAPTURE_APPROVAL_REQUIRED")
    if not _hash(APPROVED_SANDBOX_CAPABILITY_SHA256):
        gaps.append("INDEPENDENT_FROZEN_SANDBOX_CAPABILITY_APPROVAL_REQUIRED")
    if not _hash(APPROVED_SANDBOX_BROKER_SHA256):
        gaps.append("SOURCE_REVIEWED_ISOLATION_BROKER_EXECUTABLE_REQUIRED")
    if not all(_hash(APPROVED_OUTSIDE_HOST_WITNESS_SHA256.get(arm)) for arm in MODELS):
        gaps.append("INDEPENDENT_COMPLETE_Q8_AND_DENSE_164_OUTSIDE_HOST_WITNESSES_REQUIRED")
    return gaps


def _capability(capability_path: Path, broker_path: Path) -> dict:
    """Admit only a separately source-approved immutable broker and runtime."""
    _need(not security.execution_admission_gaps(),
          "existing original HumanEval sandbox capability/witness approval absent")
    _need(_hash(APPROVED_SANDBOX_CAPABILITY_SHA256)
          and _hash(APPROVED_SANDBOX_BROKER_SHA256),
          "independently source-approved frozen isolation broker/profile absent")
    raw = _read(capability_path, MAX_JSON)
    _need(_sha(raw) == APPROVED_SANDBOX_CAPABILITY_SHA256,
          "independent sandbox capability SHA mismatch")
    broker_bytes = _read(broker_path, 16 * 1024 * 1024)
    _need(_sha(broker_bytes) == APPROVED_SANDBOX_BROKER_SHA256,
          "independent reviewed native isolation broker executable SHA mismatch")
    record = _json(raw)
    controls = ("rootless_isolated_user_pid_mount_net_namespaces", "no_network_egress",
                "readonly_host_and_no_secrets", "per_task_private_tmpfs", "no_new_privileges",
                "actual_cgroup_v2_memory_and_pids_enforced", "python_I_10_second_watchdog",
                "exact_original_python313_interpreter_pinned", "immutable_sealed_code_stdin_only",
                "outside_host_pre_execution_sha_witness", "all_descendants_extinct_after",
                "separate_original_task_cgroup", "independent_process_observer_outside_sandbox")
    _need(record.get("schema") == CAPABILITY_SCHEMA
          and record.get("status") == "INDEPENDENTLY_APPROVED_FROZEN_ORIGINAL_EXECUTION"
          and record.get("broker_sha256") == APPROVED_SANDBOX_BROKER_SHA256
          and _hash(record.get("sandbox_profile_sha256"))
          and _hash(record.get("runtime_host_sha256"))
          and _hash(record.get("sandbox_image_sha256"))
          and _hash(record.get("python_executable_sha256"))
          and record.get("python_version") == "3.13"
          and record.get("entrypoint") == ["python", "-I", "candidate.py"]
          and record.get("timeout_seconds") == 10
          and record.get("address_space_bytes") == 2 * 1024**3
          and record.get("output_limit_bytes") == 1024**2
          and isinstance(record.get("controls"), dict)
          and set(record["controls"]) == set(controls)
          and all(record["controls"][field] is True for field in controls),
          "original externally reviewed isolated execution capability incomplete")
    return record


def _candidate_packet(case: dict, tasks: list[dict]) -> tuple[dict, bytes | None]:
    """Recompute program bytes from captured text; never import/execute it."""
    _need(isinstance(case, dict) and case.get("schema") == CAPTURE_SCHEMA,
          "unrecognized original frozen native case")
    i = case.get("ordinal")
    _need(type(i) is int and 0 <= i < POPULATION and case.get("id") == tasks[i]["id"],
          "original captured case id/ordering mismatch")
    _need(case.get("arm") in MODELS and case.get("model_sha256") == MODELS[case["arm"]]
          and case.get("source_items_sha256") == FROZEN_SOURCE_ITEMS_SHA256
          and case.get("canonical_sha256") == FROZEN_CANONICAL_SHA256
          and case.get("evaluator_sha256") == EVALUATOR_SHA
          and case.get("scorer_sha256") == SCORER_SHA
          and _hash(case.get("session_id"), 32)
          and _hash(case.get("native_owner_review_sha256"))
          and case.get("generated_code_executed") is False
          and case.get("frozen_correct") is None
          and type(case.get("finished")) is bool,
          "captured case is missing original source/model/session semantics")
    response = case.get("native_generation_text")
    _need(isinstance(response, str) and len(response.encode("utf-8")) <= MAX_BYTES
          and case.get("native_response_sha256") == _sha(response.encode("utf-8")),
          "original native response text/bytes SHA mismatch")
    raw = _original_program(tasks[i], response) if case["finished"] else None
    _need(raw is None or 0 < len(raw) <= MAX_BYTES, "unbounded original candidate program")
    _need(case.get("candidate_program_sha256") == (_sha(raw) if raw else None),
          "original exact scorer program SHA mismatch")
    _need(type(case.get("completion_tokens")) is int
          and 0 <= case["completion_tokens"] <= 768
          and type(case.get("seconds")) in (int, float)
          and math.isfinite(case["seconds"]) and case["seconds"] >= 0
          and type(case.get("trigram_ratio")) in (int, float)
          and math.isfinite(case["trigram_ratio"])
          and case["trigram_ratio"] == round(reap_eval._trigram_ratio(response), 4),
          "captured frozen evaluator token/health metrics invalid")
    return case, raw


def prepare_isolation_handoff(
    case: dict, *, capture_path: Path, capability_path: Path, broker_path: Path,
    canonical: Path = archived.CANONICAL,
) -> tuple[dict, bytes]:
    """Build an approved stdin-only isolated-execution request, never execute it.

    No arbitrary `--force`, caller digest or custom command line. Generated
    Python is only embedded in a bounded JSON stdin envelope for the reviewed
    external sandbox broker; never a host file, argv, shell or environment.
    """
    cap = _capability(Path(capability_path), Path(broker_path))
    tasks = canonical_tasks(Path(canonical))
    case, raw = _candidate_packet(case, tasks)
    source = _read(Path(capture_path), 32 * 1024 * 1024)
    _need(_hash(APPROVED_NATIVE_CAPTURE_COHORT_SHA256.get(case["arm"]))
          and _sha(source) == APPROVED_NATIVE_CAPTURE_COHORT_SHA256[case["arm"]]
          and source.endswith(b"\n"),
          "independently source-approved original native capture cohort missing")
    lines = source.splitlines()
    _need(len(lines) == POPULATION, "source-reviewed 164-case capture population incomplete")
    _need(_json(lines[case["ordinal"]]) == case,
          "handoff case differs from reviewed frozen native capture source")
    _need(raw is not None, "nonterminated original HumanEval is a nonpass, never executed")
    _need(case["native_owner_review_sha256"]
          == APPROVED_NATIVE_OWNER_REVIEW_SHA256.get(case["arm"]),
          "original native model owner review not source-approved")
    meta = {"schema": SCHEMA, "status": "APPROVED_BROKER_STDIN_ONLY_PENDING_OUTSIDE_HOST_WITNESS",
            "arm": case["arm"], "ordinal": case["ordinal"], "id": case["id"],
            "model_sha256": case["model_sha256"], "session_id": case["session_id"],
            "source_items_sha256": FROZEN_SOURCE_ITEMS_SHA256,
            "canonical_sha256": FROZEN_CANONICAL_SHA256,
            "candidate_program_sha256": _sha(raw),
            "capability_sha256": APPROVED_SANDBOX_CAPABILITY_SHA256,
            "sandbox_profile_sha256": cap["sandbox_profile_sha256"],
            "runtime_host_sha256": cap["runtime_host_sha256"],
            "broker_sha256": APPROVED_SANDBOX_BROKER_SHA256}
    # The caller sees the raw candidate only after independent capability
    # admission; never execute or import it on the current host.
    payload = json.dumps({"request": meta, "candidate_source_utf8": raw.decode("utf-8")},
                         sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
    _need(len(payload) <= MAX_BYTES + MAX_JSON, "sandbox handoff packet exceeds fixed bound")
    return meta, payload


def execute_approved_broker(case: dict, *, capture_path: Path,
                            broker_path: Path, capability_path: Path,
                            canonical: Path = archived.CANONICAL) -> dict:
    """Future audited executable path; currently guaranteed to refuse.

    Never invokes Podman, WSL, shell or generated Python directly. It invokes
    ONLY a separately SHA-approved native broker, which must create/observe
    the isolated Python process and return bounded structured receipts. The
    outside-host witness is independently verified by ``score_frozen_164``.
    """
    _, packet = prepare_isolation_handoff(case, broker_path=Path(broker_path),
                                          capability_path=Path(capability_path),
                                          capture_path=Path(capture_path),
                                          canonical=Path(canonical))
    _need(os.name == "nt", "frozen original native broker requires Windows owner host")
    try:
        # The approved broker owns isolation, cgroup termination and deadlines.
        # Do not let the caller substitute argv, shell commands or script paths.
        proc = subprocess.run([str(broker_path), "--protocol", "frozen-human164-stdin-v1"],
                              input=packet, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=45, check=False,
                              env={"PATH": os.environ.get("SystemRoot", "C:\\Windows") + "\\System32",
                                   "SystemRoot": os.environ.get("SystemRoot", "C:\\Windows")})
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise FrozenExecutionRefusal("independent isolation broker failed or timed out") from exc
    _need(proc.returncode == 0 and len(proc.stdout) <= MAX_JSON
          and len(proc.stderr) <= 65536,
          "original external isolation broker failed or emitted unbounded output")
    return _json(proc.stdout)


def score_frozen_164(cases: list[dict], independent_witness_path: Path,
                     *, arm: str, capability_path: Path | None = None,
                     broker_path: Path | None = None,
                     canonical: Path = archived.CANONICAL) -> dict:
    """Score ONLY after full 164 source-pinned outside-host runtime attestation.

    A single independently pinned external witness joins all case hashes,
    unique sandbox owners, interpreter, before-release stage, actual exit and
    process-tree extinction. Never trusts `correct` in an original model row.
    """
    _need(arm in MODELS and isinstance(cases, list) and len(cases) == POPULATION,
          "frozen score requires all 164 original Q8/dense items")
    _admit_capture_cohort(arm, cases)
    _need(not security.execution_admission_gaps(),
          "HumanEval execution security approval absent; archived research cannot be promoted")
    _need(capability_path is not None and broker_path is not None,
          "source-approved sandbox capability and isolation broker missing")
    capability = _capability(Path(capability_path), Path(broker_path))
    _need(_hash(APPROVED_OUTSIDE_HOST_WITNESS_SHA256.get(arm)),
          "independent external per-case witness SHA not source-approved")
    witness_raw = _read(Path(independent_witness_path), 4 * 1024 * 1024)
    _need(_sha(witness_raw) == APPROVED_OUTSIDE_HOST_WITNESS_SHA256[arm],
          "frozen outside-host complete164 witness SHA mismatch")
    witness = _json(witness_raw) if len(witness_raw) <= MAX_JSON else _large_json(witness_raw)
    tasks = canonical_tasks(Path(canonical))
    _need(witness.get("schema") == WITNESS_SCHEMA
          and witness.get("status") == "INDEPENDENTLY_VERIFIED_ORIGINAL_FROZEN_164"
          and witness.get("arm") == arm and witness.get("model_sha256") == MODELS[arm]
          and witness.get("source_items_sha256") == FROZEN_SOURCE_ITEMS_SHA256
          and witness.get("canonical_sha256") == FROZEN_CANONICAL_SHA256
          and _hash(witness.get("capability_sha256"))
          and witness.get("capability_sha256") == APPROVED_SANDBOX_CAPABILITY_SHA256
          and witness.get("sandbox_profile_sha256") == capability["sandbox_profile_sha256"]
          and witness.get("runtime_host_sha256") == capability["runtime_host_sha256"]
          and witness.get("python_executable_sha256") == capability["python_executable_sha256"]
          and isinstance(witness.get("per_case"), list)
          and len(witness["per_case"]) == POPULATION,
          "independent frozen HumanEval witness population/profile invalid")
    outcomes = []
    owners = set()
    pids = set()
    session = None
    for i, (case, proof) in enumerate(zip(cases, witness["per_case"])):
        case, source = _candidate_packet(case, tasks)
        _need(case["arm"] == arm and case["ordinal"] == i,
              "original full164 case model or order changed")
        session = case["session_id"] if session is None else session
        _need(case["session_id"] == session
              and case["native_owner_review_sha256"] == APPROVED_NATIVE_OWNER_REVIEW_SHA256.get(arm),
              "original native serving owner/session not source-approved")
        _need(type(proof) is dict and proof.get("ordinal") == i
              and proof.get("id") == case["id"]
              and proof.get("session_id") == session
              and proof.get("candidate_program_sha256") == case["candidate_program_sha256"],
              "outside-host case/order/candidate SHA mismatch")
        if source is None:
            _need(proof.get("status") == "NOT_EXECUTED_UNTERMINATED"
                  and proof.get("correct") is False
                  and proof.get("child_pid") is None
                  and proof.get("sandbox_owner") is None,
                  "original unterminated response was executed or scored")
            outcomes.append(False)
            continue
        owner = proof.get("sandbox_owner")
        child = proof.get("child_pid")
        _need(proof.get("status") in ("PYTHON_EXIT", "PYTHON_TIMEOUT_10S")
              and type(proof.get("correct")) is bool
              and _hash(proof.get("independent_trace_sha256"))
              and isinstance(owner, str) and 8 <= len(owner) <= 256 and owner not in owners
              and type(child) is int and child > 0 and child not in pids
              and proof.get("independently_observed_preexecution_source_sha256") == _sha(source)
              and proof.get("sealed_memfd_source_matches") is True
              and proof.get("python_isolated_mode") is True
              and proof.get("observer_outside_workload_namespaces") is True
              and proof.get("network_egress_denied") is True
              and proof.get("host_secrets_denied") is True
              and proof.get("cgroup_limits_enforced") is True
              and proof.get("descendants_extinct_after") is True
              and proof.get("observed_cgroup_empty_after") is True
              and type(proof.get("elapsed_milliseconds")) is int
              and 0 < proof["elapsed_milliseconds"] <= 12000,
              "original candidate isolation and independent process witness incomplete")
        owners.add(owner)
        pids.add(child)
        if proof["status"] == "PYTHON_TIMEOUT_10S":
            _need(proof.get("correct") is False and proof.get("exit_code") is None
                  and proof.get("timeout_kill_observed") is True,
                  "independently timed-out generated Python cannot pass")
        else:
            _need(type(proof.get("exit_code")) is int
                  and proof["correct"] is (proof["exit_code"] == 0)
                  and proof.get("timeout_kill_observed") is False,
                  "candidate exit and original pass@1 scoring disagree")
        outcomes.append(proof["correct"])
    _need(witness.get("session_id") == session
          and type(witness.get("independent_correct")) is int
          and witness["independent_correct"] == sum(outcomes)
          and witness.get("release_approved") is False,
          "frozen model case aggregate or scientific release status inconsistent")
    return {"schema": SCHEMA, "status": "INDEPENDENTLY_SOURCE_PINNED_FROZEN_164_CASE_RESULTS",
            "arm": arm, "model_sha256": MODELS[arm], "session_id": session,
            "source_items_sha256": FROZEN_SOURCE_ITEMS_SHA256,
            "outside_host_witness_sha256": _sha(witness_raw),
            "evaluated_original_rows": POPULATION, "correct": sum(outcomes),
            "accuracy": sum(outcomes) / POPULATION,
            "outcomes": [{"id": tasks[i]["id"], "correct": outcomes[i],
                          "terminated": cases[i]["finished"]} for i in range(POPULATION)],
            "gate2": "UNKNOWN_UNTIL_ORIGINAL_SAME_ITEM_DENSE_PAIR_AND_SELECTOR",
            "chosen": None, "release_approved": False}


def build_selector_rows(cases: list[dict], grade: dict, *, arm: str,
                        canonical: Path = archived.CANONICAL) -> dict:
    """Return exact selector-compatible JSONL/summary bytes without filesystem mutation.

    Requires already independently admitted `score_frozen_164` result; does
    not call the unsafe original host Python scorer or assert Gate2 PASS.
    """
    _need(arm in MODELS and isinstance(grade, dict)
          and grade.get("status") == "INDEPENDENTLY_SOURCE_PINNED_FROZEN_164_CASE_RESULTS"
          and grade.get("arm") == arm and grade.get("model_sha256") == MODELS[arm]
          and grade.get("source_items_sha256") == FROZEN_SOURCE_ITEMS_SHA256
          and _hash(grade.get("outside_host_witness_sha256"))
          and grade["outside_host_witness_sha256"] == APPROVED_OUTSIDE_HOST_WITNESS_SHA256.get(arm)
          and grade.get("release_approved") is False
          and isinstance(grade.get("outcomes"), list) and len(grade["outcomes"]) == POPULATION
          and isinstance(cases, list) and len(cases) == POPULATION,
          "independently source-pinned frozen 164 grade missing")
    tasks = canonical_tasks(Path(canonical))
    rows = []
    for i, (case, outcome) in enumerate(zip(cases, grade["outcomes"])):
        case, _ = _candidate_packet(case, tasks)
        _need(case["arm"] == arm and case["session_id"] == grade.get("session_id")
              and outcome.get("id") == case["id"]
              and type(outcome.get("correct")) is bool
              and outcome.get("terminated") is case["finished"]
              and (case["finished"] or not outcome["correct"]),
              "selector raw row is unpaired, nonbinary or forged")
        rows.append({"id": case["id"], "correct": outcome["correct"],
                     "terminated": case["finished"], "trigram_ratio": case["trigram_ratio"],
                     "completion_tokens": case["completion_tokens"], "seconds": case["seconds"],
                     "predicted_per_second": case["predicted_per_second"],
                     "text": case["native_generation_text"],
                     "eval_session_id": case["session_id"]})
    _need(sum(r["correct"] for r in rows) == grade.get("correct"),
          "source-pinned independent HumanEval aggregate differs from selector rows")
    raw = b"".join((json.dumps(row, sort_keys=True, ensure_ascii=False,
                                separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
                   for row in rows)
    return {"schema": SCHEMA, "status": "SELECTOR_COMPATIBLE_AUTHENTICATED_ROWS_NOT_PUBLISHED",
            "arm": arm, "row_count": POPULATION, "jsonl_bytes": raw,
            "summary_entry": {
                "n": POPULATION, "accuracy": grade["correct"] / POPULATION,
                "terminated_rate": sum(r["terminated"] for r in rows) / POPULATION,
                "degenerate_rate": sum(r["trigram_ratio"] < 0.5 for r in rows) / POPULATION,
                "mean_completion_tokens": sum(r["completion_tokens"] for r in rows) / POPULATION,
                "file_sha256": _sha(raw)},
            "frozen_gate2": "UNKNOWN_UNTIL_SAME_ITEM_Q8_VS_DENSE_SELECTOR",
            "release_approved": False}


def _large_json(raw: bytes) -> dict:
    """Bounded full-cohort external witness, retaining strict JSON semantics."""
    _need(len(raw) <= 4 * 1024 * 1024, "unbounded complete164 witness")
    try:
        value = json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite")),
                           object_pairs_hook=_unique_pairs)
    except (ValueError, UnicodeError) as exc:
        raise FrozenExecutionRefusal("invalid original full164 outside-host witness") from exc
    _need(isinstance(value, dict), "full164 independent witness must be an object")
    return value


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate source-bound witness JSON key")
        result[key] = value
    return result
