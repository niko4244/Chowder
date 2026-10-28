"""Fail-closed original REAP v2 supervisor/wave integration checks.

Production defect in HEAD fa226ac: the existing supervisor can treat 41/41
layer offload as proof of zero CPU-routed-expert offload and mark a model
fit witnessed WITHOUT a native 'successfully fit' source line. This module
provides the additive remediation; it does not monkeypatch or launch models.

Prime integration (in the existing supervisor, NOT applied by this worker):
  after _watch_sample and before watched_fit/terminal PASS, read ONLY that
  session's bounded, owned server.log startup prefix; call
      assess_runtime_integration(sample, startup=prefix,
                                 expected_model=model)
  and fail the owned run if status != RUNTIME_INTEGRATION_OBSERVED_ONLY.
  Separately call assess_bound_owner on a read-only live job/child snapshot;
  do not promote a runtime receipt without assess_bound_receipts and a
  genuinely witnessed post-close descendant census.

No function grants scientific selection, native model-launch authorization, or
termination authority over any process. All resulting states retain Gen0 BLOCKED.
"""

from __future__ import annotations

from dataclasses import astuple, fields
import hashlib
import math
from pathlib import Path
import re
import sys
from typing import Any

import psutil

from chowder_batch import reap_serve
from chowder_batch import reap_v2_dedicated_runner_admission_20261027 as native
from chowder_batch import reap_v2_frozen_wave_20261027 as wave
from chowder_batch import reap_v2_owner_bound_runtime_supervisor_20261027 as supervisor

GEN0_BLOCKED = {"status": "BLOCKED", "chosen": None, "release_approved": False}
MAX_SOURCE_BYTES = supervisor.MAX_LOG_PREFIX_BYTES
_CONTEXT = re.compile(r"\bllama_context:\s+n_ctx\s*=\s*(\d+)")
_SLOTS = re.compile(r"\bn_seq_max\s*=\s*(\d+)")
_CTX_PER_SLOT = re.compile(r"\bn_ctx_seq\s*=\s*(\d+)")
_LAYERS = re.compile(r"\boffloaded\s+(\d+)/(\d+)\s+layers\s+to GPU", re.I)
_KV = re.compile(r"\bCUDA\d+ KV buffer size\s*=\s*([0-9]+(?:\.[0-9]+)?) MiB")
_CPU = re.compile(r"\bn_cpu_moe\s*[=:]\s*(\d+)", re.I)
_MODEL = re.compile(r"loading model '([^'\r\n]+)'", re.I)
_EXPERTS = re.compile(r"\bn_expert\s*=\s*(\d+)")
_ACTIVE = re.compile(r"\bn_expert_used\s*=\s*(\d+)")


class GuardRefused(ValueError):
    """Foreign owner, incomplete native source, or unjustified runtime claim."""


def _failure(status: str, reasons: list[str], **extra) -> dict:
    return {
        "status": status if not reasons else "REFUSED",
        "reasons": sorted(set(reasons)),
        "gen0": dict(GEN0_BLOCKED),
        "model_selection_approved": False,
        "execution_approved": False,
        **extra,
    }


def _path_string(value: Any) -> str:
    # Native Windows paths can use either path separator in llama.cpp logs.
    if not isinstance(value, (str, Path)):
        return ""
    return str(value).replace("\\", "/").casefold()


def _single_integer(pattern: re.Pattern, text: str) -> int | None:
    matches = pattern.findall(text)
    return int(matches[0]) if len(matches) == 1 else None


def assess_native_startup(source: bytes, *, expected_model: Path) -> dict:
    """Independent source-bound native context/KV/routed-expert fit witness.

    Numeric layers/context/KV alone NEVER imply zero CPU experts. Require a
    source-native successful-fit marker, no CPU override/overflow, and the
    original reap_serve.parse_offload test. Missing data is UNKNOWN, not PASS.
    """
    if (
        not isinstance(source, bytes)
        or not 0 < len(source) <= MAX_SOURCE_BYTES
        or not isinstance(expected_model, Path)
        or not expected_model.is_absolute()
    ):
        return _failure(
            "SOURCE_FIT_WITNESSED_ONLY",
            ["NATIVE_STARTUP_SOURCE_INVALID"],
            residency=None,
        )
    try:
        text = source.decode("utf-8", errors="strict")
    except UnicodeError:
        return _failure(
            "SOURCE_FIT_WITNESSED_ONLY",
            ["NATIVE_STARTUP_SOURCE_INVALID"],
            residency=None,
        )
    if "\x00" in text:
        return _failure(
            "SOURCE_FIT_WITNESSED_ONLY",
            ["NATIVE_STARTUP_SOURCE_INVALID"],
            residency=None,
        )
    problems: list[str] = []
    models = _MODEL.findall(text)
    if len(models) != 1 or _path_string(models[0]) != _path_string(expected_model):
        problems.append("NATIVE_MODEL_IDENTITY")
    if not re.search(
        r"\b(?:llama_params_fit|common_fit_params)[^\n]*successfully fit", text, re.I
    ):
        problems.append("NATIVE_SUCCESSFUL_FIT_UNPROVEN")
    context_list = _CONTEXT.findall(text)
    if len(context_list) != 1:
        problems.append("NATIVE_CONTEXT_AMBIGUOUS")
    n_ctx = int(context_list[0]) if len(context_list) == 1 else None
    slots = _single_integer(_SLOTS, text)
    per_slot = _single_integer(_CTX_PER_SLOT, text)
    if n_ctx != 16384 or per_slot != 4096:
        problems.append("NATIVE_FROZEN_CONTEXT")
    if slots != 4:
        problems.append("NATIVE_FROZEN_SLOTS")
    layers = _LAYERS.findall(text)
    if len(layers) != 1 or tuple(map(int, layers[0])) != (41, 41):
        problems.append("NATIVE_LAYER_OFFLOAD")
    experts = _single_integer(_EXPERTS, text)
    active = _single_integer(_ACTIVE, text)
    if experts != 128 or active != 8:
        problems.append("NATIVE_EXPERT_GEOMETRY")
    kv = _KV.findall(text)
    if not kv:
        kv_mib = None
        problems.append("NATIVE_KV_UNOBSERVED")
    else:
        kv_mib = sum(float(v) for v in kv)
        if not math.isfinite(kv_mib) or not 0 < kv_mib <= supervisor.MAX_KV_MIB:
            problems.append("NATIVE_KV_UNBOUNDED")
    cpu = _CPU.findall(text)
    if len(cpu) > 1:
        problems.append("NATIVE_CPU_EXPERT_OFFLOAD")
    source_lines, source_cpu_offload = reap_serve.parse_offload(text)
    if (
        source_cpu_offload != 0
        or any(int(v) != 0 for v in cpu)
        or re.search(r"\boverflowing\b|\b(?:blk\.[0-9]+).*exps.*CPU", text, re.I)
    ):
        problems.append("NATIVE_CPU_EXPERT_OFFLOAD")
    if not source_lines:
        problems.append("NATIVE_SOURCE_FIT_PARSER_UNKNOWN")
    return _failure(
        "SOURCE_FIT_WITNESSED_ONLY",
        problems,
        source_sha256=hashlib.sha256(source).hexdigest(),
        original_source_parser_cpu_expert_count=source_cpu_offload,
        residency={
            "layers_offloaded": 41 if layers and layers[-1][0] == "41" else None,
            "layers_total": 41 if layers and layers[-1][1] == "41" else None,
            "context": n_ctx,
            "slots": slots,
            "context_per_slot": per_slot,
            "kv_total_mib": kv_mib,
            "cpu_expert_offload": 0
            if source_cpu_offload == 0 and not problems
            else None,
        },
    )


def rebind_wave_study(study: object) -> wave.Study:
    """Accommodate CLI __main__ vs imported Study WITHOUT widening identity."""
    names = tuple(f.name for f in fields(wave.Study))
    try:
        values = tuple(getattr(study, name) for name in names)
    except (AttributeError, TypeError) as exc:
        raise GuardRefused("FROZEN_STUDY_FIELDS_MISSING") from exc
    if (
        len(values) != 4
        or not all(type(x) is str for x in (values[0], values[1], values[3]))
        or type(values[2]) is not int
    ):
        raise GuardRefused("FROZEN_STUDY_TYPE_MISMATCH")
    try:
        original = wave.study_for(values[0])
    except (ValueError, TypeError) as exc:
        raise GuardRefused("UNREGISTERED_FROZEN_STUDY") from exc
    if tuple(astuple(original)) != values:
        raise GuardRefused("FROZEN_STUDY_IDENTITY_CHANGED")
    return original


def assess_runtime_integration(
    sample: dict, *, startup: bytes, expected_model: Path
) -> dict:
    """Companion to supervisor.validate_sample; not a substitute for owner proof."""
    resource_failures = supervisor.validate_sample(sample)
    fit = assess_native_startup(startup, expected_model=expected_model)
    reasons = [*resource_failures, *fit["reasons"]]
    if (
        isinstance(sample, dict)
        and isinstance(sample.get("native_residency"), dict)
        and fit["residency"] is not None
        and not fit["reasons"]
    ):
        for key in (
            "layers_offloaded",
            "layers_total",
            "context",
            "kv_total_mib",
            "cpu_expert_offload",
        ):
            if sample["native_residency"].get(key) != fit["residency"].get(key):
                reasons.append("NATIVE_SOURCE_RUNTIME_MISMATCH")
                break
    return _failure(
        "RUNTIME_INTEGRATION_OBSERVED_ONLY",
        reasons,
        source_fit_sha256=fit.get("source_sha256"),
        material_runtime_limits_verified=not resource_failures,
    )


def assess_bound_owner(
    owner: dict,
    *,
    study: wave.Study,
    model: Path,
    server: Path,
    output: Path,
    port: int,
) -> dict:
    """Pure read-only actual Job-member/cmdline/PID-birth/port graph audit.

    The caller MUST construct owner from fresh Win32 Job/PID-handle observations,
    never from a caller-provided receipt. No process is opened for termination.
    """
    issues = []
    try:
        study = rebind_wave_study(study)
        command = wave.command(
            study, model=model, server=server, output=output, port=port
        )
    except (ValueError, TypeError, AttributeError) as exc:
        return _failure(
            "OWNER_BOUND_OBSERVATION_ONLY",
            ["FROZEN_WAVE_COMMAND_UNKNOWN"],
            member_count=0,
        )
    if not isinstance(owner, dict):
        owner = {}
    if owner.get("active") is not True or owner.get("job_present") is not True:
        issues.append("OWNED_JOB_NOT_ACTIVE")
    if owner.get("job_limit_kill_on_close") is not True:
        issues.append("JOB_KILL_ON_CLOSE_UNPROVEN")
    members = owner.get("members")
    if not isinstance(members, dict) or not members:
        members = {}
        issues.append("JOB_MEMBER_CENSUS_MISSING")
    root_id = owner.get("root_pid")
    root_birth = owner.get("root_birth")
    root = members.get(root_id)
    if (
        type(root_id) is not int
        or root_id <= 0
        or type(root_birth) not in (int, float)
        or not math.isfinite(root_birth)
        or not isinstance(root, dict)
        or root.get("pid") != root_id
        or type(root.get("birth")) not in (float, int)
        or abs(root.get("birth", 0) - root_birth) > 0.002
    ):
        issues.append("ROOT_CREATE_TIME_MISMATCH")
    if not isinstance(root, dict) or root.get("argv") != command:
        issues.append("FROZEN_ROOT_COMMAND_CHANGED")
    if len(members) > supervisor.MAX_JOB_PROCESSES:
        issues.append("JOB_MEMBER_LIMIT")
    for pid, record in members.items():
        if (
            type(pid) is not int
            or pid <= 0
            or not isinstance(record, dict)
            or record.get("pid") != pid
            or type(record.get("birth")) not in (float, int)
            or not math.isfinite(record.get("birth"))
        ):
            issues.append("JOB_MEMBER_PID_OR_BIRTH_INVALID")
            continue
        if pid != root_id:
            if (
                record.get("birth") < root_birth
                or type(record.get("parent")) is not int
                or record["parent"] not in members
            ):
                issues.append(
                    "CHILD_CREATED_BEFORE_ROOT"
                    if record["birth"] < root_birth
                    else "FOREIGN_PARENT_CHAIN"
                )
    expected_server = [str(server), *wave.FLAGS, "--port", str(port), "-m", str(model)]
    server_members = [
        row
        for row in members.values()
        if isinstance(row, dict)
        and isinstance(row.get("argv"), list)
        and row["argv"]
        and Path(str(row["argv"][0])).name.casefold() == "llama-server.exe"
    ]
    if len(server_members) != 1 or (
        server_members and server_members[0].get("argv") != expected_server
    ):
        issues.append("NATIVE_SERVER_COMMAND_MISMATCH")
    evaluator_expected = [
        sys.executable,
        str(wave.REPO / "chowder_batch" / "reap_eval.py"),
        *command[command.index("--") + 1 :],
    ]
    for member in members.values():
        if not isinstance(member, dict) or member.get("pid") == root_id:
            continue
        if member.get("argv") not in (expected_server, evaluator_expected):
            issues.append("FOREIGN_OWNED_CHILD_COMMAND")
    listener = owner.get("owned_listener")
    if (
        not isinstance(listener, dict)
        or not server_members
        or listener.get("pid") != server_members[0].get("pid")
        or listener.get("port") != port
    ):
        issues.append("OWNED_LISTENER_MISMATCH")
    if not isinstance(owner.get("other_listeners"), list) or owner["other_listeners"]:
        issues.append("FOREIGN_LISTENER")
    return _failure("OWNER_BOUND_OBSERVATION_ONLY", issues, member_count=len(members))


def assess_bound_receipts(
    start: dict,
    terminal: dict,
    *,
    study: wave.Study,
    model: Path,
    server: Path,
    output: Path,
    port: int,
) -> dict:
    """Source-locked session join; NEVER mistake Job-close claim for extinction."""
    issues = []
    try:
        study = rebind_wave_study(study)
        argv = wave.command(study, model=model, server=server, output=output, port=port)
    except (TypeError, ValueError):
        return _failure(
            "RECEIPTS_CONSISTENT_NOT_SCIENTIFIC_APPROVAL", ["FROZEN_STUDY_INVALID"]
        )
    if not isinstance(start, dict) or not isinstance(terminal, dict):
        return _failure(
            "RECEIPTS_CONSISTENT_NOT_SCIENTIFIC_APPROVAL", ["RECEIPT_MALFORMED"]
        )
    for receipt in (start, terminal):
        if (
            receipt.get("schema") != supervisor.SCHEMA
            or receipt.get("arm") != study.arm
            or receipt.get("model_sha256") != study.model_sha256
            or receipt.get("gen0") != GEN0_BLOCKED
        ):
            issues.append("REGISTERED_RECEIPT_IDENTITY_MISMATCH")
    if start.get("status") != "PRELAUNCH_SOURCE_AND_CAPACITY_VERIFIED":
        issues.append("OWNER_START_NOT_SOURCE_ADMITTED")
    if start.get("command") != argv:
        issues.append("FROZEN_OWNER_COMMAND_CHANGED")
    if not isinstance(start.get("native_server_sha256"), str) or not re.fullmatch(
        r"[a-f0-9]{64}", start["native_server_sha256"]
    ):
        issues.append("NATIVE_BINARY_SHA_MISSING")
    a = start.get("original_offline_source_proof")
    b = terminal.get("original_offline_source_proof")
    if (
        not isinstance(a, dict)
        or not isinstance(b, dict)
        or a.get("source_offline_verified") is not True
        or b.get("source_offline_verified") is not True
        or not isinstance(a.get("sha256"), str)
        or not re.fullmatch(r"[a-f0-9]{64}", a["sha256"])
        or a["sha256"] != b.get("sha256")
    ):
        issues.append("ORIGINAL_OFFLINE_SOURCE_MISMATCH")
    if (
        terminal.get("status") != "CHILD_EXIT_ZERO_RUNTIME_WITNESSED_ONLY"
        or type(terminal.get("child_pid")) is not int
        or terminal["child_pid"] <= 0
        or type(terminal.get("child_create_time_unix")) not in (int, float)
        or not math.isfinite(terminal["child_create_time_unix"])
    ):
        issues.append("CHILD_RUNTIME_EXIT_UNVERIFIED")
    if (
        terminal.get("post_close_descendant_extinction_independently_witnessed")
        is not True
    ):
        issues.append("OWNED_TREE_EXTINCTION_UNVERIFIED")
    if terminal.get("owned_job_handle_closed_with_kill_on_close") is not True:
        issues.append("OWNED_JOB_CLOSE_UNVERIFIED")
    return _failure("RECEIPTS_CONSISTENT_NOT_SCIENTIFIC_APPROVAL", issues)


def observe_descendant_extinction(known_births: dict[int, float]) -> dict:
    """Read-only PID create-time census; never signal a foreign/reused PID."""
    if not isinstance(known_births, dict) or not known_births:
        return _failure(
            "OWNED_DESCENDANTS_EXTINCT_OBSERVED_ONLY", ["EMPTY_CHILD_PID_BIRTH_LEDGER"]
        )
    failures = []
    for pid, original_birth in known_births.items():
        if (
            type(pid) is not int
            or pid <= 0
            or type(original_birth) not in (float, int)
            or not math.isfinite(original_birth)
        ):
            failures.append("CHILD_PID_BIRTH_LEDGER_INVALID")
            continue
        try:
            p = psutil.Process(pid)
            if abs(p.create_time() - original_birth) < 0.002 and p.is_running():
                failures.append("ORIGINAL_CHILD_STILL_ALIVE")
        except psutil.NoSuchProcess:
            continue
        except (psutil.AccessDenied, psutil.Error, OSError):
            failures.append("CHILD_EXIT_UNOBSERVABLE")
    return _failure(
        "OWNED_DESCENDANTS_EXTINCT_OBSERVED_ONLY",
        failures,
        checked_child_pid_count=len(known_births),
    )
