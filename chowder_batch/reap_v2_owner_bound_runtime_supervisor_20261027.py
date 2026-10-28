"""Windows native owner-bound runtime supervisor for ONE original frozen GSM8K arm.

Prime integration, once a separately authorized resource-eligible dedicated
host exists:

    from chowder_batch.reap_v2_owner_bound_runtime_supervisor_20261027 import run_frozen_arm
    run_frozen_arm(study=wave.study_for(arm), model=model, server=server,
                   output=output, receipt_dir=existing_empty_receipt_directory,
                   port=8790, execute_authorized=True)

Default invocation refuses before any filesystem mutation, GPU work or launch.
run_frozen_arm independently repeats native source/host admission. A successful
result is only a watched native child exit, never a scientific score, speed
ratio, model selection or permission to run generated HumanEval programs.

The model runner is started CREATE_SUSPENDED and attached to a private Windows
Job Object configured KILL_ON_JOB_CLOSE and active-process limit before resume.
Only the handle created in this invocation is ever terminated. No taskkill,
Process.kill(pid), service control, Podman or GPU work is invoked by this module.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import uuid

import psutil

from chowder_batch import reap_v2_dedicated_runner_admission_20261027 as native
from chowder_batch import reap_v2_frozen_wave_20261027 as wave


GIB = 1024**3
FROZEN_FLAGS = tuple(native.FROZEN_FLAGS)
RUNTIME_PHYSICAL_FLOOR_BYTES = 7 * GIB
RUNTIME_COMMIT_FLOOR_BYTES = 8 * GIB
RUNTIME_DISK_FLOOR_BYTES = 8 * GIB
RUNTIME_GPU_FREE_MIB = 2048  # observed post-load floor, NOT startup or science
MAX_KV_MIB = 4096
MAX_WALL_SECONDS = 14400
MAX_STARTUP_GRACE_SECONDS = 900
MAX_LOG_BYTES = 256 * 2**20
MAX_LOG_PREFIX_BYTES = 256 * 1024
MAX_SAMPLES = 4096
MAX_JOB_PROCESSES = 16
SCHEMA = "chowder.reap-v2-owned-windows-runtime.v1"


class SupervisorRefused(RuntimeError):
    """Observed identity, resource or ownership does not authorize execution."""


def _number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _integer(value: object) -> bool:
    return type(value) is int and value >= 0


def parse_native_residency(text: str) -> dict | None:
    """Bounded startup-log evidence; partial logs return UNKNOWN (None)."""
    if not isinstance(text, str) or len(text) > MAX_LOG_PREFIX_BYTES:
        return None
    ctx = re.findall(r"\bn_ctx\s*=\s*(\d+)", text)
    layers = re.findall(r"\boffloaded\s+(\d+)/(\d+)\s+layers\s+to GPU", text, re.I)
    kv = re.findall(r"\bCUDA\d+ KV buffer size\s*=\s*([0-9]+(?:\.[0-9]+)?) MiB", text)
    cpu = re.findall(r"\bn_cpu_moe\s*[=:]\s*(\d+)", text)
    spilling = (
        re.search(r"\b(?:overflowing|exps.*(?:overridden|CPU))\b", text, re.I)
        is not None
    )
    if not ctx or not layers or not kv:
        return None
    count = sum(float(n) for n in kv)
    if not math.isfinite(count):
        return None
    a, b = tuple(map(int, layers[-1]))
    # Absence of an explicit n_cpu_moe marker is UNKNOWN, unless native fit
    # confirms all layers and no CPU expert override in this startup prefix.
    cpu_count = int(cpu[-1]) if cpu else (0 if a == b and not spilling else None)
    return {
        "layers_offloaded": a,
        "layers_total": b,
        "context": int(ctx[-1]),
        "kv_total_mib": count,
        "cpu_expert_offload": cpu_count,
    }


def validate_sample(
    sample: dict,
    *,
    startup_grace_seconds: int = MAX_STARTUP_GRACE_SECONDS,
    max_runtime_seconds: int = MAX_WALL_SECONDS,
) -> list[str]:
    """Pure fail-closed runtime bounds; does not launch or kill processes."""
    if not isinstance(sample, dict) or not _number(startup_grace_seconds):
        return ["RUNTIME_TELEMETRY_INVALID"]
    failures = []
    for key, bound, reason in (
        ("physical_available_bytes", RUNTIME_PHYSICAL_FLOOR_BYTES, "RAM_RUNTIME_FLOOR"),
        ("commit_headroom_bytes", RUNTIME_COMMIT_FLOOR_BYTES, "COMMIT_RUNTIME_FLOOR"),
        ("disk_free_bytes", RUNTIME_DISK_FLOOR_BYTES, "DISK_RUNTIME_FLOOR"),
    ):
        val = sample.get(key)
        if not _integer(val) or val < bound:
            failures.append(reason)
    if sample.get("root_owner_valid") is not True:
        failures.append("ROOT_PID_REUSED_OR_FOREIGN")
    if sample.get("job_processes_valid") is not True:
        failures.append("JOB_PROCESS_IDENTITY")
    if (
        not _integer(sample.get("job_process_count"))
        or sample["job_process_count"] > MAX_JOB_PROCESSES
    ):
        failures.append("JOB_PROCESS_LIMIT")
    foreign = sample.get("foreign_ports")
    if not isinstance(foreign, list) or foreign:
        failures.append("FOREIGN_LISTENER")
    extra = sample.get("foreign_processes")
    if not isinstance(extra, list) or extra:
        failures.append("FOREIGN_MODEL_PROCESS")
    elapsed = sample.get("elapsed_seconds")
    started = sample.get("startup_seconds")
    if not _number(elapsed) or elapsed > max_runtime_seconds:
        failures.append("WALL_TIME_LIMIT")
    if not _number(started):
        failures.append("STARTUP_GRACE_EXPIRED")
    if (
        not _integer(sample.get("model_log_bytes"))
        or sample["model_log_bytes"] > MAX_LOG_BYTES
    ):
        failures.append("LOG_BYTE_LIMIT")
    gpu = sample.get("gpu")
    if not isinstance(gpu, dict):
        gpu = {}
    values = [
        gpu.get(key)
        for key in (
            "total_mib",
            "used_mib",
            "free_mib",
            "reserved_mib",
            "utilization_pct",
        )
    ]
    if (
        not all(_integer(x) for x in values)
        or not isinstance(gpu.get("uuid"), str)
        or not re.fullmatch(r"GPU-[A-Za-z0-9-]{8,64}", gpu.get("uuid", ""))
        or values[1] + values[2] + values[3] != values[0]
        or values[3] > 2048
        or values[4] > 100
    ):
        failures.append("GPU_RUNTIME_COUNTERS")
    elif values[2] < RUNTIME_GPU_FREE_MIB:
        failures.append("GPU_RUNTIME_FLOOR")
    ready = sample.get("native_started") is True
    residence = sample.get("native_residency")
    if ready and not isinstance(residence, dict):
        failures.append("NATIVE_RESIDENCY_MISSING")
    if (
        not ready
        or sample.get("owned_listener") is not True
        or not isinstance(residence, dict)
    ):
        if not _number(started) or started > startup_grace_seconds:
            failures.append("STARTUP_GRACE_EXPIRED")
        return sorted(set(failures))
    if (
        not _integer(residence.get("layers_offloaded"))
        or residence.get("layers_offloaded") != 41
        or residence.get("layers_total") != 41
    ):
        failures.append("NATIVE_GPU_FIT")
    if residence.get("context") != 16384 or type(residence.get("context")) is not int:
        failures.append("NATIVE_CONTEXT_CHANGED")
    if (
        not _number(residence.get("kv_total_mib"))
        or not 0 < residence["kv_total_mib"] <= MAX_KV_MIB
    ):
        failures.append("KV_ALLOCATION_UNBOUNDED")
    if (
        residence.get("cpu_expert_offload") != 0
        or type(residence.get("cpu_expert_offload")) is not int
    ):
        failures.append("CPU_EXPERT_OFFLOAD")
    return sorted(set(failures))


def validate_frozen_command(
    command: list[str],
    study: wave.Study,
    model: Path,
    server: Path,
    output: Path,
    port: int,
) -> None:
    """Strict original wave command equality; no unregistered arguments."""
    if (
        sys.platform != "win32"
        or study != wave.study_for(study.arm)
        or tuple(wave.FLAGS) != FROZEN_FLAGS
        or not _integer(port)
        or not 1024 <= port <= 65535
        or type(command) is not list
        or command
        != wave.command(study, model=model, server=server, output=output, port=port)
        or Path(command[0]).resolve() != Path(sys.executable).resolve()
        or Path(command[1]).resolve()
        != (wave.REPO / "chowder_batch" / "reap_serve.py").resolve()
        or not Path(server).is_absolute()
        or Path(server).name.lower() != "llama-server.exe"
    ):
        raise ValueError("FROZEN_NATIVE_CHILD_COMMAND_INVALID")


def write_receipt_exclusive(path: Path, payload: dict) -> None:
    """Only caller-owned, new, fsynced receipt; never overwrite prior evidence."""
    target = Path(path)
    if (
        not target.is_absolute()
        or target.is_symlink()
        or not target.parent.is_dir()
        or target.parent.is_symlink()
        or not isinstance(payload, dict)
    ):
        raise ValueError("RECEIPT_TARGET_INVALID")
    raw = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    if len(raw) > 2 * 2**20:
        raise ValueError("RECEIPT_TOO_LARGE")
    pending = target.with_name("." + target.name + "." + uuid.uuid4().hex + ".pending")
    try:
        with pending.open("xb") as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        if sys.platform == "win32":
            pending.rename(target)  # refuses an existing destination on Windows
        else:
            os.link(pending, target)
    finally:
        pending.unlink(missing_ok=True)


def verify_frozen_sources_offline(*, receipt_path: Path) -> dict:
    """Prove both frozen original HF cohorts in a separate offline CPU child."""
    from chowder_batch import reap_v2_original_offline_source_lock_20261027 as source

    if (
        receipt_path.exists()
        or receipt_path.is_symlink()
        or not receipt_path.parent.is_dir()
    ):
        raise SupervisorRefused("OFFLINE_SOURCE_RECEIPT_NOT_EXCLUSIVE")
    child_env = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "REAP_ORIGINAL_SOURCE_OFFLINE_CHILD": "1",
    }
    command = [
        sys.executable,
        "-m",
        "chowder_batch.reap_v2_original_offline_source_lock_20261027",
        "--out",
        str(receipt_path),
    ]
    try:
        child = subprocess.run(
            command,
            cwd=wave.REPO,
            env=child_env,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SupervisorRefused("OFFLINE_SOURCE_CHECK_UNAVAILABLE") from exc
    if child.returncode != 0 or not receipt_path.is_file() or receipt_path.is_symlink():
        raise SupervisorRefused("OFFLINE_SOURCE_CHECK_REFUSED")
    if receipt_path.stat().st_size > 65536:
        raise SupervisorRefused("OFFLINE_SOURCE_RECEIPT_OVERSIZED")

    def no_duplicate_keys(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("DUPLICATE_SOURCE_RECEIPT_KEY")
            result[key] = value
        return result

    try:
        raw = receipt_path.read_bytes()
        record = json.loads(
            raw,
            object_pairs_hook=no_duplicate_keys,
            parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)),
        )
        cohorts = record["cohorts"]
        if (
            record.get("schema") != source.SCHEMA
            or record.get("status")
            != "COMPLETE_ORIGINAL_FROZEN_OFFLINE_SOURCES_MATCHED"
            or record.get("evaluator_sha256") != source.EVAL_SHA256
            or record.get("network_or_model_called") is not False
            or record.get("official_quality_scored") is not False
            or record.get("gen0") != native.GEN0_BLOCKED
            or any(
                cohorts[suite]["n"] != expected[0]
                or cohorts[suite]["full_rendered_source_sha256"] != expected[1]
                or cohorts[suite].get("full_population_and_order_verified") is not True
                for suite, expected in (
                    ("gsm8k", source.GSM),
                    ("humaneval", source.HUMAN),
                )
            )
        ):
            raise ValueError("OFFLINE_SOURCE_POPULATION_CHANGED")
    except (KeyError, TypeError, OSError, ValueError, UnicodeError) as exc:
        raise SupervisorRefused("OFFLINE_SOURCE_RECEIPT_INVALID") from exc
    return {
        "receipt": str(receipt_path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "evaluator_sha256": source.EVAL_SHA256,
        "cohorts": cohorts,
        "source_offline_verified": True,
    }


class _STARTUPINFO(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("lpReserved", wintypes.LPWSTR),
        ("lpDesktop", wintypes.LPWSTR),
        ("lpTitle", wintypes.LPWSTR),
        ("dwX", wintypes.DWORD),
        ("dwY", wintypes.DWORD),
        ("dwXSize", wintypes.DWORD),
        ("dwYSize", wintypes.DWORD),
        ("dwXCountChars", wintypes.DWORD),
        ("dwYCountChars", wintypes.DWORD),
        ("dwFillAttribute", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("wShowWindow", wintypes.WORD),
        ("cbReserved2", wintypes.WORD),
        ("lpReserved2", ctypes.c_void_p),
        ("hStdInput", wintypes.HANDLE),
        ("hStdOutput", wintypes.HANDLE),
        ("hStdError", wintypes.HANDLE),
    ]


class _PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("hProcess", wintypes.HANDLE),
        ("hThread", wintypes.HANDLE),
        ("dwProcessId", wintypes.DWORD),
        ("dwThreadId", wintypes.DWORD),
    ]


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_uint64)
        for name in (
            "ReadOperationCount",
            "WriteOperationCount",
            "OtherOperationCount",
            "ReadTransferCount",
            "WriteTransferCount",
            "OtherTransferCount",
        )
    ]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", _IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _FILETIME(ctypes.Structure):
    _fields_ = [("lo", wintypes.DWORD), ("hi", wintypes.DWORD)]


class _WindowsOwnedJob:
    """Single-use private Job Object and direct process handles, no PID killing."""

    def __init__(self):
        if sys.platform != "win32":
            raise SupervisorRefused("WINDOWS_JOB_OBJECT_REQUIRED")
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.job = None
        self.process = None
        self.thread = None
        self.pid = None
        self.birth = None
        self.active = False
        self.member_births = {}

    def _call(self, name: str, *args, result_type=wintypes.BOOL, types=None):
        api = getattr(self.kernel, name)
        api.restype = result_type
        if types is not None:
            api.argtypes = types
        result = api(*args)
        if not result:
            raise SupervisorRefused(f"WINDOWS_{name}_FAILED_{ctypes.get_last_error()}")
        return result

    def _close(self, handle) -> None:
        if handle:
            self._call("CloseHandle", handle, types=[wintypes.HANDLE])

    def start(self, command: list[str], cwd: Path):
        if self.active or self.job or self.process or self.thread:
            raise SupervisorRefused("JOB_OWNER_ALREADY_USED")
        try:
            self.job = self._call(
                "CreateJobObjectW",
                None,
                None,
                result_type=wintypes.HANDLE,
                types=[ctypes.c_void_p, wintypes.LPCWSTR],
            )
            policy = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            policy.BasicLimitInformation.LimitFlags = (
                0x2000 | 0x0008
            )  # kill on close + active limit
            policy.BasicLimitInformation.ActiveProcessLimit = MAX_JOB_PROCESSES
            self._call(
                "SetInformationJobObject",
                self.job,
                9,
                ctypes.byref(policy),
                ctypes.sizeof(policy),
                types=[wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD],
            )
            startup = _STARTUPINFO()
            startup.cb = ctypes.sizeof(startup)
            process = _PROCESS_INFORMATION()
            cmdline = ctypes.create_unicode_buffer(subprocess.list2cmdline(command))
            # Suspended process cannot spawn server/evaluator before the private
            # Job Object is securely assigned. Never use CREATE_BREAKAWAY_FROM_JOB.
            self._call(
                "CreateProcessW",
                str(Path(command[0]).resolve()),
                cmdline,
                None,
                None,
                False,
                0x00000004 | 0x08000000,
                None,
                str(cwd),
                ctypes.byref(startup),
                ctypes.byref(process),
                types=[
                    wintypes.LPCWSTR,
                    wintypes.LPWSTR,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    wintypes.BOOL,
                    wintypes.DWORD,
                    ctypes.c_void_p,
                    wintypes.LPCWSTR,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                ],
            )
            self.process, self.thread, self.pid = (
                process.hProcess,
                process.hThread,
                int(process.dwProcessId),
            )
            self._call(
                "AssignProcessToJobObject",
                self.job,
                self.process,
                types=[wintypes.HANDLE, wintypes.HANDLE],
            )
            self.birth = self._handle_create_time(self.process)
            observed = psutil.Process(self.pid)
            if abs(observed.create_time() - self.birth) > 2:
                raise SupervisorRefused("CREATED_SUSPENDED_PID_START_IDENTITY_MISMATCH")
            self._call(
                "ResumeThread",
                self.thread,
                result_type=wintypes.DWORD,
                types=[wintypes.HANDLE],
            )
            self._close(self.thread)
            self.thread = None
            self.active = True
            return self
        except Exception:
            # Child was suspended and is a direct, locally created process
            # HANDLE: if job assignment failed, terminate that handle only.
            if self.process:
                try:
                    self._call(
                        "TerminateProcess",
                        self.process,
                        127,
                        types=[wintypes.HANDLE, wintypes.UINT],
                    )
                except Exception:
                    pass
            self.close()
            raise

    def _handle_create_time(self, handle) -> float:
        create, exit_time, kernel, user = (_FILETIME() for _ in range(4))
        self._call(
            "GetProcessTimes",
            handle,
            ctypes.byref(create),
            ctypes.byref(exit_time),
            ctypes.byref(kernel),
            ctypes.byref(user),
            types=[wintypes.HANDLE] + [ctypes.POINTER(_FILETIME)] * 4,
        )
        return round(((int(create.hi) << 32) + int(create.lo)) / 10**7 - 11644473600, 6)

    def owner_valid(self) -> bool:
        if not self.active or not self.process or not self.pid:
            return False
        try:
            return (
                abs(self._handle_create_time(self.process) - self.birth) < 0.002
                and abs(psutil.Process(self.pid).create_time() - self.birth) < 2
            )
        except (psutil.Error, OSError, SupervisorRefused):
            return False

    def member_ids(self) -> list[int]:
        if not self.active:
            raise SupervisorRefused("PRIVATE_JOB_NOT_ACTIVE")
        capacity = 256
        buf = ctypes.create_string_buffer(8 + capacity * ctypes.sizeof(ctypes.c_size_t))
        self._call(
            "QueryInformationJobObject",
            self.job,
            3,
            buf,
            ctypes.sizeof(buf),
            None,
            types=[
                wintypes.HANDLE,
                ctypes.c_int,
                ctypes.c_void_p,
                wintypes.DWORD,
                ctypes.c_void_p,
            ],
        )
        assigned = int.from_bytes(buf.raw[0:4], "little")
        in_list = int.from_bytes(buf.raw[4:8], "little")
        if assigned > MAX_JOB_PROCESSES or in_list > assigned or in_list > capacity:
            raise SupervisorRefused("OWNED_JOB_PROCESS_LIMIT_OR_CENSUS_INVALID")
        psize = ctypes.sizeof(ctypes.c_size_t)
        return [
            int.from_bytes(buf.raw[8 + i * psize : 8 + (i + 1) * psize], "little")
            for i in range(in_list)
        ]

    def verify_member_births(self, members: list[int]) -> bool:
        """No job-member PID may change create-time within this observation."""
        if not self.active:
            return False
        try:
            for pid in members:
                if not self.is_member(pid):
                    return False
                when = psutil.Process(pid).create_time()
                if not math.isfinite(when) or (
                    pid in self.member_births and self.member_births[pid] != when
                ):
                    return False
                self.member_births[pid] = when
            return True
        except (psutil.Error, OSError, ValueError):
            return False

    def is_member(self, pid: int) -> bool:
        if not self.active or not _integer(pid) or pid == 0:
            return False
        handle = None
        try:
            handle = self._call(
                "OpenProcess",
                0x1000,
                False,
                pid,
                result_type=wintypes.HANDLE,
                types=[wintypes.DWORD, wintypes.BOOL, wintypes.DWORD],
            )
            belongs = wintypes.BOOL()
            self._call(
                "IsProcessInJob",
                handle,
                self.job,
                ctypes.byref(belongs),
                types=[wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)],
            )
            return bool(belongs.value)
        except (OSError, SupervisorRefused):
            return False
        finally:
            if handle:
                self._close(handle)

    def exit_code(self) -> int | None:
        if not self.active:
            raise SupervisorRefused("PRIVATE_JOB_NOT_ACTIVE")
        code = wintypes.DWORD()
        self._call(
            "GetExitCodeProcess",
            self.process,
            ctypes.byref(code),
            types=[wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)],
        )
        return None if code.value == 259 else int(code.value)

    def terminate_owned(self):
        if not self.active or not self.job or not self.process:
            raise SupervisorRefused("JOB_OWNER_IDENTITY_NOT_ESTABLISHED")
        self._call(
            "TerminateJobObject", self.job, 137, types=[wintypes.HANDLE, wintypes.UINT]
        )

    def close(self):
        self.active = False
        for name in ("thread", "process", "job"):
            handle = getattr(self, name)
            if handle:
                try:
                    self._close(handle)
                finally:
                    setattr(self, name, None)


def cancel_owned_job(owner) -> None:
    """No PID argument: termination scope is private owned Job handle only."""
    if getattr(owner, "active", None) is not True:
        raise SupervisorRefused("CANNOT_CANCEL_UNBOUND_OR_FOREIGN_PROCESS")
    owner.terminate_owned()


def _watch_sample(
    owner: _WindowsOwnedJob,
    *,
    output: Path,
    arm: str,
    port: int,
    gpu_uuid: str,
    started: float,
) -> dict:
    now = time.monotonic()
    memory = native.read_windows_memory()
    gpu = native.read_gpu()
    members = owner.member_ids()
    process_identified = bool(
        members and owner.pid in members and owner.verify_member_births(members)
    )
    census = native.read_active_processes_and_ports(port)
    foreign = [
        c
        for c in census["listeners"]
        if c["pid"] is None or not owner.is_member(c["pid"])
    ]
    owns_listener = any(
        c["port"] == port and c["pid"] is not None and owner.is_member(c["pid"])
        for c in census["listeners"]
    )
    foreign_processes = [
        p for p in census["active_conflicts"] if not owner.is_member(p["pid"])
    ]
    log_path = output / arm / "server.log"
    log_bytes = 0
    residence = None
    if log_path.is_symlink():
        raise SupervisorRefused("MODEL_LOG_SYMLINK_FORBIDDEN")
    if log_path.is_file():
        log_bytes = log_path.stat().st_size
        if log_bytes <= MAX_LOG_BYTES:
            with log_path.open("rb") as source:
                prefix = source.read(MAX_LOG_PREFIX_BYTES)
            residence = parse_native_residency(prefix.decode("utf-8", errors="replace"))
    return {
        **memory,
        "gpu": gpu,
        "gpu_source_identity_matches_preflight": gpu.get("uuid") == gpu_uuid,
        "disk_free_bytes": shutil.disk_usage(output.parent).free,
        "root_owner_valid": owner.owner_valid(),
        "job_processes_valid": process_identified,
        "job_process_count": len(members),
        "foreign_ports": foreign,
        "foreign_processes": foreign_processes,
        "owned_listener": owns_listener,
        "native_started": residence is not None,
        "native_residency": residence,
        "model_log_bytes": log_bytes,
        "startup_seconds": now - started,
        "elapsed_seconds": now - started,
        "sampled_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _run_authorized(
    *,
    study: wave.Study,
    model: Path,
    server: Path,
    output: Path,
    receipt_dir: Path,
    port: int,
    poll_seconds: float,
    max_runtime_seconds: int,
    startup_grace_seconds: int,
    owner_factory=_WindowsOwnedJob,
) -> dict:
    if sys.platform != "win32":
        raise SupervisorRefused("NATIVE_WINDOWS_JOB_REQUIRED")
    if (
        not _number(poll_seconds)
        or not 1 <= poll_seconds <= 30
        or not _integer(max_runtime_seconds)
        or not 30 <= max_runtime_seconds <= MAX_WALL_SECONDS
        or not _integer(startup_grace_seconds)
        or not 30 <= startup_grace_seconds <= MAX_STARTUP_GRACE_SECONDS
    ):
        raise SupervisorRefused("UNBOUNDED_RUNTIME_OR_SAMPLE_INTERVAL")
    wave.assert_frozen_code_identity()
    if study != wave.study_for(study.arm):
        raise SupervisorRefused("FROZEN_STUDY_DRIFT")
    model = Path(model)
    server = Path(server)
    output = Path(output)
    receipt_dir = Path(receipt_dir)
    if (
        not model.is_absolute()
        or model.is_symlink()
        or not model.is_file()
        or not server.is_absolute()
        or server.is_symlink()
        or not server.is_file()
        or not output.is_absolute()
        or not output.parent.is_dir()
        or output.parent.is_symlink()
        or not receipt_dir.is_absolute()
        or not receipt_dir.is_dir()
        or receipt_dir.is_symlink()
        or not 1024 <= port <= 65535
    ):
        raise SupervisorRefused("UNSAFE_FROZEN_PATH_OR_PORT")
    wave.validate_fresh_output(output, study.arm)
    start_receipt = receipt_dir / (study.arm + ".owner-start.json")
    terminal_receipt = receipt_dir / (study.arm + ".owner-terminal.json")
    if (
        start_receipt.exists()
        or start_receipt.is_symlink()
        or terminal_receipt.exists()
        or terminal_receipt.is_symlink()
    ):
        raise SupervisorRefused("OWNER_RECEIPT_ALREADY_EXISTS")
    cmd = wave.command(study, model=model, output=output, server=server, port=port)
    validate_frozen_command(cmd, study, model, server, output, port)
    native_arm = wave.NATIVE_ADMISSION_ARM[study.arm]
    snap = native.snapshot(
        model_path=model,
        server_path=server,
        port=port,
        arm=native_arm,
        disk_root=output.parent,
    )
    decision = native.assess(snap, native_arm, port)
    if (
        decision["status"] != "READY_FOR_CAPACITY_PROBE_ONLY"
        or snap.get("model_full_sha256") != study.model_sha256
        or snap.get("server_full_sha256") is None
    ):
        raise SupervisorRefused(
            "DEDICATED_HOST_NOT_INDEPENDENTLY_ADMITTED:" + ",".join(decision["reasons"])
        )
    # Refresh live capacity, port absence, and physical GPU identity after
    # potentially expensive full-model SHA; no unowned process is stopped.
    latest = {
        **snap,
        **native.read_windows_memory(),
        "gpu": native.read_gpu(),
        **native.read_active_processes_and_ports(port),
        "disk_free_bytes": shutil.disk_usage(output.parent).free,
    }
    if (
        native.assess(latest, native_arm, port)["status"]
        != "READY_FOR_CAPACITY_PROBE_ONLY"
    ):
        raise SupervisorRefused("DEDICATED_ADMISSION_CHANGED_BEFORE_SPAWN")
    source_lock = verify_frozen_sources_offline(
        receipt_path=receipt_dir / (study.arm + ".original-offline-source.json")
    )
    born_stat = (
        model.stat().st_size,
        model.stat().st_mtime_ns,
        server.stat().st_size,
        server.stat().st_mtime_ns,
    )
    # Publishing the start intent is exclusive and durable BEFORE creation.
    write_receipt_exclusive(
        start_receipt,
        {
            "schema": SCHEMA,
            "status": "PRELAUNCH_SOURCE_AND_CAPACITY_VERIFIED",
            "arm": study.arm,
            "model_sha256": study.model_sha256,
            "native_server_sha256": snap["server_full_sha256"],
            "original_offline_source_proof": source_lock,
            "command": cmd,
            "native_admission": decision,
            "prelaunch_gpu_uuid": latest["gpu"]["uuid"],
            "no_process_created_at_receipt_time": True,
            "gen0": dict(native.GEN0_BLOCKED),
        },
    )
    owner = owner_factory()
    samples = []
    breached = []
    watched_fit = False
    code = None
    t0 = time.monotonic()
    try:
        if born_stat != (
            model.stat().st_size,
            model.stat().st_mtime_ns,
            server.stat().st_size,
            server.stat().st_mtime_ns,
        ):
            raise SupervisorRefused("FROZEN_BINARY_CHANGED_BEFORE_LAUNCH")
        owner.start(cmd, wave.REPO)
        while True:
            code = owner.exit_code()
            if code is not None:
                if code != 0 or not watched_fit:
                    breached = ["CHILD_NONZERO_OR_NATIVE_FIT_UNWITNESSED"]
                cancel_owned_job(owner)
                break
            observation = _watch_sample(
                owner,
                output=output,
                arm=study.arm,
                port=port,
                gpu_uuid=latest["gpu"]["uuid"],
                started=t0,
            )
            faults = validate_sample(
                observation,
                startup_grace_seconds=startup_grace_seconds,
                max_runtime_seconds=max_runtime_seconds,
            )
            if observation["gpu_source_identity_matches_preflight"] is not True:
                faults.append("PHYSICAL_GPU_CHANGED")
            if watched_fit and observation["owned_listener"] is not True:
                faults.append("OWNED_LISTENER_LOST")
            if faults:
                breached = sorted(set(faults))
                cancel_owned_job(owner)
                break
            watched_fit |= bool(
                observation["native_started"] and observation["owned_listener"]
            )
            if len(samples) >= MAX_SAMPLES:
                breached = ["SAMPLE_LIMIT"]
                cancel_owned_job(owner)
                break
            samples.append(
                {
                    "seconds": round(observation["elapsed_seconds"], 3),
                    "physical_available_bytes": observation["physical_available_bytes"],
                    "commit_headroom_bytes": observation["commit_headroom_bytes"],
                    "gpu_free_mib": observation["gpu"]["free_mib"],
                    "gpu_uuid": observation["gpu"]["uuid"],
                    "native_residency": observation["native_residency"],
                    "owned_listener": observation["owned_listener"],
                    "model_log_bytes": observation["model_log_bytes"],
                    "job_process_count": observation["job_process_count"],
                }
            )
            time.sleep(poll_seconds)
    except Exception as exc:
        breached = sorted(set(breached + ["SUPERVISOR_EXCEPTION", type(exc).__name__]))
        if owner.active:
            try:
                cancel_owned_job(owner)
            except Exception:
                breached.append("JOB_CANCEL_FAILED")
    finally:
        owner.close()
    result = {
        "schema": SCHEMA,
        "arm": study.arm,
        "child_pid": owner.pid,
        "child_create_time_unix": owner.birth,
        "status": (
            "CHILD_EXIT_ZERO_RUNTIME_WITNESSED_ONLY"
            if code == 0 and watched_fit and not breached
            else "REFUSED_RUNTIME_ABORTED"
        ),
        "breaches": breached,
        "exit_code": code,
        "real_native_child_started": owner.pid is not None,
        "owned_job_handle_closed_with_kill_on_close": True,
        "post_close_descendant_extinction_independently_witnessed": False,
        "observations": samples,
        "fully_witnessed_native_fit": watched_fit,
        "model_sha256": study.model_sha256,
        "original_offline_source_proof": source_lock,
        "scientific_gate": "UNKNOWN",
        "speed_gate": "UNKNOWN",
        "gen0": dict(native.GEN0_BLOCKED),
    }
    write_receipt_exclusive(terminal_receipt, result)
    return result


def run_frozen_arm(
    *,
    study: wave.Study,
    model: Path,
    server: Path,
    output: Path,
    receipt_dir: Path,
    port: int = 8790,
    execute_authorized: bool = False,
    poll_seconds: float = 3,
    max_runtime_seconds: int = MAX_WALL_SECONDS,
    startup_grace_seconds: int = MAX_STARTUP_GRACE_SECONDS,
) -> dict:
    """Future prime integration API. Defaults to refusing before any I/O."""
    if execute_authorized is not True:
        raise SupervisorRefused("EXPLICIT_EXECUTION_AUTHORIZATION_REQUIRED")
    return _run_authorized(
        study=study,
        model=model,
        server=server,
        output=output,
        receipt_dir=receipt_dir,
        port=port,
        poll_seconds=poll_seconds,
        max_runtime_seconds=max_runtime_seconds,
        startup_grace_seconds=startup_grace_seconds,
    )
