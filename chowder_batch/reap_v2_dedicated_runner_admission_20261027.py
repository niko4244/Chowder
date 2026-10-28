"""Read-only native Windows resource admission for a dedicated REAP v2 runner.

The frozen GGUF identities come from
docs/quals/reap-v2-gen0-integrated-scientific-state-20261026.md; capacity
guidance and historical failed dense loading from
docs/quals/reap-v2-full-q4-kaggle-independent-audit-20261026.md.

An eligible observation authorizes only further supervised capacity/fit
investigation. It is never authority to start inference, modify services,
promote scientific gates, or replay completed original evaluations.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import psutil


GIB = 1024**3
FROZEN_FLAGS = ("--fit", "on", "-np", "4", "-c", "16384", "--jinja")
SCHEMA = "chowder.reap-v2-dedicated-windows-runner-admission.v1"
RAM_FREE_FLOOR = 48 * GIB
COMMIT_FREE_FLOOR = 48 * GIB
RAM_RUNTIME_FLOOR = 7 * GIB
COMMIT_RUNTIME_FLOOR = 8 * GIB
DISK_FREE_FLOOR = 8 * GIB
GPU_NATIVE_START_MIB = 12_288
GPU_CLASS_TOTAL_MIB = 48 * 1024
# The free-VRAM provision below is a conservative dedicated-runner proposal,
# not a frozen scientific threshold or a guarantee that BF16 will fit.
GPU_DEDICATED_FREE_MIB = 40 * 1024
GPU_IDLE_USED_MIB = 1024
GPU_IDLE_UTILIZATION = 10
REGISTERED_PORTS = frozenset({8790, 8890, 8892, 8894})
GEN0_BLOCKED = {"status": "BLOCKED", "chosen": None, "release_approved": False}


@dataclass(frozen=True)
class ModelSpec:
    arm: str
    original_path: str
    bytes: int
    sha256: str


FROZEN_MODELS = {
    "bf16_gguf": ModelSpec(
        "bf16_gguf",
        "F:/reap128-v2-bf16-intermediate-20261025.gguf",
        37_122_439_488,
        "80341aa776ad9b5f937c21b081b6d76de4a2ea277e5fb88c9ae2634a9da8f4c6",
    ),
    "q8": ModelSpec(
        "q8",
        "I:/reap128-v2-q8_0-20261025.gguf",
        19_748_435_968,
        "89dcaf33b61f59c83a31b718ba98b73c33437d0320ae0cc56737a5d7e43519c6",
    ),
    "q4": ModelSpec(
        "q4",
        "G:/reap128-v2-q4_k_m-20261025.gguf",
        11_563_915_264,
        "2d64b786461b99a335fdbbdf7c4d3a6758b21409bce755751714ab7cd8f63bf7",
    ),
    "dense_q4": ModelSpec(
        "dense_q4",
        "G:/llm-cold/gguf/dense-q4_k_m.gguf",
        21_166_757_984,
        "2c6d9466129e8802839566b35728e2506aaf1e8705cd137009e2f08feb9dd049",
    ),
    "dense_q8": ModelSpec(
        "dense_q8",
        "G:/llm-cold/gguf/dense-q8_0.gguf",
        36_903_139_424,
        "2ff57a74a36939c5baabf40a0d3eb06d4424ecc94cf7d4ef04b3ac02059097c2",
    ),
    "random128_q8": ModelSpec(
        "random128_q8",
        "G:/llm-cold/gguf/random128-q8_0.gguf",
        19_748_435_872,
        "601871e00403658708eb4eebf3b05f9dc54a93b326e31f2f87b48cb59f92241b",
    ),
}


class _PerformanceInformation(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("CommitTotal", ctypes.c_size_t),
        ("CommitLimit", ctypes.c_size_t),
        ("CommitPeak", ctypes.c_size_t),
        ("PhysicalTotal", ctypes.c_size_t),
        ("PhysicalAvailable", ctypes.c_size_t),
        ("SystemCache", ctypes.c_size_t),
        ("KernelTotal", ctypes.c_size_t),
        ("KernelPaged", ctypes.c_size_t),
        ("KernelNonpaged", ctypes.c_size_t),
        ("PageSize", ctypes.c_size_t),
        ("HandleCount", ctypes.c_ulong),
        ("ProcessCount", ctypes.c_ulong),
        ("ThreadCount", ctypes.c_ulong),
    ]


def read_windows_memory() -> dict:
    """GetPerformanceInfo reports committed charge, not virtual address space."""
    if sys.platform != "win32":
        raise RuntimeError("WINDOWS_REQUIRED")
    perf = _PerformanceInformation()
    perf.cb = ctypes.sizeof(perf)
    api = ctypes.WinDLL("psapi", use_last_error=True).GetPerformanceInfo
    api.argtypes = [ctypes.POINTER(_PerformanceInformation), ctypes.c_ulong]
    api.restype = ctypes.c_int
    if not api(ctypes.byref(perf), perf.cb):
        raise RuntimeError("WINDOWS_GETPERFORMANCEINFO_FAILED")
    page = int(perf.PageSize)
    if not (
        1024 <= page <= 65536
        and 0 <= perf.CommitTotal <= perf.CommitLimit
        and 0 <= perf.PhysicalAvailable <= perf.PhysicalTotal
    ):
        raise RuntimeError("WINDOWS_MEMORY_COUNTERS_INVALID")
    return {
        "physical_total_bytes": int(perf.PhysicalTotal) * page,
        "physical_available_bytes": int(perf.PhysicalAvailable) * page,
        "commit_total_bytes": int(perf.CommitTotal) * page,
        "commit_limit_bytes": int(perf.CommitLimit) * page,
        "commit_headroom_bytes": (int(perf.CommitLimit) - int(perf.CommitTotal)) * page,
    }


def _integer(v: object, *, positive: bool = False) -> bool:
    return type(v) is int and v >= (1 if positive else 0)


def parse_gpu(raw: str) -> dict:
    """Only one physical CUDA GPU; no WDDM virtual-budget substitution."""
    if not isinstance(raw, str):
        raise ValueError("GPU_COUNTERS_INVALID")
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError("GPU_COUNT_UNKNOWN_OR_MULTIPLE")
    fields = [field.strip() for field in lines[0].split(",")]
    if (
        len(fields) != 6
        or not re.fullmatch(r"GPU-[a-zA-Z0-9-]{8,64}", fields[0])
        or not fields[1]
    ):
        raise ValueError("GPU_IDENTITY_INVALID")
    try:
        numbers = list(map(int, fields[2:]))
    except ValueError as exc:
        raise ValueError("GPU_COUNTERS_INVALID") from exc
    total, used, free, busy = numbers
    reserved = total - used - free
    if (
        total < 12000
        or used < 0
        or free < 0
        or not 0 <= reserved <= 2048
        or not 0 <= busy <= 100
    ):
        raise ValueError("GPU_COUNTERS_INVALID")
    return {
        "uuid": fields[0],
        "name": fields[1],
        "total_mib": total,
        "used_mib": used,
        "free_mib": free,
        "reserved_mib": reserved,
        "utilization_pct": busy,
    }


def read_gpu() -> dict:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    if result.returncode:
        raise RuntimeError("NVIDIA_NATIVE_COUNTERS_UNAVAILABLE")
    return parse_gpu(result.stdout)


def _is_windows_secure_system(pid: int) -> bool:
    """Confirm the known blank-name Windows Secure System via native CIM."""
    if sys.platform != "win32" or not _integer(pid, positive=True):
        return False
    # The integer PID originates in psutil, never from a shell argument.
    script = (
        f"$p=Get-CimInstance Win32_Process -Filter 'ProcessId = {pid}'; "
        "if($p.Name -eq 'Secure System' -and $p.ParentProcessId -eq 4)"
        "{'SECURE_SYSTEM'}"
    )
    try:
        query = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return query.returncode == 0 and query.stdout.strip() == "SECURE_SYSTEM"


def read_active_processes_and_ports(port: int) -> dict:
    """Unknown/inaccessible processes or listeners refuse; never stop owners."""
    if not _integer(port, positive=True) or port > 65535:
        raise ValueError("PORT_INVALID")
    conflicts = []
    try:
        for process in psutil.process_iter(("pid", "name", "cmdline", "create_time")):
            pid = process.info["pid"]
            if pid == os.getpid():
                continue
            name = str(process.info.get("name") or "").lower()
            if not name:
                if process.info.get("cmdline") is None and _is_windows_secure_system(
                    pid
                ):
                    continue
                raise RuntimeError("PROCESS_IDENTITY_UNAVAILABLE")
            argv = process.info.get("cmdline")
            if argv is None and name.startswith("python"):
                raise RuntimeError("PYTHON_PROCESS_COMMANDLINE_UNAVAILABLE")
            command = " ".join(str(arg).lower() for arg in (argv or []))
            named_model = any(
                fragment in name
                for fragment in (
                    "llama-server",
                    "ollama",
                    "vllm",
                    "kobold",
                    "text-generation",
                    "podman",
                )
            )
            python_model = name.startswith("python") and any(
                term in command
                for term in (
                    "livebench",
                    "reap_",
                    "reap-",
                    "llama-server",
                    "qwen",
                    "automationbench",
                )
            )
            if named_model or python_model:
                conflicts.append(
                    {
                        "pid": pid,
                        "name": name,
                        "create_time": process.info["create_time"],
                    }
                )
        occupied = []
        for connection in psutil.net_connections(kind="tcp"):
            if (
                connection.status == psutil.CONN_LISTEN
                and connection.laddr
                and connection.laddr.port in REGISTERED_PORTS | {port}
            ):
                occupied.append({"port": connection.laddr.port, "pid": connection.pid})
    except (psutil.Error, OSError, ValueError, TypeError) as exc:
        raise RuntimeError("PROCESS_OR_PORT_CENSUS_UNAVAILABLE") from exc
    return {
        "active_conflicts": sorted(conflicts, key=lambda v: v["pid"]),
        "listeners": sorted(occupied, key=lambda v: (v["port"], v["pid"] or 0)),
        "process_census_complete": True,
        "port_census_complete": True,
    }


def verify_file(path: Path, expected_bytes: int, expected_sha256: str) -> dict:
    """Full-byte SHA and pre/post stat identity; external model paths are supported."""
    path = Path(path)
    if (
        not path.is_absolute()
        or path.is_symlink()
        or not path.is_file()
        or not _integer(expected_bytes, positive=True)
        or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
    ):
        raise ValueError("FILE_IDENTITY_OR_EXPECTED_SOURCE_INVALID")
    before = path.stat()
    if before.st_size != expected_bytes:
        raise ValueError("FILE_SIZE_MISMATCH")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_mtime_ns, before.st_size) != (
        after.st_dev,
        after.st_ino,
        after.st_mtime_ns,
        after.st_size,
    ):
        raise ValueError("FILE_CHANGED_DURING_HASH")
    if digest.hexdigest() != expected_sha256:
        raise ValueError("FILE_SHA256_MISMATCH")
    return {
        "sha256": digest.hexdigest(),
        "size_bytes": expected_bytes,
        "sha256_verified": True,
        "size_verified": True,
        "mtime_ns": after.st_mtime_ns,
    }


def assess(observation: dict, arm: str, port: int = 8790) -> dict:
    """Pure, fail-closed decision; READY means observational probe only."""
    problems = []
    if type(arm) is not str or arm not in FROZEN_MODELS:
        problems.append("UNKNOWN_FROZEN_ARM")
    if not _integer(port, positive=True) or port > 65535:
        problems.append("PORT_INVALID")
    if not isinstance(observation, dict):
        observation = {}
    for field, floor, reason in (
        ("physical_available_bytes", RAM_FREE_FLOOR, "RAM_48G_FREE"),
        ("commit_headroom_bytes", COMMIT_FREE_FLOOR, "COMMIT_48G_FREE"),
        ("disk_free_bytes", DISK_FREE_FLOOR, "DISK_8G_FREE"),
    ):
        value = observation.get(field)
        if not _integer(value) or value < floor:
            problems.append(reason)
    total = observation.get("physical_total_bytes")
    avail = observation.get("physical_available_bytes")
    charged = observation.get("commit_total_bytes")
    limit = observation.get("commit_limit_bytes")
    headroom = observation.get("commit_headroom_bytes")
    if (
        not all(_integer(v) for v in (total, avail, charged, limit, headroom))
        or avail > total
        or charged > limit
        or headroom != limit - charged
    ):
        problems.append("WINDOWS_MEMORY_COUNTERS")
    for field, reason in (
        ("model_sha256_verified", "MODEL_SHA"),
        ("model_bytes_verified", "MODEL_BYTES"),
        ("native_binary_sha256_verified", "SERVER_SHA"),
        ("process_census_complete", "PROCESS_CENSUS"),
        ("port_census_complete", "PORT_CENSUS"),
    ):
        if observation.get(field) is not True:
            problems.append(reason)
    for field, reason in (
        ("active_conflicts", "ACTIVE_PROCESSES"),
        ("listeners", "ACTIVE_LISTENERS"),
    ):
        value = observation.get(field)
        if not isinstance(value, list) or len(value):
            problems.append(reason)
    gpu = observation.get("gpu")
    if not isinstance(gpu, dict):
        gpu = {}
    values = [
        gpu.get(k)
        for k in (
            "total_mib",
            "used_mib",
            "free_mib",
            "reserved_mib",
            "utilization_pct",
        )
    ]
    if (
        not all(_integer(v) for v in values)
        or not isinstance(gpu.get("uuid"), str)
        or not re.fullmatch(r"GPU-[a-zA-Z0-9-]{8,64}", gpu.get("uuid", ""))
        or not isinstance(gpu.get("name"), str)
        or not gpu.get("name")
        or values[1] + values[2] + values[3] != values[0]
        or values[3] > 2048
        or values[4] > 100
    ):
        problems.append("GPU_COUNTERS")
    else:
        total_gpu, used, free, _, busy = values
        if total_gpu < GPU_CLASS_TOTAL_MIB:
            problems.append("GPU_48G_CLASS")
        if free < GPU_NATIVE_START_MIB:
            problems.append("GPU_12G_NATIVE_FLOOR")
        if free < GPU_DEDICATED_FREE_MIB:
            problems.append("GPU_40G_DEDICATED")
        if used > GPU_IDLE_USED_MIB or busy > GPU_IDLE_UTILIZATION:
            problems.append("GPU_NOT_IDLE")
    return {
        "status": "REFUSED" if problems else "READY_FOR_CAPACITY_PROBE_ONLY",
        "reasons": sorted(set(problems)),
        "execution_approved": False,
        "frozen_flags": list(FROZEN_FLAGS),
        "gen0": dict(GEN0_BLOCKED),
        "floors": {
            "physical_available_bytes": RAM_FREE_FLOOR,
            "commit_headroom_bytes": COMMIT_FREE_FLOOR,
            "runtime_ram_floor_bytes": RAM_RUNTIME_FLOOR,
            "runtime_commit_floor_bytes": COMMIT_RUNTIME_FLOOR,
            "disk_free_bytes": DISK_FREE_FLOOR,
            "original_native_gpu_free_mib": GPU_NATIVE_START_MIB,
            "dedicated_gpu_total_mib": GPU_CLASS_TOTAL_MIB,
            "dedicated_gpu_free_mib": GPU_DEDICATED_FREE_MIB,
        },
        "capacity_is_not_measured_model_fit": True,
    }


admit = assess


def snapshot(
    *,
    model_path: Path,
    server_path: Path,
    port: int = 8790,
    arm: str = "q4",
    disk_root: Path | None = None,
) -> dict:
    """Read OS counters first; hash huge model files only if host is eligible."""
    if sys.platform != "win32":
        raise RuntimeError("WINDOWS_REQUIRED")
    if type(arm) is not str or arm not in FROZEN_MODELS:
        raise ValueError("UNKNOWN_FROZEN_ARM")
    first = read_active_processes_and_ports(port)
    memory = read_windows_memory()
    gpu = read_gpu()
    disk_path = Path(disk_root) if disk_root else Path(model_path).parent
    available_disk = shutil.disk_usage(disk_path).free
    result = {
        **memory,
        **first,
        "gpu": gpu,
        "disk_free_bytes": available_disk,
        "model_bytes_verified": False,
        "model_sha256_verified": False,
        "native_binary_sha256_verified": False,
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "model_full_sha256": None,
        "server_full_sha256": None,
    }
    # Skip many gigabytes of model hashing when the immutable host gates
    # already refuse the attempt. This may return no model-SHA evidence.
    pre = assess(result, arm, port)
    identity_only = {"MODEL_SHA", "MODEL_BYTES", "SERVER_SHA"}
    if any(reason not in identity_only for reason in pre["reasons"]):
        return result
    original = FROZEN_MODELS[arm]
    proof = verify_file(Path(model_path), original.bytes, original.sha256)
    result["model_bytes_verified"] = proof["size_verified"]
    result["model_sha256_verified"] = proof["sha256_verified"]
    result["model_full_sha256"] = proof["sha256"]
    native = Path(server_path)
    if not native.is_absolute() or native.is_symlink() or not native.is_file():
        raise ValueError("NATIVE_BINARY_MISSING_OR_UNSAFE")
    actual_stat = native.stat()
    native_proof = verify_file(native, actual_stat.st_size, _sha_native_binary(native))
    result["native_binary_sha256_verified"] = native_proof["sha256_verified"]
    result["server_full_sha256"] = native_proof["sha256"]
    # Additional fresh source-owned observation after potentially slow hashing.
    later = read_active_processes_and_ports(port)
    result.update(later)
    result.update(read_windows_memory())
    result["gpu"] = read_gpu()
    result["disk_free_bytes"] = shutil.disk_usage(disk_path).free
    result["time_utc"] = datetime.now(timezone.utc).isoformat()
    return result


def _sha_native_binary(native: Path) -> str:
    digest = hashlib.sha256()
    with native.open("rb") as stream:
        for piece in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(piece)
    return digest.hexdigest()


def make_manifest(
    arm: str, model_path: Path, server_path: Path, port: int, observation: dict
) -> dict:
    """Pure reproducible command plan; never executes the command."""
    if not isinstance(observation, dict):
        observation = {}
    verdict = assess(observation, arm, port)
    original = FROZEN_MODELS.get(arm) if type(arm) is str else None
    proposed = (
        [
            str(server_path),
            *FROZEN_FLAGS,
            "-m",
            str(model_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]
        if original and _integer(port, positive=True) and port <= 65535
        else None
    )
    return {
        "schema": SCHEMA,
        "platform_required": "Windows",
        "arm": arm,
        "original_model": (
            {
                "canonical_path": original.original_path,
                "selected_runner_path": str(model_path),
                "bytes": original.bytes,
                "sha256": original.sha256,
            }
            if original
            else None
        ),
        "native_binary": {
            "path": str(server_path),
            "observed_sha256": observation.get("server_full_sha256"),
            "historical_native_binary_source_pin_established": False,
        },
        "proposed_command": proposed,
        "registered_flags": list(FROZEN_FLAGS),
        "port": port,
        "observed": observation,
        "report": verdict,
        "never_launches_model": True,
        "recheck_before_any_run": True,
        "requires_independent_per_arm_postload_fit_and_numerical_evidence": True,
        "cgroup_free_not_measured_by_native_windows": True,
        "dedicated_gpu_free_40g_is_provisioning_not_frozen_scientific_gate": True,
    }


def write_exclusive(path: Path, payload: dict) -> None:
    target = Path(path)
    if not target.parent.is_dir() or target.is_symlink():
        raise ValueError("EXCLUSIVE_OUTPUT_PARENT_INVALID")
    with target.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main(argv: list[str] | None = None) -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--arm", choices=sorted(FROZEN_MODELS), required=True)
    cli.add_argument(
        "--model", type=Path, help="Copied original model; full pinned SHA required"
    )
    cli.add_argument(
        "--server", type=Path, default=Path("F:/llama.cpp/bin/llama-server.exe")
    )
    cli.add_argument("--port", type=int, default=8790)
    cli.add_argument(
        "--out", type=Path, required=True, help="New exclusive JSON on target host"
    )
    args = cli.parse_args(argv)
    model = args.model if args.model else Path(FROZEN_MODELS[args.arm].original_path)
    try:
        observed = snapshot(
            model_path=model,
            server_path=args.server,
            port=args.port,
            arm=args.arm,
            disk_root=args.out.parent,
        )
    except (
        OSError,
        RuntimeError,
        ValueError,
        subprocess.SubprocessError,
        psutil.Error,
    ) as exc:
        observed = {"observation_failed": type(exc).__name__ + ": " + str(exc)}
    manifest = make_manifest(args.arm, model, args.server, args.port, observed)
    write_exclusive(args.out, manifest)
    print(
        json.dumps(
            {
                "status": manifest["report"]["status"],
                "reasons": manifest["report"]["reasons"],
                "output": str(args.out),
            },
            sort_keys=True,
        )
    )
    # Even a positive result is a preflight receipt, not an approved launch.
    return 0 if manifest["report"]["status"] == "READY_FOR_CAPACITY_PROBE_ONLY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
