"""Fail-closed native continuation of the *original* guard.v3 research run.

This uses the original runner/guard/scorer bytes from Git commit 84ff970,
not later guard.v4 worktree code. It never changes a scored task or upgrades
one-slot research evidence into a frozen four-slot Gen-0 result. The default
is read-only; --execute is required to load the original model.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import ctypes
import json
import os
from pathlib import Path, PureWindowsPath
import re
import socket
import subprocess
import sys

from chowder_batch import reap_original600_guardv3_git_pinned_reader_20261027 as reader

SCHEMA = "chowder.reap-original600-guardv3-source-locked-resume.v1"
RUNNER_MODULE = "chowder_batch.reap_q4_automationbench_full600_resumable_20261027"
PORTS = (8892, 8894)
FILE_ATTRIBUTE_REPARSE_POINT = 0x400
NATIVE_LEASE_NAME = f"Global\\Chowder_Original600_guardv3_{reader.OWNER_SHA}"


@contextmanager
def exclusive_native_resume():
    """Serialize original guard.v3 native starts across independent wrappers.

    Free-port probes and GPU snapshots alone are not an atomic reservation:
    two concurrent wrappers can both pass them before either native server
    binds. Hold a named kernel mutex from the final admission check through
    the entire original runner lifetime, without writing to frozen evidence.
    An abandoned lease is refused, not silently treated as clean ownership.
    """
    if os.name != "nt":
        raise ValueError("original native lease requires Windows")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateMutexW.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p)
    kernel.CreateMutexW.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
    kernel.WaitForSingleObject.restype = ctypes.c_uint32
    kernel.ReleaseMutex.argtypes = (ctypes.c_void_p,)
    kernel.ReleaseMutex.restype = ctypes.c_int
    kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
    kernel.CloseHandle.restype = ctypes.c_int
    handle = kernel.CreateMutexW(None, False, NATIVE_LEASE_NAME)
    if not handle:
        raise ValueError("original guard.v3 native lease unavailable")
    acquired = False
    try:
        status = kernel.WaitForSingleObject(handle, 0)
        if status == 0:  # WAIT_OBJECT_0
            acquired = True
        elif status == 0x80:  # WAIT_ABANDONED: an earlier owner died
            acquired = True
            raise ValueError("original guard.v3 native lease abandoned")
        elif status == 0x102:  # WAIT_TIMEOUT
            raise ValueError("another original guard.v3 native resume owns the lease")
        else:
            raise ValueError("original guard.v3 native lease acquisition failed")
        yield
    finally:
        release_ok = not acquired or bool(kernel.ReleaseMutex(handle))
        close_ok = bool(kernel.CloseHandle(handle))
        if not release_ok or not close_ok:
            raise ValueError("original guard.v3 native lease cleanup failed")


def assert_native_ordinary(path: Path) -> None:
    """Reject NTFS junctions/mount reparse points, not merely Python symlinks."""
    attributes = ctypes.windll.kernel32.GetFileAttributesW(str(path))
    if attributes in (-1, 0xFFFFFFFF):
        raise OSError("cannot inspect native filesystem attributes")
    if attributes & FILE_ATTRIBUTE_REPARSE_POINT or path.is_symlink():
        raise ValueError("native reparse path in original checkpoint")


def competing_model_processes() -> list[dict[str, object]]:
    """Refuse a second model load when *any* local native model server exists.

    An unreadable process inventory is not evidence of GPU exclusivity.
    The separate original runner still enforces native RAM/commit/VRAM floors.
    """
    import psutil

    competing = []
    try:
        for process in psutil.process_iter(attrs=["pid", "name", "cmdline"]):
            try:
                info = process.info
                if info["pid"] == os.getpid():
                    continue
                # psutil's default ad_value is None: an AccessDenied for one
                # requested attribute can be silently substituted instead of
                # raising. An unreadable owner is not proof of GPU exclusivity.
                if info.get("name") is None or info.get("cmdline") is None:
                    raise ValueError("native process inventory contains unreadable owner")
                name = str(info.get("name") or "").lower()
                argv = [str(x).lower() for x in (info.get("cmdline") or [])]
                # Shell commands can *mention* llama-server in their text;
                # that does not make bash/pwsh a native model server.
                python_process = name.startswith("python")
                vllm_python = (python_process and
                               (any("vllm.entrypoints" in x for x in argv)
                                or any(argv[i:i + 2] == ["-m", "vllm"]
                                       for i in range(len(argv) - 1))))
                # An unrelated GPU evaluation can use Python directly, without
                # a llama-server or vLLM executable. In particular, the Spark
                # harness is an active CUDA owner even when ports 8892/8894
                # are free. Ollama's parent `serve` process is also an owner;
                # `ollama list` and shell text mentioning these names are not.
                spark_python = (python_process and any(
                    x.replace("\\", "/").endswith("harness/run_eval.py")
                    for x in argv[1:]))
                ollama_serve = (name in ("ollama.exe", "ollama") and
                                "serve" in argv[1:])
                # WDDM's compute-app inventory may omit GPU work performed
                # inside WSL2/Podman. Treat a live VM/container host as an
                # unverified competing owner until separately source-proven
                # isolated; a harmless one-shot `podman ps` is not a server.
                container_vm = name in (
                    "vmmemwsl.exe", "vmmem.exe", "wslhost.exe", "vmwp.exe",
                )
                podman_active = (name in ("podman.exe", "podman") and
                                 any(arg in ("run", "start", "exec", "machine")
                                     for arg in argv[1:]))
                if ("llama-server" in name or "llama_server" in name
                        or vllm_python or spark_python or ollama_serve
                        or container_vm or podman_active):
                    competing.append({"pid": info["pid"], "name": name})
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
            except psutil.AccessDenied as exc:
                raise ValueError("native process inventory cannot prove exclusivity") from exc
    except psutil.Error as exc:
        raise ValueError("native process enumeration unavailable") from exc
    return competing


def gpu_python_compute_owners() -> list[dict[str, object]]:
    """Refuse unrelated Python CUDA owners missed by command-line heuristics.

    WDDM's GPU-process inventory includes graphical desktop applications and
    sometimes inaccessible system processes. It does *not* provide reliable
    per-process VRAM accounting. We therefore use its PID/name observations
    only as an additional veto for identifiable model/Python executables,
    never as evidence of free VRAM or as a replacement for native floors.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("NVIDIA GPU process inventory unavailable") from exc
    if result.returncode:
        raise ValueError("NVIDIA GPU process inventory unavailable")
    owners: dict[int, dict[str, object]] = {}
    for row in csv.reader(result.stdout.splitlines()):
        if len(row) != 2 or not row[0].strip().isdecimal():
            raise ValueError("malformed NVIDIA GPU process inventory")
        pid = int(row[0].strip())
        if pid <= 0:
            raise ValueError("invalid NVIDIA GPU process identity")
        if pid == os.getpid():
            continue
        name = PureWindowsPath(row[1].strip()).name.lower()
        if (not name or name.startswith("[insufficient permissions]")
                or name in ("[not supported]", "[unknown]")):
            raise ValueError("NVIDIA GPU process owner cannot be attributed")
        if (re.fullmatch(r"python(?:w|\d+(?:\.\d+)?)?\.exe", name)
                or name in ("llama-server.exe", "llama_server.exe",
                            "ollama.exe", "vllm.exe", "torchrun.exe",
                            "vmmemwsl.exe", "vmmem.exe", "podman.exe",
                            "wslhost.exe", "vmwp.exe", "containerd.exe")):
            owners[pid] = {"pid": pid, "name": name}
    return [owners[pid] for pid in sorted(owners)]


def native_single_gpu_uuid() -> str:
    """Refuse ambiguous device selection in the pinned first-GPU resource probe.

    The unchanged historical resource helper samples only the first GPU, but
    the native launcher does not pin a UUID. Until that original architecture
    has a reviewed explicit device binding, multi-GPU hosts are not admissible.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("NVIDIA GPU device identity unavailable") from exc
    uuids = [line.strip() for line in result.stdout.splitlines()]
    if (result.returncode or len(uuids) != 1
            or re.fullmatch(
                r"GPU-[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}",
                uuids[0],
            ) is None):
        raise ValueError("original native resource admission requires one identifiable GPU")
    return uuids[0]


def occupied_native_ports(ports: tuple[int, ...] = PORTS) -> list[int]:
    occupied = []
    for port in ports:
        for family, address in ((socket.AF_INET, "127.0.0.1"),
                                (socket.AF_INET6, "::1")):
            try:
                with socket.socket(family, socket.SOCK_STREAM) as probe:
                    if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                        probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                    probe.bind((address, port))
            except OSError:
                occupied.append(port)
                break
    return sorted(set(occupied))


def _checkpoint_directories(root: Path, completed: int) -> None:
    """Do not start an expensive model before checking foreign checkpoint paths."""
    assert_native_ordinary(root)
    assert_native_ordinary(root / "owner.json")
    for path in root.iterdir():
        assert_native_ordinary(path)
        if not path.is_dir():
            continue
        match = re.fullmatch(r"task-(\d{3})(?:\.unfinished-([0-9a-f]{16}))?", path.name)
        if (path.is_symlink() or match is None or int(match.group(1)) >= 600):
            raise ValueError("foreign or linked directory in original checkpoint")
        index = int(match.group(1))
        if match.group(2) is None and index > completed:
            raise ValueError("noncontiguous unscored original task directory")
        if match.group(2) is None and index == completed:
            children = list(path.iterdir())
            allowed = {"public-automationbench.private.json",
                       "official-evaluator.private.log"}
            if (not children or any(child.name not in allowed or not child.is_file()
                                    for child in children)):
                raise ValueError("unscored original task has foreign evidence")
            for child in children:
                assert_native_ordinary(child)


def preflight(max_new: int) -> tuple[dict, object | None]:
    if type(max_new) is not int or not 1 <= max_new <= 600:
        raise ValueError("max-new must be an integer from 1 to 600")
    if os.name != "nt":
        raise ValueError("original Windows native continuation requires Windows")
    evidence = reader.inspect()  # exact Git sources, owner, and all scored rows
    runner = sys.modules[RUNNER_MODULE]
    completed = evidence["completed_scored_tasks"]
    if (evidence["active_unscored_indices"] not in ([], [completed])
            or evidence["native_sessions_pooled_for_frozen_comparison"] is not False
            or evidence["frozen_gen0_status"] != "BLOCKED"
            or evidence["chosen"] is not None
            or evidence["release_approved"] is not False):
        raise ValueError("source-bound checkpoint or frozen decision is invalid")
    _checkpoint_directories(reader.ROOT, completed)
    if completed == 600:
        return {"schema": SCHEMA, "status": "COMPLETE_NO_NATIVE_RESTART",
                "completed_scored_tasks": completed, "model_executed": False,
                "gen0_status": "BLOCKED", "chosen": None,
                "release_approved": False}, None

    blockers = []
    try:
        processes = competing_model_processes()
    except (OSError, ValueError) as exc:
        processes = []
        blockers.append("NATIVE_PROCESS_INVENTORY_UNTRUSTED: " + str(exc)[:160])
    if processes:
        blockers.append("OTHER_NATIVE_MODEL_SERVER_ACTIVE")
    try:
        gpu_owners = gpu_python_compute_owners()
    except (OSError, ValueError) as exc:
        gpu_owners = []
        blockers.append("NVIDIA_GPU_OWNER_INVENTORY_UNTRUSTED: " + str(exc)[:160])
    if gpu_owners:
        blockers.append("OTHER_GPU_COMPUTE_PYTHON_ACTIVE")
    gpu_uuid = None
    try:
        gpu_uuid = native_single_gpu_uuid()
    except (OSError, ValueError) as exc:
        blockers.append("NATIVE_GPU_DEVICE_IDENTITY_UNTRUSTED: " + str(exc)[:160])
    ports = occupied_native_ports()
    if ports:
        blockers.append("NATIVE_PORT_OCCUPIED")
    resource = None
    try:
        # Unmodified 84ff970 conservative physical RAM/commit/VRAM floors;
        # deliberately NEVER request its optional experimental bypass.
        resource = runner._native_prestart(reader.ROOT, bypass_conservative=False)
    except (OSError, RuntimeError, ValueError) as exc:
        blockers.append("ORIGINAL_CONSERVATIVE_RESOURCE_FLOOR_UNMET: " + str(exc)[:160])
    return {
        "schema": SCHEMA,
        "status": "REFUSED_NATIVE_ADMISSION" if blockers else "READY_ORIGINAL_GUARDV3_RESUME",
        "historical_git": reader.COMMIT, "historical_guard": "guard.v3",
        "owner_sha256": reader.OWNER_SHA,
        "completed_scored_tasks": completed,
        "next_original_index": completed,
        "active_unscored_indices": evidence["active_unscored_indices"],
        "max_new": max_new,
        "blockers": blockers, "competing_model_processes": processes,
        "gpu_python_compute_owners": gpu_owners,
        "native_gpu_uuid": gpu_uuid,
        "occupied_ports": ports, "native_prestart": resource,
        "model_executed": False, "frozen_four_slot_eligible": False,
        "gen0_status": "BLOCKED", "chosen": None,
        "release_approved": False,
    }, runner


def run(*, execute: bool, max_new: int) -> dict:
    if not execute:
        report, _ = preflight(max_new)
        return report
    with exclusive_native_resume():
        # Even the source/checkpoint inspection must be performed under the
        # cross-session lease. A competing cooperating resume cannot change
        # the checkpoint between the independent audit and native launch.
        report, runner = preflight(max_new)
        if runner is None or report["status"] != "READY_ORIGINAL_GUARDV3_RESUME":
            return report
        # Recheck volatile ownership after admission, while the lease remains
        # held until the original native server and guard have been torn down.
        if (competing_model_processes() or gpu_python_compute_owners()
                or occupied_native_ports()):
            raise ValueError("native process or port ownership changed before execution")
        if native_single_gpu_uuid() != report["native_gpu_uuid"]:
            raise ValueError("original native GPU identity changed before execution")
        runner._native_prestart(reader.ROOT, bypass_conservative=False)
        result = runner.run(reader.ROOT, execute=True, max_new=max_new,
                            bypass_conservative=False)
        if (result.get("gen0_status") != "BLOCKED"
                or result.get("chosen") is not None
                or result.get("release_approved") is not False
                or result.get("original_frozen_gen0_serving_comparable") is not False):
            raise ValueError("historical runner returned a forbidden release state")
        # The pinned original runner normally tears down its owned native
        # server in finally. A guard-shutdown exception or an unsuccessful
        # child-tree cleanup must not be reported as a clean native session.
        if (competing_model_processes() or gpu_python_compute_owners()
                or occupied_native_ports()):
            raise ValueError("original native server or guard teardown unverified")
    return {**result, "historical_git": reader.COMMIT,
            "historical_owner_sha256": reader.OWNER_SHA,
            "historical_guard": "guard.v3", "frozen_four_slot_eligible": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-new", type=int, default=600)
    args = parser.parse_args()
    try:
        result = run(execute=args.execute, max_new=args.max_new)
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        return 0 if result["status"] != "REFUSED_NATIVE_ADMISSION" else 2
    except (OSError, ValueError, KeyError, RuntimeError, ImportError, TypeError) as exc:
        print(json.dumps({"status": "REFUSED_ORIGINAL_GUARDV3_SOURCE_LOCKED_RESUME",
                          "reason": str(exc)[:380], "gen0_status": "BLOCKED",
                          "chosen": None, "release_approved": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
