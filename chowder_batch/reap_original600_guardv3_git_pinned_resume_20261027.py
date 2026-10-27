"""Fail-closed native continuation of the *original* guard.v3 research run.

This uses the original runner/guard/scorer bytes from Git commit 84ff970,
not later guard.v4 worktree code. It never changes a scored task or upgrades
one-slot research evidence into a frozen four-slot Gen-0 result. The default
is read-only; --execute is required to load the original model.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import re
import socket
import sys

from chowder_batch import reap_original600_guardv3_git_pinned_reader_20261027 as reader

SCHEMA = "chowder.reap-original600-guardv3-source-locked-resume.v1"
RUNNER_MODULE = "chowder_batch.reap_q4_automationbench_full600_resumable_20261027"
PORTS = (8892, 8894)
FILE_ATTRIBUTE_REPARSE_POINT = 0x400


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
                name = str(info.get("name") or "").lower()
                argv = [str(x).lower() for x in (info.get("cmdline") or [])]
                # Shell commands can *mention* llama-server in their text;
                # that does not make bash/pwsh a native model server.
                vllm_python = (name.startswith("python") and
                               (any("vllm.entrypoints" in x for x in argv)
                                or any(argv[i:i + 2] == ["-m", "vllm"]
                                       for i in range(len(argv) - 1))))
                if ("llama-server" in name or "llama_server" in name
                        or vllm_python):
                    competing.append({"pid": info["pid"], "name": name})
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
            except psutil.AccessDenied as exc:
                raise ValueError("native process inventory cannot prove exclusivity") from exc
    except psutil.Error as exc:
        raise ValueError("native process enumeration unavailable") from exc
    return competing


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
    processes = competing_model_processes()
    if processes:
        blockers.append("OTHER_NATIVE_MODEL_SERVER_ACTIVE")
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
        "occupied_ports": ports, "native_prestart": resource,
        "model_executed": False, "frozen_four_slot_eligible": False,
        "gen0_status": "BLOCKED", "chosen": None,
        "release_approved": False,
    }, runner


def run(*, execute: bool, max_new: int) -> dict:
    report, runner = preflight(max_new)
    if not execute or runner is None or report["status"] != "READY_ORIGINAL_GUARDV3_RESUME":
        return report
    # Recheck the volatile ownership evidence immediately before the original
    # runner performs its own native admission and immutable source validation.
    if competing_model_processes() or occupied_native_ports():
        raise ValueError("native process or port ownership changed before execution")
    runner._native_prestart(reader.ROOT, bypass_conservative=False)
    result = runner.run(reader.ROOT, execute=True, max_new=max_new,
                        bypass_conservative=False)
    if (result.get("gen0_status") != "BLOCKED"
            or result.get("chosen") is not None
            or result.get("release_approved") is not False
            or result.get("original_frozen_gen0_serving_comparable") is not False):
        raise ValueError("historical runner returned a forbidden release state")
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
