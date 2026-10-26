"""Bounded *harmless* Ubuntu WSL namespace/process capability census.

Never receives original model text, a benchmark dataset or executable source.
Never runs model-generated Python. NEVER authorizes a HumanEval runner, even if
all its descriptive fields eventually look favorable. This probes the known
Kaggle shortfall on an independent local host without changing shared mounts,
supervised services, cgroup knobs, credentials or sandbox approval hashes.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

NAMES = ("user", "pid", "mnt", "net")

# Fixed benign code only: does not import networking, read private content,
# start descendants or execute arbitrary caller-provided Python. The time
# budget gives the OUTER supervisor an opportunity to inspect /proc identity.
FIXED_CHILD = '''import json, os, pathlib, sys, time
names = ("user", "pid", "mnt", "net")
result = {
    "child_namespace_ids": {n: os.readlink("/proc/self/ns/" + n) for n in names},
    "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
    "windows_mount_visible": any(pathlib.Path(p).exists() for p in ("/mnt/c", "/mnt/d", "/mnt/f")),
    "isolated_rootfs_proven": False,
    "network_egress_denial_proven": False,
    "cgroup_delegation_proven": False,
}
print(json.dumps(result, sort_keys=True), flush=True)
time.sleep(2)
'''

# This content hash is a provenance checksum for an exclusively harmless
# child; it is NOT an approved runtime/interpreter/image/sandbox signature.
CHILD_SHA256 = "02d7d8f049e46d1c4177d834ece5c98a18c13e58f9e2cba628d19d9a5c2bcf45"


def _ns_ids(pid: int | str) -> dict[str, str]:
    return {name: os.readlink(f"/proc/{pid}/ns/{name}") for name in NAMES}


def _live_descendants(parent_pid: int) -> dict[int, str]:
    """Observe only a fixed launcher's descendants, no unrelated processes."""
    lineage = {parent_pid}
    alive: dict[int, tuple[int, str]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            # /proc/PID/stat: strip `comm` safely, which can include spaces.
            raw = (entry / "stat").read_text(encoding="ascii")
            end = raw.rfind(")")
            fields = raw[end + 2:].split()
            if end < 0 or len(fields) < 20:
                continue
            ppid = int(fields[1])
            start_ticks = fields[19]
            alive[int(entry.name)] = (ppid, start_ticks)
        except (OSError, ValueError, IndexError, UnicodeError):
            continue
    changes = True
    while changes:
        changes = False
        for pid, (ppid, _) in alive.items():
            if ppid in lineage and pid not in lineage:
                lineage.add(pid)
                changes = True
    return {pid: alive[pid][1] for pid in lineage if pid != parent_pid and pid in alive}


def interpret_observation(observed: dict) -> dict:
    """A negative capability witness only, including when everything appears good."""
    if not isinstance(observed, dict):
        raise ValueError("missing original harmless namespace observation")
    host = observed.get("host_namespace_ids")
    child = observed.get("child_namespace_ids")
    if (not isinstance(host, dict) or not isinstance(child, dict)
            or set(host) != set(NAMES) or set(child) != set(NAMES)
            or any(not isinstance(value, str)
                   or re.fullmatch(r"[a-z]+:\[\d+\]", value) is None
                   for value in list(host.values()) + list(child.values()))):
        raise ValueError("namespace observation incomplete or malformed")
    separate = all(host[name] != child[name] for name in NAMES)
    problems = []
    if not separate:
        problems.append("SEPARATE_USER_PID_MOUNT_NETWORK_NAMESPACES_UNPROVEN")
    if observed.get("windows_mount_visible") is not False:
        problems.append("WINDOWS_DRIVE_MOUNTS_NOT_HIDDEN")
    if observed.get("isolated_rootfs_proven") is not True:
        problems.append("ISOLATED_READ_ONLY_ROOTFS_UNPROVEN")
    if observed.get("cgroup_v2_present") is not True:
        problems.append("CGROUP_V2_MISSING_OR_UNVERIFIED")
    if observed.get("cgroup_delegation_proven") is not True:
        problems.append("CGROUP_MEMORY_PID_DELEGATION_UNPROVEN")
    if observed.get("network_egress_denial_proven") is not True:
        problems.append("KERNEL_ENFORCED_NO_EGRESS_UNPROVEN")
    if observed.get("python_major_minor") != "3.13":
        problems.append("ORIGINAL_PYTHON_313_REQUIRED")
    if observed.get("observer_verified_descendants_gone") is not True:
        problems.append("EXTERNAL_DESCENDANT_EXTINCTION_UNPROVEN")
    if (type(observed.get("observer_tracked_descendants")) is not int
            or observed["observer_tracked_descendants"] < 1):
        problems.append("EXTERNAL_CHILD_PROCESS_ID_NOT_OBSERVED")
    return {
        "schema": "chowder.reap-v2-wsl-harmless-namespace-capability.v1",
        "status": "CAPABILITY_PROBE_ONLY_NOT_EXECUTION_APPROVAL",
        "new_user_pid_mount_network_namespaces_observed": separate,
        "unresolved_gaps": problems,
        "no_generated_program_executed": True,
        "verified_complete_humaneval_execution_sandbox": False,
        "execution_approved": False,
        "gen0_status": "BLOCKED", "chosen": None,
        "release_approved": False,
    }


def run_fixed_benign_probe() -> dict:
    """No filesystem writes; bounded owned harmless unshare child on Linux."""
    if sys.platform != "linux" or not Path("/proc/self/ns/user").is_file():
        raise RuntimeError("fixed WSL namespace capability probe requires Linux")
    if shutil.which("unshare") is None:
        raise RuntimeError("unshare is unavailable")
    host = _ns_ids("self")
    args = ["unshare", "--user", "--map-root-user", "--mount", "--pid", "--net",
            "--fork", "--mount-proc", "--kill-child=SIGKILL",
            sys.executable, "-I", "-c", FIXED_CHILD]
    started = time.monotonic()
    runner = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, shell=False)
    tracked: dict[int, str] = {}
    timed_out = False
    try:
        for _ in range(8):
            time.sleep(0.12)
            tracked.update(_live_descendants(runner.pid))
            if tracked:
                break
        try:
            stdout, stderr = runner.communicate(timeout=8)
        except subprocess.TimeoutExpired:
            timed_out = True
            runner.kill()  # Only the unique fixed-source launcher this call owns.
            stdout, stderr = runner.communicate(timeout=3)
    except BaseException:
        if runner.poll() is None:
            runner.kill()
            runner.communicate(timeout=3)
        raise
    # Never claim child death from launcher exit alone. Bind a previously
    # recorded host-visible PID/start-time to the independent post-census.
    survivors = _live_descendants(runner.pid)
    confirmed_gone = bool(tracked) and all(
        not Path(f"/proc/{pid}/stat").exists() for pid in tracked
    ) and not survivors
    if (timed_out or runner.returncode != 0 or len(stdout) > 16384
            or len(stderr) > 16384):
        raise RuntimeError("fixed harmless unshare did not complete cleanly")
    try:
        child = json.loads(stdout)
    except (ValueError, UnicodeDecodeError) as exc:
        raise RuntimeError("fixed child observation unreadable") from exc
    if not isinstance(child, dict) or set(child) != {
        "child_namespace_ids", "python_major_minor", "windows_mount_visible",
        "isolated_rootfs_proven", "network_egress_denial_proven",
        "cgroup_delegation_proven"
    }:
        raise RuntimeError("unexpected benign namespace child observation")
    cgroup_v2 = Path("/sys/fs/cgroup/cgroup.controllers").is_file()
    witness = {**child, "host_namespace_ids": host,
               "observer_tracked_descendants": len(tracked),
               "observer_verified_descendants_gone": confirmed_gone,
               "cgroup_v2_present": cgroup_v2,
               "benign_child_source_sha256": CHILD_SHA256,
               "elapsed_wall_seconds": round(time.monotonic() - started, 3)}
    return interpret_observation(witness)


def main() -> int:
    """Machine-readable finding, nonzero shell status until FULLY qualified."""
    try:
        finding = run_fixed_benign_probe()
        print(json.dumps(finding, sort_keys=True, allow_nan=False))
        # Deliberate: code has no execution approval pathway. Successful
        # observation of namespace differences is NOT a sandbox PASS.
        return 2
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(json.dumps({"status": "REFUSED_BENIGN_NAMESPACE_PROBE_ERROR",
                          "reason": type(exc).__name__,
                          "verified_complete_humaneval_execution_sandbox": False,
                          "no_generated_program_executed": True,
                          "execution_approved": False, "gen0_status": "BLOCKED"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
