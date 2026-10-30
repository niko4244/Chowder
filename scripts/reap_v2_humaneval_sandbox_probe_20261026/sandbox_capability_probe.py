"""Harmless CPU-only Linux isolation probe, not a HumanEval executor.

Only runs a fixed trusted Python snippet printing process namespace IDs and
mount flags. Never loads GGUF weights, reads generated code, sends network
requests, changes cgroups, or treats Kaggle metadata alone as sandbox proof.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys

RESULT_PATH = pathlib.Path("/kaggle/working/reap-v2-human164-sandbox-capability-native.json")
CHILD = """import json, os, pathlib
names=('user','pid','mnt','net')
result={'ns':{k:os.readlink('/proc/self/ns/'+k) for k in names},
        'uid':os.getuid(),'euid':os.geteuid(),'gid':os.getgid(),
        'groups':os.getgroups(),
        'usr_ro':bool(os.statvfs('/usr').f_flag & getattr(os,'ST_RDONLY',1)),
        'uid_map':pathlib.Path('/proc/self/uid_map').read_text()[:256]}
print(json.dumps(result,sort_keys=True))
"""
FIXED_CHILD_SHA = hashlib.sha256(CHILD.encode()).hexdigest()


def original_namespace_ids() -> dict:
    return {name: os.readlink("/proc/self/ns/" + name)
            for name in ("user", "pid", "mnt", "net")}


def run_fixed_probe(command: list[str]) -> dict:
    """Time-bound benign subprocess; no generated/untrusted Python."""
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=12,
                              stdin=subprocess.DEVNULL, check=False, env={
                                  "PATH": "/usr/bin:/bin", "PYTHONNOUSERSITE": "1"
                              })
        if proc.returncode != 0:
            return {"status": "UNAVAILABLE", "exit_code": proc.returncode,
                    "stderr_sha256": hashlib.sha256(proc.stderr.encode()).hexdigest()}
        obj = json.loads(proc.stdout)
        if (not isinstance(obj, dict) or not isinstance(obj.get("ns"), dict)
                or not all(isinstance(obj["ns"].get(name), str)
                           for name in ("user", "pid", "mnt", "net"))):
            return {"status": "UNVERIFIABLE_FIXED_PROBE_OUTPUT"}
        return {"status": "FIXED_HARMLESS_CHILD_SUCCEEDED", "observed": obj}
    except (OSError, subprocess.TimeoutExpired, ValueError, UnicodeError) as exc:
        return {"status": "UNAVAILABLE", "reason": type(exc).__name__}


def main() -> int:
    if RESULT_PATH.exists() or RESULT_PATH.is_symlink():
        raise RuntimeError("refusing to overwrite private prior sandbox probe")
    host = original_namespace_ids()
    result = {
        "schema": "reap-v2-private-human164-harmless-kaggle-sandbox-probe.v1",
        "status": "NOT_PROVEN",
        "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "no_model_loaded": True, "no_generated_python_executed": True,
        "no_network_request_issued": True, "no_cgroup_modified": True,
        "no_original_quality_gate_eligibility": True,
        "original_gen0_status": "BLOCKED", "release_approved": False,
        "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
        "host_namespace_ids": host,
        "fixed_trusted_child_sha256": FIXED_CHILD_SHA,
        "actual_cpu_kernel_uid": os.getuid(),
        "actual_cpu_kernel_euid": os.geteuid(),
        "bubblewrap_present": bool(shutil.which("bwrap")),
        "unshare_present": bool(shutil.which("unshare")),
        "cgroup_v2_present": pathlib.Path("/sys/fs/cgroup/cgroup.controllers").is_file(),
        "cgroup_kill_present": pathlib.Path("/sys/fs/cgroup/cgroup.kill").is_file(),
        "cgroup_delegation_proven": False,
    }
    child = [sys.executable, "-I", "-c", CHILD]
    if shutil.which("unshare"):
        command = ["unshare", "--user", "--map-root-user", "--mount", "--pid",
                   "--net", "--fork", *child]
        result["unshare_probe"] = run_fixed_probe(command)
    if shutil.which("bwrap"):
        command = [
            "bwrap", "--unshare-all", "--die-with-parent", "--new-session",
            "--ro-bind", "/usr", "/usr", "--proc", "/proc", "--dev", "/dev",
            "--tmpfs", "/tmp", "--chdir", "/tmp", *child,
        ]
        result["bubblewrap_probe"] = run_fixed_probe(command)
    proven = []
    for name in ("unshare_probe", "bubblewrap_probe"):
        probe = result.get(name, {})
        if probe.get("status") != "FIXED_HARMLESS_CHILD_SUCCEEDED":
            continue
        observed = probe["observed"]
        if all(observed["ns"][n] != host[n] for n in ("user", "pid", "mnt", "net")):
            proven.append(name)
    result["new_user_pid_mount_network_namespaces_proven"] = bool(proven)
    result["available_namespaced_launchers"] = proven
    # A namespace probe alone does not certify cgroup-based descendant-kill,
    # bounded memory/PIDs, no-new-privileges or runtime provenance. Never
    # auto-escalate to evaluating arbitrary candidate programs.
    result["verified_complete_humaneval_execution_sandbox"] = False
    result["status"] = "CAPABILITY_PROBE_ONLY_NOT_AN_EXECUTION_APPROVAL"
    with RESULT_PATH.open("x", encoding="utf8", newline="\n") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"],
                      "available_namespaced_launchers": proven,
                      "verified_complete_humaneval_execution_sandbox": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
