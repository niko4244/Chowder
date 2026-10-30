"""Harmless capability census for existing rootless Podman, not a replay.

The fixed child prints no model-generated text, never receives private
generations, mounts no files, and issues no network requests. Successful
execution is NOT authorization to execute model-generated code: an
independent sandbox/process-tree review and resource floor still apply.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

IMAGE_SHA256 = "274d131132dbdc6058ed77088525d77adaf0749a6d99f2cf7d42feb9a10737c8"
IMAGE_NAME = "docker.io/library/python:3.11-slim"
STATIC_CHILD = r'''
import json, os, pathlib
p=pathlib.Path
status={}
for line in p('/proc/self/status').read_text().splitlines():
    if line.startswith(('CapEff:', 'NoNewPrivs:', 'Seccomp:')):
        key, value=line.split(':',1);status[key]=value.strip()
result={'uid':os.getuid(),'gid':os.getgid(),
        'secrets_directory_empty':not any(p('/run/secrets').iterdir()),
        'uid_mapping':p('/proc/self/uid_map').read_text()[:300],
        'namespaces':{name:os.readlink('/proc/self/ns/'+name)
                      for name in ('user','pid','mnt','net')},
        'root_read_only':bool(os.statvfs('/').f_flag & os.ST_RDONLY),
        'status':status,
        'root_mounts':[line.split()[1] for line in p('/proc/self/mounts').read_text().splitlines() if len(line.split())>2][:80],
        'network_routes':p('/proc/net/route').read_text()[:2048],
        'cgroup_memory_max':p('/sys/fs/cgroup/memory.max').read_text().strip() if p('/sys/fs/cgroup/memory.max').exists() else None,
        'cgroup_pids_max':p('/sys/fs/cgroup/pids.max').read_text().strip() if p('/sys/fs/cgroup/pids.max').exists() else None}
print(json.dumps(result,sort_keys=True))
'''


def command(image: str) -> list[str]:
    """One fixed no-mount rootless container; never shell interpolate input."""
    return [
        "podman", "run", "--rm", "-i", "--pull=never", "--network=none",
        "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
        "--user=65534:65534", "--userns=keep-id", "--pid=private",
        "--ipc=private", "--pids-limit=16", "--memory=268435456",
        "--memory-swap=268435456", "--cpus=1", "--ulimit=core=0:0",
        "--tmpfs=/tmp:rw,nosuid,nodev,noexec,size=8m",
        "--tmpfs=/run/secrets:rw,nosuid,nodev,noexec,size=64k",
        "--env=PYTHONDONTWRITEBYTECODE=1", "--entrypoint=python3",
        image, "-I", "-",
    ]


def validate(child: dict, *, expected_memory_bytes: int = 268435456) -> dict:
    if type(expected_memory_bytes) is not int or expected_memory_bytes not in {
        268435456, 805306368,
    }:
        raise ValueError("unapproved fixed sandbox cgroup memory profile")
    if not isinstance(child, dict):
        raise ValueError("isolated child returned no structured census")
    status = child.get("status")
    if not isinstance(status, dict):
        raise ValueError("cannot attest sandbox kernel restrictions")
    if (child.get("uid") != 65534 or child.get("gid") != 65534
            or child.get("secrets_directory_empty") is not True
            or not isinstance(child.get("uid_mapping"), str)
            or child.get("root_read_only") is not True
            or status.get("CapEff") != "0000000000000000"
            or status.get("NoNewPrivs") != "1"
            or status.get("Seccomp") != "2"
            or child.get("cgroup_pids_max") != "16"
            or child.get("cgroup_memory_max") != str(expected_memory_bytes)):
        raise ValueError("sandbox UID, seccomp, caps, read-only or cgroup limit missing")
    names = child.get("namespaces")
    if not isinstance(names, dict) or any(
        not isinstance(names.get(x), str) or not names[x]
        for x in ("user", "pid", "mnt", "net")
    ):
        raise ValueError("sandbox namespace observation missing")
    routes = child.get("network_routes")
    if not isinstance(routes, str) or any(
        line.split()[1] == "00000000"
        for line in routes.splitlines()[1:]
        if len(line.split()) > 1
    ):
        raise ValueError("sandbox has a default network route")
    mounts = child.get("root_mounts")
    if not isinstance(mounts, list) or any(
        not isinstance(m, str) or (
            m != "/run/podman-init"
            and m.startswith(("/host", "/mnt/", "/run/podman"))
        )
        for m in mounts
    ):
        # The replay profile explicitly enables Podman's private init/PID1.
        # Its fixed in-container /run/podman-init executable appears as one
        # container-runtime mount, unlike a private host bind. Admit ONLY this
        # exact target; other /run/podman* paths and host bind roots still fail.
        raise ValueError("unexpected host bind/mount observed in sandbox")
    return {"trusted_fixed_probe_only": True, "scope": "SANDBOX_CAPABILITY_ONLY",
            "generated_programs_executed": 0,
            "independent_humaneval_full164_certified": False}


def probe() -> dict:
    # The exact existing local image ID is pinned; never pull/create images.
    query = subprocess.run(
        ["podman", "image", "inspect", IMAGE_NAME, "--format", "{{.Id}}"],
        capture_output=True, text=True, timeout=20, check=True,
    )
    if query.stdout.strip() != IMAGE_SHA256:
        raise ValueError("preexisting Python OCI image digest differs from frozen audit")
    container = subprocess.run(
        command(IMAGE_SHA256), input=STATIC_CHILD, text=True, capture_output=True,
        timeout=45, check=False,
    )
    if container.returncode != 0 or len(container.stdout) > 16_384:
        raise ValueError("rootless no-network private benign container probe refused: "
                         + container.stderr[-350:])
    child = json.loads(container.stdout)
    proof = validate(child)
    return {
        "schema": "chowder.reap-v2-harmless-private-podman-isolation-probe.v1",
        "status": "CAPABILITY_PROBE_ONLY_NOT_HUMANEVAL_APPROVAL",
        "image_id": IMAGE_SHA256, "static_child_sha256": hashlib.sha256(
            STATIC_CHILD.encode()
        ).hexdigest(),
        "census": child, **proof,
        "sandbox_qualified_for_generated_code": False,
        "gen0_status": "BLOCKED", "chosen": None, "release_approved": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.out.exists() or args.out.is_symlink() or not args.out.parent.is_dir():
            raise ValueError("harmless probe output must be new exclusive private path")
        result = probe()
        with args.out.open("x", encoding="utf8", newline="\n") as handle:
            json.dump(result, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
        print(json.dumps({"status": result["status"], "image_id": IMAGE_SHA256,
                          "generated_programs_executed": 0}, sort_keys=True))
        return 0
    except (OSError, ValueError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
        print("private Podman probe REFUSED: " + str(exc)[:180], file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
