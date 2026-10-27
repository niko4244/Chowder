"""Outside-namespace Podman VM witness of a fixed benign Python 3.13 child.

This never receives or executes model-generated code. It observes an existing
rootless Podman machine without modifying its containers, VM, or services.
Even a successful result is NOT authorization for HumanEval code execution.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

from chowder_batch.reap_humaneval_podman_probe import command, validate


IMAGE_SHA256 = "72036a705a7cff38924efd674e60654666cb283faf7b3fad160b34007f7395bd"
NAMESPACES = ("user", "pid", "mnt", "net")
BENIGN_CHILD = '''import json, os, pathlib, sys, time
p = pathlib.Path
status = {}
for line in p('/proc/self/status').read_text().splitlines():
    if line.startswith(('CapEff:', 'NoNewPrivs:', 'Seccomp:')):
        key, value = line.split(':', 1)
        status[key] = value.strip()
result = {
    'uid': os.getuid(), 'gid': os.getgid(),
    'python_major_minor': f'{sys.version_info.major}.{sys.version_info.minor}',
    'namespaces': {name: os.readlink('/proc/self/ns/' + name)
                   for name in ('user', 'pid', 'mnt', 'net')},
    'root_read_only': bool(os.statvfs('/').f_flag & os.ST_RDONLY),
    'secrets_directory_empty': not any(p('/run/secrets').iterdir()),
    'uid_mapping': p('/proc/self/uid_map').read_text()[:300],
    'network_routes': p('/proc/net/route').read_text()[:2048],
    'root_mounts': [line.split()[1] for line in p('/proc/self/mounts').read_text().splitlines()
                    if len(line.split()) > 2][:80],
    'cgroup_memory_max': p('/sys/fs/cgroup/memory.max').read_text().strip(),
    'cgroup_pids_max': p('/sys/fs/cgroup/pids.max').read_text().strip(),
    'status': status,
}
print(json.dumps(result, sort_keys=True), flush=True)
time.sleep(9)
'''

# This code runs in the VM's *outside-container* namespace. Its sole argument
# is the numeric host PID returned by Podman inspect. No user shell is invoked.
VM_OBSERVER = '''import json, os, pathlib, sys
p = pathlib.Path
pid = int(sys.argv[1])
def stat(n):
    raw = (p('/proc') / str(n) / 'stat').read_text()
    end = raw.rfind(')')
    if end < 0: raise RuntimeError('malformed proc stat')
    fields = raw[end + 2:].split()
    return {'ppid': int(fields[1]), 'start_ticks': int(fields[19])}
def ns(n):
    return {name: os.readlink(f'/proc/{n}/ns/{name}')
            for name in ('user', 'pid', 'mnt', 'net')}
lineage = {pid}
live = {}
for entry in p('/proc').iterdir():
    if entry.name.isdigit():
        try: live[int(entry.name)] = stat(int(entry.name))
        except (OSError, ValueError, IndexError): pass
for _ in range(12):
    additions = {n for n, row in live.items() if row['ppid'] in lineage}
    if additions <= lineage: break
    lineage.update(additions)
cg = (p('/proc') / str(pid) / 'cgroup').read_text().strip().splitlines()
if len(cg) != 1 or not cg[0].startswith('0::/'):
    raise RuntimeError('container cgroup v2 ownership missing')
cg_path = p('/sys/fs/cgroup') / cg[0][4:].lstrip('/')
result = {
    'observer_namespaces': ns('self'), 'container_init_pid': pid,
    'container_init_start_ticks': stat(pid)['start_ticks'],
    'container_namespaces': ns(pid),
    'descendants': [{'pid': n, 'start_ticks': live[n]['start_ticks']}
                    for n in sorted(lineage - {pid}) if n in live],
    'cgroup_path': cg[0][3:],
    'cgroup_memory_max': (cg_path / 'memory.max').read_text().strip(),
    'cgroup_pids_max': (cg_path / 'pids.max').read_text().strip(),
    'cgroup_pids_current': int((cg_path / 'pids.current').read_text().strip()),
    'network_routes': (p('/proc') / str(pid) / 'net/route').read_text()[:2048],
}
print(json.dumps(result, sort_keys=True))
'''

VM_POST_OBSERVER = '''import json, pathlib, sys
p = pathlib.Path
pid, start, cg, descendants = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
def still_alive(pid, start):
    proc = p('/proc') / str(pid) / 'stat'
    if not proc.exists(): return False
    raw = proc.read_text()
    end = raw.rfind(')')
    return end >= 0 and int(raw[end + 2:].split()[19]) == start
same_identity_alive = still_alive(pid, start)
surviving_descendants = [int(pair.split(':')[0]) for pair in descendants.split(',')
                         if still_alive(*map(int, pair.split(':')))]
cg_path = p('/sys/fs/cgroup') / cg.lstrip('/')
current = (cg_path / 'pids.current').read_text().strip() if cg_path.exists() else None
print(json.dumps({'same_container_init_alive': same_identity_alive,
                  'same_descendants_alive': surviving_descendants,
                  'cgroup_pids_current_after': current,
                  'cgroup_directory_extinct': not cg_path.exists()}, sort_keys=True))
'''


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def bounded_json(raw: str) -> dict:
    if not 0 < len(raw.encode("utf-8")) <= 32768:
        raise ValueError("bounded witness JSON absent or oversized")
    obj = json.loads(raw, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    if type(obj) is not dict:
        raise ValueError("witness JSON is not an object")
    return obj


def benign_command(name: str) -> list[str]:
    if not re.fullmatch(r"reap-benign-vm-[0-9a-f]{16}", name):
        raise ValueError("only unique fixed benign container names are permitted")
    args = command(IMAGE_SHA256)
    args[2:2] = ["--name=" + name, "--init", "--timeout=14"]
    args[args.index("--memory=268435456")] = "--memory=805306368"
    args[args.index("--memory-swap=268435456")] = "--memory-swap=805306368"
    args[args.index("--tmpfs=/tmp:rw,nosuid,nodev,noexec,size=8m")] = (
        "--tmpfs=/tmp:rw,nosuid,nodev,noexec,size=64m"
    )
    return args


def verify(child: dict, vm: dict, post: dict, *, expected_pid: int) -> dict:
    validate(child, expected_memory_bytes=805306368)
    if child.get("python_major_minor") != "3.13":
        raise ValueError("original Python 3.13 benign runtime not observed")
    if (type(expected_pid) is not int or expected_pid <= 1
            or vm.get("container_init_pid") != expected_pid
            or type(vm.get("container_init_start_ticks")) is not int
            or vm["container_init_start_ticks"] <= 0
            or vm.get("cgroup_memory_max") != "805306368"
            or vm.get("cgroup_pids_max") != "16"
            or type(vm.get("cgroup_pids_current")) is not int
            or not 1 <= vm["cgroup_pids_current"] <= 16):
        raise ValueError("independent VM PID/cgroup admission failed")
    outer = vm.get("observer_namespaces")
    inner = vm.get("container_namespaces")
    if (not isinstance(outer, dict) or not isinstance(inner, dict)
            or any(not isinstance(outer.get(n), str) or not isinstance(inner.get(n), str)
                   or outer[n] == inner[n] or inner[n] != child["namespaces"][n]
                   for n in NAMESPACES)):
        raise ValueError("VM did not independently observe four isolated namespaces")
    descendants = vm.get("descendants")
    if (not isinstance(descendants, list) or not descendants
            or any(type(d) is not dict or type(d.get("pid")) is not int
                   or type(d.get("start_ticks")) is not int
                   or d["pid"] <= 1 or d["start_ticks"] <= 0 for d in descendants)):
        raise ValueError("VM could not observe a real container process tree")
    routes = vm.get("network_routes")
    if (not isinstance(routes, str) or any(
        len(line.split()) > 1 and line.split()[1] == "00000000"
        for line in routes.splitlines()[1:]
    )):
        raise ValueError("VM observed an external default route")
    if (post.get("same_container_init_alive") is not False
            or post.get("same_descendants_alive") != []
            or not (post.get("cgroup_directory_extinct") is True
                    or post.get("cgroup_pids_current_after") == "0")):
        raise ValueError("VM cannot attest benign container process-tree extinction")
    return {"outside_vm_namespaces_observed": True,
            "independent_benign_process_tree_seen": True,
            "independent_benign_owner_extinct": True,
            "independent_benign_descendants_extinct": True,
            "no_default_route_observed": True,
            "generated_programs_executed": 0,
            "execution_approved": False,
            "independent_humaneval_full164_certified": False}


def vm_python(source: str, *args: str, timeout: int = 12) -> dict:
    # `podman machine ssh` joins its remote argv into an SSH shell command.
    # A literal multiline `-c` argument is split by that shell. Encode ONLY
    # our own immutable Python source, and permit numeric/absolute-cgroup
    # arguments without shell metacharacters or traversal.
    for index, value in enumerate(args):
        if (not re.fullmatch(r"(?:[0-9]+|[0-9:,]+|/[A-Za-z0-9_./:@-]+)", value)
                or ".." in value.split("/")):
            raise ValueError("unsafe independent VM observer argument " + str(index)
                             + ": " + repr(value[:150]))
    encoded = base64.b64encode(source.encode("utf-8")).decode("ascii")
    fixed = "'exec(__import__(\"base64\").b64decode(\"" + encoded + "\"))'"
    result = subprocess.run(
        ["podman", "machine", "ssh", "--", "python3", "-c", fixed, *args],
        capture_output=True, text=True, timeout=timeout, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("independent VM observer refused: " + result.stderr[-500:])
    return bounded_json(result.stdout)


def run_benign() -> dict:
    image = subprocess.run(
        ["podman", "image", "inspect", IMAGE_SHA256, "--format", "{{.Id}}"],
        capture_output=True, text=True, timeout=15, check=True,
    )
    if image.stdout.strip() != IMAGE_SHA256:
        raise ValueError("frozen local Python 3.13 image changed")
    name = "reap-benign-vm-" + uuid.uuid4().hex[:16]
    exists = subprocess.run(["podman", "container", "exists", name],
                            capture_output=True, timeout=10, check=False)
    if exists.returncode != 1:
        raise RuntimeError("exclusive benign container name already exists")
    started = False
    process = subprocess.Popen(benign_command(name), stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        if process.stdin is None:
            raise RuntimeError("benign child stdin missing")
        process.stdin.write(BENIGN_CHILD)
        process.stdin.close()
        process.stdin = None
        pid = None
        deadline = time.monotonic() + 6
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("benign container exited before outside witness")
            inspect = subprocess.run(
                ["podman", "inspect", "--format", "{{.State.Pid}}", name],
                capture_output=True, text=True, timeout=6, check=False,
            )
            if inspect.returncode == 0 and inspect.stdout.strip().isdigit():
                pid = int(inspect.stdout.strip())
                if pid > 1:
                    started = True
                    break
            time.sleep(0.3)
        if not started or pid is None:
            raise RuntimeError("no independently observable benign VM process")
        vm = vm_python(VM_OBSERVER, str(pid))
        stdout, stderr = process.communicate(timeout=17)
        if process.returncode != 0 or stderr and len(stderr) > 32768:
            raise RuntimeError("fixed benign container did not complete cleanly")
        child = bounded_json(stdout.strip())
        descendants = ",".join(str(row["pid"]) + ":" + str(row["start_ticks"])
                               for row in vm["descendants"])
        post = vm_python(VM_POST_OBSERVER, str(pid),
                         str(vm["container_init_start_ticks"]), vm["cgroup_path"],
                         descendants)
        result = verify(child, vm, post, expected_pid=pid)
        return {"schema": "chowder.reap-v2-benign-independent-vm-witness.v1",
                "status": "BENIGN_VM_WITNESS_ONLY_NOT_GENERATED_EXECUTION_APPROVAL",
                "image_sha256": IMAGE_SHA256, "fixed_child_sha256": sha(BENIGN_CHILD),
                "fixed_vm_observer_sha256": sha(VM_OBSERVER),
                "fixed_post_observer_sha256": sha(VM_POST_OBSERVER),
                "child": child, "vm": vm, "post": post, **result,
                "gen0_status": "BLOCKED", "chosen": None,
                "release_approved": False}
    finally:
        if process.poll() is None:
            # Only the unique container created by this fixed-source invocation.
            # Do not touch preexisting services or any other Podman container.
            process.kill()
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            if started:
                subprocess.run(["podman", "rm", "--force", name],
                               capture_output=True, timeout=12, check=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    try:
        if (not args.out.is_absolute() or args.out.exists() or args.out.is_symlink()
                or not args.out.parent.is_dir()):
            raise ValueError("output must be a new exclusive private file")
        receipt = run_benign()
        with args.out.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(receipt, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        print(json.dumps({"status": receipt["status"],
                          "generated_programs_executed": 0}, sort_keys=True))
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print("outside-VM benign witness REFUSED: " + str(error)[:300], file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
