"""Execute ONLY fixed benign fixtures with the exact frozen HumanEval scorer.

This is an independent candidate-file/cwd/10-second-timeout admission test.
The original scorer function is extracted from its SHA-pinned ZIP unchanged,
then run in a disposable rootless Python 3.13 container. The Podman VM observes
the actual candidate.py process from OUTSIDE its PID namespace and independently
verifies that its process tree/cgroup disappear. No model-generated code, model
weights, network, host bind mounts, or protected service is touched.

Passing this probe is NOT permission to execute original model generations.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid
import zipfile

from chowder_batch.reap_external_full_audit import EVALUATOR_SHA
from chowder_batch.reap_humaneval_podman_vm_witness_20261027 import (
    IMAGE_SHA256, VM_OBSERVER, VM_POST_OBSERVER, benign_command, vm_python,
)

FROZEN_ZIP = Path(
    "F:/reap-v2-kaggle-frozen-evaluator-source-20261026/"
    "original-reap-frozen-evaluator-4da43ef.zip"
)
FROZEN_ZIP_SHA = "32660296029473011586487cbf8cc1bdbf38f1450561e8bfe52db9f743985f89"
FUNCTION_SHA = "05116801507e62913901147f13432e3ef5eaa92853ab7c60a28100481279d104"
FIXTURES = {
    "pass": (
        "import time\ntime.sleep(4)\ndef f(x):\n    return x + 1\n",
        "def check(fn):\n    assert fn(2) == 3\n",
        True,
    ),
    "fail": (
        "import time\ntime.sleep(4)\ndef f(x):\n    return x + 2\n",
        "def check(fn):\n    assert fn(2) == 3\n",
        False,
    ),
    "timeout": (
        "import time\ntime.sleep(30)\ndef f(x):\n    return x + 1\n",
        "def check(fn):\n    assert fn(2) == 3\n",
        False,
    ),
}

# Executed in the Podman VM, not in the untrusted workload namespace. Its
# only arguments are decimal PIDs obtained from the VM's own process census.
VM_CANDIDATE_OBSERVER = r'''import json, os, pathlib, sys
p = pathlib.Path
result = []
for raw in sys.argv[1:]:
    pid = int(raw)
    try:
        cmd = (p('/proc') / str(pid) / 'cmdline').read_bytes().split(b'\x00')
        argv = [x.decode('utf-8') for x in cmd if x]
        cwd = os.readlink('/proc/' + str(pid) + '/cwd')
        stat = (p('/proc') / str(pid) / 'stat').read_text()
        start = int(stat[stat.rfind(')') + 2:].split()[19])
        result.append({'pid': pid, 'start_ticks': start, 'argv': argv, 'cwd': cwd})
    except (OSError, UnicodeError, ValueError, IndexError):
        continue
print(json.dumps({'processes': result}, sort_keys=True))
'''


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def frozen_function() -> str:
    if (not FROZEN_ZIP.is_file() or FROZEN_ZIP.is_symlink()
            or FROZEN_ZIP.stat().st_size > 128 * 1024
            or sha(FROZEN_ZIP.read_bytes()) != FROZEN_ZIP_SHA):
        raise ValueError("frozen original evaluator ZIP missing or changed")
    with zipfile.ZipFile(FROZEN_ZIP) as archive:
        if archive.namelist().count("chowder_batch/reap_eval.py") != 1:
            raise ValueError("original evaluator ZIP member missing/ambiguous")
        source = archive.read("chowder_batch/reap_eval.py")
    if sha(source) != EVALUATOR_SHA:
        raise ValueError("frozen original evaluator SHA mismatch")
    decoded = source.decode("utf-8")
    nodes = [n for n in ast.parse(decoded).body
             if isinstance(n, ast.FunctionDef) and n.name == "_humaneval_pass"]
    if len(nodes) != 1:
        raise ValueError("frozen original HumanEval function absent")
    function = ast.get_source_segment(decoded, nodes[0])
    if not isinstance(function, str) or sha(function.encode()) != FUNCTION_SHA:
        raise ValueError("original scorer function changed")
    return function


def fixed_source(name: str, function: str) -> str:
    if name not in FIXTURES or sha(function.encode()) != FUNCTION_SHA:
        raise ValueError("only the three source-pinned benign cases are allowed")
    code, test, _ = FIXTURES[name]
    # The original function writes its candidate to a fresh tempfile, then
    # invokes sys.executable -I with an ABSOLUTE candidate.py path, cwd=tmp,
    # and subprocess.run(timeout=10). No replacement scorer is used.
    return (
        "import hashlib, json, pathlib, re, resource, subprocess, sys, tempfile, time\n"
        "from pathlib import Path\n"
        "resource.setrlimit(resource.RLIMIT_AS, (2147483648, 2147483648))\n"
        "resource.setrlimit(resource.RLIMIT_FSIZE, (1048576, 1048576))\n"
        + function + "\n"
        + "item = " + repr({"entry_point": "f", "header": "", "test": test}) + "\n"
        + "completion = " + repr(code) + "\n"
        + "started = time.monotonic()\n"
        + "passed = _humaneval_pass(item, completion)\n"
        + "elapsed = time.monotonic() - started\n"
        + "print(json.dumps({'passed': passed, 'elapsed': elapsed, "
        "'python': sys.version_info[:2], "
        "'isolated_parent': sys.flags.isolated == 1, "
        "'rlimit_as': resource.getrlimit(resource.RLIMIT_AS)[0], "
        "'rlimit_fsize': resource.getrlimit(resource.RLIMIT_FSIZE)[0]}, "
        "sort_keys=True), flush=True)\n"
    )


def verify_candidate_processes(vm: dict, observed: dict) -> dict:
    descendants = vm.get("descendants")
    processes = observed.get("processes")
    if not isinstance(descendants, list) or not isinstance(processes, list):
        raise ValueError("outside VM candidate process evidence missing")
    allowed = {(row.get("pid"), row.get("start_ticks")) for row in descendants
               if isinstance(row, dict)}
    candidates = []
    for row in processes:
        if not isinstance(row, dict) or (row.get("pid"), row.get("start_ticks")) not in allowed:
            raise ValueError("outside observer returned an unbound process identity")
        argv, cwd = row.get("argv"), row.get("cwd")
        if (isinstance(argv, list) and len(argv) == 3
                and argv[1] == "-I" and isinstance(argv[2], str)
                and re.fullmatch(r"/tmp/tmp[A-Za-z0-9_-]+/candidate\.py", argv[2])
                and cwd == argv[2].rsplit("/", 1)[0]):
            candidates.append(row)
    if len(candidates) != 1:
        raise ValueError("VM did not witness exactly one original isolated candidate.py child")
    return candidates[0]


def verify_fixed_result(name: str, payload: dict, candidate: dict, post: dict) -> None:
    if name not in FIXTURES:
        raise ValueError("foreign fixed benign fixture")
    expected = FIXTURES[name][2]
    elapsed = payload.get("elapsed")
    if (type(payload.get("passed")) is not bool or payload["passed"] is not expected
            or payload.get("python") != [3, 13]
            or payload.get("isolated_parent") is not True
            or payload.get("rlimit_as") != 2 * 1024 ** 3
            or payload.get("rlimit_fsize") != 1024 ** 2
            or type(elapsed) not in (int, float)
            or not 0 < elapsed < 12
            or (name == "timeout" and not 9.5 <= elapsed <= 11.5)
            or (name != "timeout" and not 3.5 <= elapsed < 9.5)):
        raise ValueError("frozen scorer candidate-file or ten-second timeout semantics failed")
    if (type(candidate.get("pid")) is not int
            or type(candidate.get("start_ticks")) is not int
            or post.get("same_container_init_alive") is not False
            or post.get("same_descendants_alive") != []
            or not (post.get("cgroup_directory_extinct") is True
                    or post.get("cgroup_pids_current_after") == "0")):
        raise ValueError("outside-VM candidate process or teardown witness failed")


def run_fixed_case(name: str, function: str) -> dict:
    if name not in FIXTURES:
        raise ValueError("unsupported fixture")
    source = fixed_source(name, function)
    owner = "reap-benign-vm-" + uuid.uuid4().hex[:16]
    if subprocess.run(["podman", "container", "exists", owner],
                      capture_output=True, timeout=10, check=False).returncode != 1:
        raise RuntimeError("exclusive benign container name not available")
    cmd = benign_command(owner)
    cmd[cmd.index("--timeout=14")] = "--timeout=15"
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)
    started = False
    try:
        if process.stdin is None:
            raise RuntimeError("benign stdin missing")
        process.stdin.write(source)
        process.stdin.close()
        process.stdin = None
        pid = None
        deadline = time.monotonic() + 6
        while time.monotonic() < deadline:
            if process.poll() is not None:
                stdout, stderr = process.communicate(timeout=3)
                raise RuntimeError(
                    "benign candidate container exited before VM witness: "
                    + repr((process.returncode, stdout[-250:], stderr[-500:]))
                )
            inspected = subprocess.run(["podman", "inspect", "--format", "{{.State.Pid}}", owner],
                                       capture_output=True, text=True, timeout=6, check=False)
            if inspected.returncode == 0 and inspected.stdout.strip().isdigit():
                pid = int(inspected.stdout.strip())
                if pid > 1:
                    started = True
                    break
            time.sleep(0.2)
        if not started or pid is None:
            raise RuntimeError("no VM-observable benign container owner")
        vm = vm_python(VM_OBSERVER, str(pid))
        descendant_pids = [str(row["pid"]) for row in vm["descendants"]]
        if not descendant_pids:
            raise RuntimeError("VM did not observe child processes")
        observed = vm_python(VM_CANDIDATE_OBSERVER, *descendant_pids)
        candidate = verify_candidate_processes(vm, observed)
        stdout, stderr = process.communicate(timeout=17)
        if process.returncode != 0 or len(stdout) > 8192 or len(stderr) > 32768:
            raise RuntimeError("original frozen scorer benign container did not complete")
        payload = json.loads(stdout.strip(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        descendants = ",".join(str(row["pid"]) + ":" + str(row["start_ticks"])
                               for row in vm["descendants"])
        post = vm_python(VM_POST_OBSERVER, str(pid),
                         str(vm["container_init_start_ticks"]), vm["cgroup_path"],
                         descendants)
        verify_fixed_result(name, payload, candidate, post)
        return {"fixture": name, "fixture_sha256": sha(source.encode()),
                "original_frozen_evaluator_sha256": EVALUATOR_SHA,
                "original_frozen_function_sha256": FUNCTION_SHA,
                "runtime_image_sha256": IMAGE_SHA256,
                "candidate": candidate, "vm": vm, "post": post,
                "observed": observed, "result": payload,
                "generated_model_code_executed": False}
    finally:
        if process.poll() is None:
            process.kill()
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        # This unique name was checked absent before launch. Never touch any
        # other container; a failed removal is an error, not evidence of safety.
        if started:
            exists = subprocess.run(["podman", "container", "exists", owner],
                                    capture_output=True, timeout=10, check=False)
            if exists.returncode == 0:
                removed = subprocess.run(["podman", "rm", "--force", owner],
                                         capture_output=True, timeout=12, check=False)
                if removed.returncode != 0:
                    raise RuntimeError("could not remove owned benign test container")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        if (not args.out.is_absolute() or args.out.exists() or args.out.is_symlink()
                or not args.out.parent.is_dir()):
            raise ValueError("output must be an exclusive private file")
        function = frozen_function()
        cases = [run_fixed_case(name, function) for name in FIXTURES]
        receipt = {
            "schema": "chowder.reap-v2-original-humaneval-benign-semantics.v1",
            "status": "BENIGN_ORIGINAL_SCORER_SEMANTICS_WITNESSED_NOT_EXECUTION_APPROVAL",
            "original_frozen_evaluator_sha256": EVALUATOR_SHA,
            "original_frozen_function_sha256": FUNCTION_SHA,
            "runtime_image_sha256": IMAGE_SHA256,
            "cases": cases, "generated_programs_executed": 0,
            "independent_original_164_replay_approved": False,
            "gen0_status": "BLOCKED", "chosen": None, "release_approved": False,
        }
        with args.out.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(receipt, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        print(json.dumps({"status": receipt["status"], "cases": len(cases),
                          "generated_programs_executed": 0}))
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError,
            json.JSONDecodeError, zipfile.BadZipFile) as error:
        print("benign original scorer semantic admission REFUSED: "
              + str(error)[:350], file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
