"""Independent source-pinned audit of actual benign original scorer VM execution.

This audits an existing private immutable receipt; it never invokes Podman,
executes Python from a receipt, or approves model-generated HumanEval replay.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

EVIDENCE = Path(
    "F:/reap-v2-humaneval-original-scorer-benign-semantic-witness-20261027-0524-exclusive/"
    "original-scorer-benign-semantics.json"
)
EVIDENCE_SHA = "e24808e3870bcbdb5c07efb890d1d1443d526c66130953ff0a3afef7ebbf0ce3"
EVALUATOR_SHA = "9736714f21834a45c23599a50fa92ef3bebc6cf70967aca7e2329d26b4d73ee1"
FUNCTION_SHA = "05116801507e62913901147f13432e3ef5eaa92853ab7c60a28100481279d104"
IMAGE_SHA = "72036a705a7cff38924efd674e60654666cb283faf7b3fad160b34007f7395bd"
FIXTURE_SHA = {
    "pass": "801c277cbb286b70bb26dc7c056e6b59f6399387f72c830f755a3cb053ecde83",
    "fail": "b3b67a06f589d7183e05a8a8e6cf50ea829ca2c16aa9debaa70b70633f3423ff",
    "timeout": "3eb8227e5f5d44ba8c2260e8c35adf4e28cd4f44fa7696ec67d0e13c7f9ae246",
}


def audit(raw: bytes, *, expected_sha: str = EVIDENCE_SHA) -> dict:
    if (not isinstance(raw, bytes) or not 0 < len(raw) <= 256 * 1024
            or hashlib.sha256(raw).hexdigest() != expected_sha):
        raise ValueError("independent immutable private semantic witness SHA mismatch")
    try:
        data = json.loads(raw, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    except (UnicodeError, ValueError) as exc:
        raise ValueError("private semantic witness malformed") from exc
    if (type(data) is not dict
            or data.get("schema") != "chowder.reap-v2-original-humaneval-benign-semantics.v1"
            or data.get("status") != "BENIGN_ORIGINAL_SCORER_SEMANTICS_WITNESSED_NOT_EXECUTION_APPROVAL"
            or data.get("original_frozen_evaluator_sha256") != EVALUATOR_SHA
            or data.get("original_frozen_function_sha256") != FUNCTION_SHA
            or data.get("runtime_image_sha256") != IMAGE_SHA
            or data.get("generated_programs_executed") != 0
            or data.get("independent_original_164_replay_approved") is not False
            or data.get("gen0_status") != "BLOCKED"
            or data.get("chosen") is not None
            or data.get("release_approved") is not False):
        raise ValueError("source-bound benign-only identity or safety status missing")
    cases = data.get("cases")
    if not isinstance(cases, list) or [c.get("fixture") for c in cases
                                       if isinstance(c, dict)] != list(FIXTURE_SHA):
        raise ValueError("all three fixed original scorer cases required")
    owners = set()
    candidate_identities = set()
    results = {}
    for case in cases:
        name = case["fixture"]
        vm, post, candidate = case.get("vm"), case.get("post"), case.get("candidate")
        observed, result = case.get("observed"), case.get("result")
        if (case.get("fixture_sha256") != FIXTURE_SHA[name]
                or case.get("original_frozen_evaluator_sha256") != EVALUATOR_SHA
                or case.get("original_frozen_function_sha256") != FUNCTION_SHA
                or case.get("runtime_image_sha256") != IMAGE_SHA
                or case.get("generated_model_code_executed") is not False
                or not all(type(x) is dict for x in (vm, post, candidate, observed, result))):
            raise ValueError("fixed benign fixture/source or outside VM witness forged")
        outer, inner = vm.get("observer_namespaces"), vm.get("container_namespaces")
        if (type(outer) is not dict or type(inner) is not dict
                or any(type(outer.get(ns)) is not str or type(inner.get(ns)) is not str
                       or outer[ns] == inner[ns]
                       for ns in ("user", "pid", "mnt", "net"))
                or vm.get("cgroup_memory_max") != "805306368"
                or vm.get("cgroup_pids_max") != "16"
                or type(vm.get("cgroup_pids_current")) is not int
                or not 2 <= vm["cgroup_pids_current"] <= 16
                or type(vm.get("container_init_pid")) is not int
                or vm["container_init_pid"] <= 1
                or type(vm.get("container_init_start_ticks")) is not int
                or vm["container_init_start_ticks"] <= 0):
            raise ValueError("independent namespace, memory, PID or process census invalid")
        route = vm.get("network_routes")
        if (type(route) is not str or any(
            len(line.split()) > 1 and line.split()[1] == "00000000"
            for line in route.splitlines()[1:]
        )):
            raise ValueError("network isolation not independently observed")
        owner = vm.get("cgroup_path")
        if (type(owner) is not str or not owner.startswith("/")
                or owner in owners or ".." in owner.split("/")):
            raise ValueError("cgroup owner missing, unsafe or reused")
        owners.add(owner)
        descendants = vm.get("descendants")
        if type(descendants) is not list or not descendants:
            raise ValueError("outside VM descendant census absent")
        child_id = (candidate.get("pid"), candidate.get("start_ticks"))
        observed_ids = {(p.get("pid"), p.get("start_ticks"))
                        for p in observed.get("processes", []) if type(p) is dict}
        descendant_ids = {(p.get("pid"), p.get("start_ticks"))
                          for p in descendants if type(p) is dict}
        argv, cwd = candidate.get("argv"), candidate.get("cwd")
        if (child_id in candidate_identities or child_id not in observed_ids
                or child_id not in descendant_ids
                or type(argv) is not list or len(argv) != 3
                or argv[1] != "-I"
                or type(argv[2]) is not str
                or re.fullmatch(r"/tmp/tmp[A-Za-z0-9_-]+/candidate\.py", argv[2]) is None
                or cwd != argv[2].rsplit("/", 1)[0]):
            raise ValueError("actual original -I candidate.py identity/cwd not independently witnessed")
        candidate_identities.add(child_id)
        if (post.get("same_container_init_alive") is not False
                or post.get("same_descendants_alive") != []
                or not (post.get("cgroup_directory_extinct") is True
                        or post.get("cgroup_pids_current_after") == "0")):
            raise ValueError("independent process-tree/cgroup teardown absent")
        elapsed = result.get("elapsed")
        expected = name == "pass"
        if (result.get("passed") is not expected
                or result.get("python") != [3, 13]
                or result.get("isolated_parent") is not True
                or result.get("rlimit_as") != 2 * 1024 ** 3
                or result.get("rlimit_fsize") != 1024 ** 2
                or type(elapsed) not in (float, int)
                or not (9.5 <= elapsed <= 11.5 if name == "timeout"
                        else 3.5 <= elapsed < 9.5)):
            raise ValueError("actual original scorer pass/fail/10-second timeout not reproduced")
        results[name] = {"correct": result["passed"], "elapsed_seconds": round(elapsed, 3),
                         "isolated_candidate_pid": child_id[0]}
    return {
        "status": "INDEPENDENT_BENIGN_ORIGINAL_SCORER_SEMANTICS_VERIFIED_NO_MODEL_CODE",
        "private_receipt_sha256": expected_sha,
        "cases": results, "unique_disposable_cgroups": len(owners),
        "generated_programs_executed": 0,
        "original_164_replay_approved": False,
        "gen0_status": "BLOCKED", "chosen": None, "release_approved": False,
    }


def main() -> int:
    if (not EVIDENCE.is_file() or EVIDENCE.is_symlink()
            or EVIDENCE.resolve() != EVIDENCE):
        raise SystemExit("private original benign receipt unavailable or linked")
    print(json.dumps(audit(EVIDENCE.read_bytes()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
