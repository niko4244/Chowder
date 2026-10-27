"""Adversarial source-bound audit of real benign scorer execution evidence."""

from __future__ import annotations

import hashlib
import json

import pytest

from chowder_batch import reap_humaneval_podman_semantics_independent_audit_20261027 as audit


def original():
    """Synthetic adversarial fixture; never ship private VM process evidence."""
    cases = []
    for i, (name, fixture_sha) in enumerate(audit.FIXTURE_SHA.items()):
        pid = 101 + i
        start = 1001 + i
        path = f"/tmp/tmpabc{i}/candidate.py"
        candidate = {"pid": pid, "start_ticks": start,
                     "argv": ["/usr/local/bin/python3", "-I", path],
                     "cwd": path.rsplit("/", 1)[0]}
        cases.append({
            "fixture": name, "fixture_sha256": fixture_sha,
            "original_frozen_evaluator_sha256": audit.EVALUATOR_SHA,
            "original_frozen_function_sha256": audit.FUNCTION_SHA,
            "runtime_image_sha256": audit.IMAGE_SHA,
            "generated_model_code_executed": False,
            "candidate": candidate,
            "observed": {"processes": [candidate]},
            "vm": {
                "observer_namespaces": {x: f"{x}:[1]" for x in ("user", "pid", "mnt", "net")},
                "container_namespaces": {x: f"{x}:[2]" for x in ("user", "pid", "mnt", "net")},
                "cgroup_memory_max": "805306368", "cgroup_pids_max": "16",
                "cgroup_pids_current": 3,
                "container_init_pid": 201 + i,
                "container_init_start_ticks": 2001 + i,
                "network_routes": "Iface Destination\n",
                "cgroup_path": f"/user.slice/private{i}",
                "descendants": [{"pid": pid, "start_ticks": start}],
            },
            "post": {"same_container_init_alive": False,
                     "same_descendants_alive": [], "cgroup_directory_extinct": True},
            "result": {"passed": name == "pass",
                       "elapsed": 10.01 if name == "timeout" else 4.01,
                       "python": [3, 13], "isolated_parent": True,
                       "rlimit_as": 2 * 1024 ** 3, "rlimit_fsize": 1024 ** 2},
        })
    return {
        "schema": "chowder.reap-v2-original-humaneval-benign-semantics.v1",
        "status": "BENIGN_ORIGINAL_SCORER_SEMANTICS_WITNESSED_NOT_EXECUTION_APPROVAL",
        "original_frozen_evaluator_sha256": audit.EVALUATOR_SHA,
        "original_frozen_function_sha256": audit.FUNCTION_SHA,
        "runtime_image_sha256": audit.IMAGE_SHA,
        "generated_programs_executed": 0,
        "independent_original_164_replay_approved": False,
        "gen0_status": "BLOCKED", "chosen": None, "release_approved": False,
        "cases": cases,
    }


def check_mutation(record):
    raw = json.dumps(record, sort_keys=True).encode()
    # Give the attacker a self-consistent digest: structural security must
    # still reject the forgery even if a caller supplies an altered SHA.
    with pytest.raises(ValueError):
        audit.audit(raw, expected_sha=hashlib.sha256(raw).hexdigest())


def test_actual_private_original_scorer_semantics_independently_verified():
    if not audit.EVIDENCE.is_file():
        pytest.skip("private source-bound real VM witness not mounted in this environment")
    result = audit.audit(audit.EVIDENCE.read_bytes())
    assert result["status"].startswith("INDEPENDENT_BENIGN")
    assert list(result["cases"]) == ["pass", "fail", "timeout"]
    assert result["unique_disposable_cgroups"] == 3
    assert result["generated_programs_executed"] == 0
    assert result["original_164_replay_approved"] is False
    assert result["release_approved"] is False


@pytest.mark.parametrize("case_index", [0, 1, 2])
def test_forged_success_or_timing_rejected(case_index):
    record = original()
    record["cases"][case_index]["result"]["passed"] = (
        not record["cases"][case_index]["result"]["passed"]
    )
    check_mutation(record)
    record = original()
    record["cases"][case_index]["result"]["elapsed"] = 0.01
    check_mutation(record)


@pytest.mark.parametrize(("field", "value"), [
    ("cgroup_memory_max", "max"), ("cgroup_pids_max", "max"),
    ("cgroup_pids_current", 17),
    ("network_routes", "Iface Destination\neth0 00000000\n"),
])
def test_vm_namespace_or_resource_forgery_rejected(field, value):
    record = original()
    record["cases"][0]["vm"][field] = value
    check_mutation(record)


def test_cross_case_reused_cgroup_or_process_identity_rejected():
    record = original()
    record["cases"][1]["vm"]["cgroup_path"] = record["cases"][0]["vm"]["cgroup_path"]
    check_mutation(record)
    record = original()
    record["cases"][1]["candidate"]["pid"] = record["cases"][0]["candidate"]["pid"]
    check_mutation(record)


@pytest.mark.parametrize(("field", "value"), [
    ("same_container_init_alive", True),
    ("same_descendants_alive", [123]),
    ("cgroup_directory_extinct", False),
])
def test_post_teardown_forgery_rejected(field, value):
    record = original()
    record["cases"][0]["post"][field] = value
    check_mutation(record)


def test_candidate_argv_and_working_directory_cannot_be_substituted():
    record = original()
    record["cases"][0]["candidate"]["argv"][1] = "-c"
    check_mutation(record)
    record = original()
    record["cases"][0]["candidate"]["cwd"] = "/host"
    check_mutation(record)


def test_replaced_model_code_or_release_claim_refused():
    record = original()
    record["cases"][0]["generated_model_code_executed"] = True
    check_mutation(record)
    record = original()
    record["release_approved"] = True
    check_mutation(record)
    record = original()
    record["independent_original_164_replay_approved"] = True
    check_mutation(record)


def test_source_digest_and_missing_population_refused():
    record = original()
    record["cases"][0]["fixture_sha256"] = "0" * 64
    check_mutation(record)
    record = original()
    record["cases"] = record["cases"][:2]
    check_mutation(record)


def test_synthetic_valid_structure_and_original_digest_fail_closed():
    record = original()
    raw = json.dumps(record, sort_keys=True).encode()
    assert audit.audit(raw, expected_sha=hashlib.sha256(raw).hexdigest())[
        "unique_disposable_cgroups"
    ] == 3
    with pytest.raises(ValueError, match="SHA mismatch"):
        audit.audit(raw)
