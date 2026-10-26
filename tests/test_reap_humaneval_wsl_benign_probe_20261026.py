"""CPU-only tests: WSL namespace census never unlocks HumanEval programs."""

import hashlib
import json

import pytest

from chowder_batch import reap_humaneval_wsl_benign_probe_20261026 as probe


def test_benign_child_is_frozen_and_contains_no_dynamic_program_ingestion():
    assert "HumanEval" not in probe.FIXED_CHILD
    assert "subprocess" not in probe.FIXED_CHILD
    assert "socket" not in probe.FIXED_CHILD
    assert hashlib.sha256(probe.FIXED_CHILD.encode()).hexdigest() == probe.CHILD_SHA256


def test_namespace_capabilities_are_only_one_part_of_execution_policy():
    result = probe.interpret_observation(
        {"child_namespace_ids": {n: f"{n}:[2]" for n in probe.NAMES},
         "host_namespace_ids": {n: f"{n}:[1]" for n in probe.NAMES},
         "observer_tracked_descendants": 2,
         "observer_verified_descendants_gone": True,
         "python_major_minor": "3.13", "windows_mount_visible": False,
         "cgroup_v2_present": True, "cgroup_delegation_proven": True,
         "network_egress_denial_proven": False,
         "isolated_rootfs_proven": True})
    assert result["status"] == "CAPABILITY_PROBE_ONLY_NOT_EXECUTION_APPROVAL"
    assert result["verified_complete_humaneval_execution_sandbox"] is False
    assert result["no_generated_program_executed"] is True


@pytest.mark.parametrize("field,value", [
    ("python_major_minor", "3.12"),
    ("windows_mount_visible", True),
    ("cgroup_delegation_proven", False),
    ("observer_verified_descendants_gone", False),
    ("isolated_rootfs_proven", False),
    ("network_egress_denial_proven", False),
])
def test_actual_missing_requirement_is_reported_not_passed(field, value):
    source = {"child_namespace_ids": {n: f"{n}:[2]" for n in probe.NAMES},
              "host_namespace_ids": {n: f"{n}:[1]" for n in probe.NAMES},
              "observer_tracked_descendants": 1,
              "observer_verified_descendants_gone": True,
              "python_major_minor": "3.13", "windows_mount_visible": False,
              "cgroup_v2_present": True, "cgroup_delegation_proven": True,
              "network_egress_denial_proven": True,
              "isolated_rootfs_proven": True}
    source[field] = value
    report = probe.interpret_observation(source)
    assert report["verified_complete_humaneval_execution_sandbox"] is False
    assert report["execution_approved"] is False
    assert report["status"] == "CAPABILITY_PROBE_ONLY_NOT_EXECUTION_APPROVAL"


def test_source_has_no_model_or_private_input_cli_and_is_fixed_child_only():
    import inspect
    source = inspect.getsource(probe)
    for forbidden in ("--execute-generated", "--rows", "--prompts",
                      "--model-path", "--candidate", "shell=True"):
        assert forbidden not in source


def test_parent_refuses_malformed_and_incomplete_child_observations():
    with pytest.raises(ValueError, match="namespace|observation"):
        probe.interpret_observation({"host_namespace_ids": {}})
    with pytest.raises(ValueError, match="namespace|observation"):
        probe.interpret_observation(json.loads('{"child_namespace_ids": []}'))


def test_benign_probe_cli_returns_nonzero_for_observed_unqualified_runtime(
    monkeypatch, capsys,
):
    monkeypatch.setattr(probe, "run_fixed_benign_probe", lambda: {
        "status": "CAPABILITY_PROBE_ONLY_NOT_EXECUTION_APPROVAL",
        "verified_complete_humaneval_execution_sandbox": False,
        "execution_approved": False,
    })
    assert probe.main() == 2
    output = json.loads(capsys.readouterr().out)
    assert output["execution_approved"] is False
