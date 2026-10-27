"""Fail-closed, offline tests of the fixed benign outside-VM observer."""

from __future__ import annotations

from copy import deepcopy

import pytest

from chowder_batch import reap_humaneval_podman_vm_witness_20261027 as witness


def fixture():
    names = ("user", "pid", "mnt", "net")
    outer = {name: f"{name}:[100]" for name in names}
    inner = {name: f"{name}:[200]" for name in names}
    child = {
        "uid": 65534, "gid": 65534, "python_major_minor": "3.13",
        "secrets_directory_empty": True, "uid_mapping": "0 100000 65536",
        "namespaces": inner, "root_read_only": True,
        "root_mounts": ["/", "/tmp", "/run/secrets"],
        "network_routes": "Iface\tDestination\tGateway\n",
        "cgroup_memory_max": "805306368", "cgroup_pids_max": "16",
        "status": {"CapEff": "0000000000000000", "NoNewPrivs": "1", "Seccomp": "2"},
    }
    vm = {
        "container_init_pid": 12345, "container_init_start_ticks": 67890,
        "observer_namespaces": outer, "container_namespaces": inner,
        "descendants": [{"pid": 12346, "start_ticks": 67891}],
        "cgroup_path": "/user.slice/user-1000.slice/user@1000.service/app.slice/benign",
        "cgroup_memory_max": "805306368", "cgroup_pids_max": "16",
        "cgroup_pids_current": 2,
        "network_routes": "Iface\tDestination\tGateway\n",
    }
    post = {
        "same_container_init_alive": False, "same_descendants_alive": [],
        "cgroup_directory_extinct": True, "cgroup_pids_current_after": None,
    }
    return child, vm, post


def test_fixed_benign_witness_never_approves_execution():
    child, vm, post = fixture()
    result = witness.verify(child, vm, post, expected_pid=12345)
    assert result["generated_programs_executed"] == 0
    assert result["execution_approved"] is False
    assert result["independent_humaneval_full164_certified"] is False
    assert result["independent_benign_descendants_extinct"] is True


@pytest.mark.parametrize("location,value", [
    (("child", "uid"), 0),
    (("child", "python_major_minor"), "3.11"),
    (("child", "root_read_only"), False),
    (("child", "secrets_directory_empty"), False),
    (("child", "cgroup_memory_max"), "max"),
    (("child", "cgroup_pids_max"), "32"),
    (("child", "status", "Seccomp"), "0"),
    (("child", "status", "CapEff"), "0000000000000001"),
    (("child", "status", "NoNewPrivs"), "0"),
    (("child", "root_mounts"), ["/", "/host/private"]),
    (("child", "network_routes"), "Iface Destination\neth0 00000000\n"),
    (("vm", "container_init_pid"), 99999),
    (("vm", "container_init_start_ticks"), 0),
    (("vm", "cgroup_memory_max"), "max"),
    (("vm", "cgroup_pids_max"), "32"),
    (("vm", "cgroup_pids_current"), 0),
    (("vm", "descendants"), []),
    (("vm", "network_routes"), "Iface Destination\neth0 00000000\n"),
    (("post", "same_container_init_alive"), True),
    (("post", "same_descendants_alive"), [12346]),
    (("post", "cgroup_directory_extinct"), False),
])
def test_adversarial_witness_mutations_fail_closed(location, value):
    child, vm, post = fixture()
    objects = {"child": child, "vm": vm, "post": post}
    target = objects[location[0]]
    for key in location[1:-1]:
        target = target[key]
    target[location[-1]] = value
    if location == ("post", "cgroup_directory_extinct"):
        post["cgroup_pids_current_after"] = "1"
    with pytest.raises(ValueError):
        witness.verify(child, vm, post, expected_pid=12345)


def test_vm_must_independently_see_distinct_namespaces():
    child, vm, post = fixture()
    for name in witness.NAMESPACES:
        altered = deepcopy(vm)
        altered["observer_namespaces"][name] = child["namespaces"][name]
        with pytest.raises(ValueError):
            witness.verify(child, altered, post, expected_pid=12345)


def test_fixed_container_command_no_network_or_host_mounts():
    args = witness.benign_command("reap-benign-vm-0123456789abcdef")
    assert "--pull=never" in args
    assert "--network=none" in args
    assert "--read-only" in args
    assert "--cap-drop=ALL" in args
    assert "--memory=805306368" in args
    assert "--memory-swap=805306368" in args
    assert "--pids-limit=16" in args
    assert "--init" in args
    assert "--timeout=14" in args
    assert not any(x in args for x in ("-v", "--volume", "--mount", "--privileged"))
    assert witness.IMAGE_SHA256 in args
    with pytest.raises(ValueError):
        witness.benign_command("hermes-brainz-stack_litellm_1")


@pytest.mark.parametrize("bad", [
    "$(id)", "1;id", "../../etc/passwd", "/a/../b", "/a b", "1'",
])
def test_vm_observer_rejects_shell_metacharacters_before_spawn(bad):
    with pytest.raises(ValueError):
        witness.vm_python(witness.VM_OBSERVER, bad)


def test_vm_observer_rejects_unbounded_or_nonfinite_json():
    with pytest.raises(ValueError):
        witness.bounded_json('{"x": NaN}')
    with pytest.raises(ValueError):
        witness.bounded_json("{" + " " * 33000 + "}")
    with pytest.raises(ValueError):
        witness.bounded_json("[]")
