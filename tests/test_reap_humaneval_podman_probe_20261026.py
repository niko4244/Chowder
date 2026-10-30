"""CPU-only audit of the harmless, no-generation local capability probe."""

import copy

import pytest

from chowder_batch.reap_humaneval_podman_probe import command, validate, IMAGE_SHA256


def _fake():
    return {
        "uid": 65534, "gid": 65534, "root_read_only": True,
        "secrets_directory_empty": True, "uid_mapping": "0 1000 1",
        "status": {"CapEff": "0000000000000000", "NoNewPrivs": "1", "Seccomp": "2"},
        "namespaces": {"user": "user:[3]", "pid": "pid:[4]", "mnt": "mnt:[5]", "net": "net:[6]"},
        "cgroup_pids_max": "16", "cgroup_memory_max": "268435456",
        "network_routes": "Iface Destination Gateway Flags\n",
        "root_mounts": ["/", "/proc", "/dev", "/tmp"],
    }


def test_probe_cannot_execute_generated_code_or_read_host_mounts():
    args = command(IMAGE_SHA256)
    for required in ("--pull=never", "--network=none", "--read-only",
                     "--cap-drop=ALL", "--security-opt=no-new-privileges",
                     "--userns=keep-id", "--pid=private", "--ipc=private",
                     "--user=65534:65534", "--pids-limit=16"):
        assert required in args
    assert "-v" not in args and "--volume" not in args
    assert "--privileged" not in args and "--device" not in args
    assert args[-2:] == ["-I", "-"]


@pytest.mark.parametrize("change", [
    lambda d: d.update(uid=0),
    lambda d: d.update(secrets_directory_empty=False),
    lambda d: d.update(root_read_only=False),
    lambda d: d["status"].update(CapEff="0000000000000001"),
    lambda d: d["status"].update(Seccomp="0"),
    lambda d: d["status"].update(NoNewPrivs="0"),
    lambda d: d.update(cgroup_memory_max="max"),
    lambda d: d.update(cgroup_pids_max="max"),
    lambda d: d["namespaces"].pop("net"),
    lambda d: d.update(network_routes="Iface Destination Gateway Flags\neth0 00000000 01020304 3\n"),
    lambda d: d["root_mounts"].append("/host/secrets"),
])
def test_probe_refuses_unsafe_isolation(change):
    fields = copy.deepcopy(_fake())
    change(fields)
    with pytest.raises(ValueError):
        validate(fields)


def test_probe_succeeding_does_not_prove_generated_program_security():
    result = validate(_fake())
    assert result["generated_programs_executed"] == 0
    assert result["independent_humaneval_full164_certified"] is False


def test_replay_profile_exact_768m_cgroup_is_checked_separately():
    child = _fake()
    child["cgroup_memory_max"] = "805306368"
    child["root_mounts"].append("/run/podman-init")
    with pytest.raises(ValueError):
        validate(child)
    assert validate(child, expected_memory_bytes=805306368)[
        "independent_humaneval_full164_certified"
    ] is False
    child["cgroup_memory_max"] = "max"
    with pytest.raises(ValueError):
        validate(child, expected_memory_bytes=805306368)


@pytest.mark.parametrize("bad_mount", [
    "/run/podman-credentials", "/run/podman-init/private",
    "/run/podman-init/../secret", "/host", "/mnt/c/Users",
])
def test_exact_podman_init_runtime_mount_is_not_a_blanket_host_mount_bypass(bad_mount):
    child = _fake()
    child["cgroup_memory_max"] = "805306368"
    child["root_mounts"].extend(["/run/podman-init", bad_mount])
    with pytest.raises(ValueError, match="host bind/mount"):
        validate(child, expected_memory_bytes=805306368)
