"""CPU-only RED/GREEN failure modes for independent fixed benign host witness."""

import hashlib
import json
import copy
from pathlib import Path
import subprocess

import pytest

from chowder_batch import reap_humaneval_independent_witness_20261027 as witness
from chowder_batch import reap_humaneval_v2_attestation as v2


def sample_host_observation():
    child = {n: f"{n}:[{i + 101}]" for i, n in enumerate(witness.NAMESPACES)}
    host = {n: f"{n}:[{i + 900}]" for i, n in enumerate(witness.NAMESPACES)}
    return {
        "host_uid": 1000, "host_ns": host, "child_ns": child,
        "child": {"pid": 54321, "ppid": 555, "start_ticks": 987654321, "state": "S"},
        "descendants": {"54322": {"pid": 54322, "ppid": 54321, "start_ticks": 987654322, "state": "S"}},
        "status": {"Uid": "1000\t1000\t1000\t1000", "CapEff": "0000000000000000",
                   "NoNewPrivs": "1", "Seccomp": "2"},
        "cgroup_path": "/sys/fs/cgroup/user.slice/user-1000.slice/libpod-expected.scope",
        "cgroup_procs": "54321\n54322\n", "memory_max": str(witness.MEMORY_BYTES),
        "pids_max": str(witness.PIDS_LIMIT),
        "network_interfaces": ["lo"], "ipv4_route": "Iface Destination Gateway Flags\n",
        "ipv6_route": "", "root_read_only": True,
        "tmp_read_only": True, "work_writable": True, "host_mounts_seen": [],
        "python_executable_sha256": witness.PYTHON_SHA256,
    }


def test_source_frozen_v2_approval_is_still_explicitly_blocked():
    assert v2.APPROVED_SANDBOX_CAPABILITY_SHA256 is None
    assert v2.APPROVED_INDEPENDENT_WITNESS_SHA256 is None
    assert witness.EXECUTION_APPROVED is False
    assert hashlib.sha256(witness.BENIGN_CANDIDATE.encode()).hexdigest() == witness.CANDIDATE_SHA256
    assert hashlib.sha256(witness.LAUNCHER.encode()).hexdigest() == witness.LAUNCHER_SHA256
    assert hashlib.sha256(witness.VM_HOST_OBSERVER.encode()).hexdigest() == witness.VM_HOST_OBSERVER_SHA256
    assert witness.IMAGE_ID == "72036a705a7cff38924efd674e60654666cb283faf7b3fad160b34007f7395bd"


def test_only_exact_owned_name_image_and_nonnetwork_profile_usable():
    with pytest.raises(ValueError, match="exclusively owned"):
        witness.container_argv("postgres")
    with pytest.raises(ValueError, match="exclusively owned"):
        witness.container_argv("w10-he-benign-../../etc/passwd")
    name = "w10-he-benign-1234567890abcdef"
    cmd = witness.container_argv(name)
    assert cmd[:3] == ["podman", "run", "--detach"]
    assert cmd[-4:] == [witness.IMAGE_ID, "-I", "-c", witness.LAUNCHER]
    for element in ("--pull=never", "--network=none", "--cgroupns=private",
                    "--pid=private", "--userns=keep-id", "--read-only", "--cap-drop=ALL",
                    "--security-opt=no-new-privileges", "--user=65534:65534", "--pids-limit=16",
                    f"--memory={witness.MEMORY_BYTES}",
                    "--tmpfs=/tmp:rw,nosuid,nodev,noexec,size=8m,mode=1777"):
        assert element in cmd
    assert "--privileged" not in cmd and "--network=host" not in cmd
    assert not any(item.startswith(("-v", "--volume", "--mount")) for item in cmd)


def test_valid_synthetic_external_host_observation_only_attests_benign_properties():
    result = witness.verify_external_observation(sample_host_observation())
    assert result["external_namespaces_all_distinct"] is True
    assert result["nonroot_no_caps_seccomp2"] is True
    assert result["real_benign_descendants_seen"] == 1
    assert "original_humaneval_164_approved" not in result


def test_actual_kernel_ipv6_loopback_only_reject_routes_are_not_egress():
    observed = sample_host_observation()
    observed["ipv6_route"] = (
        "00000000000000000000000000000000 00 00000000000000000000000000000000 00 "
        "00000000000000000000000000000000 ffffffff 00000001 00000000 00200200       lo\n"
        "00000000000000000000000000000001 80 00000000000000000000000000000000 00 "
        "00000000000000000000000000000000 00000000 00000002 00000000 80200001       lo\n")
    assert witness.verify_external_observation(observed)["outside_verified_no_egress_interfaces"]
    observed["ipv6_route"] += "00000000000000000000000000000000 00 00 00 00 00 00 00 00 eth0\n"
    with pytest.raises(ValueError, match="no-egress"):
        witness.verify_external_observation(observed)


@pytest.mark.parametrize("mutate", [
    lambda w: w.update(child_ns=w["host_ns"]),
    lambda w: w["child_ns"].pop("user"),
    lambda w: w["status"].update(CapEff="0000000000000001"),
    lambda w: w["status"].update(NoNewPrivs="0"),
    lambda w: w["status"].update(Seccomp="0"),
    lambda w: w["status"].update(Uid="0\t0\t0\t0"),
    lambda w: w.update(host_uid=0),
    lambda w: w.update(memory_max="max"),
    lambda w: w.update(pids_max="max"),
    lambda w: w.update(network_interfaces=["lo", "eth0"]),
    lambda w: w.update(ipv4_route="Iface Destination Gateway Flags\neth0 00000000 FFFFFF00 0003\n"),
    lambda w: w.update(root_read_only=False),
    lambda w: w.update(host_mounts_seen=["/mnt/c"]),
    lambda w: w.update(python_executable_sha256="0" * 64),
    lambda w: w.update(descendants={}),
    lambda w: w.update(cgroup_procs="54322\n"),
    lambda w: w.update(child={"pid": 54321, "ppid": 555, "start_ticks": 0, "state": "S"}),
])
def test_every_missing_host_external_isolation_criterion_fails_closed(mutate):
    observation = sample_host_observation()
    mutate(observation)
    with pytest.raises((ValueError, TypeError)):
        witness.verify_external_observation(observation)


def test_duplicate_json_fields_and_nan_are_not_source_proof():
    with pytest.raises(ValueError, match="duplicate"):
        witness._strict_json('{"child":1,"child":1}')
    with pytest.raises(ValueError):
        witness._strict_json('{"outcome":NaN}')
    with pytest.raises(ValueError):
        witness._strict_json('[{"PASS":true}]')


def test_ssh_vm_observer_only_accepts_in_source_code_and_safe_arguments(monkeypatch):
    recorded = []

    def fake(command, **kwargs):
        recorded.append(command)
        return subprocess.CompletedProcess(command, 0, "1000\nTrue\n", "")

    monkeypatch.setattr(witness, "checked", fake)
    with pytest.raises(ValueError, match="unapproved"):
        witness._vm_python_static("print('my arbitrary code')")
    for arg in ("5;whoami", "../../etc/passwd", "1\n2", "-1"):
        with pytest.raises(ValueError, match="unsafe"):
            witness._vm_python_static(witness.VM_HOST_OBSERVER, arg)
    witness._vm_python_static(witness.VM_PREFLIGHT)
    assert recorded[-1][:4] == ["podman", "machine", "ssh", "--"]
    assert len(recorded[-1]) == 5


def test_external_process_teardown_requires_all_original_starttimes(monkeypatch):
    observation = sample_host_observation()
    observed_commands = []

    def fake(fixed, *args, **kwargs):
        observed_commands.append((fixed, args))
        return subprocess.CompletedProcess([], 0,
                                           '{"same_owned_processes_live":[],"cgroup_procs":[]}', "")

    monkeypatch.setattr(witness, "_vm_python_static", fake)
    assert witness._vm_postcheck(observation)["same_owned_processes_live"] == []
    assert observed_commands[0][0] == witness.VM_POSTCHECK
    assert "54321" in observed_commands[0][1]
    assert "54322" in observed_commands[0][1]
    monkeypatch.setattr(witness, "_vm_python_static", lambda *a, **k: subprocess.CompletedProcess(
        [], 0, '{"same_owned_processes_live":[54322],"cgroup_procs":[]}', ""))
    with pytest.raises(RuntimeError, match="extinction"):
        witness._vm_postcheck(observation)


def test_observed_host_vm_without_a_real_container_not_success(monkeypatch):
    monkeypatch.setattr(witness, "preflight", lambda: {"image_id": witness.IMAGE_ID})
    monkeypatch.setattr(witness, "checked", lambda cmd, **kw: subprocess.CompletedProcess(
        cmd, 1, "", "container absent"))
    result = witness.run_fixed_benign_witness()
    assert result["status"] == "REFUSED_INDEPENDENT_BENIGN_HOST_WITNESS"
    assert result["sandbox_qualified_for_generated_code"] is False
    assert result["owned_container_id"] is None


def test_source_has_no_private_model_runners_or_approval_pin_writes():
    source = Path(witness.__file__).read_text(encoding="utf-8")
    assert "APPROVED_SANDBOX_CAPABILITY_SHA256 =" not in source
    assert "APPROVED_INDEPENDENT_WITNESS_SHA256 =" not in source
    assert "podman machine stop" not in source
    assert "podman machine start" not in source
    assert "--pull=always" not in source
    assert "shell=True" not in source
    assert "--privileged" not in witness.container_argv("w10-he-benign-1234567890abcdef")


def negative_log(mode: str):
    observation = {
        "network": {"mode": "network", "connected": False, "errno": 101},
        "filesystem": {"mode": "filesystem", "cwd": "/work/fixed-v37-safename",
                       "inside_writable": True, "outside_denied": [True] * 4,
                       "attempts": ["/etc/fixed-he-negative.txt", "/tmp/fixed-he-negative.txt",
                                    "/fixed-he-negative.txt", "/work/../fixed-he-negative.txt"]},
        "timeout": {"mode": "timeout", "grandchild_spawned": True},
    }[mode]
    return {"kind": "fixed_negative_v37", "mode": mode,
            "candidate_sha256": witness.NEGATIVE_SOURCE_SHA256[mode],
            "python_version": [3, 13],
            "candidate_argv": ["python", "-I", "candidate.py"],
            "timeout_budget_s": 10, "generated_programs_executed": 0,
            "wall_timeout": mode == "timeout", "process_group_killed": mode == "timeout",
            "observation": observation}


def timeout_observation():
    observed = sample_host_observation()
    observed["descendants"]["54323"] = {
        "pid": 54323, "ppid": 54322, "start_ticks": 987654323, "state": "S"}
    return observed


def test_all_v37_benign_negative_sources_are_literally_pinned():
    witness.verify_negative_source_pins()
    assert hashlib.sha256(witness.VM_POSTCHECK.encode()).hexdigest() == witness.VM_POSTCHECK_SOURCE_PIN
    assert hashlib.sha256(witness.VM_PREFLIGHT.encode()).hexdigest() == witness.VM_PREFLIGHT_SOURCE_PIN
    for mode in ("network", "filesystem", "timeout"):
        source = witness.NEGATIVE_CANDIDATES[mode]
        assert hashlib.sha256(source.encode()).hexdigest() == witness.NEGATIVE_SOURCE_SHA256[mode]
        rendered = witness.NEGATIVE_LAUNCHER.replace("'__MODE__'", repr(mode))
        assert hashlib.sha256(rendered.encode()).hexdigest() == witness.NEGATIVE_LAUNCHER_SHA256[mode]
        cmd = witness.negative_container_argv("w10-he-benign-1234567890abcdef", mode)
        assert "--network=none" in cmd and "--read-only" in cmd
        assert "--tmpfs=/work:rw,nosuid,nodev,noexec,size=8m,mode=1777" in cmd
        assert "--tmpfs=/tmp:ro,nosuid,nodev,noexec,size=64k,mode=0555" in cmd
        assert "--tmpfs=/tmp:rw,nosuid,nodev,noexec,size=8m,mode=1777" not in cmd
        assert "--rm" not in cmd and "--workdir=/work" in cmd
        assert cmd[-1] == rendered and cmd[-4] == witness.IMAGE_ID
        assert all(not x.startswith("--volume") for x in cmd)
    assert "1.1.1.1" in witness.NETWORK_DENIAL_CANDIDATE
    assert "socket.connect" not in witness.FILESYSTEM_DENIAL_CANDIDATE
    assert "os.killpg(child.pid,signal.SIGKILL)" in witness.NEGATIVE_LAUNCHER


@pytest.mark.parametrize("mode", ("network", "filesystem", "timeout"))
def test_genuine_host_bound_synthetic_negative_result_passes_only_benign_scope(mode):
    observation = timeout_observation() if mode == "timeout" else sample_host_observation()
    verdict = witness.verify_negative_log(negative_log(mode), mode, observation)
    assert verdict["active_negative_mode"] == mode
    assert verdict["generated_code_approved"] is False
    assert verdict["host_observer_required"] is True


@pytest.mark.parametrize("mode,mutation", [
    ("network", lambda r: r["observation"].update(connected=True, errno=None)),
    ("network", lambda r: r["observation"].update(errno=111)),
    ("network", lambda r: r["observation"].update(errno=None)),
    ("filesystem", lambda r: r["observation"].update(inside_writable=False)),
    ("filesystem", lambda r: r["observation"].update(outside_denied=[True, False, True, True])),
    ("filesystem", lambda r: r["observation"].update(cwd="/tmp/fixed-v37-evil")),
    ("filesystem", lambda r: r["observation"].update(attempts=[])),
    ("timeout", lambda r: r.update(wall_timeout=False)),
    ("timeout", lambda r: r.update(process_group_killed=False)),
    ("timeout", lambda r: r["observation"].update(grandchild_spawned=False)),
    ("timeout", lambda r: r.update(candidate_sha256="0" * 64)),
])
def test_negative_source_or_outcome_forgery_is_refused(mode, mutation):
    row = negative_log(mode)
    mutation(row)
    host = timeout_observation() if mode == "timeout" else sample_host_observation()
    with pytest.raises(ValueError):
        witness.verify_negative_log(row, mode, host)


@pytest.mark.parametrize("mode", ["network", "filesystem", "timeout"])
def test_missing_or_mutated_production_source_hash_fails_before_launch(monkeypatch, mode):
    monkeypatch.setitem(witness.NEGATIVE_SOURCE_SHA256, mode, "")
    with pytest.raises(ValueError, match="source SHA"):
        witness.negative_container_argv("w10-he-benign-1234567890abcdef", mode)


def test_forged_host_observation_does_not_authorize_timeout(monkeypatch):
    host = timeout_observation()
    host["descendants"].pop("54323")
    with pytest.raises(ValueError, match="grandchild"):
        witness.verify_negative_log(negative_log("timeout"), "timeout", host)
    modified = copy.deepcopy(timeout_observation())
    modified["network_interfaces"].append("eth0")
    with pytest.raises(ValueError, match="no-egress"):
        witness.verify_external_observation(modified)
    monkeypatch.setattr(witness, "HOST_WITNESS_SOURCE_PIN", "0" * 64)
    with pytest.raises(ValueError, match="host observer source SHA"):
        witness.verify_negative_source_pins()


def test_missing_vm_teardown_source_sha_is_fail_closed(monkeypatch):
    monkeypatch.setattr(witness, "VM_POSTCHECK_SOURCE_PIN", "")
    with pytest.raises(ValueError, match="teardown observer source SHA"):
        witness.verify_negative_source_pins()


@pytest.mark.parametrize("field,bad", [("tmp_read_only", False),
                                        ("work_writable", False),
                                        ("tmp_read_only", None)])
def test_negative_mount_escape_claim_requires_outside_vm_statvfs(field, bad):
    observed = sample_host_observation()
    observed[field] = bad
    with pytest.raises(ValueError, match="VM-host mount observer"):
        witness.verify_negative_log(negative_log("filesystem"), "filesystem", observed)


def test_live_negative_witness_never_accepts_unowned_container(monkeypatch):
    monkeypatch.setattr(witness, "preflight", lambda: {"image_id": witness.IMAGE_ID})
    monkeypatch.setattr(witness, "checked", lambda cmd, **kw: subprocess.CompletedProcess(
        cmd, 0, "", "some other owner's container"))
    with pytest.raises(RuntimeError, match="exclusive"):
        witness.run_fixed_negative_witness("network")
