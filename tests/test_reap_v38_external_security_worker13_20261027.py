"""Independent hostile observation checks for the native V38 research boundary."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from chowder_batch import reap_humaneval_v38_actual_execution_20261027 as v38


LOADER_SHA = "3319565d5d5a6aa210151a32bd367a136a08cfb530402c51f311143de2f08e66"
OBSERVER_SHA = "c4e5fd10ae77338337357ee250d8f5e19d925f74cf520c42b2e25b94cc519425"


def test_native_source_hashes_and_research_only_approval_are_exact():
    assert hashlib.sha256(v38.V38_LOADER.encode()).hexdigest() == LOADER_SHA
    assert hashlib.sha256(v38.V38_VM_SOURCE_OBSERVER.encode()).hexdigest() == OBSERVER_SHA
    assert v38.APPROVED_V38_RESEARCH_LOADER_SHA256 == LOADER_SHA
    assert v38.APPROVED_V38_RESEARCH_HOST_OBSERVER_SHA256 == OBSERVER_SHA


def test_profile_retains_rootless_no_network_and_no_host_passwd_mount():
    args = v38._new_research_container_argv("w10-he-v38-1234567890abcdef")
    for switch in ("--passwd=false", "--network=none", "--read-only",
                   "--cap-drop=ALL", "--security-opt=no-new-privileges",
                   "--user=65534:65534", "--pids-limit=16",
                   "--memory=268435456", "--memory-swap=268435456",
                   "--tmpfs=/run/secrets:ro,nosuid,nodev,noexec,size=64k,mode=0555,notmpcopyup"):
        assert switch in args
    assert not any(a.startswith(("--volume", "--mount=", "-v", "--privileged")) for a in args)


def test_candidate_executes_the_sealed_memfd_only():
    source = v38.V38_LOADER
    for contract in ("MFD_ALLOW_SEALING", "F_SEAL_SEAL", "F_SEAL_SHRINK",
                     "F_SEAL_GROW", "F_SEAL_WRITE", "F_ADD_SEALS", "F_GET_SEALS",
                     "pass_fds=(sealed,)", "'/proc/self/fd/'+str(sealed)"):
        assert contract in source
    assert "'candidate.py'],cwd=" not in source


@pytest.mark.parametrize("bad_fd", [-1, 0, 2, 65, "3", None, True])
def test_invalid_immutable_handle_refused_before_podman(monkeypatch, bad_fd):
    monkeypatch.setattr(v38, "_bounded_exec", lambda *a, **kw: pytest.fail("unexpected subprocess"))
    with pytest.raises(ValueError, match="invalid immutable executable handle"):
        v38._source_stage_host_observe(123, "a" * 64, bad_fd)


def test_unverifiable_sealed_fd_rejected(monkeypatch):
    evidence = {"staged_program_sha256": "a" * 64, "immutable_executable_memfd": {
        "pid": 123, "fd": 4, "sha256": "b" * 64, "inode": 12, "seals": 0}}
    monkeypatch.setattr(v38, "_bounded_exec", lambda *a, **kw: SimpleNamespace(
        returncode=0, stderr="", stdout=json.dumps(evidence)))
    with pytest.raises(RuntimeError):
        v38._source_stage_host_observe(123, "a" * 64, 4)


@pytest.mark.parametrize("change", [
    {"net_interfaces": ["lo", "eth0"]},
    {"ipv4_routes": ["Iface Destination Gateway Flags RefCnt Use Metric Mask MTU Window IRTT",
                      "eth0 00000000 0100007F 0003 0 0 0 00000000 0 0 0"]},
    {"ipv6_routes": ["0" * 32 + " " + "0" * 32 + " 00 00000000 00000000 00000000 00000001 eth0"]},
    {"mount_inventory": [{"point": "/etc/passwd", "fstype": "ext4", "source": "/dev/sdd"}]},
    {"mount_inventory": [{"point": "/run/secrets", "fstype": "ext4", "source": "/dev/sdd"}]},
    {"staged_inode": {"device": 2, "inode": 3, "size": 4, "uid": 0,
                       "nlink": 1, "mode": 0o100644, "mtime_ns": 5, "ctime_ns": 6}},
    {"process_status": [{"capeff": "0000000000000001", "nnp": "1",
                         "seccomp": "2", "uid": "65534 65534 65534 65534"}]},
])
def test_forged_host_observations_are_rejected(monkeypatch, change):
    evidence = {"staged_program_sha256": "a" * 64,
                "net_interfaces": ["lo"],
                "ipv4_routes": ["Iface Destination Gateway Flags RefCnt Use Metric Mask MTU Window IRTT"],
                "ipv6_routes": [],
                "mount_inventory": [{"point": "/", "fstype": "overlay", "source": "overlay"},
                                    {"point": "/work", "fstype": "tmpfs", "source": "tmpfs"},
                                    {"point": "/tmp", "fstype": "tmpfs", "source": "tmpfs"}],
                "staged_inode": {"device": 2, "inode": 3, "size": 4, "uid": 65534,
                                 "nlink": 1, "mode": 0o100644, "mtime_ns": 5, "ctime_ns": 6}}
    evidence.update(change)
    result = SimpleNamespace(returncode=0, stdout=json.dumps(evidence), stderr="")
    monkeypatch.setattr(v38, "_bounded_exec", lambda *args, **kwargs: result)
    with pytest.raises(RuntimeError):
        v38._source_stage_host_observe(123, "a" * 64, 4)
