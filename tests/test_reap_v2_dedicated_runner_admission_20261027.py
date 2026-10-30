"""Adversarial, CPU-only checks for source-pinned dedicated-host admission."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from chowder_batch import reap_v2_dedicated_runner_admission_20261027 as admission


def census(**changes):
    sample = {
        "physical_total_bytes": 96 * admission.GIB,
        "physical_available_bytes": 65 * admission.GIB,
        "commit_total_bytes": 40 * admission.GIB,
        "commit_limit_bytes": 120 * admission.GIB,
        "commit_headroom_bytes": 80 * admission.GIB,
        "gpu": {
            "uuid": "GPU-12345678-1234-1234-1234-123456789abc",
            "name": "48GiB-class CUDA device",
            "total_mib": 65536,
            "used_mib": 200,
            "free_mib": 65100,
            "reserved_mib": 236,
            "utilization_pct": 0,
        },
        "active_conflicts": [],
        "listeners": [],
        "process_census_complete": True,
        "port_census_complete": True,
        "model_sha256_verified": True,
        "model_bytes_verified": True,
        "native_binary_sha256_verified": True,
        "disk_free_bytes": 50 * admission.GIB,
    }
    sample.update(changes)
    return sample


def test_frozen_model_catalog_is_source_pinned():
    assert admission.FROZEN_FLAGS == (
        "--fit",
        "on",
        "-np",
        "4",
        "-c",
        "16384",
        "--jinja",
    )
    assert admission.FROZEN_MODELS["q4"].bytes == 11_563_915_264
    assert admission.FROZEN_MODELS["q8"].sha256.startswith("89dcaf33")
    assert admission.FROZEN_MODELS["dense_q4"].bytes == 21_166_757_984
    assert admission.FROZEN_MODELS["dense_q8"].bytes == 36_903_139_424
    assert admission.FROZEN_MODELS["dense_q8"].sha256.startswith("2ff57a74")
    assert admission.FROZEN_MODELS["random128_q8"].sha256.startswith("601871e0")
    assert admission.FROZEN_MODELS["bf16_gguf"].bytes == 37_122_439_488


def test_eligible_preflight_is_never_scientific_or_execution_approval():
    result = admission.assess(census(), "q8", 8790)
    assert result["status"] == "READY_FOR_CAPACITY_PROBE_ONLY"
    assert result["reasons"] == []
    assert result["execution_approved"] is False
    assert result["gen0"] == {
        "status": "BLOCKED",
        "chosen": None,
        "release_approved": False,
    }


@pytest.mark.parametrize(
    "change,code",
    [
        ({"physical_available_bytes": 48 * 2**30 - 1}, "RAM_48G_FREE"),
        ({"commit_headroom_bytes": 48 * 2**30 - 1}, "COMMIT_48G_FREE"),
        ({"disk_free_bytes": 8 * 2**30 - 1}, "DISK_8G_FREE"),
        ({"model_sha256_verified": False}, "MODEL_SHA"),
        ({"model_bytes_verified": False}, "MODEL_BYTES"),
        ({"native_binary_sha256_verified": False}, "SERVER_SHA"),
        ({"process_census_complete": False}, "PROCESS_CENSUS"),
        ({"port_census_complete": False}, "PORT_CENSUS"),
        (
            {"active_conflicts": [{"pid": 40080, "name": "python.exe"}]},
            "ACTIVE_PROCESSES",
        ),
        ({"listeners": [{"port": 8890, "pid": 55032}]}, "ACTIVE_LISTENERS"),
    ],
)
def test_resource_and_identity_failures_refuse(change, code):
    result = admission.assess(census(**change), "q8", 8790)
    assert result["status"] == "REFUSED"
    assert code in result["reasons"]


@pytest.mark.parametrize(
    "gpu_change,code",
    [
        (
            {
                "total_mib": 16311,
                "free_mib": 16000,
                "used_mib": 200,
                "reserved_mib": 111,
            },
            "GPU_48G_CLASS",
        ),
        (
            {"free_mib": 12287, "used_mib": 53013, "reserved_mib": 236},
            "GPU_12G_NATIVE_FLOOR",
        ),
        (
            {"free_mib": 39000, "used_mib": 26300, "reserved_mib": 236},
            "GPU_40G_DEDICATED",
        ),
        ({"used_mib": 1200, "free_mib": 64100, "reserved_mib": 236}, "GPU_NOT_IDLE"),
        ({"utilization_pct": 15}, "GPU_NOT_IDLE"),
        ({"reserved_mib": 10}, "GPU_COUNTERS"),
    ],
)
def test_gpu_fail_closed(gpu_change, code):
    sample = census()
    sample["gpu"] = {**sample["gpu"], **gpu_change}
    assert code in admission.assess(sample, "q4", 8790)["reasons"]


@pytest.mark.parametrize(
    "mutated",
    [
        {"physical_available_bytes": True},
        {"commit_headroom_bytes": float("nan")},
        {"disk_free_bytes": "999999999999999"},
        {"listeners": None},
        {"active_conflicts": "none"},
        {"process_census_complete": 1},
        {"gpu": None},
    ],
)
def test_malformed_or_spoofed_telemetry_refuses(mutated):
    result = admission.assess(census(**mutated), "q8", 8790)
    assert result["status"] == "REFUSED"


def test_port_and_arm_are_validated():
    for port in (0, True, 65536, "8790"):
        assert admission.assess(census(), "q4", port)["status"] == "REFUSED"
    assert admission.assess(census(), "fake", 8790)["status"] == "REFUSED"
    assert admission.assess(census(), [], 8790)["status"] == "REFUSED"


def test_gpu_telemetry_parser_rejects_multi_gpu_missing_or_nonnumeric():
    good = "GPU-12345678, NVIDIA 48GB, 65536, 200, 65100, 0\n"
    parsed = admission.parse_gpu(good)
    assert parsed["reserved_mib"] == 236
    for bad in (
        good + good,
        "",
        "GPU-12345678, NVIDIA, 1000, 5, x, 2",
        "GPU-12345678, NVIDIA, 60000, 40000, 40000, 0",
    ):
        with pytest.raises(ValueError):
            admission.parse_gpu(bad)


def test_model_full_sha_size_and_bytes_are_mandatory(tmp_path):
    file = tmp_path / "original.gguf"
    file.write_bytes(b"model-original")
    spec = admission.ModelSpec(
        "fixture", "fixture", 14, hashlib.sha256(b"model-original").hexdigest()
    )
    assert admission.verify_file(file, spec.bytes, spec.sha256)["sha256_verified"]
    file.write_bytes(b"model-tampered")
    with pytest.raises(ValueError, match="SHA256"):
        admission.verify_file(file, spec.bytes, spec.sha256)
    with pytest.raises(ValueError, match="SIZE"):
        admission.verify_file(file, spec.bytes + 1, spec.sha256)


def test_manifest_frozen_command_and_no_launch(tmp_path):
    model = tmp_path / "q8.gguf"
    server = tmp_path / "llama-server.exe"
    manifest = admission.make_manifest("q8", model, server, 8790, census())
    assert manifest["proposed_command"] == [
        str(server),
        *admission.FROZEN_FLAGS,
        "-m",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        "8790",
    ]
    assert manifest["report"]["execution_approved"] is False
    assert manifest["original_model"]["sha256"] == admission.FROZEN_MODELS["q8"].sha256
    assert manifest["never_launches_model"] is True
    assert manifest["recheck_before_any_run"] is True


def test_protected_pid_and_port_are_detected_without_touching_them(monkeypatch):
    fake_process = SimpleNamespace(
        info={
            "pid": 40080,
            "name": "python.exe",
            "cmdline": ["python", "F:/lb_run.py", "livebench"],
            "create_time": 123456.0,
        }
    )
    listener = SimpleNamespace(
        status=admission.psutil.CONN_LISTEN, laddr=SimpleNamespace(port=8890), pid=55032
    )
    monkeypatch.setattr(admission.psutil, "process_iter", lambda _: [fake_process])
    monkeypatch.setattr(admission.psutil, "net_connections", lambda **_: [listener])
    data = admission.read_active_processes_and_ports(8790)
    assert data["active_conflicts"][0]["pid"] == 40080
    assert data["listeners"] == [{"port": 8890, "pid": 55032}]
    assert "ACTIVE_PROCESSES" in admission.assess(census(**data), "q4", 8790)["reasons"]


def test_native_snapshot_skips_file_hash_on_occupied_host(monkeypatch, tmp_path):
    monkeypatch.setattr(admission.sys, "platform", "win32")
    occupied = census(active_conflicts=[{"pid": 55032, "name": "llama-server.exe"}])
    process = {
        name: occupied[name]
        for name in (
            "active_conflicts",
            "listeners",
            "process_census_complete",
            "port_census_complete",
        )
    }
    memory = {
        name: occupied[name]
        for name in (
            "physical_total_bytes",
            "physical_available_bytes",
            "commit_total_bytes",
            "commit_limit_bytes",
            "commit_headroom_bytes",
        )
    }
    monkeypatch.setattr(admission, "read_active_processes_and_ports", lambda _: process)
    monkeypatch.setattr(admission, "read_windows_memory", lambda: memory)
    monkeypatch.setattr(admission, "read_gpu", lambda: occupied["gpu"])

    def no_hash(*_):
        raise AssertionError("host already refused: must not hash huge weights")

    monkeypatch.setattr(admission, "verify_file", no_hash)
    observed = admission.snapshot(
        model_path=tmp_path / "missing.gguf",
        server_path=tmp_path / "missing.exe",
        arm="q4",
        disk_root=tmp_path,
    )
    assert observed["model_sha256_verified"] is False
    assert "ACTIVE_PROCESSES" in admission.admit(observed, "q4")["reasons"]


def test_non_windows_native_snapshot_refuses_without_side_effects(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(admission.sys, "platform", "linux")
    with pytest.raises(RuntimeError, match="WINDOWS_REQUIRED"):
        admission.snapshot(
            model_path=tmp_path / "unused", server_path=tmp_path / "unused"
        )


def test_output_is_exclusive_and_non_overwriting(tmp_path):
    output = tmp_path / "dedicated.json"
    admission.write_exclusive(
        output, {"status": "REFUSED", "reasons": ["RAM_48G_FREE"]}
    )
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        admission.write_exclusive(output, {"status": "FORGED_PASS"})
    assert output.read_bytes() == original
