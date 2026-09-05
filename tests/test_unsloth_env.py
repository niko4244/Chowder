from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from chowder.cli import build_parser
from chowder.hardware import AcceleratorProfile, HardwareSnapshot
from chowder.unsloth_env import (
    CapabilityCheck,
    UnslothDoctorReport,
    UnslothEnvironmentError,
    doctor_unsloth_environment,
    setup_unsloth_environment,
    unsloth_env_dir,
    unsloth_python,
)


def _hardware(*, gpu: bool = True) -> HardwareSnapshot:
    accelerators = (
        (
            AcceleratorProfile(
                vendor="NVIDIA",
                name="NVIDIA GeForce RTX 5060 Ti",
                memory_gb=16.0,
                index=0,
                compute_capability="12.0",
            ),
        )
        if gpu
        else ()
    )
    return HardwareSnapshot(
        platform="test",
        cpu_count=8,
        ram_gb=64.0,
        storage_total_gb=1000.0,
        storage_free_gb=500.0,
        accelerators=accelerators,
    )


def _good_doctor(root: Path) -> UnslothDoctorReport:
    env = unsloth_env_dir(root)
    return UnslothDoctorReport(
        env_dir=env,
        python_executable=unsloth_python(env),
        python_version="3.13.7",
        platform="test",
        versions={"unsloth": "2026.9.2", "torch": "2.10.0"},
        cuda_version="13.0",
        accelerators=({"name": "NVIDIA GeForce RTX 5060 Ti"},),
        checks=(
            CapabilityCheck("Python", True, "3.13.7"),
            CapabilityCheck("CUDA", True, "13.0"),
            CapabilityCheck("4-bit CUDA support", True, "NF4 forward executed"),
        ),
        process_returncode=0,
    )


def test_environment_python_path_is_platform_correct(tmp_path):
    env = unsloth_env_dir(tmp_path)
    assert unsloth_python(env, system_name="Windows") == env / "Scripts" / "python.exe"
    assert unsloth_python(env, system_name="Linux") == env / "bin" / "python"


def test_setup_refuses_cpu_fallback_before_install(monkeypatch, tmp_path):
    monkeypatch.setattr("chowder.unsloth_env.detect_hardware", lambda root: _hardware(gpu=False))

    with pytest.raises(UnslothEnvironmentError, match="No NVIDIA GPU"):
        setup_unsloth_environment(tmp_path)


def test_setup_uses_uv_auto_torch_backend_and_records_resolved_environment(
    monkeypatch, tmp_path
):
    commands: list[list[str]] = []
    env = unsloth_env_dir(tmp_path)
    python_executable = unsloth_python(env)

    monkeypatch.setattr("chowder.unsloth_env.detect_hardware", lambda root: _hardware())
    monkeypatch.setattr("chowder.unsloth_env._require_uv", lambda: "/fake/uv")
    monkeypatch.setattr(
        "chowder.unsloth_env.doctor_unsloth_environment",
        lambda root: _good_doctor(Path(root)),
    )

    def fake_run(command, *, timeout=300, check=True):
        command = list(command)
        commands.append(command)
        if command[:2] == ["/fake/uv", "--version"]:
            return subprocess.CompletedProcess(command, 0, stdout="uv 0.8.15\n", stderr="")
        if command[:2] == ["/fake/uv", "venv"]:
            python_executable.parent.mkdir(parents=True, exist_ok=True)
            python_executable.touch()
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[:3] == ["/fake/uv", "pip", "check"]:
            return subprocess.CompletedProcess(command, 0, stdout="Checked 42 packages\n", stderr="")
        if command[:3] == ["/fake/uv", "pip", "freeze"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="unsloth==2026.9.2\ntorch==2.10.0\npeft==0.18.1\n",
                stderr="",
            )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("chowder.unsloth_env._run", fake_run)
    result = setup_unsloth_environment(tmp_path)

    install = next(command for command in commands if command[:3] == ["/fake/uv", "pip", "install"])
    assert install == [
        "/fake/uv",
        "pip",
        "install",
        "--python",
        str(python_executable),
        "unsloth==2026.9.2",
        "--torch-backend=auto",
    ]
    assert result.ok
    assert result.frozen_packages == (
        "unsloth==2026.9.2",
        "torch==2.10.0",
        "peft==0.18.1",
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["engine"] == "unsloth"
    assert manifest["requested"] == {
        "python": "3.13",
        "torch_backend": "auto",
        "unsloth_version": "2026.9.2",
    }
    assert manifest["hardware_preflight"]["accelerators"][0]["name"].endswith("5060 Ti")
    assert manifest["doctor"]["ok"] is True
    assert manifest["frozen_packages"][0] == "unsloth==2026.9.2"


def test_setup_surfaces_uv_dependency_conflicts_in_manifest(monkeypatch, tmp_path):
    env = unsloth_env_dir(tmp_path)
    python_executable = unsloth_python(env)
    monkeypatch.setattr("chowder.unsloth_env.detect_hardware", lambda root: _hardware())
    monkeypatch.setattr("chowder.unsloth_env._require_uv", lambda: "/fake/uv")
    monkeypatch.setattr(
        "chowder.unsloth_env.doctor_unsloth_environment",
        lambda root: _good_doctor(Path(root)),
    )

    def fake_run(command, *, timeout=300, check=True):
        command = list(command)
        if command[:2] == ["/fake/uv", "--version"]:
            return subprocess.CompletedProcess(command, 0, stdout="uv 0.8.15\n", stderr="")
        if command[:2] == ["/fake/uv", "venv"]:
            python_executable.parent.mkdir(parents=True, exist_ok=True)
            python_executable.touch()
        if command[:3] == ["/fake/uv", "pip", "check"]:
            return subprocess.CompletedProcess(
                command,
                1,
                stdout="",
                stderr="peft has incompatible dependency transformers",
            )
        if command[:3] == ["/fake/uv", "pip", "freeze"]:
            return subprocess.CompletedProcess(command, 0, stdout="unsloth==2026.9.2\n", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("chowder.unsloth_env._run", fake_run)
    result = setup_unsloth_environment(tmp_path)

    assert not result.ok
    assert result.doctor.checks[0].name == "Dependency graph"
    assert "incompatible dependency" in result.doctor.checks[0].detail
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["doctor"]["ok"] is False


def test_existing_environment_python_mismatch_fails_before_install(monkeypatch, tmp_path):
    env = unsloth_env_dir(tmp_path)
    python_executable = unsloth_python(env)
    python_executable.parent.mkdir(parents=True, exist_ok=True)
    python_executable.touch()
    monkeypatch.setattr("chowder.unsloth_env.detect_hardware", lambda root: _hardware())
    monkeypatch.setattr("chowder.unsloth_env._require_uv", lambda: "/fake/uv")

    def fake_run(command, *, timeout=300, check=True):
        command = list(command)
        if command[:2] == ["/fake/uv", "--version"]:
            return subprocess.CompletedProcess(command, 0, stdout="uv 0.8.15\n", stderr="")
        if command[0] == str(python_executable):
            return subprocess.CompletedProcess(command, 0, stdout="3.12\n", stderr="")
        raise AssertionError(f"unexpected command after version mismatch: {command}")

    monkeypatch.setattr("chowder.unsloth_env._run", fake_run)
    with pytest.raises(UnslothEnvironmentError, match="uses Python 3.12"):
        setup_unsloth_environment(tmp_path, python_request="3.13")


def test_doctor_parses_child_probe_failure_without_hiding_detail(monkeypatch, tmp_path):
    env = unsloth_env_dir(tmp_path)
    python_executable = unsloth_python(env)
    python_executable.parent.mkdir(parents=True, exist_ok=True)
    python_executable.touch()

    def fake_run(command, *, timeout=300, check=True):
        result_path = Path(command[-1])
        result_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "python_version": "3.13.7",
                    "platform": "Windows-11",
                    "versions": {"unsloth": "2026.9.2", "torch": "2.10.0"},
                    "cuda_version": "13.0",
                    "accelerators": [{"name": "NVIDIA GeForce RTX 5060 Ti"}],
                    "checks": [
                        {"name": "Python", "ok": True, "detail": "3.13.7", "required": True},
                        {
                            "name": "Triton",
                            "ok": False,
                            "detail": "ImportError: triton-windows DLL mismatch",
                            "required": True,
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 1, stdout="unsloth banner\n", stderr="trace\n")

    monkeypatch.setattr("chowder.unsloth_env._run", fake_run)
    report = doctor_unsloth_environment(tmp_path)

    assert not report.ok
    assert report.process_returncode == 1
    assert report.checks[1].name == "Triton"
    assert "DLL mismatch" in report.checks[1].detail
    assert report.stdout_tail == "unsloth banner\n"
    assert report.stderr_tail == "trace\n"


def test_doctor_missing_environment_is_a_structured_failure(tmp_path):
    report = doctor_unsloth_environment(tmp_path)
    assert not report.ok
    assert report.checks[0].name == "Environment"
    assert ".chowder" in report.checks[0].detail


def test_cli_exposes_nested_setup_and_doctor_commands(tmp_path):
    parser = build_parser()
    setup = parser.parse_args(
        [
            "setup",
            "unsloth",
            "--root",
            str(tmp_path),
            "--python",
            "3.13",
            "--unsloth-version",
            "2026.9.2",
        ]
    )
    doctor = parser.parse_args(["doctor", "unsloth", "--root", str(tmp_path)])

    assert setup.command == "setup"
    assert setup.setup_target == "unsloth"
    assert setup.python == "3.13"
    assert setup.unsloth_version == "2026.9.2"
    assert doctor.command == "doctor"
    assert doctor.doctor_target == "unsloth"
