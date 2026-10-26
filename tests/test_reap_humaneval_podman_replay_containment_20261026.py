"""CPU-only, mocked adversarial independent replay containment contracts."""

from __future__ import annotations

from types import SimpleNamespace
import subprocess

import pytest

from chowder_batch import reap_humaneval_podman_replay as replay


NAME = "reap-q8-independent-human-95876063-4"


def test_exact_replay_command_has_no_mount_or_network_exceptions():
    args = replay.replay_profile_command(NAME)
    for required in (
        "--pull=never", "--network=none", "--read-only", "--cap-drop=ALL",
        "--security-opt=no-new-privileges", "--user=65534:65534",
        "--userns=keep-id", "--pid=private", "--ipc=private",
        "--memory=805306368", "--memory-swap=805306368", "--pids-limit=16",
        "--timeout=14", "--init", "--ulimit=cpu=10:11",
        "--tmpfs=/run/secrets:rw,nosuid,nodev,noexec,size=64k",
    ):
        assert required in args
    assert all(not arg.startswith(("--mount", "--volume", "--device", "--privileged"))
               for arg in args)
    assert args[-2:] == ["-I", "-"]


def test_timeout_must_not_claim_killed_with_failed_container_cleanup(monkeypatch):
    """A Podman kill/rm error cannot silently turn into a safe timeout outcome."""
    commands = []

    def run(args, **kwargs):
        commands.append(tuple(args[:3]))
        if args[:2] == ["podman", "run"]:
            raise subprocess.TimeoutExpired(args, 23)
        if args[1:3] == ["container", "exists"]:
            return SimpleNamespace(returncode=1 if commands.count(
                ("podman", "container", "exists")) == 1 else 0)
        # The container survived even if kill/rm reported errors.
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(replay, "host_safety_guard", lambda: None)
    monkeypatch.setattr(replay.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="container|reap|owner"):
        replay.podman_exec("print('fixed benign')\n", NAME)
    assert any("kill" in part for cmd in commands for part in cmd)
    assert any("rm" in part for cmd in commands for part in cmd)


def test_timeout_must_verify_container_absent_after_cleanup(monkeypatch):
    calls = []

    def run(args, **kwargs):
        calls.append(tuple(args[:3]))
        if args[:2] == ["podman", "run"]:
            raise subprocess.TimeoutExpired(args, 23)
        if args[1:3] == ["container", "exists"]:
            return SimpleNamespace(returncode=1 if calls.count(
                ("podman", "container", "exists")) == 1 else 0)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(replay, "host_safety_guard", lambda: None)
    monkeypatch.setattr(replay.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="container|reap|owner"):
        replay.podman_exec("print(4)\n", NAME)
    assert ("podman", "container", "exists") in calls


def test_safe_timeout_records_failure_only_after_verified_cleanup(monkeypatch):
    counts = {"exists": 0}

    def run(args, **kwargs):
        if args[:2] == ["podman", "run"]:
            raise subprocess.TimeoutExpired(args, 23)
        if args[1:3] == ["container", "exists"]:
            counts["exists"] += 1
            return SimpleNamespace(returncode=1)  # Confirmed absent.
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(replay, "host_safety_guard", lambda: None)
    monkeypatch.setattr(replay.subprocess, "run", run)
    passed, reason, code, _elapsed = replay.podman_exec("print(4)\n", NAME)
    assert (passed, reason, code) == (False, "WALL_TIMEOUT_CONTAINER_KILLED", None)


def test_unexpected_or_foreign_container_name_refused_without_host_command(monkeypatch):
    calls = []
    monkeypatch.setattr(replay, "host_safety_guard", lambda: None)
    monkeypatch.setattr(replay.subprocess, "run", lambda *a, **kw: calls.append(a))
    with pytest.raises(ValueError, match="container|name|owner"):
        replay.podman_exec("print(4)\n", "unrelated-hermes-redis")
    assert not calls


def test_unterminated_humaneval_does_not_render_or_execute_generated_text(monkeypatch):
    """Nonterminated records are always 0, without candidate source handling."""
    import argparse

    input_rows = [
        {"id": f"HumanEval/{i}", "correct": False,
         "terminated": False, "text": "x" * (3 * 1024 * 1024) if i == 0 else ""}
        for i in range(164)
    ]
    canonical = [
        {"id": f"HumanEval/{i}", "test": "", "header": "", "entry_point": "candidate"}
        for i in range(164)
    ]
    monkeypatch.setattr(replay, "host_safety_guard", lambda: None)
    monkeypatch.setattr(replay, "EXECUTABLE_REPLAY_QUALIFIED", True)
    monkeypatch.setattr(replay, "check_source_binding",
                        lambda _: (canonical, input_rows, {}))
    monkeypatch.setattr(replay, "exact_program",
                        lambda *_: pytest.fail("unterminated code rendered"))
    monkeypatch.setattr(replay, "podman_exec",
                        lambda *_: pytest.fail("unterminated code executed"))
    data = replay.replay(argparse.Namespace())
    assert data["source_rows"] == 164
    assert data["actually_executed_programs"] == 0
    assert all(row["program_sha256"] is None for row in data["per_task"])


def test_unqualified_real_generated_execution_is_source_disabled(monkeypatch):
    import argparse

    monkeypatch.setattr(
        replay, "check_source_binding",
        lambda *_: pytest.fail("unqualified generated replay entered preflight"),
    )
    monkeypatch.setattr(
        replay, "podman_exec",
        lambda *_: pytest.fail("unqualified generated replay executed a program"),
    )
    assert replay.EXECUTABLE_REPLAY_QUALIFIED is False
    with pytest.raises(RuntimeError, match="generated HumanEval execution disabled"):
        replay.replay(argparse.Namespace())
