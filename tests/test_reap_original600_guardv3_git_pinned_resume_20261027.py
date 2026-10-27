"""Adversarial no-GPU tests for the Git-pinned guard.v3 continuation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import psutil

from chowder_batch import reap_original600_guardv3_git_pinned_resume_20261027 as resume


def _fake(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
          *, resource_ok: bool = True) -> SimpleNamespace:
    score = {"completed_scored_tasks": 24, "active_unscored_indices": [24],
             "native_sessions_pooled_for_frozen_comparison": False,
             "frozen_gen0_status": "BLOCKED", "chosen": None,
             "release_approved": False}
    monkeypatch.setattr(resume.reader, "inspect", lambda: score)
    monkeypatch.setattr(resume.reader, "ROOT", tmp_path)
    (tmp_path / "owner.json").write_text("fixture", encoding="utf-8")
    monkeypatch.setattr(resume, "competing_model_processes", lambda: [])
    monkeypatch.setattr(resume, "occupied_native_ports", lambda: [])

    def prestart(*_args, **kwargs):
        assert kwargs == {"bypass_conservative": False}
        if not resource_ok:
            raise RuntimeError("real 72k physical floor unmet")
        return {"gpu_free_mib": 15000, "ram_available_gib": 40,
                "commit_available_gib": 45}

    runner = SimpleNamespace(_native_prestart=prestart)
    monkeypatch.setitem(sys.modules, resume.RUNNER_MODULE, runner)
    return runner


@pytest.mark.parametrize("value", [0, -1, 601, True, 1.5])
def test_invalid_max_new_fails_closed(value: object) -> None:
    with pytest.raises(ValueError, match="max-new"):
        resume.preflight(value)


def test_genuine_native_preflight_is_read_only() -> None:
    if os.name != "nt":
        pytest.skip("native Windows-only evaluator")
    owner = resume.reader.ROOT / "owner.json"
    before = hashlib.sha256(owner.read_bytes()).hexdigest()
    result = subprocess.run(
        [sys.executable, "-m", "chowder_batch.reap_original600_guardv3_git_pinned_resume_20261027"],
        cwd=resume.reader.REPO, text=True, capture_output=True, timeout=120,
    )
    data = json.loads(result.stdout)
    assert data["gen0_status"] == "BLOCKED"
    assert data["chosen"] is None and data["release_approved"] is False
    assert data["status"] in ("REFUSED_NATIVE_ADMISSION",
                              "READY_ORIGINAL_GUARDV3_RESUME",
                              "COMPLETE_NO_NATIVE_RESTART")
    assert data["model_executed"] is False
    assert hashlib.sha256(owner.read_bytes()).hexdigest() == before


def test_competing_model_refused_even_with_abundant_memory(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _fake(monkeypatch, tmp_path)
    monkeypatch.setattr(resume, "competing_model_processes",
                        lambda: [{"pid": 123, "name": "llama-server.exe"}])
    result, _ = resume.preflight(600)
    assert result["status"] == "REFUSED_NATIVE_ADMISSION"
    assert "OTHER_NATIVE_MODEL_SERVER_ACTIVE" in result["blockers"]


def test_physically_occupied_port_refused(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _fake(monkeypatch, tmp_path)
    monkeypatch.setattr(resume, "occupied_native_ports", lambda: [8892])
    result, _ = resume.preflight(600)
    assert result["status"] == "REFUSED_NATIVE_ADMISSION"
    assert result["occupied_ports"] == [8892]


def test_conservative_ram_commit_gpu_floor_cannot_be_bypassed(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _fake(monkeypatch, tmp_path, resource_ok=False)
    result, _ = resume.preflight(600)
    assert result["status"] == "REFUSED_NATIVE_ADMISSION"
    assert any("CONSERVATIVE_RESOURCE_FLOOR" in b for b in result["blockers"])


def test_incomplete_task_not_silently_scored(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _fake(monkeypatch, tmp_path)
    (tmp_path / "task-025").mkdir()
    with pytest.raises(ValueError, match="noncontiguous"):
        resume.preflight(600)


def test_foreign_directory_fails_before_gpu_load(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _fake(monkeypatch, tmp_path)
    (tmp_path / "foreign-owner").mkdir()
    with pytest.raises(ValueError, match="foreign"):
        resume.preflight(600)


def test_incomplete_original_task_with_foreign_file_refused(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _fake(monkeypatch, tmp_path)
    item = tmp_path / "task-024"
    item.mkdir()
    (item / "receipt.json").write_text("forged", encoding="utf-8")
    with pytest.raises(ValueError, match="foreign evidence"):
        resume.preflight(600)


def test_ntfs_reparse_path_refused_before_model_load(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _fake(monkeypatch, tmp_path)
    original = resume.assert_native_ordinary

    def reparse(path: Path):
        if path.name == "owner.json":
            raise ValueError("native reparse path in original checkpoint")
        return original(path)

    monkeypatch.setattr(resume, "assert_native_ordinary", reparse)
    with pytest.raises(ValueError, match="native reparse"):
        resume.preflight(600)


def test_shell_mention_is_not_a_model_server(
        monkeypatch: pytest.MonkeyPatch) -> None:
    entries = [
        SimpleNamespace(info={"pid": 111, "name": "bash.exe",
                              "cmdline": ["bash", "-c", "grep llama-server file"]}),
        SimpleNamespace(info={"pid": 112, "name": "llama-server.exe",
                              "cmdline": ["F:/llama.cpp/bin/llama-server.exe"]}),
        SimpleNamespace(info={"pid": 113, "name": "python.exe",
                              "cmdline": ["python", "-m", "vllm"]}),
    ]
    monkeypatch.setattr(psutil, "process_iter", lambda **_kwargs: iter(entries))
    assert resume.competing_model_processes() == [
        {"pid": 112, "name": "llama-server.exe"},
        {"pid": 113, "name": "python.exe"},
    ]


def test_process_inventory_access_failure_is_not_free_gpu(
        monkeypatch: pytest.MonkeyPatch) -> None:
    def deny(**_kwargs):
        raise psutil.AccessDenied(pid=123)

    monkeypatch.setattr(psutil, "process_iter", deny)
    with pytest.raises(ValueError, match="enumeration unavailable"):
        resume.competing_model_processes()


def test_execute_cannot_start_when_competitor_active(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner = _fake(monkeypatch, tmp_path)
    monkeypatch.setattr(resume, "competing_model_processes",
                        lambda: [{"pid": 999, "name": "llama-server.exe"}])
    runner.run = lambda *_args, **_kwargs: pytest.fail("GPU load must not start")
    result = resume.run(execute=True, max_new=600)
    assert result["status"] == "REFUSED_NATIVE_ADMISSION"


def test_race_after_admission_refuses_before_gpu_load(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner = _fake(monkeypatch, tmp_path)
    seen = iter([[], [{"pid": 99, "name": "llama-server.exe"}]])
    monkeypatch.setattr(resume, "competing_model_processes", lambda: next(seen))
    runner.run = lambda *_args, **_kwargs: pytest.fail("GPU load must not start")
    with pytest.raises(ValueError, match="ownership changed"):
        resume.run(execute=True, max_new=600)


def test_only_original_runner_is_invoked_without_resource_override(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner = _fake(monkeypatch, tmp_path)
    called = []

    def original_run(root: Path, **kwargs):
        called.append((root, kwargs))
        return {"status": "ORIGINAL_PUBLIC600_PARTIAL_REAL_EXECUTION_RESUMABLE",
                "gen0_status": "BLOCKED", "chosen": None,
                "release_approved": False,
                "original_frozen_gen0_serving_comparable": False}

    runner.run = original_run
    result = resume.run(execute=True, max_new=2)
    assert called == [(tmp_path, {"execute": True, "max_new": 2,
                                   "bypass_conservative": False})]
    assert result["historical_guard"] == "guard.v3"
    assert result["frozen_four_slot_eligible"] is False


def test_runner_cannot_promote_release_state(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner = _fake(monkeypatch, tmp_path)
    runner.run = lambda *_args, **_kwargs: {
        "gen0_status": "PASS", "chosen": "Q4", "release_approved": True,
        "original_frozen_gen0_serving_comparable": True,
    }
    with pytest.raises(ValueError, match="forbidden release state"):
        resume.run(execute=True, max_new=2)


def test_complete_checkpoint_never_starts_another_native_session(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _fake(monkeypatch, tmp_path)
    monkeypatch.setattr(resume.reader, "inspect", lambda: {
        "completed_scored_tasks": 600, "active_unscored_indices": [],
        "native_sessions_pooled_for_frozen_comparison": False,
        "frozen_gen0_status": "BLOCKED", "chosen": None,
        "release_approved": False,
    })
    result = resume.run(execute=True, max_new=600)
    assert result["status"] == "COMPLETE_NO_NATIVE_RESTART"
    assert result["model_executed"] is False
