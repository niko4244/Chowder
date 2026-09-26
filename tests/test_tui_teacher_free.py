"""Tests for the teacher-free distillation TUI screen.

The screen is a renderer: what it shows comes from
:mod:`chowder.teacher_free_evidence` (tested separately, without Textual).
These tests cover the two things the screen itself owns — where it reads from,
and how it launches the preflight worker.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("textual")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chowder import tui_teacher_free as tui  # noqa: E402
from chowder.teacher_free_evidence import GREEN_STATES, stage_state  # noqa: E402
from chowder.worker_env import chowder_source_root  # noqa: E402

EXP = Path(__file__).resolve().parents[1] / "experiments" / "teacher_free_distill"


def test_screen_reads_the_repo_pilot_directory_by_default(monkeypatch):
    monkeypatch.delenv("TFD_PILOT_DIR", raising=False)
    assert tui.pilot_dir() == EXP


def test_screen_reads_an_override_directory_when_asked(monkeypatch, tmp_path):
    monkeypatch.setenv("TFD_PILOT_DIR", str(tmp_path))
    assert tui.pilot_dir() == tmp_path


def test_preflight_launch_passes_the_worker_environment(tmp_path, monkeypatch):
    """A fresh interpreter would otherwise resolve the editable install and
    validate a different Chowder's masking code as this checkout's evidence."""
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs.get("env")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(tui.subprocess, "run", fake_run)
    tui.run_preflight(tmp_path)
    assert captured["env"] is not None, "preflight launched without env=worker_env()"
    assert captured["env"]["PYTHONPATH"].split(os.pathsep)[0] == str(chowder_source_root())
    assert captured["env"]["PYTHONUNBUFFERED"] == "1"
    assert str(tmp_path / "preflight.py") in captured["command"]


def test_preflight_timeout_never_raises(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd="preflight.py", timeout=1)

    monkeypatch.setattr(tui.subprocess, "run", boom)
    tui.run_preflight(tmp_path)  # must not raise; the screen stays alive


def test_existing_but_unchecked_artifacts_are_not_green(tmp_path, monkeypatch):
    """Regression: an empty replay summary and a bare recipe must not read as
    completed work."""
    (tmp_path / "recipes").mkdir()
    (tmp_path / "recipes" / "a.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "replay_summary.json").write_text(
        json.dumps({"rows": 3, "verified": 0, "not_green": 2, "setup_failed": 1,
                    "skipped": 0}), encoding="utf-8")
    states = {s["stage"]: s for s in stage_state(tmp_path)}
    assert states["training"]["state"] not in GREEN_STATES
    assert states["evaluation"]["state"] not in GREEN_STATES


def test_screen_mounts_and_renders_artifact_state(monkeypatch, tmp_path):
    from textual.app import App

    (tmp_path / "sources.json").write_text(json.dumps({"sources": {
        "ot3": {"approved": True, "license": "Apache-2.0", "revision": "abc",
                "review_reference": "HF API"}}}), encoding="utf-8")
    monkeypatch.setenv("TFD_PILOT_DIR", str(tmp_path))

    class Host(App[None]):
        def compose(self):
            yield tui.TeacherFreeDistillScreen()

    import asyncio
    app = Host()

    async def run_it():
        async with app.run_test() as pilot:
            await pilot.pause()
            text = str(app.query_one("#tfd_stages").render())
            assert "Teacher-free distillation pilot" in text
            assert "source_review" in text
            assert "reviewed" in text
            assert str(tmp_path) in text  # the screen says where it reads from

    asyncio.run(run_it())


def test_screen_state_against_the_real_repo_artifacts():
    """The real experiment directory should derive coherent stage state."""
    states = {s["stage"]: s for s in stage_state(EXP)}
    assert states["source_review"]["state"] == "reviewed"
    assert states["training_preflight"]["state"] in ("passed", "failed", "pending")
