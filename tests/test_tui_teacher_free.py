"""Tests for the teacher-free distillation TUI screen."""
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("textual")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chowder.tui_teacher_free import TeacherFreeDistillScreen, stage_state  # noqa: E402

EXP = Path(__file__).resolve().parents[1] / "experiments" / "teacher_free_distill"


def test_stage_state_reads_real_artifacts(tmp_path):
    states = {s["stage"]: s for s in stage_state(tmp_path)}
    assert states["source_review"]["state"] == "missing"
    assert states["data_collection"]["state"] == "pending"
    assert states["training_preflight"]["state"] == "pending"
    assert states["training"]["state"] == "pending"

    (tmp_path / "sources.json").write_text(json.dumps({"sources": {
        "open_thoughts3": {"approved": True},
        "mixture_of_thoughts": {"approved": False}}}))
    (tmp_path / "preflight_result.json").write_text(json.dumps({"ok": True, "passed": 8, "total": 8}))
    (tmp_path / "recipes").mkdir()
    (tmp_path / "recipes" / "a.json").write_text("{}")
    states = {s["stage"]: s for s in stage_state(tmp_path)}
    assert states["source_review"]["state"] == "done"
    assert "mixture_of_thoughts" in states["source_review"]["detail"]
    assert states["training_preflight"]["state"] == "passed"
    assert states["training"]["state"] == "recipes_ready"
    assert "operator authorization" in states["training"]["detail"]


def test_stage_state_against_real_repo_artifacts():
    """The real experiment directory should derive coherent stage state."""
    states = {s["stage"]: s for s in stage_state(EXP)}
    assert states["source_review"]["state"] == "done"
    assert states["training_preflight"]["state"] in ("passed", "failed", "pending")


def test_screen_mounts_in_app():
    from textual.app import App

    class Host(App[None]):
        def compose(self):
            yield TeacherFreeDistillScreen()

    import asyncio
    app = Host()

    async def run_it():
        async with app.run_test() as pilot:
            await pilot.pause()
            text = str(app.query_one("#tfd_stages").render())
            assert "source_review" in text
            assert "Teacher-free distillation pilot" in text

    asyncio.run(run_it())
