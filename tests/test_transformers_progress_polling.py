from __future__ import annotations

import json

from chowder.backends.transformers_peft import TransformersPeftExecutor
from chowder.run_events import TrainingProgressEvent


def test_poll_progress_once_returns_last_step_when_file_missing(tmp_path):
    executor = TransformersPeftExecutor()
    result = executor._poll_progress_once(tmp_path / "progress.json", "e1", None)
    assert result is None


def test_poll_progress_once_reports_and_returns_new_step(tmp_path):
    progress_path = tmp_path / "progress.json"
    progress_path.write_text(
        json.dumps(
            {
                "step": 10,
                "max_steps": 100,
                "epoch": 0.5,
                "loss": 1.234,
                "learning_rate": 2e-4,
                "wall_seconds": 12.5,
            }
        ),
        encoding="utf-8",
    )
    seen: list[TrainingProgressEvent] = []
    executor = TransformersPeftExecutor()
    executor.bind_progress_callback(seen.append)

    result = executor._poll_progress_once(progress_path, "e1", None)

    assert result == 10
    assert len(seen) == 1
    event = seen[0]
    assert event.experiment_id == "e1"
    assert event.step == 10
    assert event.max_steps == 100
    assert event.epoch == 0.5
    assert event.loss == 1.234
    assert event.learning_rate == 2e-4
    assert event.wall_seconds == 12.5


def test_poll_progress_once_is_a_noop_when_step_is_unchanged(tmp_path):
    progress_path = tmp_path / "progress.json"
    progress_path.write_text(json.dumps({"step": 5}), encoding="utf-8")
    seen: list[TrainingProgressEvent] = []
    executor = TransformersPeftExecutor()
    executor.bind_progress_callback(seen.append)

    result = executor._poll_progress_once(progress_path, "e1", last_step=5)

    assert result == 5
    assert seen == []


def test_poll_progress_once_reports_again_when_step_advances(tmp_path):
    progress_path = tmp_path / "progress.json"
    progress_path.write_text(json.dumps({"step": 6}), encoding="utf-8")
    seen: list[TrainingProgressEvent] = []
    executor = TransformersPeftExecutor()
    executor.bind_progress_callback(seen.append)

    result = executor._poll_progress_once(progress_path, "e1", last_step=5)

    assert result == 6
    assert len(seen) == 1
    assert seen[0].step == 6


def test_poll_progress_once_tolerates_malformed_json(tmp_path):
    progress_path = tmp_path / "progress.json"
    progress_path.write_text("not json{{{", encoding="utf-8")
    seen: list[TrainingProgressEvent] = []
    executor = TransformersPeftExecutor()
    executor.bind_progress_callback(seen.append)

    result = executor._poll_progress_once(progress_path, "e1", last_step=3)

    assert result == 3
    assert seen == []


def test_poll_progress_once_tolerates_non_object_json(tmp_path):
    progress_path = tmp_path / "progress.json"
    progress_path.write_text("[1, 2, 3]", encoding="utf-8")
    executor = TransformersPeftExecutor()
    result = executor._poll_progress_once(progress_path, "e1", last_step=None)
    assert result is None


def test_poll_progress_once_survives_a_raising_callback(tmp_path):
    """A caller's callback misbehaving must not propagate out of the
    poller -- the worst case is a missed update, not a crashed run."""
    progress_path = tmp_path / "progress.json"
    progress_path.write_text(json.dumps({"step": 1}), encoding="utf-8")
    executor = TransformersPeftExecutor()

    def boom(event):
        raise RuntimeError("ui callback exploded")

    executor.bind_progress_callback(boom)
    result = executor._poll_progress_once(progress_path, "e1", last_step=None)
    assert result == 1  # step still advances even though the callback failed


def test_poll_progress_once_with_no_bound_callback_still_advances_step(tmp_path):
    progress_path = tmp_path / "progress.json"
    progress_path.write_text(json.dumps({"step": 7}), encoding="utf-8")
    executor = TransformersPeftExecutor()  # bind_progress_callback never called
    result = executor._poll_progress_once(progress_path, "e1", last_step=None)
    assert result == 7


def test_bind_progress_callback_can_be_cleared():
    executor = TransformersPeftExecutor()
    seen = []
    executor.bind_progress_callback(seen.append)
    executor.bind_progress_callback(None)
    assert executor._progress_callback is None
