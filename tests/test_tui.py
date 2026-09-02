from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from textual.widgets import Button, Static

from chowder.backends.transformers_peft import TransformersPeftExecutor, TransformersPeftRunSpec
from chowder.cancellation import CancellationToken
from chowder.hardware import AcceleratorProfile, HardwareSnapshot
from chowder.memory_preflight import MemoryEstimate
from chowder.models import Experiment, ExperimentResult, Hypothesis
from chowder.project import ProjectValidationError
from chowder.provenance import sha256_file
from chowder.recursive_repair import RecursiveRepairStopReason
from chowder.registry import RunRegistry
from chowder.run_events import (
    CheckpointEvent,
    FailureEvent,
    PromotionEvent,
    RepairEvent,
    RunEvent,
    TrainingProgressEvent,
)
from chowder.tui import ChowderTUI


def _snapshot(n_gpus: int) -> HardwareSnapshot:
    return HardwareSnapshot(
        platform="Linux",
        cpu_count=8,
        ram_gb=32.0,
        storage_total_gb=200.0,
        storage_free_gb=150.0,
        accelerators=tuple(
            AcceleratorProfile("nvidia", f"GPU{i}", 15.0, index=i) for i in range(n_gpus)
        ),
    )


def _set(app: ChowderTUI, **input_overrides: str) -> None:
    for widget_id, value in input_overrides.items():
        app.query_one(f"#{widget_id}").value = value


def _write_matching_checkpoint(app: ChowderTUI, work_dir: Path, *, step: int) -> Path:
    """A checkpoint whose manifest is derived from the app's own current
    payload, so it is guaranteed valid against whatever _build_payload()
    currently produces -- rather than hand-duplicating its config shape and
    risking drift from the real generator."""
    payload = app._build_payload()
    config = payload["config"]
    spec = TransformersPeftRunSpec.from_resolved_config(
        config,
        work_dir=work_dir,
        output_dir=work_dir / "unused",
        seed=1,
        hardware=app._current_execution_context().hardware,
    )
    if spec.dataset_sha256 is None:
        spec = replace(spec, dataset_sha256=sha256_file(spec.dataset))
    bound_inputs = dict(TransformersPeftExecutor._bound_inputs(spec))

    trainer_dir = work_dir / ".chowder" / "runs" / "e1-abc" / "adapter" / "trainer"
    checkpoint_dir = trainer_dir / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True)
    (trainer_dir / "chowder-checkpoint-manifest.json").write_text(
        json.dumps(bound_inputs), encoding="utf-8"
    )
    return checkpoint_dir


@pytest.mark.asyncio
async def test_quantization_auto_omits_the_key(tmp_path):
    app = ChowderTUI(project_path=str(tmp_path / "project.json"))
    async with app.run_test():
        payload = app._build_payload()
    assert "quantization" not in payload["config"]["backend"]


@pytest.mark.asyncio
async def test_quantization_explicit_value_is_included(tmp_path):
    app = ChowderTUI(project_path=str(tmp_path / "project.json"))
    async with app.run_test():
        _set(app, quantization="4bit")
        payload = app._build_payload()
    assert payload["config"]["backend"]["quantization"] == "4bit"


@pytest.mark.asyncio
async def test_gradient_checkpointing_auto_omits_the_key(tmp_path):
    app = ChowderTUI(project_path=str(tmp_path / "project.json"))
    async with app.run_test():
        payload = app._build_payload()
    assert "gradient_checkpointing" not in payload["config"]["backend"]["training"]


@pytest.mark.asyncio
async def test_gradient_checkpointing_explicit_true_is_included(tmp_path):
    app = ChowderTUI(project_path=str(tmp_path / "project.json"))
    async with app.run_test():
        _set(app, gradient_checkpointing="true")
        payload = app._build_payload()
    assert payload["config"]["backend"]["training"]["gradient_checkpointing"] is True


@pytest.mark.asyncio
async def test_gradient_checkpointing_explicit_false_is_included(tmp_path):
    app = ChowderTUI(project_path=str(tmp_path / "project.json"))
    async with app.run_test():
        _set(app, gradient_checkpointing="FALSE")
        payload = app._build_payload()
    assert payload["config"]["backend"]["training"]["gradient_checkpointing"] is False


@pytest.mark.asyncio
async def test_gradient_checkpointing_invalid_value_raises(tmp_path):
    app = ChowderTUI(project_path=str(tmp_path / "project.json"))
    async with app.run_test():
        _set(app, gradient_checkpointing="sometimes")
        with pytest.raises(ProjectValidationError, match="gradient checkpointing"):
            app._build_payload()


@pytest.mark.asyncio
async def test_active_accelerator_count_auto_uses_detected_gpu_count(tmp_path):
    app = ChowderTUI(project_path=str(tmp_path / "project.json"))
    async with app.run_test():
        app._hardware = _snapshot(2)
        payload = app._build_payload()
    assert payload["config"]["backend"]["runtime"]["active_accelerator_count"] == 2


@pytest.mark.asyncio
async def test_active_accelerator_count_auto_with_no_hardware_scanned_yet_defaults_to_zero(
    tmp_path,
):
    """The background hardware scan on_mount() kicks off can genuinely
    finish before this test's own code runs (a real race, not just a local
    timing accident -- observed passing locally and failing on CI), so this
    forces the "not scanned yet" state directly rather than hoping the scan
    hasn't completed."""
    app = ChowderTUI(project_path=str(tmp_path / "project.json"))
    async with app.run_test():
        app._hardware = None
        payload = app._build_payload()
    assert payload["config"]["backend"]["runtime"]["active_accelerator_count"] == 0


@pytest.mark.asyncio
async def test_active_accelerator_count_explicit_value_overrides_detected_count(tmp_path):
    app = ChowderTUI(project_path=str(tmp_path / "project.json"))
    async with app.run_test():
        _set(app, active_accelerator_count="1")
        app._hardware = _snapshot(2)
        payload = app._build_payload()
    assert payload["config"]["backend"]["runtime"]["active_accelerator_count"] == 1


@pytest.mark.asyncio
async def test_active_accelerator_count_negative_raises(tmp_path):
    app = ChowderTUI(project_path=str(tmp_path / "project.json"))
    async with app.run_test():
        _set(app, active_accelerator_count="-1")
        with pytest.raises(ProjectValidationError, match="negative"):
            app._build_payload()


@pytest.mark.asyncio
async def test_active_accelerator_count_non_integer_raises(tmp_path):
    app = ChowderTUI(project_path=str(tmp_path / "project.json"))
    async with app.run_test():
        _set(app, active_accelerator_count="two")
        with pytest.raises(ProjectValidationError, match="'auto' or an integer"):
            app._build_payload()


@pytest.mark.asyncio
async def test_precision_auto_is_passed_through_as_an_explicit_value(tmp_path):
    """Unlike quantization/gradient_checkpointing, precision's "auto" is a
    real value the worker itself resolves -- it must never be omitted."""
    app = ChowderTUI(project_path=str(tmp_path / "project.json"))
    async with app.run_test():
        payload = app._build_payload()
    assert payload["config"]["backend"]["precision"] == "auto"


# --- checkpoint / resume -------------------------------------------------


@pytest.mark.asyncio
async def test_save_strategy_no_omits_checkpoint_keys(tmp_path):
    app = ChowderTUI(project_path=str(tmp_path / "project.json"))
    async with app.run_test():
        payload = app._build_payload()
    training = payload["config"]["backend"]["training"]
    assert "save_strategy" not in training
    assert "save_steps" not in training
    assert "save_total_limit" not in training
    assert "resume_from_checkpoint" not in payload["config"]["backend"]


@pytest.mark.asyncio
async def test_save_strategy_epoch_is_included_without_save_steps(tmp_path):
    app = ChowderTUI(project_path=str(tmp_path / "project.json"))
    async with app.run_test():
        _set(app, save_strategy="epoch")
        payload = app._build_payload()
    training = payload["config"]["backend"]["training"]
    assert training["save_strategy"] == "epoch"
    assert "save_steps" not in training


@pytest.mark.asyncio
async def test_save_strategy_steps_requires_and_includes_save_steps(tmp_path):
    app = ChowderTUI(project_path=str(tmp_path / "project.json"))
    async with app.run_test():
        _set(app, save_strategy="steps", save_steps="50")
        payload = app._build_payload()
    training = payload["config"]["backend"]["training"]
    assert training["save_strategy"] == "steps"
    assert training["save_steps"] == 50


@pytest.mark.asyncio
async def test_save_total_limit_included_only_when_set(tmp_path):
    app = ChowderTUI(project_path=str(tmp_path / "project.json"))
    async with app.run_test():
        _set(app, save_strategy="epoch", save_total_limit="3")
        payload = app._build_payload()
    assert payload["config"]["backend"]["training"]["save_total_limit"] == 3


@pytest.mark.asyncio
async def test_invalid_save_strategy_raises(tmp_path):
    app = ChowderTUI(project_path=str(tmp_path / "project.json"))
    async with app.run_test():
        _set(app, save_strategy="sometimes")
        with pytest.raises(ProjectValidationError, match="save strategy"):
            app._build_payload()


@pytest.mark.asyncio
async def test_resume_from_checkpoint_included_only_when_set(tmp_path):
    app = ChowderTUI(project_path=str(tmp_path / "project.json"))
    async with app.run_test():
        payload = app._build_payload()
        assert "resume_from_checkpoint" not in payload["config"]["backend"]
        _set(app, resume_from_checkpoint="/ckpt/run-1/checkpoint-100")
        payload = app._build_payload()
    assert (
        payload["config"]["backend"]["resume_from_checkpoint"]
        == "/ckpt/run-1/checkpoint-100"
    )


# --- autonomous repair ----------------------------------------------------


@pytest.mark.asyncio
async def test_repair_omitted_when_corpus_is_blank(tmp_path):
    app = ChowderTUI(project_path=str(tmp_path / "project.json"))
    async with app.run_test():
        payload = app._build_payload()
    assert "repair" not in payload


@pytest.mark.asyncio
async def test_repair_section_built_when_corpus_is_set(tmp_path):
    app = ChowderTUI(project_path=str(tmp_path / "project.json"))
    async with app.run_test():
        _set(
            app,
            repair_corpus="repair.jsonl",
            repair_extra_epochs="2.0",
            repair_estimated_gpu_hours="0.5",
            repair_max_depth="3",
        )
        payload = app._build_payload()
    repair = payload["repair"]
    assert repair["corpus_files"] == ["repair.jsonl"]
    assert repair["policy"]["max_depth"] == 3
    variant = repair["variants"][0]
    assert variant["estimated_gpu_hours"] == 0.5
    assert variant["training_patch"] == {"epochs": 2.0}
    assert variant["name"]


# --- cancel ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_button_starts_disabled(tmp_path):
    app = ChowderTUI(project_path=str(tmp_path / "project.json"))
    async with app.run_test():
        assert app.query_one("#cancel", Button).disabled is True


@pytest.mark.asyncio
async def test_set_running_enables_and_disables_cancel_button(tmp_path):
    app = ChowderTUI(project_path=str(tmp_path / "project.json"))
    async with app.run_test():
        app._set_running(True)
        assert app.query_one("#cancel", Button).disabled is False
        app._set_running(False)
        assert app.query_one("#cancel", Button).disabled is True


@pytest.mark.asyncio
async def test_cancel_press_requests_the_active_token(tmp_path):
    app = ChowderTUI(project_path=str(tmp_path / "project.json"))
    async with app.run_test():
        token = CancellationToken()
        app._cancellation = token
        app.on_button_pressed(Button.Pressed(app.query_one("#cancel", Button)))
    assert token.requested is True


@pytest.mark.asyncio
async def test_cancel_press_with_no_active_run_is_a_noop(tmp_path):
    app = ChowderTUI(project_path=str(tmp_path / "project.json"))
    async with app.run_test():
        assert app._cancellation is None
        app.on_button_pressed(Button.Pressed(app.query_one("#cancel", Button)))  # must not raise


# --- history -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_history_reports_a_missing_registry(tmp_path):
    app = ChowderTUI(project_path=str(tmp_path / "project.json"))
    async with app.run_test():
        _set(app, work_dir=str(tmp_path))
        summary = app._history_summary()
    assert "No run history found" in summary


@pytest.mark.asyncio
async def test_history_reports_an_empty_registry(tmp_path):
    registry_path = tmp_path / ".chowder" / "runs.db"
    registry_path.parent.mkdir(parents=True)
    with RunRegistry(registry_path):
        pass

    app = ChowderTUI(project_path=str(tmp_path / "project.json"))
    async with app.run_test():
        _set(app, work_dir=str(tmp_path))
        summary = app._history_summary()
    assert "No results recorded yet" in summary


@pytest.mark.asyncio
async def test_history_summarizes_recorded_results(tmp_path):
    registry_path = tmp_path / ".chowder" / "runs.db"
    registry_path.parent.mkdir(parents=True)
    with RunRegistry(registry_path) as registry:
        registry.record_experiment(
            Experiment("e1", None, Hypothesis("obs", "cause", "fix"), {}, 0.5)
        )
        registry.record_result(
            ExperimentResult("e1", {"quality": 0.9}, 0.3, artifact_ref="/adapter")
        )

    app = ChowderTUI(project_path=str(tmp_path / "project.json"))
    async with app.run_test():
        _set(app, work_dir=str(tmp_path))
        summary = app._history_summary()
    assert "e1" in summary
    assert "quality=0.9000" in summary
    assert "adapter" in summary


@pytest.mark.asyncio
async def test_history_button_press_does_not_raise(tmp_path):
    """Exercises the same dispatch path _history_summary()'s own content is
    already verified through above -- RichLog doesn't expose a plain-text
    getter to also assert on what got logged here."""
    app = ChowderTUI(project_path=str(tmp_path / "project.json"))
    async with app.run_test():
        _set(app, work_dir=str(tmp_path))
        app.on_button_pressed(Button.Pressed(app.query_one("#history", Button)))


# --- cancel: races, repeated clicks, and status wording ----------------------


def _fake_outcome(*, candidate_error=None, repair_stop_reason=None, promoted_experiment_id=None):
    candidate = SimpleNamespace(error=candidate_error, artifact=None)
    generation = SimpleNamespace(candidates=(candidate,), promoted=None)
    repair = (
        SimpleNamespace(stop_reason=repair_stop_reason) if repair_stop_reason is not None else None
    )
    return SimpleNamespace(
        generation=generation,
        repair=repair,
        promoted_experiment_id=promoted_experiment_id,
    )


@pytest.mark.asyncio
async def test_start_then_immediate_cancel_reaches_run_project_already_requested(
    tmp_path, monkeypatch
):
    """Simulates pressing Start then Cancel before the worker thread has
    necessarily run at all -- the token passed into run_project() must
    already carry the cancellation, since it is created on the main thread
    before the Cancel button is even enabled."""
    captured = {}

    def fake_run_project(project_path, *, on_event=None, cancellation=None):
        captured["cancellation"] = cancellation
        return _fake_outcome(candidate_error="cancelled before start")

    monkeypatch.setattr("chowder.tui.run_project", fake_run_project)

    app = ChowderTUI(project_path=str(tmp_path / "project.json"))
    async with app.run_test():
        _set(app, work_dir=str(tmp_path))
        app.on_button_pressed(Button.Pressed(app.query_one("#start", Button)))
        # Enabled and bound immediately -- no window where a click would
        # find self._cancellation still None.
        assert app.query_one("#cancel", Button).disabled is False
        assert app._cancellation is not None
        app.on_button_pressed(Button.Pressed(app.query_one("#cancel", Button)))
        assert app._cancellation.requested is True
        await app._training_worker.wait()

    assert captured["cancellation"] is not None
    assert captured["cancellation"].requested is True


@pytest.mark.asyncio
async def test_repeated_cancel_clicks_do_not_raise(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "chowder.tui.run_project",
        lambda *a, **k: _fake_outcome(candidate_error="cancelled before start"),
    )
    app = ChowderTUI(project_path=str(tmp_path / "project.json"))
    async with app.run_test():
        _set(app, work_dir=str(tmp_path))
        app.on_button_pressed(Button.Pressed(app.query_one("#start", Button)))
        cancel_button = app.query_one("#cancel", Button)
        for _ in range(5):
            app.on_button_pressed(Button.Pressed(cancel_button))
        assert app._cancellation.requested is True
        await app._training_worker.wait()


@pytest.mark.asyncio
async def test_status_reads_cancelled_when_the_candidate_was_cancelled(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "chowder.tui.run_project",
        lambda *a, **k: _fake_outcome(candidate_error="cancelled: RuntimeError: worker terminated"),
    )
    app = ChowderTUI(project_path=str(tmp_path / "project.json"))
    async with app.run_test():
        _set(app, work_dir=str(tmp_path))
        app.on_button_pressed(Button.Pressed(app.query_one("#start", Button)))
        await app._training_worker.wait()
        assert app.query_one("#status", Static).content == "Cancelled"


@pytest.mark.asyncio
async def test_status_reads_cancelled_when_the_repair_loop_stopped_as_cancelled(
    tmp_path, monkeypatch
):
    """The last-run candidate can complete normally right before the token
    was set -- only the repair loop's own stop reason says this run was
    cancelled. That must still read as "Cancelled", not as an unpromoted
    success."""
    monkeypatch.setattr(
        "chowder.tui.run_project",
        lambda *a, **k: _fake_outcome(
            candidate_error=None, repair_stop_reason=RecursiveRepairStopReason.CANCELLED
        ),
    )
    app = ChowderTUI(project_path=str(tmp_path / "project.json"))
    async with app.run_test():
        _set(app, work_dir=str(tmp_path))
        app.on_button_pressed(Button.Pressed(app.query_one("#start", Button)))
        await app._training_worker.wait()
        assert app.query_one("#status", Static).content == "Cancelled"


@pytest.mark.asyncio
async def test_status_reads_failed_for_a_non_cancellation_error(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "chowder.tui.run_project",
        lambda *a, **k: _fake_outcome(candidate_error="RuntimeError: dataset not found"),
    )
    app = ChowderTUI(project_path=str(tmp_path / "project.json"))
    async with app.run_test():
        _set(app, work_dir=str(tmp_path))
        app.on_button_pressed(Button.Pressed(app.query_one("#start", Button)))
        await app._training_worker.wait()
        status = app.query_one("#status", Static).content
        assert status.startswith("Failed:")
        assert "Cancelled" not in status


# --- checkpoint discovery ------------------------------------------------


@pytest.mark.asyncio
async def test_discover_checkpoints_finds_and_reports_a_valid_checkpoint(tmp_path):
    (tmp_path / "train.jsonl").write_text('{"text":"hello"}\n', encoding="utf-8")
    app = ChowderTUI(project_path=str(tmp_path / "project.json"))
    async with app.run_test():
        _set(app, work_dir=str(tmp_path))
        checkpoint_dir = _write_matching_checkpoint(app, tmp_path, step=250)

        app.on_button_pressed(
            Button.Pressed(app.query_one("#discover_checkpoints", Button))
        )
        assert len(app._discovered_checkpoints) == 1
        found = app._discovered_checkpoints[0]
        assert found.valid is True
        assert found.checkpoint_dir == checkpoint_dir
        assert app.query_one("#resume_best", Button).disabled is False
        panel_text = app.query_one("#checkpoints", Static).content
        assert "step 250" in panel_text
        assert "verified" in panel_text


@pytest.mark.asyncio
async def test_discover_checkpoints_with_none_found_disables_resume_best(tmp_path):
    (tmp_path / "train.jsonl").write_text('{"text":"hello"}\n', encoding="utf-8")
    app = ChowderTUI(project_path=str(tmp_path / "project.json"))
    async with app.run_test():
        _set(app, work_dir=str(tmp_path))
        app.on_button_pressed(
            Button.Pressed(app.query_one("#discover_checkpoints", Button))
        )
        assert app._discovered_checkpoints == ()
        assert app.query_one("#resume_best", Button).disabled is True
        assert "No interrupted runs" in app.query_one("#checkpoints", Static).content


@pytest.mark.asyncio
async def test_discover_checkpoints_reports_an_incompatible_checkpoint(tmp_path):
    (tmp_path / "train.jsonl").write_text('{"text":"hello"}\n', encoding="utf-8")
    app = ChowderTUI(project_path=str(tmp_path / "project.json"))
    async with app.run_test():
        _set(app, work_dir=str(tmp_path))
        _write_matching_checkpoint(app, tmp_path, step=100)
        # Change the config after the checkpoint was "produced" -- a real
        # mismatch, the same way a user editing the base model would be.
        _set(app, base_model="org/a-different-model")

        app.on_button_pressed(
            Button.Pressed(app.query_one("#discover_checkpoints", Button))
        )
        assert len(app._discovered_checkpoints) == 1
        found = app._discovered_checkpoints[0]
        assert found.valid is False
        assert "base_model" in found.mismatches
        assert app.query_one("#resume_best", Button).disabled is True
        panel_text = app.query_one("#checkpoints", Static).content
        assert "MISMATCH" in panel_text


@pytest.mark.asyncio
async def test_resume_best_fills_the_resume_field_with_the_valid_checkpoint(tmp_path):
    (tmp_path / "train.jsonl").write_text('{"text":"hello"}\n', encoding="utf-8")
    app = ChowderTUI(project_path=str(tmp_path / "project.json"))
    async with app.run_test():
        _set(app, work_dir=str(tmp_path))
        checkpoint_dir = _write_matching_checkpoint(app, tmp_path, step=250)
        app.on_button_pressed(
            Button.Pressed(app.query_one("#discover_checkpoints", Button))
        )
        app.on_button_pressed(Button.Pressed(app.query_one("#resume_best", Button)))
        assert app._value("resume_from_checkpoint") == str(checkpoint_dir)


@pytest.mark.asyncio
async def test_resume_best_is_a_noop_with_nothing_valid_discovered(tmp_path):
    app = ChowderTUI(project_path=str(tmp_path / "project.json"))
    async with app.run_test():
        _set(app, work_dir=str(tmp_path), resume_from_checkpoint="")
        app.on_button_pressed(Button.Pressed(app.query_one("#resume_best", Button)))
        assert app._value("resume_from_checkpoint") == ""


@pytest.mark.asyncio
async def test_start_fresh_clears_the_resume_field_without_touching_disk(tmp_path):
    (tmp_path / "train.jsonl").write_text('{"text":"hello"}\n', encoding="utf-8")
    app = ChowderTUI(project_path=str(tmp_path / "project.json"))
    async with app.run_test():
        _set(app, work_dir=str(tmp_path))
        checkpoint_dir = _write_matching_checkpoint(app, tmp_path, step=250)
        _set(app, resume_from_checkpoint=str(checkpoint_dir))

        app.on_button_pressed(Button.Pressed(app.query_one("#start_fresh", Button)))
        assert app._value("resume_from_checkpoint") == ""
        # Start Fresh never deletes evidence -- the checkpoint itself is
        # untouched on disk.
        assert checkpoint_dir.is_dir()


@pytest.mark.asyncio
async def test_discover_checkpoints_reports_a_build_payload_error_gracefully(tmp_path):
    app = ChowderTUI(project_path=str(tmp_path / "project.json"))
    async with app.run_test():
        _set(app, work_dir=str(tmp_path), metric_name="")  # required field left blank
        app.on_button_pressed(
            Button.Pressed(app.query_one("#discover_checkpoints", Button))
        )
        assert app._discovered_checkpoints == ()
        assert app.query_one("#resume_best", Button).disabled is True
        panel_text = app.query_one("#checkpoints", Static).content
        assert "failed" in panel_text.lower()


# --- memory estimate ---------------------------------------------------------


def _fake_estimate(**overrides) -> MemoryEstimate:
    fields = dict(
        device="cuda",
        frozen_params=1_000_000,
        trainable_params=1_000,
        max_length=512,
        measured_peak_gb_at_batch_1=2.0,
        measured_peak_gb_at_batch_2=3.5,
        per_example_activation_gb=1.5,
        configured_batch_size=1,
        estimated_peak_gb=2.0,
        per_rank_available_gb=16.0,
        fits=True,
        recommendations=(),
        from_cache=False,
    )
    fields.update(overrides)
    return MemoryEstimate(**fields)


@pytest.mark.asyncio
async def test_estimate_memory_reports_a_fitting_estimate(tmp_path, monkeypatch):
    (tmp_path / "train.jsonl").write_text('{"text":"hello"}\n', encoding="utf-8")
    monkeypatch.setattr(
        "chowder.tui.estimate_memory_requirements", lambda **kwargs: _fake_estimate()
    )
    app = ChowderTUI(project_path=str(tmp_path / "project.json"))
    async with app.run_test():
        _set(app, work_dir=str(tmp_path))
        app.on_button_pressed(Button.Pressed(app.query_one("#estimate_memory", Button)))
        assert app.query_one("#estimate_memory", Button).disabled is True
        await app._memory_estimate_worker.wait()
        assert app.query_one("#estimate_memory", Button).disabled is False
        panel_text = app.query_one("#memory_estimate", Static).content
        assert "cuda" in panel_text
        assert "Fits within available VRAM" in panel_text


@pytest.mark.asyncio
async def test_estimate_memory_reports_a_non_fitting_estimate_with_recommendations(
    tmp_path, monkeypatch
):
    (tmp_path / "train.jsonl").write_text('{"text":"hello"}\n', encoding="utf-8")
    monkeypatch.setattr(
        "chowder.tui.estimate_memory_requirements",
        lambda **kwargs: _fake_estimate(
            fits=False,
            recommendations=("switch backend.quantization to '4bit'", "overage: 4.00 GB"),
        ),
    )
    app = ChowderTUI(project_path=str(tmp_path / "project.json"))
    async with app.run_test():
        _set(app, work_dir=str(tmp_path))
        app.on_button_pressed(Button.Pressed(app.query_one("#estimate_memory", Button)))
        await app._memory_estimate_worker.wait()
        panel_text = app.query_one("#memory_estimate", Static).content
        assert "Does NOT fit" in panel_text
        assert "switch backend.quantization to '4bit'" in panel_text


@pytest.mark.asyncio
async def test_estimate_memory_reports_a_worker_failure_and_reenables_the_button(
    tmp_path, monkeypatch
):
    (tmp_path / "train.jsonl").write_text('{"text":"hello"}\n', encoding="utf-8")

    def _raise(**kwargs):
        raise RuntimeError("memory preflight worker failed with exit code 1")

    monkeypatch.setattr("chowder.tui.estimate_memory_requirements", _raise)
    app = ChowderTUI(project_path=str(tmp_path / "project.json"))
    async with app.run_test():
        _set(app, work_dir=str(tmp_path))
        app.on_button_pressed(Button.Pressed(app.query_one("#estimate_memory", Button)))
        await app._memory_estimate_worker.wait()
        assert app.query_one("#estimate_memory", Button).disabled is False
        panel_text = app.query_one("#memory_estimate", Static).content
        assert "Memory estimate failed" in panel_text
        assert "worker failed with exit code 1" in panel_text


@pytest.mark.asyncio
async def test_estimate_memory_reports_a_build_payload_error_without_starting_a_worker(
    tmp_path, monkeypatch
):
    """A config validation error (e.g. a required field left blank) must
    reject before spawning the expensive worker -- mirrors
    test_discover_checkpoints_reports_a_build_payload_error_gracefully."""
    called = False

    def _should_not_run(**kwargs):
        nonlocal called
        called = True
        return _fake_estimate()

    monkeypatch.setattr("chowder.tui.estimate_memory_requirements", _should_not_run)
    app = ChowderTUI(project_path=str(tmp_path / "project.json"))
    async with app.run_test():
        _set(app, work_dir=str(tmp_path), metric_name="")  # required field left blank
        app.on_button_pressed(Button.Pressed(app.query_one("#estimate_memory", Button)))
        assert app._memory_estimate_worker is None
        assert called is False
        # The button must never have been left disabled by a run that never started.
        assert app.query_one("#estimate_memory", Button).disabled is False


# --- run-status panel -------------------------------------------------------


@pytest.mark.asyncio
async def test_render_run_status_reports_not_running_before_any_run(tmp_path):
    app = ChowderTUI(project_path=str(tmp_path / "project.json"))
    async with app.run_test():
        assert app._render_run_status() == "Not running"


@pytest.mark.asyncio
async def test_reset_run_status_sets_project_model_and_starting_stage(tmp_path):
    app = ChowderTUI(project_path=str(tmp_path / "project.json"))
    async with app.run_test():
        app._reset_run_status(project="My Project", model="org/model")
        rendered = app._render_run_status()
    assert "Project: My Project" in rendered
    assert "Model: org/model" in rendered
    assert "Stage: starting" in rendered


def test_estimate_remaining_seconds_computes_a_linear_eta():
    status = {"step": 10, "max_steps": 100, "wall_seconds": 20.0}
    eta = ChowderTUI._estimate_remaining_seconds(status)
    assert eta == pytest.approx(180.0)  # 2s/step * 90 remaining steps


def test_estimate_remaining_seconds_none_at_or_past_max_steps():
    assert ChowderTUI._estimate_remaining_seconds({"step": 100, "max_steps": 100, "wall_seconds": 20.0}) is None


def test_estimate_remaining_seconds_none_without_max_steps():
    assert ChowderTUI._estimate_remaining_seconds({"step": 10, "wall_seconds": 20.0}) is None


def test_estimate_remaining_seconds_none_with_zero_step():
    assert (
        ChowderTUI._estimate_remaining_seconds({"step": 0, "max_steps": 100, "wall_seconds": 0.0})
        is None
    )


@pytest.mark.asyncio
async def test_render_run_status_includes_training_progress_and_eta(tmp_path):
    app = ChowderTUI(project_path=str(tmp_path / "project.json"))
    async with app.run_test():
        app._reset_run_status(project="p", model="m")
        app._run_status.update(
            step=5, max_steps=10, epoch=0.5, loss=1.2345, learning_rate=2e-4, wall_seconds=10.0
        )
        rendered = app._render_run_status()
    assert "step 5/10" in rendered
    assert "epoch 0.50" in rendered
    assert "loss 1.2345" in rendered
    assert "remaining" in rendered


@pytest.mark.asyncio
async def test_render_run_status_includes_checkpoint_repair_failure_and_promotion(tmp_path):
    app = ChowderTUI(project_path=str(tmp_path / "project.json"))
    async with app.run_test():
        app._reset_run_status(project="p", model="m")
        app._run_status.update(
            checkpoint_dir="/ckpt/e1",
            repair_depth=2,
            repair_failure_signature="abcdef0123456789",
            repair_stop_reason="max_depth",
            failure_count=3,
            repair_plan_count=1,
            promoted_experiment_id="e1-repair-1",
            promoted_metrics={"quality": 0.9},
        )
        rendered = app._render_run_status()
    assert "Checkpoint: /ckpt/e1" in rendered
    assert "Repair: depth 2" in rendered
    assert "cluster abcdef012345" in rendered
    assert "stopped: max_depth" in rendered
    assert "Failures harvested: 3 (1 repair plan(s))" in rendered
    assert "Promoted: e1-repair-1 (quality=0.9000)" in rendered


@pytest.mark.asyncio
async def test_update_run_status_is_a_noop_before_a_run_starts(tmp_path):
    """self._run_status starts empty (no _reset_run_status call yet) --
    events arriving before that must not crash or fabricate a run."""
    app = ChowderTUI(project_path=str(tmp_path / "project.json"))
    async with app.run_test():
        assert app._run_status == {}
        app._update_run_status(RunEvent(stage="train", message="x"))
        assert app._run_status == {}


def _fake_run_project_with_events(events_to_emit, outcome):
    def fake_run_project(project_path, *, on_event=None, cancellation=None):
        if on_event is not None:
            for event in events_to_emit:
                on_event(event)
        return outcome

    return fake_run_project


@pytest.mark.asyncio
async def test_run_status_panel_updates_live_through_the_real_event_pipeline(
    tmp_path, monkeypatch
):
    """Drives _reset_run_status -> a real @work(thread=True) worker calling
    event_sink from a genuine background thread -> _update_run_status's
    call_from_thread marshaling -> the actual #run_status widget. Exercises
    the real cross-thread path, not just the pure rendering logic covered
    above."""
    events = [
        RunEvent(stage="train", message="starting", experiment_id="e1"),
        TrainingProgressEvent(
            experiment_id="e1",
            step=5,
            max_steps=10,
            epoch=0.5,
            loss=1.5,
            learning_rate=2e-4,
            wall_seconds=3.0,
        ),
        CheckpointEvent(experiment_id="e1", checkpoint_dir="/ckpt/e1", step=5),
        RepairEvent(target_experiment_id="e1", depth=1, failure_signature="sig123"),
        FailureEvent(experiment_id="e1", failure_count=2, repair_plan_count=1),
        PromotionEvent(experiment_id="e1", metrics={"quality": 0.95}),
    ]
    monkeypatch.setattr(
        "chowder.tui.run_project",
        _fake_run_project_with_events(events, _fake_outcome(promoted_experiment_id="e1")),
    )

    app = ChowderTUI(project_path=str(tmp_path / "project.json"))
    async with app.run_test():
        _set(app, work_dir=str(tmp_path))
        app.on_button_pressed(Button.Pressed(app.query_one("#start", Button)))
        await app._training_worker.wait()
        panel_text = app.query_one("#run_status", Static).content

    assert "step 5/10" in panel_text
    assert "Checkpoint: /ckpt/e1" in panel_text
    assert "Repair: depth 1" in panel_text
    assert "Failures harvested: 2" in panel_text
    assert "Promoted: e1 (quality=0.9500)" in panel_text
