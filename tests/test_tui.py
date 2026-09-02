from __future__ import annotations

import pytest
from textual.widgets import Button

from chowder.cancellation import CancellationToken
from chowder.hardware import AcceleratorProfile, HardwareSnapshot
from chowder.models import Experiment, ExperimentResult, Hypothesis
from chowder.project import ProjectValidationError
from chowder.registry import RunRegistry
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
