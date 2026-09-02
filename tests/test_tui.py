from __future__ import annotations

import pytest

from chowder.hardware import AcceleratorProfile, HardwareSnapshot
from chowder.project import ProjectValidationError
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


async def _app(tmp_path, **input_overrides):
    app = ChowderTUI(project_path=str(tmp_path / "project.json"))
    async with app.run_test() as pilot:
        for widget_id, value in input_overrides.items():
            widget = app.query_one(f"#{widget_id}")
            widget.value = value
        yield app, pilot


@pytest.mark.asyncio
async def test_quantization_auto_omits_the_key(tmp_path):
    async for app, _ in _app(tmp_path):
        payload = app._build_payload()
        assert "quantization" not in payload["config"]["backend"]


@pytest.mark.asyncio
async def test_quantization_explicit_value_is_included(tmp_path):
    async for app, _ in _app(tmp_path, quantization="4bit"):
        payload = app._build_payload()
        assert payload["config"]["backend"]["quantization"] == "4bit"


@pytest.mark.asyncio
async def test_gradient_checkpointing_auto_omits_the_key(tmp_path):
    async for app, _ in _app(tmp_path):
        payload = app._build_payload()
        assert "gradient_checkpointing" not in payload["config"]["backend"]["training"]


@pytest.mark.asyncio
async def test_gradient_checkpointing_explicit_true_is_included(tmp_path):
    async for app, _ in _app(tmp_path, gradient_checkpointing="true"):
        payload = app._build_payload()
        assert payload["config"]["backend"]["training"]["gradient_checkpointing"] is True


@pytest.mark.asyncio
async def test_gradient_checkpointing_explicit_false_is_included(tmp_path):
    async for app, _ in _app(tmp_path, gradient_checkpointing="FALSE"):
        payload = app._build_payload()
        assert payload["config"]["backend"]["training"]["gradient_checkpointing"] is False


@pytest.mark.asyncio
async def test_gradient_checkpointing_invalid_value_raises(tmp_path):
    async for app, _ in _app(tmp_path, gradient_checkpointing="sometimes"):
        with pytest.raises(ProjectValidationError, match="gradient checkpointing"):
            app._build_payload()


@pytest.mark.asyncio
async def test_active_accelerator_count_auto_uses_detected_gpu_count(tmp_path):
    async for app, _ in _app(tmp_path):
        app._hardware = _snapshot(2)
        payload = app._build_payload()
        assert payload["config"]["backend"]["runtime"]["active_accelerator_count"] == 2


@pytest.mark.asyncio
async def test_active_accelerator_count_auto_with_no_hardware_scanned_yet_defaults_to_zero(
    tmp_path,
):
    async for app, _ in _app(tmp_path):
        assert app._hardware is None
        payload = app._build_payload()
        assert payload["config"]["backend"]["runtime"]["active_accelerator_count"] == 0


@pytest.mark.asyncio
async def test_active_accelerator_count_explicit_value_overrides_detected_count(tmp_path):
    async for app, _ in _app(tmp_path, active_accelerator_count="1"):
        app._hardware = _snapshot(2)
        payload = app._build_payload()
        assert payload["config"]["backend"]["runtime"]["active_accelerator_count"] == 1


@pytest.mark.asyncio
async def test_active_accelerator_count_negative_raises(tmp_path):
    async for app, _ in _app(tmp_path, active_accelerator_count="-1"):
        with pytest.raises(ProjectValidationError, match="negative"):
            app._build_payload()


@pytest.mark.asyncio
async def test_active_accelerator_count_non_integer_raises(tmp_path):
    async for app, _ in _app(tmp_path, active_accelerator_count="two"):
        with pytest.raises(ProjectValidationError, match="'auto' or an integer"):
            app._build_payload()


@pytest.mark.asyncio
async def test_precision_auto_is_passed_through_as_an_explicit_value(tmp_path):
    """Unlike quantization/gradient_checkpointing, precision's "auto" is a
    real value the worker itself resolves -- it must never be omitted."""
    async for app, _ in _app(tmp_path):
        payload = app._build_payload()
        assert payload["config"]["backend"]["precision"] == "auto"
