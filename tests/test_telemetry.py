from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from chowder.executors import ExecutionContext
from chowder.memory import HardwareProfile
from chowder.telemetry import LayerTelemetry, RuntimeTelemetry, collect_runtime_telemetry

_REAL_ML_SMOKE = pytest.mark.skipif(
    os.environ.get("CHOWDER_REAL_ML_SMOKE") != "1",
    reason="real ML smoke requires CHOWDER_REAL_ML_SMOKE=1 and train dependencies",
)
_TINY_MODEL = "trl-internal-testing/tiny-LlamaForCausalLM-3.2"


def _config(**backend_overrides):
    backend = {
        "type": "transformers-peft",
        "base_model": "org/model",
        "dataset": "train.jsonl",
        "max_length": 64,
        "quantization": "none",
        "precision": "fp32",
        "lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]},
    }
    backend.update(backend_overrides)
    return {"backend": backend}


def _hardware(vram_gb=16.0, pools=(16.0,)):
    return HardwareProfile(vram_gb, 64.0, 500.0, 12.0, 40.0, 3.0, accelerator_vram_gb=pools)


def _context(hardware=None):
    return ExecutionContext(hardware or _hardware(), ".", seed=1)


def _fake_measured(**overrides):
    measured = {
        "device": "cuda",
        "frozen_params": 1000,
        "trainable_params": 10,
        "max_length": 64,
        "peak_vram_gb_bs1": 1.0,
        "peak_vram_gb_bs2": 1.5,
        "layer_telemetry": [
            {
                "name": "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default",
                "module_type": "Linear",
                "trainable_params": 4,
                "frozen_params": 0,
                "activation_bytes": 256,
            },
            {
                "name": "base_model.model.lm_head",
                "module_type": "Linear",
                "trainable_params": 0,
                "frozen_params": 900,
                "activation_bytes": 32768,
            },
        ],
        "optimizer_state_bytes": 8192,
    }
    measured.update(overrides)
    return measured


# --- data model ---------------------------------------------------------


def test_total_activation_bytes_sums_every_layer():
    telemetry = RuntimeTelemetry(
        device="cuda",
        max_length=64,
        frozen_params=900,
        trainable_params=4,
        optimizer_state_bytes=8192,
        layers=(
            LayerTelemetry("a", "Linear", 1, 0, 100),
            LayerTelemetry("b", "Linear", 0, 1, 250),
        ),
    )
    assert telemetry.total_activation_bytes == 350


def test_total_activation_bytes_is_zero_with_no_layers():
    telemetry = RuntimeTelemetry(
        device="cpu", max_length=64, frozen_params=0, trainable_params=0,
        optimizer_state_bytes=0, layers=(),
    )
    assert telemetry.total_activation_bytes == 0


def test_top_activation_layers_returns_the_largest_first():
    telemetry = RuntimeTelemetry(
        device="cuda", max_length=64, frozen_params=0, trainable_params=0,
        optimizer_state_bytes=0,
        layers=(
            LayerTelemetry("small", "Linear", 0, 0, 10),
            LayerTelemetry("big", "Linear", 0, 0, 1000),
            LayerTelemetry("medium", "Linear", 0, 0, 100),
        ),
    )
    top = telemetry.top_activation_layers(2)
    assert [layer.name for layer in top] == ["big", "medium"]


def test_top_activation_layers_never_returns_more_than_requested():
    telemetry = RuntimeTelemetry(
        device="cuda", max_length=64, frozen_params=0, trainable_params=0,
        optimizer_state_bytes=0,
        layers=(LayerTelemetry("only", "Linear", 0, 0, 10),),
    )
    assert len(telemetry.top_activation_layers(5)) == 1


# --- collect_runtime_telemetry orchestration (mocked worker) -----------


def test_collect_runtime_telemetry_parses_the_worker_result(tmp_path):
    with patch(
        "chowder.telemetry._run_dry_run_worker", return_value=_fake_measured()
    ) as mock_worker:
        telemetry = collect_runtime_telemetry(
            resolved_config=_config(), context=_context(), work_dir=tmp_path, use_cache=False
        )
    mock_worker.assert_called_once()
    assert telemetry.device == "cuda"
    assert telemetry.optimizer_state_bytes == 8192
    assert len(telemetry.layers) == 2
    assert telemetry.layers[0].name.endswith("lora_A.default")
    assert telemetry.layers[0].trainable_params == 4
    assert telemetry.layers[1].activation_bytes == 32768
    assert telemetry.from_cache is False


def test_collect_runtime_telemetry_uses_the_cache_on_a_second_call(tmp_path):
    config = _config()
    context = _context()
    with patch(
        "chowder.telemetry._run_dry_run_worker", return_value=_fake_measured()
    ) as mock_worker:
        first = collect_runtime_telemetry(resolved_config=config, context=context, work_dir=tmp_path)
        second = collect_runtime_telemetry(resolved_config=config, context=context, work_dir=tmp_path)
    mock_worker.assert_called_once()
    assert first.from_cache is False
    assert second.from_cache is True
    assert second.total_activation_bytes == first.total_activation_bytes


def test_collect_runtime_telemetry_shares_the_cache_with_memory_estimate(tmp_path):
    """One real measurement pass produces both the VRAM-fit fields
    memory_preflight.estimate_memory_requirements() wants and the
    telemetry fields this module wants -- calling one first must warm the
    cache for the other, not trigger a second real worker spawn."""
    from chowder.memory_preflight import estimate_memory_requirements

    config = _config()
    context = _context()
    with patch(
        "chowder.memory_preflight._run_dry_run_worker", return_value=_fake_measured()
    ) as mock_worker:
        estimate = estimate_memory_requirements(resolved_config=config, context=context, work_dir=tmp_path)
        telemetry = collect_runtime_telemetry(resolved_config=config, context=context, work_dir=tmp_path)
    mock_worker.assert_called_once()
    assert estimate.from_cache is False
    assert telemetry.from_cache is True
    assert telemetry.optimizer_state_bytes == 8192


def test_collect_runtime_telemetry_refuses_a_stale_cache_entry_without_telemetry(tmp_path):
    """A memory_calibration.json entry written before this telemetry field
    existed (an older Chowder version) must not be trusted as if it had
    real telemetry -- it must trigger a fresh real measurement instead of
    silently reporting zero layers."""
    from chowder.memory_preflight import _calibration_key, _write_cache_entry
    from chowder.backends.transformers_peft import TransformersPeftRunSpec

    config = _config()
    context = _context()
    spec = TransformersPeftRunSpec.from_resolved_config(
        config, work_dir=tmp_path, output_dir=tmp_path / "scratch", seed=context.seed,
        hardware=context.hardware,
    )
    key = _calibration_key(spec=spec, context=context)
    # A pre-Phase-7 cache entry: no layer_telemetry/optimizer_state_bytes.
    _write_cache_entry(
        tmp_path, key,
        {"device": "cuda", "frozen_params": 1000, "trainable_params": 10,
         "max_length": 64, "peak_vram_gb_bs1": 1.0, "peak_vram_gb_bs2": 1.5},
    )
    with patch(
        "chowder.telemetry._run_dry_run_worker", return_value=_fake_measured()
    ) as mock_worker:
        telemetry = collect_runtime_telemetry(resolved_config=config, context=context, work_dir=tmp_path)
    mock_worker.assert_called_once()
    assert telemetry.from_cache is False
    assert len(telemetry.layers) == 2


def test_collect_runtime_telemetry_use_cache_false_always_calls_the_worker(tmp_path):
    config = _config()
    context = _context()
    with patch(
        "chowder.telemetry._run_dry_run_worker", return_value=_fake_measured()
    ) as mock_worker:
        collect_runtime_telemetry(resolved_config=config, context=context, work_dir=tmp_path, use_cache=False)
        collect_runtime_telemetry(resolved_config=config, context=context, work_dir=tmp_path, use_cache=False)
    assert mock_worker.call_count == 2


def test_collect_runtime_telemetry_handles_no_layers_gracefully(tmp_path):
    with patch(
        "chowder.telemetry._run_dry_run_worker",
        return_value=_fake_measured(layer_telemetry=[]),
    ):
        telemetry = collect_runtime_telemetry(
            resolved_config=_config(), context=_context(), work_dir=tmp_path, use_cache=False
        )
    assert telemetry.layers == ()
    assert telemetry.total_activation_bytes == 0


def test_collect_runtime_telemetry_defaults_optimizer_state_bytes_when_absent(tmp_path):
    measured = _fake_measured()
    del measured["optimizer_state_bytes"]
    with patch("chowder.telemetry._run_dry_run_worker", return_value=measured):
        telemetry = collect_runtime_telemetry(
            resolved_config=_config(), context=_context(), work_dir=tmp_path, use_cache=False
        )
    assert telemetry.optimizer_state_bytes == 0


# --- real end-to-end (actual model load + forward/backward + optimizer step) --


@_REAL_ML_SMOKE
def test_real_dry_run_collects_leaf_module_telemetry_and_a_real_optimizer_step():
    from chowder.backends.memory_preflight_worker import dry_run
    from chowder.backends.transformers_peft import TransformersPeftRunSpec

    spec = TransformersPeftRunSpec(
        base_model=_TINY_MODEL,
        dataset="unused.jsonl",
        output_dir="/tmp/telemetry-real-check",
        max_length=32,
        quantization="none",
        precision="fp32",
        lora_r=4,
        lora_alpha=8,
        target_modules=("q_proj", "v_proj"),
        batch_size=1,
        gradient_checkpointing=False,
    )
    result = dry_run(spec)

    layers = result["layer_telemetry"]
    assert len(layers) > 0
    # LoRA A/B adapters must show up as real trainable leaf modules.
    lora_layers = [entry for entry in layers if "lora_A" in entry["name"] or "lora_B" in entry["name"]]
    assert lora_layers
    assert all(entry["trainable_params"] > 0 for entry in lora_layers)
    # Every entry has a real, non-negative measured activation size.
    assert all(entry["activation_bytes"] >= 0 for entry in layers)
    assert any(entry["activation_bytes"] > 0 for entry in layers)
    # A real optimizer step was taken; AdamW's exp_avg/exp_avg_sq state
    # allocation is never free.
    assert result["optimizer_state_bytes"] > 0


@_REAL_ML_SMOKE
def test_real_collect_runtime_telemetry_end_to_end(tmp_path):
    from chowder.hardware import detect_hardware
    from chowder.project_runner import hardware_profile_from_snapshot

    hardware = hardware_profile_from_snapshot(detect_hardware(str(tmp_path)))
    context = ExecutionContext(hardware, str(tmp_path), seed=7)
    config = {
        "backend": {
            "type": "transformers-peft",
            "base_model": _TINY_MODEL,
            "dataset": "unused.jsonl",
            "max_length": 32,
            "quantization": "none",
            "precision": "fp32",
            "lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]},
            "training": {"batch_size": 2},
        }
    }

    telemetry = collect_runtime_telemetry(resolved_config=config, context=context, work_dir=tmp_path)
    assert telemetry.frozen_params > 0
    assert telemetry.trainable_params > 0
    assert len(telemetry.layers) > 0
    assert telemetry.total_activation_bytes > 0
    assert telemetry.optimizer_state_bytes > 0
    top = telemetry.top_activation_layers(3)
    assert len(top) == 3
    assert top[0].activation_bytes >= top[1].activation_bytes >= top[2].activation_bytes
