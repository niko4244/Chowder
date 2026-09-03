from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from chowder.backends.transformers_peft import TransformersPeftRunSpec
from chowder.executors import ExecutionContext
from chowder.memory import HardwareProfile
from chowder.optimizer_tiering import (
    OptimizerTieringExperiment,
    OptimizerVariantMeasurement,
    _read_cache_entry,
    _tiering_calibration_key,
    _write_cache_entry,
    run_optimizer_tiering_experiment,
)

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


def _spec(**overrides):
    fields = dict(
        base_model="org/model",
        dataset="train.jsonl",
        output_dir="/tmp/out",
        max_length=64,
        quantization="none",
        precision="fp32",
        lora_r=4,
        lora_alpha=8,
        target_modules=("q_proj", "v_proj"),
        batch_size=1,
        gradient_checkpointing=False,
    )
    fields.update(overrides)
    return TransformersPeftRunSpec(**fields)


def _fake_available(**overrides):
    measured = {
        "available": True,
        "device": "cuda",
        "batch_size": 2,
        "max_length": 64,
        "variants": {
            "adamw": {"step_seconds": 0.001, "state_bytes": 1824},
            "paged_adamw": {"step_seconds": 0.0012, "state_bytes": 1792},
            "paged_adamw_8bit": {"step_seconds": 0.0015, "state_bytes": 1792},
        },
    }
    measured.update(overrides)
    return measured


def _fake_unavailable(reason="bitsandbytes is not installed; install chowder-ai[qlora]"):
    return {"available": False, "device": "cpu", "reason": reason}


def _fake_memory_estimate(peak_gb: float):
    from chowder.memory_preflight import MemoryEstimate

    return MemoryEstimate(
        device="cuda", frozen_params=1000, trainable_params=10, max_length=64,
        measured_peak_gb_at_batch_1=peak_gb, measured_peak_gb_at_batch_2=peak_gb,
        per_example_activation_gb=0.0, configured_batch_size=2, estimated_peak_gb=peak_gb,
        per_rank_available_gb=16.0, fits=True,
    )


# --- calibration key ------------------------------------------------------


def test_tiering_calibration_key_is_stable_for_identical_inputs():
    spec = _spec()
    context = _context()
    a = _tiering_calibration_key(spec=spec, context=context, batch_size=4)
    b = _tiering_calibration_key(spec=spec, context=context, batch_size=4)
    assert a == b


def test_tiering_calibration_key_changes_with_batch_size():
    spec = _spec()
    context = _context()
    a = _tiering_calibration_key(spec=spec, context=context, batch_size=2)
    b = _tiering_calibration_key(spec=spec, context=context, batch_size=8)
    assert a != b


# --- cache read/write -------------------------------------------------------


def test_tiering_write_then_read_cache_entry_round_trips(tmp_path):
    _write_cache_entry(tmp_path, "key1", {"available": True})
    assert _read_cache_entry(tmp_path, "key1") == {"available": True}


def test_tiering_cache_uses_its_own_file(tmp_path):
    _write_cache_entry(tmp_path, "key1", {"available": True})
    assert (tmp_path / ".chowder" / "optimizer_tiering.json").exists()
    assert not (tmp_path / ".chowder" / "activation_offload.json").exists()
    assert not (tmp_path / ".chowder" / "memory_calibration.json").exists()


# --- run_optimizer_tiering_experiment orchestration (mocked workers) -------


def test_experiment_reports_unavailable_without_bitsandbytes(tmp_path):
    with patch(
        "chowder.optimizer_tiering._run_tiering_worker", return_value=_fake_unavailable()
    ) as mock_worker:
        experiment = run_optimizer_tiering_experiment(
            resolved_config=_config(), context=_context(), work_dir=tmp_path, use_cache=False
        )
    mock_worker.assert_called_once()
    assert experiment.available is False
    assert experiment.required is False
    assert experiment.recommended is False
    assert experiment.variants == ()
    assert experiment.reason is not None


def test_experiment_reports_all_three_variants(tmp_path):
    with (
        patch("chowder.optimizer_tiering._run_tiering_worker", return_value=_fake_available()),
        patch("chowder.optimizer_tiering.estimate_memory_requirements", return_value=_fake_memory_estimate(1.0)),
    ):
        experiment = run_optimizer_tiering_experiment(
            resolved_config=_config(), context=_context(), work_dir=tmp_path, use_cache=False
        )
    names = {v.name for v in experiment.variants}
    assert names == {"adamw", "paged_adamw", "paged_adamw_8bit"}
    assert experiment.variant("paged_adamw_8bit").state_bytes == 1792
    assert experiment.variant("nonexistent") is None


def test_experiment_not_recommended_when_not_required_and_penalty_too_high(tmp_path):
    measured = _fake_available(
        variants={
            "adamw": {"step_seconds": 0.1, "state_bytes": 1824},
            "paged_adamw": {"step_seconds": 1.0, "state_bytes": 1792},
            "paged_adamw_8bit": {"step_seconds": 1.2, "state_bytes": 1792},
        }
    )
    with (
        patch("chowder.optimizer_tiering._run_tiering_worker", return_value=measured),
        patch("chowder.optimizer_tiering.estimate_memory_requirements", return_value=_fake_memory_estimate(1.0)),
    ):
        experiment = run_optimizer_tiering_experiment(
            resolved_config=_config(),
            context=_context(_hardware(vram_gb=16.0, pools=(16.0,))),
            work_dir=tmp_path,
            use_cache=False,
        )
    assert experiment.required is False
    assert experiment.wall_time_penalty_ratio == pytest.approx(10.0)
    assert experiment.recommended is False


def test_experiment_recommended_when_not_required_but_penalty_acceptable(tmp_path):
    measured = _fake_available(
        variants={
            "adamw": {"step_seconds": 1.0, "state_bytes": 1824},
            "paged_adamw": {"step_seconds": 1.1, "state_bytes": 1792},
            "paged_adamw_8bit": {"step_seconds": 1.2, "state_bytes": 1792},
        }
    )
    with (
        patch("chowder.optimizer_tiering._run_tiering_worker", return_value=measured),
        patch("chowder.optimizer_tiering.estimate_memory_requirements", return_value=_fake_memory_estimate(1.0)),
    ):
        experiment = run_optimizer_tiering_experiment(
            resolved_config=_config(),
            context=_context(_hardware(vram_gb=16.0, pools=(16.0,))),
            work_dir=tmp_path,
            use_cache=False,
        )
    assert experiment.required is False
    assert experiment.wall_time_penalty_ratio == pytest.approx(1.1)
    assert experiment.recommended is True


def test_experiment_required_and_recommended_when_combined_footprint_does_not_fit(tmp_path):
    """Required overrides the penalty ratio entirely, just like activation
    offload's policy: fitting at all beats not fitting, however slow."""
    measured = _fake_available(
        variants={
            "adamw": {"step_seconds": 0.1, "state_bytes": 1824},
            "paged_adamw": {"step_seconds": 5.0, "state_bytes": 1792},
            "paged_adamw_8bit": {"step_seconds": 5.5, "state_bytes": 1792},
        }
    )
    with (
        patch("chowder.optimizer_tiering._run_tiering_worker", return_value=measured),
        patch("chowder.optimizer_tiering.estimate_memory_requirements", return_value=_fake_memory_estimate(8.0)),
    ):
        experiment = run_optimizer_tiering_experiment(
            resolved_config=_config(),
            context=_context(_hardware(vram_gb=8.5, pools=(8.5,))),
            work_dir=tmp_path,
            use_cache=False,
        )
    assert experiment.required is True
    assert experiment.wall_time_penalty_ratio == pytest.approx(50.0)
    assert experiment.recommended is True


def test_experiment_uses_the_cache_on_a_second_call(tmp_path):
    config = _config()
    context = _context()
    with (
        patch("chowder.optimizer_tiering._run_tiering_worker", return_value=_fake_available()) as mock_worker,
        patch("chowder.optimizer_tiering.estimate_memory_requirements", return_value=_fake_memory_estimate(1.0)),
    ):
        first = run_optimizer_tiering_experiment(resolved_config=config, context=context, work_dir=tmp_path)
        second = run_optimizer_tiering_experiment(resolved_config=config, context=context, work_dir=tmp_path)
    mock_worker.assert_called_once()
    assert first.from_cache is False
    assert second.from_cache is True
    assert second.wall_time_penalty_ratio == first.wall_time_penalty_ratio


def test_experiment_use_cache_false_always_calls_the_worker(tmp_path):
    config = _config()
    context = _context()
    with (
        patch("chowder.optimizer_tiering._run_tiering_worker", return_value=_fake_available()) as mock_worker,
        patch("chowder.optimizer_tiering.estimate_memory_requirements", return_value=_fake_memory_estimate(1.0)),
    ):
        run_optimizer_tiering_experiment(resolved_config=config, context=context, work_dir=tmp_path, use_cache=False)
        run_optimizer_tiering_experiment(resolved_config=config, context=context, work_dir=tmp_path, use_cache=False)
    assert mock_worker.call_count == 2


def test_experiment_explicit_batch_size_overrides_configured_one(tmp_path):
    config = _config(training={"batch_size": 2})
    captured = {}

    def _fake_worker(spec, *, batch_size, work_dir, timeout_seconds):
        captured["batch_size"] = batch_size
        return _fake_available(batch_size=batch_size)

    with (
        patch("chowder.optimizer_tiering._run_tiering_worker", side_effect=_fake_worker),
        patch("chowder.optimizer_tiering.estimate_memory_requirements", return_value=_fake_memory_estimate(1.0)),
    ):
        experiment = run_optimizer_tiering_experiment(
            resolved_config=config, context=_context(), work_dir=tmp_path, batch_size=16, use_cache=False
        )
    assert captured["batch_size"] == 16
    assert experiment.batch_size == 16


# --- real end-to-end (actual model load + timed optimizer.step comparison) --


@_REAL_ML_SMOKE
def test_real_optimizer_tiering_worker_measures_genuine_variants():
    from chowder.backends.optimizer_tiering_worker import run_experiment

    spec = _spec(base_model=_TINY_MODEL, dataset="unused.jsonl", max_length=64)
    result = run_experiment(spec, batch_size=4)

    if not result["available"]:
        pytest.skip(f"optimizer tiering unavailable in this environment: {result.get('reason')}")
    variants = result["variants"]
    assert set(variants) == {"adamw", "paged_adamw", "paged_adamw_8bit"}
    for entry in variants.values():
        assert entry["step_seconds"] > 0
        assert entry["state_bytes"] > 0


@_REAL_ML_SMOKE
def test_real_run_optimizer_tiering_experiment_end_to_end(tmp_path):
    from chowder.hardware import detect_hardware
    from chowder.project_runner import hardware_profile_from_snapshot

    hardware = hardware_profile_from_snapshot(detect_hardware(str(tmp_path)))
    context = ExecutionContext(hardware, str(tmp_path), seed=7)
    config = {
        "backend": {
            "type": "transformers-peft",
            "base_model": _TINY_MODEL,
            "dataset": "unused.jsonl",
            "max_length": 64,
            "quantization": "none",
            "precision": "fp32",
            "lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]},
            "training": {"batch_size": 4},
        }
    }

    experiment = run_optimizer_tiering_experiment(resolved_config=config, context=context, work_dir=tmp_path)
    if not experiment.available:
        pytest.skip(f"optimizer tiering unavailable in this environment: {experiment.reason}")
    assert experiment.batch_size == 4
    assert experiment.model_peak_vram_gb > 0
    assert experiment.wall_time_penalty_ratio > 0
    assert isinstance(experiment.recommended, bool)
    assert experiment.variant("adamw") is not None
    assert experiment.variant("paged_adamw") is not None
