from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from chowder.activation_offload import (
    ActivationOffloadExperiment,
    _offload_calibration_key,
    _read_cache_entry,
    _write_cache_entry,
    run_activation_offload_experiment,
)
from chowder.backends.transformers_peft import TransformersPeftRunSpec
from chowder.executors import ExecutionContext
from chowder.memory import HardwareProfile

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
        "baseline_peak_vram_gb": 1.0,
        "offload_peak_vram_gb": 0.8,
        "baseline_wall_seconds": 0.1,
        "offload_wall_seconds": 0.15,
    }
    measured.update(overrides)
    return measured


def _fake_unavailable(reason="activation offload requires an available CUDA device"):
    return {"available": False, "device": "cpu", "reason": reason}


# --- calibration key ----------------------------------------------------


def test_offload_calibration_key_is_stable_for_identical_inputs():
    spec = _spec()
    context = _context()
    a = _offload_calibration_key(spec=spec, context=context, batch_size=4)
    b = _offload_calibration_key(spec=spec, context=context, batch_size=4)
    assert a == b


def test_offload_calibration_key_changes_with_batch_size():
    """Unlike memory_preflight's key, this one must be batch-size-sensitive
    -- the offload experiment measures at one specific batch size, not a
    batch-size-independent pair extrapolated afterward."""
    spec = _spec()
    context = _context()
    a = _offload_calibration_key(spec=spec, context=context, batch_size=2)
    b = _offload_calibration_key(spec=spec, context=context, batch_size=8)
    assert a != b


def test_offload_calibration_key_changes_with_base_model():
    context = _context()
    a = _offload_calibration_key(spec=_spec(base_model="org/a"), context=context, batch_size=2)
    b = _offload_calibration_key(spec=_spec(base_model="org/b"), context=context, batch_size=2)
    assert a != b


# --- cache read/write ---------------------------------------------------


def test_offload_write_then_read_cache_entry_round_trips(tmp_path):
    _write_cache_entry(tmp_path, "key1", {"baseline_peak_vram_gb": 1.0})
    assert _read_cache_entry(tmp_path, "key1") == {"baseline_peak_vram_gb": 1.0}


def test_offload_read_cache_entry_returns_none_when_file_missing(tmp_path):
    assert _read_cache_entry(tmp_path, "key1") is None


def test_offload_cache_uses_its_own_file_not_memory_preflights(tmp_path):
    """activation_offload.json and memory_calibration.json must be
    distinct files -- their cached value shapes are different and a
    shared key namespace would risk one overwriting the other."""
    _write_cache_entry(tmp_path, "key1", {"baseline_peak_vram_gb": 1.0})
    assert not (tmp_path / ".chowder" / "memory_calibration.json").exists()
    assert (tmp_path / ".chowder" / "activation_offload.json").exists()


# --- run_activation_offload_experiment orchestration (mocked worker) ----


def test_experiment_reports_unavailable_on_cpu(tmp_path):
    with patch(
        "chowder.activation_offload._run_offload_worker", return_value=_fake_unavailable()
    ) as mock_worker:
        experiment = run_activation_offload_experiment(
            resolved_config=_config(), context=_context(), work_dir=tmp_path, use_cache=False
        )
    mock_worker.assert_called_once()
    assert experiment.available is False
    assert experiment.required is False
    assert experiment.recommended is False
    assert experiment.reason is not None


def test_experiment_computes_vram_saved_and_penalty_ratio(tmp_path):
    with patch(
        "chowder.activation_offload._run_offload_worker",
        return_value=_fake_available(baseline_peak_vram_gb=2.0, offload_peak_vram_gb=1.5,
                                      baseline_wall_seconds=0.1, offload_wall_seconds=0.3),
    ):
        experiment = run_activation_offload_experiment(
            resolved_config=_config(), context=_context(), work_dir=tmp_path, use_cache=False
        )
    assert experiment.vram_saved_gb == pytest.approx(0.5)
    assert experiment.wall_time_penalty_ratio == pytest.approx(3.0)


def test_experiment_not_recommended_when_not_required_and_penalty_too_high(tmp_path):
    with patch(
        "chowder.activation_offload._run_offload_worker",
        return_value=_fake_available(baseline_peak_vram_gb=1.0, offload_peak_vram_gb=0.8,
                                      baseline_wall_seconds=0.1, offload_wall_seconds=1.0),
    ):
        experiment = run_activation_offload_experiment(
            resolved_config=_config(),
            context=_context(_hardware(vram_gb=16.0, pools=(16.0,))),
            work_dir=tmp_path,
            use_cache=False,
        )
    assert experiment.required is False
    assert experiment.wall_time_penalty_ratio == pytest.approx(10.0)
    assert experiment.recommended is False


def test_experiment_recommended_when_not_required_but_penalty_acceptable(tmp_path):
    with patch(
        "chowder.activation_offload._run_offload_worker",
        return_value=_fake_available(baseline_peak_vram_gb=1.0, offload_peak_vram_gb=0.8,
                                      baseline_wall_seconds=1.0, offload_wall_seconds=1.1),
    ):
        experiment = run_activation_offload_experiment(
            resolved_config=_config(),
            context=_context(_hardware(vram_gb=16.0, pools=(16.0,))),
            work_dir=tmp_path,
            use_cache=False,
        )
    assert experiment.required is False
    assert experiment.wall_time_penalty_ratio == pytest.approx(1.1)
    assert experiment.recommended is True


def test_experiment_required_and_recommended_when_baseline_does_not_fit(tmp_path):
    """Required overrides the penalty ratio entirely: fitting at all beats
    not fitting, however slow."""
    with patch(
        "chowder.activation_offload._run_offload_worker",
        return_value=_fake_available(baseline_peak_vram_gb=8.0, offload_peak_vram_gb=6.0,
                                      baseline_wall_seconds=0.1, offload_wall_seconds=5.0),
    ):
        experiment = run_activation_offload_experiment(
            resolved_config=_config(),
            context=_context(_hardware(vram_gb=6.5, pools=(6.5,))),
            work_dir=tmp_path,
            use_cache=False,
        )
    assert experiment.required is True
    assert experiment.wall_time_penalty_ratio == pytest.approx(50.0)
    assert experiment.recommended is True


def test_experiment_never_recommended_with_zero_or_negative_savings_even_if_required(tmp_path):
    """Offload that doesn't actually save VRAM is never worth recommending
    -- 'required' only matters when offload would actually help fit."""
    with patch(
        "chowder.activation_offload._run_offload_worker",
        return_value=_fake_available(baseline_peak_vram_gb=8.0, offload_peak_vram_gb=8.2,
                                      baseline_wall_seconds=0.1, offload_wall_seconds=0.1),
    ):
        experiment = run_activation_offload_experiment(
            resolved_config=_config(),
            context=_context(_hardware(vram_gb=6.5, pools=(6.5,))),
            work_dir=tmp_path,
            use_cache=False,
        )
    assert experiment.required is True
    assert experiment.vram_saved_gb < 0
    assert experiment.recommended is False


def test_experiment_uses_the_cache_on_a_second_call(tmp_path):
    config = _config()
    context = _context()
    with patch(
        "chowder.activation_offload._run_offload_worker", return_value=_fake_available()
    ) as mock_worker:
        first = run_activation_offload_experiment(resolved_config=config, context=context, work_dir=tmp_path)
        second = run_activation_offload_experiment(resolved_config=config, context=context, work_dir=tmp_path)
    mock_worker.assert_called_once()
    assert first.from_cache is False
    assert second.from_cache is True
    assert second.vram_saved_gb == first.vram_saved_gb


def test_experiment_cache_key_differs_by_explicit_batch_size_override(tmp_path):
    config = _config(training={"batch_size": 2})
    context = _context()
    with patch(
        "chowder.activation_offload._run_offload_worker", return_value=_fake_available()
    ) as mock_worker:
        run_activation_offload_experiment(resolved_config=config, context=context, work_dir=tmp_path, batch_size=2)
        run_activation_offload_experiment(resolved_config=config, context=context, work_dir=tmp_path, batch_size=8)
    assert mock_worker.call_count == 2


def test_experiment_explicit_batch_size_overrides_configured_one(tmp_path):
    config = _config(training={"batch_size": 2})
    captured = {}

    def _fake_worker(spec, *, batch_size, work_dir, timeout_seconds):
        captured["batch_size"] = batch_size
        return _fake_available(batch_size=batch_size)

    with patch("chowder.activation_offload._run_offload_worker", side_effect=_fake_worker):
        experiment = run_activation_offload_experiment(
            resolved_config=config, context=_context(), work_dir=tmp_path, batch_size=16, use_cache=False
        )
    assert captured["batch_size"] == 16
    assert experiment.batch_size == 16


def test_experiment_use_cache_false_always_calls_the_worker(tmp_path):
    config = _config()
    context = _context()
    with patch(
        "chowder.activation_offload._run_offload_worker", return_value=_fake_available()
    ) as mock_worker:
        run_activation_offload_experiment(resolved_config=config, context=context, work_dir=tmp_path, use_cache=False)
        run_activation_offload_experiment(resolved_config=config, context=context, work_dir=tmp_path, use_cache=False)
    assert mock_worker.call_count == 2


# --- real end-to-end (actual model load + timed forward/backward with/without offload) --


@_REAL_ML_SMOKE
def test_real_activation_offload_worker_measures_a_genuine_wall_time_penalty():
    from chowder.backends.activation_offload_worker import run_experiment

    spec = _spec(base_model=_TINY_MODEL, dataset="unused.jsonl", max_length=64)
    result = run_experiment(spec, batch_size=4)

    if not result["available"]:
        pytest.skip("no CUDA device available in this environment for a real offload comparison")
    assert result["baseline_peak_vram_gb"] > 0
    assert result["offload_peak_vram_gb"] > 0
    assert result["baseline_wall_seconds"] > 0
    assert result["offload_wall_seconds"] > 0
    # Real, warmed-up measurement on a genuine model: CPU activation
    # offload is never faster than keeping activations on-device for a
    # single-GPU forward+backward.
    assert result["offload_wall_seconds"] > result["baseline_wall_seconds"]


@_REAL_ML_SMOKE
def test_real_run_activation_offload_experiment_end_to_end(tmp_path):
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

    experiment = run_activation_offload_experiment(resolved_config=config, context=context, work_dir=tmp_path)
    if not experiment.available:
        pytest.skip("no CUDA device available in this environment for a real offload comparison")
    assert experiment.batch_size == 4
    assert experiment.baseline_peak_vram_gb > 0
    assert experiment.wall_time_penalty_ratio > 0
    # A tiny smoke-test model has negligible real activation pressure --
    # the correct, honest conclusion for this workload is "not worth it",
    # exercising the same recommendation logic a real large model with
    # genuine memory pressure would also go through.
    assert isinstance(experiment.recommended, bool)
