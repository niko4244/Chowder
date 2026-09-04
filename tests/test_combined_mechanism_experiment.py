from __future__ import annotations

import os

import pytest

from chowder.activation_offload import ActivationOffloadExperiment
from chowder.combined_mechanism_experiment import (
    _config_with_mechanisms,
    run_combined_mechanism_experiment,
)
from chowder.executors import ExecutionContext, TrainingArtifact
from chowder.frozen_layer_streaming import FrozenLayerStreamingExperiment
from chowder.memory import HardwareProfile
from chowder.optimizer_tiering import OptimizerTieringExperiment, OptimizerVariantMeasurement

_REAL_ML_SMOKE = pytest.mark.skipif(
    os.environ.get("CHOWDER_REAL_ML_SMOKE") != "1",
    reason="real ML smoke requires CHOWDER_REAL_ML_SMOKE=1 and train dependencies",
)
_TINY_MODEL = "trl-internal-testing/tiny-LlamaForCausalLM-3.2"


def _hardware() -> HardwareProfile:
    return HardwareProfile(16, 64, 500, 12, 40, 3)


def _base_resolved_config(tmp_path) -> dict:
    dataset = tmp_path / "train.jsonl"
    dataset.write_text('{"text": "hello world"}\n', encoding="utf-8")
    return {
        "backend": {
            "type": "transformers-peft",
            "base_model": "org/model",
            "dataset": str(dataset),
            "max_length": 64,
            "quantization": "none",
            "precision": "fp32",
            "lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]},
            "training": {
                "learning_rate": 0.001,
                "batch_size": 2,
                "activation_offload": "off",
                "optimizer_tiering": "off",
                "frozen_layer_streaming": "off",
            },
        },
    }


class FakeTrainer:
    """Records the resolved_config each call actually received and returns
    a controllable TrainingArtifact -- mirrors the FakeTrainer pattern
    already used by test_successive_halving.py, so fast tests never spawn
    a real training subprocess."""

    def __init__(self, telemetry_by_call):
        self._telemetry_by_call = list(telemetry_by_call)
        self.calls: list[dict] = []

    def run(self, experiment, context):
        training = context.resolved_config["backend"]["training"]
        self.calls.append(dict(training))
        telemetry = self._telemetry_by_call[len(self.calls) - 1]
        return TrainingArtifact(
            run_id=f"run-{experiment.experiment_id}",
            experiment_id=experiment.experiment_id,
            artifact_ref="unused",
            gpu_hours=0.01,
            telemetry=telemetry,
        )


def _fake_experiments(*, ao_savings=0.0, ot_state_bytes=0, fls_savings=0.0):
    activation_offload_exp = ActivationOffloadExperiment(
        device="cuda:0", available=True, batch_size=2, max_length=64,
        baseline_peak_vram_gb=1.0, offload_peak_vram_gb=1.0 - ao_savings,
        vram_saved_gb=ao_savings, baseline_wall_seconds=1.0, offload_wall_seconds=1.1,
        wall_time_penalty_ratio=1.1, per_rank_available_gb=16.0, required=False, recommended=ao_savings > 0,
    )
    optimizer_tiering_exp = OptimizerTieringExperiment(
        device="cuda:0", available=True, batch_size=2, max_length=64,
        variants=(
            OptimizerVariantMeasurement(name="adamw", step_seconds=1.0, state_bytes=ot_state_bytes),
            OptimizerVariantMeasurement(name="paged_adamw_8bit", step_seconds=1.1, state_bytes=1),
        ),
        model_peak_vram_gb=1.0, per_rank_available_gb=16.0,
        wall_time_penalty_ratio=1.1, required=False, recommended=ot_state_bytes > 0,
    )
    frozen_layer_streaming_exp = FrozenLayerStreamingExperiment(
        device="cuda:0", available=True, batch_size=2, max_length=64,
        baseline_peak_vram_gb=1.0, streamed_peak_vram_gb=1.0 - fls_savings,
        vram_saved_gb=fls_savings, baseline_wall_seconds=1.0, streamed_wall_seconds=1.1,
        wall_time_penalty_ratio=1.1, bytes_transferred_per_step=0,
        per_rank_available_gb=16.0, required=False, recommended=fls_savings > 0,
    )
    return activation_offload_exp, optimizer_tiering_exp, frozen_layer_streaming_exp


def _patch_mechanism_experiments(monkeypatch, *, ao_savings=0.0, ot_state_bytes=0, fls_savings=0.0):
    ao_exp, ot_exp, fls_exp = _fake_experiments(
        ao_savings=ao_savings, ot_state_bytes=ot_state_bytes, fls_savings=fls_savings
    )
    monkeypatch.setattr(
        "chowder.combined_mechanism_experiment.run_activation_offload_experiment",
        lambda **kwargs: ao_exp,
    )
    monkeypatch.setattr(
        "chowder.combined_mechanism_experiment.run_optimizer_tiering_experiment",
        lambda **kwargs: ot_exp,
    )
    monkeypatch.setattr(
        "chowder.combined_mechanism_experiment.run_frozen_layer_streaming_experiment",
        lambda **kwargs: fls_exp,
    )


def _patch_trainer(monkeypatch, telemetry_by_call):
    fake = FakeTrainer(telemetry_by_call)
    monkeypatch.setattr(
        "chowder.combined_mechanism_experiment.TransformersPeftExecutor", lambda: fake
    )
    return fake


def _context(tmp_path, resolved_config):
    return ExecutionContext(
        hardware=_hardware(), work_dir=str(tmp_path), seed=1, resolved_config=resolved_config
    )


def test_rejects_fewer_than_two_mechanisms(tmp_path):
    config = _base_resolved_config(tmp_path)
    with pytest.raises(ValueError, match="at least 2 mechanisms"):
        run_combined_mechanism_experiment(
            mechanisms=("activation_offload",),
            resolved_config=config, context=_context(tmp_path, config), work_dir=tmp_path,
        )


def test_rejects_unknown_mechanism(tmp_path):
    config = _base_resolved_config(tmp_path)
    with pytest.raises(ValueError, match="unknown mechanism"):
        run_combined_mechanism_experiment(
            mechanisms=("activation_offload", "not_real"),
            resolved_config=config, context=_context(tmp_path, config), work_dir=tmp_path,
        )


def test_rejects_duplicate_mechanisms(tmp_path):
    config = _base_resolved_config(tmp_path)
    with pytest.raises(ValueError, match="must not repeat"):
        run_combined_mechanism_experiment(
            mechanisms=("activation_offload", "activation_offload"),
            resolved_config=config, context=_context(tmp_path, config), work_dir=tmp_path,
        )


def test_config_with_mechanisms_enables_only_the_chosen_combination(tmp_path):
    config = _base_resolved_config(tmp_path)
    patched = _config_with_mechanisms(
        config, mechanisms=frozenset({"activation_offload"}), max_steps=4
    )
    training = patched["backend"]["training"]
    assert training["activation_offload"] == "always"
    assert training["optimizer_tiering"] == "off"
    assert training["frozen_layer_streaming"] == "off"
    assert training["detailed_timing_telemetry"] is True
    assert training["max_steps"] == 4
    assert training["save_strategy"] == "no"
    # the original config is untouched
    assert config["backend"]["training"]["activation_offload"] == "off"


def test_predicted_combined_peak_sums_only_the_requested_mechanisms(tmp_path, monkeypatch):
    config = _base_resolved_config(tmp_path)
    _patch_mechanism_experiments(monkeypatch, ao_savings=0.3, ot_state_bytes=0, fls_savings=0.2)
    _patch_trainer(
        monkeypatch,
        [
            {"peak_vram_gb": 1.0, "train_runtime_seconds": 1.0},  # baseline
            {"peak_vram_gb": 0.6, "train_runtime_seconds": 1.2},  # combined
        ],
    )

    result = run_combined_mechanism_experiment(
        mechanisms=("activation_offload", "frozen_layer_streaming"),
        resolved_config=config, context=_context(tmp_path, config), work_dir=tmp_path,
    )

    # predicted = baseline_peak(1.0) - (ao 0.3 + fls 0.2) = 0.5, NOT including
    # optimizer_tiering's savings even though that experiment also ran
    assert result.predicted_combined_peak_vram_gb == pytest.approx(0.5)
    assert result.actual_combined_peak_vram_gb == pytest.approx(0.6)
    assert result.prediction_error_gb == pytest.approx(0.5 - 0.6)
    assert result.per_mechanism_predicted_savings_gb == {
        "activation_offload": pytest.approx(0.3),
        "frozen_layer_streaming": pytest.approx(0.2),
    }


def test_mechanisms_are_stored_sorted_regardless_of_input_order(tmp_path, monkeypatch):
    config = _base_resolved_config(tmp_path)
    _patch_mechanism_experiments(monkeypatch)
    _patch_trainer(
        monkeypatch,
        [
            {"peak_vram_gb": 1.0, "train_runtime_seconds": 1.0},
            {"peak_vram_gb": 1.0, "train_runtime_seconds": 1.0},
        ],
    )
    result = run_combined_mechanism_experiment(
        mechanisms=("optimizer_tiering", "activation_offload"),
        resolved_config=config, context=_context(tmp_path, config), work_dir=tmp_path,
    )
    assert result.mechanisms == ("activation_offload", "optimizer_tiering")


def test_telemetry_fields_come_from_the_combined_run_not_the_baseline(tmp_path, monkeypatch):
    config = _base_resolved_config(tmp_path)
    _patch_mechanism_experiments(monkeypatch)
    _patch_trainer(
        monkeypatch,
        [
            {
                "peak_vram_gb": 1.0, "train_runtime_seconds": 1.0,
                "forward_seconds": 999.0,  # must NOT leak into the result
            },
            {
                "peak_vram_gb": 1.0, "train_runtime_seconds": 1.0,
                "forward_seconds": 0.4, "backward_seconds": 0.2, "optimizer_seconds": 0.05,
                "avg_gpu_utilization_percent": 12.5, "optimizer_state_bytes": 4096,
                "frozen_layer_streaming_bytes_transferred": 3072,
                "activation_offload_bytes_transferred": 12345,
            },
        ],
    )
    result = run_combined_mechanism_experiment(
        mechanisms=("activation_offload", "frozen_layer_streaming"),
        resolved_config=config, context=_context(tmp_path, config), work_dir=tmp_path,
    )
    assert result.forward_seconds == 0.4
    assert result.backward_seconds == 0.2
    assert result.optimizer_seconds == 0.05
    assert result.avg_gpu_utilization_percent == 12.5
    assert result.optimizer_state_bytes == 4096
    assert result.frozen_layer_streaming_bytes_transferred == 3072
    assert result.activation_offload_bytes_transferred == 12345


def test_baseline_run_has_every_mechanism_off(tmp_path, monkeypatch):
    config = _base_resolved_config(tmp_path)
    _patch_mechanism_experiments(monkeypatch)
    trainer = _patch_trainer(
        monkeypatch,
        [
            {"peak_vram_gb": 1.0, "train_runtime_seconds": 1.0},
            {"peak_vram_gb": 1.0, "train_runtime_seconds": 1.0},
        ],
    )
    run_combined_mechanism_experiment(
        mechanisms=("activation_offload", "optimizer_tiering"),
        resolved_config=config, context=_context(tmp_path, config), work_dir=tmp_path,
    )
    baseline_call, combined_call = trainer.calls
    assert baseline_call["activation_offload"] == "off"
    assert baseline_call["optimizer_tiering"] == "off"
    assert baseline_call["frozen_layer_streaming"] == "off"
    assert combined_call["activation_offload"] == "always"
    assert combined_call["optimizer_tiering"] == "always"
    assert combined_call["frozen_layer_streaming"] == "off"


@_REAL_ML_SMOKE
def test_real_combined_activation_offload_and_frozen_layer_streaming(tmp_path):
    """End-to-end with a REAL tiny model: one real baseline training run
    and one real training run with activation_offload + frozen_layer_streaming
    both set to "always" simultaneously, compared against what naively
    summing each mechanism's own independent real experiment predicts.

    This exercises exactly the finding this module's own development
    surfaced: a full Trainer.train() run's real peak VRAM can show far
    less benefit than summing each mechanism's own isolated forward+backward
    experiment would predict -- proving or correcting that assumption is
    the entire point of this module, so the real test only requires the
    real numbers to be internally consistent, not that the additive
    prediction turns out to be correct.
    """
    import json

    train_path = tmp_path / "train.jsonl"
    rows = [{"text": f"Question: what is {i}? Answer: {i * 2}"} for i in range(20)]
    train_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    resolved_config = {
        "backend": {
            "type": "transformers-peft",
            "base_model": _TINY_MODEL,
            "dataset": str(train_path),
            "max_length": 256,
            "quantization": "none",
            "precision": "fp32",
            "lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]},
            "training": {
                "learning_rate": 0.001,
                "batch_size": 8,
                "gradient_accumulation_steps": 1,
                "logging_steps": 1,
                "gradient_checkpointing": False,
            },
            "runtime": {"timeout_seconds": 180.0},
        },
    }
    context = ExecutionContext(
        hardware=_hardware(), work_dir=str(tmp_path), seed=1, resolved_config=resolved_config
    )

    result = run_combined_mechanism_experiment(
        mechanisms=("activation_offload", "frozen_layer_streaming"),
        resolved_config=resolved_config,
        context=context,
        work_dir=tmp_path,
    )

    assert result.mechanisms == ("activation_offload", "frozen_layer_streaming")
    assert result.baseline_peak_vram_gb > 0
    assert result.actual_combined_peak_vram_gb > 0
    assert result.baseline_wall_seconds > 0
    assert result.combined_wall_seconds > 0
    assert result.forward_seconds is not None
    assert result.backward_seconds is not None
    assert result.optimizer_seconds is not None
    # prediction_error_gb is real signed data either way -- both directions
    # are legitimate findings, so this only asserts it was computed, not a sign
    assert isinstance(result.prediction_error_gb, float)
