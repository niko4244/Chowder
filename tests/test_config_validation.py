import pytest

from chowder.config_validation import (
    ConfigValidationError,
    TRANSFORMERS_BACKEND_SCHEMA_VERSION,
    validate_transformers_backend_config,
)
from chowder.cycle import ExperimentCycleRunner
from chowder.engine import EvolutionEngine
from chowder.executors import CostEstimate, ExecutionContext
from chowder.memory import HardwareProfile
from chowder.models import Experiment, ExperimentResult, Goal, Hypothesis, MetricTarget


def _backend_config():
    return {
        "seed": 7,
        "backend": {
            "schema_version": TRANSFORMERS_BACKEND_SCHEMA_VERSION,
            "type": "transformers-peft",
            "base_model": "example/model",
            "dataset": "train.jsonl",
            "dataset_sha256": "d" * 64,
            "text_field": "text",
            "max_length": 512,
            "quantization": "4bit",
            "precision": "fp16",
            "training": {
                "epochs": 1.0,
                "learning_rate": 2e-4,
                "batch_size": 1,
                "gradient_accumulation_steps": 4,
                "logging_steps": 10,
                "gradient_checkpointing": True,
            },
            "lora": {
                "r": 16,
                "alpha": 32,
                "dropout": 0.05,
                "target_modules": ["q_proj", "v_proj"],
                "use_rslora": False,
            },
            "runtime": {
                "timeout_seconds": 3600,
                "active_accelerator_count": 2,
            },
            "profile": {
                "estimated_steps": 100,
                "seconds_per_step": 2.0,
                "peak_vram_gb": 15.0,
                "source": "measured",
                "active_accelerator_count": 2,
            },
        },
        "evaluation": {"estimated_gpu_hours": 0.05},
    }


def test_valid_t4x2_backend_config_is_accepted():
    validate_transformers_backend_config(_backend_config())


def test_top_level_repair_metadata_is_not_rejected():
    config = _backend_config()
    config["repair"] = {
        "plan_id": "plan-1",
        "parent_adapter_sha256": "a" * 64,
        "variant": "default",
    }
    validate_transformers_backend_config(config)


def test_replay_provenance_metadata_used_by_repair_pipeline_is_allowed():
    config = _backend_config()
    config["backend"]["replay"] = {
        "dataset": "replay.jsonl",
        "sha256": "r" * 64,
        "ratio": 1.0,
        "manifest": "replay.manifest.json",
        "manifest_sha256": "m" * 64,
    }
    validate_transformers_backend_config(config)


def test_misspelled_training_key_fails_closed():
    config = _backend_config()
    config["backend"]["training"]["learning_rtae"] = config["backend"]["training"].pop(
        "learning_rate"
    )
    with pytest.raises(ConfigValidationError, match="learning_rtae"):
        validate_transformers_backend_config(config)


def test_misspelled_backend_key_fails_closed():
    config = _backend_config()
    config["backend"]["quantiztion"] = config["backend"].pop("quantization")
    with pytest.raises(ConfigValidationError, match="quantiztion"):
        validate_transformers_backend_config(config)


def test_future_schema_version_is_rejected():
    config = _backend_config()
    config["backend"]["schema_version"] = TRANSFORMERS_BACKEND_SCHEMA_VERSION + 1
    with pytest.raises(ConfigValidationError, match="unsupported backend.schema_version"):
        validate_transformers_backend_config(config)


@pytest.mark.parametrize("bad", [True, 1.5, "2"])
def test_active_accelerator_count_requires_real_integer(bad):
    config = _backend_config()
    config["backend"]["runtime"]["active_accelerator_count"] = bad
    with pytest.raises(ConfigValidationError, match="must be an integer"):
        validate_transformers_backend_config(config)


def test_unknown_custom_backend_is_not_valid_for_transformers_executor():
    config = _backend_config()
    config["backend"]["type"] = "custom-research-backend"
    with pytest.raises(ConfigValidationError, match="unsupported backend.type"):
        validate_transformers_backend_config(config)


def _hardware():
    return HardwareProfile(16, 32, 100, 12, 40, 3)


def _experiment():
    return Experiment(
        "candidate",
        None,
        Hypothesis("obs", "cause", "fix"),
        {},
        0.5,
    )


def _engine():
    goal = Goal((MetricTarget("score", minimum=1),), gpu_hour_budget=2.0)
    return EvolutionEngine(goal, ExperimentResult("base", {"score": 1.0}, 0.0))


class _MustNotRunTransformersTrainer:
    name = "transformers-peft"

    def __init__(self):
        self.profile_called = False
        self.run_called = False

    def profile(self, experiment, context):
        self.profile_called = True
        return CostEstimate(gpu_hours=0.1)

    def run(self, experiment, context):
        self.run_called = True
        raise AssertionError("invalid config must never reach training")

    def cancel(self, run_id):
        return None


class _MustNotEvaluate:
    name = "unused"

    def profile(self, experiment, context):
        raise NotImplementedError

    def evaluate(self, *, experiment, artifact, context):
        raise AssertionError("invalid config must never reach evaluation")

    def cancel(self, run_id):
        return None


def test_cycle_rejects_typo_before_profile_and_charges_zero_compute(tmp_path):
    engine = _engine()
    experiment = _experiment()
    engine.propose((experiment,))
    config = _backend_config()
    config["backend"]["training"]["learning_rtae"] = config["backend"]["training"].pop(
        "learning_rate"
    )
    trainer = _MustNotRunTransformersTrainer()
    runner = ExperimentCycleRunner(
        engine=engine,
        trainer=trainer,
        evaluator=_MustNotEvaluate(),
        context=ExecutionContext(_hardware(), str(tmp_path), seed=1),
        base_config=config,
    )

    outcome = runner.run_generation((experiment,))

    assert not trainer.profile_called
    assert not trainer.run_called
    assert outcome.promoted is None
    assert outcome.candidates[0].error.startswith("preflight ConfigValidationError")
    assert engine.spent_gpu_hours == 0
    assert engine.reserved_gpu_hours == 0


class _CustomTrainer:
    name = "custom-executor"

    def profile(self, experiment, context):
        raise NotImplementedError

    def run(self, experiment, context):
        raise RuntimeError("expected custom runtime path")

    def cancel(self, run_id):
        return None


def test_custom_executor_is_not_forced_through_transformers_schema(tmp_path):
    engine = _engine()
    experiment = _experiment()
    engine.propose((experiment,))
    runner = ExperimentCycleRunner(
        engine=engine,
        trainer=_CustomTrainer(),
        evaluator=_MustNotEvaluate(),
        context=ExecutionContext(_hardware(), str(tmp_path), seed=1),
        base_config={"backend": {"type": "custom", "totally_custom_key": 42}},
    )

    outcome = runner.run_generation((experiment,))

    assert outcome.candidates[0].execution_failure is not None
    assert "ConfigValidationError" not in (outcome.candidates[0].error or "")
