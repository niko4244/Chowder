import os

import pytest

from chowder.backends.transformers_peft import TransformersPeftExecutor
from chowder.cycle import ExperimentCycleRunner
from chowder.engine import EvolutionEngine
from chowder.evaluators.transformers_text import TransformersTextEvaluator
from chowder.executors import ExecutionContext
from chowder.memory import HardwareProfile
from chowder.model_compatibility import (
    IncompatibleModelArchitectureError,
    check_causal_lm_architecture,
)
from chowder.models import Experiment, ExperimentResult, Goal, Hypothesis, MetricTarget


# These tests monkeypatch the real `transformers` package directly (rather
# than a chowder-owned wrapper), since the whole point of this check is
# introspecting transformers' own AutoModelForCausalLM registry -- so, like
# test_real_ml_training.py, they need [train] actually installed to even
# import the thing they're patching, not just to exercise a real backend.
pytestmark = pytest.mark.skipif(
    os.environ.get("CHOWDER_REAL_ML_SMOKE") != "1",
    reason="real ML smoke requires CHOWDER_REAL_ML_SMOKE=1 and train dependencies",
)


class _FakeCausalLMConfig:
    model_type = "fake-causal"


class _FakeEncoderOnlyConfig:
    model_type = "fake-encoder-only"


def test_passes_for_an_architecture_registered_under_causal_lm(monkeypatch):
    monkeypatch.setattr(
        "transformers.AutoConfig.from_pretrained",
        lambda *a, **k: _FakeCausalLMConfig(),
    )
    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM._model_mapping",
        {_FakeCausalLMConfig: object},
    )
    check_causal_lm_architecture(
        base_model="org/model", revision=None, offline=False, label="t"
    )


def test_raises_for_an_architecture_not_registered_under_causal_lm(monkeypatch):
    monkeypatch.setattr(
        "transformers.AutoConfig.from_pretrained",
        lambda *a, **k: _FakeEncoderOnlyConfig(),
    )
    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM._model_mapping",
        {_FakeCausalLMConfig: object},
    )
    with pytest.raises(IncompatibleModelArchitectureError, match="fake-encoder-only"):
        check_causal_lm_architecture(
            base_model="org/model", revision=None, offline=False, label="t"
        )


def test_skips_silently_when_model_mapping_introspection_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        "transformers.AutoConfig.from_pretrained",
        lambda *a, **k: _FakeEncoderOnlyConfig(),
    )

    class _NoMappingAutoModel:
        pass

    monkeypatch.setattr("transformers.AutoModelForCausalLM", _NoMappingAutoModel)
    check_causal_lm_architecture(
        base_model="org/model", revision=None, offline=False, label="t"
    )


def test_passes_revision_and_offline_through_to_from_pretrained(monkeypatch):
    seen = {}

    def fake_from_pretrained(base_model, *, revision, trust_remote_code, local_files_only):
        seen["base_model"] = base_model
        seen["revision"] = revision
        seen["trust_remote_code"] = trust_remote_code
        seen["local_files_only"] = local_files_only
        return _FakeCausalLMConfig()

    monkeypatch.setattr("transformers.AutoConfig.from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM._model_mapping",
        {_FakeCausalLMConfig: object},
    )
    check_causal_lm_architecture(
        base_model="org/model", revision="abc123", offline=True, label="t"
    )
    assert seen == {
        "base_model": "org/model",
        "revision": "abc123",
        "trust_remote_code": False,
        "local_files_only": True,
    }


# --- dispatch through ExperimentCycleRunner's real preflight -----------------


def _engine():
    return EvolutionEngine(
        Goal((MetricTarget("quality", minimum=0.8),), gpu_hour_budget=10),
        ExperimentResult("base", {"quality": 0.7}, 0),
    )


def _experiment():
    return Experiment("e1", None, Hypothesis("obs", "cause", "fix"), {}, 1.0)


def _hardware():
    return HardwareProfile(16, 64, 500, 12, 40, 3)


def test_cycle_rejects_incompatible_architecture_before_subprocess_spawn(tmp_path, monkeypatch):
    """A real TransformersPeftExecutor + TransformersTextEvaluator, with a
    simulated non-causal-LM architecture -- proves the candidate is rejected
    at preflight, before engine.resize_reservation() commits any GPU-hours
    and before subprocess.Popen is ever reached (mocked to raise if
    called)."""
    monkeypatch.setattr(
        "transformers.AutoConfig.from_pretrained",
        lambda *a, **k: _FakeEncoderOnlyConfig(),
    )
    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM._model_mapping",
        {_FakeCausalLMConfig: object},
    )

    def should_not_launch(*args, **kwargs):
        raise AssertionError("worker must not launch for an incompatible architecture")

    monkeypatch.setattr(
        "chowder.backends.transformers_peft.subprocess.Popen", should_not_launch
    )

    engine = _engine()
    experiment = _experiment()
    engine.propose((experiment,))
    runner = ExperimentCycleRunner(
        engine=engine,
        trainer=TransformersPeftExecutor(),
        evaluator=TransformersTextEvaluator(),
        context=ExecutionContext(_hardware(), str(tmp_path), 1),
        base_config={
            "backend": {"type": "transformers-peft", "base_model": "org/not-a-causal-lm"}
        },
    )
    outcome = runner.run_generation((experiment,))
    candidate = outcome.candidates[0]
    assert candidate.error is not None
    assert candidate.error.startswith("preflight")
    assert "IncompatibleModelArchitectureError" in candidate.error
    assert engine.spent_gpu_hours == 0
    assert engine.reserved_gpu_hours == 0
    assert not engine.has_reservation(experiment.experiment_id)


def test_cycle_architecture_check_skipped_when_base_model_is_absent(tmp_path, monkeypatch):
    def should_not_be_called(*a, **k):
        raise AssertionError("AutoConfig.from_pretrained must not be called")

    monkeypatch.setattr("transformers.AutoConfig.from_pretrained", should_not_be_called)

    engine = _engine()
    experiment = _experiment()
    engine.propose((experiment,))
    runner = ExperimentCycleRunner(
        engine=engine,
        trainer=TransformersPeftExecutor(),
        evaluator=TransformersTextEvaluator(),
        context=ExecutionContext(_hardware(), str(tmp_path), 1),
        base_config={"backend": {"type": "transformers-peft"}},
    )
    outcome = runner.run_generation((experiment,))
    # No base_model configured -- training itself fails for an unrelated
    # reason once it actually launches, but the architecture check must not
    # have been the thing that rejected it, and must never have touched the
    # network to get there.
    candidate = outcome.candidates[0]
    assert candidate.error is None or "IncompatibleModelArchitectureError" not in candidate.error


class _NamedLikeRealTrainer:
    """Shares the real executor's .name but is not an instance of it."""

    name = "transformers-peft"

    def profile(self, experiment, context):
        raise NotImplementedError

    def run(self, experiment, context):
        from chowder.executors import TrainingArtifact

        return TrainingArtifact(
            "train-1", experiment.experiment_id, "/artifact", 0.4, evidence={"sha": "x"}
        )

    def cancel(self, run_id):
        pass


class _NamedLikeRealEvaluator:
    """Shares the real evaluator's .name but is not an instance of it."""

    name = "transformers-text"

    def profile(self, experiment, context):
        raise NotImplementedError

    def evaluate(self, *, experiment, artifact, context):
        from chowder.executors import EvaluationOutcome

        return EvaluationOutcome(
            "eval-1", experiment.experiment_id, artifact.artifact_ref, {"quality": 0.85}, 0.1, {}
        )

    def cancel(self, run_id):
        pass


def test_cycle_architecture_check_ignores_stubs_that_only_share_the_name(tmp_path, monkeypatch):
    """Dispatch is by real type (isinstance), not by the .name string a test
    double happens to share with the real backends -- a stub merely named
    "transformers-peft"/"transformers-text" with a fake base_model must not
    trigger a real network call or be rejected for an architecture mismatch
    that has nothing to do with it."""

    def should_not_be_called(*a, **k):
        raise AssertionError("AutoConfig.from_pretrained must not be called for a stub")

    monkeypatch.setattr("transformers.AutoConfig.from_pretrained", should_not_be_called)

    engine = _engine()
    experiment = _experiment()
    engine.propose((experiment,))
    runner = ExperimentCycleRunner(
        engine=engine,
        trainer=_NamedLikeRealTrainer(),
        evaluator=_NamedLikeRealEvaluator(),
        context=ExecutionContext(_hardware(), str(tmp_path), 1),
        base_config={
            "backend": {"type": "transformers-peft", "base_model": "org/whatever"}
        },
    )
    outcome = runner.run_generation((experiment,))
    candidate = outcome.candidates[0]
    assert candidate.error is None
    assert candidate.result is not None
