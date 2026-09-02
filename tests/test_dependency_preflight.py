import pytest

from chowder.backends.transformers_peft import TransformersPeftExecutor
from chowder.cycle import ExperimentCycleRunner
from chowder.dependency_preflight import MissingDependencyError, check_dependencies
from chowder.engine import EvolutionEngine
from chowder.evaluators.transformers_text import TransformersTextEvaluator
from chowder.executors import EvaluationOutcome, ExecutionContext, TrainingArtifact
from chowder.memory import HardwareProfile
from chowder.models import Experiment, ExperimentResult, Goal, Hypothesis, MetricTarget


def test_passes_when_every_package_is_importable(monkeypatch):
    monkeypatch.setattr(
        "chowder.dependency_preflight.importlib.util.find_spec", lambda name: object()
    )
    check_dependencies(packages=("torch", "transformers"), quantization="none", label="t")


def test_raises_when_a_base_package_is_missing(monkeypatch):
    present = {"torch"}
    monkeypatch.setattr(
        "chowder.dependency_preflight.importlib.util.find_spec",
        lambda name: object() if name in present else None,
    )
    with pytest.raises(MissingDependencyError, match="missing required package.*transformers"):
        check_dependencies(packages=("torch", "transformers"), quantization="none", label="t")


def test_message_names_every_missing_package(monkeypatch):
    monkeypatch.setattr("chowder.dependency_preflight.importlib.util.find_spec", lambda name: None)
    with pytest.raises(MissingDependencyError) as excinfo:
        check_dependencies(packages=("torch", "transformers", "peft"), quantization="none", label="t")
    message = str(excinfo.value)
    assert "torch" in message
    assert "transformers" in message
    assert "peft" in message


def test_message_includes_the_label(monkeypatch):
    monkeypatch.setattr("chowder.dependency_preflight.importlib.util.find_spec", lambda name: None)
    with pytest.raises(MissingDependencyError, match="some-workload is missing"):
        check_dependencies(packages=("torch",), quantization="none", label="some-workload")


def test_no_packages_required_never_raises():
    check_dependencies(packages=(), quantization="none", label="anything")


def test_bitsandbytes_only_checked_when_quantization_is_4bit(monkeypatch):
    # torch/transformers present, bitsandbytes missing -- should only matter
    # when quantization is actually "4bit".
    present = {"torch", "transformers"}
    monkeypatch.setattr(
        "chowder.dependency_preflight.importlib.util.find_spec",
        lambda name: object() if name in present else None,
    )
    check_dependencies(packages=("torch", "transformers"), quantization="none", label="t")
    with pytest.raises(MissingDependencyError, match="bitsandbytes"):
        check_dependencies(packages=("torch", "transformers"), quantization="4bit", label="t")


def test_error_message_recommends_the_right_extras(monkeypatch):
    monkeypatch.setattr("chowder.dependency_preflight.importlib.util.find_spec", lambda name: None)
    with pytest.raises(MissingDependencyError, match=r"chowder-ai\[train\]$"):
        check_dependencies(packages=("torch",), quantization="none", label="t")
    with pytest.raises(MissingDependencyError, match=r"chowder-ai\[train\] and chowder-ai\[qlora\]$"):
        check_dependencies(packages=("torch",), quantization="4bit", label="t")


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


def test_cycle_rejects_missing_dependency_before_subprocess_spawn(tmp_path, monkeypatch):
    """A real TransformersPeftExecutor + TransformersTextEvaluator, with a
    simulated missing torch install -- proves the candidate is rejected at
    preflight, before engine.resize_reservation() commits any GPU-hours and
    before subprocess.Popen is ever reached (mocked to raise if called)."""
    monkeypatch.setattr(
        "chowder.dependency_preflight.importlib.util.find_spec",
        lambda name: None if name == "torch" else object(),
    )

    def should_not_launch(*args, **kwargs):
        raise AssertionError("worker must not launch when a dependency is missing")

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
        base_config={"backend": {"type": "transformers-peft"}},
    )
    outcome = runner.run_generation((experiment,))
    candidate = outcome.candidates[0]
    assert candidate.error is not None
    assert candidate.error.startswith("preflight")
    assert "torch" in candidate.error
    assert engine.spent_gpu_hours == 0
    assert engine.reserved_gpu_hours == 0
    assert not engine.has_reservation(experiment.experiment_id)


class _NamedLikeRealTrainer:
    """Shares the real executor's .name but is not an instance of it."""

    name = "transformers-peft"

    def profile(self, experiment, context):
        raise NotImplementedError

    def run(self, experiment, context):
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
        return EvaluationOutcome(
            "eval-1", experiment.experiment_id, artifact.artifact_ref, {"quality": 0.85}, 0.1, {}
        )

    def cancel(self, run_id):
        pass


def test_cycle_dependency_check_ignores_stubs_that_only_share_the_name(tmp_path, monkeypatch):
    """Dispatch is by real type (isinstance), not by the .name string a test
    double happens to share with the real backends -- a stub merely named
    "transformers-peft"/"transformers-text" must not be rejected for a
    simulated missing torch install it has nothing to do with."""
    monkeypatch.setattr(
        "chowder.dependency_preflight.importlib.util.find_spec", lambda name: None
    )
    engine = _engine()
    experiment = _experiment()
    engine.propose((experiment,))
    runner = ExperimentCycleRunner(
        engine=engine,
        trainer=_NamedLikeRealTrainer(),
        evaluator=_NamedLikeRealEvaluator(),
        context=ExecutionContext(_hardware(), str(tmp_path), 1),
        # Sharing the real name also routes through the real strict config
        # validation (_validate_trainer_config dispatches by name, not
        # isinstance -- unlike the dependency check this test exists to
        # verify), so this still needs a config that satisfies that.
        base_config={"backend": {"type": "transformers-peft"}},
    )
    outcome = runner.run_generation((experiment,))
    candidate = outcome.candidates[0]
    assert candidate.error is None
    assert candidate.result is not None
