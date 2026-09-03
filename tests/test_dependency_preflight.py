import pytest

from chowder.backends.transformers_peft import TransformersPeftExecutor
from chowder.cycle import ExperimentCycleRunner, _check_dependencies, _check_memory_fit
from chowder.dependency_preflight import (
    InsufficientDiskSpaceError,
    MissingDependencyError,
    check_dependencies,
    check_disk_space,
)
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


def test_check_dependencies_requires_bitsandbytes_when_optimizer_tiering_always(monkeypatch):
    """optimizer_tiering="always" bypasses run_optimizer_tiering_experiment's
    own graceful degradation entirely and unconditionally constructs a real
    paged optimizer -- unlike quantization="4bit" being the only other
    bitsandbytes trigger, this must be checked even when quantization is
    "none"."""
    present = {"torch", "transformers", "peft", "datasets", "accelerate"}
    monkeypatch.setattr(
        "chowder.dependency_preflight.importlib.util.find_spec",
        lambda name: object() if name in present else None,
    )
    resolved = {
        "backend": {
            "type": "transformers-peft",
            "quantization": "none",
            "training": {"optimizer_tiering": "always"},
        }
    }
    context = ExecutionContext(_hardware(), ".", 1, resolved_config=resolved)
    with pytest.raises(MissingDependencyError, match="bitsandbytes"):
        _check_dependencies(TransformersPeftExecutor(), None, resolved, context)


def test_check_dependencies_does_not_require_bitsandbytes_for_optimizer_tiering_auto(monkeypatch):
    """"auto" is not checked explicitly here: its own real experiment
    already degrades gracefully to unavailable/not-recommended when
    bitsandbytes is missing, so a separate preflight rejection would only
    block a config that would otherwise train successfully with tiering
    simply declining."""
    present = {"torch", "transformers", "peft", "datasets", "accelerate"}
    monkeypatch.setattr(
        "chowder.dependency_preflight.importlib.util.find_spec",
        lambda name: object() if name in present else None,
    )
    resolved = {
        "backend": {
            "type": "transformers-peft",
            "quantization": "none",
            "training": {"optimizer_tiering": "auto"},
        }
    }
    context = ExecutionContext(_hardware(), ".", 1, resolved_config=resolved)
    _check_dependencies(TransformersPeftExecutor(), None, resolved, context)


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


def test_cycle_rejects_optimizer_tiering_always_without_bitsandbytes(tmp_path, monkeypatch):
    """A real TransformersPeftExecutor with optimizer_tiering="always" and a
    simulated missing bitsandbytes install -- proves this is rejected at
    preflight (config time), not discovered inside the spawned worker after
    GPU-hours were already reserved and paged_adamw_32bit construction
    fails deep inside TrainingArguments/Trainer."""
    present = {"torch", "transformers", "peft", "datasets", "accelerate"}
    monkeypatch.setattr(
        "chowder.dependency_preflight.importlib.util.find_spec",
        lambda name: object() if name in present else None,
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
        base_config={
            "backend": {
                "type": "transformers-peft",
                "training": {"optimizer_tiering": "always"},
            }
        },
    )
    outcome = runner.run_generation((experiment,))
    candidate = outcome.candidates[0]
    assert candidate.error is not None
    assert candidate.error.startswith("preflight")
    assert "bitsandbytes" in candidate.error
    assert engine.spent_gpu_hours == 0
    assert engine.reserved_gpu_hours == 0
    assert not engine.has_reservation(experiment.experiment_id)


def test_cycle_rejects_a_recipe_that_does_not_fit_when_memory_preflight_is_enabled(
    tmp_path, monkeypatch
):
    """A real TransformersPeftExecutor with memory_preflight="always" and a
    simulated real dry-run measurement that clearly does not fit -- proves
    this is rejected at preflight (config time), not discovered as a real
    OOM deep inside a spawned training subprocess after GPU-hours were
    already reserved. Completes the "memory preflight integration" the
    real dry run (memory_preflight.py) already had a mechanism for but was
    never wired into the automated cycle before this policy existed."""
    monkeypatch.setattr(
        "chowder.dependency_preflight.importlib.util.find_spec", lambda name: object()
    )
    monkeypatch.setattr("chowder.cycle.check_causal_lm_architecture", lambda **kwargs: None)
    monkeypatch.setattr(
        "chowder.memory_preflight._run_dry_run_worker",
        lambda *args, **kwargs: {
            "device": "cuda",
            "frozen_params": 1000,
            "trainable_params": 10,
            "max_length": 64,
            "peak_vram_gb_bs1": 100.0,
            "peak_vram_gb_bs2": 150.0,
        },
    )

    def should_not_launch(*args, **kwargs):
        raise AssertionError("worker must not launch when the recipe does not fit")

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
            "backend": {
                "type": "transformers-peft",
                "base_model": "org/model",
                "dataset": str(tmp_path / "train.jsonl"),
                "memory_preflight": "always",
            }
        },
    )
    (tmp_path / "train.jsonl").write_text('{"text":"hello"}\n')
    outcome = runner.run_generation((experiment,))
    candidate = outcome.candidates[0]
    assert candidate.error is not None
    assert candidate.error.startswith("preflight")
    assert "not expected to fit" in candidate.error
    assert engine.spent_gpu_hours == 0
    assert engine.reserved_gpu_hours == 0
    assert not engine.has_reservation(experiment.experiment_id)


def test_check_memory_fit_allows_a_fitting_recipe(tmp_path, monkeypatch):
    """The rejection is specifically about a recipe that does not fit --
    memory_preflight="always" combined with a real measurement that
    clearly does fit must never raise. Direct unit-level call, matching
    this file's own established pattern for "doesn't reject" cases
    (e.g. test_passes_when_every_package_is_importable)."""
    monkeypatch.setattr(
        "chowder.memory_preflight._run_dry_run_worker",
        lambda *args, **kwargs: {
            "device": "cuda",
            "frozen_params": 1000,
            "trainable_params": 10,
            "max_length": 64,
            "peak_vram_gb_bs1": 1.0,
            "peak_vram_gb_bs2": 1.5,
        },
    )
    resolved = {
        "backend": {
            "type": "transformers-peft",
            "base_model": "org/model",
            "memory_preflight": "always",
        }
    }
    context = ExecutionContext(_hardware(), str(tmp_path), 1)
    _check_memory_fit(TransformersPeftExecutor(), None, resolved, context)  # must not raise


def test_check_memory_fit_is_a_no_op_when_policy_is_off(tmp_path, monkeypatch):
    def should_not_measure(*args, **kwargs):
        raise AssertionError("must not run a dry-run worker when memory_preflight is off")

    monkeypatch.setattr("chowder.memory_preflight._run_dry_run_worker", should_not_measure)
    resolved = {"backend": {"type": "transformers-peft", "base_model": "org/model"}}
    context = ExecutionContext(_hardware(), str(tmp_path), 1)
    _check_memory_fit(TransformersPeftExecutor(), None, resolved, context)  # must not raise


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


# --- disk-space preflight ----------------------------------------------------


def test_check_disk_space_passes_when_plenty_free(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "chowder.dependency_preflight.shutil.disk_usage",
        lambda path: type("Usage", (), {"free": 10 * 1024**3})(),
    )
    check_disk_space(path=tmp_path, minimum_free_gb=2.0, label="t")


def test_check_disk_space_raises_when_below_minimum(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "chowder.dependency_preflight.shutil.disk_usage",
        lambda path: type("Usage", (), {"free": 1 * 1024**3})(),
    )
    with pytest.raises(InsufficientDiskSpaceError, match="2.00 GB.*1.00 GB"):
        check_disk_space(path=tmp_path, minimum_free_gb=2.0, label="t")


def test_check_disk_space_zero_minimum_never_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "chowder.dependency_preflight.shutil.disk_usage",
        lambda path: type("Usage", (), {"free": 0})(),
    )
    check_disk_space(path=tmp_path, minimum_free_gb=0.0, label="t")


def test_check_disk_space_measures_nearest_existing_ancestor(tmp_path, monkeypatch):
    seen: list = []

    def fake_disk_usage(path):
        seen.append(path)
        return type("Usage", (), {"free": 10 * 1024**3})()

    monkeypatch.setattr("chowder.dependency_preflight.shutil.disk_usage", fake_disk_usage)
    missing = tmp_path / "does" / "not" / "exist"
    check_disk_space(path=missing, minimum_free_gb=1.0, label="t")
    assert seen == [tmp_path]


def test_cycle_rejects_insufficient_disk_space_before_subprocess_spawn(tmp_path, monkeypatch):
    """A real TransformersPeftExecutor + TransformersTextEvaluator, with a
    simulated near-empty disk -- proves the candidate is rejected at
    preflight, before engine.resize_reservation() commits any GPU-hours and
    before subprocess.Popen is ever reached (mocked to raise if called)."""
    monkeypatch.setattr(
        "chowder.dependency_preflight.shutil.disk_usage",
        lambda path: type("Usage", (), {"free": 1024})(),  # 1 KB free
    )
    # Must reach the disk-space check regardless of whether [train] is
    # actually installed on the machine running this test -- simulate every
    # package as present so the earlier dependency check always passes.
    monkeypatch.setattr(
        "chowder.dependency_preflight.importlib.util.find_spec", lambda name: object()
    )

    def should_not_launch(*args, **kwargs):
        raise AssertionError("worker must not launch when disk space is insufficient")

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
    assert "InsufficientDiskSpaceError" in candidate.error
    assert engine.spent_gpu_hours == 0
    assert engine.reserved_gpu_hours == 0
    assert not engine.has_reservation(experiment.experiment_id)


def test_cycle_disk_space_check_ignores_stubs_that_only_share_the_name(tmp_path, monkeypatch):
    """Even with a configured minimum no simulated-low-disk mock could
    satisfy, a stub that only shares the real backends' .name must not be
    rejected -- dispatch is by isinstance, matching _check_dependencies."""
    monkeypatch.setattr(
        "chowder.dependency_preflight.shutil.disk_usage",
        lambda path: type("Usage", (), {"free": 0})(),  # 0 bytes free
    )
    engine = _engine()
    experiment = _experiment()
    engine.propose((experiment,))
    runner = ExperimentCycleRunner(
        engine=engine,
        trainer=_NamedLikeRealTrainer(),
        evaluator=_NamedLikeRealEvaluator(),
        context=ExecutionContext(_hardware(), str(tmp_path), 1),
        base_config={
            "backend": {"type": "transformers-peft", "min_free_disk_gb": 5.0}
        },
    )
    outcome = runner.run_generation((experiment,))
    candidate = outcome.candidates[0]
    assert candidate.error is None
    assert candidate.result is not None
