from __future__ import annotations

import os
from pathlib import Path

import pytest

from chowder.cycle import ExperimentCycleRunner
from chowder.engine import EvolutionEngine
from chowder.executors import EvaluationOutcome, ExecutionContext, TrainingArtifact
from chowder.memory import HardwareProfile
from chowder.models import Experiment, ExperimentResult, Goal, Hypothesis, MetricTarget
from chowder.successive_halving import run_successive_halving

_REAL_ML_SMOKE = pytest.mark.skipif(
    os.environ.get("CHOWDER_REAL_ML_SMOKE") != "1",
    reason="real ML smoke requires CHOWDER_REAL_ML_SMOKE=1 and train dependencies",
)
_TINY_MODEL = "trl-internal-testing/tiny-LlamaForCausalLM-3.2"


def _lineage_root(experiment_id: str) -> str:
    return experiment_id.split("-r")[0]


class FakeTrainer:
    """Writes a real, minimal on-disk checkpoint layout so
    successive_halving's own real filesystem discovery (globbing
    trainer/checkpoint-N) is exercised the same way it would be against
    a real TransformersPeftExecutor -- without spawning a real training
    subprocess. Records the resolved config each call actually received
    so tests can assert on real round-to-round wiring (max_steps,
    resume_from_checkpoint)."""

    name = "fake-trainer"

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.calls: list[dict] = []

    def profile(self, experiment, context):
        raise NotImplementedError

    def run(self, experiment, context):
        backend = context.resolved_config.get("backend", {})
        training = backend.get("training", {})
        self.calls.append(
            {
                "experiment_id": experiment.experiment_id,
                "max_steps": training.get("max_steps"),
                "resume_from_checkpoint": backend.get("resume_from_checkpoint"),
                "save_strategy": training.get("save_strategy"),
            }
        )
        output_dir = self.base_dir / experiment.experiment_id / "adapter"
        output_dir.mkdir(parents=True, exist_ok=True)
        if training.get("save_strategy") == "steps":
            max_steps = int(training.get("max_steps", 1))
            (output_dir / "trainer" / f"checkpoint-{max_steps}").mkdir(parents=True, exist_ok=True)
        return TrainingArtifact(
            f"train-{experiment.experiment_id}", experiment.experiment_id, str(output_dir), 0.01,
            evidence={"sha": "x"},
        )

    def cancel(self, run_id):
        pass


class FakeEvaluator:
    name = "fake-eval"

    def __init__(self, scores: dict[str, float]):
        self.scores = scores

    def profile(self, experiment, context):
        raise NotImplementedError

    def evaluate(self, *, experiment, artifact, context):
        score = self.scores[_lineage_root(experiment.experiment_id)]
        return EvaluationOutcome(
            f"eval-{experiment.experiment_id}", experiment.experiment_id, artifact.artifact_ref,
            {"quality": score}, 0.001, {"suite": "fake"},
        )

    def cancel(self, run_id):
        pass


def _hardware():
    return HardwareProfile(16, 64, 500, 12, 40, 3)


def _experiment(name: str, hours: float = 0.02) -> Experiment:
    return Experiment(name, None, Hypothesis("obs", "cause", "fix"), {}, hours)


def _runner(tmp_path, trainer: FakeTrainer, evaluator: FakeEvaluator, *, minimum=0.0, budget=100.0):
    goal = Goal((MetricTarget("quality", minimum=minimum),), gpu_hour_budget=budget, max_parallel_candidates=8)
    baseline = ExperimentResult("baseline", {"quality": 0.05}, 0.0)
    engine = EvolutionEngine(goal, baseline)
    return ExperimentCycleRunner(
        engine=engine,
        trainer=trainer,
        evaluator=evaluator,
        context=ExecutionContext(_hardware(), str(tmp_path), 1),
        base_config={
            "backend": {
                "type": "transformers-peft",
                "base_model": "org/model",
                "dataset": "train.jsonl",
                "lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]},
                "training": {"batch_size": 1},
            }
        },
    )


def test_round_zero_sets_max_steps_and_save_strategy(tmp_path):
    trainer = FakeTrainer(tmp_path)
    evaluator = FakeEvaluator({"a": 0.9, "b": 0.8})
    runner = _runner(tmp_path, trainer, evaluator)
    run_successive_halving(
        runner, (_experiment("a"), _experiment("b")),
        initial_max_steps=2, min_survivors=1,
    )
    for call in trainer.calls:
        if call["experiment_id"] in ("a", "b"):
            assert call["max_steps"] == 2
            assert call["save_strategy"] == "steps"
            assert call["resume_from_checkpoint"] is None


def test_survivor_advances_with_doubled_steps_and_real_resume_path(tmp_path):
    """4 candidates so round 0's ceil(4*0.5)=2 survivors genuinely need
    a second round to narrow to min_survivors=1 -- with only 2 initial
    candidates, round 0 alone would already reach exactly 1 survivor
    and there would be nothing left to test in round 1."""
    trainer = FakeTrainer(tmp_path)
    evaluator = FakeEvaluator({"a": 0.9, "b": 0.8, "c": 0.2, "d": 0.1})
    runner = _runner(tmp_path, trainer, evaluator)
    outcome = run_successive_halving(
        runner, tuple(_experiment(n) for n in ("a", "b", "c", "d")),
        initial_max_steps=2, step_multiplier=2.0, survival_fraction=0.5, min_survivors=1,
    )
    assert outcome.total_rounds == 2
    round1_call = next(c for c in trainer.calls if c["experiment_id"] == "a-r1")
    assert round1_call["max_steps"] == 4
    assert round1_call["resume_from_checkpoint"] is not None
    assert "checkpoint-2" in round1_call["resume_from_checkpoint"]


def test_only_the_final_round_is_promoted(tmp_path):
    """Round 0's winner ("a") must NOT become the baseline just because
    it topped round 0 -- only round 1's real winner does."""
    trainer = FakeTrainer(tmp_path)
    evaluator = FakeEvaluator({"a": 0.9, "b": 0.8, "c": 0.2, "d": 0.1})
    runner = _runner(tmp_path, trainer, evaluator)
    outcome = run_successive_halving(
        runner, tuple(_experiment(n) for n in ("a", "b", "c", "d")),
        initial_max_steps=2, min_survivors=1,
    )
    assert outcome.promoted is not None
    assert outcome.promoted.experiment_id == "a-r1"
    assert runner.engine.baseline.experiment_id == "a-r1"


def test_cutoff_elimination_is_tracked_separately_from_gate_rejection(tmp_path):
    trainer = FakeTrainer(tmp_path)
    evaluator = FakeEvaluator({"a": 0.9, "b": 0.8, "c": 0.7, "d": 0.6})
    runner = _runner(tmp_path, trainer, evaluator, minimum=0.0)
    outcome = run_successive_halving(
        runner, tuple(_experiment(n) for n in ("a", "b", "c", "d")),
        initial_max_steps=2, survival_fraction=0.5, min_survivors=1,
    )
    round0 = outcome.rounds[0]
    assert set(round0.survivor_experiment_ids) == {"a", "b"}
    assert set(round0.eliminated_by_cutoff_experiment_ids) == {"c", "d"}
    assert round0.eliminated_by_gate_experiment_ids == ()


def test_gate_rejected_candidates_are_never_treated_as_survivors(tmp_path):
    """A candidate that fails the hard regression gate must be excluded
    from survival entirely, distinct from merely ranking below the
    cutoff -- gate rejection is never a probabilistic ranking call."""
    trainer = FakeTrainer(tmp_path)
    # "b" scores below the goal's minimum -- a real hard gate rejection,
    # not just a low rank.
    evaluator = FakeEvaluator({"a": 0.9, "b": 0.01})
    runner = _runner(tmp_path, trainer, evaluator, minimum=0.5)
    outcome = run_successive_halving(
        runner, (_experiment("a"), _experiment("b")),
        initial_max_steps=2, min_survivors=1,
    )
    round0 = outcome.rounds[0]
    assert round0.eliminated_by_gate_experiment_ids == ("b",)
    assert round0.eliminated_by_cutoff_experiment_ids == ()
    assert round0.survivor_experiment_ids == ("a",)
    # Only one real survivor -- this must already be the final round.
    assert outcome.total_rounds == 1
    assert outcome.promoted.experiment_id == "a"


def test_stops_immediately_once_min_survivors_is_reached(tmp_path):
    trainer = FakeTrainer(tmp_path)
    evaluator = FakeEvaluator({"a": 0.9})
    runner = _runner(tmp_path, trainer, evaluator)
    outcome = run_successive_halving(
        runner, (_experiment("a"),), initial_max_steps=2, min_survivors=1,
    )
    assert outcome.total_rounds == 1
    assert outcome.promoted.experiment_id == "a"


def test_max_rounds_caps_the_search_even_with_more_survivors_possible(tmp_path):
    trainer = FakeTrainer(tmp_path)
    evaluator = FakeEvaluator({"a": 0.9, "b": 0.8, "c": 0.7, "d": 0.6})
    runner = _runner(tmp_path, trainer, evaluator)
    outcome = run_successive_halving(
        runner, tuple(_experiment(n) for n in ("a", "b", "c", "d")),
        initial_max_steps=2, survival_fraction=0.5, min_survivors=1, max_rounds=1,
    )
    assert outcome.total_rounds == 1
    # Round 0's own winner is promoted since max_rounds forced a stop --
    # still only ever the LAST round actually run, never an earlier one
    # promoted out from under a still-running search.
    assert outcome.promoted.experiment_id == "a"


def test_rejects_invalid_survival_fraction(tmp_path):
    trainer = FakeTrainer(tmp_path)
    evaluator = FakeEvaluator({"a": 0.9})
    runner = _runner(tmp_path, trainer, evaluator)
    with pytest.raises(ValueError, match="survival_fraction"):
        run_successive_halving(runner, (_experiment("a"),), initial_max_steps=2, survival_fraction=0.0)
    with pytest.raises(ValueError, match="survival_fraction"):
        run_successive_halving(runner, (_experiment("a"),), initial_max_steps=2, survival_fraction=1.0)


def test_rejects_invalid_step_multiplier(tmp_path):
    trainer = FakeTrainer(tmp_path)
    evaluator = FakeEvaluator({"a": 0.9})
    runner = _runner(tmp_path, trainer, evaluator)
    with pytest.raises(ValueError, match="step_multiplier"):
        run_successive_halving(runner, (_experiment("a"),), initial_max_steps=2, step_multiplier=1.0)


def test_total_gpu_hours_sums_every_real_round(tmp_path):
    trainer = FakeTrainer(tmp_path)
    evaluator = FakeEvaluator({"a": 0.9, "b": 0.1})
    runner = _runner(tmp_path, trainer, evaluator)
    outcome = run_successive_halving(
        runner, (_experiment("a"), _experiment("b")), initial_max_steps=2, min_survivors=1,
    )
    expected = sum(
        c.result.gpu_hours
        for r in outcome.rounds
        for c in r.generation.candidates
        if c.result is not None
    )
    assert outcome.total_gpu_hours == pytest.approx(expected)
    assert outcome.total_gpu_hours > 0


# --- real end-to-end (real Trainer, controllable fake evaluation) -----------


@_REAL_ML_SMOKE
def test_real_successive_halving_chains_a_real_checkpoint_across_rounds(tmp_path):
    """Real TransformersPeftExecutor end to end: round 0 trains 4 real
    candidates for real, round 1 genuinely resumes the real winner's
    real checkpoint and trains additional real steps on top of it (not
    a restart from scratch -- global_step proves this), and only the
    real round-1 winner is promoted."""
    import json

    from chowder.backends.transformers_peft import TransformersPeftExecutor

    data = tmp_path / "train.jsonl"
    rows = [{"text": f"Question: what is {i}? Answer: {i * 2}"} for i in range(20)]
    data.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    scores = {"cand-a": 0.95, "cand-b": 0.85, "cand-c": 0.4, "cand-d": 0.3}
    evaluator = FakeEvaluator(scores)

    goal = Goal((MetricTarget("quality", minimum=0.2),), gpu_hour_budget=100.0, max_parallel_candidates=8)
    baseline = ExperimentResult("baseline", {"quality": 0.1}, 0.0)
    engine = EvolutionEngine(goal, baseline)
    runner = ExperimentCycleRunner(
        engine=engine,
        trainer=TransformersPeftExecutor(),
        evaluator=evaluator,
        context=ExecutionContext(_hardware(), str(tmp_path), 1),
        base_config={
            "backend": {
                "type": "transformers-peft",
                "base_model": _TINY_MODEL,
                "dataset": str(data),
                "max_length": 64,
                "quantization": "none",
                "precision": "fp32",
                "lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]},
                "training": {
                    "learning_rate": 0.001, "batch_size": 1,
                    "gradient_accumulation_steps": 1, "logging_steps": 1,
                    "gradient_checkpointing": False,
                },
                "runtime": {"timeout_seconds": 180.0},
            }
        },
    )
    experiments = tuple(_experiment(n) for n in scores)

    outcome = run_successive_halving(
        runner, experiments, initial_max_steps=2, step_multiplier=2.0,
        survival_fraction=0.5, min_survivors=1,
    )

    assert outcome.total_rounds == 2
    round0, round1 = outcome.rounds
    assert set(round0.survivor_experiment_ids) == {"cand-a", "cand-b"}
    assert set(round0.eliminated_by_cutoff_experiment_ids) == {"cand-c", "cand-d"}
    assert round1.survivor_experiment_ids == ("cand-a-r1",)
    assert outcome.promoted is not None
    assert outcome.promoted.experiment_id == "cand-a-r1"
    assert engine.baseline.experiment_id == "cand-a-r1"

    round1_winner = next(c for c in round1.generation.candidates if c.experiment_id == "cand-a-r1")
    assert round1_winner.artifact is not None
    # 2 real steps in round 0 + 2 more real steps in round 1 = 4 --
    # proves this is a real resume, not a restart from scratch.
    assert round1_winner.artifact.telemetry["global_step"] == 4
