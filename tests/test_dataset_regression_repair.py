import json
from pathlib import Path

import pytest

from chowder.contamination import write_holdout_fingerprint_index
from chowder.cycle import ExperimentCycleRunner
from chowder.dataset_regression_repair import (
    build_training_regression_repair_plan,
    build_training_regression_repair_request,
    run_training_regression_repair,
    verified_last_good_checkpoint_adapter,
)
from chowder.engine import EvolutionEngine
from chowder.executors import EvaluationOutcome, ExecutionContext, TrainingArtifact
from chowder.local_corpus_provider import LocalCorpusRepairProvider
from chowder.memory import HardwareProfile
from chowder.models import Experiment, ExperimentResult, Goal, Hypothesis, MetricTarget
from chowder.provenance import sha256_directory, sha256_file
from chowder.repair_candidates import RepairVariant
from chowder.training_sample_clusters import TrainingSampleCluster


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _cluster():
    return TrainingSampleCluster(
        cluster_id="c" * 64,
        member_row_indices=(3, 7),
        representative_excerpt="repetitive offending training row",
        mean_influence_score=0.4,
        max_influence_score=0.55,
        member_confidences=("high", "medium"),
    )


class FakeTrainer:
    name = "fake-trainer"

    def profile(self, experiment, context):
        raise NotImplementedError

    def run(self, experiment, context):
        artifact = Path(context.work_dir) / f"artifact-{experiment.experiment_id}"
        artifact.mkdir(exist_ok=False)
        (artifact / "adapter_config.json").write_text('{"peft_type":"LORA"}\n', encoding="utf-8")
        (artifact / "adapter.bin").write_bytes(experiment.experiment_id.encode("utf-8"))
        backend = context.resolved_config["backend"]
        dataset = Path(str(backend["dataset"]))
        if not dataset.is_absolute():
            dataset = Path(context.work_dir) / dataset
        return TrainingArtifact(
            run_id=f"train-{experiment.experiment_id}",
            experiment_id=experiment.experiment_id,
            artifact_ref=str(artifact),
            gpu_hours=0.2,
            evidence={
                "test": True,
                "dataset_sha256": sha256_file(dataset.resolve()),
                "artifact_sha256": sha256_directory(artifact),
            },
        )

    def cancel(self, run_id):
        pass


class FakeEvaluator:
    """Every repair-tagged candidate scores above the gate; the parent does not."""

    name = "fake-eval"

    def profile(self, experiment, context):
        raise NotImplementedError

    def cancel(self, run_id):
        pass

    def evaluate(self, *, experiment, artifact, context):
        is_repair = "repair" in experiment.tags
        quality = 0.86 if is_repair else 0.65
        return EvaluationOutcome(
            run_id=f"eval-{experiment.experiment_id}",
            experiment_id=experiment.experiment_id,
            source_artifact_ref=artifact.artifact_ref,
            metrics={"quality": quality},
            gpu_hours=0.05,
            evidence={},
        )


class AlwaysRegressingEvaluator(FakeEvaluator):
    """No candidate -- repair or otherwise -- ever clears the gate."""

    def evaluate(self, *, experiment, artifact, context):
        outcome = super().evaluate(experiment=experiment, artifact=artifact, context=context)
        return EvaluationOutcome(
            outcome.run_id,
            outcome.experiment_id,
            outcome.source_artifact_ref,
            {"quality": 0.5},
            outcome.gpu_hours,
            outcome.evidence,
        )


def _runner(tmp_path, *, evaluator=None):
    _write_jsonl(
        tmp_path / "base.jsonl",
        [
            {"text": "original training example one"},
            {"text": "original training example two"},
        ],
    )
    engine = EvolutionEngine(
        Goal(
            (MetricTarget("quality", minimum=0.8),),
            gpu_hour_budget=4.0,
            max_parallel_candidates=2,
        ),
        ExperimentResult("baseline", {"quality": 0.7}, 0.0),
    )
    context = ExecutionContext(HardwareProfile(16, 64, 500, 12, 40, 3), str(tmp_path), 7)
    return ExperimentCycleRunner(
        engine=engine,
        trainer=FakeTrainer(),
        evaluator=evaluator or FakeEvaluator(),
        context=context,
        base_config={
            "backend": {
                "base_model": "example/model",
                "dataset": "base.jsonl",
                "training": {"learning_rate": 1e-4, "epochs": 2},
                "lora": {"r": 16, "alpha": 32},
            },
            "evaluation": {"type": "transformers-text"},
        },
    )


def _provider(tmp_path):
    corpus = tmp_path / "repair-corpus.jsonl"
    _write_jsonl(
        corpus,
        [
            {
                "example_id": "counter-1",
                "suite": "training-corpus",
                "strategy": "near_neighbor_reasoning",
                "prompt": "independent counterexample one",
                "expected": "answer one",
            },
            {
                "example_id": "counter-2",
                "suite": "training-corpus",
                "strategy": "near_neighbor_reasoning",
                "prompt": "independent counterexample two",
                "expected": "answer two",
            },
        ],
    )
    return LocalCorpusRepairProvider([corpus], max_examples=2, examples_per_failure=2)


def _holdout(tmp_path):
    path = tmp_path / "holdout-index.jsonl"
    write_holdout_fingerprint_index([("benchmark prompt", "benchmark answer")], path)
    return path


def _parent_experiment():
    return Experiment(
        experiment_id="parent",
        parent_id=None,
        hypothesis=Hypothesis("baseline run", "n/a", "initial training"),
        config_patch={},
        estimated_gpu_hours=0.5,
    )


def test_build_request_and_plan_carry_honest_diagnostic_identity():
    cluster = _cluster()
    request = build_training_regression_repair_request(
        cluster, good_checkpoint_dir="/ckpt/good", bad_checkpoint_dir="/ckpt/bad"
    )
    plan = build_training_regression_repair_plan(
        cluster, good_checkpoint_dir="/ckpt/good", bad_checkpoint_dir="/ckpt/bad"
    )
    assert request.evaluator == "dataset-influence"
    assert request.suite == "training-corpus"
    assert request.failure_kind == "training_example_regression"
    assert request.failure_count == 2
    assert request.cluster_id == cluster.cluster_id
    assert plan.cluster_id == cluster.cluster_id
    assert plan.direct_training_allowed is False
    assert plan.requires_independent_source is True
    assert plan.source_failure_ids == ("3", "7")


def test_identical_checkpoint_directories_are_rejected():
    with pytest.raises(ValueError, match="must differ"):
        build_training_regression_repair_request(
            _cluster(), good_checkpoint_dir="/ckpt/x", bad_checkpoint_dir="/ckpt/x"
        )


def test_verified_last_good_checkpoint_adapter_detects_tamper(tmp_path):
    checkpoint = tmp_path / "checkpoint-good"
    checkpoint.mkdir()
    (checkpoint / "adapter.bin").write_bytes(b"weights")
    verified = verified_last_good_checkpoint_adapter(checkpoint)
    assert verified.sha256 == sha256_directory(checkpoint)

    with open(checkpoint / "adapter.bin", "ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="content changed"):
        verified.verify()


def test_training_regression_repair_generates_counterexamples_and_promotes(tmp_path):
    runner = _runner(tmp_path)
    parent = _parent_experiment()
    runner.engine.propose((parent,))
    parent_generation = runner.run_generation((parent,))
    assert parent_generation.promoted is None  # 0.65 < minimum 0.8
    good_checkpoint = parent_generation.candidates[0].artifact.artifact_ref

    bad_checkpoint = tmp_path / "checkpoint-bad"
    bad_checkpoint.mkdir()
    (bad_checkpoint / "adapter.bin").write_bytes(b"regressed-weights")

    outcome = run_training_regression_repair(
        runner=runner,
        parent_id="parent",
        cluster=_cluster(),
        good_checkpoint_dir=good_checkpoint,
        bad_checkpoint_dir=str(bad_checkpoint),
        provider=_provider(tmp_path),
        holdout_fingerprint_files=(_holdout(tmp_path),),
        variants=(
            RepairVariant("lr-low", 0.3, training_patch={"learning_rate": 5e-5}),
        ),
    )

    assert len(outcome.candidates) == 1
    candidate = outcome.candidates[0]
    assert candidate.result is not None
    assert outcome.promoted is not None
    assert outcome.promoted.metrics["quality"] == pytest.approx(0.86)
    assert runner.engine.baseline.experiment_id.startswith("repair-")

    proposed = runner.engine.graph.nodes[outcome.promoted.experiment_id]
    assert proposed.parent_id == "parent"
    assert proposed.config_patch["backend"]["parent_adapter"] == {
        "path": str(Path(good_checkpoint).resolve()),
        "sha256": sha256_directory(Path(good_checkpoint)),
    }
    assert proposed.config_patch["repair"]["continuation"] is True


def test_training_regression_repair_that_fails_to_improve_is_not_promoted(tmp_path):
    """The explicit failed-repair -> refused-promotion proof: even a repair
    population built entirely correctly (real counterexamples, real
    contamination audit, real continuation from the last-good checkpoint) is
    rejected -- not silently accepted -- when it does not clear the same real
    gate every other candidate is held to."""
    runner = _runner(tmp_path, evaluator=AlwaysRegressingEvaluator())
    parent = _parent_experiment()
    runner.engine.propose((parent,))
    parent_generation = runner.run_generation((parent,))
    good_checkpoint = parent_generation.candidates[0].artifact.artifact_ref

    bad_checkpoint = tmp_path / "checkpoint-bad"
    bad_checkpoint.mkdir()
    (bad_checkpoint / "adapter.bin").write_bytes(b"regressed-weights")

    baseline_before = runner.engine.baseline
    outcome = run_training_regression_repair(
        runner=runner,
        parent_id="parent",
        cluster=_cluster(),
        good_checkpoint_dir=good_checkpoint,
        bad_checkpoint_dir=str(bad_checkpoint),
        provider=_provider(tmp_path),
        holdout_fingerprint_files=(_holdout(tmp_path),),
        variants=(RepairVariant("default", 0.2),),
    )

    assert outcome.promoted is None
    assert outcome.ranking[0].decision.accepted is False
    assert runner.engine.baseline is baseline_before
    repair_candidate_id = outcome.candidates[0].experiment_id
    assert runner.engine.graph.nodes[repair_candidate_id].status.value == "rejected"


def test_fresh_start_repair_allows_lora_topology_change(tmp_path):
    runner = _runner(tmp_path)
    parent = _parent_experiment()
    runner.engine.propose((parent,))
    parent_generation = runner.run_generation((parent,))
    good_checkpoint = parent_generation.candidates[0].artifact.artifact_ref
    bad_checkpoint = tmp_path / "checkpoint-bad"
    bad_checkpoint.mkdir()
    (bad_checkpoint / "adapter.bin").write_bytes(b"regressed-weights")

    outcome = run_training_regression_repair(
        runner=runner,
        parent_id="parent",
        cluster=_cluster(),
        good_checkpoint_dir=good_checkpoint,
        bad_checkpoint_dir=str(bad_checkpoint),
        provider=_provider(tmp_path),
        holdout_fingerprint_files=(_holdout(tmp_path),),
        variants=(RepairVariant("fresh-rank8", 0.3, lora_patch={"r": 8}),),
        continue_from_last_good=False,
    )

    assert len(outcome.candidates) == 1
    candidate_id = outcome.candidates[0].experiment_id
    proposed = runner.engine.graph.nodes[candidate_id]
    assert "parent_adapter" not in proposed.config_patch["backend"]
    assert proposed.config_patch["backend"]["lora"] == {"r": 8}
    assert proposed.config_patch["repair"]["continuation"] is False


def test_continuation_repair_rejects_lora_topology_variant_before_provider(tmp_path):
    runner = _runner(tmp_path)
    parent = _parent_experiment()
    runner.engine.propose((parent,))
    parent_generation = runner.run_generation((parent,))
    good_checkpoint = parent_generation.candidates[0].artifact.artifact_ref

    class ProviderMustNotRun:
        name = "must-not-run"
        version = "1"

        def propose(self, request):
            raise AssertionError("provider must not run for invalid continuation topology")

    with pytest.raises(ValueError, match="cannot change LoRA topology"):
        run_training_regression_repair(
            runner=runner,
            parent_id="parent",
            cluster=_cluster(),
            good_checkpoint_dir=good_checkpoint,
            bad_checkpoint_dir=str(tmp_path / "checkpoint-bad-unused"),
            provider=ProviderMustNotRun(),
            holdout_fingerprint_files=(_holdout(tmp_path),),
            variants=(RepairVariant("rank-change", 0.2, lora_patch={"r": 8}),),
        )
    assert runner.engine.outstanding_candidates == 0
