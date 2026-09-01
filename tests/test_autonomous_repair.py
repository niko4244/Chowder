import json
from pathlib import Path

import pytest

from chowder.autonomous_repair import run_single_hop_autonomous_repair
from chowder.contamination import write_holdout_fingerprint_index
from chowder.cycle import ExperimentCycleRunner
from chowder.engine import EvolutionEngine
from chowder.executors import EvaluationOutcome, ExecutionContext, TrainingArtifact
from chowder.failures import FailureRecord, FailureSourceRole
from chowder.local_corpus_provider import LocalCorpusRepairProvider
from chowder.memory import HardwareProfile
from chowder.models import Experiment, ExperimentResult, Goal, Hypothesis, MetricTarget
from chowder.provenance import sha256_file
from chowder.repair_candidates import RepairVariant


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class FakeTrainer:
    name = "fake-trainer"

    def profile(self, experiment, context):
        raise NotImplementedError

    def run(self, experiment, context):
        artifact = Path(context.work_dir) / f"artifact-{experiment.experiment_id}"
        artifact.mkdir(exist_ok=False)
        (artifact / "adapter.bin").write_bytes(experiment.experiment_id.encode("utf-8"))
        return TrainingArtifact(
            run_id=f"train-{experiment.experiment_id}",
            experiment_id=experiment.experiment_id,
            artifact_ref=str(artifact),
            gpu_hours=0.2,
            evidence={"test": True},
        )

    def cancel(self, run_id):
        pass


class FakeEvaluator:
    name = "fake-eval"

    def __init__(self, holdout_path):
        self.holdout_path = Path(holdout_path).resolve()
        self.holdout_sha = sha256_file(self.holdout_path)

    def evaluate(self, *, experiment, artifact, context):
        is_repair = "repair" in experiment.tags
        quality = 0.86 if is_repair else 0.65
        return EvaluationOutcome(
            run_id=f"eval-{experiment.experiment_id}",
            experiment_id=experiment.experiment_id,
            source_artifact_ref=artifact.artifact_ref,
            metrics={"quality": quality},
            gpu_hours=0.05,
            evidence={
                "evaluator": "transformers-text",
                "protocol_sha256": "p" * 64,
                "holdout_fingerprint_sha256": {"reasoning": self.holdout_sha},
                "suite_evidence": {
                    "reasoning": {
                        "holdout_fingerprints_file": str(self.holdout_path),
                        "holdout_fingerprints_sha256": self.holdout_sha,
                    }
                },
            },
        )


def _harvest(outcome):
    if outcome.experiment_id.startswith("repair-"):
        return ()
    return (
        FailureRecord(
            failure_id="f" * 64,
            experiment_id=outcome.experiment_id,
            evaluation_run_id=outcome.run_id,
            evaluator="transformers-text",
            suite="reasoning",
            row_index=0,
            protocol_sha256="p" * 64,
            artifact_sha256="a" * 64,
            source_role=FailureSourceRole.GATE_HOLDOUT,
            prompt="hidden benchmark prompt",
            expected="hidden answer",
            prediction="wrong answer",
            score=0.0,
            failure_kind="answer_mismatch",
        ),
    )


def _runner(tmp_path):
    holdout_index = tmp_path / "holdout-index.jsonl"
    write_holdout_fingerprint_index(
        [("hidden benchmark prompt", "hidden answer")], holdout_index
    )
    engine = EvolutionEngine(
        Goal((MetricTarget("quality", minimum=0.8),), gpu_hour_budget=4.0, max_parallel_candidates=2),
        ExperimentResult("baseline", {"quality": 0.7}, 0.0),
    )
    context = ExecutionContext(
        HardwareProfile(16, 64, 500, 12, 40, 3),
        str(tmp_path),
        7,
    )
    return ExperimentCycleRunner(
        engine=engine,
        trainer=FakeTrainer(),
        evaluator=FakeEvaluator(holdout_index),
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
        failure_harvester=_harvest,
    )


def _provider(tmp_path):
    corpus = tmp_path / "repair-corpus.jsonl"
    _write_jsonl(
        corpus,
        [
            {
                "example_id": "independent-1",
                "suite": "reasoning",
                "strategy": "near_neighbor_reasoning",
                "prompt": "independent reasoning example one",
                "expected": "answer one",
            },
            {
                "example_id": "independent-2",
                "suite": "reasoning",
                "strategy": "near_neighbor_reasoning",
                "prompt": "independent reasoning example two",
                "expected": "answer two",
            },
        ],
    )
    return LocalCorpusRepairProvider([corpus], max_examples=2, examples_per_failure=2)


def _source_experiment():
    return Experiment(
        experiment_id="source",
        parent_id=None,
        hypothesis=Hypothesis("quality regression", "reasoning weakness", "candidate change"),
        config_patch={},
        estimated_gpu_hours=0.5,
    )


def test_single_hop_autonomous_repair_runs_rejected_candidate_to_promoted_repair(tmp_path):
    runner = _runner(tmp_path)
    source = _source_experiment()
    assert runner.engine.propose((source,)) == (source,)

    source_generation = runner.run_generation((source,))
    assert source_generation.promoted is None
    assert source_generation.candidates[0].repair_plans
    assert runner.engine.graph.nodes["source"].status.value == "rejected"

    outcome = run_single_hop_autonomous_repair(
        runner=runner,
        source_generation=source_generation,
        provider=_provider(tmp_path),
        variants=(
            RepairVariant("lr-low", 0.3, training_patch={"learning_rate": 5e-5}),
            RepairVariant("rank-low", 0.3, lora_patch={"r": 8}),
        ),
    )

    assert outcome.target.candidate.experiment_id == "source"
    assert outcome.target.plan.requires_independent_source
    assert len(outcome.population.proposed_candidates) == 2
    assert all(candidate.parent_id == "source" for candidate in outcome.population.proposed_candidates)
    assert outcome.promoted is not None
    assert outcome.promoted.metrics["quality"] == pytest.approx(0.86)
    assert runner.engine.baseline.experiment_id.startswith("repair-")
    assert runner.engine.outstanding_candidates == 0
    assert runner.engine.spent_gpu_hours == pytest.approx(0.75)


def test_repair_coordinator_rejects_non_rejected_source_candidate(tmp_path):
    runner = _runner(tmp_path)
    source = _source_experiment()
    runner.engine.propose((source,))

    class PassingEvaluator(FakeEvaluator):
        def evaluate(self, *, experiment, artifact, context):
            result = super().evaluate(experiment=experiment, artifact=artifact, context=context)
            return EvaluationOutcome(
                result.run_id,
                result.experiment_id,
                result.source_artifact_ref,
                {"quality": 0.9},
                result.gpu_hours,
                result.evidence,
            )

    runner.evaluator = PassingEvaluator(tmp_path / "holdout-index.jsonl")
    generation = runner.run_generation((source,))
    assert generation.promoted is not None

    with pytest.raises(ValueError, match="no independently repairable rejected candidate"):
        run_single_hop_autonomous_repair(
            runner=runner,
            source_generation=generation,
            provider=_provider(tmp_path),
            variants=(RepairVariant("default", 0.2),),
        )


def test_repair_coordinator_reverifies_holdout_index_before_source_provider(tmp_path):
    runner = _runner(tmp_path)
    source = _source_experiment()
    runner.engine.propose((source,))
    generation = runner.run_generation((source,))

    holdout = tmp_path / "holdout-index.jsonl"
    holdout.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="holdout fingerprint digest changed"):
        run_single_hop_autonomous_repair(
            runner=runner,
            source_generation=generation,
            provider=_provider(tmp_path),
            variants=(RepairVariant("default", 0.2),),
        )

    assert runner.engine.outstanding_candidates == 0
