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
from chowder.provenance import sha256_directory, sha256_file
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
        (artifact / "adapter_config.json").write_text(
            '{"peft_type":"LORA"}\n', encoding="utf-8"
        )
        (artifact / "adapter.bin").write_bytes(
            experiment.experiment_id.encode("utf-8")
        )
        backend = context.resolved_config["backend"]
        dataset = Path(str(backend["dataset"]))
        if not dataset.is_absolute():
            dataset = Path(context.work_dir) / dataset
        dataset = dataset.resolve()
        replay_sha = None
        replay = backend.get("replay")
        if isinstance(replay, dict) and isinstance(replay.get("dataset"), str):
            replay_path = Path(replay["dataset"])
            if not replay_path.is_absolute():
                replay_path = Path(context.work_dir) / replay_path
            replay_sha = sha256_file(replay_path.resolve())
        return TrainingArtifact(
            run_id=f"train-{experiment.experiment_id}",
            experiment_id=experiment.experiment_id,
            artifact_ref=str(artifact),
            gpu_hours=0.2,
            evidence={
                "test": True,
                "dataset_sha256": sha256_file(dataset),
                "replay_dataset_sha256": replay_sha,
                "artifact_sha256": sha256_directory(artifact),
            },
        )

    def cancel(self, run_id):
        pass


class FakeEvaluator:
    name = "fake-eval"

    def __init__(self, holdout_path):
        self.holdout_path = Path(holdout_path).resolve()
        self.holdout_sha = sha256_file(self.holdout_path)

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
    _write_jsonl(
        tmp_path / "base.jsonl",
        [
            {"text": "original training example one"},
            {"text": "original training example two"},
            {"text": "original training example three"},
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
    return LocalCorpusRepairProvider(
        [corpus], max_examples=2, examples_per_failure=2
    )


def _source_experiment():
    return Experiment(
        experiment_id="source",
        parent_id=None,
        hypothesis=Hypothesis(
            "quality regression", "reasoning weakness", "candidate change"
        ),
        config_patch={},
        estimated_gpu_hours=0.5,
    )


def test_single_hop_autonomous_repair_runs_rejected_candidate_to_promoted_repair(
    tmp_path,
):
    runner = _runner(tmp_path)
    source = _source_experiment()
    assert runner.engine.propose((source,)) == (source,)

    source_generation = runner.run_generation((source,))
    assert source_generation.promoted is None
    assert source_generation.candidates[0].repair_plans
    assert runner.engine.graph.nodes["source"].status.value == "rejected"

    source_artifact = source_generation.candidates[0].artifact
    assert source_artifact is not None
    source_adapter_sha = source_artifact.evidence["artifact_sha256"]
    base_sha = sha256_file(tmp_path / "base.jsonl")
    outcome = run_single_hop_autonomous_repair(
        runner=runner,
        source_generation=source_generation,
        provider=_provider(tmp_path),
        variants=(
            RepairVariant(
                "lr-low", 0.3, training_patch={"learning_rate": 5e-5}
            ),
            RepairVariant("epochs-low", 0.3, training_patch={"epochs": 1}),
        ),
    )

    assert outcome.target.candidate.experiment_id == "source"
    assert outcome.target.plan.requires_independent_source
    assert len(outcome.population.proposed_candidates) == 2
    assert all(
        candidate.parent_id == "source"
        for candidate in outcome.population.proposed_candidates
    )
    for candidate in outcome.population.proposed_candidates:
        backend = candidate.config_patch["backend"]
        replay = backend["replay"]
        replay_path = Path(replay["dataset"])
        assert replay_path.parent == (tmp_path / ".chowder" / "replay-history").resolve()
        assert replay["sha256"] == sha256_file(replay_path)
        assert replay["ratio"] == pytest.approx(1.0)
        assert Path(replay["manifest"]).is_file()
        assert replay["manifest_sha256"] == sha256_file(replay["manifest"])
        assert backend["text_field"] == "text"
        replay_rows = [json.loads(line)["text"] for line in replay_path.read_text().splitlines()]
        assert replay_rows == [
            "original training example one",
            "original training example two",
            "original training example three",
        ]
        manifest = json.loads(Path(replay["manifest"]).read_text())
        assert manifest["sources"] == [
            {
                "role": "parent_primary_training",
                "row_count": 3,
                "source_sha256": base_sha,
                "text_field": "text",
            }
        ]
        assert backend["dataset_sha256"] == candidate.config_patch["repair"][
            "repair_dataset_sha256"
        ]
        assert backend["parent_adapter"] == {
            "path": str(Path(source_artifact.artifact_ref).resolve()),
            "sha256": source_adapter_sha,
        }
        assert candidate.config_patch["repair"]["continuation"] is True
        assert candidate.config_patch["repair"]["replay_manifest_sha256"] == replay[
            "manifest_sha256"
        ]
    assert outcome.promoted is not None
    assert outcome.promoted.metrics["quality"] == pytest.approx(0.86)
    assert runner.engine.baseline.experiment_id.startswith("repair-")
    assert runner.engine.outstanding_candidates == 0
    assert runner.engine.spent_gpu_hours == pytest.approx(0.75)


def test_repair_coordinator_rejects_lora_topology_variant_before_provider(tmp_path):
    runner = _runner(tmp_path)
    source = _source_experiment()
    runner.engine.propose((source,))
    generation = runner.run_generation((source,))

    class ProviderMustNotRun:
        name = "must-not-run"
        version = "1"

        def propose(self, request):
            raise AssertionError("provider must not run for invalid continuation topology")

    with pytest.raises(ValueError, match="cannot change LoRA topology"):
        run_single_hop_autonomous_repair(
            runner=runner,
            source_generation=generation,
            provider=ProviderMustNotRun(),
            variants=(RepairVariant("rank-change", 0.2, lora_patch={"r": 8}),),
        )
    assert runner.engine.outstanding_candidates == 0


def test_continue_from_parent_false_allows_lora_topology_change_with_no_parent_adapter(
    tmp_path,
):
    runner = _runner(tmp_path)
    source = _source_experiment()
    runner.engine.propose((source,))
    source_generation = runner.run_generation((source,))
    assert source_generation.promoted is None

    outcome = run_single_hop_autonomous_repair(
        runner=runner,
        source_generation=source_generation,
        provider=_provider(tmp_path),
        variants=(RepairVariant("fresh-start-rank8", 0.3, lora_patch={"r": 8}),),
        continue_from_parent=False,
    )

    assert len(outcome.population.proposed_candidates) == 1
    candidate = outcome.population.proposed_candidates[0]
    backend = candidate.config_patch["backend"]
    assert backend["lora"] == {"r": 8}
    assert "parent_adapter" not in backend
    assert candidate.config_patch["repair"]["continuation"] is False
    assert candidate.config_patch["repair"]["parent_adapter_sha256"] is None
    # replay (data rehearsal) is orthogonal to adapter continuation -- a
    # fresh-start repair still gets the parent's training history rehearsed
    assert "replay" in backend
    assert outcome.promoted is not None


def test_repair_coordinator_rejects_non_rejected_source_candidate(tmp_path):
    runner = _runner(tmp_path)
    source = _source_experiment()
    runner.engine.propose((source,))

    class PassingEvaluator(FakeEvaluator):
        def evaluate(self, *, experiment, artifact, context):
            result = super().evaluate(
                experiment=experiment, artifact=artifact, context=context
            )
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

    with pytest.raises(
        ValueError, match="no independently repairable rejected candidate"
    ):
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


def test_repair_coordinator_rejects_tampered_parent_replay_before_provider(tmp_path):
    runner = _runner(tmp_path)
    source = _source_experiment()
    runner.engine.propose((source,))
    generation = runner.run_generation((source,))

    with open(tmp_path / "base.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"text": "post-training mutation"}) + "\n")

    class ProviderMustNotRun:
        name = "must-not-run"
        version = "1"

        def propose(self, request):
            raise AssertionError("provider must not see a repair request after replay tamper")

    with pytest.raises(ValueError, match="replay history source content changed"):
        run_single_hop_autonomous_repair(
            runner=runner,
            source_generation=generation,
            provider=ProviderMustNotRun(),
            variants=(RepairVariant("default", 0.2),),
        )
    assert runner.engine.outstanding_candidates == 0


def test_repair_coordinator_rejects_tampered_parent_adapter_before_provider(tmp_path):
    runner = _runner(tmp_path)
    source = _source_experiment()
    runner.engine.propose((source,))
    generation = runner.run_generation((source,))
    artifact = generation.candidates[0].artifact
    assert artifact is not None
    with open(Path(artifact.artifact_ref) / "adapter.bin", "ab") as handle:
        handle.write(b"tampered")

    class ProviderMustNotRun:
        name = "must-not-run"
        version = "1"

        def propose(self, request):
            raise AssertionError("provider must not run after adapter tamper")

    with pytest.raises(ValueError, match="parent adapter content changed"):
        run_single_hop_autonomous_repair(
            runner=runner,
            source_generation=generation,
            provider=ProviderMustNotRun(),
            variants=(RepairVariant("default", 0.2),),
        )
    assert runner.engine.outstanding_candidates == 0


def test_repair_coordinator_can_explicitly_disable_replay(tmp_path):
    runner = _runner(tmp_path)
    source = _source_experiment()
    runner.engine.propose((source,))
    generation = runner.run_generation((source,))
    outcome = run_single_hop_autonomous_repair(
        runner=runner,
        source_generation=generation,
        provider=_provider(tmp_path),
        variants=(RepairVariant("default", 0.2),),
        replay_ratio=None,
    )
    candidate = outcome.population.proposed_candidates[0]
    assert "replay" not in candidate.config_patch["backend"]
    assert "parent_adapter" in candidate.config_patch["backend"]
