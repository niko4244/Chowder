import json

from chowder.autonomous_repair import RepairTarget, _verified_parent_replay
from chowder.cycle import CandidateCycleOutcome, ExperimentCycleRunner
from chowder.engine import EvolutionEngine
from chowder.executors import ExecutionContext, TrainingArtifact
from chowder.failures import FailureCluster, FailureSourceRole, RepairPlan
from chowder.memory import HardwareProfile
from chowder.models import Experiment, ExperimentResult, Goal, Hypothesis, MetricTarget
from chowder.provenance import sha256_file
from chowder.replay_history import ReplayHistorySource, materialize_replay_history


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _cluster():
    return FailureCluster(
        cluster_id="c" * 64,
        evaluator="transformers-text",
        suite="reasoning",
        protocol_sha256="p" * 64,
        source_role=FailureSourceRole.GATE_HOLDOUT,
        failure_kind="answer_mismatch",
        failure_ids=("f" * 64,),
    )


def _plan():
    return RepairPlan(
        plan_id="r" * 64,
        cluster_id="c" * 64,
        observation="still failing",
        suspected_cause="remaining weakness",
        intervention="next independent repair",
        source_failure_ids=("f" * 64,),
        direct_training_allowed=False,
        requires_independent_source=True,
    )


def test_second_hop_replay_contains_prior_history_plus_parent_repair_data(tmp_path):
    root = tmp_path / "root.jsonl"
    _jsonl(root, [{"text": "root one"}, {"text": "root two"}])
    prior = materialize_replay_history(
        sources=(
            ReplayHistorySource(
                str(root), sha256_file(root), role="parent_primary_training"
            ),
        ),
        work_dir=tmp_path,
        ratio=1.0,
    )

    repair_a = tmp_path / "repair-a.jsonl"
    _jsonl(
        repair_a,
        [
            {"text": "repair a one"},
            {"text": "root two"},
        ],
    )
    repair_sha = sha256_file(repair_a)

    experiment = Experiment(
        experiment_id="repair-a",
        parent_id=None,
        hypothesis=Hypothesis("obs", "cause", "repair"),
        config_patch={
            "backend": {
                "dataset": str(repair_a),
                "dataset_sha256": repair_sha,
                "text_field": "text",
                "replay": {
                    "dataset": prior.path,
                    "sha256": prior.sha256,
                    "ratio": 1.0,
                    "manifest": prior.manifest_path,
                    "manifest_sha256": prior.manifest_sha256,
                },
            }
        },
        estimated_gpu_hours=0.2,
    )
    engine = EvolutionEngine(
        Goal((MetricTarget("quality", minimum=0.8),), gpu_hour_budget=2.0),
        ExperimentResult("baseline", {"quality": 0.7}, 0.0),
    )
    engine.graph.add(experiment)
    runner = ExperimentCycleRunner(
        engine=engine,
        trainer=object(),
        evaluator=object(),
        context=ExecutionContext(
            HardwareProfile(16, 64, 500, 12, 40, 3), str(tmp_path), 9
        ),
    )
    artifact = TrainingArtifact(
        run_id="train-repair-a",
        experiment_id="repair-a",
        artifact_ref=str(tmp_path / "unused-adapter"),
        gpu_hours=0.1,
        evidence={
            "dataset_sha256": repair_sha,
            "replay_dataset_sha256": prior.sha256,
        },
    )
    target = RepairTarget(
        candidate=CandidateCycleOutcome(
            experiment_id="repair-a", artifact=artifact
        ),
        cluster=_cluster(),
        plan=_plan(),
    )

    cumulative = _verified_parent_replay(
        runner=runner,
        target=target,
        ratio=1.0,
    )
    texts = [
        json.loads(line)["text"]
        for line in open(cumulative.path, encoding="utf-8")
    ]
    assert texts == ["root one", "root two", "repair a one"]
    manifest = json.load(open(cumulative.manifest_path, encoding="utf-8"))
    assert [source["role"] for source in manifest["sources"]] == [
        "prior_replay_history",
        "parent_primary_training",
    ]
    assert [source["source_sha256"] for source in manifest["sources"]] == [
        prior.sha256,
        repair_sha,
    ]
