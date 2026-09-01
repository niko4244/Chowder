from chowder.executors import TrainingArtifact
from chowder.models import Experiment, ExperimentResult, Hypothesis
from chowder.provenance import EvidenceManifest
from chowder.registry import RunRegistry


def test_registry_persists_result_manifest_and_lineage(tmp_path):
    db = tmp_path / "runs.db"
    root = Experiment("root", None, Hypothesis("o", "c", "i"), {}, 1)
    child = Experiment("child", "root", Hypothesis("o2", "c2", "i2"), {"lr": 1e-5}, 1)
    result = ExperimentResult("child", {"score": 2.0}, 0.8, "adapter://child")
    manifest = EvidenceManifest("child", "root", {"lr": 1e-5}, "abc", {"score": 2.0}, 42)

    with RunRegistry(db) as registry:
        registry.record_experiment(root)
        registry.record_experiment(child)
        registry.record_result(result)
        digest = registry.record_manifest(manifest)
        assert registry.lineage("child") == ("root",)
        assert list(registry.list_results()) == [result]
        assert len(digest) == 64


def test_registry_records_training_artifacts_before_evaluation(tmp_path):
    db = tmp_path / "runs.db"
    experiment = Experiment(
        "e1", None, Hypothesis("obs", "cause", "fix"), {}, 1.0
    )
    artifact = TrainingArtifact(
        "run-1", "e1", "./adapter", 0.25,
        telemetry={"train_loss": 0.4}, evidence={"dataset_sha256": "abc"}
    )
    with RunRegistry(db) as registry:
        registry.record_experiment(experiment)
        registry.record_training_artifact(artifact)
        loaded = tuple(registry.list_training_artifacts())
    assert loaded == (artifact,)
