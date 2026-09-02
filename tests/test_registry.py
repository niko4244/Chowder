from chowder.executors import TrainingArtifact
from chowder.models import Experiment, ExperimentResult, Hypothesis
from chowder.provenance import EvidenceManifest
from chowder.registry import RunRegistry
from chowder.run_events import PromotionEvent, RepairEvent, RunEvent


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


def test_registry_persists_run_events_in_order(tmp_path):
    db = tmp_path / "runs.db"
    with RunRegistry(db) as registry:
        registry.record_event(RunEvent(stage="prepare", message="loaded"))
        registry.record_event(
            RepairEvent(target_experiment_id="e1", depth=1, failure_signature="sig")
        )
        registry.record_event(
            PromotionEvent(experiment_id="e1-repair-1", metrics={"quality": 0.9})
        )
        events = list(registry.list_events())

    assert [e.event_type for e in events] == ["RunEvent", "RepairEvent", "PromotionEvent"]
    assert events[0].experiment_id is None
    assert events[0].payload == {"stage": "prepare", "message": "loaded", "experiment_id": None}
    assert events[1].experiment_id == "e1"
    assert events[1].payload["failure_signature"] == "sig"
    assert events[2].experiment_id == "e1-repair-1"
    assert events[2].payload["metrics"] == {"quality": 0.9}
    # event_id increases monotonically with insertion order
    assert [e.event_id for e in events] == sorted(e.event_id for e in events)


def test_registry_filters_events_by_experiment_id(tmp_path):
    db = tmp_path / "runs.db"
    with RunRegistry(db) as registry:
        registry.record_event(RunEvent(stage="train", message="a", experiment_id="e1"))
        registry.record_event(RunEvent(stage="train", message="b", experiment_id="e2"))
        registry.record_event(RunEvent(stage="train", message="c", experiment_id="e1"))
        e1_events = list(registry.list_events(experiment_id="e1"))

    assert [e.payload["message"] for e in e1_events] == ["a", "c"]
