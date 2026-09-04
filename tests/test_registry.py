import pytest

from chowder.combined_mechanism_experiment import CombinedMechanismExperiment
from chowder.executors import TrainingArtifact
from chowder.models import Experiment, ExperimentResult, ExperimentStatus, Hypothesis
from chowder.provenance import EvidenceManifest
from chowder.registry import RegistryInvariantError, RunRegistry
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


def test_registry_list_experiments_round_trips_config_and_lineage(tmp_path):
    db = tmp_path / "runs.db"
    root = Experiment(
        "root", None, Hypothesis("o", "c", "i"),
        {"backend": {"training": {"learning_rate": 1e-3}}}, 1.0,
    )
    child = Experiment(
        "child", "root", Hypothesis("o2", "c2", "i2"),
        {"backend": {"lora": {"r": 8}}}, 2.0,
        status=ExperimentStatus.PASSED,
    )
    with RunRegistry(db) as registry:
        registry.record_experiments((root, child))
        listed = list(registry.list_experiments())

    assert [e.experiment_id for e in listed] == ["root", "child"]
    assert listed[0].parent_id is None
    assert listed[0].config_patch == {"backend": {"training": {"learning_rate": 1e-3}}}
    assert listed[0].status == ExperimentStatus.PLANNED
    assert listed[1].parent_id == "root"
    assert listed[1].config_patch == {"backend": {"lora": {"r": 8}}}
    assert listed[1].status == ExperimentStatus.PASSED
    assert listed[1].hypothesis == Hypothesis("o2", "c2", "i2")
    # tags are not part of the persisted schema and always round-trip empty
    assert listed[1].tags == ()


def _combined_mechanism_experiment(**overrides) -> CombinedMechanismExperiment:
    defaults = dict(
        experiment_key="combo-key",
        mechanisms=("activation_offload", "frozen_layer_streaming"),
        baseline_peak_vram_gb=1.0,
        predicted_combined_peak_vram_gb=0.5,
        actual_combined_peak_vram_gb=0.6,
        prediction_error_gb=-0.1,
        baseline_wall_seconds=1.0,
        combined_wall_seconds=1.3,
        wall_time_penalty_ratio=1.3,
        per_mechanism_predicted_savings_gb={"activation_offload": 0.3, "frozen_layer_streaming": 0.2},
        forward_seconds=0.4,
        backward_seconds=0.2,
        optimizer_seconds=0.05,
        avg_gpu_utilization_percent=12.5,
        optimizer_state_bytes=4096.0,
        frozen_layer_streaming_bytes_transferred=3072.0,
        activation_offload_bytes_transferred=12345.0,
    )
    defaults.update(overrides)
    return CombinedMechanismExperiment(**defaults)


def test_registry_round_trips_combined_mechanism_experiment(tmp_path):
    db = tmp_path / "runs.db"
    experiment = _combined_mechanism_experiment()
    with RunRegistry(db) as registry:
        registry.record_combined_mechanism_experiment(experiment)
        listed = list(registry.list_combined_mechanism_experiments())
    assert listed == [experiment]


def test_registry_combined_mechanism_experiment_replay_is_idempotent(tmp_path):
    db = tmp_path / "runs.db"
    experiment = _combined_mechanism_experiment()
    with RunRegistry(db) as registry:
        registry.record_combined_mechanism_experiment(experiment)
        registry.record_combined_mechanism_experiment(experiment)
        listed = list(registry.list_combined_mechanism_experiments())
    assert len(listed) == 1


def test_registry_rejects_divergent_combined_mechanism_experiment_replay(tmp_path):
    db = tmp_path / "runs.db"
    first = _combined_mechanism_experiment()
    second = _combined_mechanism_experiment(actual_combined_peak_vram_gb=0.99)
    with RunRegistry(db) as registry:
        registry.record_combined_mechanism_experiment(first)
        with pytest.raises(RegistryInvariantError, match="different content"):
            registry.record_combined_mechanism_experiment(second)
