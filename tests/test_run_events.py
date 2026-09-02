from chowder.run_events import (
    CheckpointEvent,
    EvaluationProgressEvent,
    FailureEvent,
    PromotionEvent,
    RepairEvent,
    RunEvent,
    TrainingProgressEvent,
    event_experiment_id,
    event_payload,
    event_type_name,
)


def test_run_event_defaults_and_shape():
    event = RunEvent(stage="prepare", message="loaded project 'x'")
    assert event.experiment_id is None
    assert event_type_name(event) == "RunEvent"
    assert event_experiment_id(event) is None
    assert event_payload(event) == {
        "stage": "prepare",
        "message": "loaded project 'x'",
        "experiment_id": None,
    }


def test_training_progress_event_optional_fields_default_to_none():
    event = TrainingProgressEvent(
        experiment_id="e1",
        step=5,
        max_steps=None,
        epoch=None,
        loss=None,
        learning_rate=None,
        wall_seconds=1.5,
    )
    assert event_experiment_id(event) == "e1"
    assert event_payload(event)["step"] == 5
    assert event_payload(event)["loss"] is None


def test_evaluation_progress_event_shape():
    event = EvaluationProgressEvent(
        experiment_id="e1", suite="quality", rows_scored=3, rows_total=10, wall_seconds=0.5
    )
    assert event_type_name(event) == "EvaluationProgressEvent"
    assert event_experiment_id(event) == "e1"


def test_checkpoint_event_defaults_resumed_to_false():
    event = CheckpointEvent(experiment_id="e1", checkpoint_dir="/ckpt/e1", step=50)
    assert event.resumed is False
    assert event_experiment_id(event) == "e1"


def test_repair_event_uses_target_experiment_id_not_experiment_id():
    """RepairEvent has no `experiment_id` field of its own -- the target
    being repaired is `target_experiment_id`. event_experiment_id() must
    still resolve it, since the registry keys persisted events off
    whichever field a given event type actually carries."""
    event = RepairEvent(target_experiment_id="e1", depth=1, failure_signature="sig")
    assert not hasattr(event, "experiment_id")
    assert event_experiment_id(event) == "e1"
    assert event.stop_reason is None
    assert event.stop_detail is None


def test_failure_event_shape():
    event = FailureEvent(experiment_id="e1", failure_count=3, repair_plan_count=1)
    assert event_experiment_id(event) == "e1"
    assert event_payload(event) == {
        "experiment_id": "e1",
        "failure_count": 3,
        "repair_plan_count": 1,
    }


def test_promotion_event_metrics_default_to_empty_dict():
    event = PromotionEvent(experiment_id="e1")
    assert dict(event.metrics) == {}
    event2 = PromotionEvent(experiment_id="e1", metrics={"quality": 0.9})
    assert event_payload(event2)["metrics"] == {"quality": 0.9}


def test_event_experiment_id_returns_none_when_neither_field_is_present():
    class _Bare:
        pass

    assert event_experiment_id(_Bare()) is None
