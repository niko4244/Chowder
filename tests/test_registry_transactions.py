import sqlite3

import pytest

from chowder.executors import EvaluationOutcome
from chowder.failures import FailureRecord, FailureSourceRole
from chowder.models import Experiment, Hypothesis
from chowder.registry import RunRegistry


def _experiment(name, parent=None):
    return Experiment(
        experiment_id=name,
        parent_id=parent,
        hypothesis=Hypothesis("obs", "cause", "fix"),
        config_patch={},
        estimated_gpu_hours=0.2,
    )


def _failure(failure_id, *, experiment_id="e1", evaluation_run_id="eval-1"):
    return FailureRecord(
        failure_id=failure_id,
        experiment_id=experiment_id,
        evaluation_run_id=evaluation_run_id,
        evaluator="transformers-text",
        suite="quality",
        row_index=0,
        protocol_sha256="p" * 64,
        artifact_sha256="a" * 64,
        source_role=FailureSourceRole.GATE_HOLDOUT,
        prompt="prompt",
        expected="expected",
        prediction="wrong",
        score=0.0,
        failure_kind="answer_mismatch",
    )


def test_record_experiments_rolls_back_complete_batch_on_foreign_key_failure(tmp_path):
    with RunRegistry(tmp_path / "runs.db") as registry:
        root = _experiment("root")
        registry.record_experiment(root)
        valid_child = _experiment("valid-child", "root")
        invalid_child = _experiment("invalid-child", "missing")

        with pytest.raises(sqlite3.IntegrityError):
            registry.record_experiments((valid_child, invalid_child))

        assert registry.has_experiment("root")
        assert not registry.has_experiment("valid-child")
        assert not registry.has_experiment("invalid-child")


def test_record_experiments_supports_parent_child_in_same_transaction(tmp_path):
    with RunRegistry(tmp_path / "runs.db") as registry:
        root = _experiment("root")
        child = _experiment("child", "root")
        grandchild = _experiment("grandchild", "child")
        registry.record_experiments((root, child, grandchild))

        assert registry.has_experiment("root")
        assert registry.has_experiment("child")
        assert registry.has_experiment("grandchild")
        assert registry.lineage("grandchild") == ("child", "root")


def test_record_failures_rolls_back_complete_batch_on_foreign_key_failure(tmp_path):
    with RunRegistry(tmp_path / "runs.db") as registry:
        experiment = _experiment("e1")
        registry.record_experiment(experiment)
        registry.record_evaluation_outcome(
            EvaluationOutcome(
                run_id="eval-1",
                experiment_id="e1",
                source_artifact_ref="adapter://e1",
                metrics={"quality": 0.5},
                gpu_hours=0.0,
                evidence={},
            )
        )
        valid = _failure("v" * 64)
        invalid = _failure("i" * 64, evaluation_run_id="missing-eval")

        with pytest.raises(sqlite3.IntegrityError):
            registry.record_failures((valid, invalid))

        assert tuple(registry.list_failures()) == ()
