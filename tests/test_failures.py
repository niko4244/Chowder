import json

import pytest

from chowder.executors import EvaluationOutcome
from chowder.failures import (
    FailureRecord,
    FailureSourceRole,
    cluster_failures,
    harvest_transformers_text_failures,
    plan_repairs,
    write_direct_repair_dataset,
)


def _outcome(tmp_path):
    predictions = tmp_path / "predictions-reasoning.jsonl"
    rows = [
        {"prompt": "2+2?", "expected": "4", "prediction": "4", "score": 1.0},
        {"prompt": "3+5?", "expected": "8", "prediction": "", "score": 0.0},
        {"prompt": "9-2?", "expected": "7", "prediction": "6", "score": 0.0},
    ]
    predictions.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return EvaluationOutcome(
        run_id="eval-1",
        experiment_id="exp-1",
        source_artifact_ref=str(tmp_path / "adapter"),
        metrics={"reasoning": 1 / 3},
        gpu_hours=0.01,
        evidence={
            "evaluator": "transformers-text",
            "protocol_sha256": "a" * 64,
            "artifact_sha256": "b" * 64,
            "suite_evidence": {
                "reasoning": {
                    "predictions_file": str(predictions),
                    "source_role": "gate_holdout",
                }
            },
        },
    )


def test_harvester_records_failed_rows_only_and_is_deterministic(tmp_path):
    outcome = _outcome(tmp_path)
    first = harvest_transformers_text_failures(outcome)
    second = harvest_transformers_text_failures(outcome)
    assert len(first) == 2
    assert [item.failure_id for item in first] == [item.failure_id for item in second]
    assert first[0].failure_kind == "empty_prediction"
    assert first[1].failure_kind == "answer_mismatch"
    assert all(item.source_role is FailureSourceRole.GATE_HOLDOUT for item in first)
    assert all(not item.training_eligible for item in first)


def test_failure_clustering_separates_failure_kinds(tmp_path):
    failures = harvest_transformers_text_failures(_outcome(tmp_path))
    clusters = cluster_failures(failures)
    assert len(clusters) == 2
    assert {cluster.failure_kind for cluster in clusters} == {"empty_prediction", "answer_mismatch"}
    assert all(len(cluster.failure_ids) == 1 for cluster in clusters)


def test_repair_plans_for_holdout_require_independent_source(tmp_path):
    failures = harvest_transformers_text_failures(_outcome(tmp_path))
    plans = plan_repairs(cluster_failures(failures))
    assert plans
    assert all(plan.requires_independent_source for plan in plans)
    assert all(not plan.direct_training_allowed for plan in plans)
    assert all("never copy holdout" in plan.intervention for plan in plans)


def test_direct_dataset_refuses_gate_holdout_leakage(tmp_path):
    failures = harvest_transformers_text_failures(_outcome(tmp_path))
    with pytest.raises(ValueError, match="non-training-eligible"):
        write_direct_repair_dataset(failures, tmp_path / "repair.jsonl")


def test_direct_dataset_allows_explicit_repair_source(tmp_path):
    failure = FailureRecord(
        failure_id="f" * 64,
        experiment_id="exp",
        evaluation_run_id="repair-eval",
        evaluator="repair-source-eval",
        suite="independent-dev-corpus",
        row_index=0,
        protocol_sha256="a" * 64,
        artifact_sha256="b" * 64,
        source_role=FailureSourceRole.REPAIR_SOURCE,
        prompt="What is 2+3?",
        expected="5",
        prediction="4",
        score=0.0,
        failure_kind="answer_mismatch",
    )
    path = tmp_path / "repair.jsonl"
    digest = write_direct_repair_dataset([failure], path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert len(digest) == 64
    assert payload["failure_id"] == failure.failure_id
    assert payload["source_role"] == "repair_source"
    assert "Assistant: 5" in payload["text"]
