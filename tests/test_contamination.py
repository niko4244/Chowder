import json

import pytest

from chowder.contamination import (
    RepairExample,
    audit_repair_examples,
    write_holdout_fingerprint_index,
    write_verified_repair_dataset,
)
from chowder.failures import FailureRecord, FailureSourceRole
from chowder.repair_data import write_verified_failure_repair_dataset


def _index(tmp_path):
    path = tmp_path / "holdout.jsonl"
    digest = write_holdout_fingerprint_index(
        [("What is 2 + 2?", "4"), ("Capital of France?", "Paris")], path
    )
    return path, digest


def test_holdout_index_is_hash_only_and_deterministic(tmp_path):
    path, first_digest = _index(tmp_path)
    raw = path.read_text(encoding="utf-8")
    assert "What is 2 + 2?" not in raw
    assert "Paris" not in raw
    second = tmp_path / "holdout-2.jsonl"
    second_digest = write_holdout_fingerprint_index(
        [("Capital of France?", "Paris"), ("What is 2 + 2?", "4")], second
    )
    assert first_digest == second_digest
    rows = [json.loads(line) for line in raw.splitlines()]
    assert all(len(row["prompt_sha256"]) == 64 for row in rows)
    assert all(len(row["pair_sha256"]) == 64 for row in rows)


def test_audit_catches_normalized_prompt_overlap_even_with_changed_answer(tmp_path):
    path, _ = _index(tmp_path)
    audit = audit_repair_examples(
        [RepairExample("  WHAT   is 2 + 2? ", "five", "independent-source-1")],
        [path],
    )
    assert audit.clean is False
    assert len(audit.prompt_overlap_sha256) == 1
    assert len(audit.pair_overlap_sha256) == 0


def test_audit_allows_independent_near_neighbor_examples(tmp_path):
    path, _ = _index(tmp_path)
    examples = [RepairExample("What is 3 + 2?", "5", "independent-source-2")]
    audit = audit_repair_examples(examples, [path])
    assert audit.clean is True
    digest, written_audit = write_verified_repair_dataset(
        examples, [path], tmp_path / "repair.jsonl"
    )
    assert len(digest) == 64
    assert written_audit.clean is True
    payload = json.loads((tmp_path / "repair.jsonl").read_text(encoding="utf-8"))
    assert payload["source_id"] == "independent-source-2"


def test_verified_writer_refuses_contaminated_repair_example(tmp_path):
    path, _ = _index(tmp_path)
    with pytest.raises(ValueError, match="overlaps holdout"):
        write_verified_repair_dataset(
            [RepairExample("Capital of France?", "Paris", "copied")],
            [path],
            tmp_path / "repair.jsonl",
        )


def test_autonomous_failure_repair_writer_requires_eligible_and_clean_source(tmp_path):
    path, _ = _index(tmp_path)
    clean = FailureRecord(
        failure_id="c" * 64,
        experiment_id="repair-exp",
        evaluation_run_id="repair-eval",
        evaluator="repair-source-eval",
        suite="independent",
        row_index=0,
        protocol_sha256="a" * 64,
        artifact_sha256="b" * 64,
        source_role=FailureSourceRole.REPAIR_SOURCE,
        prompt="What is 3 + 3?",
        expected="6",
        prediction="7",
        score=0.0,
        failure_kind="answer_mismatch",
    )
    digest, audit = write_verified_failure_repair_dataset(
        [clean], [path], tmp_path / "verified-repair.jsonl"
    )
    assert len(digest) == 64
    assert audit.clean

    contaminated = FailureRecord(
        failure_id="d" * 64,
        experiment_id="repair-exp",
        evaluation_run_id="repair-eval",
        evaluator="repair-source-eval",
        suite="independent",
        row_index=1,
        protocol_sha256="a" * 64,
        artifact_sha256="b" * 64,
        source_role=FailureSourceRole.REPAIR_SOURCE,
        prompt="what is 2 + 2?",
        expected="4",
        prediction="5",
        score=0.0,
        failure_kind="answer_mismatch",
    )
    with pytest.raises(ValueError, match="overlaps holdout"):
        write_verified_failure_repair_dataset(
            [contaminated], [path], tmp_path / "blocked.jsonl"
        )
