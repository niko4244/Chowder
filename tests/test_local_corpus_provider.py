import hashlib
import json

import pytest

from chowder.failures import FailureCluster, FailureSourceRole, RepairPlan
from chowder.local_corpus_provider import LocalCorpusRepairProvider
from chowder.repair_requests import RepairStrategy, build_repair_request, request_repair_sources


def _cluster(kind="answer_mismatch"):
    return FailureCluster(
        cluster_id="c" * 64,
        evaluator="transformers-text",
        suite="reasoning",
        protocol_sha256="a" * 64,
        source_role=FailureSourceRole.GATE_HOLDOUT,
        failure_kind=kind,
        failure_ids=("f" * 64,),
    )


def _plan():
    return RepairPlan(
        plan_id="p" * 64,
        cluster_id="c" * 64,
        observation="failed reasoning",
        suspected_cause="reasoning weakness",
        intervention="independent near-neighbor examples",
        source_failure_ids=("f" * 64,),
        direct_training_allowed=False,
        requires_independent_source=True,
    )


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_local_corpus_provider_selects_by_sanitized_metadata_deterministically(tmp_path):
    corpus = tmp_path / "repair-corpus.jsonl"
    _write_jsonl(
        corpus,
        [
            {
                "example_id": "exact-high",
                "suite": "reasoning",
                "strategy": "near_neighbor_reasoning",
                "prompt": "What is 3+3?",
                "expected": "6",
                "priority": 10,
            },
            {
                "example_id": "wildcard",
                "suite": "*",
                "strategy": "near_neighbor_reasoning",
                "prompt": "What is 4+4?",
                "expected": "8",
            },
            {
                "example_id": "wrong-suite",
                "suite": "coding",
                "strategy": "near_neighbor_reasoning",
                "prompt": "Write a loop",
                "expected": "for i in range(3): pass",
                "priority": 100,
            },
            {
                "example_id": "wrong-strategy",
                "suite": "reasoning",
                "strategy": "format_control",
                "prompt": "Answer briefly",
                "expected": "yes",
                "priority": 100,
            },
        ],
    )
    request = build_repair_request(plan=_plan(), cluster=_cluster())
    assert request.strategy is RepairStrategy.NEAR_NEIGHBOR_REASONING
    provider = LocalCorpusRepairProvider([corpus], max_examples=4, examples_per_failure=2)

    first = request_repair_sources(provider=provider, request=request)
    second = request_repair_sources(provider=provider, request=request)

    assert first == second
    assert len(first.examples) == 2
    assert [example.prompt for example in first.examples] == ["What is 3+3?", "What is 4+4?"]
    assert len(first.sources) == 1
    expected_sha = hashlib.sha256(corpus.read_bytes()).hexdigest()
    assert first.sources[0].content_sha256 == expected_sha
    assert expected_sha in first.sources[0].ref


def test_local_corpus_provider_honors_failure_kind_filter(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    _write_jsonl(
        corpus,
        [
            {
                "example_id": "refusal-only",
                "suite": "reasoning",
                "strategy": "near_neighbor_reasoning",
                "failure_kind": "refusal_or_unknown",
                "prompt": "Independent question",
                "expected": "Independent answer",
            }
        ],
    )
    provider = LocalCorpusRepairProvider([corpus])
    request = build_repair_request(plan=_plan(), cluster=_cluster("answer_mismatch"))
    with pytest.raises(ValueError, match="enough eligible examples"):
        request_repair_sources(provider=provider, request=request)


def test_local_corpus_provider_rejects_duplicate_row_ids(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    _write_jsonl(
        corpus,
        [
            {"example_id": "dup", "prompt": "A", "expected": "B"},
            {"example_id": "dup", "prompt": "C", "expected": "D"},
        ],
    )
    provider = LocalCorpusRepairProvider([corpus])
    request = build_repair_request(plan=_plan(), cluster=_cluster())
    with pytest.raises(ValueError, match="duplicate example_id"):
        request_repair_sources(provider=provider, request=request)
