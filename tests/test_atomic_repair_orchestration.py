import json
from pathlib import Path

import pytest

from chowder.contamination import write_holdout_fingerprint_index
from chowder.engine import EvolutionEngine
from chowder.failures import FailureCluster, FailureSourceRole, RepairPlan
from chowder.local_corpus_provider import LocalCorpusRepairProvider
from chowder.models import Experiment, ExperimentResult, Goal, Hypothesis, MetricTarget
from chowder.registry import RegistryInvariantError, RunRegistry
from chowder.repair_candidates import RepairVariant
from chowder.repair_orchestrator import prepare_and_propose_repair_population
from chowder.repair_requests import build_repair_request


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _parent():
    return Experiment(
        experiment_id="parent",
        parent_id=None,
        hypothesis=Hypothesis("baseline failure", "reasoning weakness", "repair"),
        config_patch={
            "backend": {
                "base_model": "example/model",
                "dataset": "base.jsonl",
                "training": {"epochs": 2, "learning_rate": 1e-4},
                "lora": {"r": 16},
            },
            "evaluation": {
                "type": "transformers-text",
                "suites": [{"name": "quality", "dataset": "holdout.jsonl"}],
            },
        },
        estimated_gpu_hours=1.0,
    )


def _engine():
    engine = EvolutionEngine(
        Goal((MetricTarget("quality", minimum=0.8),), gpu_hour_budget=5.0, max_parallel_candidates=2),
        ExperimentResult("baseline", {"quality": 0.7}, 0.0),
    )
    engine.graph.add(_parent())
    return engine


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
        observation="reasoning failure",
        suspected_cause="answer selection",
        intervention="independent repair examples",
        source_failure_ids=("f" * 64,),
        direct_training_allowed=False,
        requires_independent_source=True,
    )


def _provider(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    _write_jsonl(
        corpus,
        [
            {
                "example_id": "ex-1",
                "suite": "reasoning",
                "strategy": "near_neighbor_reasoning",
                "prompt": "3+3?",
                "expected": "6",
            },
            {
                "example_id": "ex-2",
                "suite": "reasoning",
                "strategy": "near_neighbor_reasoning",
                "prompt": "4+4?",
                "expected": "8",
            },
        ],
    )
    return LocalCorpusRepairProvider([corpus], max_examples=2, examples_per_failure=2)


def _holdout(tmp_path):
    path = tmp_path / "holdout-index.jsonl"
    write_holdout_fingerprint_index([("2+2?", "4")], path)
    return path


def _variants():
    return (
        RepairVariant("lr-low", 0.2, training_patch={"learning_rate": 5e-5}),
        RepairVariant("rank-low", 0.3, lora_patch={"r": 8}),
    )


def test_registry_failure_withdraws_engine_proposals_and_removes_repair_artifacts(tmp_path):
    engine = _engine()
    with RunRegistry(tmp_path / "runs.db") as registry:
        # Deliberately do not persist parent. Registry lineage preflight fails
        # after engine proposal succeeds, exercising the cross-system rollback.
        with pytest.raises(RegistryInvariantError, match="unknown persisted parent"):
            prepare_and_propose_repair_population(
                engine=engine,
                parent_id="parent",
                plan=_plan(),
                request=build_repair_request(plan=_plan(), cluster=_cluster()),
                provider=_provider(tmp_path),
                holdout_fingerprint_files=(_holdout(tmp_path),),
                variants=_variants(),
                work_dir=tmp_path,
                registry=registry,
            )

        assert set(engine.graph.nodes) == {"parent"}
        assert engine.outstanding_candidates == 0
        assert engine.reserved_gpu_hours == 0
        assert engine.spent_gpu_hours == 0
        repairs_root = tmp_path / ".chowder" / "repairs"
        assert not repairs_root.exists() or not any(repairs_root.iterdir())


def test_successful_registry_batch_matches_engine_proposals(tmp_path):
    engine = _engine()
    with RunRegistry(tmp_path / "runs.db") as registry:
        registry.record_experiment(engine.graph.nodes["parent"])
        outcome = prepare_and_propose_repair_population(
            engine=engine,
            parent_id="parent",
            plan=_plan(),
            request=build_repair_request(plan=_plan(), cluster=_cluster()),
            provider=_provider(tmp_path),
            holdout_fingerprint_files=(_holdout(tmp_path),),
            variants=_variants(),
            work_dir=tmp_path,
            registry=registry,
        )

        assert len(outcome.proposed_candidates) == 2
        assert engine.outstanding_candidates == 2
        assert engine.reserved_gpu_hours == pytest.approx(0.5)
        for candidate in outcome.proposed_candidates:
            assert registry.has_experiment(candidate.experiment_id)
            assert registry.lineage(candidate.experiment_id) == ("parent",)
            assert candidate.experiment_id in engine.graph.nodes
        assert Path(outcome.repair_dir).is_dir()
