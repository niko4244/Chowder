import json
from pathlib import Path

import pytest

from chowder.contamination import write_holdout_fingerprint_index
from chowder.engine import EvolutionEngine
from chowder.failures import FailureCluster, FailureSourceRole, RepairPlan
from chowder.local_corpus_provider import LocalCorpusRepairProvider
from chowder.models import Experiment, ExperimentResult, Goal, Hypothesis, MetricTarget
from chowder.provenance import sha256_file
from chowder.repair_candidates import RepairVariant, VerifiedReplayDataset
from chowder.repair_orchestrator import prepare_and_propose_repair_population


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _cluster():
    return FailureCluster(
        cluster_id="c" * 64,
        evaluator="transformers-text",
        suite="reasoning",
        protocol_sha256="a" * 64,
        source_role=FailureSourceRole.GATE_HOLDOUT,
        failure_kind="answer_mismatch",
        failure_ids=("f" * 64,),
    )


def _plan():
    return RepairPlan(
        plan_id="p" * 64,
        cluster_id="c" * 64,
        observation="reasoning failures",
        suspected_cause="answer-selection weakness",
        intervention="add independent near-neighbor examples",
        source_failure_ids=("f" * 64,),
        direct_training_allowed=False,
        requires_independent_source=True,
    )


def _engine(max_parallel=2, budget=5.0):
    engine = EvolutionEngine(
        Goal(
            (MetricTarget("quality", minimum=0.8),),
            gpu_hour_budget=budget,
            max_parallel_candidates=max_parallel,
        ),
        ExperimentResult("baseline", {"quality": 0.7}, 0.0),
    )
    parent = Experiment(
        experiment_id="parent",
        parent_id=None,
        hypothesis=Hypothesis("candidate failed", "reasoning weakness", "repair"),
        config_patch={
            "backend": {
                "base_model": "example/model",
                "dataset": "base.jsonl",
                "training": {"epochs": 2, "learning_rate": 1e-4},
                "lora": {"r": 16, "alpha": 32},
            },
            "evaluation": {
                "type": "transformers-text",
                "suites": [{"name": "quality", "dataset": "holdout.jsonl"}],
            },
        },
        estimated_gpu_hours=1.0,
    )
    engine.graph.add(parent)
    return engine


def _provider(tmp_path, *, contaminated=False):
    corpus = tmp_path / "repair-corpus.jsonl"
    prompt, expected = (("2+2?", "4") if contaminated else ("3+3?", "6"))
    _write_jsonl(
        corpus,
        [
            {
                "example_id": "ex-1",
                "suite": "reasoning",
                "strategy": "near_neighbor_reasoning",
                "prompt": prompt,
                "expected": expected,
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
    return LocalCorpusRepairProvider(
        [corpus], max_examples=2, examples_per_failure=2
    )


def _holdout(tmp_path):
    path = tmp_path / "holdout-index.jsonl"
    write_holdout_fingerprint_index([("2+2?", "4")], path)
    return path


def _replay(tmp_path, ratio=1.0):
    path = tmp_path / "base.jsonl"
    _write_jsonl(
        path,
        [
            {"text": "parent one"},
            {"text": "parent two"},
            {"text": "parent three"},
        ],
    )
    return VerifiedReplayDataset(str(path), sha256_file(path), ratio)


def test_orchestrator_materializes_verified_data_and_reserves_budgeted_population(tmp_path):
    engine = _engine(max_parallel=2)
    variants = (
        RepairVariant("lr-low", 0.2, training_patch={"learning_rate": 5e-5}),
        RepairVariant("rank-low", 0.3, lora_patch={"r": 8}),
        RepairVariant("epochs-low", 0.4, training_patch={"epochs": 1}),
    )
    outcome = prepare_and_propose_repair_population(
        engine=engine,
        parent_id="parent",
        plan=_plan(),
        cluster=_cluster(),
        provider=_provider(tmp_path),
        holdout_fingerprint_files=(_holdout(tmp_path),),
        variants=variants,
        work_dir=tmp_path,
    )

    assert len(outcome.generated_candidates) == 3
    assert len(outcome.proposed_candidates) == 2
    assert len(outcome.deferred_candidates) == 1
    assert engine.outstanding_candidates == 2
    assert engine.reserved_gpu_hours == pytest.approx(0.5)
    assert Path(outcome.materialized.dataset_path).is_file()
    assert Path(outcome.materialized.source_manifest_path).is_file()
    assert outcome.materialized.contamination_audit.clean

    for candidate in outcome.proposed_candidates:
        assert candidate.parent_id == "parent"
        assert "evaluation" not in candidate.config_patch
        resolved = engine.resolve_config(candidate.experiment_id)
        assert resolved["backend"]["base_model"] == "example/model"
        assert resolved["evaluation"]["type"] == "transformers-text"


def test_replay_adjusts_reserved_compute_upper_bound(tmp_path):
    engine = _engine(max_parallel=2, budget=2.0)
    replay = _replay(tmp_path, ratio=1.0)
    outcome = prepare_and_propose_repair_population(
        engine=engine,
        parent_id="parent",
        plan=_plan(),
        cluster=_cluster(),
        provider=_provider(tmp_path),
        holdout_fingerprint_files=(_holdout(tmp_path),),
        variants=(RepairVariant("default", 0.25),),
        work_dir=tmp_path,
        replay=replay,
    )
    assert len(outcome.proposed_candidates) == 1
    candidate = outcome.proposed_candidates[0]
    assert candidate.estimated_gpu_hours == pytest.approx(0.5)
    assert engine.reserved_gpu_hours == pytest.approx(0.5)
    assert candidate.config_patch["repair"]["repair_only_estimated_gpu_hours"] == pytest.approx(0.25)
    assert candidate.config_patch["repair"]["replay_adjusted_estimated_gpu_hours"] == pytest.approx(0.5)


def test_contaminated_provider_is_rejected_before_engine_mutation_and_retry_dir_is_cleaned(tmp_path):
    engine = _engine()
    with pytest.raises(ValueError, match="overlaps holdout"):
        prepare_and_propose_repair_population(
            engine=engine,
            parent_id="parent",
            plan=_plan(),
            cluster=_cluster(),
            provider=_provider(tmp_path, contaminated=True),
            holdout_fingerprint_files=(_holdout(tmp_path),),
            variants=(RepairVariant("default", 0.2),),
            work_dir=tmp_path,
        )

    assert engine.outstanding_candidates == 0
    assert engine.reserved_gpu_hours == 0
    assert set(engine.graph.nodes) == {"parent"}
    repair_root = tmp_path / ".chowder" / "repairs"
    assert not repair_root.exists() or not any(repair_root.iterdir())


def test_orchestrator_refuses_work_when_no_variant_fits_remaining_budget(tmp_path):
    engine = _engine(budget=0.1)
    with pytest.raises(ValueError, match="fits the remaining GPU-hour budget"):
        prepare_and_propose_repair_population(
            engine=engine,
            parent_id="parent",
            plan=_plan(),
            cluster=_cluster(),
            provider=_provider(tmp_path),
            holdout_fingerprint_files=(_holdout(tmp_path),),
            variants=(RepairVariant("too-large", 0.2),),
            work_dir=tmp_path,
        )
    assert not (tmp_path / ".chowder" / "repairs").exists()


def test_orchestrator_refuses_variant_that_only_fits_without_replay(tmp_path):
    engine = _engine(budget=0.3)
    with pytest.raises(ValueError, match="replay-adjusted"):
        prepare_and_propose_repair_population(
            engine=engine,
            parent_id="parent",
            plan=_plan(),
            cluster=_cluster(),
            provider=_provider(tmp_path),
            holdout_fingerprint_files=(_holdout(tmp_path),),
            variants=(RepairVariant("repair-only-fits", 0.2),),
            work_dir=tmp_path,
            replay=_replay(tmp_path, ratio=1.0),
        )
    assert engine.outstanding_candidates == 0
    assert not (tmp_path / ".chowder" / "repairs").exists()
