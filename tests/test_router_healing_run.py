"""Regression tests for `chowder.router_healing_run`.

The load-bearing claim these tests defend is that healing cannot promote
itself: a regression, an incomplete evidence row, or a cross-protocol
comparison must all be refused by the real gate through this path. No
torch, no GPU, no checkpoint -- the accountability layer is pure data.
"""
from __future__ import annotations

import json

import pytest

from chowder.models import ExperimentResult, OptimizationDirection
from chowder.parent_eval import PARENT_DIMENSIONS
from chowder.router_healing_run import (
    EFFICIENCY_METRIC,
    RouterHealingRunError,
    RouterHealingSpec,
    build_healing_experiment,
    build_router_healing_goal,
    experiment_result_from_outcome,
    healing_experiment_id,
    judge_router_healing,
    load_router_delta,
    save_router_delta,
)

V3_PROTOCOL = "6a18a4e4f03df8caac32c668662ba9c610038f1668b37353c3f5566571e45dde"
OTHER_PROTOCOL = "c5e964df3a2a7917" + "0" * 48
BASE_MANIFEST = "d382d54f159f7b6c" + "0" * 48
CORPUS_DIGEST = "a05451e901d819a5" + "0" * 48

SUITE_NAMES = {dim: f"suite-{dim.replace('_', '-')}-v1" for dim in PARENT_DIMENSIONS}


def _spec(**overrides) -> RouterHealingSpec:
    kwargs = dict(
        base_model_dir=r"F:\Local Models\HuggingFace\Qwen\Qwen3.8-27B-MoE-E16",
        base_manifest_sha256=BASE_MANIFEST,
        corpus_path=r"C:\corpus.jsonl",
        corpus_sha256=CORPUS_DIGEST,
        max_steps=20,
        learning_rate=1e-4,
        seq_len=256,
        seed=20260907,
        protocol_sha256=V3_PROTOCOL,
    )
    kwargs.update(overrides)
    return RouterHealingSpec(**kwargs)


def _metrics(value: float = 1.0, *, experts_per_tok: float = 16.0, **overrides) -> dict[str, float]:
    metrics = {name: value for name in SUITE_NAMES.values()}
    metrics[EFFICIENCY_METRIC] = experts_per_tok
    metrics.update(overrides)
    return metrics


def _result(metrics: dict[str, float], *, protocol: str = V3_PROTOCOL, experiment_id: str = "exp-x") -> ExperimentResult:
    return ExperimentResult(
        experiment_id=experiment_id,
        metrics=metrics,
        gpu_hours=0.5,
        evidence={"evaluation_protocol_sha256": protocol},
    )


def test_spec_digest_is_stable_and_id_is_derived():
    spec = _spec()
    assert spec.digest() == _spec().digest()
    assert healing_experiment_id(spec).startswith("exp-router-healing-")
    # a different budget is a different run
    assert _spec(max_steps=40).digest() != spec.digest()


def test_spec_rejects_placeholder_digests_and_budgets():
    with pytest.raises(ValueError):
        _spec(base_manifest_sha256="not-a-digest")
    with pytest.raises(ValueError):
        _spec(max_steps=0)
    with pytest.raises(ValueError):
        _spec(learning_rate=0.0)


def test_healing_experiment_carries_real_gpu_hour_estimate():
    spec = _spec()
    exp = build_healing_experiment(spec, parent_experiment_id=None, estimated_gpu_hours=1.5)
    assert exp.experiment_id == healing_experiment_id(spec)
    assert exp.estimated_gpu_hours == 1.5
    assert "router-healing" in exp.tags
    assert exp.config_patch["experts_frozen"] is True
    with pytest.raises(RouterHealingRunError):
        build_healing_experiment(spec, parent_experiment_id=None, estimated_gpu_hours=0.0)


def test_goal_covers_every_protected_dimension_plus_efficiency():
    goal = build_router_healing_goal(gpu_hour_budget=2.0, dimension_suite_names=SUITE_NAMES)
    assert goal.require_protocol_match is True
    assert len(goal.metrics) == len(PARENT_DIMENSIONS) + 1
    assert {m.name for m in goal.metrics} == set(SUITE_NAMES.values()) | {EFFICIENCY_METRIC}
    assert all(m.regression_tolerance == 0.0 for m in goal.metrics)
    efficiency = goal.target(EFFICIENCY_METRIC)
    assert efficiency is not None
    assert efficiency.direction is OptimizationDirection.MINIMIZE


def test_goal_refuses_to_silently_drop_a_dimension():
    partial = dict(SUITE_NAMES)
    partial.pop("behavior")
    with pytest.raises(RouterHealingRunError):
        build_router_healing_goal(gpu_hour_budget=2.0, dimension_suite_names=partial)


def test_healing_that_cuts_experts_while_holding_capability_is_accepted():
    goal = build_router_healing_goal(gpu_hour_budget=2.0, dimension_suite_names=SUITE_NAMES)
    baseline = _result(_metrics(0.8, experts_per_tok=16.0), experiment_id="exp-parent-a")
    candidate = _result(_metrics(0.8, experts_per_tok=3.0), experiment_id="exp-healed")
    decision = judge_router_healing(goal=goal, baseline=baseline, candidate=candidate)
    assert decision.accepted, decision.reason
    assert decision.score > 0


def test_healing_with_no_efficiency_gain_is_refused_as_no_improvement():
    """A flat run that changes nothing is not a promotion -- the whole point
    of healing is the compute win, so holding capability alone is not enough."""
    goal = build_router_healing_goal(gpu_hour_budget=2.0, dimension_suite_names=SUITE_NAMES)
    baseline = _result(_metrics(0.8, experts_per_tok=16.0), experiment_id="exp-parent-a")
    candidate = _result(_metrics(0.8, experts_per_tok=16.0), experiment_id="exp-healed")
    decision = judge_router_healing(goal=goal, baseline=baseline, candidate=candidate)
    assert not decision.accepted
    assert "improve" in decision.reason


def test_efficiency_gain_cannot_buy_a_capability_regression():
    """The load-bearing guard: even a huge experts-per-token reduction must
    not promote a model that broke a protected dimension."""
    goal = build_router_healing_goal(gpu_hour_budget=2.0, dimension_suite_names=SUITE_NAMES)
    baseline = _result(_metrics(0.8, experts_per_tok=16.0), experiment_id="exp-parent-a")
    hurt = _metrics(0.8, experts_per_tok=1.0)  # maximal efficiency win
    hurt["suite-reasoning-v1"] = 0.6  # but reasoning broke
    candidate = _result(hurt, experiment_id="exp-healed")
    decision = judge_router_healing(goal=goal, baseline=baseline, candidate=candidate)
    assert not decision.accepted
    assert "suite-reasoning-v1" in decision.regressions


def test_cross_protocol_comparison_is_refused_before_scores_are_weighed():
    goal = build_router_healing_goal(gpu_hour_budget=2.0, dimension_suite_names=SUITE_NAMES)
    baseline = _result(_metrics(0.8), protocol=OTHER_PROTOCOL, experiment_id="exp-parent-a")
    candidate = _result(_metrics(1.0), protocol=V3_PROTOCOL, experiment_id="exp-healed")
    decision = judge_router_healing(goal=goal, baseline=baseline, candidate=candidate)
    # candidate scores strictly better on every metric, and is STILL refused
    assert not decision.accepted
    assert "protocol" in decision.reason


def test_experiment_result_bridge_accepts_mapping_and_object_rows():
    row = {
        "experiment_id": "exp-healed",
        "metrics": {"suite-reasoning-v1": 1.0},
        "gpu_hours": 0.25,
        "source_artifact_ref": "/runs/healed",
        "evidence": {"evaluation_protocol_sha256": V3_PROTOCOL},
    }
    from_mapping = experiment_result_from_outcome(row)
    assert from_mapping.experiment_id == "exp-healed"
    assert from_mapping.artifact_ref == "/runs/healed"

    class _Row:
        experiment_id = "exp-healed"
        metrics = {"suite-reasoning-v1": 1.0}
        gpu_hours = 0.25
        source_artifact_ref = "/runs/healed"
        evidence = {"evaluation_protocol_sha256": V3_PROTOCOL}

    from_object = experiment_result_from_outcome(_Row())
    assert from_object.metrics == from_mapping.metrics


def test_experiment_result_bridge_refuses_incomplete_rows():
    with pytest.raises(RouterHealingRunError):
        experiment_result_from_outcome({"experiment_id": "exp-healed"})


def test_router_delta_roundtrip_and_base_mismatch_fails_closed(tmp_path):
    spec = _spec()
    tensors = {
        "model.layers.0.mlp.gate.weight": [[0.1, 0.2], [0.3, 0.4]],
        "model.layers.0.mlp.shared_expert_gate.weight": [[0.5, 0.6]],
    }
    artifact = save_router_delta(tensors, tmp_path / "delta.json", spec=spec, steps_completed=20)
    assert artifact.trainable_param_count == 6
    assert artifact.steps_completed == 20
    assert artifact.spec_digest == spec.digest()

    loaded = load_router_delta(tmp_path / "delta.json", expected_base_manifest_sha256=BASE_MANIFEST)
    assert loaded["tensors"].keys() == tensors.keys()

    with pytest.raises(RouterHealingRunError):
        load_router_delta(tmp_path / "delta.json", expected_base_manifest_sha256="f" * 64)


def test_router_delta_refuses_overwrite_and_empty_payloads(tmp_path):
    spec = _spec()
    path = tmp_path / "delta.json"
    save_router_delta({"a": [1.0]}, path, spec=spec, steps_completed=1)
    with pytest.raises(RouterHealingRunError):
        save_router_delta({"a": [1.0]}, path, spec=spec, steps_completed=1)
    with pytest.raises(RouterHealingRunError):
        save_router_delta({}, tmp_path / "empty.json", spec=spec, steps_completed=1)


def test_delta_sha256_matches_file_bytes(tmp_path):
    import hashlib

    spec = _spec()
    path = tmp_path / "delta.json"
    artifact = save_router_delta({"a": [1.0, 2.0]}, path, spec=spec, steps_completed=3)
    on_disk = hashlib.sha256(path.read_bytes()).hexdigest()
    assert artifact.delta_sha256 == on_disk
    # the sidecar is readable JSON, not an opaque blob
    assert json.loads(path.read_text(encoding="utf-8"))["steps_completed"] == 3
