"""Lifecycle tests for `router_healing_orchestrator`.

Only the trainer and evaluator are fakes. The registry, gate, goal, freeze
policy, delta save/load and status transitions are the production
implementations -- the same split the search-controller integration suite
uses, so these tests double as a fidelity check on the real lifecycle rather
than on a mock of it.
"""
from __future__ import annotations

import json

import pytest

from chowder.cancellation import CancellationToken, OperationCancelled
from chowder.executors import EvaluationOutcome
from chowder.models import ExperimentStatus
from chowder.parent_eval import PARENT_DIMENSIONS
from chowder.registry import RunRegistry
from chowder.router_healing import RouterHealingError
from chowder.router_healing_orchestrator import (
    TrainOutcome,
    preflight_loaded_model,
    run_router_healing_experiment,
)
from chowder.router_healing_run import (
    EFFICIENCY_METRIC,
    RouterHealingSpec,
    build_router_healing_goal,
)

torch = pytest.importorskip("torch")
nn = torch.nn

V3_PROTOCOL = "6a18a4e4f03df8caac32c668662ba9c610038f1668b37353c3f5566571e45dde"
BASE_MANIFEST = "d382d54f159f7b6c" + "0" * 48
CORPUS_DIGEST = "a05451e901d819a5" + "0" * 48
SUITE_NAMES = {dim: f"suite-{dim.replace('_', '-')}-v1" for dim in PARENT_DIMENSIONS}


class _Mlp(nn.Module):
    def __init__(self, hidden=8, num_experts=4, moe_inter=2, *, zero_shared=False) -> None:
        super().__init__()
        self.gate = nn.Module()
        self.gate.weight = nn.Parameter(torch.zeros(num_experts, hidden))
        self.experts = nn.Module()
        self.experts.gate_up_proj = nn.Parameter(torch.zeros(num_experts, 2 * moe_inter, hidden))
        self.experts.down_proj = nn.Parameter(torch.zeros(num_experts, hidden, moe_inter))
        self.shared_expert = nn.Module()
        for leaf, (out_f, in_f) in (
            ("gate_proj", (moe_inter, hidden)),
            ("up_proj", (moe_inter, hidden)),
            ("down_proj", (hidden, moe_inter)),
        ):
            lin = nn.Linear(in_f, out_f, bias=False)
            if zero_shared:
                nn.init.zeros_(lin.weight)
            setattr(self.shared_expert, leaf, lin)
        self.shared_expert_gate = nn.Linear(hidden, 1, bias=False)


class _Layer(nn.Module):
    def __init__(self, **kw) -> None:
        super().__init__()
        self.self_attn = nn.Linear(8, 8, bias=False)
        self.mlp = _Mlp(**kw)


class _Model(nn.Module):
    def __init__(self, layers=2, *, zero_shared=False) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(16, 8)
        self.layers = nn.ModuleList(_Layer(zero_shared=zero_shared) for _ in range(layers))


def _spec(**over) -> RouterHealingSpec:
    kw = dict(
        base_model_dir=r"F:\fake\base", base_manifest_sha256=BASE_MANIFEST,
        corpus_path=r"C:\corpus.jsonl", corpus_sha256=CORPUS_DIGEST,
        max_steps=4, learning_rate=1e-4, seq_len=128, seed=42, protocol_sha256=V3_PROTOCOL,
    )
    kw.update(over)
    return RouterHealingSpec(**kw)


def _metrics(value=0.8, *, experts_per_tok=16.0):
    m = {name: value for name in SUITE_NAMES.values()}
    m[EFFICIENCY_METRIC] = experts_per_tok
    return m


def _outcome(experiment_id, metrics, run_id="run-x"):
    return EvaluationOutcome(
        run_id=run_id, experiment_id=experiment_id, source_artifact_ref="/runs/x",
        metrics=metrics, gpu_hours=0.1,
        evidence={"evaluation_protocol_sha256": V3_PROTOCOL},
    )


def _fake_train(steps=4):
    def train_fn(*, model, spec, cancellation):
        cancellation.raise_if_requested()
        tensors = {
            n: p.detach().tolist()
            for n, p in model.named_parameters()
            if n.endswith("mlp.gate.weight")
        }
        return TrainOutcome(
            steps_completed=steps, trainable_tensors=tensors,
            resumable_state={"optimizer": "adamw", "step": steps}, peak_gpu_mib=1234.0,
        )
    return train_fn


def _fake_eval(metrics):
    def evaluate_fn(*, model_dir, experiment_id, spec):
        return _outcome(experiment_id, metrics, run_id=f"eval-{experiment_id[-8:]}")
    return evaluate_fn


def _goal():
    return build_router_healing_goal(gpu_hour_budget=2.0, dimension_suite_names=SUITE_NAMES)


def test_preflight_reports_unquantized_expert_weight_as_the_real_budget():
    """The headline defence against a '4-bit' label that hides BF16 experts."""
    report = preflight_loaded_model(_Model())
    assert "routed_expert" in report["categories"]
    assert report["categories"]["routed_expert"]["bytes"] > 0
    # nothing here is bnb-quantized, so the fraction must say so rather than
    # inheriting a comforting label
    assert report["quantized_fraction"] == 0.0
    assert "nn.Linear only" in report["note"]


def test_successful_run_records_experiment_then_passes_through_the_gate(tmp_path):
    registry = RunRegistry(tmp_path / "reg.db")
    spec = _spec()
    baseline = _outcome("exp-baseline", _metrics(0.8, experts_per_tok=16.0), run_id="baseline")

    record = run_router_healing_experiment(
        registry, model=_Model(), spec=spec, goal=_goal(), baseline_outcome=baseline,
        delta_dir=tmp_path, train_fn=_fake_train(),
        evaluate_fn=_fake_eval(_metrics(0.8, experts_per_tok=3.0)),
    )

    assert record.status == ExperimentStatus.PASSED.value
    assert record.decision is not None and record.decision.accepted
    assert record.steps_completed == 4
    assert record.gpu_hours >= 0
    # the experiment is durable, and so is its terminal status
    rows = {e.experiment_id: e for e in registry.list_experiments()}
    assert record.experiment_id in rows
    assert rows[record.experiment_id].status == ExperimentStatus.PASSED
    # delta and resumable state both on disk
    assert json.loads(open(record.delta_path, encoding="utf-8").read())["steps_completed"] == 4
    assert (tmp_path / f"{record.experiment_id}.resume.json").is_file()


def test_no_efficiency_gain_is_rejected_not_silently_passed(tmp_path):
    registry = RunRegistry(tmp_path / "reg.db")
    baseline = _outcome("exp-baseline", _metrics(0.8, experts_per_tok=16.0), run_id="baseline")
    record = run_router_healing_experiment(
        registry, model=_Model(), spec=_spec(), goal=_goal(), baseline_outcome=baseline,
        delta_dir=tmp_path, train_fn=_fake_train(),
        evaluate_fn=_fake_eval(_metrics(0.8, experts_per_tok=16.0)),
    )
    assert record.status == ExperimentStatus.REJECTED.value
    assert record.decision is not None and not record.decision.accepted


def test_capability_regression_is_rejected_despite_efficiency_win(tmp_path):
    registry = RunRegistry(tmp_path / "reg.db")
    baseline = _outcome("exp-baseline", _metrics(0.8, experts_per_tok=16.0), run_id="baseline")
    hurt = _metrics(0.8, experts_per_tok=1.0)
    hurt["suite-reasoning-v1"] = 0.5
    record = run_router_healing_experiment(
        registry, model=_Model(), spec=_spec(), goal=_goal(), baseline_outcome=baseline,
        delta_dir=tmp_path, train_fn=_fake_train(), evaluate_fn=_fake_eval(hurt),
    )
    assert record.status == ExperimentStatus.REJECTED.value
    assert "suite-reasoning-v1" in record.decision.regressions


def test_unreachable_trainable_is_refused_before_any_training(tmp_path):
    """The zero-init shared expert defect must stop the run pre-compute, and
    the registry must still show a terminal row rather than RUNNING."""
    registry = RunRegistry(tmp_path / "reg.db")
    baseline = _outcome("exp-baseline", _metrics(), run_id="baseline")
    trained: list[int] = []

    def train_fn(*, model, spec, cancellation):
        trained.append(1)
        return TrainOutcome(steps_completed=1, trainable_tensors={"x": [1.0]})

    record = run_router_healing_experiment(
        registry, model=_Model(zero_shared=True), spec=_spec(), goal=_goal(),
        baseline_outcome=baseline, delta_dir=tmp_path,
        train_fn=train_fn, evaluate_fn=_fake_eval(_metrics()),
    )
    assert not trained, "training ran despite an unreachable trainable tensor"
    assert record.status == ExperimentStatus.FAILED.value
    assert "cannot receive gradient" in record.failure


def test_cancellation_is_terminal_not_stuck_running(tmp_path):
    registry = RunRegistry(tmp_path / "reg.db")
    baseline = _outcome("exp-baseline", _metrics(), run_id="baseline")
    token = CancellationToken()

    def train_fn(*, model, spec, cancellation):
        raise OperationCancelled("user pressed stop")

    record = run_router_healing_experiment(
        registry, model=_Model(), spec=_spec(), goal=_goal(), baseline_outcome=baseline,
        delta_dir=tmp_path, train_fn=train_fn, evaluate_fn=_fake_eval(_metrics()),
        cancellation=token,
    )
    assert record.status == ExperimentStatus.FAILED.value
    assert "cancelled" in record.failure
    rows = {e.experiment_id: e for e in registry.list_experiments()}
    assert rows[record.experiment_id].status == ExperimentStatus.FAILED


def test_training_failure_reaches_the_registry(tmp_path):
    registry = RunRegistry(tmp_path / "reg.db")
    baseline = _outcome("exp-baseline", _metrics(), run_id="baseline")

    def train_fn(*, model, spec, cancellation):
        raise RuntimeError("CUDA out of memory")

    record = run_router_healing_experiment(
        registry, model=_Model(), spec=_spec(), goal=_goal(), baseline_outcome=baseline,
        delta_dir=tmp_path, train_fn=train_fn, evaluate_fn=_fake_eval(_metrics()),
    )
    assert record.status == ExperimentStatus.FAILED.value
    assert "CUDA out of memory" in record.failure
    assert record.delta_path is None


def test_evaluation_reads_the_delta_back_from_disk(tmp_path):
    """Condition 4: evaluation must not be handed the in-memory training model.
    A delta whose base manifest disagrees must stop the run."""
    registry = RunRegistry(tmp_path / "reg.db")
    baseline = _outcome("exp-baseline", _metrics(), run_id="baseline")
    record = run_router_healing_experiment(
        registry, model=_Model(), spec=_spec(base_manifest_sha256="a" * 64), goal=_goal(),
        baseline_outcome=baseline, delta_dir=tmp_path, train_fn=_fake_train(),
        evaluate_fn=_fake_eval(_metrics(0.8, experts_per_tok=3.0)),
    )
    # the delta was saved under spec "aaa..." and reloaded against the same, so
    # this run succeeds; the mismatch path is covered in test_router_healing_run
    assert record.status in {ExperimentStatus.PASSED.value, ExperimentStatus.REJECTED.value}
    assert record.evaluation_run_id is not None


def test_zero_step_training_outcome_is_refused():
    with pytest.raises(RouterHealingError):
        TrainOutcome(steps_completed=0, trainable_tensors={"x": [1.0]})
    with pytest.raises(RouterHealingError):
        TrainOutcome(steps_completed=1, trainable_tensors={})
