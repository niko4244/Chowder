"""End-to-end regression test for the autonomous search controller.

Every other test in this suite exercises one module against hand-built
inputs. This file exercises the *seams*: the real `EvolutionEngine`, the
real `ExperimentCycleRunner`, the real hard gate (`gate.evaluate_candidate`
via `tournament.rank_candidates`), the real `RunRegistry`, the real UCB1
candidate prioritizer, the real successive-halving scheduler and the real
autonomous-repair machinery, all driven through one loop.

The ONLY faked components are the trainer and the evaluator -- i.e. the GPU
boundary. Everything on this side of that boundary is the production object:
real reservations, real settlement arithmetic, real gate decisions, real
SQLite persistence, real on-disk artifact/checkpoint layouts hashed with the
real `provenance.sha256_*` helpers, real contamination auditing and real
replay materialization. Faking anything else would defeat the purpose --
these tests exist to prove the joins hold, not that the parts work alone.

Invariants proven here:
  * exact GPU-hour accounting (reserved -> spent -> remaining) across
    multiple rounds, including candidates that fail, are gate-rejected,
    are cut off, and are withdrawn;
  * no double reservation, and withdrawal/settlement releasing exactly
    what was reserved;
  * no gate bypass on any path -- including the repair path, where a
    repair that fails its own independent evaluation must never replace
    the working baseline;
  * cancellation mid-search leaving an exact ledger, and a real
    checkpoint resume afterwards;
  * immutable provenance -- recorded evidence is never mutated after the
    fact, and a hash recorded at training time still verifies at the end
    of the loop.

Test names deliberately avoid the substring "real": this suite is often run
with `-k "not real"` to exclude the GPU-only smoke tests, and none of these
need hardware.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

from chowder.autonomous_repair import run_single_hop_autonomous_repair
from chowder.cancellation import CancellationToken
from chowder.candidate_selection import prioritize_candidates
from chowder.contamination import write_holdout_fingerprint_index
from chowder.cycle import ExperimentCycleRunner
from chowder.engine import EvolutionEngine
from chowder.executors import EvaluationOutcome, ExecutionContext, TrainingArtifact
from chowder.failures import FailureRecord, FailureSourceRole
from chowder.graph import GraphInvariantError
from chowder.local_corpus_provider import LocalCorpusRepairProvider
from chowder.memory import HardwareProfile
from chowder.models import (
    Experiment,
    ExperimentResult,
    ExperimentStatus,
    Goal,
    Hypothesis,
    MetricTarget,
)
from chowder.provenance import sha256_directory, sha256_file
from chowder.registry import RegistryInvariantError, RunRegistry
from chowder.repair_candidates import RepairVariant
from chowder.successive_halving import run_successive_halving

# Every GPU-hour figure in this file derives from these four constants, so
# each asserted total below is arithmetic a reader can check by hand rather
# than a number copied out of a previous run.
_TRAIN_GPU_HOURS = 0.2
_EVAL_GPU_HOURS = 0.05
_CANDIDATE_COST = _TRAIN_GPU_HOURS + _EVAL_GPU_HOURS
# Declared in base_config; ExperimentCycleRunner adds it to the profiled
# training estimate when it resizes a reservation, so every reservation in
# this file is exactly `experiment.estimated_gpu_hours + this`.
_EVAL_RESERVE_GPU_HOURS = 0.05

_PROTOCOL_SHA = "p" * 64
_BASELINE_QUALITY = 0.7

# successive_halving names a survivor's next-round child `<parent>-r<N>`,
# and a round-2 child of that `<parent>-r1-r2`. Strip the whole suffix chain
# to recover the lineage root a fake evaluator scores against.
_HALVING_SUFFIX = re.compile(r"(?:-r\d+)+$")


def _lineage_root(experiment_id: str) -> str:
    return _HALVING_SUFFIX.sub("", experiment_id)


def _write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


# --- fakes: the GPU boundary, and nothing beyond it ---------------------------


class FakeTrainer:
    """No GPU and no model -- but real on-disk output.

    Writes a real adapter directory (hashed with the real
    `provenance.sha256_directory`) and the real `trainer/checkpoint-N`
    layout that `successive_halving._latest_checkpoint_dir` and
    `checkpoint_bisect._checkpoints_for_run` actually glob for, and reads
    its dataset/replay paths back out of the real resolved config the real
    `ExperimentGraph` produced. Records what each call actually received so
    round-to-round wiring can be asserted on.
    """

    name = "fake-trainer"

    def __init__(
        self,
        work_dir,
        *,
        failing_ids=(),
        cancelling_ids=(),
        cancellation: CancellationToken | None = None,
        ledger_probe=None,
    ):
        self.work_dir = Path(work_dir)
        self.failing_ids = frozenset(failing_ids)
        self.cancelling_ids = frozenset(cancelling_ids)
        self.cancellation = cancellation
        self.ledger_probe = ledger_probe
        self.calls: list[dict] = []

    def profile(self, experiment, context):
        # Same as every other executor double in this suite: the runner
        # falls back to the standing reservation, which keeps the
        # reservation arithmetic below exactly predictable.
        raise NotImplementedError

    def cancel(self, run_id):
        pass

    def _dataset_sha(self, backend, key: str = "dataset") -> str | None:
        raw = backend.get(key)
        if not isinstance(raw, str) or not raw.strip():
            return None
        path = Path(raw)
        if not path.is_absolute():
            path = self.work_dir / path
        return sha256_file(path.resolve())

    @staticmethod
    def _write_checkpoints(artifact_dir: Path, training) -> None:
        if training.get("save_strategy") != "steps":
            return
        max_steps = int(training.get("max_steps", 1))
        save_steps = max(1, int(training.get("save_steps", max_steps)))
        steps = sorted({*range(save_steps, max_steps + 1, save_steps), max_steps})
        for step in steps:
            checkpoint = artifact_dir / "trainer" / f"checkpoint-{step}"
            checkpoint.mkdir(parents=True, exist_ok=True)
            (checkpoint / "adapter.bin").write_bytes(f"{artifact_dir.name}@{step}".encode())

    def run(self, experiment, context):
        backend = dict(context.resolved_config["backend"])
        training = dict(backend.get("training", {}))
        self.calls.append(
            {
                "experiment_id": experiment.experiment_id,
                "max_steps": training.get("max_steps"),
                "resume_from_checkpoint": backend.get("resume_from_checkpoint"),
                "parent_adapter": backend.get("parent_adapter"),
                "ledger": (
                    self.ledger_probe(experiment.experiment_id)
                    if self.ledger_probe is not None
                    else None
                ),
            }
        )
        if experiment.experiment_id in self.cancelling_ids:
            # Models `CancellationToken.request()` successfully terminating an
            # in-flight worker: the token flips, and the interrupted call
            # raises instead of returning an artifact.
            self.cancellation.request()
            raise RuntimeError("worker terminated")
        if experiment.experiment_id in self.failing_ids:
            raise RuntimeError("CUDA error: device-side assert triggered")

        artifact_dir = self.work_dir / "artifacts" / experiment.experiment_id
        artifact_dir.mkdir(parents=True, exist_ok=False)
        (artifact_dir / "adapter_config.json").write_text(
            '{"peft_type":"LORA"}\n', encoding="utf-8"
        )
        (artifact_dir / "adapter.bin").write_bytes(experiment.experiment_id.encode("utf-8"))
        self._write_checkpoints(artifact_dir, training)

        replay = backend.get("replay")
        replay_sha = (
            self._dataset_sha(replay) if isinstance(replay, dict) else None
        )
        return TrainingArtifact(
            run_id=f"train-{experiment.experiment_id}",
            experiment_id=experiment.experiment_id,
            artifact_ref=str(artifact_dir),
            gpu_hours=_TRAIN_GPU_HOURS,
            evidence={
                "dataset_sha256": self._dataset_sha(backend),
                "replay_dataset_sha256": replay_sha,
                "artifact_sha256": sha256_directory(artifact_dir),
            },
        )


class FakeEvaluator:
    """No GPU -- but real, complete evaluation *evidence*.

    Emits the protocol digest and holdout-fingerprint evidence the real
    autonomous-repair coordinator re-verifies before it will build any
    repair, so the repair path below runs its real provenance checks
    against real files rather than being handed a pre-blessed object.
    """

    name = "fake-eval"

    def __init__(self, holdout_index, scores, *, repair_score: float | None = None):
        self.holdout_path = Path(holdout_index).resolve()
        self.holdout_sha = sha256_file(self.holdout_path)
        self.scores = dict(scores)
        self.repair_score = repair_score
        self.calls: list[str] = []

    def profile(self, experiment, context):
        raise NotImplementedError

    def cancel(self, run_id):
        pass

    def _quality(self, experiment_id: str) -> float:
        if self.repair_score is not None and experiment_id.startswith("repair-"):
            return self.repair_score
        return self.scores[_lineage_root(experiment_id)]

    def evaluate(self, *, experiment, artifact, context):
        self.calls.append(experiment.experiment_id)
        return EvaluationOutcome(
            run_id=f"eval-{experiment.experiment_id}",
            experiment_id=experiment.experiment_id,
            source_artifact_ref=artifact.artifact_ref,
            metrics={"quality": self._quality(experiment.experiment_id)},
            gpu_hours=_EVAL_GPU_HOURS,
            evidence={
                "evaluator": "fake-eval",
                "protocol_sha256": _PROTOCOL_SHA,
                "holdout_fingerprint_sha256": {"reasoning": self.holdout_sha},
                "suite_evidence": {
                    "reasoning": {
                        "holdout_fingerprints_file": str(self.holdout_path),
                        "holdout_fingerprints_sha256": self.holdout_sha,
                    }
                },
            },
        )


# --- shared harness -----------------------------------------------------------


def _harvest(evaluation):
    """Only non-repair candidates produce failures, so the repair loop has
    something to diagnose on the first hop and terminates on the second."""
    if evaluation.experiment_id.startswith("repair-"):
        return ()
    return (
        FailureRecord(
            failure_id="f" * 64,
            experiment_id=evaluation.experiment_id,
            evaluation_run_id=evaluation.run_id,
            evaluator="fake-eval",
            suite="reasoning",
            row_index=0,
            protocol_sha256=_PROTOCOL_SHA,
            artifact_sha256="a" * 64,
            source_role=FailureSourceRole.GATE_HOLDOUT,
            prompt="hidden benchmark prompt",
            expected="hidden answer",
            prediction="wrong answer",
            score=0.0,
            failure_kind="answer_mismatch",
        ),
    )


def _seed_work_dir(tmp_path: Path) -> Path:
    """Write the real training corpus and the real holdout fingerprint index
    the repair path re-hashes. Returns the holdout index path."""
    _write_jsonl(
        tmp_path / "base.jsonl",
        [
            {"text": "original training example one"},
            {"text": "original training example two"},
            {"text": "original training example three"},
        ],
    )
    holdout_index = tmp_path / "holdout-index.jsonl"
    write_holdout_fingerprint_index(
        [("hidden benchmark prompt", "hidden answer")], holdout_index
    )
    return holdout_index


def _base_config() -> dict:
    return {
        "backend": {
            "base_model": "example/model",
            "dataset": "base.jsonl",
            "text_field": "text",
            "training": {"learning_rate": 1e-4, "epochs": 2, "batch_size": 1},
            "lora": {"r": 16, "alpha": 32},
        },
        "evaluation": {
            "type": "fake-eval",
            "estimated_gpu_hours": _EVAL_RESERVE_GPU_HOURS,
        },
    }


def _goal(*, budget: float, minimum: float = 0.8, max_parallel: int = 8) -> Goal:
    return Goal(
        (MetricTarget("quality", minimum=minimum),),
        gpu_hour_budget=budget,
        max_parallel_candidates=max_parallel,
    )


def _engine(*, budget: float, minimum: float = 0.8, max_parallel: int = 8) -> EvolutionEngine:
    return EvolutionEngine(
        _goal(budget=budget, minimum=minimum, max_parallel=max_parallel),
        ExperimentResult("baseline", {"quality": _BASELINE_QUALITY}, 0.0),
    )


def _runner(tmp_path, engine, trainer, evaluator, *, registry=None, cancellation=None):
    return ExperimentCycleRunner(
        engine=engine,
        trainer=trainer,
        evaluator=evaluator,
        context=ExecutionContext(HardwareProfile(16, 64, 500, 12, 40, 3), str(tmp_path), 7),
        base_config=_base_config(),
        registry=registry,
        failure_harvester=_harvest,
        cancellation=cancellation,
    )


def _experiment(experiment_id, patch=None, *, hours=0.2, parent=None) -> Experiment:
    return Experiment(
        experiment_id=experiment_id,
        parent_id=parent,
        hypothesis=Hypothesis("quality regression", "reasoning weakness", "candidate change"),
        config_patch=dict(patch or {}),
        estimated_gpu_hours=hours,
    )


def _assert_ledger_balances(engine, *, spent: float) -> None:
    """The whole point of the accounting invariant: nothing is left
    reserved, nothing was lost, and spent + remaining is the budget."""
    assert engine.spent_gpu_hours == pytest.approx(spent)
    assert engine.reserved_gpu_hours == pytest.approx(0.0)
    assert engine.outstanding_candidates == 0
    assert engine.remaining_budget == pytest.approx(engine.goal.gpu_hour_budget - spent)


def _repair_provider(tmp_path: Path) -> LocalCorpusRepairProvider:
    corpus = tmp_path / "repair-corpus.jsonl"
    _write_jsonl(
        corpus,
        [
            {
                "example_id": "independent-1",
                "suite": "reasoning",
                "strategy": "near_neighbor_reasoning",
                "prompt": "independent reasoning example one",
                "expected": "answer one",
            },
            {
                "example_id": "independent-2",
                "suite": "reasoning",
                "strategy": "near_neighbor_reasoning",
                "prompt": "independent reasoning example two",
                "expected": "answer two",
            },
        ],
    )
    return LocalCorpusRepairProvider([corpus], max_examples=2, examples_per_failure=2)


def _run_rejection_then_repair(tmp_path, registry, *, repair_score: float):
    """Drive the real chain: a candidate is trained, independently
    evaluated, gate-REJECTED, its failures clustered into a real repair
    plan, and a real two-variant repair population is built from an
    independent corpus (real contamination audit, real replay
    materialization, real parent-adapter hash binding) and re-evaluated.

    `repair_score` decides whether that repair clears the gate, which is
    the only difference between the promotion and the failed-canary case.
    """
    holdout_index = _seed_work_dir(tmp_path)
    engine = _engine(budget=4.0, max_parallel=4)
    trainer = FakeTrainer(tmp_path)
    evaluator = FakeEvaluator(
        holdout_index, {"source": 0.65}, repair_score=repair_score
    )
    runner = _runner(tmp_path, engine, trainer, evaluator, registry=registry)

    source = _experiment("source", hours=0.5)
    registry.record_experiment(source)
    assert engine.propose((source,)) == (source,)
    source_generation = runner.run_generation((source,))
    assert source_generation.promoted is None
    assert engine.graph.nodes["source"].status is ExperimentStatus.REJECTED
    # Captured before any repair exists, so a caller can assert the very
    # same object is still the baseline afterwards.
    baseline_before_repair = engine.baseline

    repair_outcome = run_single_hop_autonomous_repair(
        runner=runner,
        source_generation=source_generation,
        provider=_repair_provider(tmp_path),
        variants=(
            RepairVariant("lr-low", 0.3, training_patch={"learning_rate": 5e-5}),
            RepairVariant("epochs-low", 0.3, training_patch={"epochs": 1}),
        ),
    )
    return runner, source_generation, repair_outcome, baseline_before_repair


# --- 1. the full loop: history -> prioritization -> halving -> promotion -------


def test_full_search_loop_from_history_to_promotion_balances_the_gpu_hour_ledger(tmp_path):
    """History -> UCB1 prioritization -> staged successive halving with a
    real checkpoint resume -> hard gate -> promotion, with the GPU-hour
    ledger balancing exactly at the end.

    A registry is deliberately NOT attached to the halving runner: doing so
    currently crashes, which
    `test_successive_halving_persists_the_child_experiments_it_creates`
    documents as a real bug. The registry seam is still exercised for real
    here (the historical campaign writes to it, and the prioritizer reads
    its history straight back out of it) and again in the repair tests
    below.
    """
    holdout_index = _seed_work_dir(tmp_path)

    with RunRegistry(tmp_path / "runs.db") as registry:
        # -- a real prior campaign, persisted, that later becomes history --
        history_engine = _engine(budget=10.0)
        history_runner = _runner(
            tmp_path,
            history_engine,
            FakeTrainer(tmp_path),
            FakeEvaluator(holdout_index, {"hist-lr": 0.90, "hist-lora": 0.55}),
            registry=registry,
        )
        history_experiments = (
            _experiment("hist-lr", {"backend": {"training": {"learning_rate": 3e-4}}}),
            _experiment("hist-lora", {"backend": {"lora": {"r": 32}}}),
        )
        for experiment in history_experiments:
            registry.record_experiment(experiment)
        history_engine.propose(history_experiments)
        history_generation = history_runner.run_generation(history_experiments)
        assert history_generation.promoted is not None
        assert history_generation.promoted.experiment_id == "hist-lr"

        # -- history reconstructed from durable state, not from memory --
        persisted = {e.experiment_id: e for e in registry.list_experiments()}
        history = tuple(
            (persisted[result.experiment_id], result) for result in registry.list_results()
        )
        assert len(history) == 2

        # -- a fresh campaign, prioritized by what that history taught --
        engine = _engine(budget=10.0)
        trainer = FakeTrainer(tmp_path)
        evaluator = FakeEvaluator(
            holdout_index,
            {"lr-tune": 0.90, "warmup": 0.85, "decay": 0.80, "lora-rank": 0.55},
        )
        runner = _runner(tmp_path, engine, trainer, evaluator)

        pool = (
            # Deliberately worst-first on input: the historically bad arm
            # (backend.lora.r) leads, the historically good arm
            # (backend.training.learning_rate) is second, two never-tried
            # arms are last.
            _experiment("lora-rank", {"backend": {"lora": {"r": 8}}}),
            _experiment("lr-tune", {"backend": {"training": {"learning_rate": 5e-5}}}),
            _experiment("warmup", {"backend": {"training": {"warmup_ratio": 0.1}}}),
            _experiment("decay", {"backend": {"training": {"weight_decay": 0.01}}}),
        )
        prioritized = prioritize_candidates(
            pool, history=history, goal=engine.goal, baseline=engine.baseline
        )
        assert [e.experiment_id for e in prioritized] == [
            "warmup",  # untried arm: UCB1 +inf, stable-sorted before "decay"
            "decay",  # untried arm
            "lr-tune",  # historically rewarding arm
            "lora-rank",  # historically punished arm, runs last
        ]

        outcome = run_successive_halving(
            runner,
            prioritized,
            initial_max_steps=2,
            step_multiplier=2.0,
            survival_fraction=0.5,
            min_survivors=1,
        )

    # -- the gate, not the priority order, decides who wins --
    assert outcome.total_rounds == 2
    round0, round1 = outcome.rounds
    assert round0.survivor_experiment_ids == ("lr-tune", "warmup")
    assert round0.eliminated_by_gate_experiment_ids == ("lora-rank",)
    assert round0.eliminated_by_cutoff_experiment_ids == ("decay",)
    assert outcome.promoted is not None
    assert outcome.promoted.experiment_id == "lr-tune-r1"
    assert engine.baseline is outcome.promoted
    # The candidate prioritization ran first is NOT the one promoted --
    # UCB1 only decides who spends GPU-hours first, never who wins.
    assert prioritized[0].experiment_id != outcome.promoted.experiment_id

    # -- a gate-rejected candidate never advances a round --
    round1_ids = {c.experiment_id for c in round1.generation.candidates}
    assert round1_ids == {"lr-tune-r1", "warmup-r1"}
    assert not any(call["experiment_id"].startswith("lora-rank-r") for call in trainer.calls)

    # -- round 1 genuinely resumed round 0's checkpoint, not a restart --
    resume_call = next(c for c in trainer.calls if c["experiment_id"] == "lr-tune-r1")
    round0_artifact = next(
        c.artifact for c in round0.generation.candidates if c.experiment_id == "lr-tune"
    )
    expected_checkpoint = Path(round0_artifact.artifact_ref) / "trainer" / "checkpoint-2"
    assert resume_call["resume_from_checkpoint"] == str(expected_checkpoint)
    assert expected_checkpoint.is_dir()
    assert resume_call["max_steps"] == 4  # doubled budget for the survivor

    # -- exact accounting: 4 candidates in round 0 + 2 in round 1 --
    expected_spend = 6 * _CANDIDATE_COST
    _assert_ledger_balances(engine, spent=expected_spend)
    assert outcome.total_gpu_hours == pytest.approx(expected_spend)
    for experiment_id in ("lr-tune", "warmup", "decay", "lora-rank", "lr-tune-r1", "warmup-r1"):
        assert not engine.has_reservation(experiment_id)
        with pytest.raises(ValueError, match="no active reservation"):
            engine.reservation_for(experiment_id)


# --- 2. no gate bypass, including the repair path -----------------------------


def test_a_repair_that_fails_its_own_gate_never_replaces_the_baseline(tmp_path):
    """The roadmap's "auto-revert on failed canary" claim, driven end to end
    for the first time: a rejected candidate is really repaired (independent
    sources, contamination audit, replay rehearsal, parent-adapter hash
    binding) and the repair is really trained and really re-evaluated --
    and because the repair's own independent evaluation regresses, nothing
    is promoted and the working baseline object is untouched.
    """
    with RunRegistry(tmp_path / "runs.db") as registry:
        runner, source_generation, outcome, baseline_before_repair = (
            _run_rejection_then_repair(tmp_path, registry, repair_score=0.60)
        )
        engine = runner.engine
        assert baseline_before_repair.experiment_id == "baseline"

        # The repair machinery really ran: two repair candidates were
        # trained, each continuing from the source's exact hashed weights.
        repair_ids = tuple(
            e.experiment_id for e in outcome.population.proposed_candidates
        )
        assert len(repair_ids) == 2
        source_artifact = source_generation.candidates[0].artifact
        assert source_artifact is not None
        for repair_id in repair_ids:
            call = next(c for c in runner.trainer.calls if c["experiment_id"] == repair_id)
            assert call["parent_adapter"] == {
                "path": str(Path(source_artifact.artifact_ref).resolve()),
                "sha256": source_artifact.evidence["artifact_sha256"],
            }

        # ...and every one of them was still refused by the hard gate.
        assert outcome.promoted is None
        assert outcome.repair_generation.ranking
        assert all(
            ranked.decision.accepted is False
            for ranked in outcome.repair_generation.ranking
        )
        # Identity, not equality: the working baseline object was never
        # even replaced, on any path through the repair loop.
        assert engine.baseline is baseline_before_repair
        assert engine.graph.nodes["source"].status is ExperimentStatus.REJECTED
        for repair_id in repair_ids:
            assert engine.graph.nodes[repair_id].status is ExperimentStatus.REJECTED

        # Rejection is durable, not just in-memory.
        statuses = {e.experiment_id: e.status for e in registry.list_experiments()}
        assert statuses["source"] is ExperimentStatus.REJECTED
        for repair_id in repair_ids:
            assert statuses[repair_id] is ExperimentStatus.REJECTED

        # The failed canary still cost exactly what it really used, and the
        # engine's own ledger agrees with the independently persisted
        # per-candidate costs -- a cross-check that does not go through the
        # same constants the fakes report.
        _assert_ledger_balances(engine, spent=3 * _CANDIDATE_COST)
        assert engine.spent_gpu_hours == pytest.approx(
            sum(result.gpu_hours for result in registry.list_results())
        )


# --- 3. no double reservation; withdrawal releases exactly what it reserved ----


def test_budget_is_reserved_once_per_candidate_and_withdrawal_releases_exactly_that(
    tmp_path,
):
    """Reservations are the search controller's only protection against
    overcommitting a finite GPU-hour budget, so this checks them from the
    inside: a `ledger_probe` reads the live engine at the moment each
    candidate is actually training, where a double reservation would show
    up as an inflated total.
    """
    holdout_index = _seed_work_dir(tmp_path)
    engine = _engine(budget=10.0)

    def probe(experiment_id):
        # A second reservation for the same candidate would inflate either
        # its own figure or the running total.
        return (
            engine.reservation_for(experiment_id),
            engine.reserved_gpu_hours,
            engine.spent_gpu_hours,
        )

    trainer = FakeTrainer(tmp_path, ledger_probe=probe)
    evaluator = FakeEvaluator(holdout_index, {f"cand-{i}": 0.90 for i in range(4)})
    runner = _runner(tmp_path, engine, trainer, evaluator)

    pool = tuple(_experiment(f"cand-{i}", hours=0.2) for i in range(4))
    assert engine.propose(pool) == pool
    assert engine.reserved_gpu_hours == pytest.approx(4 * 0.2)
    assert engine.outstanding_candidates == 4

    # Re-proposing an already-proposed candidate is refused atomically:
    # the ledger must be untouched, not silently doubled.
    before = (engine.reserved_gpu_hours, engine.spent_gpu_hours, engine.outstanding_candidates)
    with pytest.raises(GraphInvariantError, match="duplicate experiment id"):
        engine.propose((pool[0],))
    assert (
        engine.reserved_gpu_hours,
        engine.spent_gpu_hours,
        engine.outstanding_candidates,
    ) == before

    # Withdrawing two never-run candidates releases exactly their estimates.
    withdrawn = engine.withdraw_proposals(("cand-2", "cand-3"))
    assert tuple(e.experiment_id for e in withdrawn) == ("cand-2", "cand-3")
    assert engine.reserved_gpu_hours == pytest.approx(2 * 0.2)
    assert engine.outstanding_candidates == 2
    assert "cand-2" not in engine.graph.nodes and "cand-3" not in engine.graph.nodes
    assert engine.spent_gpu_hours == pytest.approx(0.0)

    outcome = runner.run_round(pool[:2], promote=True)
    assert outcome.promoted is not None

    # Each surviving candidate was resized exactly once, from 0.2 to
    # 0.2 + the declared evaluation reserve, and the running total moved by
    # exactly that delta -- never by a whole extra reservation.
    expected_reservation = 0.2 + _EVAL_RESERVE_GPU_HOURS
    ledgers = [call["ledger"] for call in trainer.calls]
    assert len(ledgers) == 2
    for index, (own, total, spent) in enumerate(ledgers, start=1):
        assert own == pytest.approx(expected_reservation)
        assert total == pytest.approx(2 * 0.2 + index * _EVAL_RESERVE_GPU_HOURS)
        assert spent == pytest.approx(0.0)  # nothing settles until adjudication

    _assert_ledger_balances(engine, spent=2 * _CANDIDATE_COST)


# --- 4. cancellation, then resume ---------------------------------------------


def test_cancelled_search_charges_only_what_ran_and_resumes_from_a_checkpoint(tmp_path):
    """A cancellation mid-round must leave the ledger exact -- the
    interrupted candidate charged conservatively, the not-yet-started one
    charged nothing, no reservation stranded -- and the search must be able
    to pick back up from the real checkpoint the completed candidate wrote.
    """
    holdout_index = _seed_work_dir(tmp_path)
    engine = _engine(budget=10.0)
    token = CancellationToken()
    trainer = FakeTrainer(tmp_path, cancelling_ids=("interrupted",), cancellation=token)
    evaluator = FakeEvaluator(
        holdout_index, {"completed": 0.90, "interrupted": 0.90, "skipped": 0.90, "resumed": 0.95}
    )
    runner = _runner(tmp_path, engine, trainer, evaluator, cancellation=token)

    checkpointing = {"backend": {"training": {"max_steps": 4, "save_strategy": "steps", "save_steps": 2}}}
    interrupted_round = (
        _experiment("completed", checkpointing, hours=0.2),
        _experiment("interrupted", checkpointing, hours=0.2),
        _experiment("skipped", checkpointing, hours=0.2),
    )
    assert engine.propose(interrupted_round) == interrupted_round
    # promote=False: an interrupted search must not crown a winner from a
    # round it never finished.
    outcome = runner.run_round(interrupted_round, promote=False)

    assert outcome.promoted is None
    assert engine.baseline.experiment_id == "baseline"
    errors = {c.experiment_id: c.error for c in outcome.candidates}
    assert errors["completed"] is None
    assert errors["interrupted"].startswith("cancelled: ")
    assert errors["skipped"] == "cancelled before start"
    # The skipped candidate never reached the trainer at all.
    assert [c["experiment_id"] for c in trainer.calls] == ["completed", "interrupted"]
    # Cancelling is not an anomaly: no investigation budget was spent on it.
    assert next(c for c in outcome.candidates if c.experiment_id == "interrupted").executor_analysis is None

    # completed: really trained + evaluated. interrupted: charged its full
    # reservation (0.2 + the declared eval reserve), conservatively, because
    # the worker's own spend is unknowable after a kill. skipped: nothing.
    interrupted_charge = 0.2 + _EVAL_RESERVE_GPU_HOURS
    _assert_ledger_balances(engine, spent=_CANDIDATE_COST + interrupted_charge)

    # -- resume: a fresh token, and a real child continuing the real
    # checkpoint the completed candidate actually wrote to disk --
    completed_artifact = next(
        c.artifact for c in outcome.candidates if c.experiment_id == "completed"
    )
    checkpoint = Path(completed_artifact.artifact_ref) / "trainer" / "checkpoint-4"
    assert checkpoint.is_dir()

    runner.cancellation = CancellationToken()
    resumed = _experiment(
        "resumed",
        {
            "backend": {
                "resume_from_checkpoint": str(checkpoint),
                "training": {"max_steps": 8, "save_strategy": "steps", "save_steps": 4},
            }
        },
        hours=0.2,
        parent="completed",
    )
    assert engine.propose((resumed,)) == (resumed,)
    resumed_outcome = runner.run_round((resumed,), promote=True)

    resume_call = next(c for c in trainer.calls if c["experiment_id"] == "resumed")
    assert resume_call["resume_from_checkpoint"] == str(checkpoint)
    assert resume_call["max_steps"] == 8
    assert resumed_outcome.promoted is not None
    assert resumed_outcome.promoted.experiment_id == "resumed"
    assert engine.baseline is resumed_outcome.promoted
    _assert_ledger_balances(
        engine, spent=2 * _CANDIDATE_COST + interrupted_charge
    )


# --- 5. immutable provenance --------------------------------------------------


def test_recorded_evidence_is_immutable_and_its_hashes_still_verify_afterwards(tmp_path):
    """Everything the loop records is durable scientific evidence: a later
    stage of the same loop -- including a repair that gets promoted over the
    candidate it repaired -- must never rewrite what an earlier stage wrote,
    and a digest recorded at training time must still verify at the end.
    """
    with RunRegistry(tmp_path / "runs.db") as registry:
        holdout_index = _seed_work_dir(tmp_path)
        engine = _engine(budget=4.0, max_parallel=4)
        trainer = FakeTrainer(tmp_path)
        evaluator = FakeEvaluator(holdout_index, {"source": 0.65}, repair_score=0.86)
        runner = _runner(tmp_path, engine, trainer, evaluator, registry=registry)

        source = _experiment("source", hours=0.5)
        registry.record_experiment(source)
        engine.propose((source,))
        source_generation = runner.run_generation((source,))
        assert source_generation.promoted is None

        # Snapshot every durable row the rejected candidate produced.
        def _snapshot():
            return {
                "training": [
                    (a.run_id, a.experiment_id, a.artifact_ref, a.gpu_hours, dict(a.evidence))
                    for a in registry.list_training_artifacts()
                ],
                "evaluation": [
                    (e.run_id, e.experiment_id, dict(e.metrics), e.gpu_hours, dict(e.evidence))
                    for e in registry.list_evaluation_outcomes()
                ],
                "results": [
                    (r.experiment_id, dict(r.metrics), r.gpu_hours, r.artifact_ref, dict(r.evidence))
                    for r in registry.list_results()
                ],
                "failures": [f.failure_id for f in registry.list_failures()],
            }

        before = _snapshot()
        assert len(before["training"]) == 1
        source_artifact = source_generation.candidates[0].artifact
        recorded_sha = before["training"][0][4]["artifact_sha256"]
        assert recorded_sha == source_artifact.evidence["artifact_sha256"]

        repair_outcome = run_single_hop_autonomous_repair(
            runner=runner,
            source_generation=source_generation,
            provider=_repair_provider(tmp_path),
            variants=(
                RepairVariant("lr-low", 0.3, training_patch={"learning_rate": 5e-5}),
                RepairVariant("epochs-low", 0.3, training_patch={"epochs": 1}),
            ),
        )
        assert repair_outcome.promoted is not None
        assert engine.baseline.experiment_id.startswith("repair-")

        after = _snapshot()
        # Promotion of a repair appended rows; it rewrote none of the
        # rejected candidate's own evidence.
        for table in ("training", "evaluation", "results"):
            assert after[table][: len(before[table])] == before[table]
            assert len(after[table]) > len(before[table])
        assert after["failures"] == before["failures"]
        # The rejected candidate stays rejected even though its repair won.
        statuses = {e.experiment_id: e.status for e in registry.list_experiments()}
        assert statuses["source"] is ExperimentStatus.REJECTED

        # The digest taken at training time still verifies against the real
        # bytes on disk after the whole loop, and it is exactly the digest
        # the repair bound itself to.
        assert sha256_directory(Path(source_artifact.artifact_ref)) == recorded_sha
        for candidate in repair_outcome.population.proposed_candidates:
            backend = candidate.config_patch["backend"]
            assert backend["parent_adapter"]["sha256"] == recorded_sha
            replay = backend["replay"]
            assert replay["sha256"] == sha256_file(Path(replay["dataset"]))
            assert replay["manifest_sha256"] == sha256_file(Path(replay["manifest"]))

        # Cross-table consistency: the protocol digest the gate saw in the
        # promoted result is the one the evaluation run actually recorded.
        promoted_row = next(
            r for r in registry.list_results()
            if r.experiment_id == repair_outcome.promoted.experiment_id
        )
        promoted_evaluation = next(
            e for e in registry.list_evaluation_outcomes()
            if e.experiment_id == repair_outcome.promoted.experiment_id
        )
        assert (
            promoted_row.evidence["evaluation_protocol_sha256"]
            == promoted_evaluation.evidence["protocol_sha256"]
            == _PROTOCOL_SHA
        )

        # A divergent replay of an already-recorded run is refused outright;
        # an identical replay is idempotent.
        original = next(
            a for a in registry.list_training_artifacts() if a.experiment_id == "source"
        )
        with pytest.raises(RegistryInvariantError, match="already exists with different content"):
            registry.record_training_artifact(
                replace(original, gpu_hours=original.gpu_hours + 1.0)
            )
        registry.record_training_artifact(original)
        assert len(tuple(registry.list_training_artifacts())) == len(after["training"])

        original_result = next(
            r for r in registry.list_results() if r.experiment_id == "source"
        )
        with pytest.raises(RegistryInvariantError, match="already exists with different content"):
            registry.record_result(replace(original_result, metrics={"quality": 0.99}))
        registry.record_result(original_result)
        assert len(tuple(registry.list_results())) == len(after["results"])

        _assert_ledger_balances(engine, spent=3 * _CANDIDATE_COST)
        assert engine.spent_gpu_hours == pytest.approx(
            sum(result.gpu_hours for result in registry.list_results())
        )


# --- 6. a real bug this integration test found --------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "REAL BUG (production, not test): run_successive_halving() creates the "
        "round-1+ child experiments itself and proposes them to the engine, but "
        "never records them in the RunRegistry. ExperimentCycleRunner._record_status"
        "() then calls RunRegistry.update_experiment_status(), which raises "
        "RegistryInvariantError('unknown persisted experiment id: <parent>-r1') "
        "outside any try block -- so any successive-halving search with a registry "
        "attached dies at the start of round 1, leaving every round-1 reservation "
        "outstanding and unsettled (a permanently leaked GPU-hour reservation). "
        "Neither module's own tests catch it: test_successive_halving.py never "
        "attaches a registry, and test_cycle.py never runs a multi-round search. "
        "Remove this marker once the fix lands."
    ),
)
def test_successive_halving_persists_the_child_experiments_it_creates(tmp_path):
    holdout_index = _seed_work_dir(tmp_path)
    with RunRegistry(tmp_path / "runs.db") as registry:
        engine = _engine(budget=10.0)
        runner = _runner(
            tmp_path,
            engine,
            FakeTrainer(tmp_path),
            FakeEvaluator(
                holdout_index,
                {"lr-tune": 0.90, "warmup": 0.85, "decay": 0.80, "lora-rank": 0.55},
            ),
            registry=registry,
        )
        pool = (
            _experiment("lr-tune", {"backend": {"training": {"learning_rate": 5e-5}}}),
            _experiment("warmup", {"backend": {"training": {"warmup_ratio": 0.1}}}),
            _experiment("decay", {"backend": {"training": {"weight_decay": 0.01}}}),
            _experiment("lora-rank", {"backend": {"lora": {"r": 8}}}),
        )
        for experiment in pool:
            registry.record_experiment(experiment)

        outcome = run_successive_halving(
            runner, pool, initial_max_steps=2, survival_fraction=0.5, min_survivors=1
        )

        assert outcome.total_rounds == 2
        assert outcome.promoted is not None
        assert outcome.promoted.experiment_id == "lr-tune-r1"
        # The child a real round actually ran must be durably recorded, with
        # its real lineage back to the candidate it continued.
        assert registry.has_experiment("lr-tune-r1")
        assert registry.lineage("lr-tune-r1") == ("lr-tune",)
        _assert_ledger_balances(engine, spent=6 * _CANDIDATE_COST)
