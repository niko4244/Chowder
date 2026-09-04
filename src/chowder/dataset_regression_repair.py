"""Targeted repair for training-data-induced regressions.

Bridges the diagnostic pipeline (`dataset_influence.py`'s
`TrainingExampleInfluence`, clustered by `training_sample_clusters.py`'s
`TrainingSampleCluster`) to the existing, already-hardened repair machinery in
`repair_orchestrator.py` / `repair_candidates.py` / `contamination.py` --
reuses all of it (contamination auditing, provider boundary, replay,
gate-based promotion) rather than rebuilding a parallel system.

Distinct from `autonomous_repair.py`'s eval-gate-failure repair path: there
the "parent" being repaired is a gate-REJECTED candidate, and continuation
resumes from its own (rejected) adapter weights. Here the "parent" is the
ORIGINAL run whose checkpoint sequence a regression was bisected out of
(`checkpoint_bisect.py`'s `CheckpointBisectOutcome`), and continuation must
resume from the LAST-GOOD checkpoint -- resuming from the regressed (bad)
checkpoint would just continue training on top of the very state being
repaired.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from .cycle import ExperimentCycleRunner, GenerationOutcome
from .failures import RepairPlan
from .provenance import sha256_directory
from .repair_candidates import RepairVariant, VerifiedParentAdapter, VerifiedReplayDataset
from .repair_orchestrator import prepare_and_propose_repair_population
from .repair_requests import RepairRequest, RepairSourceProvider, strategy_for_failure_kind
from .training_sample_clusters import TrainingSampleCluster

TRAINING_REGRESSION_FAILURE_KIND = "training_example_regression"
TRAINING_REGRESSION_EVALUATOR = "dataset-influence"
TRAINING_REGRESSION_SUITE = "training-corpus"


def _canonical_digest(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _regression_identity(
    cluster: TrainingSampleCluster, *, good_checkpoint_dir: str, bad_checkpoint_dir: str
) -> dict[str, object]:
    if not good_checkpoint_dir.strip() or not bad_checkpoint_dir.strip():
        raise ValueError("both checkpoint directories are required")
    if good_checkpoint_dir == bad_checkpoint_dir:
        raise ValueError("good and bad checkpoint directories must differ")
    return {
        "cluster_id": cluster.cluster_id,
        "member_row_indices": list(cluster.member_row_indices),
        "good_checkpoint_dir": good_checkpoint_dir,
        "bad_checkpoint_dir": bad_checkpoint_dir,
    }


def build_training_regression_repair_request(
    cluster: TrainingSampleCluster, *, good_checkpoint_dir: str, bad_checkpoint_dir: str
) -> RepairRequest:
    """Leak-resistant repair request for a dataset-influence-diagnosed regression.

    Reuses ``RepairRequest`` exactly (same type, same downstream provider/
    contamination machinery as the eval-gate repair path) but is built from a
    ``TrainingSampleCluster`` instead of a ``FailureCluster``: there is no
    real eval "evaluator/suite" for a training-corpus regression, so those
    fields carry honest sentinel values naming the diagnostic source instead.
    """
    identity = _regression_identity(
        cluster, good_checkpoint_dir=good_checkpoint_dir, bad_checkpoint_dir=bad_checkpoint_dir
    )
    protocol_sha256 = _canonical_digest(identity)
    strategy = strategy_for_failure_kind(TRAINING_REGRESSION_FAILURE_KIND)
    return RepairRequest(
        request_id=f"training-regression-repair-request-{protocol_sha256[:16]}",
        plan_id=_canonical_digest({**identity, "role": "plan"}),
        cluster_id=cluster.cluster_id,
        evaluator=TRAINING_REGRESSION_EVALUATOR,
        suite=TRAINING_REGRESSION_SUITE,
        failure_kind=TRAINING_REGRESSION_FAILURE_KIND,
        strategy=strategy,
        failure_count=len(cluster.member_row_indices),
        protocol_sha256=protocol_sha256,
    )


def build_training_regression_repair_plan(
    cluster: TrainingSampleCluster, *, good_checkpoint_dir: str, bad_checkpoint_dir: str
) -> RepairPlan:
    """RepairPlan for a dataset-influence-diagnosed regression.

    ``direct_training_allowed`` is always False and
    ``requires_independent_source`` always True: the whole point of this
    diagnostic is that these exact training rows measurably worsened the
    model, so retraining directly on them again is never the repair -- only
    independently sourced counterexamples covering the same topic are.
    """
    identity = _regression_identity(
        cluster, good_checkpoint_dir=good_checkpoint_dir, bad_checkpoint_dir=bad_checkpoint_dir
    )
    plan_id = _canonical_digest({**identity, "role": "plan"})
    return RepairPlan(
        plan_id=plan_id,
        cluster_id=cluster.cluster_id,
        observation=(
            f"{len(cluster.member_row_indices)} training examples in cluster "
            f"{cluster.cluster_id[:12]} show measurably worse loss between "
            f"{good_checkpoint_dir} and {bad_checkpoint_dir} "
            f"(mean influence {cluster.mean_influence_score:.4f}, "
            f"max {cluster.max_influence_score:.4f})"
        ),
        suspected_cause=(
            "training-data-induced regression: these training rows measurably "
            "worsened held checkpoint loss between the last-good and "
            "first-regressing checkpoint"
        ),
        intervention=(
            "add independently sourced counterexamples covering this cluster's "
            "topic; never retrain directly on the offending training rows or "
            "copy holdout prompts/answers into training data"
        ),
        source_failure_ids=tuple(str(index) for index in cluster.member_row_indices),
        direct_training_allowed=False,
        requires_independent_source=True,
    )


def verified_last_good_checkpoint_adapter(good_checkpoint_dir: str | Path) -> VerifiedParentAdapter:
    """Bind repair continuation to the exact last-good checkpoint's real weights.

    Deliberately separate from `autonomous_repair.py`'s
    ``_verified_parent_adapter``, which binds to a gate-REJECTED candidate's
    own final artifact and reads a pre-declared SHA-256 from its evaluation
    evidence. Here the continuation source is a checkpoint mid-run, with no
    ``ExperimentResult``/evaluation evidence of its own -- the directory's
    real content hash is computed directly instead.
    """
    path = Path(good_checkpoint_dir).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"last-good checkpoint not found: {path}")
    parent = VerifiedParentAdapter(str(path), sha256_directory(path))
    parent.verify()
    return parent


def run_training_regression_repair(
    *,
    runner: ExperimentCycleRunner,
    parent_id: str,
    cluster: TrainingSampleCluster,
    good_checkpoint_dir: str,
    bad_checkpoint_dir: str,
    provider: RepairSourceProvider,
    holdout_fingerprint_files: Iterable[str | Path],
    variants: tuple[RepairVariant, ...],
    continue_from_last_good: bool = True,
    replay: VerifiedReplayDataset | None = None,
) -> GenerationOutcome:
    """Repair a training-data-induced regression via independent counterexamples.

    Generates independent counterexamples for the diagnosed cluster (through
    the same leak-resistant provider boundary and holdout-contamination audit
    the eval-gate repair path uses) and trains a repair population that
    resumes from the LAST-GOOD checkpoint -- never the regressed one.

    ``continue_from_last_good=False`` builds a fresh-start repair population
    instead (no parent adapter; variants may set ``lora_patch``), matching
    ``autonomous_repair.run_single_hop_autonomous_repair``'s own
    ``continue_from_parent`` design.

    The repair population is gated by the real, unmodified promotion gate
    (via ``runner.run_generation``) -- exactly like every other candidate in
    this codebase. There is no separate "did this fix the target regression"
    check: a repair that does not clear the gate is rejected and
    ``GenerationOutcome.promoted`` is ``None``, the same honest outcome any
    other failed candidate produces.
    """
    if parent_id not in runner.engine.graph.nodes:
        raise ValueError("repair parent must already exist in the experiment graph")
    if not variants:
        raise ValueError("training-regression repair requires at least one repair variant")
    if continue_from_last_good and any(variant.lora_patch for variant in variants):
        raise ValueError("continuation repair cannot change LoRA topology")

    plan = build_training_regression_repair_plan(
        cluster, good_checkpoint_dir=good_checkpoint_dir, bad_checkpoint_dir=bad_checkpoint_dir
    )
    request = build_training_regression_repair_request(
        cluster, good_checkpoint_dir=good_checkpoint_dir, bad_checkpoint_dir=bad_checkpoint_dir
    )
    parent_adapter = (
        verified_last_good_checkpoint_adapter(good_checkpoint_dir)
        if continue_from_last_good
        else None
    )

    population = prepare_and_propose_repair_population(
        engine=runner.engine,
        parent_id=parent_id,
        plan=plan,
        request=request,
        provider=provider,
        holdout_fingerprint_files=holdout_fingerprint_files,
        variants=variants,
        work_dir=runner.context.work_dir,
        registry=runner.registry,
        replay=replay,
        parent_adapter=parent_adapter,
    )
    if not population.proposed_candidates:
        raise ValueError(
            "training-regression repair population produced no budget-admissible candidates"
        )
    return runner.run_generation(population.proposed_candidates)
