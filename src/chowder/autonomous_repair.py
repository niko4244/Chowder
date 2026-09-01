from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .cycle import CandidateCycleOutcome, ExperimentCycleRunner, GenerationOutcome
from .failures import FailureCluster, RepairPlan, cluster_failures
from .provenance import sha256_directory, sha256_file
from .repair_candidates import (
    RepairVariant,
    VerifiedParentAdapter,
    VerifiedReplayDataset,
)
from .repair_orchestrator import (
    RepairPopulationOutcome,
    prepare_and_propose_repair_population,
)
from .repair_requests import RepairSourceProvider
from .replay_history import ReplayHistorySource, materialize_replay_history


@dataclass(frozen=True)
class RepairTarget:
    candidate: CandidateCycleOutcome
    cluster: FailureCluster
    plan: RepairPlan


@dataclass(frozen=True)
class AutonomousRepairOutcome:
    source_generation: GenerationOutcome
    target: RepairTarget
    population: RepairPopulationOutcome
    repair_generation: GenerationOutcome

    @property
    def promoted(self):
        return self.repair_generation.promoted


def _ranked_rejections(generation: GenerationOutcome) -> dict[str, object]:
    return {
        ranked.result.experiment_id: ranked
        for ranked in generation.ranking
        if not ranked.decision.accepted
    }


def _candidate_by_id(
    generation: GenerationOutcome,
) -> dict[str, CandidateCycleOutcome]:
    return {candidate.experiment_id: candidate for candidate in generation.candidates}


def _repairable_target(
    generation: GenerationOutcome,
    *,
    candidate_id: str | None = None,
) -> RepairTarget:
    rejected = _ranked_rejections(generation)
    candidates = _candidate_by_id(generation)

    if candidate_id is not None:
        if candidate_id not in rejected:
            raise ValueError("requested repair candidate was not gate-rejected")
        candidate_ids = (candidate_id,)
    else:
        candidate_ids = tuple(
            ranked.result.experiment_id
            for ranked in generation.ranking
            if not ranked.decision.accepted
        )

    for experiment_id in candidate_ids:
        candidate = candidates.get(experiment_id)
        if candidate is None:
            raise ValueError("ranked candidate is missing from generation outcomes")
        if (
            candidate.error is not None
            or candidate.result is None
            or candidate.evaluation is None
        ):
            continue
        if candidate.diagnostic_error is not None:
            continue
        if not candidate.harvested_failures or not candidate.repair_plans:
            continue

        clusters = {
            cluster.cluster_id: cluster
            for cluster in cluster_failures(candidate.harvested_failures)
        }
        repairable: list[tuple[int, str, FailureCluster, RepairPlan]] = []
        for plan in candidate.repair_plans:
            if not plan.requires_independent_source:
                continue
            cluster = clusters.get(plan.cluster_id)
            if cluster is None:
                raise ValueError(
                    "repair plan cluster is missing from harvested failure evidence"
                )
            if tuple(sorted(plan.source_failure_ids)) != tuple(
                sorted(cluster.failure_ids)
            ):
                raise ValueError(
                    "repair plan failure lineage does not match harvested cluster"
                )
            repairable.append((len(cluster.failure_ids), plan.plan_id, cluster, plan))

        if repairable:
            repairable.sort(key=lambda row: (-row[0], row[1]))
            _, _, cluster, plan = repairable[0]
            return RepairTarget(candidate=candidate, cluster=cluster, plan=plan)

    if candidate_id is not None:
        raise ValueError(
            "requested rejected candidate has no independently repairable diagnostics"
        )
    raise ValueError("generation has no independently repairable rejected candidate")


def _verified_holdout_files(candidate: CandidateCycleOutcome) -> tuple[Path, ...]:
    evaluation = candidate.evaluation
    if evaluation is None:
        raise ValueError("repair target is missing evaluation evidence")
    evidence = evaluation.evidence
    suite_evidence = evidence.get("suite_evidence")
    declared_hashes = evidence.get("holdout_fingerprint_sha256")
    if not isinstance(suite_evidence, Mapping) or not isinstance(
        declared_hashes, Mapping
    ):
        raise ValueError("evaluation is missing holdout fingerprint evidence")

    files: list[Path] = []
    seen: set[Path] = set()
    if set(suite_evidence) != set(declared_hashes):
        raise ValueError("holdout fingerprint suite evidence is inconsistent")

    for suite_name in sorted(suite_evidence):
        suite = suite_evidence[suite_name]
        declared = declared_hashes[suite_name]
        if not isinstance(suite, Mapping):
            raise ValueError(f"suite evidence for {suite_name!r} is invalid")
        path_raw = suite.get("holdout_fingerprints_file")
        suite_declared = suite.get("holdout_fingerprints_sha256")
        if (
            not isinstance(path_raw, str)
            or not isinstance(declared, str)
            or not isinstance(suite_declared, str)
        ):
            raise ValueError(
                f"suite {suite_name!r} is missing fingerprint path/digest"
            )
        if declared != suite_declared:
            raise ValueError(
                f"suite {suite_name!r} fingerprint digest declarations disagree"
            )
        path = Path(path_raw).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"holdout fingerprint file not found: {path}")
        actual = sha256_file(path)
        if actual != declared:
            raise ValueError(
                f"suite {suite_name!r} holdout fingerprint digest changed"
            )
        if path not in seen:
            files.append(path)
            seen.add(path)
    if not files:
        raise ValueError("evaluation contains no holdout fingerprint files")
    return tuple(files)


def _resolved_path(raw: str, work_dir: str | Path) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = Path(work_dir) / path
    return path.resolve()


def _verified_parent_replay(
    *,
    runner: ExperimentCycleRunner,
    target: RepairTarget,
    ratio: float,
) -> VerifiedReplayDataset:
    """Materialize cumulative rehearsal from all history available to the parent.

    If the parent already used replay, that verified history is carried forward
    first. The parent's primary training dataset is then appended and exact-text
    deduplicated. The result is normalized to a canonical `text` JSONL corpus.
    """

    artifact = target.candidate.artifact
    if artifact is None:
        raise ValueError("repair target has no training artifact for replay provenance")
    recorded_primary_sha = artifact.evidence.get("dataset_sha256")
    if not isinstance(recorded_primary_sha, str) or len(recorded_primary_sha) != 64:
        raise ValueError(
            "repair target training artifact is missing dataset SHA-256 provenance"
        )

    resolved = runner.engine.resolve_config(
        target.candidate.experiment_id, runner.base_config
    )
    backend = resolved.get("backend")
    if not isinstance(backend, Mapping):
        raise ValueError("repair target resolved config has no backend mapping")

    dataset_raw = backend.get("dataset")
    if not isinstance(dataset_raw, str) or not dataset_raw.strip():
        raise ValueError("repair target resolved config has no training dataset")
    dataset_path = _resolved_path(dataset_raw, runner.context.work_dir)
    declared_primary_sha = backend.get("dataset_sha256")
    if declared_primary_sha is not None and declared_primary_sha != recorded_primary_sha:
        raise ValueError("parent training dataset provenance disagrees with resolved config")

    text_field = str(backend.get("text_field", "text")).strip()
    if not text_field:
        raise ValueError("parent backend text_field is empty")

    sources: list[ReplayHistorySource] = []
    replay_config = backend.get("replay")
    if replay_config is not None:
        if not isinstance(replay_config, Mapping):
            raise ValueError("parent backend replay section is invalid")
        replay_raw = replay_config.get("dataset")
        replay_sha = replay_config.get("sha256")
        if replay_raw is not None or replay_sha is not None:
            if not isinstance(replay_raw, str) or not isinstance(replay_sha, str):
                raise ValueError("parent replay history is missing dataset/SHA binding")
            recorded_replay_sha = artifact.evidence.get("replay_dataset_sha256")
            if not isinstance(recorded_replay_sha, str) or recorded_replay_sha != replay_sha:
                raise ValueError(
                    "parent training artifact does not prove the configured replay history"
                )
            prior_manifest = replay_config.get("manifest")
            prior_manifest_sha = replay_config.get("manifest_sha256")
            prior = VerifiedReplayDataset(
                path=str(_resolved_path(replay_raw, runner.context.work_dir)),
                sha256=replay_sha,
                ratio=float(replay_config.get("ratio", 1.0)),
                manifest_path=(
                    str(_resolved_path(prior_manifest, runner.context.work_dir))
                    if isinstance(prior_manifest, str)
                    else None
                ),
                manifest_sha256=(
                    str(prior_manifest_sha)
                    if prior_manifest_sha is not None
                    else None
                ),
            )
            prior_path = prior.verify()
            sources.append(
                ReplayHistorySource(
                    path=str(prior_path),
                    sha256=prior.sha256,
                    text_field=text_field,
                    role="prior_replay_history",
                )
            )

    sources.append(
        ReplayHistorySource(
            path=str(dataset_path),
            sha256=recorded_primary_sha,
            text_field=text_field,
            role="parent_primary_training",
        )
    )
    return materialize_replay_history(
        sources=sources,
        work_dir=runner.context.work_dir,
        ratio=float(ratio),
    )


def _verified_parent_adapter(target: RepairTarget) -> VerifiedParentAdapter:
    artifact = target.candidate.artifact
    if artifact is None:
        raise ValueError("repair target has no training artifact to continue")
    artifact_sha = artifact.evidence.get("artifact_sha256")
    if not isinstance(artifact_sha, str) or len(artifact_sha) != 64:
        raise ValueError(
            "repair target training artifact is missing adapter SHA-256 provenance"
        )
    adapter_path = Path(artifact.artifact_ref).resolve()
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"repair target adapter not found: {adapter_path}")
    actual = sha256_directory(adapter_path)
    if actual != artifact_sha:
        raise ValueError("parent adapter content changed after evaluation")
    parent = VerifiedParentAdapter(str(adapter_path), artifact_sha)
    parent.verify()
    return parent


def run_single_hop_autonomous_repair(
    *,
    runner: ExperimentCycleRunner,
    source_generation: GenerationOutcome,
    provider: RepairSourceProvider,
    variants: tuple[RepairVariant, ...],
    candidate_id: str | None = None,
    replay_ratio: float | None = 1.0,
) -> AutonomousRepairOutcome:
    """Repair one rejected adapter with cumulative, provenance-bound rehearsal."""

    if not variants:
        raise ValueError("autonomous repair requires at least one repair variant")
    if any(variant.lora_patch for variant in variants):
        raise ValueError(
            "autonomous continuation repair cannot change LoRA topology"
        )

    target = _repairable_target(source_generation, candidate_id=candidate_id)
    node = runner.engine.graph.nodes.get(target.candidate.experiment_id)
    if node is None:
        raise ValueError("repair target is missing from experiment graph")
    if node.status.value != "rejected":
        raise ValueError("repair target graph status is not rejected")

    holdout_files = _verified_holdout_files(target.candidate)
    replay = (
        _verified_parent_replay(
            runner=runner, target=target, ratio=float(replay_ratio)
        )
        if replay_ratio is not None
        else None
    )
    parent_adapter = _verified_parent_adapter(target)

    population = prepare_and_propose_repair_population(
        engine=runner.engine,
        parent_id=target.candidate.experiment_id,
        plan=target.plan,
        cluster=target.cluster,
        provider=provider,
        holdout_fingerprint_files=holdout_files,
        variants=variants,
        work_dir=runner.context.work_dir,
        registry=runner.registry,
        replay=replay,
        parent_adapter=parent_adapter,
    )
    if not population.proposed_candidates:
        raise ValueError("repair population produced no budget-admissible candidates")

    repair_generation = runner.run_generation(population.proposed_candidates)
    return AutonomousRepairOutcome(
        source_generation=source_generation,
        target=target,
        population=population,
        repair_generation=repair_generation,
    )
