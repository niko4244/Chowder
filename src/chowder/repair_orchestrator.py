from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .engine import EvolutionEngine
from .failures import FailureCluster, RepairPlan
from .models import Experiment
from .registry import RunRegistry
from .repair_candidates import (
    RepairVariant,
    VerifiedRepairDataset,
    build_autonomous_repair_population,
)
from .repair_requests import (
    MaterializedRepairProposal,
    RepairRequest,
    RepairSourceProposal,
    RepairSourceProvider,
    build_repair_request,
    materialize_repair_proposal,
    request_repair_sources,
)


@dataclass(frozen=True)
class RepairPopulationOutcome:
    request: RepairRequest
    proposal: RepairSourceProposal
    materialized: MaterializedRepairProposal
    generated_candidates: tuple[Experiment, ...]
    proposed_candidates: tuple[Experiment, ...]
    repair_dir: str

    @property
    def deferred_candidates(self) -> tuple[Experiment, ...]:
        accepted = {candidate.experiment_id for candidate in self.proposed_candidates}
        return tuple(
            candidate
            for candidate in self.generated_candidates
            if candidate.experiment_id not in accepted
        )


def _provider_slug(provider: RepairSourceProvider) -> str:
    raw = f"{provider.name}\x00{provider.version}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def prepare_and_propose_repair_population(
    *,
    engine: EvolutionEngine,
    parent_id: str,
    plan: RepairPlan,
    cluster: FailureCluster,
    provider: RepairSourceProvider,
    holdout_fingerprint_files: Iterable[str | Path],
    variants: Iterable[RepairVariant],
    work_dir: str | Path,
    registry: RunRegistry | None = None,
) -> RepairPopulationOutcome:
    """Prepare provenance-safe repair data and reserve candidate experiments.

    This function intentionally stops before training. Candidate admission and
    compute reservation are delegated to ``EvolutionEngine.propose`` so repair
    automation cannot bypass parallelism or GPU-hour budgets.
    """

    if parent_id not in engine.graph.nodes:
        raise ValueError("repair parent must already exist in the experiment graph")
    if engine.outstanding_candidates >= engine.goal.max_parallel_candidates:
        raise ValueError("no parallel candidate slots are available for repair")

    variant_rows = tuple(variants)
    if not variant_rows:
        raise ValueError("at least one repair variant is required")
    if not any(variant.estimated_gpu_hours <= engine.remaining_budget for variant in variant_rows):
        raise ValueError("no repair variant fits the remaining GPU-hour budget")

    holdout_files = tuple(Path(path).resolve() for path in holdout_fingerprint_files)
    if not holdout_files:
        raise ValueError("repair orchestration requires holdout fingerprint indexes")
    if any(not path.is_file() for path in holdout_files):
        missing = [str(path) for path in holdout_files if not path.is_file()]
        raise FileNotFoundError(f"holdout fingerprint indexes not found: {missing}")

    request = build_repair_request(plan=plan, cluster=cluster)
    proposal = request_repair_sources(provider=provider, request=request)

    root = Path(work_dir).resolve() / ".chowder" / "repairs"
    repair_dir = root / f"{request.request_id}-{_provider_slug(provider)}"
    repair_dir.mkdir(parents=True, exist_ok=False)
    dataset_path = repair_dir / "repair.jsonl"
    manifest_path = repair_dir / "sources.json"

    materialized = materialize_repair_proposal(
        request=request,
        proposal=proposal,
        provider=provider,
        holdout_fingerprint_files=holdout_files,
        dataset_path=dataset_path,
        source_manifest_path=manifest_path,
    )
    verified = VerifiedRepairDataset(
        path=materialized.dataset_path,
        sha256=materialized.dataset_sha256,
        contamination_audit=materialized.contamination_audit,
        source_manifest_path=materialized.source_manifest_path,
        source_manifest_sha256=materialized.source_manifest_sha256,
    )
    generated = build_autonomous_repair_population(
        parent_id=parent_id,
        plan=plan,
        dataset=verified,
        variants=variant_rows,
    )
    proposed = engine.propose(generated)

    if registry is not None:
        for experiment in proposed:
            registry.record_experiment(experiment)

    return RepairPopulationOutcome(
        request=request,
        proposal=proposal,
        materialized=materialized,
        generated_candidates=generated,
        proposed_candidates=proposed,
        repair_dir=str(repair_dir),
    )
