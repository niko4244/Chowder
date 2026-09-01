from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol, runtime_checkable

from .contamination import ContaminationAudit
from .failures import FailureCluster, RepairPlan
from .repair_sources import (
    RepairSource,
    SourcedRepairExample,
    write_provenanced_repair_dataset,
)


class RepairStrategy(str, Enum):
    CONCISE_ANSWER = "concise_answer"
    CALIBRATED_ANSWERING = "calibrated_answering"
    FORMAT_CONTROL = "format_control"
    NEAR_NEIGHBOR_REASONING = "near_neighbor_reasoning"


def strategy_for_failure_kind(failure_kind: str) -> RepairStrategy:
    if failure_kind == "empty_prediction":
        return RepairStrategy.CONCISE_ANSWER
    if failure_kind == "refusal_or_unknown":
        return RepairStrategy.CALIBRATED_ANSWERING
    if failure_kind == "overlong_mismatch":
        return RepairStrategy.FORMAT_CONTROL
    return RepairStrategy.NEAR_NEIGHBOR_REASONING


@dataclass(frozen=True)
class RepairRequest:
    """Aggregate repair intent that deliberately excludes raw evaluation rows."""

    request_id: str
    plan_id: str
    cluster_id: str
    evaluator: str
    suite: str
    failure_kind: str
    strategy: RepairStrategy
    failure_count: int
    protocol_sha256: str
    source_failure_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.failure_count <= 0:
            raise ValueError("repair request failure_count must be positive")
        if self.failure_count != len(self.source_failure_ids):
            raise ValueError("repair request failure_count does not match failure IDs")
        if len(self.protocol_sha256) != 64:
            raise ValueError("repair request protocol_sha256 is invalid")
        if not all(len(failure_id) == 64 for failure_id in self.source_failure_ids):
            raise ValueError("repair request failure IDs must be SHA-256 digests")
        if self.strategy is not strategy_for_failure_kind(self.failure_kind):
            raise ValueError("repair request strategy does not match failure kind")

    def to_provider_payload(self) -> dict[str, object]:
        """Return the complete provider-visible payload.

        Raw prompts, expected answers, model predictions, row indexes, and
        free-form plan prose are not members of this type and cannot cross this
        provider boundary through the canonical serializer.
        """

        return {
            "request_id": self.request_id,
            "plan_id": self.plan_id,
            "cluster_id": self.cluster_id,
            "evaluator": self.evaluator,
            "suite": self.suite,
            "failure_kind": self.failure_kind,
            "strategy": self.strategy.value,
            "failure_count": self.failure_count,
            "protocol_sha256": self.protocol_sha256,
            "source_failure_ids": list(self.source_failure_ids),
        }


@dataclass(frozen=True)
class RepairSourceProposal:
    request_id: str
    provider_name: str
    provider_version: str
    sources: tuple[RepairSource, ...]
    examples: tuple[SourcedRepairExample, ...]

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise ValueError("repair proposal request_id is required")
        if not self.provider_name.strip():
            raise ValueError("repair proposal provider_name is required")
        if not self.provider_version.strip():
            raise ValueError("repair proposal provider_version is required")
        if not self.sources:
            raise ValueError("repair proposal contains no sources")
        if not self.examples:
            raise ValueError("repair proposal contains no examples")


@dataclass(frozen=True)
class MaterializedRepairProposal:
    request_id: str
    provider_name: str
    provider_version: str
    dataset_path: str
    dataset_sha256: str
    contamination_audit: ContaminationAudit
    source_manifest_path: str
    source_manifest_sha256: str


@runtime_checkable
class RepairSourceProvider(Protocol):
    """Provider sees only a leak-resistant RepairRequest, never FailureRecord."""

    name: str
    version: str

    def propose(self, request: RepairRequest) -> RepairSourceProposal:
        ...


def _canonical_digest(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_repair_request(*, plan: RepairPlan, cluster: FailureCluster) -> RepairRequest:
    if plan.cluster_id != cluster.cluster_id:
        raise ValueError("repair plan does not belong to failure cluster")
    plan_ids = tuple(sorted(plan.source_failure_ids))
    cluster_ids = tuple(sorted(cluster.failure_ids))
    if plan_ids != cluster_ids:
        raise ValueError("repair plan failure lineage does not match failure cluster")
    if not plan.requires_independent_source:
        raise ValueError("repair request is only for plans requiring independent source material")

    strategy = strategy_for_failure_kind(cluster.failure_kind)
    identity = {
        "plan_id": plan.plan_id,
        "cluster_id": cluster.cluster_id,
        "evaluator": cluster.evaluator,
        "suite": cluster.suite,
        "failure_kind": cluster.failure_kind,
        "strategy": strategy.value,
        "protocol_sha256": cluster.protocol_sha256,
        "failure_ids": list(cluster_ids),
    }
    return RepairRequest(
        request_id=f"repair-request-{_canonical_digest(identity)[:16]}",
        plan_id=plan.plan_id,
        cluster_id=cluster.cluster_id,
        evaluator=cluster.evaluator,
        suite=cluster.suite,
        failure_kind=cluster.failure_kind,
        strategy=strategy,
        failure_count=len(cluster_ids),
        protocol_sha256=cluster.protocol_sha256,
        source_failure_ids=cluster_ids,
    )


def validate_repair_proposal(
    *,
    request: RepairRequest,
    proposal: RepairSourceProposal,
    provider: RepairSourceProvider | None = None,
) -> None:
    if proposal.request_id != request.request_id:
        raise ValueError("repair proposal belongs to a different request")
    if provider is not None:
        if proposal.provider_name != provider.name or proposal.provider_version != provider.version:
            raise ValueError("repair proposal provider identity does not match provider")

    source_ids = [source.source_id for source in proposal.sources]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("repair proposal contains duplicate source IDs")
    example_ids = [example.example_id for example in proposal.examples]
    if len(example_ids) != len(set(example_ids)):
        raise ValueError("repair proposal contains duplicate example IDs")
    declared_sources = set(source_ids)
    undeclared = sorted(
        {example.source_id for example in proposal.examples if example.source_id not in declared_sources}
    )
    if undeclared:
        raise ValueError(f"repair proposal examples reference undeclared sources: {undeclared}")


def materialize_repair_proposal(
    *,
    request: RepairRequest,
    proposal: RepairSourceProposal,
    holdout_fingerprint_files: tuple[str | Path, ...],
    dataset_path: str | Path,
    source_manifest_path: str | Path,
    provider: RepairSourceProvider | None = None,
) -> MaterializedRepairProposal:
    """Verify provider identity, then run provenance + contamination admission."""

    validate_repair_proposal(request=request, proposal=proposal, provider=provider)
    dataset_sha, audit, manifest_sha = write_provenanced_repair_dataset(
        examples=proposal.examples,
        sources=proposal.sources,
        holdout_fingerprint_files=holdout_fingerprint_files,
        dataset_path=dataset_path,
        manifest_path=source_manifest_path,
    )
    return MaterializedRepairProposal(
        request_id=request.request_id,
        provider_name=proposal.provider_name,
        provider_version=proposal.provider_version,
        dataset_path=str(Path(dataset_path).resolve()),
        dataset_sha256=dataset_sha,
        contamination_audit=audit,
        source_manifest_path=str(Path(source_manifest_path).resolve()),
        source_manifest_sha256=manifest_sha,
    )
