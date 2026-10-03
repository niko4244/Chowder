"""Claims and findings: what the evidence licenses, no more.

A `ResearchFinding` carries typed `Claim`s, not prose. A claim's status
(`provisional` / `replicated` / `rejected` / `superseded`) is decided by
Chowder against the replication policy and the cited observations — never by
the provider that proposed the idea. An unsupported claim is recorded as
unsupported; a one-benchmark delta with transfer failure is recorded as
non-transferable, never as a capability gain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

CLAIM_STATUSES = ("proposed", "provisional", "replicated", "rejected", "superseded")


@dataclass(frozen=True)
class ClaimEvidence:
    """One cited piece of evidence; a run id or an observation id."""

    kind: str            # run | observation | finding
    ref: str             # the id

    def __post_init__(self) -> None:
        if self.kind not in ("run", "observation", "finding"):
            raise ValueError(f"unknown evidence kind: {self.kind}")
        if not self.ref:
            raise ValueError("evidence needs a reference")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "ref": self.ref}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClaimEvidence":
        unknown = sorted(set(data) - {"kind", "ref"})
        if unknown:
            raise ValueError(f"unknown claim-evidence keys: {unknown}")
        return cls(kind=str(data["kind"]), ref=str(data["ref"]))


@dataclass(frozen=True)
class Claim:
    """A scoped, status-carrying statement the cited runs license."""

    claim_id: str
    statement: str
    scope: str                       # e.g. "qwen3-family, lora r16, replay 0.25"
    status: str = "provisional"
    supporting_experiments: tuple[str, ...] = ()   # run ids
    contradicting_experiments: tuple[str, ...] = ()
    affected_capabilities: tuple[str, ...] = ()
    conditions: tuple[str, ...] = ()
    confidence_basis: str = ""
    replication_count: int = 0

    def __post_init__(self) -> None:
        if not self.claim_id:
            raise ValueError("claim_id is required")
        if not self.statement.strip():
            raise ValueError("a claim needs a statement")
        if not self.scope.strip():
            raise ValueError("a claim states its scope; an unscoped claim is folklore")
        if self.status not in CLAIM_STATUSES:
            raise ValueError(f"unknown claim status: {self.status}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "statement": self.statement,
            "scope": self.scope,
            "status": self.status,
            "supporting_experiments": list(self.supporting_experiments),
            "contradicting_experiments": list(self.contradicting_experiments),
            "affected_capabilities": list(self.affected_capabilities),
            "conditions": list(self.conditions),
            "confidence_basis": self.confidence_basis,
            "replication_count": self.replication_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Claim":
        _KEYS = (
            "claim_id", "statement", "scope", "status", "supporting_experiments",
            "contradicting_experiments", "affected_capabilities", "conditions",
            "confidence_basis", "replication_count",
        )
        unknown = sorted(set(data) - set(_KEYS))
        if unknown:
            raise ValueError(f"unknown claim keys (fail-closed): {unknown}")
        return cls(
            claim_id=str(data["claim_id"]),
            statement=str(data["statement"]),
            scope=str(data["scope"]),
            status=str(data.get("status", "provisional")),
            supporting_experiments=tuple(str(e) for e in data.get("supporting_experiments", ())),
            contradicting_experiments=tuple(str(e) for e in data.get("contradicting_experiments", ())),
            affected_capabilities=tuple(str(c) for c in data.get("affected_capabilities", ())),
            conditions=tuple(str(c) for c in data.get("conditions", ())),
            confidence_basis=str(data.get("confidence_basis", "")),
            replication_count=int(data.get("replication_count", 0)),
        )


@dataclass(frozen=True)
class ReplicationPolicy:
    """The stage ladder distinguishing interesting / reproducible /
    generalizable. Stage names are Chowder-owned; the provider cannot
    re-label them."""

    seeds_for_replication: int = 3
    require_transfer_stage: bool = True
    min_survivors_for_generalization: int = 2

    def __post_init__(self) -> None:
        if self.seeds_for_replication < 2:
            raise ValueError("replication requires at least 2 seeds; one run is an anecdote")

    def claim_status_for(
        self,
        *,
        successful_runs: int,
        transfer_supported: bool | None,
    ) -> str:
        """Mechanical status: the number decides the label, not a reviewer."""
        if successful_runs <= 0:
            return "rejected"
        if successful_runs < self.seeds_for_replication:
            return "provisional"
        if self.require_transfer_stage and transfer_supported is not True:
            return "provisional"
        if self.require_transfer_stage and transfer_supported:
            return "replicated"
        return "replicated"

    def to_dict(self) -> dict[str, Any]:
        return {
            "seeds_for_replication": self.seeds_for_replication,
            "require_transfer_stage": self.require_transfer_stage,
            "min_survivors_for_generalization": self.min_survivors_for_generalization,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReplicationPolicy":
        unknown = sorted(set(data) - {"seeds_for_replication",
                                      "require_transfer_stage",
                                      "min_survivors_for_generalization"})
        if unknown:
            raise ValueError(f"unknown replication-policy keys: {unknown}")
        return cls(
            seeds_for_replication=int(data.get("seeds_for_replication", 3)),
            require_transfer_stage=bool(data.get("require_transfer_stage", True)),
            min_survivors_for_generalization=int(data.get("min_survivors_for_generalization", 2)),
        )


@dataclass(frozen=True)
class ResearchFinding:
    """One finding = claims + the observations that license them."""

    finding_id: str
    hypothesis_id: str
    claims: tuple[Claim, ...]
    observation_ids: tuple[str, ...]
    provider: str = ""
    mission_id: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        if not self.finding_id:
            raise ValueError("finding_id is required")
        if not self.hypothesis_id:
            raise ValueError("a finding reports on a hypothesis")
        if not self.claims:
            raise ValueError("a finding carries at least one claim (possibly a rejected one)")
        if not self.observation_ids:
            raise ValueError(
                "no run, no empirical claim: a finding must cite the observations "
                "(and through them, runs) that license it"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "hypothesis_id": self.hypothesis_id,
            "claims": [c.to_dict() for c in self.claims],
            "observation_ids": list(self.observation_ids),
            "provider": self.provider,
            "mission_id": self.mission_id,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResearchFinding":
        unknown = sorted(set(data) - {"finding_id", "hypothesis_id", "claims",
                                      "observation_ids", "provider", "mission_id", "note"})
        if unknown:
            raise ValueError(f"unknown finding keys (fail-closed): {unknown}")
        return cls(
            finding_id=str(data["finding_id"]),
            hypothesis_id=str(data["hypothesis_id"]),
            claims=tuple(Claim.from_dict(c) for c in data.get("claims", ())),
            observation_ids=tuple(str(o) for o in data.get("observation_ids", ())),
            provider=str(data.get("provider", "")),
            mission_id=str(data.get("mission_id", "")),
            note=str(data.get("note", "")),
        )
