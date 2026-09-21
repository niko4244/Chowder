"""Dataset discovery: candidate sources enter as UNREVIEWED, never trainable.

Discovery ingests *metadata* about a candidate source (from Hugging Face,
papers, GitHub, curated catalogs). It never automatically approves training
use: every candidate must pass the inspect -> license -> quality ->
contamination -> register workflow, and the registration decision is explicit.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, replace
from typing import Any

from .data_registry import DataRegistry, DataSource, now_iso

# Trust classes a candidate may hold while under review.
_REVIEW_CLASSES = ("QUARANTINE",)

_LICENSE_DENY_PATTERNS = (
    r"non-?commercial",
    r"research[- ]only",
    r"cc[- ]by[- ]nc",
    r"all rights reserved",
)

# Benchmarks must never enter the training pool wholesale.
_PROTECTED_ID_PATTERN = re.compile(r"(test|eval|benchmark|protected)", re.IGNORECASE)

# Source types accepted by the underlying data registry.
_ALLOWED_SOURCE_TYPES = frozenset({"synthetic", "human", "web", "code", "paper", "mixed"})


@dataclass(frozen=True)
class Candidate:
    """A discovered-but-unreviewed data source candidate."""

    candidate_id: str
    name: str
    origin: str  # e.g. "huggingface", "github", "paper"
    url: str
    revision: str
    license_declared: str
    domain: str
    language: tuple[str, ...]
    source_type: str  # synthetic | human | web | code | research
    example_count: int
    token_estimate: int
    notes: tuple[str, ...] = field(default_factory=tuple)

    def content_fingerprint(self) -> str:
        """Stable identity of the candidate's declared metadata."""
        payload = "|".join(
            (
                self.candidate_id,
                self.origin,
                self.url,
                self.revision,
                self.license_declared,
                str(self.example_count),
            )
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class InspectionReport:
    """Mechanical findings from the inspection stage."""

    candidate_id: str
    license_ok: bool
    license_reason: str
    looks_like_benchmark: bool
    benchmark_reason: str
    metadata_complete: bool
    missing_fields: tuple[str, ...]

    @property
    def blockers(self) -> tuple[str, ...]:
        blockers: list[str] = []
        if not self.license_ok:
            blockers.append(f"license: {self.license_reason}")
        if self.looks_like_benchmark:
            blockers.append(f"benchmark-shaped: {self.benchmark_reason}")
        if not self.metadata_complete:
            blockers.append(f"missing metadata: {', '.join(self.missing_fields)}")
        return tuple(blockers)


class DataDiscovery:
    """The DISCOVER -> INSPECT -> REGISTER workflow."""

    def __init__(self, registry: DataRegistry | None = None) -> None:
        self.registry = registry if registry is not None else DataRegistry()
        self._candidates: dict[str, Candidate] = {}
        self._rejected: dict[str, tuple[str, ...]] = {}

    def discover(self, candidate: Candidate) -> Candidate:
        """Record a discovered candidate. Registration is a separate step."""
        if candidate.candidate_id in self._candidates:
            raise ValueError(f"candidate already discovered: {candidate.candidate_id}")
        self._candidates[candidate.candidate_id] = candidate
        return candidate

    def candidates(self) -> tuple[Candidate, ...]:
        return tuple(self._candidates.values())

    def rejected(self) -> dict[str, tuple[str, ...]]:
        return dict(self._rejected)

    def inspect(self, candidate_id: str) -> InspectionReport:
        """Mechanical inspection: license, benchmark-shape, metadata completeness."""
        candidate = self._require(candidate_id)
        license_ok, license_reason = self._check_license(candidate)
        looks_like_benchmark, benchmark_reason = self._check_benchmark_shape(candidate)
        missing = self._missing_metadata(candidate)
        return InspectionReport(
            candidate_id=candidate.candidate_id,
            license_ok=license_ok,
            license_reason=license_reason,
            looks_like_benchmark=looks_like_benchmark,
            benchmark_reason=benchmark_reason,
            metadata_complete=not missing,
            missing_fields=missing,
        )

    def register(
        self,
        candidate_id: str,
        *,
        pii_reviewed: bool,
        secrets_reviewed: bool,
        quality_score: float,
        contamination_status: str,
        inclusion_reason: str,
    ) -> DataSource:
        """Explicitly register a candidate into the registry as QUARANTINE.

        Even a fully approved candidate enters as QUARANTINE with
        ``trainable() == False``; promotion out of quarantine is a separate,
        evidence-backed decision made through :func:`chowder.growth.data_registry.admit`.
        """
        candidate = self._require(candidate_id)
        report = self.inspect(candidate_id)
        if report.blockers:
            self._rejected[candidate_id] = report.blockers
            raise PermissionError(
                f"candidate {candidate_id} failed inspection: {report.blockers}"
            )
        if not (pii_reviewed and secrets_reviewed):
            raise PermissionError(
                f"candidate {candidate_id} cannot register without PII and secret review"
            )
        source = DataSource(
            source_id=candidate.candidate_id,
            dataset_name=candidate.name,
            revision=candidate.revision,
            url=candidate.url,
            license=candidate.license_declared,
            permitted_training_use=True,  # license passed inspection; admit() re-checks
            domain=candidate.domain,
            language=", ".join(candidate.language) or "unknown",
            source_type=(
                candidate.source_type
                if candidate.source_type in _ALLOWED_SOURCE_TYPES
                else "mixed"
            ),
            verification="unverified",
            trust_class="QUARANTINE",
            example_count=candidate.example_count,
            token_estimate=candidate.token_estimate,
            provenance=f"discovered via {candidate.origin}; fingerprint "
            f"{candidate.content_fingerprint()[:16]}",
            acquisition_timestamp=now_iso(),
            source_hash=candidate.content_fingerprint(),
            contamination_relationship=contamination_status,
            quality_score=quality_score,
            pii_reviewed=pii_reviewed,
            secrets_reviewed=secrets_reviewed,
            inclusion_decision="pending",
            notes="; ".join(candidate.notes) or inclusion_reason,
        )
        return self.registry.register(source)

    def _require(self, candidate_id: str) -> Candidate:
        candidate = self._candidates.get(candidate_id)
        if candidate is None:
            raise KeyError(f"unknown candidate: {candidate_id}")
        return candidate

    @staticmethod
    def _check_license(candidate: Candidate) -> tuple[bool, str]:
        declared = candidate.license_declared.strip().lower()
        if not declared or declared in {"unknown", "unspecified", "none"}:
            return False, "no license declared"
        for pattern in _LICENSE_DENY_PATTERNS:
            if re.search(pattern, declared):
                return False, f"license blocks training use ({declared})"
        return True, declared

    @staticmethod
    def _check_benchmark_shape(candidate: Candidate) -> tuple[bool, str]:
        if _PROTECTED_ID_PATTERN.search(candidate.candidate_id) or _PROTECTED_ID_PATTERN.search(
            candidate.name
        ):
            return True, "identifier suggests evaluation/protected material"
        return False, ""

    @staticmethod
    def _missing_metadata(candidate: Candidate) -> tuple[str, ...]:
        missing: list[str] = []
        if not candidate.url:
            missing.append("url")
        if not candidate.revision:
            missing.append("revision")
        if candidate.example_count <= 0:
            missing.append("example_count")
        if candidate.token_estimate <= 0:
            missing.append("token_estimate")
        if candidate.source_type not in {"synthetic", "human", "web", "code", "research"}:
            missing.append("source_type")
        return tuple(missing)


__all__ = [
    "Candidate",
    "DataDiscovery",
    "InspectionReport",
]
