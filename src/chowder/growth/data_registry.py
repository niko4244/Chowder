"""Training-data registry.

Chowder never ingests anonymous untracked blobs into a production
curriculum. Every source carries accountable provenance: exact revision,
license, permitted-use declaration, verification class, contamination
relationship, quality, PII/secret review, and an explicit inclusion
decision.

Trust classes:

- GOLD: objectively verified (executable tests, symbolic/numeric check,
  authoritative answer keys).
- SILVER: strongly verified (multi-judge agreement, trusted curated
  reasoning corpora, citation-supported factual answers).
- BRONZE: weakly verified (filtered educational web text, general
  instruction corpora).
- QUARANTINE: unverified synthetic/raw scraped material. NEVER
  automatically trained on; promotion of quarantine data to a verified
  class requires the verification evidence to be registered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

TRUST_CLASSES = frozenset({"GOLD", "SILVER", "BRONZE", "QUARANTINE"})

SOURCE_TYPES = frozenset({"synthetic", "human", "web", "code", "paper", "mixed"})

VERIFICATION_TYPES = frozenset(
    {
        "executable_tests",
        "symbolic_numeric",
        "authoritative_key",
        "multi_judge",
        "citation_supported",
        "curated_trusted",
        "heuristic_filter",
        "unverified",
    }
)

#: Which verification types can back which trust classes.
_CLASS_FLOOR: dict[str, frozenset[str]] = {
    "GOLD": frozenset({"executable_tests", "symbolic_numeric", "authoritative_key"}),
    "SILVER": frozenset({"multi_judge", "citation_supported", "curated_trusted"}),
    "BRONZE": frozenset({"heuristic_filter", "curated_trusted"}),
    "QUARANTINE": frozenset({"unverified"}),
}

#: The largest public corpora are DATA RESERVOIRS: streamed/sampled, never
#: bulk-downloaded by default (see docs/DATA_POLICY.md).
RESERVOIR_SOURCES = frozenset({"fineweb-edu", "dclm-baseline", "the-stack-v2"})


@dataclass(frozen=True)
class DataSource:
    """One accountable training-data source."""

    source_id: str
    dataset_name: str
    revision: str  # exact revision/pin; "latest" refused
    url: str
    license: str  # SPDX or explicit terms; "unknown" refused for inclusion
    permitted_training_use: bool
    domain: str
    language: str
    source_type: str  # SOURCE_TYPES
    verification: str  # VERIFICATION_TYPES
    trust_class: str  # TRUST_CLASSES
    example_count: int | None
    token_estimate: int | None
    provenance: str  # who produced it and how
    acquisition_timestamp: str  # ISO instant
    source_hash: str  # hash of the acquired artifact/manifest
    contamination_relationship: str  # CLEAN | POSSIBLE | KNOWN_CONTAMINATION | UNKNOWN
    quality_score: float  # 0..1
    difficulty_distribution: Mapping[str, float] = field(default_factory=dict)
    pii_reviewed: bool = False
    secrets_reviewed: bool = False
    inclusion_decision: str = "pending"  # pending | included | excluded
    exclusion_reason: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if self.revision.strip().lower() in {"latest", "current", ""}:
            raise ValueError(
                f"{self.source_id}: revision must be pinned (got {self.revision!r})"
            )
        if self.trust_class not in TRUST_CLASSES:
            raise ValueError(f"{self.source_id}: unknown trust class {self.trust_class!r}")
        if self.verification not in VERIFICATION_TYPES:
            raise ValueError(f"{self.source_id}: unknown verification {self.verification!r}")
        if self.verification not in _CLASS_FLOOR[self.trust_class]:
            raise ValueError(
                f"{self.source_id}: trust class {self.trust_class} requires verification in "
                f"{sorted(_CLASS_FLOOR[self.trust_class])}, got {self.verification!r}"
            )
        if self.source_type not in SOURCE_TYPES:
            raise ValueError(f"{self.source_id}: unknown source_type {self.source_type!r}")
        if not 0.0 <= self.quality_score <= 1.0:
            raise ValueError(f"{self.source_id}: quality_score must be in [0, 1]")
        if not self.license.strip() or self.license.strip().lower() in {"unknown", "unspecified"}:
            raise ValueError(
                f"{self.source_id}: license must be recorded (SPDX or explicit terms); "
                "'unknown' is refused -- licensing is a registration requirement"
            )
        if self.inclusion_decision not in {"pending", "included", "excluded"}:
            raise ValueError(f"{self.source_id}: invalid inclusion_decision")

    @property
    def trainable(self) -> bool:
        """May this source enter a production curriculum?"""
        return (
            self.inclusion_decision == "included"
            and self.permitted_training_use
            and self.trust_class in {"GOLD", "SILVER", "BRONZE"}
            and self.contamination_relationship in {"CLEAN", "POSSIBLE_CLEARED"}
        )

    def to_dict(self) -> dict[str, Any]:
        data = {
            f: getattr(self, f)
            for f in self.__dataclass_fields__  # type: ignore[attr-defined]
        }
        data["difficulty_distribution"] = dict(self.difficulty_distribution)
        return data


class DataRegistry:
    """The registered set of training-data sources."""

    def __init__(self) -> None:
        self._sources: dict[str, DataSource] = {}

    def __len__(self) -> int:
        return len(self._sources)

    def __iter__(self):
        return iter(self._sources.values())

    def register(self, source: DataSource) -> DataSource:
        if source.source_id in self._sources:
            raise ValueError(f"source {source.source_id!r} already registered")
        self._sources[source.source_id] = source
        return source

    def get(self, source_id: str) -> DataSource | None:
        return self._sources.get(source_id)

    def trainable_sources(self) -> tuple[DataSource, ...]:
        return tuple(s for s in self if s.trainable)

    def by_class(self, trust_class: str) -> tuple[DataSource, ...]:
        return tuple(s for s in self if s.trust_class == trust_class)

    def sources(self) -> tuple[DataSource, ...]:
        return tuple(self._sources.values())


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def quarantine(source: DataSource) -> DataSource:
    """Force a source into quarantine (e.g. a failed contamination check or
    missing verification evidence). Returns the quarantined copy."""
    if source.trust_class == "QUARANTINE":
        return source
    return _replace(source, trust_class="QUARANTINE", verification="unverified")


def admit(
    source: DataSource,
    *,
    decision: str,
    reason: str = "",
    contamination_relationship: str | None = None,
) -> DataSource:
    """Record an explicit inclusion/exclusion decision (with the reason the
    decision was made -- auditability, not vibes)."""
    if decision not in {"included", "excluded"}:
        raise ValueError("decision must be 'included' or 'excluded'")
    updated = source
    if contamination_relationship is not None:
        updated = _replace(
            updated, contamination_relationship=contamination_relationship
        )
    updated = _replace(updated, inclusion_decision=decision, exclusion_reason=reason)
    if decision == "included" and not updated.permitted_training_use:
        raise ValueError(
            f"{source.source_id}: cannot include a source whose license does not "
            "permit training use"
        )
    return updated


def _replace(source: DataSource, **changes: Any) -> DataSource:
    from dataclasses import replace as _dc_replace

    return _dc_replace(source, **changes)


def seed_registry() -> DataRegistry:
    """The initial source catalog: reputable open corpora registered with
    pinned revisions. Reservoir sources are marked for STREAMING/SAMPLING,
    not bulk download (docs/DATA_POLICY.md). Training-use flags reflect the
    licenses as understood at registration; re-verify before each campaign.

    Contamination relationships start at UNKNOWN until the firewall checks
    each source; ``admit()`` records the check result.
    """
    registry = DataRegistry()

    def _add(**kwargs: Any) -> None:
        base = dict(
            acquisition_timestamp=now_iso(),
            source_hash="pending-hash-on-acquisition",
            example_count=None,
            token_estimate=None,
            contamination_relationship="UNKNOWN",
            difficulty_distribution={},
            notes="",
        )
        base.update(kwargs)
        registry.register(DataSource(**base))

    # ---------------- reasoning ----------------
    _add(
        source_id="openthoughts3",
        dataset_name="OpenThoughts3 (reasoning traces)",
        revision="2025-08-snapshot",
        url="https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M",
        license="Apache-2.0",
        permitted_training_use=True,
        domain="reasoning",
        language="en",
        source_type="synthetic",
        verification="multi_judge",
        trust_class="SILVER",
        example_count=1_200_000,
        token_estimate=1_800_000_000,
        provenance="synthetic reasoning traces distilled from strong open models, verified per OpenThoughts methodology",
        quality_score=0.75,
        notes="verify current HF card terms before each campaign",
    )
    _add(
        source_id="openr1-math-220k",
        dataset_name="OpenR1-Math-220k",
        revision="2025-02",
        url="https://huggingface.co/datasets/open-r1/OpenR1-Math-220k",
        license="Apache-2.0",
        permitted_training_use=True,
        domain="math",
        language="en",
        source_type="synthetic",
        verification="multi_judge",
        trust_class="SILVER",
        example_count=220_000,
        token_estimate=500_000_000,
        provenance="R1-distilled math reasoning traces (NuminaMath-derived problems)",
        quality_score=0.72,
    )
    # ---------------- coding ----------------
    _add(
        source_id="swe-smith",
        dataset_name="SWE-smith (repo bug instances)",
        revision="2025-06",
        url="https://huggingface.co/datasets/SWE-bench/SWE-smith",
        license="MIT",
        permitted_training_use=True,
        domain="coding",
        language="en+code",
        source_type="code",
        verification="executable_tests",
        trust_class="GOLD",
        example_count=50_000,
        token_estimate=300_000_000,
        provenance="synthetic repository bug instances from real repos; each with failing tests",
        quality_score=0.8,
    )
    _add(
        source_id="taco-exec",
        dataset_name="TACO (execution-verified subset)",
        revision="2023-11",
        url="https://huggingface.co/datasets/BAAI/TACO",
        license="Apache-2.0",
        permitted_training_use=True,
        domain="coding",
        language="en+code",
        source_type="code",
        verification="executable_tests",
        trust_class="GOLD",
        example_count=25_000,
        token_estimate=120_000_000,
        provenance="competitive programming problems with executable test verification",
        quality_score=0.75,
    )
    # ---------------- general knowledge reservoirs (streamed only) ----------
    _add(
        source_id="fineweb-edu",
        dataset_name="FineWeb-Edu (educational web text)",
        revision="2024-06-v2",
        url="https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu",
        license="ODC-BY",
        permitted_training_use=True,
        domain="general",
        language="en",
        source_type="web",
        verification="heuristic_filter",
        trust_class="BRONZE",
        example_count=1_300_000_000,
        token_estimate=3_000_000_000_000,
        provenance="CommonCrawl filtered by educational-quality classifier (HuggingFaceFW)",
        quality_score=0.6,
        notes="RESERVOIR: stream/sample only; never bulk-download locally",
    )
    _add(
        source_id="dclm-baseline",
        dataset_name="DCLM baseline (filtered web)",
        revision="2024-07",
        url="https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0",
        license="ODC-BY",
        permitted_training_use=True,
        domain="general",
        language="en",
        source_type="web",
        verification="heuristic_filter",
        trust_class="BRONZE",
        example_count=2_400_000_000,
        token_estimate=3_600_000_000_000,
        provenance="DataComp-LM filtered CommonCrawl",
        quality_score=0.6,
        notes="RESERVOIR: stream/sample only",
    )
    # ---------------- research ----------------
    _add(
        source_id="core-bench-pairs",
        dataset_name="CORE-Bench paper/code pairs",
        revision="2024-09",
        url="https://github.com/hendrycks/core-bench",
        license="MIT (framework); task repos retain their licenses",
        permitted_training_use=True,
        domain="research",
        language="en+code",
        source_type="paper",
        verification="executable_tests",
        trust_class="GOLD",
        example_count=360,
        token_estimate=50_000_000,
        provenance="computational reproduction tasks derived from published papers with executable pipelines",
        quality_score=0.7,
    )
    # ---------------- quarantine by default ----------------
    _add(
        source_id="raw-synthetic-unreviewed",
        dataset_name="Raw synthetic generations (unreviewed)",
        revision="rolling",
        url="internal://growth/synthetic-unreviewed",
        license="internal",
        permitted_training_use=True,
        domain="mixed",
        language="en",
        source_type="synthetic",
        verification="unverified",
        trust_class="QUARANTINE",
        example_count=None,
        token_estimate=None,
        provenance="freshly generated synthetic material awaiting critic/verifier review",
        quality_score=0.2,
        notes="NEVER train directly; promote only after verification evidence is registered",
    )

    return registry
