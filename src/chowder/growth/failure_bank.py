"""Failure bank: every meaningful model failure as durable evidence.

Records carry digests rather than raw protected-benchmark text (protected
material lives only in the contamination firewall), so the bank can inform
the curriculum without leaking protected sets into training-accessible
storage. Recurrence tracking (first/last seen generation, recurrence count,
repaired status) turns the bank into the anti-forgetting memory: a repaired
failure class becomes a protected regression probe.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping

from .failure_taxonomy import FailureTaxonomy


def prompt_digest(text: str) -> str:
    """sha256 of a prompt/output -- the durable reference instead of raw
    protected text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FailureRecord:
    """One observed failure of one model version on one evaluation sample."""

    failure_id: str
    model_version: str
    benchmark_qualified_id: str  # benchmark@version
    sample_ref: str  # opaque sample identifier (never raw protected text)
    prompt_digest_hex: str
    output_digest_hex: str
    expected_behavior: str  # free-text description; no protected raw text
    score: float  # measured score on the sample (0..1 typical)
    categories: tuple[str, ...]  # failure taxonomy categories
    verifier_evidence: str  # digest/summary of verifier output
    confidence: float  # 0..1 confidence that this is a true failure
    first_seen_generation: str
    last_seen_generation: str
    recurrence_count: int
    repaired: bool
    repair_generation: str | None
    notes: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = {
            f: getattr(self, f)
            for f in self.__dataclass_fields__  # type: ignore[attr-defined]
        }
        data["categories"] = list(self.categories)
        data["extra"] = dict(self.extra)
        return data


class FailureBank:
    """Append/recurrence-update store of failures with taxonomy-backed
    classification and aggregation."""

    def __init__(self, taxonomy: FailureTaxonomy | None = None) -> None:
        self.taxonomy = taxonomy or FailureTaxonomy()
        self._failures: dict[str, FailureRecord] = {}

    def __len__(self) -> int:
        return len(self._failures)

    def __iter__(self):
        return iter(self._failures.values())

    def record(
        self,
        *,
        model_version: str,
        benchmark_qualified_id: str,
        sample_ref: str,
        prompt: str,
        output: str,
        expected_behavior: str,
        score: float,
        verifier_evidence: str,
        confidence: float,
        generation: str,
        categories: tuple[str, ...] | None = None,
        notes: str = "",
    ) -> FailureRecord:
        """Record a failure (or update recurrence if the same failure was
        already banked for this model/benchmark/sample)."""
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        failure_id = (
            f"{model_version}|{benchmark_qualified_id}|{sample_ref}|{prompt_digest(prompt)[:12]}"
        )
        existing = self._failures.get(failure_id)
        if existing is not None:
            updated = FailureRecord(
                failure_id=existing.failure_id,
                model_version=existing.model_version,
                benchmark_qualified_id=existing.benchmark_qualified_id,
                sample_ref=existing.sample_ref,
                prompt_digest_hex=existing.prompt_digest_hex,
                output_digest_hex=prompt_digest(output),
                expected_behavior=existing.expected_behavior,
                score=score,
                categories=existing.categories,
                verifier_evidence=verifier_evidence,
                confidence=max(existing.confidence, confidence),
                first_seen_generation=existing.first_seen_generation,
                last_seen_generation=generation,
                recurrence_count=existing.recurrence_count + 1,
                repaired=False,
                repair_generation=existing.repair_generation,
                notes=notes or existing.notes,
            )
            self._failures[failure_id] = updated
            return updated

        if categories is None:
            categories = self.taxonomy.classify(
                " ".join([expected_behavior, verifier_evidence, notes])
            ) or ("unclassified",)
        record = FailureRecord(
            failure_id=failure_id,
            model_version=model_version,
            benchmark_qualified_id=benchmark_qualified_id,
            sample_ref=sample_ref,
            prompt_digest_hex=prompt_digest(prompt),
            output_digest_hex=prompt_digest(output),
            expected_behavior=expected_behavior,
            score=score,
            categories=categories,
            verifier_evidence=verifier_evidence,
            confidence=confidence,
            first_seen_generation=generation,
            last_seen_generation=generation,
            recurrence_count=1,
            repaired=False,
            repair_generation=None,
            notes=notes,
        )
        self._failures[failure_id] = record
        return record

    def mark_repaired(self, failure_id: str, *, generation: str) -> FailureRecord:
        """A convincingly repaired failure becomes anti-forgetting evidence:
        its class joins the protected regression probe set."""
        record = self._failures.get(failure_id)
        if record is None:
            raise KeyError(f"unknown failure {failure_id!r}")
        updated = FailureRecord(
            **{
                **record.to_dict(),
                "repaired": True,
                "repair_generation": generation,
            },
        )
        self._failures[failure_id] = updated
        return updated

    def get(self, failure_id: str) -> FailureRecord | None:
        return self._failures.get(failure_id)

    def by_category(self, category: str) -> tuple[FailureRecord, ...]:
        return tuple(f for f in self if category in f.categories)

    def open_failures(self, *, generation: str | None = None) -> tuple[FailureRecord, ...]:
        """Unrepaired failures; optionally only those still failing at
        ``generation`` (recurred there)."""
        return tuple(
            f
            for f in self
            if not f.repaired and (generation is None or f.last_seen_generation == generation)
        )

    def repaired_classes(self) -> tuple[str, ...]:
        """Categories with at least one repaired failure: the anti-forgetting
        probe seeds."""
        return tuple(
            sorted(
                {c for f in self if f.repaired for c in f.categories if c != "unclassified"}
            )
        )

    def category_counts(self, *, generation: str | None = None) -> dict[str, int]:
        counts: dict[str, int] = {}
        for failure in self:
            if generation is not None and failure.last_seen_generation != generation:
                continue
            for category in failure.categories:
                counts[category] = counts.get(category, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def high_frequency(
        self, *, min_recurrence: int = 2, min_confidence: float = 0.6
    ) -> tuple[FailureRecord, ...]:
        """Failures worth curriculum attention: recurring and confidently
        diagnosed."""
        return tuple(
            f
            for f in self
            if f.recurrence_count >= min_recurrence and f.confidence >= min_confidence
        )
