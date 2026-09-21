"""Generational lineage: never lose provenance from parent to Model N+1.

Every promoted (or rejected) candidate is recorded with its full ancestry:
parent version, adapter/checkpoint reference, dataset manifest, curriculum
manifest, training recipe, training evidence, evaluation report, and the
promotion decision. Anti-forgetting is part of lineage: once a failure class
is convincingly repaired, its probe becomes PROTECTED -- future generations
must keep passing it, and lineage tracks which probes a generation was
required to hold.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .promotion import PromotionDecision


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class GenerationRecord:
    version: str
    parent_version: str | None
    cycle_id: str
    base_model: Mapping[str, Any]  # revision/quant/inference engine
    adapter_ref: str | None
    checkpoint_ref: str | None
    dataset_manifest_ref: str
    curriculum_manifest_ref: str
    recipe: Mapping[str, Any]
    training_evidence_ref: str
    evaluation_report_ref: str
    promotion: Mapping[str, Any]  # PromotionDecision.to_dict()
    required_probes: tuple[str, ...] = ()  # anti-forgetting probes this gen must hold
    frontier_snapshot_id: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "parent_version": self.parent_version,
            "cycle_id": self.cycle_id,
            "base_model": dict(self.base_model),
            "adapter_ref": self.adapter_ref,
            "checkpoint_ref": self.checkpoint_ref,
            "dataset_manifest_ref": self.dataset_manifest_ref,
            "curriculum_manifest_ref": self.curriculum_manifest_ref,
            "recipe": dict(self.recipe),
            "training_evidence_ref": self.training_evidence_ref,
            "evaluation_report_ref": self.evaluation_report_ref,
            "promotion": dict(self.promotion),
            "required_probes": list(self.required_probes),
            "frontier_snapshot_id": self.frontier_snapshot_id,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class GenerationAdjudicationRevision:
    """A superseding adjudication of one generation's promotion decision.

    Historical records are never mutated: a correction is an append that
    references the original record's digest and names its reason codes. The
    effective verdict for a generation is the latest revision's verdict (or
    the original record's verdict when no revision exists) -- resolved
    deterministically by iteration order in the append-only file.
    """

    generation_version: str
    supersedes_adjudication_id: str  # "original" or a prior revision id
    revision_id: str
    reason_codes: tuple[str, ...]
    policy_version: str
    new_verdict: str
    evidence_refs: tuple[str, ...]
    recorded_at: str
    original_decision_digest: str  # sha256 of the original record's promotion payload
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation_version": self.generation_version,
            "supersedes_adjudication_id": self.supersedes_adjudication_id,
            "revision_id": self.revision_id,
            "reason_codes": list(self.reason_codes),
            "policy_version": self.policy_version,
            "new_verdict": self.new_verdict,
            "evidence_refs": list(self.evidence_refs),
            "recorded_at": self.recorded_at,
            "original_decision_digest": self.original_decision_digest,
            "notes": self.notes,
        }


def _decision_digest(record_dict: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(record_dict.get("promotion", {}), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class GenerationLedger:
    """Append-only ledger of generation records, persisted as JSON."""

    def __init__(self, root: Path | str) -> None:
        self._path = Path(root) / "generations.json"
        self._revisions_path = Path(root) / "adjudication_revisions.json"
        self._records: dict[str, GenerationRecord] = {}
        self._revisions: list[GenerationAdjudicationRevision] = []
        if self._path.exists():
            for item in json.loads(self._path.read_text(encoding="utf-8")):
                record = GenerationRecord(
                    version=item["version"],
                    parent_version=item.get("parent_version"),
                    cycle_id=item["cycle_id"],
                    base_model=item.get("base_model", {}),
                    adapter_ref=item.get("adapter_ref"),
                    checkpoint_ref=item.get("checkpoint_ref"),
                    dataset_manifest_ref=item.get("dataset_manifest_ref", ""),
                    curriculum_manifest_ref=item.get("curriculum_manifest_ref", ""),
                    recipe=item.get("recipe", {}),
                    training_evidence_ref=item.get("training_evidence_ref", ""),
                    evaluation_report_ref=item.get("evaluation_report_ref", ""),
                    promotion=item.get("promotion", {}),
                    required_probes=tuple(item.get("required_probes", ())),
                    frontier_snapshot_id=item.get("frontier_snapshot_id"),
                    notes=item.get("notes", ""),
                )
                self._records[record.version] = record
        if self._revisions_path.exists():
            for item in json.loads(self._revisions_path.read_text(encoding="utf-8")):
                self._revisions.append(
                    GenerationAdjudicationRevision(
                        generation_version=item["generation_version"],
                        supersedes_adjudication_id=item["supersedes_adjudication_id"],
                        revision_id=item["revision_id"],
                        reason_codes=tuple(item["reason_codes"]),
                        policy_version=item["policy_version"],
                        new_verdict=item["new_verdict"],
                        evidence_refs=tuple(item["evidence_refs"]),
                        recorded_at=item["recorded_at"],
                        original_decision_digest=item["original_decision_digest"],
                        notes=item.get("notes", ""),
                    )
                )

    def record(
        self,
        *,
        version: str,
        parent_version: str | None,
        cycle_id: str,
        base_model: Mapping[str, Any],
        dataset_manifest_ref: str,
        curriculum_manifest_ref: str,
        recipe: Mapping[str, Any],
        training_evidence_ref: str,
        evaluation_report_ref: str,
        promotion: PromotionDecision,
        adapter_ref: str | None = None,
        checkpoint_ref: str | None = None,
        required_probes: tuple[str, ...] = (),
        frontier_snapshot_id: str | None = None,
        notes: str = "",
    ) -> GenerationRecord:
        if version in self._records:
            raise ValueError(f"generation {version} already recorded")
        record = GenerationRecord(
            version=version,
            parent_version=parent_version,
            cycle_id=cycle_id,
            base_model=dict(base_model),
            adapter_ref=adapter_ref,
            checkpoint_ref=checkpoint_ref,
            dataset_manifest_ref=dataset_manifest_ref,
            curriculum_manifest_ref=curriculum_manifest_ref,
            recipe=dict(recipe),
            training_evidence_ref=training_evidence_ref,
            evaluation_report_ref=evaluation_report_ref,
            promotion=promotion.to_dict(),
            required_probes=tuple(required_probes),
            frontier_snapshot_id=frontier_snapshot_id,
            notes=notes,
        )
        self._records[version] = record
        self._flush()
        return record

    def get(self, version: str) -> GenerationRecord:
        if version not in self._records:
            raise KeyError(f"unknown generation: {version}")
        return self._records[version]

    # ---------------- superseding adjudications (append-only) ----------------

    def append_adjudication_revision(
        self,
        *,
        generation_version: str,
        reason_codes: Sequence[str],
        policy_version: str,
        new_verdict: str,
        evidence_refs: Sequence[str] = (),
        notes: str = "",
        recorded_at: str | None = None,
    ) -> GenerationAdjudicationRevision:
        """Append a superseding adjudication without touching original bytes.

        The revision references the original record's promotion digest, so a
        reader can verify the original adjudication being superseded is the
        one on disk. Duplicate revision ids and revisions of unknown
        generations refuse.
        """
        if generation_version not in self._records:
            raise KeyError(
                f"cannot adjudicate unknown generation: {generation_version!r}"
            )
        prior = [r for r in self._revisions if r.generation_version == generation_version]
        if prior:
            supersedes = prior[-1].revision_id
        else:
            supersedes = "original"
        revision_id = f"{generation_version}-adjudication-{len(prior) + 1:03d}"
        if any(r.revision_id == revision_id for r in self._revisions):
            raise ValueError(f"adjudication revision {revision_id!r} already exists")
        revision = GenerationAdjudicationRevision(
            generation_version=generation_version,
            supersedes_adjudication_id=supersedes,
            revision_id=revision_id,
            reason_codes=tuple(reason_codes),
            policy_version=policy_version,
            new_verdict=new_verdict,
            evidence_refs=tuple(evidence_refs),
            recorded_at=recorded_at or _utc_now(),
            original_decision_digest=_decision_digest(self._records[generation_version].to_dict()),
            notes=notes,
        )
        self._revisions.append(revision)
        self._flush_revisions()
        return revision

    def revisions_for(self, version: str) -> tuple[GenerationAdjudicationRevision, ...]:
        return tuple(r for r in self._revisions if r.generation_version == version)

    def effective_verdict(self, version: str) -> str:
        """The current verdict: the latest revision's, else the original's.

        Deterministic by construction: revisions are stored append-only and
        the last one for the generation wins. The original record is never
        consulted for its verdict once superseded, but remains fully intact
        and inspectable for audit.
        """
        revisions = self.revisions_for(version)
        if revisions:
            return revisions[-1].new_verdict
        return self.get(version).promotion.get("verdict", "UNKNOWN")

    def _flush_revisions(self) -> None:
        self._revisions_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [revision.to_dict() for revision in self._revisions]
        self._revisions_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )

    def ancestry(self, version: str) -> list[GenerationRecord]:
        """Root-first chain from Generation 0 (or the earliest record) down."""
        chain: list[GenerationRecord] = []
        current: str | None = version
        seen: set[str] = set()
        while current and current in self._records and current not in seen:
            seen.add(current)
            record = self._records[current]
            chain.append(record)
            current = record.parent_version
        chain.reverse()
        return chain

    def versions(self) -> list[str]:
        return list(self._records)

    def _flush(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = [record.to_dict() for record in self._records.values()]
        self._path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )


@dataclass(frozen=True)
class RegressionProbe:
    """A protected anti-forgetting probe derived from a repaired failure."""

    probe_id: str
    benchmark_qualified_id: str
    failure_class: str
    min_score: float  # floor the probe must hold
    created_generation: str
    repaired_generation: str
    status: str = "PROTECTED"  # PROTECTED | RETIRED (with reason)


class RegressionMemory:
    """Anti-forgetting: repaired failure classes become protected probes.

    Every future generation must hold these probes; a violation is a hard
    promotion blocker (the promotion input lists them as protected
    benchmarks). Probes are never silently retired -- retirement requires an
    explicit reason.
    """

    def __init__(self, root: Path | str) -> None:
        self._path = Path(root) / "regression_probes.json"
        self._probes: dict[str, RegressionProbe] = {}
        if self._path.exists():
            for item in json.loads(self._path.read_text(encoding="utf-8")):
                probe = RegressionProbe(
                    probe_id=item["probe_id"],
                    benchmark_qualified_id=item["benchmark_qualified_id"],
                    failure_class=item["failure_class"],
                    min_score=item["min_score"],
                    created_generation=item["created_generation"],
                    repaired_generation=item["repaired_generation"],
                    status=item.get("status", "PROTECTED"),
                )
                self._probes[probe.probe_id] = probe

    def add_probe(
        self,
        *,
        probe_id: str,
        benchmark_qualified_id: str,
        failure_class: str,
        min_score: float,
        created_generation: str,
        repaired_generation: str,
    ) -> RegressionProbe:
        if probe_id in self._probes:
            raise ValueError(f"probe {probe_id} already exists")
        probe = RegressionProbe(
            probe_id=probe_id,
            benchmark_qualified_id=benchmark_qualified_id,
            failure_class=failure_class,
            min_score=min_score,
            created_generation=created_generation,
            repaired_generation=repaired_generation,
        )
        self._probes[probe_id] = probe
        self._flush()
        return probe

    def protected_benchmark_ids(self) -> tuple[str, ...]:
        return tuple(
            probe.benchmark_qualified_id
            for probe in self._probes.values()
            if probe.status == "PROTECTED"
        )

    def hold_report(
        self,
        scores: Mapping[str, float],
    ) -> dict[str, Any]:
        """Did every protected probe hold? MACHINE-READABLE, honest UNKNOWNs."""
        rows: list[dict[str, Any]] = []
        held = 0
        violated = 0
        unknown = 0
        for probe in self._probes.values():
            if probe.status != "PROTECTED":
                continue
            if probe.benchmark_qualified_id not in scores:
                unknown += 1
                rows.append(
                    {
                        "probe_id": probe.probe_id,
                        "benchmark": probe.benchmark_qualified_id,
                        "verdict": "UNKNOWN",
                    }
                )
                continue
            score = scores[probe.benchmark_qualified_id]
            verdict = "held" if score >= probe.min_score else "REGRESSED"
            if verdict == "held":
                held += 1
            else:
                violated += 1
            rows.append(
                {
                    "probe_id": probe.probe_id,
                    "benchmark": probe.benchmark_qualified_id,
                    "score": score,
                    "min_score": probe.min_score,
                    "verdict": verdict,
                }
            )
        return {
            "held": held,
            "violated": violated,
            "unknown": unknown,
            "rows": rows,
            "all_held": violated == 0 and unknown == 0,
        }

    def retire(self, probe_id: str, reason: str) -> None:
        probe = self._probes.get(probe_id)
        if probe is None:
            raise KeyError(f"unknown probe: {probe_id}")
        if not reason:
            raise ValueError("retiring a probe requires an explicit reason")
        self._probes[probe_id] = RegressionProbe(
            probe_id=probe.probe_id,
            benchmark_qualified_id=probe.benchmark_qualified_id,
            failure_class=probe.failure_class,
            min_score=probe.min_score,
            created_generation=probe.created_generation,
            repaired_generation=probe.repaired_generation,
            status="RETIRED",
        )
        self._flush()

    def _flush(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = [probe.__dict__ for probe in self._probes.values()]
        self._path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
