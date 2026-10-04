"""The Intervention Evidence Store: durable, scoped, tamper-evident memory
of what interventions actually did, so the loop stops rediscovering failures
it has already measured.

Every record is a *scoped* claim, never a universal truth: it names the model
family, checkpoint identity, architecture, evaluation suite, intervention
family and parameters, software/runtime versions, hardware class, sample size,
evidence quality, date and source -- and a state from :data:`EvidenceState`.
Candidate generation reads priors from these records; nothing here promotes
anything automatically, and a prior modifies a proposal's chances, never the
promotion gate.

The store is an append-only JSONL with a sha256 hash chain (each record's
``record_digest`` covers its content *and* the previous record's digest), so
editing or removing a recorded outcome breaks the chain and :meth:`audit`
fails. That is the point: a memory that can be quietly rewritten is not
memory, it is whichever story survives.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "EvidenceState",
    "EvidenceRecord",
    "EvidenceStore",
    "PriorAdjustment",
    "prior_for_family",
]


class EvidenceState(str, Enum):
    """What a measured outcome licenses saying about a scoped intervention."""

    #: Reproduced improvement in this scope; modest extra budget is rational.
    PROMISING = "promising"
    #: No measured effect either way in this scope.
    NEUTRAL = "neutral"
    #: Measured harm or a failed hypothesis in this scope.
    FAILED = "failed"
    #: Rejected by a preregistered rule (a decision, not a measurement).
    REJECTED = "rejected"
    #: Ran but could not answer the question (too few samples, broken eval).
    INCONCLUSIVE = "inconclusive"
    #: Cannot apply to this architecture at all (structurally, not empirically).
    ARCHITECTURE_INCOMPATIBLE = "architecture-incompatible"
    #: Claimed once, not reproduced when re-measured in this scope.
    NOT_REPRODUCED = "not-reproduced"
    #: Qualified for production use in this scope by measured evidence.
    PRODUCTION_QUALIFIED = "production-qualified"


_EVIDENCE_SCHEMA = "chowder-intervention-evidence-v1"


@dataclass(frozen=True)
class EvidenceRecord:
    """One scoped, dated claim about what an intervention did.

    Every scope field is required: a record that cannot say *where it
    applies* is exactly the universal-truth shortcut the store exists to
    prevent. ``None`` is honest only for fields the run genuinely could not
    know (say, a hub model's local weight digest).
    """

    record_id: str
    state: EvidenceState
    family_id: str
    #: Scope -- the boundaries of the claim.
    model_family: str
    checkpoint_identity: str
    architecture: str
    eval_suite: str
    intervention_parameters: Mapping[str, Any]
    software_runtime: Mapping[str, str]
    hardware_class: str
    #: Quality of the measurement itself.
    sample_size: int
    evidence_quality: str  # paired-seeds | single-run | aggregate-only | telemetry
    measured_effect: Mapping[str, float]  # dimension -> paired delta
    date: str  # ISO date of the measurement
    source_run: str  # run/kernel/artifact identifier the claim came from
    source_branch: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["state"] = self.state.value
        payload["schema"] = _EVIDENCE_SCHEMA
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvidenceRecord":
        expected = set(EvidenceRecord.__dataclass_fields__)  # type: ignore[attr-defined]
        missing = sorted(expected - set(payload))
        if missing:
            raise ValueError(f"evidence record is missing scope fields {missing}")
        return cls(
            record_id=str(payload["record_id"]),
            state=EvidenceState(str(payload["state"])),
            family_id=str(payload["family_id"]),
            model_family=str(payload["model_family"]),
            checkpoint_identity=str(payload["checkpoint_identity"]),
            architecture=str(payload["architecture"]),
            eval_suite=str(payload["eval_suite"]),
            intervention_parameters=dict(payload["intervention_parameters"]),
            software_runtime={str(k): str(v) for k, v in dict(payload["software_runtime"]).items()},
            hardware_class=str(payload["hardware_class"]),
            sample_size=int(payload["sample_size"]),
            evidence_quality=str(payload["evidence_quality"]),
            measured_effect={str(k): float(v) for k, v in dict(payload["measured_effect"]).items()},
            date=str(payload["date"]),
            source_run=str(payload["source_run"]),
            source_branch=(str(payload["source_branch"]) if payload.get("source_branch") else None),
            notes=str(payload.get("notes", "")),
        )


def _digest_record(payload: Mapping[str, Any], previous_digest: str) -> str:
    chain = {"record": payload, "previous": previous_digest}
    encoded = json.dumps(chain, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class EvidenceStore:
    """An append-only, hash-chained JSONL of evidence records."""

    path: Path
    _chain_tip: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if self.path.exists():
            self._chain_tip = self._load_and_audit()

    def _rows(self) -> list[tuple[dict[str, Any], str]]:
        rows: list[tuple[dict[str, Any], str]] = []
        if not self.path.exists():
            return rows
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                digest = str(row.get("record_digest", ""))
                rows.append((row, digest))
        return rows

    def _load_and_audit(self) -> str:
        previous = ""
        for row, digest in self._rows():
            payload = {k: v for k, v in row.items() if k != "record_digest"}
            expected = _digest_record(payload, previous)
            if expected != digest:
                raise ValueError(
                    f"evidence store {self.path} failed its hash chain at "
                    f"record {payload.get('record_id')!r}: recorded digest "
                    f"{digest[:12]} != computed {expected[:12]} -- a recorded "
                    "outcome was mutated or removed, and the memory is now "
                    "refusing to lie"
                )
            previous = digest
        return previous

    def audit(self) -> dict[str, Any]:
        """Verify the whole chain; returns a report the caller can record."""
        previous = ""
        count = 0
        for _row, _digest in self._rows():
            count += 1
            payload = {k: v for k, v in _row.items() if k != "record_digest"}
            expected = _digest_record(payload, previous)
            if expected != _digest:
                raise ValueError(
                    f"evidence store {self.path} failed its hash chain at "
                    f"record {payload.get('record_id')!r}"
                )
            previous = expected
        return {"records": count, "chain_tip": previous, "schema": _EVIDENCE_SCHEMA}

    def record(self, entry: EvidenceRecord) -> str:
        """Append one scoped record; returns its digest."""
        for existing, _digest in self._rows():
            if existing.get("record_id") == entry.record_id:
                raise ValueError(
                    f"evidence record {entry.record_id!r} already exists; the "
                    "store is append-only -- supersede with a new record that "
                    "names the one it updates"
                )
        payload = entry.to_dict()
        digest = _digest_record(payload, self._chain_tip)
        row = dict(payload, record_digest=digest)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        self._chain_tip = digest
        return digest

    def records_for_family(
        self, family_id: str, *, model_family: str | None = None
    ) -> tuple[EvidenceRecord, ...]:
        """Every record about one family, optionally narrowed to a model family."""
        found: list[EvidenceRecord] = []
        for row, _digest in self._rows():
            payload = {k: v for k, v in row.items() if k != "record_digest"}
            record = EvidenceRecord.from_dict(payload)
            if record.family_id != family_id:
                continue
            if model_family is not None and record.model_family != model_family:
                continue
            found.append(record)
        return tuple(found)


#: How strongly each state bends a proposal's prior. A prior NEVER promotes:
#: it changes how much exploration budget a family starts with, and the
#: promotion gate stays exactly what the campaign declared.
_STATE_PRIORS: Mapping[EvidenceState, float] = {
    EvidenceState.PROMISING: 1.25,
    EvidenceState.NEUTRAL: 1.0,
    EvidenceState.INCONCLUSIVE: 0.9,
    EvidenceState.NOT_REPRODUCED: 0.5,
    EvidenceState.FAILED: 0.35,
    EvidenceState.REJECTED: 0.2,
    EvidenceState.ARCHITECTURE_INCOMPATIBLE: 0.0,
    EvidenceState.PRODUCTION_QUALIFIED: 1.5,
}


@dataclass(frozen=True)
class PriorAdjustment:
    """What history says about proposing a family again, and why."""

    family_id: str
    model_family: str
    multiplier: float
    reasons: tuple[str, ...]

    @property
    def excluded(self) -> bool:
        """True when history says this family cannot apply here at all."""
        return self.multiplier <= 0.0


def prior_for_family(
    store: EvidenceStore,
    *,
    family_id: str,
    model_family: str,
    architecture: str,
) -> PriorAdjustment:
    """Derive the proposal prior for a family in a concrete scope.

    Rules, in the order the mandate states them:

    * an architecture-incompatible record in scope excludes the family --
      it cannot apply, whatever its average outcome elsewhere;
    * repeated measured failure in scope shrinks the prior hard;
    * promising-in-scope gets a modest boost;
    * untested in this scope -> neutral, i.e. an exploration candidate;
    * nothing here reaches 1.5x twice: even a production-qualified family
      never promotes automatically.
    """
    records = [
        record
        for record in store.records_for_family(family_id)
        if record.model_family == model_family
    ]
    if not records:
        return PriorAdjustment(
            family_id=family_id,
            model_family=model_family,
            multiplier=1.0,
            reasons=("no in-scope history: an exploration candidate",),
        )

    incompatible = [r for r in records if r.state is EvidenceState.ARCHITECTURE_INCOMPATIBLE]
    if incompatible:
        matching_arch = [
            r for r in incompatible if not r.architecture or r.architecture == architecture
        ]
        if matching_arch:
            return PriorAdjustment(
                family_id=family_id,
                model_family=model_family,
                multiplier=0.0,
                reasons=(
                    "architecture-incompatible in scope "
                    f"({matching_arch[0].record_id}): the family cannot apply "
                    f"to {architecture!r}"
                ),
            )

    multiplier = 1.0
    reasons: list[str] = []
    failed = [r for r in records if r.state in (EvidenceState.FAILED, EvidenceState.REJECTED)]
    promising = [r for r in records if r.state is EvidenceState.PROMISING]
    reproduced_failure = [
        r for r in records if r.state is EvidenceState.NOT_REPRODUCED
    ]
    qualified = [r for r in records if r.state is EvidenceState.PRODUCTION_QUALIFIED]

    if failed:
        multiplier *= _STATE_PRIORS[EvidenceState.FAILED] ** min(len(failed), 3)
        reasons.append(f"{len(failed)} measured failure(s) in scope shrink the prior hard")
    if reproduced_failure:
        multiplier *= _STATE_PRIORS[EvidenceState.NOT_REPRODUCED]
        reasons.append("a previously promising result was not reproduced in scope")
    if promising:
        multiplier *= _STATE_PRIORS[EvidenceState.PROMISING]
        reasons.append(f"{len(promising)} promising result(s) in scope earn a modest boost")
    if qualified and not failed:
        multiplier *= _STATE_PRIORS[EvidenceState.PRODUCTION_QUALIFIED]
        reasons.append(
            "production-qualified in scope -- a stronger prior, still never an "
            "automatic promotion"
        )
    if not reasons:
        reasons.append("in-scope history is neutral or inconclusive; prior unchanged")
        multiplier = _STATE_PRIORS[EvidenceState.INCONCLUSIVE] if all(
            r.state is EvidenceState.INCONCLUSIVE for r in records
        ) else 1.0
    return PriorAdjustment(
        family_id=family_id,
        model_family=model_family,
        multiplier=round(multiplier, 4),
        reasons=tuple(reasons),
    )
