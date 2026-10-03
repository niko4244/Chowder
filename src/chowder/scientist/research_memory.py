"""ResearchMemory: durable append-only scientific memory.

Same discipline as GrowthState: JSONL records under one root, append-only,
read back before the next research decision. Every record carries provenance
to exact runs (`run_id`s resolved against the run registry at query time), so
"have we already tried this mechanism on this model family?" is answered from
durable evidence, never from an LLM's recollection. Contradiction and
supersession are recorded; records are never silently edited.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .findings import Claim, ResearchFinding
from .hypothesis import Hypothesis
from .observation import ExperimentObservation
from .proposal import ExperimentProposal

SCHEMA_VERSION = 1


def _append_jsonl(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(document, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a torn final line from a crash is skipped, never fatal
    return rows


class ResearchMemory:
    """Append-only research records rooted beside the growth state.

    Files: hypotheses.jsonl, experiment-proposals.jsonl,
    experiment-results.jsonl (observations), findings.jsonl,
    rejected-hypotheses.jsonl, method-effects.jsonl.
    """

    def __init__(
        self,
        root: Path,
        *,
        run_exists: Callable[[str], bool] | None = None,
        run_complete: Callable[[str], bool] | None = None,
    ) -> None:
        self.root = Path(root)
        #: Resolvers default to "accept" so the memory layer is usable without
        #: a registry (tests, offline analysis); production wires them to the
        #: RunRegistry. When a resolver refuses, dangling evidence fails loudly.
        self._run_exists = run_exists or (lambda run_id: True)
        self._run_complete = run_complete or (lambda run_id: True)

    # -- paths ---------------------------------------------------------------

    @property
    def hypotheses_path(self) -> Path:
        return self.root / "hypotheses.jsonl"

    @property
    def proposals_path(self) -> Path:
        return self.root / "experiment-proposals.jsonl"

    @property
    def observations_path(self) -> Path:
        return self.root / "experiment-results.jsonl"

    @property
    def findings_path(self) -> Path:
        return self.root / "findings.jsonl"

    @property
    def rejected_path(self) -> Path:
        return self.root / "rejected-hypotheses.jsonl"

    @property
    def effects_path(self) -> Path:
        return self.root / "method-effects.jsonl"

    @property
    def refusals_path(self) -> Path:
        return self.root / "refusals.jsonl"

    # -- append ---------------------------------------------------------------

    def record_hypothesis(self, hypothesis: Hypothesis) -> None:
        _append_jsonl(self.hypotheses_path, {
            "schema_version": SCHEMA_VERSION,
            "record": hypothesis.to_dict(),
        })

    def record_proposal(self, proposal: ExperimentProposal, *, admitted: bool,
                        refusal_reasons: tuple[str, ...] = ()) -> None:
        _append_jsonl(self.proposals_path, {
            "schema_version": SCHEMA_VERSION,
            "admitted": admitted,
            "refusal_reasons": list(refusal_reasons),
            "record": proposal.to_dict(),
        })

    def record_observation(self, observation: ExperimentObservation) -> None:
        if not self._run_exists(observation.run_id):
            raise ValueError(
                f"REFUSAL_RUN_UNKNOWN: observation {observation.observation_id} cites "
                f"run {observation.run_id!r}, which the run registry does not know"
            )
        if observation.status == "complete" and not self._run_complete(observation.run_id):
            raise ValueError(
                f"REFUSAL_RUN_INCOMPLETE: observation {observation.observation_id} "
                f"claims completion for run {observation.run_id!r}, which is not "
                "complete in the run registry"
            )
        _append_jsonl(self.observations_path, {
            "schema_version": SCHEMA_VERSION,
            "record": observation.to_dict(),
        })

    def record_finding(self, finding: ResearchFinding) -> None:
        self._assert_observations_resolve(finding.observation_ids)
        for claim in finding.claims:
            self._assert_claim_runs_resolve(claim)
            self._assert_hardware_context(claim, finding.observation_ids)
        _append_jsonl(self.findings_path, {"schema_version": SCHEMA_VERSION,
                                           "record": finding.to_dict()})

    def _assert_hardware_context(self, claim: Any, observation_ids: tuple[str, ...]) -> None:
        """The hardware-context rule (docs/COMPUTE_PROVIDERS.md §3): a
        hardware-dependent (efficiency) claim may only reach `replicated` when
        every cited run shares one hardware class — a throughput number is
        about that hardware, not about the model. Cross-hardware replication
        is exactly how a quality claim earns its status, so quality claims
        (hardware_dependent=False) are not restricted here."""
        if not isinstance(claim, Claim):
            claim = Claim.from_dict(dict(claim))
        if not claim.hardware_dependent or claim.status != "replicated":
            return
        observations = {o["record"]["observation_id"]: o["record"]
                        for o in _read_jsonl(self.observations_path)}
        cited = [observations[oid] for oid in observation_ids if oid in observations]
        hardware_classes = {str(o.get("hardware_class", "")) for o in cited
                            if o.get("status", "complete") == "complete"}
        if len(hardware_classes) > 1:
            raise ValueError(
                f"REFUSAL_HARDWARE_CONTEXT: claim {claim.claim_id} is hardware-"
                f"dependent yet cites runs from {sorted(hardware_classes)}; an "
                "efficiency claim replicates only on one hardware class"
            )

    def record_rejected_hypothesis(self, hypothesis: Hypothesis, *,
                                   reason: str) -> None:
        if hypothesis.status != "rejected":
            raise ValueError("only a rejected hypothesis enters the rejected ledger")
        _append_jsonl(self.rejected_path, {
            "schema_version": SCHEMA_VERSION,
            "reason": reason,
            "record": hypothesis.to_dict(),
        })

    def record_method_effect(self, effect: dict[str, Any]) -> None:
        """A method-effects row: mechanism -> measured effect with evidence."""
        if "mechanism" not in effect or "evidence_run_ids" not in effect:
            raise ValueError("a method effect names its mechanism and its evidence runs")
        for run_id in effect["evidence_run_ids"]:
            if not self._run_exists(str(run_id)):
                raise ValueError(
                    f"REFUSAL_RUN_UNKNOWN: method effect cites unknown run {run_id!r}"
                )
        _append_jsonl(self.effects_path, {
            "schema_version": SCHEMA_VERSION,
            "record": dict(effect),
        })

    def record_refusal(self, *, provider: str, kind: str, detail: str,
                       payload: dict[str, Any] | None = None) -> None:
        _append_jsonl(self.refusals_path, {
            "schema_version": SCHEMA_VERSION,
            "provider": provider,
            "kind": kind,
            "detail": detail,
            "payload": payload or {},
        })

    # -- queries ---------------------------------------------------------------

    def _assert_observations_resolve(self, observation_ids: Iterable[str]) -> None:
        known = {row["record"]["observation_id"]
                 for row in _read_jsonl(self.observations_path)}
        dangling = [o for o in observation_ids if o not in known]
        if dangling:
            raise ValueError(
                f"REFUSAL_OBSERVATION_UNKNOWN: finding cites unknown observations: {dangling}"
            )

    def _assert_claim_runs_resolve(self, claim: Any) -> None:
        if not isinstance(claim, Claim):
            claim = Claim.from_dict(dict(claim))
        for ref in claim.supporting_experiments:
            if not self._run_exists(ref):
                raise ValueError(
                    f"REFUSAL_RUN_UNKNOWN: claim {claim.claim_id} cites unknown run {ref!r}"
                )

    def observations(self, *, mission_id: str | None = None) -> list[ExperimentObservation]:
        rows = _read_jsonl(self.observations_path)
        out = [ExperimentObservation.from_dict(r["record"]) for r in rows]
        if mission_id is not None:
            out = [o for o in out if o.mission_id == mission_id]
        return out

    def hypotheses(self, *, mission_id: str | None = None,
                   status: str | None = None) -> list[Hypothesis]:
        out = [Hypothesis.from_dict(r["record"])
               for r in _read_jsonl(self.hypotheses_path)]
        if mission_id is not None:
            out = [h for h in out if h.mission_id == mission_id]
        if status is not None:
            out = [h for h in out if h.status == status]
        return out

    def findings(self, *, mission_id: str | None = None) -> list[ResearchFinding]:
        out = [ResearchFinding.from_dict(r["record"])
               for r in _read_jsonl(self.findings_path)]
        if mission_id is not None:
            out = [f for f in out if f.mission_id == mission_id]
        return out

    def has_tried_mechanism(self, mechanism: str, *,
                            model_family: str | None = None) -> bool:
        """The durable prior: has this mechanism been tried (on this family)?"""
        needle = mechanism.strip().lower()
        for row in _read_jsonl(self.effects_path):
            record = row.get("record", {})
            if needle not in str(record.get("mechanism", "")).lower():
                continue
            if model_family is None:
                return True
            if model_family.lower() in str(record.get("model_family", "")).lower():
                return True
        return False

    def method_effects(self, *, mechanism: str | None = None) -> list[dict[str, Any]]:
        rows = _read_jsonl(self.effects_path)
        out = [r.get("record", {}) for r in rows]
        if mechanism is not None:
            needle = mechanism.strip().lower()
            out = [e for e in out if needle in str(e.get("mechanism", "")).lower()]
        return out
