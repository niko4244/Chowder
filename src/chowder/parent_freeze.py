"""Deterministic four-parent (A/B/C/D) evidence-reporting and freeze layer.

Roadmap Priority 0 Phase 4 needs a way to turn four *already completed*
protected parent-tournament runs (`parent_tournament.py` /
`parent_eval.py`'s `evaluation_runs` rows) into one evidence-backed
parent-selection decision, without re-running or re-interpreting the
evaluation protocol itself. This module is that consumer.

Hard boundary this module respects
-----------------------------------
This module never imports GPU/model-loading code, never launches a
subprocess, and never writes anywhere under a protected tournament output
directory or registry. It reads:

- `evaluation_runs` rows through the registry's existing public API
  (`list_evaluation_outcomes` / `list_experiments`) -- the same shape every
  other Chowder evaluation produces via `parent_eval.record_parent_tournament_result`;
- optionally, already-written `predictions-*.jsonl` files under a
  completed run's recorded `source_artifact_ref` directory, strictly
  read-only, to build a best-effort item-level failure inventory. Their
  content is re-hashed and checked against the digest the tournament
  itself recorded (`prediction_file_sha256`) before being trusted, so a
  directory that has changed since the run is reported as unavailable
  rather than silently mis-attributed.

It never touches `parent_eval.py`, `parent_tournament.py`,
`evaluators/base_text_worker.py`, or protected suite content, and performs
no evaluation of its own.

Two artifacts, deliberately different strictness
--------------------------------------------------
`build_selection_packet` is tolerant: it reports whatever evidence exists
for the bound roles, including partial evidence, and never raises for
"role missing" or "gate failed" -- every gate's pass/fail is itself
evidence, recorded in the packet.

`freeze_selected_parent` is strict: it re-checks every fail-closed gate
against the packet and raises the *first* violated one. A
`ParentFreezeRecord` can only be produced once every parent has complete,
mutually consistent, protocol-matching, tokenizer-comparable evidence.

Why some mission-requested fields are `None` for every run today
------------------------------------------------------------------
`worker_attempts` and `commit_headroom_gib_at_launch` are computed by
`parent_tournament._run_worker` but are never written into the
`ParentEvalReport` evidence dict that gets persisted (only `wall_seconds`,
`peak_gpu_mib_sampled`, `worker_runtime`, `worker_versions`, and
`suite_evidence` are copied across in `evaluate_parent`) -- confirmed by
reading both the registered evidence shape and the real retry7 artifacts
on disk (no companion file records them either). This is a real gap in
the upstream tournament code, which this module is not permitted to
modify. Rather than guess, every `ParentEvidence` row reports these two
fields as `None` with an explicit entry in `evidence_gaps` naming the
reason, exactly the "no fabricated scores" discipline `parent_eval.py`
itself documents.

Similarly, tokenizer identity (class/vocab-size/serialized-asset hash) is
measured by `parent_tournament.tokenizer_evidence` before any model load
but is never persisted into the `evaluation_runs` row -- only a free-text
note ("identity hashed from serialized assets on disk") survives. This
module therefore takes tokenizer evidence as an explicit optional
argument the caller supplies (by calling the existing, unmodified
`parent_tournament.tokenizer_evidence` against each parent's still-local
checkpoint, exactly the "reuse existing abstractions" instruction) --
never re-measured or guessed here, and fails closed by default when it is
missing.

Why "protocol version" is enforced via digest comparison, not a stored field
-------------------------------------------------------------------------------
`ParentEvalSpec.to_dict()` (and therefore `.digest()`) does not serialize
the dataclass's own `protocol_version` field -- it is scorer-side semantics
in `base_text_worker.py`, not a spec field. Two runs cannot be told apart
by a stored "protocol_version" string because none is stored. What *is*
provably different between the retry6 (v1) and retry7 (v2) protocols is
the generation budget (`max_new_tokens`), which *is* part of every suite's
`to_dict()` and therefore part of the digest. This module's protocol gate
therefore compares each parent's recorded `evaluation_protocol_sha256`
against a caller-supplied `expected_protocol_sha256` -- computed by the
caller from the live, current `parent_suite_content.build_tournament_spec`
(the exact function `parent_tournament.run_tournament` itself calls) --
rather than trusting a "protocol_version" label that does not exist in
the persisted evidence.

No blended scalar
------------------
`parent_eval.py` deliberately keeps `capability_mean` and `behavior_mean`
un-blended, and nowhere in this program's specification is there a single
combined score. `default_decision_rule` therefore never invents one: it
classifies every dimension delta *against the native control (A)* in item
units (six items per protected-suite dimension, exactly
`parent_tournament.compare_reports`'s convention, generalized from two
parents to four), and recommends B (the documented primary development
parent) only when the evidence shows no clear-difference capability or
behavior regression against A; otherwise it names the specific
dimensions that block an automatic call and returns "no_automatic_selection"
rather than guessing. Callers may substitute their own explicit,
equally-testable `decision_rule`.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .parent_eval import BEHAVIOR_DIMENSION, CAPABILITY_DIMENSIONS, PARENT_DIMENSIONS

#: Protected suite v1 authors exactly six items per dimension
#: (`parent_suite_content.py`; independently confirmed by counting real
#: `predictions-*.jsonl` rows under a completed retry7 run). Mirrors
#: `parent_tournament.compare_reports`'s private `_ITEMS_PER_SUITE` --
#: duplicated as a single named constant, with this provenance note,
#: rather than importing a module-private symbol from a file this module
#: must never modify.
ITEMS_PER_SUITE = 6
EXPECTED_SUITE_COUNT = len(PARENT_DIMENSIONS)
EXPECTED_TOTAL_ITEMS = ITEMS_PER_SUITE * EXPECTED_SUITE_COUNT  # 54

PARENT_ROLES: tuple[str, ...] = ("A", "B", "C", "D")

#: The evidence tag `parent_tournament.evaluate_parent` stamps onto every
#: tournament row's evidence dict via `aggregate_parent_result`. Used to
#: distinguish parent-tournament rows from any other `evaluation_runs`
#: row a shared registry might hold.
_TOURNAMENT_EVIDENCE_TAG = "qwen38-native-sparse"

_TIE_THRESHOLD_ITEMS = 0.5
_WEAK_SIGNAL_THRESHOLD_ITEMS = 1.5


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class ParentFreezeError(ValueError):
    """Base: the freeze record cannot be produced honestly from this evidence."""


class MissingParentEvidenceError(ParentFreezeError):
    """One or more bound roles have no matching `evaluation_runs` row."""


class DuplicateParentEvidenceError(ParentFreezeError):
    """A role has more than one mutually inconsistent tournament row."""


class ProtocolMismatchError(ParentFreezeError):
    """A parent's protocol digest does not match the reference (or another parent)."""


class SuiteContentMismatchError(ParentFreezeError):
    """Two parents' protected-suite content digests disagree for the same suite."""


class TokenizerComparabilityError(ParentFreezeError):
    """Tokenizer identity evidence is missing or provably not identical."""


class RevisionMismatchError(ParentFreezeError):
    """A parent's recorded revision does not match its pinned revision."""


class IncompleteDimensionCoverageError(ParentFreezeError):
    """A parent's report does not cover all nine dimensions with the expected suite shape."""


class MalformedEvidenceError(ParentFreezeError):
    """A persisted `evaluation_runs` row does not have the shape this module requires."""


@dataclass(frozen=True)
class RoleBinding:
    """Binds one tournament role (A/B/C/D) to a parent label and its pin.

    `label` must equal `ParentEvalReport.base_model` (== `LocalParent.label`)
    for that parent's tournament rows. `expected_revision` is the full
    40-character commit sha this program pins that parent to
    (`docs/QWEN38_SPARSE_PROGRAM.md` / `qwen38_campaign.py`'s `ParentPin`),
    never a branch name or short hash.
    """

    role: str
    label: str
    expected_revision: str

    def __post_init__(self) -> None:
        if self.role not in PARENT_ROLES:
            raise ValueError(f"role must be one of {PARENT_ROLES}, got {self.role!r}")
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("RoleBinding.label must be a non-empty string")
        rev = self.expected_revision
        if not isinstance(rev, str) or len(rev) != 40 or any(c not in "0123456789abcdef" for c in rev.lower()):
            raise ValueError(
                f"RoleBinding.expected_revision must be a full 40-character commit sha, "
                f"not a branch name or short hash: {rev!r}"
            )

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "label": self.label, "expected_revision": self.expected_revision}


@dataclass(frozen=True)
class ItemClassification:
    """One failing (or truncation-flagged) protected item, hash-identified.

    Raw prompt/expected/prediction text is never carried in this record --
    only its position and a hash of the prompt -- mirroring the hash-only
    discipline `contamination.py` and `parent_eval.build_protected_suite_dir`
    already use for protected content. Full text remains inspectable
    directly from the run's `predictions-*.jsonl` file by anyone with
    access to it; this module does not re-export it into a packet that
    might be shared more widely than the raw run directory.
    """

    suite: str
    row_index: int
    prompt_sha256: str
    score: float
    likely_truncation: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "row_index": self.row_index,
            "prompt_sha256": self.prompt_sha256,
            "score": self.score,
            "likely_truncation": self.likely_truncation,
        }


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


@dataclass(frozen=True)
class ParentEvidence:
    """One role's full evidence view, built from its `evaluation_runs` row(s)."""

    role: str
    label: str
    run_id: str
    experiment_id: str
    revision: str | None
    revision_matches_pin: bool
    evaluation_protocol_sha256: str
    protocol_matches_reference: bool
    model_manifest_sha256: str | None
    suite_digests: Mapping[str, str]
    dimension_scores: Mapping[str, float | None]
    dimension_suite_counts: Mapping[str, int]
    capability_mean: float | None
    behavior_mean: float | None
    worker_runtime: Mapping[str, Any]
    worker_versions: Mapping[str, Any]
    wall_seconds: float | None
    peak_gpu_mib_sampled: int | None
    commit_headroom_gib_at_launch: float | None
    worker_attempts: int | None
    prediction_file_sha256: Mapping[str, str]
    tokenizer: Mapping[str, Any] | None
    item_failures: tuple[ItemClassification, ...]
    item_inventory_source: str
    evidence_gaps: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "label": self.label,
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "revision": self.revision,
            "revision_matches_pin": self.revision_matches_pin,
            "evaluation_protocol_sha256": self.evaluation_protocol_sha256,
            "protocol_matches_reference": self.protocol_matches_reference,
            "model_manifest_sha256": self.model_manifest_sha256,
            "suite_digests": dict(self.suite_digests),
            "dimension_scores": dict(self.dimension_scores),
            "dimension_suite_counts": dict(self.dimension_suite_counts),
            "capability_mean": self.capability_mean,
            "behavior_mean": self.behavior_mean,
            "worker_runtime": dict(self.worker_runtime),
            "worker_versions": dict(self.worker_versions),
            "wall_seconds": self.wall_seconds,
            "peak_gpu_mib_sampled": self.peak_gpu_mib_sampled,
            "commit_headroom_gib_at_launch": self.commit_headroom_gib_at_launch,
            "worker_attempts": self.worker_attempts,
            "prediction_file_sha256": dict(self.prediction_file_sha256),
            "tokenizer": dict(self.tokenizer) if self.tokenizer is not None else None,
            "item_failures": [item.to_dict() for item in self.item_failures],
            "item_inventory_source": self.item_inventory_source,
            "evidence_gaps": list(self.evidence_gaps),
        }


@dataclass(frozen=True)
class DimensionComparison:
    """One dimension's cross-role values plus every pairwise item-unit delta."""

    dimension: str
    values: Mapping[str, float | None]
    pairwise: Mapping[str, Mapping[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "values": dict(self.values),
            "pairwise": {key: dict(value) for key, value in self.pairwise.items()},
        }


def classify_delta(left: float | None, right: float | None) -> dict[str, Any]:
    if left is None or right is None:
        return {"delta": None, "classification": "unmeasured"}
    delta = round(right - left, 6)
    magnitude_items = abs(delta) * ITEMS_PER_SUITE
    if magnitude_items < _TIE_THRESHOLD_ITEMS:
        classification = "tie"
    elif magnitude_items < _WEAK_SIGNAL_THRESHOLD_ITEMS:
        classification = "weak-signal"
    else:
        classification = "clear-difference"
    return {"delta": delta, "classification": classification}


def compare_parents(evidence_by_role: Mapping[str, ParentEvidence]) -> tuple[DimensionComparison, ...]:
    """Pairwise, item-unit comparison across every present role, per dimension.

    Generalizes `parent_tournament.compare_reports` (which requires exactly
    two parents) to however many of A/B/C/D have evidence, without ever
    combining dimensions or roles into one scalar. Absent roles are simply
    omitted from `values`/`pairwise` for that dimension -- there is nothing
    to compare them against.
    """
    roles = sorted(evidence_by_role)
    comparisons: list[DimensionComparison] = []
    for dimension in PARENT_DIMENSIONS:
        values = {role: evidence_by_role[role].dimension_scores.get(dimension) for role in roles}
        pairwise: dict[str, dict[str, Any]] = {}
        for i, left_role in enumerate(roles):
            for right_role in roles[i + 1 :]:
                key = f"{left_role}_vs_{right_role}"
                pairwise[key] = classify_delta(values[left_role], values[right_role])
        comparisons.append(DimensionComparison(dimension=dimension, values=values, pairwise=pairwise))
    return tuple(comparisons)


@dataclass(frozen=True)
class ParentSelectionPacket:
    """The full, always-buildable evidence view across the bound roles."""

    generated_at: str
    role_bindings: Mapping[str, RoleBinding]
    expected_protocol_sha256: str
    parents: Mapping[str, ParentEvidence]
    missing_roles: tuple[str, ...]
    dimension_comparisons: tuple[DimensionComparison, ...]
    gate_results: tuple[GateResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "role_bindings": {role: binding.to_dict() for role, binding in self.role_bindings.items()},
            "expected_protocol_sha256": self.expected_protocol_sha256,
            "parents": {role: evidence.to_dict() for role, evidence in self.parents.items()},
            "missing_roles": list(self.missing_roles),
            "dimension_comparisons": [comparison.to_dict() for comparison in self.dimension_comparisons],
            "gate_results": [gate.to_dict() for gate in self.gate_results],
        }

    def canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def digest(self) -> str:
        return _digest(self.to_dict())

    def all_gates_passed(self) -> bool:
        return not self.missing_roles and all(gate.passed for gate in self.gate_results)


class _RegistryRow:
    """Normalized view of one `EvaluationOutcome`-shaped registry row."""

    __slots__ = ("run_id", "experiment_id", "source_artifact_ref", "metrics", "evidence")

    def __init__(self, run_id: str, experiment_id: str, source_artifact_ref: str,
                 metrics: Mapping[str, Any], evidence: Mapping[str, Any]) -> None:
        self.run_id = run_id
        self.experiment_id = experiment_id
        self.source_artifact_ref = source_artifact_ref
        self.metrics = metrics
        self.evidence = evidence


def _rows_for_label(registry: Any, label: str) -> list[_RegistryRow]:
    """Every persisted tournament row for `label`.

    Matched by the same `parent_program` marker `aggregate_parent_result`
    stamps onto every tournament report, combined with *either* the
    nested report's `base_model` (the semantic match) *or* the
    `exp-parent-baseline-{label}` experiment_id convention
    `parent_tournament.evaluate_parent` uses (a structural match). The
    structural fallback matters: a row whose evidence payload is
    corrupted or schema-drifted (no readable `base_model`) would
    otherwise be silently invisible -- indistinguishable from "this
    parent was never evaluated" -- instead of surfacing as the malformed
    row it is.
    """
    matches: list[_RegistryRow] = []
    expected_experiment_id = f"exp-parent-baseline-{label}"
    for outcome in registry.list_evaluation_outcomes():
        evidence = outcome.evidence
        if not isinstance(evidence, Mapping):
            continue
        if evidence.get("parent_program") != _TOURNAMENT_EVIDENCE_TAG:
            continue
        report = evidence.get("parent_eval_report")
        base_model = report.get("base_model") if isinstance(report, Mapping) else None
        if base_model != label and outcome.experiment_id != expected_experiment_id:
            continue
        matches.append(
            _RegistryRow(
                run_id=outcome.run_id,
                experiment_id=outcome.experiment_id,
                source_artifact_ref=outcome.source_artifact_ref,
                metrics=outcome.metrics,
                evidence=evidence,
            )
        )
    return matches


def _dedupe_or_raise(label: str, rows: Sequence[_RegistryRow]) -> _RegistryRow:
    if len(rows) == 1:
        return rows[0]
    canonical = {_digest(row.evidence) for row in rows}
    if len(canonical) == 1:
        # identical replays: idempotent, same discipline as
        # RunRegistry._insert_immutable's replay tolerance.
        return rows[0]
    raise DuplicateParentEvidenceError(
        f"parent {label!r} has {len(rows)} tournament rows with mutually "
        f"inconsistent evidence (run_ids: {[row.run_id for row in rows]}); "
        "a parent-selection decision requires exactly one authoritative "
        "result per parent, so this refuses to guess which is correct"
    )


def _read_item_inventory(
    run_dir: str,
    suite_digests: Mapping[str, str],
    prediction_file_sha256: Mapping[str, str],
) -> tuple[tuple[ItemClassification, ...], str]:
    """Best-effort, read-only item-level failure inventory.

    Returns `([], "unavailable")` whenever the run directory, or any
    individual prediction file, cannot be trusted (missing, or its current
    sha256 no longer matches what the tournament recorded at run time) --
    never a partial or silently-approximate inventory.
    """
    root = Path(run_dir)
    if not root.is_dir():
        return (), "unavailable"
    failures: list[ItemClassification] = []
    for filename, recorded_sha256 in sorted(prediction_file_sha256.items()):
        path = root / filename
        if not path.is_file():
            return (), "unavailable"
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != recorded_sha256:
            return (), "unavailable"
        suite_name = filename[len("predictions-") : -len(".jsonl")] if filename.startswith("predictions-") else filename
        with path.open("r", encoding="utf-8") as handle:
            for row_index, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                score = float(row.get("score", 0.0))
                if score >= 1.0:
                    continue
                prediction = str(row.get("prediction", ""))
                # A response that never closes its thinking span within the
                # generation budget is a truncation/infrastructure
                # candidate, not evidence the model reasoned to a wrong
                # answer -- heuristic, not certain, and reported as such.
                likely_truncation = "</think>" not in prediction
                failures.append(
                    ItemClassification(
                        suite=suite_name,
                        row_index=row_index,
                        prompt_sha256=hashlib.sha256(str(row.get("prompt", "")).encode("utf-8")).hexdigest(),
                        score=score,
                        likely_truncation=likely_truncation,
                    )
                )
    return tuple(failures), "predictions_files"


def _build_parent_evidence(
    binding: RoleBinding,
    row: _RegistryRow,
    *,
    expected_protocol_sha256: str,
    tokenizer_evidence: Mapping[str, Any] | None,
    read_item_inventory: bool,
) -> ParentEvidence:
    evidence = row.evidence
    report = evidence.get("parent_eval_report")
    if not isinstance(report, Mapping):
        raise MalformedEvidenceError(f"parent {binding.label!r} row {row.run_id!r} has no parent_eval_report")
    protocol_sha256 = evidence.get("evaluation_protocol_sha256") or report.get("evaluation_protocol_sha256")
    if not isinstance(protocol_sha256, str) or len(protocol_sha256) != 64:
        raise MalformedEvidenceError(
            f"parent {binding.label!r} row {row.run_id!r} has no evaluation_protocol_sha256"
        )
    dimensions = report.get("dimensions")
    if not isinstance(dimensions, Mapping):
        raise MalformedEvidenceError(f"parent {binding.label!r} row {row.run_id!r} has no dimensions table")

    dimension_scores: dict[str, float | None] = {}
    dimension_suite_counts: dict[str, int] = {}
    suite_digests: dict[str, str] = {}
    suite_evidence = evidence.get("suite_evidence")
    if not isinstance(suite_evidence, Mapping):
        suite_evidence = {}
    for dim in PARENT_DIMENSIONS:
        entry = dimensions.get(dim)
        if not isinstance(entry, Mapping):
            dimension_scores[dim] = None
            dimension_suite_counts[dim] = 0
            continue
        dimension_scores[dim] = entry.get("mean")
        suite_metrics = entry.get("suite_metrics") or {}
        dimension_suite_counts[dim] = len(suite_metrics)
        for suite_name in suite_metrics:
            suite_info = suite_evidence.get(suite_name)
            if isinstance(suite_info, Mapping):
                digest = suite_info.get("holdout_fingerprints_sha256")
                if isinstance(digest, str):
                    suite_digests[suite_name] = digest

    revision = report.get("revision")
    gaps: list[str] = []
    if "commit_headroom_gib_at_launch" not in evidence:
        gaps.append(
            "commit_headroom_gib_at_launch: computed by parent_tournament._run_worker "
            "at launch time but not copied into the persisted evidence dict "
            "(evaluate_parent only forwards wall_seconds/peak_gpu_mib_sampled/"
            "worker_runtime/worker_versions/suite_evidence); not recoverable "
            "post hoc for an already-completed run"
        )
    if "worker_attempts" not in evidence:
        gaps.append(
            "worker_attempts: same gap as commit_headroom_gib_at_launch -- "
            "computed in memory, never persisted"
        )
    if tokenizer_evidence is None:
        gaps.append(
            "tokenizer: not persisted in evaluation_runs evidence; the caller "
            "did not supply tokenizer_evidence (measured via the unmodified "
            "parent_tournament.tokenizer_evidence) for this role"
        )

    prediction_file_sha256 = evidence.get("prediction_file_sha256") or {}
    item_failures: tuple[ItemClassification, ...] = ()
    item_inventory_source = "unavailable"
    if read_item_inventory and isinstance(prediction_file_sha256, Mapping) and prediction_file_sha256:
        item_failures, item_inventory_source = _read_item_inventory(
            row.source_artifact_ref, suite_digests, prediction_file_sha256
        )

    return ParentEvidence(
        role=binding.role,
        label=binding.label,
        run_id=row.run_id,
        experiment_id=row.experiment_id,
        revision=revision,
        revision_matches_pin=(revision == binding.expected_revision),
        evaluation_protocol_sha256=protocol_sha256,
        protocol_matches_reference=(protocol_sha256 == expected_protocol_sha256),
        model_manifest_sha256=evidence.get("model_manifest_sha256"),
        suite_digests=suite_digests,
        dimension_scores=dimension_scores,
        dimension_suite_counts=dimension_suite_counts,
        capability_mean=report.get("capability_mean"),
        behavior_mean=report.get("behavior_mean"),
        worker_runtime=evidence.get("worker_runtime") or {},
        worker_versions=evidence.get("worker_versions") or {},
        wall_seconds=evidence.get("wall_seconds"),
        peak_gpu_mib_sampled=evidence.get("peak_gpu_mib_sampled"),
        commit_headroom_gib_at_launch=evidence.get("commit_headroom_gib_at_launch"),
        worker_attempts=evidence.get("worker_attempts"),
        prediction_file_sha256=dict(prediction_file_sha256) if isinstance(prediction_file_sha256, Mapping) else {},
        tokenizer=dict(tokenizer_evidence) if tokenizer_evidence is not None else None,
        item_failures=item_failures,
        item_inventory_source=item_inventory_source,
        evidence_gaps=tuple(gaps),
    )


def _dimension_coverage_gate(evidence: ParentEvidence) -> GateResult:
    missing = [dim for dim in PARENT_DIMENSIONS if evidence.dimension_scores.get(dim) is None]
    wrong_count = [
        dim for dim in PARENT_DIMENSIONS
        if evidence.dimension_scores.get(dim) is not None and evidence.dimension_suite_counts.get(dim) != 1
    ]
    if missing or wrong_count:
        return GateResult(
            name=f"dimension_coverage[{evidence.role}]",
            passed=False,
            detail=(
                f"missing dimensions: {missing or 'none'}; dimensions with an "
                f"unexpected suite count (expected exactly 1 per dimension, "
                f"6 items each -- protected suite v1's authored shape): {wrong_count or 'none'}"
            ),
        )
    return GateResult(name=f"dimension_coverage[{evidence.role}]", passed=True, detail="9/9 dimensions, 1 suite each")


def _revision_gate(evidence: ParentEvidence) -> GateResult:
    return GateResult(
        name=f"revision_pin[{evidence.role}]",
        passed=evidence.revision_matches_pin,
        detail=(
            "recorded revision matches the pinned revision" if evidence.revision_matches_pin
            else f"recorded revision {evidence.revision!r} does not match the pinned revision"
        ),
    )


def _protocol_gate(evidence: ParentEvidence, expected_protocol_sha256: str) -> GateResult:
    return GateResult(
        name=f"protocol_digest[{evidence.role}]",
        passed=evidence.protocol_matches_reference,
        detail=(
            "matches the reference protocol digest" if evidence.protocol_matches_reference
            else (
                f"recorded {evidence.evaluation_protocol_sha256[:12]}... does not match "
                f"reference {expected_protocol_sha256[:12]}... (a run scored under a "
                "different generation budget or suite definition -- including retry6/"
                "protocol v1 -- produces a different digest here)"
            )
        ),
    )


def _suite_digest_gate(evidence_by_role: Mapping[str, ParentEvidence]) -> GateResult:
    by_suite: dict[str, dict[str, str]] = {}
    for role, evidence in evidence_by_role.items():
        for suite_name, digest in evidence.suite_digests.items():
            by_suite.setdefault(suite_name, {})[role] = digest
    conflicts = {
        suite_name: roles
        for suite_name, roles in by_suite.items()
        if len(set(roles.values())) > 1
    }
    if conflicts:
        return GateResult(
            name="suite_content_digest",
            passed=False,
            detail=f"protected-suite content digest disagrees across roles for: {sorted(conflicts)}",
        )
    return GateResult(name="suite_content_digest", passed=True, detail="every shared suite has one content digest")


def _tokenizer_gate(
    evidence_by_role: Mapping[str, ParentEvidence], *, allow_missing_tokenizer_evidence: bool
) -> GateResult:
    missing = [role for role, evidence in evidence_by_role.items() if evidence.tokenizer is None]
    if missing:
        if allow_missing_tokenizer_evidence:
            return GateResult(
                name="tokenizer_comparability",
                passed=True,
                detail=(
                    f"tokenizer evidence missing for {missing} but the gate was "
                    "explicitly bypassed via allow_missing_tokenizer_evidence=True"
                ),
            )
        return GateResult(
            name="tokenizer_comparability",
            passed=False,
            detail=f"tokenizer evidence missing for role(s): {missing}",
        )
    fields = ("tokenizer_class", "vocab_size", "identity_sha256")
    reference_role = sorted(evidence_by_role)[0]
    reference = evidence_by_role[reference_role].tokenizer
    mismatches = []
    for role, evidence in evidence_by_role.items():
        for key in fields:
            if evidence.tokenizer.get(key) != reference.get(key):
                mismatches.append(f"{role}.{key}={evidence.tokenizer.get(key)!r}")
    if mismatches:
        return GateResult(
            name="tokenizer_comparability",
            passed=False,
            detail=f"tokenizer identity diverges from {reference_role}'s: {mismatches}",
        )
    return GateResult(name="tokenizer_comparability", passed=True, detail="tokenizer identity matches across all roles")


def build_selection_packet(
    registry: Any,
    *,
    role_bindings: Mapping[str, RoleBinding],
    expected_protocol_sha256: str,
    tokenizer_evidence: Mapping[str, Mapping[str, Any]] | None = None,
    allow_missing_tokenizer_evidence: bool = False,
    read_item_inventory: bool = True,
    now: str | None = None,
) -> ParentSelectionPacket:
    """Build the full evidence packet from whatever is currently persisted.

    Never raises for missing or inconsistent evidence -- every check's
    result is recorded as a `GateResult` instead, so a partial tournament
    (e.g. only A and B done) still produces an inspectable packet. Only
    `freeze_selected_parent` (below) treats a failing gate as fatal.

    `expected_protocol_sha256` should be computed by the caller from the
    live, current `parent_suite_content.build_tournament_spec(frozen_root).digest()`
    (or, in tests, a synthetic reference digest) -- never hardcoded here.
    """
    if set(role_bindings) != set(PARENT_ROLES):
        raise ValueError(f"role_bindings must cover exactly {PARENT_ROLES}, got {sorted(role_bindings)}")
    if not isinstance(expected_protocol_sha256, str) or len(expected_protocol_sha256) != 64:
        raise ValueError("expected_protocol_sha256 must be a 64-character sha256 hex digest")

    parents: dict[str, ParentEvidence] = {}
    missing_roles: list[str] = []
    for role in PARENT_ROLES:
        binding = role_bindings[role]
        rows = _rows_for_label(registry, binding.label)
        if not rows:
            missing_roles.append(role)
            continue
        row = _dedupe_or_raise(binding.label, rows)
        tokenizer = None
        if tokenizer_evidence is not None:
            tokenizer = tokenizer_evidence.get(role)
        parents[role] = _build_parent_evidence(
            binding,
            row,
            expected_protocol_sha256=expected_protocol_sha256,
            tokenizer_evidence=tokenizer,
            read_item_inventory=read_item_inventory,
        )

    gate_results: list[GateResult] = []
    for role, evidence in parents.items():
        gate_results.append(_dimension_coverage_gate(evidence))
        gate_results.append(_revision_gate(evidence))
        gate_results.append(_protocol_gate(evidence, expected_protocol_sha256))
    if parents:
        gate_results.append(_suite_digest_gate(parents))
        gate_results.append(_tokenizer_gate(parents, allow_missing_tokenizer_evidence=allow_missing_tokenizer_evidence))
    gate_results.append(
        GateResult(
            name="all_roles_present",
            passed=not missing_roles,
            detail="all four roles have evidence" if not missing_roles else f"missing role(s): {missing_roles}",
        )
    )

    return ParentSelectionPacket(
        generated_at=now or _utcnow(),
        role_bindings=dict(role_bindings),
        expected_protocol_sha256=expected_protocol_sha256,
        parents=parents,
        missing_roles=tuple(missing_roles),
        dimension_comparisons=compare_parents(parents),
        gate_results=tuple(gate_results),
    )


@dataclass(frozen=True)
class ParentFreezeRecord:
    """The single machine-readable "select this baseline" artifact."""

    packet_digest: str
    frozen_at: str
    selected_role: str | None
    selected_label: str | None
    rationale: str
    dimension_rationale: Mapping[str, str]
    packet: ParentSelectionPacket

    def to_dict(self) -> dict[str, Any]:
        return {
            "packet_digest": self.packet_digest,
            "frozen_at": self.frozen_at,
            "selected_role": self.selected_role,
            "selected_label": self.selected_label,
            "rationale": self.rationale,
            "dimension_rationale": dict(self.dimension_rationale),
            "packet": self.packet.to_dict(),
        }

    def canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def digest(self) -> str:
        return _digest(self.to_dict())


def _find_comparison(packet: ParentSelectionPacket, dimension: str) -> DimensionComparison:
    for comparison in packet.dimension_comparisons:
        if comparison.dimension == dimension:
            return comparison
    raise MalformedEvidenceError(f"packet has no comparison entry for dimension {dimension!r}")


def default_decision_rule(packet: ParentSelectionPacket) -> tuple[str | None, str, Mapping[str, str]]:
    """Reference-relative, item-unit, testable default rule.

    Never blends capability and behavior, and never ranks by a combined
    scalar. Recommends B (the documented primary development parent, per
    `docs/QWEN38_SPARSE_PROGRAM.md`) unless the evidence shows a
    clear-difference *regression* against the native control A on any
    capability or behavior dimension; in that case it looks for a
    candidate among {B, C, D} with no such regression. If none is unique,
    it returns `None` (no automatic selection) rather than guessing, and
    every dimension's rationale is recorded either way.

    Returns (selected_role_or_None, rationale, per_dimension_rationale).
    """
    reference = "A"
    dimension_rationale: dict[str, str] = {}
    regressions: dict[str, list[str]] = {"B": [], "C": [], "D": []}
    for dimension in PARENT_DIMENSIONS:
        comparison = _find_comparison(packet, dimension)
        for candidate in ("B", "C", "D"):
            # `reference` ("A") sorts first alphabetically among A/B/C/D, so
            # `compare_parents`'s pairwise key is always f"A_vs_{candidate}"
            # with delta = candidate_value - reference_value (see
            # `classify_delta`'s left/right convention) -- never the
            # reverse pairing.
            key = f"{reference}_vs_{candidate}"
            entry = comparison.pairwise.get(key)
            if entry is None:
                continue
            classification = entry["classification"]
            delta = entry["delta"]
            if classification != "clear-difference" or delta is None:
                continue
            if delta < 0:
                regressions[candidate].append(dimension)
        dimension_rationale[dimension] = json.dumps(comparison.to_dict()["pairwise"], sort_keys=True)

    clean_candidates = [candidate for candidate in ("B", "C", "D") if not regressions[candidate]]
    if "B" in clean_candidates:
        return (
            "B",
            "B (documented primary development parent) shows no clear-difference "
            "capability or behavior regression against native control A; recommended "
            "as the frozen baseline.",
            dimension_rationale,
        )
    if len(clean_candidates) == 1:
        only = clean_candidates[0]
        return (
            only,
            f"B shows a clear-difference regression against A on {regressions['B']}; "
            f"{only} is the only candidate among B/C/D with no such regression, so it "
            "is recommended instead.",
            dimension_rationale,
        )
    return (
        None,
        "no automatic selection: B shows a clear-difference regression against A on "
        f"{regressions['B']}, and no single alternative among B/C/D is clean "
        f"(regressions: {regressions}); this requires human review rather than a "
        "guessed pick.",
        dimension_rationale,
    )


def validate_packet_for_freeze(packet: ParentSelectionPacket) -> None:
    """Raise the first violated fail-closed gate, or return None when clean."""
    if packet.missing_roles:
        raise MissingParentEvidenceError(f"missing evidence for role(s): {packet.missing_roles}")
    for gate in packet.gate_results:
        if gate.passed:
            continue
        if gate.name.startswith("dimension_coverage"):
            raise IncompleteDimensionCoverageError(gate.detail)
        if gate.name.startswith("revision_pin"):
            raise RevisionMismatchError(gate.detail)
        if gate.name.startswith("protocol_digest"):
            raise ProtocolMismatchError(gate.detail)
        if gate.name == "suite_content_digest":
            raise SuiteContentMismatchError(gate.detail)
        if gate.name == "tokenizer_comparability":
            raise TokenizerComparabilityError(gate.detail)
        if gate.name == "all_roles_present":
            raise MissingParentEvidenceError(gate.detail)
        raise ParentFreezeError(f"{gate.name}: {gate.detail}")


def freeze_selected_parent(
    packet: ParentSelectionPacket,
    *,
    decision_rule: Callable[[ParentSelectionPacket], tuple[str | None, str, Mapping[str, str]]] | None = None,
    now: str | None = None,
) -> ParentFreezeRecord:
    """Produce the frozen baseline record, or raise the first failed gate.

    `decision_rule` defaults to `default_decision_rule`. A caller may
    substitute another explicit, testable rule; this function does not
    itself invent a ranking.
    """
    validate_packet_for_freeze(packet)
    rule = decision_rule or default_decision_rule
    selected_role, rationale, dimension_rationale = rule(packet)
    selected_label = packet.parents[selected_role].label if selected_role is not None else None
    return ParentFreezeRecord(
        packet_digest=packet.digest(),
        frozen_at=now or _utcnow(),
        selected_role=selected_role,
        selected_label=selected_label,
        rationale=rationale,
        dimension_rationale=dimension_rationale,
        packet=packet,
    )
