"""Local-vs-Kaggle backend equivalence qualification (Qwen3.8 program, Phase 3).

Kaggle T4 results for parents C/D may never enter the same four-parent
tournament comparison as local A/B results until this module says so.
This module builds the item-level, dimension-level, and environment-level
comparison the mission requires, and turns it into a machine-readable
`BackendQualificationRecord` -- it performs no evaluation itself and never
touches the live retry7 registry or output directory.

Reused, not reinvented
------------------------
- `evaluators.base_text_worker._final_answer` and `._normalize` -- the
  exact thinking-aware answer-extraction and scoring-normalization
  functions the real worker already uses. This module imports them
  directly (they are Chowder-internal, not re-implemented) rather than
  re-deriving equivalent logic that could silently drift from the real
  scorer.
- `parent_eval.PARENT_DIMENSIONS` for the nine-dimension order.
- `parent_freeze.ITEMS_PER_SUITE` / `parent_freeze.classify_delta` for the
  exact tie/weak-signal/clear-difference item-unit convention already
  established for the four-parent comparison, so a dimension delta means
  the same thing here as it does in the tournament freeze packet.

A known, hardware-forced divergence this module expects and handles
----------------------------------------------------------------------
`evaluators.base_text_worker._dtype` raises when `precision="bf16"` is
requested on a device where `torch.cuda.is_bf16_supported()` is False.
T4 GPUs (Kaggle's free-tier accelerator, compute capability 7.5) do not
support bf16 tensor cores -- confirmed by reading that check directly,
not assumed. A byte-identical protocol digest between a local bf16 run
and a Kaggle run is therefore *impossible* while reusing the unmodified
worker unchanged, because `precision` is itself part of
`ParentEvalSpec.to_dict()` and therefore part of the digest. This module
does not paper over that: a protocol digest mismatch is fatal to
qualification *unless* the caller explicitly declares which spec fields
are expected to differ and why (`declared_digest_divergence_reasons`),
and every other gate (item scores, suite content digest, tokenizer
identity, dimension totals) still has to match exactly. This is the only
sanctioned exception; this module manufactures no other leniency and
"fixes" nothing about the Kaggle run's generation settings.

Multi-GPU note
--------------
`base_text_worker.evaluate` pins the whole (quantized) model onto exactly
one CUDA device index (`model_kwargs["device_map"] = {"": index}`); it
has no code path that splits a model across two GPUs. A real Kaggle
qualification or evaluation run therefore targets a single T4, not both
at once -- this module's `BackendFingerprint.gpu_count` field records
whatever was actually used, honestly, rather than assuming Kaggle's
"T4 x2" allocation implies two-GPU inference.

Protected-content handling
---------------------------
`ItemComparison` never carries raw prompt/expected/prediction text --
only a prompt hash and the extracted-answer/score comparison outcome,
mirroring `parent_freeze.ItemClassification`'s hash-only discipline. Raw
text may be used transiently by the comparison functions (which read
real `predictions-*.jsonl` rows) but is never carried into a returned
dataclass or its `to_dict()`.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .evaluators.base_text_worker import _final_answer, _normalize
from .parent_eval import PARENT_DIMENSIONS
from .parent_freeze import ITEMS_PER_SUITE, classify_delta


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class BackendEquivalenceError(ValueError):
    """The equivalence comparison cannot be built or qualified honestly."""


@dataclass(frozen=True)
class BackendFingerprint:
    """Every environment field the mission requires recorded, honestly.

    `gpu_models`/`gpu_count` describe what was *actually used* for this
    run, not the accelerator type Kaggle offered -- a single-T4 run
    records `gpu_count=1`, even under a "T4 x2" Kaggle allocation.
    """

    python_version: str
    torch_version: str
    transformers_version: str
    bitsandbytes_version: str
    accelerate_version: str
    cuda_runtime_version: str | None
    gpu_models: tuple[str, ...]
    gpu_count: int
    device_map_summary: str
    quantization: str
    dtype: str
    tokenizer_identity_sha256: str | None
    chowder_commit_sha: str

    def __post_init__(self) -> None:
        if isinstance(self.gpu_count, bool) or not isinstance(self.gpu_count, int) or self.gpu_count < 0:
            raise ValueError("gpu_count must be a non-negative int")
        if len(self.chowder_commit_sha) != 40 or any(
            c not in "0123456789abcdef" for c in self.chowder_commit_sha.lower()
        ):
            raise ValueError("chowder_commit_sha must be a full 40-character commit sha")

    def to_dict(self) -> dict[str, Any]:
        return {
            "python_version": self.python_version,
            "torch_version": self.torch_version,
            "transformers_version": self.transformers_version,
            "bitsandbytes_version": self.bitsandbytes_version,
            "accelerate_version": self.accelerate_version,
            "cuda_runtime_version": self.cuda_runtime_version,
            "gpu_models": list(self.gpu_models),
            "gpu_count": self.gpu_count,
            "device_map_summary": self.device_map_summary,
            "quantization": self.quantization,
            "dtype": self.dtype,
            "tokenizer_identity_sha256": self.tokenizer_identity_sha256,
            "chowder_commit_sha": self.chowder_commit_sha,
        }


@dataclass(frozen=True)
class ItemComparison:
    """One protected item's local-vs-Kaggle outcome. Hash-only prompt id."""

    suite: str
    row_index: int
    prompt_sha256: str
    local_score: float
    kaggle_score: float
    scores_equal: bool
    answers_equal: bool
    normalized_answers_equal: bool
    local_truncated: bool
    kaggle_truncated: bool
    truncation_difference: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "row_index": self.row_index,
            "prompt_sha256": self.prompt_sha256,
            "local_score": self.local_score,
            "kaggle_score": self.kaggle_score,
            "scores_equal": self.scores_equal,
            "answers_equal": self.answers_equal,
            "normalized_answers_equal": self.normalized_answers_equal,
            "local_truncated": self.local_truncated,
            "kaggle_truncated": self.kaggle_truncated,
            "truncation_difference": self.truncation_difference,
        }


def _is_truncated(prediction: str) -> bool:
    """Same heuristic `parent_freeze._read_item_inventory` uses: a
    response that never closes its thinking span within the generation
    budget is a truncation candidate, not a considered wrong answer."""
    return "<think>" in prediction and "</think>" not in prediction


def _read_predictions(directory: Path) -> dict[str, list[dict[str, Any]]]:
    """suite_name -> ordered list of {prompt, expected, prediction, score} rows."""
    by_suite: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(directory.glob("predictions-*.jsonl")):
        suite_name = path.name[len("predictions-") : -len(".jsonl")]
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        by_suite[suite_name] = rows
    return by_suite


def compare_items(
    local_predictions_dir: str | Path, kaggle_predictions_dir: str | Path
) -> tuple[ItemComparison, ...]:
    """Item-level local-vs-Kaggle comparison, read directly from each
    side's real `predictions-*.jsonl` files.

    Items are matched positionally within each shared suite name (the
    same suite, evaluated under the same frozen dataset ordering on both
    backends) -- a suite present on only one side is a structural
    mismatch this function refuses rather than silently skipping.
    """
    local_by_suite = _read_predictions(Path(local_predictions_dir))
    kaggle_by_suite = _read_predictions(Path(kaggle_predictions_dir))
    if set(local_by_suite) != set(kaggle_by_suite):
        raise BackendEquivalenceError(
            f"suite sets differ between local and Kaggle predictions: "
            f"local only {sorted(set(local_by_suite) - set(kaggle_by_suite))}, "
            f"kaggle only {sorted(set(kaggle_by_suite) - set(local_by_suite))}"
        )
    comparisons: list[ItemComparison] = []
    for suite_name in sorted(local_by_suite):
        local_rows = local_by_suite[suite_name]
        kaggle_rows = kaggle_by_suite[suite_name]
        if len(local_rows) != len(kaggle_rows):
            raise BackendEquivalenceError(
                f"suite {suite_name!r} has {len(local_rows)} local item(s) but "
                f"{len(kaggle_rows)} Kaggle item(s); cannot compare positionally"
            )
        for row_index, (local_row, kaggle_row) in enumerate(zip(local_rows, kaggle_rows)):
            local_prompt = str(local_row.get("prompt", ""))
            kaggle_prompt = str(kaggle_row.get("prompt", ""))
            if local_prompt != kaggle_prompt:
                raise BackendEquivalenceError(
                    f"suite {suite_name!r} item {row_index}: prompt text differs between "
                    "local and Kaggle -- the two backends evaluated different content, "
                    "positional score comparison would be meaningless"
                )
            local_prediction = str(local_row.get("prediction", ""))
            kaggle_prediction = str(kaggle_row.get("prediction", ""))
            local_answer = _final_answer(local_prediction)
            kaggle_answer = _final_answer(kaggle_prediction)
            local_score = float(local_row.get("score", 0.0))
            kaggle_score = float(kaggle_row.get("score", 0.0))
            comparisons.append(
                ItemComparison(
                    suite=suite_name,
                    row_index=row_index,
                    prompt_sha256=hashlib.sha256(local_prompt.encode("utf-8")).hexdigest(),
                    local_score=local_score,
                    kaggle_score=kaggle_score,
                    scores_equal=(local_score == kaggle_score),
                    answers_equal=(local_answer == kaggle_answer),
                    normalized_answers_equal=(_normalize(local_answer) == _normalize(kaggle_answer)),
                    local_truncated=_is_truncated(local_prediction),
                    kaggle_truncated=_is_truncated(kaggle_prediction),
                    truncation_difference=(_is_truncated(local_prediction) != _is_truncated(kaggle_prediction)),
                )
            )
    return tuple(comparisons)


@dataclass(frozen=True)
class DimensionEquivalence:
    dimension: str
    local_mean: float | None
    kaggle_mean: float | None
    delta: float | None
    classification: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "local_mean": self.local_mean,
            "kaggle_mean": self.kaggle_mean,
            "delta": self.delta,
            "classification": self.classification,
        }


def compare_dimensions(
    local_report: Mapping[str, Any], kaggle_report: Mapping[str, Any]
) -> tuple[DimensionEquivalence, ...]:
    local_dims = local_report.get("dimensions") or {}
    kaggle_dims = kaggle_report.get("dimensions") or {}
    comparisons = []
    for dimension in PARENT_DIMENSIONS:
        local_mean = (local_dims.get(dimension) or {}).get("mean")
        kaggle_mean = (kaggle_dims.get(dimension) or {}).get("mean")
        outcome = classify_delta(local_mean, kaggle_mean)
        comparisons.append(
            DimensionEquivalence(
                dimension=dimension,
                local_mean=local_mean,
                kaggle_mean=kaggle_mean,
                delta=outcome["delta"],
                classification=outcome["classification"],
            )
        )
    return tuple(comparisons)


@dataclass(frozen=True)
class EquivalenceReport:
    """The full local-vs-Kaggle comparison for one parent's qualification run."""

    parent_label: str
    generated_at: str
    local_protocol_sha256: str
    kaggle_protocol_sha256: str
    protocol_digest_equal: bool
    local_suite_digests: Mapping[str, str]
    kaggle_suite_digests: Mapping[str, str]
    suite_digest_equal: bool
    local_tokenizer: Mapping[str, Any] | None
    kaggle_tokenizer: Mapping[str, Any] | None
    tokenizer_identity_equal: bool
    item_comparisons: tuple[ItemComparison, ...]
    dimension_comparisons: tuple[DimensionEquivalence, ...]
    local_capability_mean: float | None
    kaggle_capability_mean: float | None
    local_behavior_mean: float | None
    kaggle_behavior_mean: float | None
    declared_digest_divergence_reasons: tuple[str, ...]
    local_environment: BackendFingerprint
    kaggle_environment: BackendFingerprint

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_label": self.parent_label,
            "generated_at": self.generated_at,
            "local_protocol_sha256": self.local_protocol_sha256,
            "kaggle_protocol_sha256": self.kaggle_protocol_sha256,
            "protocol_digest_equal": self.protocol_digest_equal,
            "local_suite_digests": dict(self.local_suite_digests),
            "kaggle_suite_digests": dict(self.kaggle_suite_digests),
            "suite_digest_equal": self.suite_digest_equal,
            "local_tokenizer": dict(self.local_tokenizer) if self.local_tokenizer else None,
            "kaggle_tokenizer": dict(self.kaggle_tokenizer) if self.kaggle_tokenizer else None,
            "tokenizer_identity_equal": self.tokenizer_identity_equal,
            "item_comparisons": [item.to_dict() for item in self.item_comparisons],
            "dimension_comparisons": [dim.to_dict() for dim in self.dimension_comparisons],
            "local_capability_mean": self.local_capability_mean,
            "kaggle_capability_mean": self.kaggle_capability_mean,
            "local_behavior_mean": self.local_behavior_mean,
            "kaggle_behavior_mean": self.kaggle_behavior_mean,
            "declared_digest_divergence_reasons": list(self.declared_digest_divergence_reasons),
            "local_environment": self.local_environment.to_dict(),
            "kaggle_environment": self.kaggle_environment.to_dict(),
            "item_score_agreement_count": sum(1 for item in self.item_comparisons if item.scores_equal),
            "item_answer_agreement_count": sum(1 for item in self.item_comparisons if item.answers_equal),
            "item_total_count": len(self.item_comparisons),
        }

    def canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def digest(self) -> str:
        return _digest(self.to_dict())

    @property
    def item_total_count(self) -> int:
        return len(self.item_comparisons)

    @property
    def all_scores_equal(self) -> bool:
        return bool(self.item_comparisons) and all(item.scores_equal for item in self.item_comparisons)

    @property
    def all_answers_equal(self) -> bool:
        return bool(self.item_comparisons) and all(item.answers_equal for item in self.item_comparisons)

    @property
    def any_systematic_truncation_difference(self) -> bool:
        """More than one item disagreeing on truncation is treated as
        systematic (a single isolated case is reported but not, by
        itself, proof of a systematic generation-budget difference)."""
        return sum(1 for item in self.item_comparisons if item.truncation_difference) > 1


def _suite_digests_from_evidence(report: Mapping[str, Any], suite_evidence: Mapping[str, Any]) -> dict[str, str]:
    digests: dict[str, str] = {}
    dimensions = report.get("dimensions") or {}
    for entry in dimensions.values():
        for suite_name in (entry or {}).get("suite_metrics", {}):
            info = suite_evidence.get(suite_name)
            if isinstance(info, Mapping):
                digest = info.get("holdout_fingerprints_sha256")
                if isinstance(digest, str):
                    digests[suite_name] = digest
    return digests


def build_equivalence_report(
    *,
    parent_label: str,
    local_report: Mapping[str, Any],
    kaggle_report: Mapping[str, Any],
    local_suite_evidence: Mapping[str, Any],
    kaggle_suite_evidence: Mapping[str, Any],
    local_predictions_dir: str | Path,
    kaggle_predictions_dir: str | Path,
    local_tokenizer: Mapping[str, Any] | None,
    kaggle_tokenizer: Mapping[str, Any] | None,
    local_environment: BackendFingerprint,
    kaggle_environment: BackendFingerprint,
    declared_digest_divergence_reasons: Sequence[str] = (),
    now: str | None = None,
) -> EquivalenceReport:
    """Build the full equivalence report from two completed parent runs.

    `local_report`/`kaggle_report` are `ParentEvalReport.to_dict()`-shaped
    (the same evidence shape `parent_freeze.py` consumes).
    `*_suite_evidence` is each run's `evidence["suite_evidence"]` (per-suite
    `holdout_fingerprints_sha256`, exactly as `parent_tournament.evaluate_parent`
    records it). This function performs no scoring of its own -- every
    score comes from what each backend's own (unmodified) worker already
    recorded.
    """
    local_protocol = local_report.get("evaluation_protocol_sha256")
    kaggle_protocol = kaggle_report.get("evaluation_protocol_sha256")
    if not isinstance(local_protocol, str) or not isinstance(kaggle_protocol, str):
        raise BackendEquivalenceError("both reports must carry evaluation_protocol_sha256")

    local_suite_digests = _suite_digests_from_evidence(local_report, local_suite_evidence)
    kaggle_suite_digests = _suite_digests_from_evidence(kaggle_report, kaggle_suite_evidence)
    shared_suites = set(local_suite_digests) & set(kaggle_suite_digests)
    suite_digest_equal = bool(shared_suites) and all(
        local_suite_digests[name] == kaggle_suite_digests[name] for name in shared_suites
    )

    tokenizer_identity_equal = False
    if local_tokenizer is not None and kaggle_tokenizer is not None:
        tokenizer_identity_equal = all(
            local_tokenizer.get(key) == kaggle_tokenizer.get(key)
            for key in ("tokenizer_class", "vocab_size", "identity_sha256")
        )

    return EquivalenceReport(
        parent_label=parent_label,
        generated_at=now or _utcnow(),
        local_protocol_sha256=local_protocol,
        kaggle_protocol_sha256=kaggle_protocol,
        protocol_digest_equal=(local_protocol == kaggle_protocol),
        local_suite_digests=local_suite_digests,
        kaggle_suite_digests=kaggle_suite_digests,
        suite_digest_equal=suite_digest_equal,
        local_tokenizer=local_tokenizer,
        kaggle_tokenizer=kaggle_tokenizer,
        tokenizer_identity_equal=tokenizer_identity_equal,
        item_comparisons=compare_items(local_predictions_dir, kaggle_predictions_dir),
        dimension_comparisons=compare_dimensions(local_report, kaggle_report),
        local_capability_mean=local_report.get("capability_mean"),
        kaggle_capability_mean=kaggle_report.get("capability_mean"),
        local_behavior_mean=local_report.get("behavior_mean"),
        kaggle_behavior_mean=kaggle_report.get("behavior_mean"),
        declared_digest_divergence_reasons=tuple(declared_digest_divergence_reasons),
        local_environment=local_environment,
        kaggle_environment=kaggle_environment,
    )


@dataclass(frozen=True)
class BackendQualificationRecord:
    """The single machine-readable qualification artifact.

    `status` is one of:
    - "qualified": byte-for-byte expected outcome -- protocol digests
      match, every gate passes, every item score and answer agree.
    - "qualified_with_acknowledged_differences": every item score agrees
      (the hard bar) and every non-precision gate passes, but the
      protocol digest differs *only* for reasons the caller declared and
      justified up front (e.g. the T4 bf16 incompatibility), and/or some
      item answers differ only after normalization (harmless formatting,
      explicitly documented rather than silently ignored).
    - "not_qualified": some gate failed, or the protocol/suite digest
      diverges without a declared, justified reason. Kaggle C/D results
      must not enter the four-parent tournament while this is the status.
    """

    qualification_id: str
    status: str
    parent_used: str
    report_digest: str
    item_score_agreement_count: int
    item_answer_agreement_count: int
    item_total_count: int
    dimension_agreement: Mapping[str, str]
    suite_digest_equal: bool
    tokenizer_identity_equal: bool
    protocol_digest_equal: bool
    declared_digest_divergence_reasons: tuple[str, ...]
    local_environment: Mapping[str, Any]
    kaggle_environment: Mapping[str, Any]
    qualification_timestamp: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "qualification_id": self.qualification_id,
            "status": self.status,
            "parent_used": self.parent_used,
            "report_digest": self.report_digest,
            "item_score_agreement_count": self.item_score_agreement_count,
            "item_answer_agreement_count": self.item_answer_agreement_count,
            "item_total_count": self.item_total_count,
            "dimension_agreement": dict(self.dimension_agreement),
            "suite_digest_equal": self.suite_digest_equal,
            "tokenizer_identity_equal": self.tokenizer_identity_equal,
            "protocol_digest_equal": self.protocol_digest_equal,
            "declared_digest_divergence_reasons": list(self.declared_digest_divergence_reasons),
            "local_environment": dict(self.local_environment),
            "kaggle_environment": dict(self.kaggle_environment),
            "qualification_timestamp": self.qualification_timestamp,
            "detail": self.detail,
        }

    def canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def digest(self) -> str:
        return _digest(self.to_dict())

    @property
    def is_qualified(self) -> bool:
        return self.status in ("qualified", "qualified_with_acknowledged_differences")


def qualify_backend(
    report: EquivalenceReport,
    *,
    qualification_id: str,
    now: str | None = None,
) -> BackendQualificationRecord:
    """Turn one equivalence report into a pass/fail qualification record.

    Never relaxes the item-score bar: `not_qualified` unless every item's
    score agrees. A protocol-digest mismatch is only tolerated when the
    report carries `declared_digest_divergence_reasons` -- this function
    does not decide *for itself* that a divergence is acceptable, it only
    checks that one was declared and that everything else still matches.
    """
    dimension_agreement = {dim.dimension: dim.classification for dim in report.dimension_comparisons}
    reasons: list[str] = []

    if report.item_total_count == 0:
        return BackendQualificationRecord(
            qualification_id=qualification_id,
            status="not_qualified",
            parent_used=report.parent_label,
            report_digest=report.digest(),
            item_score_agreement_count=0,
            item_answer_agreement_count=0,
            item_total_count=0,
            dimension_agreement=dimension_agreement,
            suite_digest_equal=report.suite_digest_equal,
            tokenizer_identity_equal=report.tokenizer_identity_equal,
            protocol_digest_equal=report.protocol_digest_equal,
            declared_digest_divergence_reasons=report.declared_digest_divergence_reasons,
            local_environment=report.local_environment.to_dict(),
            kaggle_environment=report.kaggle_environment.to_dict(),
            qualification_timestamp=now or _utcnow(),
            detail="no items were compared; an empty comparison is not qualification evidence",
        )

    if not report.all_scores_equal:
        mismatched = sum(1 for item in report.item_comparisons if not item.scores_equal)
        reasons.append(f"{mismatched}/{report.item_total_count} item score(s) disagree")
    if not report.suite_digest_equal:
        reasons.append("protected-suite content digest disagrees between local and Kaggle")
    if not report.tokenizer_identity_equal:
        reasons.append("tokenizer identity is not provably equal between local and Kaggle")
    if not report.protocol_digest_equal and not report.declared_digest_divergence_reasons:
        reasons.append(
            "protocol digest disagrees and no declared_digest_divergence_reasons were supplied "
            "to justify it -- an undeclared protocol difference fails closed"
        )
    non_tie_dimensions = [dim for dim, cls in dimension_agreement.items() if cls not in ("tie",)]
    if not report.all_scores_equal and non_tie_dimensions:
        reasons.append(f"dimensions with a non-tie delta: {non_tie_dimensions}")

    if reasons:
        return BackendQualificationRecord(
            qualification_id=qualification_id,
            status="not_qualified",
            parent_used=report.parent_label,
            report_digest=report.digest(),
            item_score_agreement_count=sum(1 for item in report.item_comparisons if item.scores_equal),
            item_answer_agreement_count=sum(1 for item in report.item_comparisons if item.answers_equal),
            item_total_count=report.item_total_count,
            dimension_agreement=dimension_agreement,
            suite_digest_equal=report.suite_digest_equal,
            tokenizer_identity_equal=report.tokenizer_identity_equal,
            protocol_digest_equal=report.protocol_digest_equal,
            declared_digest_divergence_reasons=report.declared_digest_divergence_reasons,
            local_environment=report.local_environment.to_dict(),
            kaggle_environment=report.kaggle_environment.to_dict(),
            qualification_timestamp=now or _utcnow(),
            detail="; ".join(reasons),
        )

    if report.protocol_digest_equal and report.all_answers_equal and not report.any_systematic_truncation_difference:
        status = "qualified"
        detail = "byte-for-byte expected outcome: protocol digests match, every item score and answer agree"
    else:
        status = "qualified_with_acknowledged_differences"
        detail_parts = ["every item score agrees"]
        if not report.protocol_digest_equal:
            detail_parts.append(
                f"protocol digest differs for declared reason(s): {list(report.declared_digest_divergence_reasons)}"
            )
        if not report.all_answers_equal:
            # Every item already has an equal score (the hard bar, checked
            # above). A raw-answer difference alongside an equal score of
            # 1.0 is structurally impossible for normalized_exact_match
            # (both sides would have to normalize to the same expected
            # value, hence to each other) -- so any remaining raw-answer
            # difference here is two *equally wrong* (score 0.0) answers
            # that happen to differ from one another, which is harmless
            # formatting/content noise, not an equivalence concern. Still
            # documented explicitly rather than silently declaring
            # byte-level identity.
            differing = sum(1 for item in report.item_comparisons if not item.answers_equal)
            detail_parts.append(
                f"{differing} item(s) have textually different (but equally-scored) raw answers"
            )
        if report.any_systematic_truncation_difference:
            detail_parts.append("systematic truncation difference observed between backends")
        detail = "; ".join(detail_parts)

    return BackendQualificationRecord(
        qualification_id=qualification_id,
        status=status,
        parent_used=report.parent_label,
        report_digest=report.digest(),
        item_score_agreement_count=sum(1 for item in report.item_comparisons if item.scores_equal),
        item_answer_agreement_count=sum(1 for item in report.item_comparisons if item.answers_equal),
        item_total_count=report.item_total_count,
        dimension_agreement=dimension_agreement,
        suite_digest_equal=report.suite_digest_equal,
        tokenizer_identity_equal=report.tokenizer_identity_equal,
        protocol_digest_equal=report.protocol_digest_equal,
        declared_digest_divergence_reasons=report.declared_digest_divergence_reasons,
        local_environment=report.local_environment.to_dict(),
        kaggle_environment=report.kaggle_environment.to_dict(),
        qualification_timestamp=now or _utcnow(),
        detail=detail,
    )
