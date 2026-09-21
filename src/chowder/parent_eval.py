"""Protected parent-evaluation suite for the Qwen3.8 parent tournament.

Roadmap Priority 0's Phase 4 requires all four parent candidates
(docs/QWEN38_SPARSE_PROGRAM.md: A control, B primary, C and D
comparisons) to be evaluated under *one identical protocol* before any
architecture surgery, and Phase 13 bans training on protected evaluator
content outright. This module is the tournament's harness: the suite
schema, the protocol fingerprint that makes "same protocol" checkable
rather than asserted, dimension aggregation with capability and behavior
kept separate, and the persistence plumbing that turns a parent run into
registry evidence under the same `evaluation_runs` rows every other
evaluation uses.

What this module deliberately is NOT
------------------------------------
It is not a runner. It performs no I/O against a model: constructing a
spec, fingerprinting it, aggregating a worker's result payload, and
persisting an `EvaluationOutcome` are all offline operations. The
existing subprocess workers (`base_text_worker.py` today; lm-eval for
external benchmarks later) do the actual generation; their payload
(`metrics`/`evidence`, keyed by suite name) plugs into
`aggregate_parent_result` unchanged. This follows the same
specification/execution split as `BaseTextEvalSpec`/`base_text_worker`.

Nine dimensions, one rule about separation
------------------------------------------
The mission's nine evaluation dimensions, in fixed order:

  reasoning, coding, knowledge, calibration, self_correction,
  instruction_following, agentic, thinking_efficiency, behavior

Eight are capability dimensions. `behavior` (unnecessary refusal on a
benign protected suite) is not: the mission explicitly separates
capability from behavior ("do NOT reduce refusal rate to intelligence").
Mechanically, that means:

- behavior results are never blended into the capability aggregate —
  `capability_mean` is defined over the eight capability dimensions only;
- behavior is reported alongside (its own 0..1 mean, its own per-suite
  metrics) with its dimension label, so a parent-selection decision can
  *see* both without either contaminating the other's number;
- the parent-evaluation protocol fingerprint covers both (the identical
  protocol requirement applies to the whole suite), while suite-level
  dataset content stays out of spec digests (it is fingerprinted
  separately, hash-only, at run time by the worker).

Honesty rules this module enforces mechanically
-----------------------------------------------
1. **Coverage is complete or the spec is invalid.** A spec whose suites
   do not cover exactly the nine dimensions is rejected — silently
   evaluating three of nine dimensions would produce a tournament row
   that looks like evidence.
2. **Fingerprints cover meaning, not candidates.** Like
   `BaseTextEvalSpec.digest()`, the parent-spec fingerprint excludes
   base-model identity, revisions, output paths, and seeds — those are
   per-candidate state. It includes the full suite definitions (names,
   dimension labels, dataset refs, scoring, generation limits,
   chat-template policy). Two parents evaluated under different suite
   definitions produce different fingerprints, and that difference is
   visible evidence, not noise.
3. **Protection is proven, not promised.** `build_protected_suite_dir`
   writes hash-only fingerprint indexes (no raw prompts, reusing
   `contamination.write_holdout_fingerprint_index`), and every suite row
   the spec carries is auditable against those indexes. Training/repair
   datasets destined for a tournament parent are checked against the
   same indexes with `audit_repair_examples` — the existing
   Contamination Guard machinery, pointed at the tournament holdout.
4. **No fabricated scores.** Aggregation reads only what the worker
   reported; a dimension with no suite metrics is `None` in the report,
   never zero, never imputed.
5. **Compatibility is evidence-based.** Parent D's tokenizer class
   differs from A/C's (docs/QWEN38_SPARSE_PROGRAM.md); class names are
   not identity. The tokenizer gate here records class *and* vocab-size
   evidence and refuses a tournament when identity is not provably
   equal — mirroring `teacher_fabric`'s fail-closed stance.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping

from .contamination import (
    RepairExample,
    audit_repair_examples,
    write_holdout_fingerprint_index,
)
from .executors import EvaluationOutcome

#: The mission's nine dimensions, in fixed report order. Eight capability
#: dimensions plus `behavior`, which is scored separately by construction.
PARENT_DIMENSIONS: tuple[str, ...] = (
    "reasoning",
    "coding",
    "knowledge",
    "calibration",
    "self_correction",
    "instruction_following",
    "agentic",
    "thinking_efficiency",
    "behavior",
)

CAPABILITY_DIMENSIONS: frozenset[str] = frozenset(PARENT_DIMENSIONS) - {"behavior"}
BEHAVIOR_DIMENSION = "behavior"


class ParentDimension(str, Enum):
    """The nine tournament dimensions, validated by membership not spelling."""

    REASONING = "reasoning"
    CODING = "coding"
    KNOWLEDGE = "knowledge"
    CALIBRATION = "calibration"
    SELF_CORRECTION = "self_correction"
    INSTRUCTION_FOLLOWING = "instruction_following"
    AGENTIC = "agentic"
    THINKING_EFFICIENCY = "thinking_efficiency"
    BEHAVIOR = "behavior"

    @property
    def is_capability(self) -> bool:
        return self is not ParentDimension.BEHAVIOR


class ParentSuiteValidationError(ValueError):
    """A parent suite definition does not cover the nine dimensions, or
    otherwise violates the tournament's structural rules."""


class ParentTokenizerMismatch(ValueError):
    """Parent tokenizer identity is not provably equal across the
    tournament. Raised, never approximated: cross-parent score
    comparability rests on shared tokenization (mirrors
    `teacher_fabric.TokenizerIncompatibilityError`'s fail-closed rule)."""


@dataclass(frozen=True)
class ParentSuiteSpec:
    """One protected suite: a dataset probed for one dimension.

    Mirrors `EvalSuiteSpec`'s shape so the existing workers can execute
    it directly, and adds the dimension label the tournament aggregates
    over. `dataset` is a path or Hub dataset reference resolved by the
    worker at run time; suite-level dataset *content* never enters the
    spec fingerprint (content identity is proven separately, hash-only).
    """

    name: str
    dimension: str
    dataset: str
    prompt_field: str = "prompt"
    expected_field: str = "expected"
    scoring: str = "normalized_exact_match"
    # v2 (2026-09-07): thinking models spend generation budget on visible
    # chain-of-thought before answering; 64 tokens truncated the answer on
    # every retry6 item. 256 covers observed thinking (~40-150 tokens) plus
    # the answer with headroom. The budget is protocol identity (hashed).
    max_new_tokens: int = 256
    use_chat_template: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("parent suite name is required")
        try:
            ParentDimension(self.dimension)
        except ValueError as exc:
            raise ValueError(
                f"parent suite {self.name!r} has unknown dimension {self.dimension!r}; "
                f"expected one of {', '.join(PARENT_DIMENSIONS)}"
            ) from exc
        if not isinstance(self.dataset, str) or not self.dataset.strip():
            raise ValueError(f"parent suite {self.name!r} dataset is required")
        if self.scoring not in {"exact_match", "normalized_exact_match"}:
            raise ValueError(f"parent suite {self.name!r} scoring must be exact_match or normalized_exact_match")
        if isinstance(self.max_new_tokens, bool) or not isinstance(self.max_new_tokens, int) or self.max_new_tokens <= 0:
            raise ValueError(f"parent suite {self.name!r} max_new_tokens must be a positive int")
        for label in ("prompt_field", "expected_field"):
            value = getattr(self, label)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"parent suite {self.name!r} {label} is required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dimension": self.dimension,
            "dataset": self.dataset,
            "prompt_field": self.prompt_field,
            "expected_field": self.expected_field,
            "scoring": self.scoring,
            "max_new_tokens": self.max_new_tokens,
            "use_chat_template": self.use_chat_template,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ParentSuiteSpec":
        return cls(
            name=data["name"],
            dimension=data["dimension"],
            dataset=data["dataset"],
            prompt_field=data.get("prompt_field", "prompt"),
            expected_field=data.get("expected_field", "expected"),
            scoring=data.get("scoring", "normalized_exact_match"),
            max_new_tokens=data.get("max_new_tokens", 256),
            use_chat_template=bool(data.get("use_chat_template", False)),
        )


@dataclass(frozen=True)
class ParentEvalSpec:
    """The full tournament protocol: suites covering all nine dimensions.

    The fingerprint deliberately mirrors `BaseTextEvalSpec.digest()`'s
    exclusions: candidate identity (base_model, revisions), output
    locations, seeds, and device are per-run state and excluded; suite
    definitions (which decide what a score *means*) are included. All
    four parents evaluated under this spec therefore share one
    fingerprint, and any protocol change is a visible fingerprint change.
    """

    suites: tuple[ParentSuiteSpec, ...]
    precision: str = "bf16"
    quantization: str = "none"
    max_model_len: int | None = None
    require_thinking_efficiency_telemetry: bool = True
    # Explicit protocol semantics version. The digest already hashes the
    # budget; this marker additionally captures scorer-side semantics that
    # are not per-suite fields -- v2 = thinking-aware answer extraction
    # (score against the text after the last "</think>") in the worker.
    # v3 (2026-09-09) = canonical chat-template rendering: every parent's
    # prompts render through ONE pinned template (the official
    # Qwen/Qwen3.8-27B one, digest-pinned in canonical_chat_template.py)
    # instead of each parent's own re-serialized template. Motivated by the
    # real C/D gate outcome: C/D tokenize identically to A but carry
    # different templates, so v2's own-template rendering broke
    # comparability in a way no tokenizer-identity gate could see.
    protocol_version: str = "v2"
    # v3 digest-additive fields: absent from to_dict() unless enabled, so
    # every v2 spec's canonical JSON (and therefore its recorded digest,
    # including retry7's c5e964df...) reproduces byte-for-byte.
    canonical_rendering: bool = False
    canonical_template_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.suites:
            raise ValueError("parent evaluation requires at least one suite")
        if len({suite.name for suite in self.suites}) != len(self.suites):
            raise ValueError("parent evaluation suite names must be unique")
        if self.precision not in {"auto", "bf16", "fp16", "fp32"}:
            raise ValueError(f"unsupported parent evaluation precision: {self.precision}")
        if self.quantization not in {"none", "4bit"}:
            raise ValueError(f"unsupported parent evaluation quantization: {self.quantization}")
        if self.max_model_len is not None and (
            isinstance(self.max_model_len, bool) or self.max_model_len <= 0
        ):
            raise ValueError("parent evaluation max_model_len must be a positive int or None")
        covered = {suite.dimension for suite in self.suites}
        missing = [dim for dim in PARENT_DIMENSIONS if dim not in covered]
        if missing:
            raise ParentSuiteValidationError(
                "parent evaluation does not cover every tournament dimension; "
                f"missing: {', '.join(missing)}. A partial suite produces a "
                "tournament row that looks like evidence but is not."
            )
        if self.canonical_rendering:
            # Fail closed at construction: a v3 spec must pin exactly the
            # canonical template the module embeds, and must say v3.
            from chowder.canonical_chat_template import canonical_template_sha256

            pinned = canonical_template_sha256()
            if self.canonical_template_sha256 != pinned:
                raise ParentSuiteValidationError(
                    "canonical_rendering requires canonical_template_sha256 "
                    f"== the embedded canonical template digest ({pinned}); "
                    f"got {self.canonical_template_sha256!r}. A template "
                    "change is a protocol change."
                )
            if self.protocol_version != "v3":
                raise ParentSuiteValidationError(
                    "canonical_rendering is a v3 protocol feature; set "
                    "protocol_version='v3'"
                )
        elif self.canonical_template_sha256 is not None:
            raise ParentSuiteValidationError(
                "canonical_template_sha256 without canonical_rendering is "
                "meaningless; enable canonical_rendering or drop the pin"
            )

    def suites_for_dimension(self, dimension: str) -> tuple[ParentSuiteSpec, ...]:
        ParentDimension(dimension)  # validates
        return tuple(suite for suite in self.suites if suite.dimension == dimension)

    def to_dict(self) -> dict[str, Any]:
        data = {
            "suites": [suite.to_dict() for suite in self.suites],
            "precision": self.precision,
            "quantization": self.quantization,
            "max_model_len": self.max_model_len,
            "require_thinking_efficiency_telemetry": self.require_thinking_efficiency_telemetry,
            "protocol_version": self.protocol_version,
        }
        if self.canonical_rendering:
            # v3-only keys: their absence from v2 JSON keeps every v2
            # digest byte-identical to its recorded value.
            data["canonical_rendering"] = True
            data["canonical_template_sha256"] = self.canonical_template_sha256
        return data

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def digest(self) -> str:
        """Protocol fingerprint over suite definitions + decoding policy.

        Excludes candidate-specific state (base model, revision, output
        paths, seed, device) exactly as `BaseTextEvalSpec.digest()` does;
        stores as `evaluation_protocol_sha256` so the promotion gate's
        strict matching applies to tournament rows too.
        """
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ParentEvalSpec":
        return cls(
            suites=tuple(ParentSuiteSpec.from_dict(suite) for suite in data["suites"]),
            precision=data.get("precision", "bf16"),
            quantization=data.get("quantization", "none"),
            max_model_len=data.get("max_model_len"),
            require_thinking_efficiency_telemetry=bool(
                data.get("require_thinking_efficiency_telemetry", True)
            ),
            # Fix (2026-09-09): protocol_version was silently dropped on
            # round-trip (re-defaulted to v2) -- a recorded v3 spec reloaded
            # from JSON would have claimed v2. Also restores the v3 fields.
            protocol_version=data.get("protocol_version", "v2"),
            canonical_rendering=bool(data.get("canonical_rendering", False)),
            canonical_template_sha256=data.get("canonical_template_sha256"),
        )


@dataclass(frozen=True)
class ParentTokenizerEvidence:
    """Recorded tokenizer identity for one parent — evidence, not a name."""

    tokenizer_class: str
    vocab_size: int
    identity_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.tokenizer_class, str) or not self.tokenizer_class.strip():
            raise ValueError("tokenizer_class must be a non-empty string")
        if isinstance(self.vocab_size, bool) or not isinstance(self.vocab_size, int) or self.vocab_size <= 0:
            raise ValueError("vocab_size must be a positive int")
        if not isinstance(self.identity_sha256, str) or len(self.identity_sha256) != 64:
            raise ValueError("identity_sha256 must be a 64-character sha256 hex digest")


def ensure_parent_tokenizer_compatible(
    reference: ParentTokenizerEvidence,
    candidate: ParentTokenizerEvidence,
) -> None:
    """Fail closed unless parent tokenizers are provably identical.

    Class name, vocab size, and identity hash must all match. Class-name
    equality alone is not identity (docs/QWEN38_SPARSE_PROGRAM.md: D
    reports `TokenizersBackend` where A/C report `Qwen2Tokenizer`); the
    identity hash (hash of the tokenizer's serialized assets) is the
    deciding evidence. No fuzzy matching, no "close enough".
    """
    if (
        reference.tokenizer_class == candidate.tokenizer_class
        and reference.vocab_size == candidate.vocab_size
        and reference.identity_sha256 == candidate.identity_sha256
    ):
        return
    raise ParentTokenizerMismatch(
        "parent tokenizer identities differ: reference "
        f"({reference.tokenizer_class}, vocab={reference.vocab_size}, "
        f"identity={reference.identity_sha256[:12]}...) vs candidate "
        f"({candidate.tokenizer_class}, vocab={candidate.vocab_size}, "
        f"identity={candidate.identity_sha256[:12]}...); cross-parent score "
        "comparability requires provably shared tokenization, so the "
        "tournament refuses this pairing (fail closed). Re-serialize the "
        "tokenizer from identical assets or exclude the parent."
    )


@dataclass(frozen=True)
class DimensionScore:
    """One dimension's aggregate plus the suites behind it."""

    dimension: str
    mean: float | None
    suite_metrics: Mapping[str, float]


@dataclass(frozen=True)
class ParentEvalReport:
    """Aggregated tournament row for one parent.

    `capability_mean` is over the eight capability dimensions only —
    behavior never blends in. `behavior_mean` is reported alongside.
    Either may be `None` when its evidence is absent; nothing is imputed.
    """

    base_model: str
    revision: str | None
    evaluation_protocol_sha256: str
    dimensions: Mapping[str, DimensionScore]
    capability_mean: float | None
    behavior_mean: float | None
    evidence: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_model": self.base_model,
            "revision": self.revision,
            "evaluation_protocol_sha256": self.evaluation_protocol_sha256,
            "dimensions": {
                dimension: {
                    "mean": score.mean,
                    "suite_metrics": dict(score.suite_metrics),
                }
                for dimension, score in self.dimensions.items()
            },
            "capability_mean": self.capability_mean,
            "behavior_mean": self.behavior_mean,
            "evidence": dict(self.evidence),
        }


def aggregate_parent_result(
    *,
    spec: ParentEvalSpec,
    base_model: str,
    revision: str | None,
    metrics: Mapping[str, float],
    evidence: Mapping[str, Any] | None = None,
) -> ParentEvalReport:
    """Aggregate a worker's per-suite payload into a tournament row.

    `metrics` is keyed by suite name (the workers' payload shape, e.g.
    `base_text_worker`'s `metrics[suite.name] = correct / len(rows)`).
    Suites not present in `metrics` contribute `None` means for their
    dimension — absent evidence is reported, never imputed. Capability
    and behavior are aggregated separately by construction.
    """
    evidence = dict(evidence or {})
    dimensions: dict[str, DimensionScore] = {}
    capability_values: list[float] = []
    behavior_values: list[float] = []
    for dimension in PARENT_DIMENSIONS:
        suite_metrics: dict[str, float] = {}
        for suite in spec.suites_for_dimension(dimension):
            value = metrics.get(suite.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            number = float(value)
            if number != number or number in (float("inf"), float("-inf")):
                continue
            suite_metrics[suite.name] = number
        mean = fmean_or_none(suite_metrics.values())
        dimensions[dimension] = DimensionScore(
            dimension=dimension, mean=mean, suite_metrics=suite_metrics
        )
        if mean is None:
            continue
        if dimension == BEHAVIOR_DIMENSION:
            behavior_values.append(mean)
        else:
            capability_values.append(mean)

    capability_mean = fmean_or_none(capability_values)
    behavior_mean = fmean_or_none(behavior_values)
    return ParentEvalReport(
        base_model=base_model,
        revision=revision,
        evaluation_protocol_sha256=spec.digest(),
        dimensions=dimensions,
        capability_mean=capability_mean,
        behavior_mean=behavior_mean,
        evidence={
            **evidence,
            "parent_program": "qwen38-native-sparse",
            "capability_dimensions": sorted(CAPABILITY_DIMENSIONS),
            "behavior_dimension": BEHAVIOR_DIMENSION,
        },
    )


def fmean_or_none(values: Iterable[float]) -> float | None:
    collected = tuple(values)
    if not collected:
        return None
    return sum(collected) / len(collected)


def build_protected_suite_dir(
    suites: Iterable[tuple[str, Iterable[tuple[str, str]]]],
    output_dir: str,
) -> dict[str, str]:
    """Write hash-only fingerprint indexes for the protected suites.

    `suites` yields (suite_name, (prompt, expected) pairs). Each suite
    gets `<output_dir>/<suite_name>.fingerprints.jsonl` via
    `contamination.write_holdout_fingerprint_index` (hash-only: raw
    prompt/answer text is never stored). Returns suite name -> index
    sha256; those digests are the *protection evidence* recorded with a
    tournament run and the indexes every later training/repair dataset
    for that parent is audited against.
    """
    digests: dict[str, str] = {}
    for suite_name, examples in suites:
        if not suite_name.strip():
            raise ValueError("protected suite name must be non-empty")
        digests[suite_name] = write_holdout_fingerprint_index(
            examples, f"{output_dir.rstrip('/')}/{suite_name}.fingerprints.jsonl"
        )
    return digests


def audit_training_examples_against_tournament(
    examples: Iterable[tuple[str, str]],
    protected_index_paths: Iterable[str],
    *,
    source_id_prefix: str = "tournament-protected",
) -> Any:
    """Run the Contamination Guard audit of training data vs the holdout.

    Thin composition over `contamination.audit_repair_examples` with a
    tournament-specific source prefix; `clean=False` is a hard stop for
    any training run that would touch protected evaluation content
    (Phase 13's ban, enforced mechanically).
    """
    rows = [
        RepairExample(prompt=prompt, expected=expected, source_id=f"{source_id_prefix}-{index}")
        for index, (prompt, expected) in enumerate(examples)
    ]
    return audit_repair_examples(rows, protected_index_paths)


def record_parent_tournament_result(
    registry: Any,
    *,
    report: ParentEvalReport,
    run_id: str,
    experiment_id: str,
    artifact_ref: str,
    gpu_hours: float,
) -> EvaluationOutcome:
    """Persist a tournament row as a normal `evaluation_runs` row.

    `experiment_id` must reference an experiment already persisted in
    this registry — `evaluation_runs` carries a real foreign key on it,
    and an unanchored evaluation row would silently fall outside every
    experiment-linked query the meta-controller runs. Fail closed here
    rather than inventing a placeholder experiment: the caller (the
    tournament runner) decides the anchoring semantics.

    Metrics are the per-suite scores plus the two aggregate keys; the
    evidence dict carries the full report, the protocol fingerprint
    (so strict protocol matching applies), and the provenance the brief
    requires (parent program, dimension map, capability/behavior split).
    Returns the recorded outcome for chaining.
    """
    flat_metrics: dict[str, float] = {}
    for score in report.dimensions.values():
        flat_metrics.update(score.suite_metrics)
    if report.capability_mean is not None:
        flat_metrics["capability_mean"] = report.capability_mean
    if report.behavior_mean is not None:
        flat_metrics["behavior_mean"] = report.behavior_mean
    if hasattr(registry, "has_experiment") and not registry.has_experiment(experiment_id):
        raise ValueError(
            f"experiment_id {experiment_id!r} is not persisted in this registry; "
            "evaluation runs are foreign-keyed to experiments, so record the "
            "parent-evaluation experiment before its outcome"
        )
    outcome = EvaluationOutcome(
        run_id=run_id,
        experiment_id=experiment_id,
        source_artifact_ref=artifact_ref,
        metrics=flat_metrics,
        gpu_hours=gpu_hours,
        evidence={
            # the run's own provenance rides at the top level; the full
            # report stays nested. Structural keys win over anything the
            # caller supplied under the same name.
            **report.evidence,
            "evaluation_protocol_sha256": report.evaluation_protocol_sha256,
            "parent_eval_report": report.to_dict(),
        },
    )
    registry.record_evaluation_outcome(outcome)
    return outcome
