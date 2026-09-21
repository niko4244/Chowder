"""Tests for the protected parent-evaluation harness (Qwen3.8 Phase 4).

Offline and deterministic: no model I/O, no network. The harness is a
specification/aggregation/persistence layer — the subprocess workers that
execute generation are exercised by their own suites. Each test targets
one honesty rule claimed in `chowder.parent_eval`'s module docstring:
complete-or-invalid coverage, no fabricated scores, capability/behavior
separation, fingerprint meaning, proven protection, and registry
persistence under the ordinary evaluation_runs schema.
"""

import hashlib
import json

import pytest

from chowder.executors import EvaluationOutcome
from chowder.parent_eval import (
    BEHAVIOR_DIMENSION,
    CAPABILITY_DIMENSIONS,
    PARENT_DIMENSIONS,
    ParentDimension,
    DimensionScore,
    ParentEvalReport,
    ParentEvalSpec,
    ParentSuiteValidationError,
    ParentSuiteSpec,
    ParentTokenizerEvidence,
    ParentTokenizerMismatch,
    aggregate_parent_result,
    audit_training_examples_against_tournament,
    build_protected_suite_dir,
    ensure_parent_tokenizer_compatible,
    record_parent_tournament_result,
)
from chowder.models import Experiment, Hypothesis
from chowder.registry import RunRegistry


def _suite(dimension: str, **overrides) -> ParentSuiteSpec:
    defaults = {
        "name": f"suite-{dimension}",
        "dimension": dimension,
        "dataset": f"protected/{dimension}-v1",
    }
    defaults.update(overrides)
    return ParentSuiteSpec(**defaults)


def _full_suites() -> tuple[ParentSuiteSpec, ...]:
    return tuple(_suite(dimension) for dimension in PARENT_DIMENSIONS)


@pytest.fixture()
def spec() -> ParentEvalSpec:
    return ParentEvalSpec(suites=_full_suites())


@pytest.fixture()
def full_metrics(spec: ParentEvalSpec) -> dict[str, float]:
    return {suite.name: 0.5 for suite in spec.suites}


@pytest.fixture()
def registry(tmp_path):
    with RunRegistry(tmp_path / "runs.db") as reg:
        reg.record_experiment(
            Experiment(
                experiment_id="parent-tournament:A",
                parent_id=None,
                hypothesis=Hypothesis(
                    "base parents must be measured before surgery",
                    "no baseline exists",
                    "evaluate all four parents under one protocol",
                ),
                config_patch={"parent_program": "qwen38-native-sparse"},
                estimated_gpu_hours=2.0,
            )
        )
        yield reg


# ---------------------------------------------------------------------------
# Coverage: complete or the spec is invalid
# ---------------------------------------------------------------------------


def test_nine_dimensions_covered_and_reported_in_order(spec, full_metrics):
    report = aggregate_parent_result(
        spec=spec,
        base_model="Qwen/Qwen3.8-27B",
        revision="1d4bf0f2",
        metrics=full_metrics,
    )
    assert tuple(report.dimensions) == PARENT_DIMENSIONS
    assert all(score.mean == 0.5 for score in report.dimensions.values())
    assert report.capability_mean == 0.5
    assert report.behavior_mean == 0.5
    assert CAPABILITY_DIMENSIONS == frozenset(PARENT_DIMENSIONS) - {BEHAVIOR_DIMENSION}


def test_missing_dimension_rejected_not_silently_partial():
    suites = tuple(_suite(dimension) for dimension in PARENT_DIMENSIONS if dimension != "behavior")
    with pytest.raises(ParentSuiteValidationError) as excinfo:
        ParentEvalSpec(suites=suites)
    assert "behavior" in str(excinfo.value)


def test_duplicate_suite_names_rejected(spec):
    doubled = spec.suites + (spec.suites[0],)
    with pytest.raises(ValueError, match="unique"):
        ParentEvalSpec(suites=doubled)


def test_suite_spec_validation():
    with pytest.raises(ValueError, match="unknown dimension"):
        _suite("charm")
    with pytest.raises(ValueError, match="scoring"):
        _suite("coding", scoring="vibes")
    with pytest.raises(ValueError, match="name"):
        _suite("coding", name="  ")
    with pytest.raises(ValueError, match="max_new_tokens"):
        _suite("coding", max_new_tokens=0)
    with pytest.raises(ValueError, match="max_new_tokens"):
        _suite("coding", max_new_tokens=True)  # bool is not an int here


# ---------------------------------------------------------------------------
# Capability / behavior separation
# ---------------------------------------------------------------------------


def test_capability_mean_never_blends_behavior(spec, full_metrics):
    metrics = {**full_metrics, "suite-behavior": 1.0}
    report = aggregate_parent_result(
        spec=spec, base_model="A", revision=None, metrics=metrics
    )
    assert report.capability_mean == 0.5
    assert report.behavior_mean == 1.0
    assert report.capability_mean != report.behavior_mean


def test_capability_absent_when_all_capability_dimensions_absent(spec):
    report = aggregate_parent_result(spec=spec, base_model="A", revision=None, metrics={})
    assert report.capability_mean is None
    assert report.behavior_mean is None


def test_behavior_present_with_capability_absent_is_reported_separately(spec):
    report = aggregate_parent_result(
        spec=spec, base_model="A", revision=None, metrics={"suite-behavior": 0.9}
    )
    assert report.capability_mean is None
    assert report.behavior_mean == 0.9
    assert report.dimensions["behavior"].suite_metrics == {"suite-behavior": 0.9}


def test_absent_suite_is_reported_none_not_imputed(spec, full_metrics):
    metrics = dict(full_metrics)
    del metrics["suite-reasoning"]
    report = aggregate_parent_result(
        spec=spec, base_model="A", revision=None, metrics=metrics
    )
    assert report.dimensions["reasoning"].mean is None
    assert report.dimensions["reasoning"].suite_metrics == {}
    # the capability mean averages only dimensions with evidence
    assert report.capability_mean == pytest.approx(0.5)
    assert report.dimensions["reasoning"].dimension == "reasoning"


def test_nonfinite_worker_values_are_dropped_not_propagated(spec, full_metrics):
    metrics = {**full_metrics, "suite-coding": float("nan"), "suite-agentic": float("inf")}
    report = aggregate_parent_result(
        spec=spec, base_model="A", revision=None, metrics=metrics
    )
    assert report.dimensions["coding"].mean is None
    assert report.dimensions["agentic"].mean is None
    assert "suite-coding" not in report.dimensions["coding"].suite_metrics


def test_report_evidence_carries_program_markers(spec, full_metrics):
    report = aggregate_parent_result(
        spec=spec, base_model="A", revision=None, metrics=full_metrics, evidence={"gpu": "b"}
    )
    assert report.evidence["parent_program"] == "qwen38-native-sparse"
    assert report.evidence["behavior_dimension"] == "behavior"
    assert report.evidence["capability_dimensions"] == sorted(CAPABILITY_DIMENSIONS)
    assert report.evidence["gpu"] == "b"


def test_dimension_enum_flags_behavior_as_non_capability():
    assert ParentDimension("behavior").is_capability is False
    assert ParentDimension("reasoning").is_capability is True


# ---------------------------------------------------------------------------
# Protocol fingerprint: covers meaning, not candidates
# ---------------------------------------------------------------------------


def test_fingerprint_excludes_candidate_identity(spec, full_metrics):
    first = aggregate_parent_result(
        spec=spec, base_model="Qwen/Qwen3.8-27B", revision="aaaa", metrics=full_metrics
    )
    second = aggregate_parent_result(
        spec=spec,
        base_model="orcarouter/Qwen3.8-27B-Uncensored",
        revision="404ea47a",
        metrics=full_metrics,
    )
    assert first.evaluation_protocol_sha256 == second.evaluation_protocol_sha256


def test_fingerprint_changes_when_protocol_meaning_changes():
    suites_a = _full_suites()
    suites_b = tuple(
        _suite(s.dimension, max_new_tokens=128) if s.dimension == "coding" else s
        for s in suites_a
    )
    spec_a = ParentEvalSpec(suites=suites_a)
    spec_b = ParentEvalSpec(suites=suites_b)
    assert spec_a.digest() != spec_b.digest()
    # and a pure scoring change is also a meaning change
    suites_c = tuple(
        _suite(s.dimension, scoring="exact_match") if s.dimension == "coding" else s
        for s in suites_a
    )
    assert spec_a.digest() != ParentEvalSpec(suites=suites_c).digest()


def test_fingerprint_survives_dict_round_trip(spec):
    assert ParentEvalSpec.from_dict(spec.to_dict()).digest() == spec.digest()


def test_suites_for_dimension_filters_and_validates(spec):
    assert {s.name for s in spec.suites_for_dimension("coding")} == {"suite-coding"}
    with pytest.raises(ValueError):
        spec.suites_for_dimension("charm")


# ---------------------------------------------------------------------------
# Tokenizer gate: evidence-based, fail closed
# ---------------------------------------------------------------------------


def _tok_evidence(cls="Qwen2Tokenizer", vocab=151936, digest=None):
    return ParentTokenizerEvidence(
        tokenizer_class=cls,
        vocab_size=vocab,
        identity_sha256=digest or ("a" * 64),
    )


def test_tokenizer_gate_accepts_provable_identity():
    assert ensure_parent_tokenizer_compatible(_tok_evidence(), _tok_evidence()) is None


@pytest.mark.parametrize(
    "candidate",
    [
        _tok_evidence(cls="TokenizersBackend"),
        _tok_evidence(vocab=151646),
        _tok_evidence(digest="b" * 64),
    ],
)
def test_tokenizer_gate_fails_closed_on_any_identity_difference(candidate):
    with pytest.raises(ParentTokenizerMismatch) as excinfo:
        ensure_parent_tokenizer_compatible(_tok_evidence(), candidate)
    assert "fail closed" in str(excinfo.value)


def test_tokenizer_evidence_validates_itself():
    with pytest.raises(ValueError, match="identity_sha256"):
        _tok_evidence(digest="short")
    with pytest.raises(ValueError, match="vocab_size"):
        ParentTokenizerEvidence(tokenizer_class="Qwen2Tokenizer", vocab_size=0, identity_sha256="a" * 64)
    with pytest.raises(ValueError, match="vocab_size"):
        ParentTokenizerEvidence(tokenizer_class="Qwen2Tokenizer", vocab_size=True, identity_sha256="a" * 64)


# ---------------------------------------------------------------------------
# Protection: hash-only indexes, real audits
# ---------------------------------------------------------------------------


def test_protected_suite_dir_is_hash_only_and_deterministic(tmp_path):
    examples = [("What is 2+2?", "4"), ("Capital of France?", "Paris")]
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    digests_a = build_protected_suite_dir([("suite-knowledge", examples)], str(out_a))
    digests_b = build_protected_suite_dir([("suite-knowledge", examples)], str(out_b))

    index_path = out_a / "suite-knowledge.fingerprints.jsonl"
    assert index_path.exists()
    raw = index_path.read_text(encoding="utf-8")
    assert "What is 2+2?" not in raw and "Paris" not in raw
    for line in raw.splitlines():
        row = json.loads(line)
        assert set(row) == {"prompt_sha256", "pair_sha256"}

    assert digests_a["suite-knowledge"] == digests_b["suite-knowledge"]
    assert digests_a["suite-knowledge"] == hashlib.sha256(index_path.read_bytes()).hexdigest()


def test_protected_suite_dir_deduplicates_and_rejects_blank_names(tmp_path):
    examples = [("q1", "e1"), ("q1", "e1")]
    digests = build_protected_suite_dir([("s", examples)], str(tmp_path / "out"))
    lines = (tmp_path / "out" / "s.fingerprints.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    with pytest.raises(ValueError, match="non-empty"):
        build_protected_suite_dir([("  ", examples)], str(tmp_path / "out2"))


def test_audit_clean_and_contaminated(tmp_path):
    protected = [("unique-protected-prompt-1", "a1"), ("unique-protected-prompt-2", "a2")]
    digests = build_protected_suite_dir([("suite-reasoning", protected)], str(tmp_path))
    index_paths = [str(tmp_path / "suite-reasoning.fingerprints.jsonl")]

    clean = audit_training_examples_against_tournament(
        [("benign-training-prompt", "b1")], index_paths
    )
    assert clean.clean is True
    assert clean.overlap_count == 0

    contaminated = audit_training_examples_against_tournament(
        [("benign-training-prompt", "b1"), protected[0]], index_paths
    )
    assert contaminated.clean is False
    assert contaminated.overlap_count >= 1
    assert contaminated.holdout_index_sha256 == tuple(sorted(digests.values()))


# ---------------------------------------------------------------------------
# Persistence: ordinary evaluation_runs rows
# ---------------------------------------------------------------------------


def test_record_round_trip_through_registry(registry, spec, full_metrics):
    report = aggregate_parent_result(
        spec=spec,
        base_model="Qwen/Qwen3.8-27B",
        revision="1d4bf0f2",
        metrics=full_metrics,
        evidence={"accelerator": "rtx5060ti"},
    )
    outcome = record_parent_tournament_result(
        registry,
        report=report,
        run_id="run-parent-1",
        experiment_id="parent-tournament:A",
        artifact_ref="artifacts/parent-a.json",
        gpu_hours=1.25,
    )
    assert isinstance(outcome, EvaluationOutcome)
    stored = [o for o in registry.list_evaluation_outcomes() if o.run_id == "run-parent-1"]
    assert len(stored) == 1
    row = stored[0]
    assert row.experiment_id == "parent-tournament:A"
    assert row.metrics["suite-knowledge"] == 0.5
    assert row.metrics["capability_mean"] == 0.5
    assert row.metrics["behavior_mean"] == 0.5
    assert row.gpu_hours == 1.25
    assert row.evidence["evaluation_protocol_sha256"] == report.evaluation_protocol_sha256
    replayed = row.evidence["parent_eval_report"]
    assert replayed["capability_mean"] == 0.5
    assert replayed["behavior_mean"] == 0.5
    assert set(replayed["dimensions"]) == set(PARENT_DIMENSIONS)
    assert row.evidence["accelerator"] == "rtx5060ti"


def test_record_omits_absent_aggregates_instead_of_zero(registry, spec):
    metrics = {"suite-behavior": 0.9}
    report = aggregate_parent_result(spec=spec, base_model="A", revision=None, metrics=metrics)
    outcome = record_parent_tournament_result(
        registry,
            report=report,
            run_id="run-parent-2",
            experiment_id="parent-tournament:A",
            artifact_ref="x",
            gpu_hours=0.0,
        
    )
    assert "capability_mean" not in outcome.metrics
    assert outcome.metrics["behavior_mean"] == 0.9


def test_record_refuses_an_all_absent_row(registry, spec):
    report = aggregate_parent_result(spec=spec, base_model="A", revision=None, metrics={})
    with pytest.raises(ValueError, match="metrics"):
        record_parent_tournament_result(
            registry,
            report=report,
            run_id="run-parent-3",
            experiment_id="parent-tournament:A",
            artifact_ref="x",
            gpu_hours=0.0,
        )


def test_record_fails_closed_without_anchored_experiment(registry, spec, full_metrics):
    report = aggregate_parent_result(
        spec=spec, base_model="A", revision=None, metrics=full_metrics
    )
    with pytest.raises(ValueError, match="not persisted"):
        record_parent_tournament_result(
            registry,
            report=report,
            run_id="run-parent-4",
            experiment_id="parent-tournament:never-recorded",
            artifact_ref="x",
            gpu_hours=0.0,
        )


def test_report_round_trips_through_its_dict(spec, full_metrics):
    report = aggregate_parent_result(
        spec=spec, base_model="B", revision="r", metrics=full_metrics
    )
    payload = report.to_dict()
    rebuilt = ParentEvalReport(
        base_model=payload["base_model"],
        revision=payload["revision"],
        evaluation_protocol_sha256=payload["evaluation_protocol_sha256"],
        dimensions={
            dim: DimensionScore(
                dimension=dim,
                mean=entry["mean"],
                suite_metrics=entry["suite_metrics"],
            )
            for dim, entry in payload["dimensions"].items()
        },
        capability_mean=payload["capability_mean"],
        behavior_mean=payload["behavior_mean"],
        evidence=payload["evidence"],
    )
    assert rebuilt.to_dict() == payload
