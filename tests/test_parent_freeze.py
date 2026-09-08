"""Regression tests for `chowder.parent_freeze`.

Every fixture builds registry rows through the real, unmodified
`parent_eval.aggregate_parent_result` / `record_parent_tournament_result`
and a real (temp-file, CPU-only) `RunRegistry` -- never a hand-rolled
evidence dict -- so these tests double as a fidelity check that this
module's parsing matches what `parent_tournament.py` actually persists.
Scores are synthetic fixtures (explicitly permitted for tests); no
network, no GPU, no model loads.
"""
from __future__ import annotations

import hashlib

import pytest

from chowder.models import Experiment, ExperimentStatus, Hypothesis
from chowder.parent_eval import (
    PARENT_DIMENSIONS,
    ParentEvalSpec,
    ParentSuiteSpec,
    aggregate_parent_result,
    record_parent_tournament_result,
)
from chowder.parent_freeze import (
    DuplicateParentEvidenceError,
    IncompleteDimensionCoverageError,
    MalformedEvidenceError,
    MissingParentEvidenceError,
    ParentFreezeError,
    ProtocolMismatchError,
    RevisionMismatchError,
    RoleBinding,
    SuiteContentMismatchError,
    TokenizerComparabilityError,
    classify_delta,
    build_selection_packet,
    default_decision_rule,
    freeze_selected_parent,
    validate_packet_for_freeze,
)
from chowder.registry import RunRegistry

PIN_A = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
PIN_B = "404ea47aaa5d8a8b00049c9e9750089aca011ab2"
PIN_C = "a58c3b53b3ce71551eafde2ed5ec8df48e0f4ff8"
PIN_D = "81c73940f94023f7d64e3ae6abcc653fc837d415"

ROLE_BINDINGS = {
    "A": RoleBinding(role="A", label="parent-a", expected_revision=PIN_A),
    "B": RoleBinding(role="B", label="parent-b", expected_revision=PIN_B),
    "C": RoleBinding(role="C", label="parent-c", expected_revision=PIN_C),
    "D": RoleBinding(role="D", label="parent-d", expected_revision=PIN_D),
}
PINS = {"A": PIN_A, "B": PIN_B, "C": PIN_C, "D": PIN_D}


def _suite_name(dimension: str) -> str:
    return f"suite-{dimension}-v1"


def _build_spec(*, max_new_tokens: int = 256, extra_suite_for: str | None = None) -> ParentEvalSpec:
    suites = [
        ParentSuiteSpec(
            name=_suite_name(dimension),
            dimension=dimension,
            dataset=f"synthetic://{dimension}",
            max_new_tokens=max_new_tokens,
        )
        for dimension in PARENT_DIMENSIONS
    ]
    if extra_suite_for is not None:
        suites.append(
            ParentSuiteSpec(
                name=f"suite-{extra_suite_for}-v2-extra",
                dimension=extra_suite_for,
                dataset=f"synthetic://{extra_suite_for}-extra",
                max_new_tokens=max_new_tokens,
            )
        )
    return ParentEvalSpec(suites=tuple(suites))


def _fingerprint(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _record(
    registry: RunRegistry,
    *,
    label: str,
    revision: str,
    spec: ParentEvalSpec,
    scores: dict[str, float],
    suite_digest_seed: str = "protected-v1",
    run_id: str | None = None,
    manifest_sha256: str = "m" * 64,
) -> None:
    experiment_id = f"exp-parent-baseline-{label}"
    if not registry.has_experiment(experiment_id):
        registry.record_experiment(
            Experiment(
                experiment_id=experiment_id,
                parent_id=None,
                hypothesis=Hypothesis(
                    observation="synthetic fixture",
                    suspected_cause="synthetic fixture",
                    intervention="synthetic fixture",
                    expected_deltas={},
                ),
                config_patch={"parent_baseline": True},
                estimated_gpu_hours=0.01,
                status=ExperimentStatus.PASSED,
                tags=("parent-tournament", "protected-suite-v1"),
            )
        )
    suite_evidence = {
        suite.name: {"holdout_fingerprints_sha256": _fingerprint(f"{suite_digest_seed}:{suite.name}")}
        for suite in spec.suites
    }
    report = aggregate_parent_result(
        spec=spec,
        base_model=label,
        revision=revision,
        metrics=scores,
        evidence={
            "model_manifest_sha256": manifest_sha256,
            "wall_seconds": 12.3,
            "peak_gpu_mib_sampled": 1000,
            "worker_runtime": {"python": "3.11"},
            "worker_versions": {"transformers": "5.16.1"},
            "suite_evidence": suite_evidence,
            "prediction_file_sha256": {},
        },
    )
    record_parent_tournament_result(
        registry,
        report=report,
        run_id=run_id or f"tournament-{label}-1",
        experiment_id=experiment_id,
        artifact_ref=f"/nonexistent/{label}",
        gpu_hours=0.01,
    )


def _full_scores(spec: ParentEvalSpec, *, value: float = 1.0, skip: str | None = None) -> dict[str, float]:
    scores = {}
    for suite in spec.suites:
        if skip is not None and suite.name == _suite_name(skip):
            continue
        scores[suite.name] = value
    return scores


def _valid_registry(tmp_path, *, b_value: float = 1.0) -> tuple[RunRegistry, ParentEvalSpec]:
    registry = RunRegistry(tmp_path / "registry.db")
    spec = _build_spec()
    _record(registry, label="parent-a", revision=PIN_A, spec=spec, scores=_full_scores(spec, value=1.0))
    _record(registry, label="parent-b", revision=PIN_B, spec=spec, scores=_full_scores(spec, value=b_value))
    _record(registry, label="parent-c", revision=PIN_C, spec=spec, scores=_full_scores(spec, value=1.0))
    _record(registry, label="parent-d", revision=PIN_D, spec=spec, scores=_full_scores(spec, value=1.0))
    return registry, spec


def _tokenizer_evidence_for_all(identity: str = "tok-identity") -> dict[str, dict]:
    return {
        role: {"tokenizer_class": "Qwen2Tokenizer", "vocab_size": 151936, "identity_sha256": identity}
        for role in ("A", "B", "C", "D")
    }


def test_four_valid_parents_produce_freeze_with_b_selected(tmp_path):
    registry, spec = _valid_registry(tmp_path)
    packet = build_selection_packet(
        registry,
        role_bindings=ROLE_BINDINGS,
        expected_protocol_sha256=spec.digest(),
        tokenizer_evidence=_tokenizer_evidence_for_all(),
    )
    assert packet.all_gates_passed()
    record = freeze_selected_parent(packet)
    assert record.selected_role == "B"
    assert record.selected_label == "parent-b"
    assert record.packet_digest == packet.digest()


def test_incomplete_parent_fails_closed(tmp_path):
    registry = RunRegistry(tmp_path / "registry.db")
    spec = _build_spec()
    _record(registry, label="parent-a", revision=PIN_A, spec=spec, scores=_full_scores(spec))
    _record(registry, label="parent-b", revision=PIN_B, spec=spec, scores=_full_scores(spec))
    _record(registry, label="parent-c", revision=PIN_C, spec=spec, scores=_full_scores(spec))
    # D never evaluated.
    packet = build_selection_packet(
        registry,
        role_bindings=ROLE_BINDINGS,
        expected_protocol_sha256=spec.digest(),
        tokenizer_evidence=_tokenizer_evidence_for_all(),
    )
    assert packet.missing_roles == ("D",)
    assert not packet.all_gates_passed()
    with pytest.raises(MissingParentEvidenceError):
        freeze_selected_parent(packet)


def test_protocol_mismatch_fails_closed(tmp_path):
    registry = RunRegistry(tmp_path / "registry.db")
    good_spec = _build_spec(max_new_tokens=256)
    bad_spec = _build_spec(max_new_tokens=64)  # retry6-style budget: different digest
    _record(registry, label="parent-a", revision=PIN_A, spec=good_spec, scores=_full_scores(good_spec))
    _record(registry, label="parent-b", revision=PIN_B, spec=good_spec, scores=_full_scores(good_spec))
    _record(registry, label="parent-c", revision=PIN_C, spec=good_spec, scores=_full_scores(good_spec))
    _record(registry, label="parent-d", revision=PIN_D, spec=bad_spec, scores=_full_scores(bad_spec))
    packet = build_selection_packet(
        registry,
        role_bindings=ROLE_BINDINGS,
        expected_protocol_sha256=good_spec.digest(),
        tokenizer_evidence=_tokenizer_evidence_for_all(),
    )
    assert not packet.all_gates_passed()
    with pytest.raises(ProtocolMismatchError):
        freeze_selected_parent(packet)


def test_suite_digest_mismatch_fails_closed(tmp_path):
    registry = RunRegistry(tmp_path / "registry.db")
    spec = _build_spec()
    _record(registry, label="parent-a", revision=PIN_A, spec=spec, scores=_full_scores(spec), suite_digest_seed="v1")
    _record(registry, label="parent-b", revision=PIN_B, spec=spec, scores=_full_scores(spec), suite_digest_seed="v1")
    _record(registry, label="parent-c", revision=PIN_C, spec=spec, scores=_full_scores(spec), suite_digest_seed="v1")
    # D's protected suite content digest disagrees, though the protocol (structure) matches.
    _record(registry, label="parent-d", revision=PIN_D, spec=spec, scores=_full_scores(spec), suite_digest_seed="DIFFERENT-CONTENT")
    packet = build_selection_packet(
        registry,
        role_bindings=ROLE_BINDINGS,
        expected_protocol_sha256=spec.digest(),
        tokenizer_evidence=_tokenizer_evidence_for_all(),
    )
    assert not packet.all_gates_passed()
    with pytest.raises(SuiteContentMismatchError):
        freeze_selected_parent(packet)


def test_tokenizer_mismatch_fails_closed(tmp_path):
    registry, spec = _valid_registry(tmp_path)
    tokenizer_evidence = _tokenizer_evidence_for_all()
    tokenizer_evidence["D"] = {"tokenizer_class": "TokenizersBackend", "vocab_size": 151936, "identity_sha256": "different"}
    packet = build_selection_packet(
        registry,
        role_bindings=ROLE_BINDINGS,
        expected_protocol_sha256=spec.digest(),
        tokenizer_evidence=tokenizer_evidence,
    )
    assert not packet.all_gates_passed()
    with pytest.raises(TokenizerComparabilityError):
        freeze_selected_parent(packet)


def test_missing_tokenizer_evidence_fails_closed_unless_bypassed(tmp_path):
    registry, spec = _valid_registry(tmp_path)
    packet = build_selection_packet(
        registry,
        role_bindings=ROLE_BINDINGS,
        expected_protocol_sha256=spec.digest(),
        tokenizer_evidence=None,
    )
    assert not packet.all_gates_passed()
    with pytest.raises(TokenizerComparabilityError):
        freeze_selected_parent(packet)

    bypass_packet = build_selection_packet(
        registry,
        role_bindings=ROLE_BINDINGS,
        expected_protocol_sha256=spec.digest(),
        tokenizer_evidence=None,
        allow_missing_tokenizer_evidence=True,
    )
    record = freeze_selected_parent(bypass_packet)
    assert record.selected_role == "B"


def test_duplicate_inconsistent_rows_fail_closed(tmp_path):
    registry = RunRegistry(tmp_path / "registry.db")
    spec = _build_spec()
    _record(registry, label="parent-a", revision=PIN_A, spec=spec, scores=_full_scores(spec), run_id="run-1")
    # Second, differently-scored row for the same label -- a real
    # inconsistency, not a replay.
    _record(registry, label="parent-a", revision=PIN_A, spec=spec, scores=_full_scores(spec, value=0.5), run_id="run-2")
    _record(registry, label="parent-b", revision=PIN_B, spec=spec, scores=_full_scores(spec))
    _record(registry, label="parent-c", revision=PIN_C, spec=spec, scores=_full_scores(spec))
    _record(registry, label="parent-d", revision=PIN_D, spec=spec, scores=_full_scores(spec))
    with pytest.raises(DuplicateParentEvidenceError):
        build_selection_packet(
            registry,
            role_bindings=ROLE_BINDINGS,
            expected_protocol_sha256=spec.digest(),
            tokenizer_evidence=_tokenizer_evidence_for_all(),
        )


def test_identical_duplicate_rows_are_tolerated(tmp_path):
    registry = RunRegistry(tmp_path / "registry.db")
    spec = _build_spec()
    scores = _full_scores(spec)
    _record(registry, label="parent-a", revision=PIN_A, spec=spec, scores=scores, run_id="run-1")
    # Re-recording with the *same* run_id and identical content is exactly
    # RunRegistry's own idempotent-replay discipline -- record_experiment's
    # has_experiment guard means the second call is a no-op experiment
    # record, and record_parent_tournament_result's evaluation outcome
    # insert is likewise idempotent for identical content.
    _record(registry, label="parent-a", revision=PIN_A, spec=spec, scores=scores, run_id="run-1")
    _record(registry, label="parent-b", revision=PIN_B, spec=spec, scores=scores)
    _record(registry, label="parent-c", revision=PIN_C, spec=spec, scores=scores)
    _record(registry, label="parent-d", revision=PIN_D, spec=spec, scores=scores)
    packet = build_selection_packet(
        registry,
        role_bindings=ROLE_BINDINGS,
        expected_protocol_sha256=spec.digest(),
        tokenizer_evidence=_tokenizer_evidence_for_all(),
    )
    assert packet.all_gates_passed()


def test_missing_dimension_fails_closed(tmp_path):
    registry = RunRegistry(tmp_path / "registry.db")
    spec = _build_spec()
    _record(registry, label="parent-a", revision=PIN_A, spec=spec, scores=_full_scores(spec))
    _record(registry, label="parent-b", revision=PIN_B, spec=spec, scores=_full_scores(spec))
    _record(registry, label="parent-c", revision=PIN_C, spec=spec, scores=_full_scores(spec))
    # D is missing the "coding" suite entirely -- an incomplete run.
    _record(registry, label="parent-d", revision=PIN_D, spec=spec, scores=_full_scores(spec, skip="coding"))
    packet = build_selection_packet(
        registry,
        role_bindings=ROLE_BINDINGS,
        expected_protocol_sha256=spec.digest(),
        tokenizer_evidence=_tokenizer_evidence_for_all(),
    )
    assert not packet.all_gates_passed()
    with pytest.raises(IncompleteDimensionCoverageError):
        freeze_selected_parent(packet)


def test_malformed_evidence_fails_closed(tmp_path):
    registry = RunRegistry(tmp_path / "registry.db")
    spec = _build_spec()
    _record(registry, label="parent-a", revision=PIN_A, spec=spec, scores=_full_scores(spec))
    _record(registry, label="parent-b", revision=PIN_B, spec=spec, scores=_full_scores(spec))
    _record(registry, label="parent-c", revision=PIN_C, spec=spec, scores=_full_scores(spec))

    # D's row is recognizable by its experiment_id (the exact naming
    # convention parent_tournament.py uses) but its evidence payload is
    # corrupted -- no parent_eval_report at all. A truncated write or a
    # schema drift must be reported, never silently treated as "missing".
    from chowder.executors import EvaluationOutcome

    experiment_id = "exp-parent-baseline-parent-d"
    registry.record_experiment(
        Experiment(
            experiment_id=experiment_id,
            parent_id=None,
            hypothesis=Hypothesis(observation="x", suspected_cause="x", intervention="x", expected_deltas={}),
            config_patch={},
            estimated_gpu_hours=0.01,
            status=ExperimentStatus.PASSED,
            tags=(),
        )
    )
    registry.record_evaluation_outcome(
        EvaluationOutcome(
            run_id="tournament-parent-d-corrupt",
            experiment_id=experiment_id,
            source_artifact_ref="/nonexistent/parent-d",
            metrics={"placeholder": 0.0},
            gpu_hours=0.01,
            evidence={"parent_program": "qwen38-native-sparse"},  # no parent_eval_report
        )
    )
    with pytest.raises(MalformedEvidenceError):
        build_selection_packet(
            registry,
            role_bindings=ROLE_BINDINGS,
            expected_protocol_sha256=spec.digest(),
            tokenizer_evidence=_tokenizer_evidence_for_all(),
        )


def test_revision_mismatch_fails_closed(tmp_path):
    registry = RunRegistry(tmp_path / "registry.db")
    spec = _build_spec()
    _record(registry, label="parent-a", revision=PIN_A, spec=spec, scores=_full_scores(spec))
    _record(registry, label="parent-b", revision=PIN_B, spec=spec, scores=_full_scores(spec))
    _record(registry, label="parent-c", revision=PIN_C, spec=spec, scores=_full_scores(spec))
    # D recorded under a revision that does not match its pin (moved main, or a stale cache).
    wrong_revision = "0" * 40
    _record(registry, label="parent-d", revision=wrong_revision, spec=spec, scores=_full_scores(spec))
    packet = build_selection_packet(
        registry,
        role_bindings=ROLE_BINDINGS,
        expected_protocol_sha256=spec.digest(),
        tokenizer_evidence=_tokenizer_evidence_for_all(),
    )
    assert not packet.all_gates_passed()
    with pytest.raises(RevisionMismatchError):
        freeze_selected_parent(packet)


def test_item_unit_classification_tie_weak_signal_clear_difference():
    # Item-unit thresholds: |delta| * 6 < 0.5 -> tie; < 1.5 -> weak-signal; else clear-difference.
    assert classify_delta(0.5, 0.5)["classification"] == "tie"
    assert classify_delta(0.5, 0.5 + (0.4 / 6))["classification"] == "tie"
    assert classify_delta(0.5, 0.5 + (1.0 / 6))["classification"] == "weak-signal"
    assert classify_delta(0.5, 0.5 + (1.4 / 6))["classification"] == "weak-signal"
    assert classify_delta(0.5, 0.5 + (2.0 / 6))["classification"] == "clear-difference"
    assert classify_delta(None, 0.5)["classification"] == "unmeasured"


def test_deterministic_digest_and_freeze_changes_with_material_evidence(tmp_path):
    registry, spec = _valid_registry(tmp_path)
    packet_1 = build_selection_packet(
        registry,
        role_bindings=ROLE_BINDINGS,
        expected_protocol_sha256=spec.digest(),
        tokenizer_evidence=_tokenizer_evidence_for_all(),
        now="2026-09-07T00:00:00+00:00",
    )
    packet_2 = build_selection_packet(
        registry,
        role_bindings=ROLE_BINDINGS,
        expected_protocol_sha256=spec.digest(),
        tokenizer_evidence=_tokenizer_evidence_for_all(),
        now="2026-09-07T00:00:00+00:00",
    )
    assert packet_1.digest() == packet_2.digest()
    record_1 = freeze_selected_parent(packet_1, now="2026-09-07T00:00:01+00:00")
    record_2 = freeze_selected_parent(packet_2, now="2026-09-07T00:00:01+00:00")
    assert record_1.digest() == record_2.digest()

    # A materially different score changes the digest.
    changed_dir = tmp_path / "changed"
    changed_dir.mkdir()
    registry_2, spec_2 = _valid_registry(changed_dir, b_value=0.999999)
    packet_3 = build_selection_packet(
        registry_2,
        role_bindings=ROLE_BINDINGS,
        expected_protocol_sha256=spec_2.digest(),
        tokenizer_evidence=_tokenizer_evidence_for_all(),
        now="2026-09-07T00:00:00+00:00",
    )
    assert packet_3.digest() != packet_1.digest()


def test_no_automatic_selection_when_b_regresses_and_no_clean_alternative(tmp_path):
    registry = RunRegistry(tmp_path / "registry.db")
    spec = _build_spec()
    full = _full_scores(spec, value=1.0)
    # B regresses on "reasoning" (clear difference: 2+ items worse than A).
    b_scores = dict(full)
    b_scores[_suite_name("reasoning")] = 1.0 - (2.0 / 6)
    # C and D each regress on a different dimension too, so no candidate is clean.
    c_scores = dict(full)
    c_scores[_suite_name("coding")] = 1.0 - (2.0 / 6)
    d_scores = dict(full)
    d_scores[_suite_name("knowledge")] = 1.0 - (2.0 / 6)
    _record(registry, label="parent-a", revision=PIN_A, spec=spec, scores=full)
    _record(registry, label="parent-b", revision=PIN_B, spec=spec, scores=b_scores)
    _record(registry, label="parent-c", revision=PIN_C, spec=spec, scores=c_scores)
    _record(registry, label="parent-d", revision=PIN_D, spec=spec, scores=d_scores)
    packet = build_selection_packet(
        registry,
        role_bindings=ROLE_BINDINGS,
        expected_protocol_sha256=spec.digest(),
        tokenizer_evidence=_tokenizer_evidence_for_all(),
    )
    record = freeze_selected_parent(packet)
    assert record.selected_role is None
    assert "no automatic selection" in record.rationale


def test_alternative_selected_when_b_regresses_but_c_is_clean(tmp_path):
    registry = RunRegistry(tmp_path / "registry.db")
    spec = _build_spec()
    full = _full_scores(spec, value=1.0)
    b_scores = dict(full)
    b_scores[_suite_name("reasoning")] = 1.0 - (2.0 / 6)
    d_scores = dict(full)
    d_scores[_suite_name("knowledge")] = 1.0 - (2.0 / 6)
    _record(registry, label="parent-a", revision=PIN_A, spec=spec, scores=full)
    _record(registry, label="parent-b", revision=PIN_B, spec=spec, scores=b_scores)
    _record(registry, label="parent-c", revision=PIN_C, spec=spec, scores=full)
    _record(registry, label="parent-d", revision=PIN_D, spec=spec, scores=d_scores)
    packet = build_selection_packet(
        registry,
        role_bindings=ROLE_BINDINGS,
        expected_protocol_sha256=spec.digest(),
        tokenizer_evidence=_tokenizer_evidence_for_all(),
    )
    record = freeze_selected_parent(packet)
    assert record.selected_role == "C"


def test_validate_packet_for_freeze_returns_none_when_clean(tmp_path):
    registry, spec = _valid_registry(tmp_path)
    packet = build_selection_packet(
        registry,
        role_bindings=ROLE_BINDINGS,
        expected_protocol_sha256=spec.digest(),
        tokenizer_evidence=_tokenizer_evidence_for_all(),
    )
    assert validate_packet_for_freeze(packet) is None
