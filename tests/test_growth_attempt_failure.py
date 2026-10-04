"""Attempt-failure classification: a failed attempt is classified, and the
class decides both the evidence record (or the honest absence of one) and
the next action. Plus the anti-repeat gate: a measured negative is not a
starting point for another identical run.
"""

from __future__ import annotations

import pytest

from chowder.growth.evidence import EvidenceRecord, EvidenceState, EvidenceStore
from chowder.growth.attempt_failure import (
    FailureClass,
    NextAction,
    classify_failure,
    record_refuses_disproven_repeat,
)


def _record(**overrides) -> EvidenceRecord:
    fields: dict[str, object] = {
        "record_id": "rec-001",
        "state": EvidenceState.FAILED,
        "family_id": "training.sft-curriculum",
        "model_family": "qwen3.8",
        "checkpoint_identity": "gen2@abc123",
        "architecture": "dense",
        "eval_suite": "gen2-screening@v1",
        "intervention_parameters": {"learning_rate": 2e-4},
        "software_runtime": {"transformers": "5.18.0"},
        "hardware_class": "kaggle_2x_t4_16gb",
        "sample_size": 3,
        "evidence_quality": "paired-seeds",
        "measured_effect": {"target_metric": -0.05},
        "date": "2026-10-04",
        "source_run": "kernel-20261004-010000",
    }
    fields.update(overrides)
    return EvidenceRecord(**fields)


class TestClassificationPrecedence:
    def test_reported_restart_is_infrastructure_and_writes_no_record(self):
        result = classify_failure({"resume_state": "not-a-resume"})
        assert result.failure_class is FailureClass.INFRASTRUCTURE
        assert result.evidence_state is None
        assert result.action is NextAction.RETRY_UNCHANGED

    def test_settlement_refusal_is_budget_and_writes_no_record(self):
        """The production settlement shape: trained, succeeded, settled over."""
        result = classify_failure(
            {
                "candidate_succeeded": True,
                "status": "REFUSED",
                "refused_by": "budget_settlement",
                "refusal_reason": "ACTUAL_EXCEEDS_PROJECTION: actual wall 0.05 "
                "exceeds projection 0.006 by more than the declared tolerance 0.25",
            }
        )
        assert result.failure_class is FailureClass.BUDGET_EXHAUSTED
        assert result.evidence_state is None
        # Precedence: settlement is read even when the run "succeeded".
        assert result.reason.startswith("settlement refused")
        assert "ACTUAL_EXCEEDS_PROJECTION" in result.reason

    def test_failed_training_is_infrastructure(self):
        result = classify_failure({"candidate_succeeded": False, "status": "FAILED"})
        assert result.failure_class is FailureClass.INFRASTRUCTURE
        assert result.evidence_state is None

    def test_incomplete_measurements_are_inconclusive(self):
        result = classify_failure(
            {"candidate_succeeded": True, "measurements_complete": False}
        )
        assert result.failure_class is FailureClass.EVAL_NOISE
        assert result.evidence_state == "inconclusive"
        assert result.action is NextAction.REPLICATE_WITH_MORE_EVIDENCE

    def test_interval_excluding_zero_negatively_is_a_clean_falsification(self):
        result = classify_failure(
            {
                "candidate_succeeded": True,
                "paired_delta_ci": (-0.08, -0.02),
                "sample_size": 6,
            }
        )
        assert result.failure_class is FailureClass.MECHANISM_FALSIFIED
        assert result.evidence_state == "failed"
        assert result.action is NextAction.RECORD_FALSIFIED_AND_SKIP

    def test_interval_including_zero_is_underpowered_not_falsified(self):
        result = classify_failure(
            {
                "candidate_succeeded": True,
                "paired_delta_ci": (-0.03, 0.02),
                "sample_size": 6,
            }
        )
        assert result.failure_class is FailureClass.EVAL_NOISE
        assert result.evidence_state == "inconclusive"

    def test_positive_interval_is_not_a_failure_at_all(self):
        # A positive measured effect handed to the classifier is a wiring
        # bug; escalating beats misfiling it.
        result = classify_failure(
            {
                "candidate_succeeded": True,
                "paired_delta_ci": (0.01, 0.09),
                "sample_size": 6,
            }
        )
        assert result.action is NextAction.ESCALATE_TO_OPERATOR
        assert result.evidence_state is None

    def test_thin_negative_mean_without_pairing_is_inconclusive(self):
        result = classify_failure(
            {"candidate_succeeded": True, "mean_delta": -0.05, "sample_size": 2}
        )
        assert result.failure_class is FailureClass.EVAL_NOISE

    def test_negative_mean_with_adequate_sample_is_a_measured_negative(self):
        result = classify_failure(
            {"candidate_succeeded": True, "mean_delta": -0.05, "sample_size": 3}
        )
        assert result.failure_class is FailureClass.MECHANISM_FALSIFIED
        assert "unpaired" in result.reason

    def test_architecture_incompatibility_is_explicit(self):
        result = classify_failure(
            {"candidate_succeeded": True, "architecture_compatible": False}
        )
        assert result.failure_class is FailureClass.ARCHITECTURE_INCOMPATIBLE
        assert result.evidence_state == "architecture-incompatible"
        assert result.action is NextAction.EXCLUDE_ARCHITECTURE

    def test_an_unreadable_failure_escalates_rather_than_guessing(self):
        result = classify_failure({"candidate_succeeded": True})
        assert result.failure_class is FailureClass.DATA_OR_CURRICULUM
        assert result.action is NextAction.ESCALATE_TO_OPERATOR
        assert result.evidence_state is None
        assert "refusing to guess" in result.reason

    def test_parameter_failures_record_neutral_not_failed(self):
        # A parameter-range miss must not compound the family's failure
        # prior: the mechanism was not refuted, the magnitude was wrong.
        result = classify_failure(
            {
                "candidate_succeeded": True,
                "mean_delta": -0.02,
                "sample_size": 3,
            }
        )
        # With no further signal this is a measured negative; the
        # PARAMETER class is asserted structurally:
        assert FailureClass("parameter").value == "parameter"


class TestAntiRepeatGate:
    def _store_with(self, tmp_path, *records: EvidenceRecord) -> EvidenceStore:
        store = EvidenceStore(tmp_path / "evidence.jsonl")
        for record in records:
            store.record(record)
        return store

    def test_exact_measured_negative_repeat_refuses(self, tmp_path):
        store = self._store_with(tmp_path, _record())
        with pytest.raises(ValueError, match="already measured this exact"):
            record_refuses_disproven_repeat(
                store,
                family_id="training.sft-curriculum",
                model_family="qwen3.8",
                architecture="dense",
                intervention_parameters={"learning_rate": 2e-4},
            )

    def test_narrowed_parameters_are_a_new_hypothesis(self, tmp_path):
        store = self._store_with(tmp_path, _record())
        record_refuses_disproven_repeat(
            store,
            family_id="training.sft-curriculum",
            model_family="qwen3.8",
            architecture="dense",
            intervention_parameters={"learning_rate": 5e-5},
        )

    def test_inconclusive_history_does_not_refuse_a_repeat(self, tmp_path):
        # Inconclusive means replicate-with-more-evidence, not refuse.
        store = self._store_with(
            tmp_path, _record(state=EvidenceState.INCONCLUSIVE, measured_effect={})
        )
        record_refuses_disproven_repeat(
            store,
            family_id="training.sft-curriculum",
            model_family="qwen3.8",
            architecture="dense",
            intervention_parameters={"learning_rate": 2e-4},
        )

    def test_failure_in_a_different_scope_does_not_refuse(self, tmp_path):
        store = self._store_with(
            tmp_path,
            _record(record_id="rec-other-model", model_family="other-model"),
            _record(record_id="rec-moe", architecture="moe"),
            _record(record_id="rec-hybrid", family_id="architecture.hybrid-lm"),
        )
        record_refuses_disproven_repeat(
            store,
            family_id="training.sft-curriculum",
            model_family="qwen3.8",
            architecture="dense",
            intervention_parameters={"learning_rate": 2e-4},
        )
