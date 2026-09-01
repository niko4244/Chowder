import pytest

from chowder.incident import compute_fingerprint
from chowder.investigation import (
    Investigation,
    InvestigationStatus,
    RemediationOutcome,
    RemediationRecord,
    RemediationRegistry,
    route_failure,
)
from chowder.models import Hypothesis

from fixtures_incidents import (
    DPO_LOGITS_FP32_UPCAST_OOM,
    GATED_DELTA_RULE_CUBLAS_FAILURE,
    PEFT_KBIT_PREP_OOM,
    QWEN3_5_CONV1D_NO_ENGINE,
)


def _resolved_record(fingerprint_sha256: str, signature_kind, config_patch=None) -> RemediationRecord:
    return RemediationRecord(
        remediation_id=f"fix-for-{fingerprint_sha256[:8]}",
        fingerprint_sha256=fingerprint_sha256,
        signature_kind=signature_kind,
        description="a fix that worked",
        config_patch=config_patch or {"applied": True},
        outcome=RemediationOutcome.RESOLVED,
        attempts_used=1,
        gpu_hours_spent=0.1,
    )


def test_route_failure_with_empty_registry_opens_investigation():
    capture = PEFT_KBIT_PREP_OOM
    fingerprint = compute_fingerprint(capture)
    result = route_failure(
        capture, fingerprint, RemediationRegistry(),
        gpu_hour_budget=1.0, investigation_id="inv-1",
    )
    assert isinstance(result, Investigation)
    assert result.status is InvestigationStatus.OPEN


def test_route_failure_returns_known_fix_on_exact_fingerprint_match():
    capture = PEFT_KBIT_PREP_OOM
    fingerprint = compute_fingerprint(capture)
    record = _resolved_record(fingerprint.fingerprint_sha256, fingerprint.signature_kind)
    registry = RemediationRegistry(records=(record,))
    result = route_failure(
        capture, fingerprint, registry, gpu_hour_budget=1.0, investigation_id="inv-2",
    )
    assert result is record


def test_route_failure_does_not_auto_apply_same_class_different_incident():
    """The core safety property: a resolved fix for one CUDA_OOM incident
    must NOT be auto-applied to a different CUDA_OOM incident just because
    they share a signature_kind. Grounded in two real OOMs from today that
    happened in genuinely different places (PEFT's kbit-prep pass vs.
    accelerate's fp32 logits upcast) -- collapsing them would have wasted
    a real attempt on the wrong fix."""
    resolved_fingerprint = compute_fingerprint(PEFT_KBIT_PREP_OOM)
    new_fingerprint = compute_fingerprint(DPO_LOGITS_FP32_UPCAST_OOM)
    assert resolved_fingerprint.signature_kind == new_fingerprint.signature_kind
    assert resolved_fingerprint.fingerprint_sha256 != new_fingerprint.fingerprint_sha256

    record = _resolved_record(resolved_fingerprint.fingerprint_sha256, resolved_fingerprint.signature_kind)
    registry = RemediationRegistry(records=(record,))
    result = route_failure(
        DPO_LOGITS_FP32_UPCAST_OOM, new_fingerprint, registry,
        gpu_hour_budget=1.0, investigation_id="inv-3",
    )
    assert isinstance(result, Investigation)


def test_route_failure_does_not_apply_unresolved_record():
    capture = QWEN3_5_CONV1D_NO_ENGINE
    fingerprint = compute_fingerprint(capture)
    failed_record = RemediationRecord(
        remediation_id="attempted-and-failed",
        fingerprint_sha256=fingerprint.fingerprint_sha256,
        signature_kind=fingerprint.signature_kind,
        description="tried, did not work",
        config_patch={"tried": True},
        outcome=RemediationOutcome.DID_NOT_RESOLVE,
        attempts_used=1,
        gpu_hours_spent=0.1,
    )
    registry = RemediationRegistry(records=(failed_record,))
    result = route_failure(
        capture, fingerprint, registry, gpu_hour_budget=1.0, investigation_id="inv-4",
    )
    assert isinstance(result, Investigation)


def test_add_hypothesis_rejects_budget_exhausted():
    fingerprint = compute_fingerprint(QWEN3_5_CONV1D_NO_ENGINE)
    investigation = Investigation(
        investigation_id="inv-5",
        fingerprint=fingerprint,
        capture=QWEN3_5_CONV1D_NO_ENGINE,
        gpu_hour_budget=0.0,
    )
    hypothesis = Hypothesis(
        observation="conv1d raises 'no engine' under gradient checkpointing",
        suspected_cause="cuDNN v8 kernel-selection gap for this op configuration",
        intervention="disable cuDNN for conv ops",
    )
    with pytest.raises(ValueError, match="no remaining GPU-hour budget"):
        investigation.add_hypothesis(hypothesis, config_patch={"cudnn_enabled": False})


def test_add_hypothesis_rejects_repeating_a_failed_intervention():
    fingerprint = compute_fingerprint(GATED_DELTA_RULE_CUBLAS_FAILURE)
    already_tried = RemediationRecord(
        remediation_id="eager-attention-attempt",
        fingerprint_sha256="some-other-fingerprint",
        signature_kind=fingerprint.signature_kind,
        description="forced eager attention",
        config_patch={"attn_implementation": "eager"},
        outcome=RemediationOutcome.DID_NOT_RESOLVE,
        attempts_used=1,
        gpu_hours_spent=0.1,
    )
    registry = RemediationRegistry(records=(already_tried,))
    investigation = Investigation(
        investigation_id="inv-6",
        fingerprint=fingerprint,
        capture=GATED_DELTA_RULE_CUBLAS_FAILURE,
        gpu_hour_budget=5.0,
    )
    hypothesis = Hypothesis(
        observation="cuBLAS execution failed in the fused kernel",
        suspected_cause="attn_implementation routes through the same fused kernel regardless",
        intervention="force eager attention",
    )
    with pytest.raises(ValueError, match="already failed"):
        investigation.add_hypothesis(
            hypothesis, config_patch={"attn_implementation": "eager"}, registry=registry
        )

    # A genuinely different intervention for the same incident is fine.
    different_hypothesis = Hypothesis(
        observation="cuBLAS execution failed in the fused kernel",
        suspected_cause="library version gap in this fused kernel's implementation",
        intervention="bump transformers to a version known to load this model successfully elsewhere",
    )
    trial = investigation.add_hypothesis(
        different_hypothesis, config_patch={"transformers_version": "5.10.2"}, registry=registry
    )
    assert trial in investigation.trials
    assert investigation.status is InvestigationStatus.HYPOTHESIS_TESTING


def test_resolve_transitions_status_and_accumulates_spend():
    fingerprint = compute_fingerprint(QWEN3_5_CONV1D_NO_ENGINE)
    investigation = Investigation(
        investigation_id="inv-7",
        fingerprint=fingerprint,
        capture=QWEN3_5_CONV1D_NO_ENGINE,
        gpu_hour_budget=1.0,
    )
    hypothesis = Hypothesis(observation="o", suspected_cause="c", intervention="i")
    trial = investigation.add_hypothesis(hypothesis, config_patch={"x": 1})
    remediation = _resolved_record(fingerprint.fingerprint_sha256, fingerprint.signature_kind)
    investigation.resolve(trial, remediation)
    assert investigation.status is InvestigationStatus.RESOLVED
    assert investigation.gpu_hours_spent == pytest.approx(0.1)


def test_resolve_rejects_non_resolving_remediation():
    fingerprint = compute_fingerprint(QWEN3_5_CONV1D_NO_ENGINE)
    investigation = Investigation(
        investigation_id="inv-8",
        fingerprint=fingerprint,
        capture=QWEN3_5_CONV1D_NO_ENGINE,
        gpu_hour_budget=1.0,
    )
    hypothesis = Hypothesis(observation="o", suspected_cause="c", intervention="i")
    trial = investigation.add_hypothesis(hypothesis, config_patch={"x": 1})
    not_resolved = RemediationRecord(
        remediation_id="r",
        fingerprint_sha256=fingerprint.fingerprint_sha256,
        signature_kind=fingerprint.signature_kind,
        description="d",
        config_patch={"x": 1},
        outcome=RemediationOutcome.DID_NOT_RESOLVE,
        attempts_used=1,
        gpu_hours_spent=0.1,
    )
    with pytest.raises(ValueError, match="non-resolving"):
        investigation.resolve(trial, not_resolved)


def test_record_failed_trial_abandons_once_budget_exhausted():
    fingerprint = compute_fingerprint(GATED_DELTA_RULE_CUBLAS_FAILURE)
    investigation = Investigation(
        investigation_id="inv-9",
        fingerprint=fingerprint,
        capture=GATED_DELTA_RULE_CUBLAS_FAILURE,
        gpu_hour_budget=0.15,
    )
    h1 = Hypothesis(observation="o1", suspected_cause="c1", intervention="i1")
    h2 = Hypothesis(observation="o2", suspected_cause="c2", intervention="i2")
    trial1 = investigation.add_hypothesis(h1, config_patch={"x": 1})
    failed1 = RemediationRecord(
        remediation_id="r1", fingerprint_sha256=fingerprint.fingerprint_sha256,
        signature_kind=fingerprint.signature_kind, description="d",
        config_patch={"x": 1}, outcome=RemediationOutcome.DID_NOT_RESOLVE,
        attempts_used=1, gpu_hours_spent=0.1,
    )
    investigation.record_failed_trial(trial1, failed1)
    assert investigation.status is InvestigationStatus.HYPOTHESIS_TESTING
    assert investigation.remaining_budget() == pytest.approx(0.05)

    trial2 = investigation.add_hypothesis(h2, config_patch={"y": 2})
    failed2 = RemediationRecord(
        remediation_id="r2", fingerprint_sha256=fingerprint.fingerprint_sha256,
        signature_kind=fingerprint.signature_kind, description="d",
        config_patch={"y": 2}, outcome=RemediationOutcome.DID_NOT_RESOLVE,
        attempts_used=1, gpu_hours_spent=0.1,
    )
    investigation.record_failed_trial(trial2, failed2)
    assert investigation.status is InvestigationStatus.ABANDONED
