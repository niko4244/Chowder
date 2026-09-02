"""Run all 11 real dev-set fixtures through the same loop the walking
skeleton (Task 5) proved on one incident -- still entirely within the dev
set, no scoring, no hidden fixtures. Confirms the machinery (generator +
ranking + remediation runner + closeout) handles the full, already-known
variety end-to-end: three CUDA_OOM incidents (two needing different fixes,
one -- found live, later -- that neither known fix resolves), the two
CUDA-RuntimeError incidents that must stay distinct, and two incidents
(device-mismatch, and now the third OOM) that are honestly abandoned rather
than given a fix that was never actually confirmed.
"""
from typing import Any, Mapping

from chowder.benchmark import run_investigation
from chowder.hypothesis_generation import RuleBasedGenerator
from chowder.incident import FailureCapture, compute_fingerprint
from chowder.investigation import (
    Investigation,
    InvestigationStatus,
    RemediationOutcome,
    RemediationRegistry,
    config_patch_digest,
)
from chowder.replay import ReplayGroundTruth

from fixtures_incidents import (
    ALL_DEV_FIXTURES,
    DPO_BACKWARD_PASS_OOM_NEAR_LIMIT,
    DPO_LOGITS_FP32_UPCAST_OOM,
    DPO_TRAINER_DEVICE_MAP_AUTO_MISMATCH,
    GATED_DELTA_RULE_CUBLAS_FAILURE,
    HF_HUB_TRANSIENT_DROP,
    KAGGLE_DATASET_FLATTENED_PATH,
    PEFT_KBIT_PREP_OOM,
    QWEN3_5_ARCH_NOT_RECOGNIZED,
    QWEN3_5_CONV1D_NO_ENGINE,
    STALE_EXECUTOR_ARTIFACT_MISSING_FLAG,
    WRONG_ACCELERATOR_PROVISIONED,
)

# Per-fixture expected terminal status and (if resolved) which config_patch
# should be the one that resolved it. DPO_TRAINER_DEVICE_MAP_AUTO_MISMATCH
# is ABANDONED, not RESOLVED -- this exact incident was still unresolved in
# the real training session these fixtures are transcribed from, and the
# ground truth below deliberately does not invent a fix that was never
# actually confirmed.
_EXPECTATIONS: tuple[tuple[FailureCapture, InvestigationStatus, Mapping[str, Any] | None], ...] = (
    (HF_HUB_TRANSIENT_DROP, InvestigationStatus.RESOLVED, {"resume_download": True}),
    (
        PEFT_KBIT_PREP_OOM,
        InvestigationStatus.RESOLVED,
        {"allocator_conf": "expandable_segments:True"},
    ),
    (
        KAGGLE_DATASET_FLATTENED_PATH,
        InvestigationStatus.RESOLVED,
        {"dataset_path_resolution": "search_by_filename"},
    ),
    (
        QWEN3_5_ARCH_NOT_RECOGNIZED,
        InvestigationStatus.RESOLVED,
        {"transformers_version": "5.10.2"},
    ),
    (QWEN3_5_CONV1D_NO_ENGINE, InvestigationStatus.RESOLVED, {"cudnn_enabled": False}),
    (
        GATED_DELTA_RULE_CUBLAS_FAILURE,
        InvestigationStatus.RESOLVED,
        {"transformers_version": "5.10.2"},
    ),
    (
        WRONG_ACCELERATOR_PROVISIONED,
        InvestigationStatus.RESOLVED,
        {"kernel_metadata.machine_shape": "NvidiaTeslaT4"},
    ),
    (
        STALE_EXECUTOR_ARTIFACT_MISSING_FLAG,
        InvestigationStatus.RESOLVED,
        {"resync_kernel_dataset": True},
    ),
    (DPO_TRAINER_DEVICE_MAP_AUTO_MISMATCH, InvestigationStatus.ABANDONED, None),
    (DPO_LOGITS_FP32_UPCAST_OOM, InvestigationStatus.RESOLVED, {"max_length": 1024}),
    (DPO_BACKWARD_PASS_OOM_NEAR_LIMIT, InvestigationStatus.ABANDONED, None),
)

# Ground truth for every candidate the generator could plausibly propose for
# each fixture's signature_kind, not just the one this run happens to try --
# ReplayGroundTruth treats an unlisted patch as a hard error by design
# (replay.py), so every candidate the table could produce must be covered.
#
# Keyed by incident_id, not the FailureCapture itself: FailureCapture is a
# frozen dataclass, but its environment carries a plain dict
# (installed_packages), which is unhashable -- frozen alone does not make a
# dataclass usable as a dict/set key when one of its fields isn't hashable.
_GROUND_TRUTH_OUTCOMES: Mapping[str, Mapping[str, RemediationOutcome]] = {
    HF_HUB_TRANSIENT_DROP.incident_id: {
        config_patch_digest({"resume_download": True}): RemediationOutcome.RESOLVED,
    },
    PEFT_KBIT_PREP_OOM.incident_id: {
        config_patch_digest(
            {"allocator_conf": "expandable_segments:True"}
        ): RemediationOutcome.RESOLVED,
        config_patch_digest({"max_length": 1024}): RemediationOutcome.DID_NOT_RESOLVE,
    },
    KAGGLE_DATASET_FLATTENED_PATH.incident_id: {
        config_patch_digest(
            {"dataset_path_resolution": "search_by_filename"}
        ): RemediationOutcome.RESOLVED,
    },
    QWEN3_5_ARCH_NOT_RECOGNIZED.incident_id: {
        config_patch_digest({"transformers_version": "5.10.2"}): RemediationOutcome.RESOLVED,
    },
    QWEN3_5_CONV1D_NO_ENGINE.incident_id: {
        config_patch_digest({"cudnn_enabled": False}): RemediationOutcome.RESOLVED,
    },
    GATED_DELTA_RULE_CUBLAS_FAILURE.incident_id: {
        config_patch_digest({"transformers_version": "5.10.2"}): RemediationOutcome.RESOLVED,
    },
    WRONG_ACCELERATOR_PROVISIONED.incident_id: {
        config_patch_digest(
            {"kernel_metadata.machine_shape": "NvidiaTeslaT4"}
        ): RemediationOutcome.RESOLVED,
    },
    STALE_EXECUTOR_ARTIFACT_MISSING_FLAG.incident_id: {
        config_patch_digest({"resync_kernel_dataset": True}): RemediationOutcome.RESOLVED,
    },
    DPO_TRAINER_DEVICE_MAP_AUTO_MISMATCH.incident_id: {
        config_patch_digest({"device_map": "balanced_low_0"}): RemediationOutcome.DID_NOT_RESOLVE,
    },
    DPO_LOGITS_FP32_UPCAST_OOM.incident_id: {
        config_patch_digest(
            {"allocator_conf": "expandable_segments:True"}
        ): RemediationOutcome.DID_NOT_RESOLVE,
        config_patch_digest({"max_length": 1024}): RemediationOutcome.RESOLVED,
    },
    # Confirmed real evidence, not a guess: allocator_conf was already active
    # the whole time this incident occurred and did not prevent it; a
    # max-length cap (tightened twice, 900 tokens then 850) got past the
    # unrelated forward-pass OOM but never closed this incident's own
    # backward-pass gap. Neither known CUDA_OOM remediation resolves this --
    # matches DPO_TRAINER_DEVICE_MAP_AUTO_MISMATCH's honesty discipline.
    DPO_BACKWARD_PASS_OOM_NEAR_LIMIT.incident_id: {
        config_patch_digest(
            {"allocator_conf": "expandable_segments:True"}
        ): RemediationOutcome.DID_NOT_RESOLVE,
        config_patch_digest({"max_length": 1024}): RemediationOutcome.DID_NOT_RESOLVE,
    },
}


def _run_one(
    capture: FailureCapture,
    registry: RemediationRegistry,
    *,
    investigation_id: str,
) -> tuple[Investigation, RemediationRegistry]:
    """Thin wrapper around chowder.benchmark.run_investigation (Task 8),
    which promoted this file's original hand-written loop to real source
    once the benchmark scorer needed the same orchestration. A uniform
    1.0-hour budget is enough for every dev fixture, including
    DPO_TRAINER_DEVICE_MAP_AUTO_MISMATCH: it no longer needs a specially
    tight budget to reach ABANDONED, now that Investigation.abandon()
    covers "ran out of candidate hypotheses" as its own terminal path,
    independent of budget exhaustion.
    """
    truth = ReplayGroundTruth(
        fingerprint_sha256=compute_fingerprint(capture).fingerprint_sha256,
        outcomes=_GROUND_TRUTH_OUTCOMES[capture.incident_id],
    )
    return run_investigation(
        capture, truth, RuleBasedGenerator(), registry, investigation_id=investigation_id
    )


def _resolving_record(investigation: Investigation):
    """The RESOLVED remediation, not just any trial that has a remediation
    attached -- a multi-candidate investigation can carry a DID_NOT_RESOLVE
    record on an earlier trial before the one that actually resolved it."""
    return next(
        t.remediation
        for t in investigation.trials
        if t.remediation is not None and t.remediation.outcome is RemediationOutcome.RESOLVED
    )


def test_all_dev_fixtures_expectation_table_is_complete():
    """Guards the test itself, not just the code under test: if a fixture is
    ever added to ALL_DEV_FIXTURES without updating this file's expectation
    and ground-truth tables, this fails loudly instead of the new fixture
    silently being skipped."""
    assert len(ALL_DEV_FIXTURES) == 11
    assert {capture.incident_id for capture, _, _ in _EXPECTATIONS} == {
        c.incident_id for c in ALL_DEV_FIXTURES
    }
    assert set(_GROUND_TRUTH_OUTCOMES) == {c.incident_id for c in ALL_DEV_FIXTURES}


def test_all_dev_fixtures_resolve_or_are_correctly_abandoned():
    registry = RemediationRegistry()
    for index, (capture, expected_status, expected_patch) in enumerate(_EXPECTATIONS):
        investigation, registry = _run_one(
            capture, registry, investigation_id=f"inv-devrun-{index}"
        )
        assert investigation.status is expected_status, (
            f"{capture.incident_id}: expected {expected_status}, got {investigation.status}"
        )
        if expected_patch is not None:
            assert dict(_resolving_record(investigation).config_patch) == expected_patch


def test_conv1d_and_cublas_incidents_never_collapse_through_the_investigation_path():
    """Regression, re-checked at the investigation level (test_incident.py
    already checks it at the fingerprinting level): these two real
    incidents are both CUDA RuntimeErrors in the same custom-op family, but
    the fix for one (disable cuDNN) does nothing for the other (needs a
    library bump instead) -- they must resolve via genuinely different
    config patches, not share a remediation."""
    registry = RemediationRegistry()
    conv1d_investigation, registry = _run_one(
        QWEN3_5_CONV1D_NO_ENGINE, registry, investigation_id="inv-conv1d"
    )
    cublas_investigation, registry = _run_one(
        GATED_DELTA_RULE_CUBLAS_FAILURE, registry, investigation_id="inv-cublas"
    )

    assert conv1d_investigation.fingerprint.fingerprint_sha256 != (
        cublas_investigation.fingerprint.fingerprint_sha256
    )
    conv1d_record = _resolving_record(conv1d_investigation)
    cublas_record = _resolving_record(cublas_investigation)
    assert dict(conv1d_record.config_patch) != dict(cublas_record.config_patch)


def test_recurring_oom_incidents_need_different_fixes_through_the_investigation_path():
    """Regression, re-checked at the investigation level: both OOMs share
    signature_kind=CUDA_OOM, but the allocator-config fix that resolves the
    kbit-prep OOM does NOT resolve the fp32-logits-upcast OOM, which instead
    needs the sequence-length trim -- proving the ranked multi-candidate
    loop actually falls through to a second hypothesis rather than the
    first candidate happening to work for both by coincidence of the test
    setup."""
    registry = RemediationRegistry()
    kbit_investigation, registry = _run_one(
        PEFT_KBIT_PREP_OOM, registry, investigation_id="inv-kbit-oom"
    )
    upcast_investigation, registry = _run_one(
        DPO_LOGITS_FP32_UPCAST_OOM, registry, investigation_id="inv-upcast-oom"
    )

    assert kbit_investigation.fingerprint.fingerprint_sha256 != (
        upcast_investigation.fingerprint.fingerprint_sha256
    )
    kbit_record = _resolving_record(kbit_investigation)
    upcast_record = _resolving_record(upcast_investigation)
    assert dict(kbit_record.config_patch) == {"allocator_conf": "expandable_segments:True"}
    assert dict(upcast_record.config_patch) == {"max_length": 1024}

    # The upcast investigation must have actually tried and failed the
    # allocator-config candidate first (ranking puts it first: same probe
    # corroboration, lower estimated cost) before falling through -- not
    # simply never proposed it.
    upcast_trial_patches = {dict(t.config_patch).get("allocator_conf") for t in upcast_investigation.trials}
    assert "expandable_segments:True" in upcast_trial_patches
    failed_allocator_trial = next(
        t for t in upcast_investigation.trials if dict(t.config_patch).get("allocator_conf")
    )
    assert failed_allocator_trial.remediation is not None
    assert failed_allocator_trial.remediation.outcome is RemediationOutcome.DID_NOT_RESOLVE
