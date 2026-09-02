from chowder.incident import (
    EnvironmentSnapshot,
    FailureCapture,
    SignatureKind,
    classify_signature,
    compute_fingerprint,
)

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


def _capture(exception_type: str, message: str, traceback_text: str = "") -> FailureCapture:
    return FailureCapture(
        incident_id="synthetic",
        experiment_id="synthetic-experiment",
        executor_name="synthetic-executor",
        occurred_at="2026-01-01T00:00:00Z",
        exception_type=exception_type,
        exception_message=message,
        traceback_text=traceback_text,
        environment=EnvironmentSnapshot(hardware_summary="1x Fake GPU", accelerator_count=1),
    )


def test_classify_signature_oom():
    kind = classify_signature("torch.OutOfMemoryError", "CUDA out of memory. Tried to allocate 1.00 GiB.", "")
    assert kind is SignatureKind.CUDA_OOM


def test_classify_signature_device_mismatch_wins_over_generic_cuda_error():
    kind = classify_signature(
        "RuntimeError",
        "Expected all tensors to be on the same device, but got index is on cuda:0",
        "",
    )
    assert kind is SignatureKind.CUDA_DEVICE_MISMATCH


def test_classify_signature_unknown_for_novel_message():
    kind = classify_signature("RuntimeError", "the flux capacitor overheated", "")
    assert kind is SignatureKind.UNKNOWN


def test_fingerprint_normalizes_volatile_numbers():
    a = _capture("torch.OutOfMemoryError", "CUDA out of memory. Tried to allocate 1.00 GiB.")
    b = _capture("torch.OutOfMemoryError", "CUDA out of memory. Tried to allocate 7.25 GiB.")
    assert compute_fingerprint(a).fingerprint_sha256 == compute_fingerprint(b).fingerprint_sha256


def test_fingerprint_is_deterministic():
    a = _capture("RuntimeError", "boom")
    assert compute_fingerprint(a).fingerprint_sha256 == compute_fingerprint(a).fingerprint_sha256


def test_fingerprint_differs_across_signature_kinds():
    oom = _capture("torch.OutOfMemoryError", "CUDA out of memory. Tried to allocate 1.00 GiB.")
    mismatch = _capture("RuntimeError", "Expected all tensors to be on the same device")
    assert compute_fingerprint(oom).fingerprint_sha256 != compute_fingerprint(mismatch).fingerprint_sha256
    assert compute_fingerprint(oom).signature_kind != compute_fingerprint(mismatch).signature_kind


# --- Grounded against today's real incidents -----------------------------
#
# These are not synthetic. If the classifier gets any of today's actual
# failures wrong, the rule set is wrong, not the test.

def test_hf_hub_drop_classifies_as_network_transient():
    assert compute_fingerprint(HF_HUB_TRANSIENT_DROP).signature_kind is SignatureKind.NETWORK_TRANSIENT


def test_peft_kbit_prep_classifies_as_oom():
    assert compute_fingerprint(PEFT_KBIT_PREP_OOM).signature_kind is SignatureKind.CUDA_OOM


def test_kaggle_flattened_dataset_path_classifies_as_artifact_not_found():
    assert (
        compute_fingerprint(KAGGLE_DATASET_FLATTENED_PATH).signature_kind
        is SignatureKind.ARTIFACT_NOT_FOUND
    )


def test_qwen3_5_arch_not_recognized_classifies_as_dependency_incompatible():
    assert (
        compute_fingerprint(QWEN3_5_ARCH_NOT_RECOGNIZED).signature_kind
        is SignatureKind.DEPENDENCY_INCOMPATIBLE
    )


def test_conv1d_no_engine_classifies_as_kernel_unavailable():
    assert (
        compute_fingerprint(QWEN3_5_CONV1D_NO_ENGINE).signature_kind
        is SignatureKind.CUDA_KERNEL_UNAVAILABLE
    )


def test_gated_delta_rule_cublas_classifies_as_execution_failed():
    assert (
        compute_fingerprint(GATED_DELTA_RULE_CUBLAS_FAILURE).signature_kind
        is SignatureKind.CUDA_EXECUTION_FAILED
    )


def test_wrong_accelerator_classifies_as_hardware_incompatible():
    assert (
        compute_fingerprint(WRONG_ACCELERATOR_PROVISIONED).signature_kind
        is SignatureKind.HARDWARE_INCOMPATIBLE
    )


def test_stale_flag_classifies_as_config_invalid():
    assert (
        compute_fingerprint(STALE_EXECUTOR_ARTIFACT_MISSING_FLAG).signature_kind
        is SignatureKind.CONFIG_INVALID
    )


def test_device_map_auto_mismatch_classifies_as_device_mismatch():
    assert (
        compute_fingerprint(DPO_TRAINER_DEVICE_MAP_AUTO_MISMATCH).signature_kind
        is SignatureKind.CUDA_DEVICE_MISMATCH
    )


def test_two_distinct_cuda_kernel_incidents_do_not_collapse_into_one_class():
    """The conv1d "no engine" fix (disabling cuDNN) did not fix the later
    gated-delta-rule cuBLAS failure -- two real, distinct incidents in the
    same custom-op family. A fingerprint that conflated them would have
    made Chowder declare victory after the wrong fix."""
    conv1d = compute_fingerprint(QWEN3_5_CONV1D_NO_ENGINE)
    gated_delta = compute_fingerprint(GATED_DELTA_RULE_CUBLAS_FAILURE)
    assert conv1d.signature_kind != gated_delta.signature_kind
    assert conv1d.fingerprint_sha256 != gated_delta.fingerprint_sha256


def test_recurring_oom_class_stays_distinguishable_across_training_phases():
    """Three real OOMs now (a third was found live, days later, while
    actually trying to unblock DPO_TRAINER_DEVICE_MAP_AUTO_MISMATCH): same
    signature_kind (all are capacity failures), but different incidents --
    different executor phase, different hardware state, different
    forward-vs-backward-pass location. Collapsing any pair would make
    Chowder think a fix for one automatically covers the others -- which
    the real remediation history disproves directly: the length-cap fix
    that resolved the fp32-upcast OOM did NOT resolve the backward-pass
    one."""
    kbit_prep = compute_fingerprint(PEFT_KBIT_PREP_OOM)
    fp32_upcast = compute_fingerprint(DPO_LOGITS_FP32_UPCAST_OOM)
    backward_pass = compute_fingerprint(DPO_BACKWARD_PASS_OOM_NEAR_LIMIT)
    assert kbit_prep.signature_kind is SignatureKind.CUDA_OOM
    assert fp32_upcast.signature_kind is SignatureKind.CUDA_OOM
    assert backward_pass.signature_kind is SignatureKind.CUDA_OOM
    fingerprints = {kbit_prep.fingerprint_sha256, fp32_upcast.fingerprint_sha256,
                    backward_pass.fingerprint_sha256}
    assert len(fingerprints) == 3


def test_backward_pass_oom_classifies_as_oom():
    """The newest dev fixture (added after the first ten, discovered live
    during a real attempt to unblock a different incident) -- confirms it
    lands in the same coarse class as the other two OOMs despite failing in
    accelerate's backward() call rather than the forward pass either of
    them hit."""
    assert compute_fingerprint(DPO_BACKWARD_PASS_OOM_NEAR_LIMIT).signature_kind is SignatureKind.CUDA_OOM


def test_all_dev_fixtures_produce_a_known_signature_kind():
    """None of today's real incidents should fall back to UNKNOWN -- if one
    does, the rule set has a real gap the fixture just found."""
    for capture in ALL_DEV_FIXTURES:
        fingerprint = compute_fingerprint(capture)
        assert fingerprint.signature_kind is not SignatureKind.UNKNOWN, (
            f"{capture.incident_id} classified as UNKNOWN -- rule set gap"
        )


def test_all_dev_fixtures_have_unique_fingerprints():
    """Eleven distinct real incidents; none should accidentally collide."""
    fingerprints = {compute_fingerprint(c).fingerprint_sha256 for c in ALL_DEV_FIXTURES}
    assert len(fingerprints) == len(ALL_DEV_FIXTURES)
