"""Real executor incidents captured from a live Kaggle DPO training session
(niko4244/frontier-lowram-autoresearch, cycle-4 math training, 2026-08-31 to
2026-09-01). These are the Investigator's development-set fixtures: exact
exception text and environment detail transcribed from the actual run logs,
not invented. Held-out/mutated variants for the hidden evaluation set belong
in a separate fixture module once the investigation layer exists to be
scored against them -- mixing dev and eval fixtures in one file defeats the
point of the split.

Each fixture is the raw evidence only (a FailureCapture). What Chowder
*should* have concluded from it is recorded in the docstring, not baked into
the fixture itself, so future investigation-layer tests can assert against
that expectation independently.
"""
from __future__ import annotations

from chowder.incident import EnvironmentSnapshot, FailureCapture

_RTX_5060TI_ENV = EnvironmentSnapshot(
    hardware_summary="1x NVIDIA GeForce RTX 5060 Ti, 16311 MiB, sm_120 (Blackwell)",
    accelerator_count=1,
    installed_packages={"transformers": "5.5.4", "torch": "2.7", "peft": "0.19.1"},
)

_KAGGLE_T4_ENV = EnvironmentSnapshot(
    hardware_summary="1x Tesla T4, sm_75",
    accelerator_count=1,
    installed_packages={
        "trl": "0.24.0",
        "transformers": "5.5.4",
        "peft": "0.19.1",
        "bitsandbytes": "0.50.2",
        "accelerate": "1.6.0",
    },
)

_KAGGLE_T4_ENV_5_10_2 = EnvironmentSnapshot(
    hardware_summary="2x Tesla T4, sm_75",
    accelerator_count=2,
    installed_packages={
        "trl": "0.24.0",
        "transformers": "5.10.2",
        "peft": "0.19.1",
        "bitsandbytes": "0.49.2",
        "accelerate": "1.13.0",
    },
)


# 1. NETWORK_TRANSIENT -- expected: retry (the HF Hub cache resumes the
#    broken shard via HTTP Range on the next attempt within the same
#    container; do not restart the whole ~19GB download).
HF_HUB_TRANSIENT_DROP = FailureCapture(
    incident_id="2026-08-29-seed43-hf-hub-drop",
    experiment_id="dpo-cycle3-mixed-v2-seed43",
    executor_name="kaggle-train-dpo-trl",
    occurred_at="2026-08-29T21:xx:00Z",
    exception_type="httpx.RemoteProtocolError",
    exception_message=(
        "peer closed connection without sending complete message body "
        "(received 4581139313 bytes, expected 4987757928)"
    ),
    traceback_text=(
        "File \"transformers/modeling_utils.py\", line 4057, in from_pretrained\n"
        "  checkpoint_files, sharded_metadata = _get_resolved_checkpoint_files(...)\n"
        "File \"huggingface_hub/file_download.py\", line 421, in http_get\n"
        "  for chunk in response.iter_bytes(chunk_size=constants.DOWNLOAD_CHUNK_SIZE):\n"
        "httpx.RemoteProtocolError: peer closed connection without sending complete "
        "message body (received 4581139313 bytes, expected 4987757928)"
    ),
    environment=_KAGGLE_T4_ENV,
    attempt_number=1,
    gpu_hours_spent=0.02,
)


# 2. CUDA_OOM -- expected: distinguish from a code-correctness failure and
#    reduce memory pressure (allocator config, batch/sequence trimming, or
#    more accelerator memory), not a blind retry.
PEFT_KBIT_PREP_OOM = FailureCapture(
    incident_id="2026-08-29-seed43-kbit-oom",
    experiment_id="dpo-cycle3-mixed-v2-seed43",
    executor_name="kaggle-train-dpo-trl",
    occurred_at="2026-08-29T22:xx:00Z",
    exception_type="torch.OutOfMemoryError",
    exception_message=(
        "CUDA out of memory. Tried to allocate 3.79 GiB. GPU 0 has a total capacity "
        "of 14.56 GiB of which 3.44 GiB is free. Including non-PyTorch memory, this "
        "process has 11.12 GiB memory in use."
    ),
    traceback_text=(
        "File \"peft/utils/other.py\", line 186, in prepare_model_for_kbit_training\n"
        "  param.data = param.data.to(torch.float32)\n"
        "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 3.79 GiB."
    ),
    environment=_KAGGLE_T4_ENV,
    attempt_number=1,
    gpu_hours_spent=0.05,
)


# 3. ARTIFACT_NOT_FOUND -- expected: recognize a Kaggle-specific packaging
#    quirk (a single-directory dataset upload gets flattened into the
#    dataset root instead of keeping the folder name) and locate by file,
#    not by assumed directory structure -- not a data-integrity problem.
KAGGLE_DATASET_FLATTENED_PATH = FailureCapture(
    incident_id="2026-08-29-math500-baseline-locate-dir",
    experiment_id="cycle4-math500-baseline-v1",
    executor_name="kaggle-run-math500-baseline",
    occurred_at="2026-08-29T14:xx:00Z",
    exception_type="FileNotFoundError",
    exception_message=(
        "code/ not found under /kaggle/input. Tree:\n"
        "/kaggle/input/datasets/nikmarco/qwythos-cycle4-math500-eval-code/math_eval.py\n"
        "/kaggle/input/datasets/nikmarco/qwythos-cycle4-math500-eval-code/answer_extract.py"
    ),
    traceback_text=(
        "File \"/kaggle/src/script.py\", line 59, in locate_dir\n"
        "  raise FileNotFoundError(f\"{name}/ not found under {INPUT}. Tree:\\n\" + ...)\n"
        "FileNotFoundError: code/ not found under /kaggle/input."
    ),
    environment=_KAGGLE_T4_ENV,
    attempt_number=1,
)


# 4. DEPENDENCY_INCOMPATIBLE -- expected: recognize a version-gap on a new
#    model architecture and consult evidence of a *known-working* pin
#    (another executor in the same project already loaded this exact model)
#    before trying arbitrary versions.
QWEN3_5_ARCH_NOT_RECOGNIZED = FailureCapture(
    incident_id="2026-08-30-math500-baseline-arch-not-recognized",
    experiment_id="cycle4-math500-baseline-v1",
    executor_name="kaggle-run-math500-baseline",
    occurred_at="2026-08-30T02:xx:00Z",
    exception_type="ValueError",
    exception_message=(
        "The checkpoint you are trying to load has model type `qwen3_5` but "
        "Transformers does not recognize this architecture. This could be because "
        "of an issue with the checkpoint, or because your version of Transformers "
        "is out of date."
    ),
    traceback_text=(
        "File \"transformers/models/auto/configuration_auto.py\", line 1384, in from_pretrained\n"
        "  config_class = CONFIG_MAPPING[config_dict[\"model_type\"]]\n"
        "KeyError: 'qwen3_5'\n"
        "...\n"
        "ValueError: The checkpoint you are trying to load has model type `qwen3_5` "
        "but Transformers does not recognize this architecture."
    ),
    environment=EnvironmentSnapshot(
        hardware_summary="1x Tesla T4, sm_75",
        accelerator_count=1,
        installed_packages={},  # stock Kaggle image, no pins applied yet -- that's the bug
    ),
    attempt_number=1,
    gpu_hours_spent=0.02,
)


# 5. CUDA_KERNEL_UNAVAILABLE -- expected: recognize as a backend/kernel
#    availability gap (not a numerical or data problem) and route around the
#    specific op path (e.g. disable the accelerated backend for that op),
#    not touch training data or hyperparameters.
QWEN3_5_CONV1D_NO_ENGINE = FailureCapture(
    incident_id="2026-08-31-cycle4-train-v1-cudnn-no-engine",
    experiment_id="dpo-cycle4-math-v1",
    executor_name="kaggle-train-dpo-trl",
    occurred_at="2026-08-31T00:xx:00Z",
    exception_type="RuntimeError",
    exception_message="GET was unable to find an engine to execute this computation",
    traceback_text=(
        "File \"transformers/models/qwen3_5/modeling_qwen3_5.py\", line 474, in forward\n"
        "  mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])\n"
        "File \"torch/nn/modules/conv.py\", line 370, in _conv_forward\n"
        "  return F.conv1d(...)\n"
        "RuntimeError: GET was unable to find an engine to execute this computation"
    ),
    environment=_KAGGLE_T4_ENV,
    attempt_number=1,
    gpu_hours_spent=0.15,
)


# 6. CUDA_EXECUTION_FAILED -- expected: recognize as distinct from
#    CUDA_KERNEL_UNAVAILABLE even though both are CUDA RuntimeErrors in the
#    same custom op family; the remediation that fixed #5 (disabling cuDNN)
#    did NOT fix this one -- a real test of whether fingerprinting correctly
#    keeps these two incidents separate rather than collapsing them.
GATED_DELTA_RULE_CUBLAS_FAILURE = FailureCapture(
    incident_id="2026-08-31-cycle4-train-v2-cublas-execution-failed",
    experiment_id="dpo-cycle4-math-v1",
    executor_name="kaggle-train-dpo-trl",
    occurred_at="2026-08-31T01:xx:00Z",
    exception_type="RuntimeError",
    exception_message=(
        "CUDA error: CUBLAS_STATUS_EXECUTION_FAILED when calling "
        "`cublasGemmStridedBatchedEx(handle, opa, opb, m, n, k, alpha_ptr, a, "
        "CUDA_R_16F, lda, stridea, b, CUDA_R_16F, ldb, strideb, beta_ptr, c, ...)`"
    ),
    traceback_text=(
        "File \"transformers/models/qwen3_5/modeling_qwen3_5.py\", line 303, in "
        "torch_chunk_gated_delta_rule\n"
        "  + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new\n"
        "RuntimeError: CUDA error: CUBLAS_STATUS_EXECUTION_FAILED"
    ),
    environment=_KAGGLE_T4_ENV,
    attempt_number=2,
    gpu_hours_spent=0.02,
)


# 7. HARDWARE_INCOMPATIBLE -- expected: recognize the *root* problem is the
#    wrong accelerator was provisioned (a P100, sm_60, below this PyTorch
#    build's sm_70 floor), not attempt a software fix for what is actually a
#    provisioning mistake -- discover the correct machine_shape value rather
#    than retrying with different library versions.
WRONG_ACCELERATOR_PROVISIONED = FailureCapture(
    incident_id="2026-09-01-cycle4-train-v6-wrong-gpu",
    experiment_id="dpo-cycle4-math-v1",
    executor_name="kaggle-train-dpo-trl",
    occurred_at="2026-09-01T14:xx:00Z",
    exception_type="RuntimeError",
    exception_message="Error named symbol not found at line 62 in file /src/csrc/ops.cu",
    traceback_text=(
        "UserWarning: \n"
        "    Found GPU0 Tesla P100-PCIE-16GB which is of cuda capability 6.0.\n"
        "    Minimum and Maximum cuda capability supported by this version of "
        "PyTorch is (7.0) - (12.0)\n"
        "UserWarning: Tesla P100-PCIE-16GB with CUDA capability sm_60 is not "
        "compatible with the current PyTorch installation.\n"
        "...\n"
        "RuntimeError: Error named symbol not found at line 62 in file /src/csrc/ops.cu"
    ),
    environment=EnvironmentSnapshot(
        hardware_summary="1x Tesla P100-PCIE-16GB, sm_60 (unintended -- machine_shape "
        "string requested T4x2, provider allocated P100)",
        accelerator_count=1,
        installed_packages=_KAGGLE_T4_ENV_5_10_2.installed_packages,
        config_patch={"kernel_metadata.machine_shape": "NvidiaTeslaT4x2"},
    ),
    attempt_number=1,
    gpu_hours_spent=0.08,
)


# 8. CONFIG_INVALID -- expected: recognize this as a config/argument defect
#    introduced by the *previous* remediation (a code change was made but not
#    propagated to where the executor actually reads it), not a new class of
#    infrastructure failure -- and check "did my last fix actually ship
#    everywhere it needed to" before generating new hypotheses.
STALE_EXECUTOR_ARTIFACT_MISSING_FLAG = FailureCapture(
    incident_id="2026-09-01-cycle4-train-v5-device-map-not-recognized",
    experiment_id="dpo-cycle4-math-v1",
    executor_name="kaggle-train-dpo-trl",
    occurred_at="2026-09-01T14:xx:30Z",
    exception_type="SystemExit",
    exception_message="train_dpo_trl.py: error: unrecognized arguments: --device-map auto",
    traceback_text=(
        "usage: train_dpo_trl.py [-h] --pairs PAIRS --out OUT [--base BASE] ...\n"
        "train_dpo_trl.py: error: unrecognized arguments: --device-map auto"
    ),
    environment=_KAGGLE_T4_ENV_5_10_2,
    attempt_number=1,
    gpu_hours_spent=0.01,
)


# 9. CUDA_DEVICE_MISMATCH -- expected: recognize as a placement-contract gap
#    between a raw device_map="auto" load and a trainer that assumes a
#    single accelerator.device for its inputs (a known rough edge, not a
#    hardware or data fault) -- and that the *previous* incident (#7) is
#    what proves multi-GPU is genuinely available here, so this is a new,
#    unrelated failure class, not a recurrence.
DPO_TRAINER_DEVICE_MAP_AUTO_MISMATCH = FailureCapture(
    incident_id="2026-09-01-cycle4-train-v7-device-mismatch",
    experiment_id="dpo-cycle4-math-v1",
    executor_name="kaggle-train-dpo-trl",
    occurred_at="2026-09-01T15:xx:00Z",
    exception_type="RuntimeError",
    exception_message=(
        "Expected all tensors to be on the same device, but got index is on cuda:0, "
        "different from other tensors on cuda:1 (when checking argument in method "
        "wrapper_CUDA__index_select)"
    ),
    traceback_text=(
        "File \"transformers/models/qwen3_5/modeling_qwen3_5.py\", line 1170, in forward\n"
        "  inputs_embeds = self.embed_tokens(input_ids)\n"
        "File \"torch/nn/modules/sparse.py\", line 191, in forward\n"
        "  return F.embedding(...)\n"
        "RuntimeError: Expected all tensors to be on the same device, but got index "
        "is on cuda:0, different from other tensors on cuda:1"
    ),
    environment=EnvironmentSnapshot(
        hardware_summary="2x Tesla T4, sm_75",
        accelerator_count=2,
        installed_packages=_KAGGLE_T4_ENV_5_10_2.installed_packages,
        config_patch={"device_map": "auto"},
    ),
    attempt_number=1,
    gpu_hours_spent=0.02,
)


# 10. CUDA_OOM (recurrence, different phase) -- expected: same
#     signature_kind as #2, but a *different* incident: different executor
#     run, different training phase (this one is in the trainer's fp32-logits
#     upcast, not PEFT's kbit-prep pass), different hardware (this run had
#     access to 2x T4 by this point, even though it wasn't using the second
#     one yet -- --device-map auto came one attempt later). A correct
#     fingerprint keeps this distinct from #2 while both still resolve to
#     signature_kind=CUDA_OOM.
DPO_LOGITS_FP32_UPCAST_OOM = FailureCapture(
    incident_id="2026-09-01-cycle4-train-v4-fp32-upcast-oom",
    experiment_id="dpo-cycle4-math-v1",
    executor_name="kaggle-train-dpo-trl",
    occurred_at="2026-09-01T13:xx:00Z",
    exception_type="torch.OutOfMemoryError",
    exception_message=(
        "CUDA out of memory. Tried to allocate 5.50 GiB. GPU 0 has a total capacity "
        "of 14.56 GiB of which 2.52 GiB is free. Including non-PyTorch memory, this "
        "process has 12.04 GiB memory in use."
    ),
    traceback_text=(
        "File \"accelerate/utils/operations.py\", line 782, in _convert_to_fp32\n"
        "  return tensor.float()\n"
        "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 5.50 GiB."
    ),
    environment=EnvironmentSnapshot(
        hardware_summary="2x Tesla T4 available, sm_75 (device_map={'': 0} in use -- "
        "second GPU present but unused)",
        accelerator_count=2,
        installed_packages=_KAGGLE_T4_ENV_5_10_2.installed_packages,
        config_patch={"max_length": 2048, "bf16": True, "device_map": "0"},
    ),
    attempt_number=4,
    gpu_hours_spent=0.03,
)


ALL_DEV_FIXTURES: tuple[FailureCapture, ...] = (
    HF_HUB_TRANSIENT_DROP,
    PEFT_KBIT_PREP_OOM,
    KAGGLE_DATASET_FLATTENED_PATH,
    QWEN3_5_ARCH_NOT_RECOGNIZED,
    QWEN3_5_CONV1D_NO_ENGINE,
    GATED_DELTA_RULE_CUBLAS_FAILURE,
    WRONG_ACCELERATOR_PROVISIONED,
    STALE_EXECUTOR_ARTIFACT_MISSING_FLAG,
    DPO_TRAINER_DEVICE_MAP_AUTO_MISMATCH,
    DPO_LOGITS_FP32_UPCAST_OOM,
)
