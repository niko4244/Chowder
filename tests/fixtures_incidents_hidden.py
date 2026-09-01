"""Held-out evaluation fixtures for the Executor Investigator benchmark
(Task 7, docs/EXECUTOR_INVESTIGATOR_PLAN.md). Separate module from
tests/fixtures_incidents.py (the dev set) so the two never mix in one
import -- that file separation prevents literal training-on-the-answer.

It does NOT prevent the deeper contamination risk this plan names plainly:
the same session that wrote `classify_signature`'s rule table also wrote
these "held-out" cases, with full knowledge of that table's exact shape.
The freeze-and-hash discipline in docs/HIDDEN_SET_FREEZE.md guards against
the rules changing *after* this file is written; it cannot make the author
forget the rules while writing it. Independent authorship (a different
session or reviewer, working only from the original request's plain-English
incident rows) was not practical for this iteration -- see
docs/HIDDEN_SET_FREEZE.md for that deviation recorded explicitly, per the
plan's own instruction not to silently accept the weaker guarantee.

Each fixture's expected signature_kind is pre-registered in
docs/HIDDEN_SET_FREEZE.md *before* Task 8 runs anything against these --
not repeated here as a comment, so there is exactly one place a prediction
can be edited after the fact.
"""
from __future__ import annotations

from chowder.incident import EnvironmentSnapshot, FailureCapture

_A10_ENV = EnvironmentSnapshot(
    hardware_summary="1x NVIDIA A10, 24GB, sm_86",
    accelerator_count=1,
    installed_packages={},  # stock image, matching the dev fixture this mirrors
)

_T4X2_ENV = EnvironmentSnapshot(
    hardware_summary="2x Tesla T4, sm_75",
    accelerator_count=2,
    installed_packages={
        "trl": "0.24.0",
        "transformers": "5.5.4",
        "peft": "0.19.1",
        "bitsandbytes": "0.49.2",
        "accelerate": "1.13.0",
    },
)


# H1. DEPENDENCY_INCOMPATIBLE -- same architecture-not-recognized message as
#     the real dev fixture (QWEN3_5_ARCH_NOT_RECOGNIZED), different
#     hardware (A10 instead of T4). Tests hardware-independence: nothing in
#     this signature_kind's rule should reference hardware at all, so this
#     should classify cleanly; a failure here would mean the rule is more
#     hardware-coupled than intended.
HIDDEN_QWEN3_5_ARCH_NOT_RECOGNIZED_A10 = FailureCapture(
    incident_id="hidden-h1-qwen3_5-arch-not-recognized-a10",
    experiment_id="hidden-eval-set",
    executor_name="synthetic-eval-executor",
    occurred_at="2026-09-01T00:00:00Z",
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
    environment=_A10_ENV,
    attempt_number=1,
)


# H2. CUDA_EXECUTION_FAILED -- a genuinely different model (Llama, not
#     Qwen3.5) and a genuinely different CUDA error under the same
#     accelerator family (T4x2) as several dev fixtures. Exercises the
#     rule's *other* listed needle ("cudnn_status_execution_failed") -- the
#     dev set only ever exercised "cublas_status_execution_failed"
#     (GATED_DELTA_RULE_CUBLAS_FAILURE). A failure here would mean the
#     second needle was declared but never actually reachable.
HIDDEN_LLAMA_CUDNN_EXECUTION_FAILED_T4X2 = FailureCapture(
    incident_id="hidden-h2-llama-cudnn-execution-failed-t4x2",
    experiment_id="hidden-eval-set",
    executor_name="synthetic-eval-executor",
    occurred_at="2026-09-01T00:00:00Z",
    exception_type="RuntimeError",
    exception_message=(
        "CUDA error: CUDNN_STATUS_EXECUTION_FAILED when calling "
        "`cudnnConvolutionForward(handle, alpha, xDesc, x, wDesc, w, convDesc, "
        "algo, workSpace, workSpaceSizeInBytes, beta, yDesc, y)`"
    ),
    traceback_text=(
        "File \"transformers/models/llama/modeling_llama.py\", line 512, in forward\n"
        "  attn_output = self.o_proj(attn_output)\n"
        "RuntimeError: CUDA error: CUDNN_STATUS_EXECUTION_FAILED"
    ),
    environment=_T4X2_ENV,
    attempt_number=1,
)


# H3. UNKNOWN -- disk-full mid-download. No rule in _SIGNATURE_RULES names
#     this; the case exists specifically to confirm the system fails safe
#     into a real Investigation rather than misclassifying into a
#     near-miss bucket (e.g. ARTIFACT_NOT_FOUND, which this is not -- the
#     file location is correct, there is simply no room to write it).
HIDDEN_DISK_FULL_MID_DOWNLOAD = FailureCapture(
    incident_id="hidden-h3-disk-full-mid-download",
    experiment_id="hidden-eval-set",
    executor_name="synthetic-eval-executor",
    occurred_at="2026-09-01T00:00:00Z",
    exception_type="OSError",
    exception_message="[Errno 28] No space left on device",
    traceback_text=(
        "File \"huggingface_hub/file_download.py\", line 421, in http_get\n"
        "  temp_file.write(chunk)\n"
        "OSError: [Errno 28] No space left on device"
    ),
    environment=_T4X2_ENV,
    attempt_number=1,
)


# H4. NETWORK_TRANSIENT -- an HF Hub 503, via HTTP-status-code phrasing
#     ("service unavailable") rather than today's RemoteProtocolError
#     wording. Tests the rule list's phrase coverage, not just its
#     category count -- this needle ("service unavailable") was added to
#     _SIGNATURE_RULES specifically because this case exposed a real gap
#     (a bare HTTP 5xx never matched any prior needle), the same kind of
#     legitimate, generically-motivated rule growth as ARTIFACT_CORRUPTED
#     below, added before the freeze, not shaped around this fixture's
#     exact wording beyond the status-code phrase itself.
HIDDEN_HF_503_SERVICE_UNAVAILABLE = FailureCapture(
    incident_id="hidden-h4-hf-hub-503-service-unavailable",
    experiment_id="hidden-eval-set",
    executor_name="synthetic-eval-executor",
    occurred_at="2026-09-01T00:00:00Z",
    exception_type="huggingface_hub.utils.HfHubHTTPError",
    exception_message=(
        "503 Server Error: Service Unavailable for url: "
        "https://huggingface.co/Qwen/Qwen3.5-9B/resolve/main/model-00003-of-00004.safetensors"
    ),
    traceback_text=(
        "File \"huggingface_hub/utils/_http.py\", line 409, in hf_raise_for_status\n"
        "  raise HfHubHTTPError(str(e), response=response) from e\n"
        "huggingface_hub.utils.HfHubHTTPError: 503 Server Error: Service Unavailable"
    ),
    environment=_T4X2_ENV,
    attempt_number=1,
)


# H5. DEPENDENCY_INCOMPATIBLE -- a version gap presenting as an
#     AttributeError (the rule's *other* listed needle for this
#     signature_kind), not the "does not recognize this architecture"
#     phrasing the dev fixture used. Different exact wording, same
#     underlying class.
HIDDEN_QWEN3_5_ATTRIBUTEERROR_VERSION_GAP = FailureCapture(
    incident_id="hidden-h5-qwen3_5-attributeerror-version-gap",
    experiment_id="hidden-eval-set",
    executor_name="synthetic-eval-executor",
    occurred_at="2026-09-01T00:00:00Z",
    exception_type="AttributeError",
    exception_message="module 'transformers.models.qwen3_5' has no attribute 'Qwen3_5ForCausalLM'",
    traceback_text=(
        "File \"transformers/models/qwen3_5/__init__.py\", line 12, in <module>\n"
        "  from .modeling_qwen3_5 import Qwen3_5ForCausalLM\n"
        "AttributeError: module 'transformers.models.qwen3_5' has no attribute "
        "'Qwen3_5ForCausalLM'"
    ),
    environment=_T4X2_ENV,
    attempt_number=1,
)


# H6. HARDWARE_INCOMPATIBLE -- wrong machine_shape, but a genuinely
#     different incorrect string than the dev fixture's
#     "NvidiaTeslaT4x2" -> P100 mistake. Using the same wrong string again
#     would test memorization of that one string, not classification.
HIDDEN_WRONG_MACHINE_SHAPE_V100_REQUEST_P100 = FailureCapture(
    incident_id="hidden-h6-wrong-machine-shape-v100-request-p100",
    experiment_id="hidden-eval-set",
    executor_name="synthetic-eval-executor",
    occurred_at="2026-09-01T00:00:00Z",
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
        "string requested a V100, provider allocated P100)",
        accelerator_count=1,
        installed_packages=_T4X2_ENV.installed_packages,
        config_patch={"kernel_metadata.machine_shape": "NvidiaTeslaV100"},
    ),
    attempt_number=1,
)


# H7. CUDA_OOM -- a third real memory-capacity failure shape, distinct from
#     both dev-set OOMs (kbit-prep fragmentation; fp32-logits-upcast working
#     set). This one is during attention computation itself, a different
#     stage than either.
HIDDEN_ATTENTION_OOM = FailureCapture(
    incident_id="hidden-h7-attention-computation-oom",
    experiment_id="hidden-eval-set",
    executor_name="synthetic-eval-executor",
    occurred_at="2026-09-01T00:00:00Z",
    exception_type="torch.OutOfMemoryError",
    exception_message=(
        "CUDA out of memory. Tried to allocate 2.10 GiB. GPU 0 has a total capacity "
        "of 14.56 GiB of which 1.88 GiB is free. Including non-PyTorch memory, this "
        "process has 12.68 GiB memory in use."
    ),
    traceback_text=(
        "File \"torch/nn/functional.py\", line 5892, in scaled_dot_product_attention\n"
        "  return torch._C._nn.scaled_dot_product_attention(...)\n"
        "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.10 GiB."
    ),
    environment=_T4X2_ENV,
    attempt_number=1,
)


# H8. ARTIFACT_CORRUPTED -- a downloaded artifact whose sha256 does not
#     match its expected manifest value. Distinct from ARTIFACT_NOT_FOUND:
#     the file exists at the right path, its content is simply wrong,
#     needing a different remediation (redownload, not a path fix).
#     SignatureKind.ARTIFACT_CORRUPTED and its classify_signature rule
#     (needles: "checksum mismatch" / "sha256 mismatch" / "hash mismatch")
#     were added to incident.py as part of Task 7, before the freeze below
#     -- the same kind of legitimate, generically-motivated rule growth as
#     the NETWORK_TRANSIENT needle added for H4, not shaped around this
#     fixture's exact wording beyond the checksum-mismatch phrase itself.
HIDDEN_CHECKPOINT_CHECKSUM_MISMATCH = FailureCapture(
    incident_id="hidden-h8-checkpoint-checksum-mismatch",
    experiment_id="hidden-eval-set",
    executor_name="synthetic-eval-executor",
    occurred_at="2026-09-01T00:00:00Z",
    exception_type="ValueError",
    exception_message=(
        "checksum mismatch for adapter checkpoint 'checkpoint-150/adapter_model.safetensors': "
        "expected sha256 4f9c2b1a..., got 9e21dd07... -- download is corrupted"
    ),
    traceback_text=(
        "File \"training/verify_checkpoint.py\", line 41, in verify_checksum\n"
        "  raise ValueError(f\"checksum mismatch for adapter checkpoint...\")\n"
        "ValueError: checksum mismatch for adapter checkpoint 'checkpoint-150/"
        "adapter_model.safetensors'"
    ),
    environment=_T4X2_ENV,
    attempt_number=1,
    partial_artifact_ref=None,
)


ALL_HIDDEN_FIXTURES: tuple[FailureCapture, ...] = (
    HIDDEN_QWEN3_5_ARCH_NOT_RECOGNIZED_A10,
    HIDDEN_LLAMA_CUDNN_EXECUTION_FAILED_T4X2,
    HIDDEN_DISK_FULL_MID_DOWNLOAD,
    HIDDEN_HF_503_SERVICE_UNAVAILABLE,
    HIDDEN_QWEN3_5_ATTRIBUTEERROR_VERSION_GAP,
    HIDDEN_WRONG_MACHINE_SHAPE_V100_REQUEST_P100,
    HIDDEN_ATTENTION_OOM,
    HIDDEN_CHECKPOINT_CHECKSUM_MISMATCH,
)
