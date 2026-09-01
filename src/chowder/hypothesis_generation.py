from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

from .incident import IncidentFingerprint, SignatureKind
from .models import Hypothesis


@dataclass(frozen=True)
class HypothesisCandidate:
    """One hypothesis paired with the concrete change it resolves to, plus
    a pre-run cost estimate.

    The hypothesis/config_patch split mirrors the one
    `Investigation.add_hypothesis` already enforces (`Hypothesis.intervention`
    is free text; `config_patch` is what actually gets tried).
    `estimated_gpu_hours` exists purely so `ranking.py` has something to
    tiebreak equally-corroborated candidates on -- it is the generator's own
    guess, not a measured cost.
    """

    hypothesis: Hypothesis
    config_patch: Mapping[str, Any]
    estimated_gpu_hours: float = 0.0


@runtime_checkable
class HypothesisGenerator(Protocol):
    """Chowder does not generate hypotheses itself -- this is a pluggable
    boundary, mirroring `TrainingExecutor`/`DiagnosticProbe`. A real
    LLM-backed or heuristic-heavy generator is out of scope for this plan;
    what matters here is that the investigation machinery around it works
    regardless of which generator is plugged in.
    """

    def generate(self, fingerprint: IncidentFingerprint) -> tuple[HypothesisCandidate, ...]:
        ...


# Rule-based candidate table, keyed by the coarse signature_kind. Covers the
# 9 signature kinds the 10 real dev fixtures classify into (CUDA_OOM appears
# twice); UNKNOWN and the two hidden-set-only kinds Task 7 introduces
# deliberately have no entry -- the generator must fail safe rather than
# fabricate a plausible-looking fix for a class it hasn't been taught.
#
# CUDA_OOM carries two ordered candidates because a single fix does not
# generalize across this project's own two real OOM incidents (kbit-prep
# fragmentation vs. a genuinely larger working set at long sequence length)
# -- the same lesson the conv1d-vs-cuBLAS incidents taught for
# CUDA_KERNEL_UNAVAILABLE vs. CUDA_EXECUTION_FAILED, replicated here inside
# a single signature_kind instead of across two.
#
# CUDA_DEVICE_MISMATCH carries a candidate but not a confirmed fix: this
# exact incident (DPO_TRAINER_DEVICE_MAP_AUTO_MISMATCH) was still unresolved
# in the real training session this benchmark is built from. The candidate
# here is a genuine next thing to try, not a known-working answer -- the
# dev fixture's ground truth (tests/test_dev_fixture_run.py) marks it
# DID_NOT_RESOLVE accordingly, so this stays honest rather than inventing a
# fix that was never actually confirmed.
_CANDIDATES_BY_SIGNATURE: Mapping[SignatureKind, tuple[HypothesisCandidate, ...]] = {
    SignatureKind.CUDA_KERNEL_UNAVAILABLE: (
        HypothesisCandidate(
            hypothesis=Hypothesis(
                observation=(
                    "a CUDA op raised 'unable to find an engine to execute this "
                    "computation' rather than an OOM or a numerical error"
                ),
                suspected_cause=(
                    "the accelerated cuDNN backend has no engine for this "
                    "op/shape/dtype combination on this hardware"
                ),
                intervention="disable cuDNN so the op falls back to a supported kernel path",
            ),
            config_patch={"cudnn_enabled": False},
            estimated_gpu_hours=0.02,
        ),
    ),
    SignatureKind.CUDA_EXECUTION_FAILED: (
        HypothesisCandidate(
            hypothesis=Hypothesis(
                observation=(
                    "a custom CUDA op raises CUBLAS_STATUS_EXECUTION_FAILED, distinct "
                    "from the 'no engine' kernel-unavailable message even though both "
                    "are RuntimeErrors in the same custom-op family"
                ),
                suspected_cause=(
                    "a library/runtime version gap in the custom op's cuBLAS call "
                    "path, not an attention-implementation flag the op bypasses anyway"
                ),
                intervention="bump the library pin known to fix this custom op family",
            ),
            config_patch={"transformers_version": "5.10.2"},
            estimated_gpu_hours=0.02,
        ),
    ),
    SignatureKind.CUDA_DEVICE_MISMATCH: (
        HypothesisCandidate(
            hypothesis=Hypothesis(
                observation=(
                    "a device_map='auto' load spans multiple GPUs, then a manual "
                    "forward call inside the trainer raises a same-device mismatch"
                ),
                suspected_cause=(
                    "a known rough edge between raw multi-GPU sharding and a "
                    "trainer that assumes a single accelerator device for its own "
                    "inputs"
                ),
                intervention="try an alternative device_map placement strategy instead of 'auto'",
            ),
            config_patch={"device_map": "balanced_low_0"},
            estimated_gpu_hours=0.05,
        ),
    ),
    SignatureKind.HARDWARE_INCOMPATIBLE: (
        HypothesisCandidate(
            hypothesis=Hypothesis(
                observation=(
                    "the provisioned accelerator reports a compute capability "
                    "below this PyTorch build's floor"
                ),
                suspected_cause=(
                    "the requested machine_shape string does not provision the "
                    "accelerator its name implies"
                ),
                intervention=(
                    "use the machine_shape string actually known to provision "
                    "the intended accelerator"
                ),
            ),
            config_patch={"kernel_metadata.machine_shape": "NvidiaTeslaT4"},
            estimated_gpu_hours=0.01,
        ),
    ),
    SignatureKind.NETWORK_TRANSIENT: (
        HypothesisCandidate(
            hypothesis=Hypothesis(
                observation="a large download's connection was dropped mid-transfer",
                suspected_cause="a transient network interruption, not a corrupted or missing source file",
                intervention=(
                    "retry the download so the HTTP cache resumes the broken "
                    "shard by range instead of restarting it"
                ),
            ),
            config_patch={"resume_download": True},
            estimated_gpu_hours=0.0,
        ),
    ),
    SignatureKind.DEPENDENCY_INCOMPATIBLE: (
        HypothesisCandidate(
            hypothesis=Hypothesis(
                observation="the installed library does not recognize this checkpoint's model_type",
                suspected_cause="the installed library version predates support for this architecture",
                intervention="bump the library to a version pin known to support this architecture",
            ),
            config_patch={"transformers_version": "5.10.2"},
            estimated_gpu_hours=0.02,
        ),
    ),
    SignatureKind.CONFIG_INVALID: (
        HypothesisCandidate(
            hypothesis=Hypothesis(
                observation="the executor rejects a command-line flag as unrecognized",
                suspected_cause=(
                    "a prior code change added the flag locally but the artifact "
                    "the executor actually runs was never re-synced"
                ),
                intervention="re-sync the updated script to wherever the executor reads it from before retrying",
            ),
            config_patch={"resync_kernel_dataset": True},
            estimated_gpu_hours=0.0,
        ),
    ),
    SignatureKind.ARTIFACT_NOT_FOUND: (
        HypothesisCandidate(
            hypothesis=Hypothesis(
                observation=(
                    "an expected file is missing from where the executor "
                    "assumed it would be, even though it was uploaded"
                ),
                suspected_cause=(
                    "packaging flattened the uploaded directory structure, so "
                    "the assumed path no longer matches"
                ),
                intervention="locate the required file by filename instead of assuming a fixed directory layout",
            ),
            config_patch={"dataset_path_resolution": "search_by_filename"},
            estimated_gpu_hours=0.0,
        ),
    ),
    SignatureKind.CUDA_OOM: (
        HypothesisCandidate(
            hypothesis=Hypothesis(
                observation=(
                    "training raised a CUDA OutOfMemoryError with GPU memory "
                    "already mostly in use before the allocation that failed"
                ),
                suspected_cause="allocator fragmentation from repeated small tensor churn, not raw insufficient capacity",
                intervention="enable the expandable-segments CUDA allocator to reduce fragmentation",
            ),
            config_patch={"allocator_conf": "expandable_segments:True"},
            estimated_gpu_hours=0.02,
        ),
        HypothesisCandidate(
            hypothesis=Hypothesis(
                observation=(
                    "training raised a CUDA OutOfMemoryError while upcasting "
                    "or preparing full-vocabulary logits at long sequence length"
                ),
                suspected_cause="the working set at this sequence length exceeds available memory outright, not merely fragmented",
                intervention="trim the maximum sequence length so the same batch fits in memory",
            ),
            config_patch={"max_length": 1024},
            estimated_gpu_hours=0.05,
        ),
    ),
}


@dataclass(frozen=True)
class RuleBasedGenerator:
    """Deterministic, table-driven hypothesis generator covering all 10 real
    dev-set incidents' signature kinds (Task 6 extends Task 5's single-entry
    version).

    Deliberately thin -- its job is making the investigation machinery
    runnable against real incident classes, not claiming agent-like
    reasoning. An unmodeled signature_kind (UNKNOWN, or a hidden-set-only
    kind Task 7 introduces) returns no candidates rather than fabricating
    one.
    """

    def generate(self, fingerprint: IncidentFingerprint) -> tuple[HypothesisCandidate, ...]:
        return _CANDIDATES_BY_SIGNATURE.get(fingerprint.signature_kind, ())
