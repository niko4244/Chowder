from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

from .incident import IncidentFingerprint, SignatureKind
from .models import Hypothesis


@dataclass(frozen=True)
class HypothesisCandidate:
    """One hypothesis paired with the concrete change it resolves to --
    mirrors the split `Investigation.add_hypothesis` already enforces
    (`Hypothesis.intervention` is free text; `config_patch` is what
    actually gets tried).
    """

    hypothesis: Hypothesis
    config_patch: Mapping[str, Any]


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


_MINIMAL_CANDIDATES: Mapping[SignatureKind, tuple[HypothesisCandidate, ...]] = {
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
        ),
    ),
}


@dataclass(frozen=True)
class MinimalRuleBasedGenerator:
    """Task 5's walking-skeleton generator: exactly one hardcoded candidate,
    for exactly one signature_kind (CUDA_KERNEL_UNAVAILABLE), matching the
    real fix for `QWEN3_5_CONV1D_NO_ENGINE`. Deliberately thin -- its job
    here is proving the investigation machinery composes end to end, not
    claiming real diagnostic coverage. Task 6 extends this to the full
    signature_kind-keyed table across all 10 dev fixtures; an unmodeled
    signature_kind returns no candidates rather than fabricating one.
    """

    def generate(self, fingerprint: IncidentFingerprint) -> tuple[HypothesisCandidate, ...]:
        return _MINIMAL_CANDIDATES.get(fingerprint.signature_kind, ())
