"""Teacher Fabric, Slice A: provider-neutral teacher signal schemas.

A teacher is a source of training signal, not necessarily a locally
executable model. This module is the first slice of that architecture: the
data contracts (signal taxonomy, capability declaration, request, signal,
durable artifact), the provider protocol, capability negotiation over a
registry, and the tokenizer-compatibility gate. It contains **no network
code, no storage, and no real provider** -- `FakeTeacherProvider` is an
explicit, deterministic test double, not a stub pretending to be a remote
teacher, and nothing here performs or enables I/O.

Deliberately not in this slice (each is a later slice with its own tests):
content-addressed signal storage and dedup (Slice B), black-box teacher
operations wired into the Regression Surgeon (Slice C), cost accounting /
deterministic query escalation (Slice D), the selected-token scorer and
objective-level tests (Slice E), real remote providers (Slice F), remote
job manifests (Slice G), remote distillation microjobs (Slice H),
multi-teacher balancing (Slice I), teacher-selection research (Slice J).
See docs/TEACHER_FABRIC.md for the full design, the threat model, and the
open research questions.

Honesty rules this module enforces mechanically
-----------------------------------------------
1. **Capabilities are declared, never inferred.** A provider says what it
   can do via `TeacherCapabilities`; nothing in this module derives a
   capability from a provider's name, type, or reputation. The one
   deliberate gap: the signal taxonomy includes `full_logits` (research-only
   mode), but the capability vocabulary has no corresponding flag -- so
   `supports(SignalKind.FULL_LOGITS)` is False for every capability block
   and Slice A cannot negotiate it. Inventing an implicit mapping (say,
   from `topk_logprobs`) would be exactly the inference the brief forbids;
   a later slice that implements full-logits mode must extend the
   capability vocabulary explicitly.
2. **Token-level alignment fails closed.** The token-aligned signal kinds
   (`sampled_token_logprobs`, `topk_logprobs`, `full_logits`,
   `hidden_projection`) are meaningful only when teacher and student
   tokenize identically. If either side's tokenizer identity is unknown,
   or the two identity hashes differ, the request is refused outright
   (`TokenizerIncompatibilityError`) -- never approximated, never
   auto-downgraded. Downgrading to text/ranking/reward/critique
   supervision is available only as an explicit caller choice
   (`downgrade_request` with a named destination kind).
3. **Costs are provider-reported evidence, not estimates dressed as
   facts.** `TeacherSignal.monetary_cost_usd` and `token_counts` carry
   what the provider *reported*; a provider that reports nothing leaves
   them at their empty/None defaults, and those stay `None`/empty in the
   artifact rather than being imputed. GPU cost reuses the existing
   `ResourceUsage` primitive so a GPU-backed teacher (rented inference)
   accounts through the same `accelerator_seconds` ledger everything else
   uses.
4. **Artifacts are self-contained and hashable.** A
   `TeacherSignalArtifact` embeds the request and the signal it answers,
   carries every provenance field the brief requires (teacher stable id,
   provider type, model/revision, tokenizer identity, schema version,
   request/prompt/student-trajectory digests, resolved parameters,
   timestamp, latency, monetary/token/GPU cost, payload content hash,
   parent job-manifest digest, provenance/licensing metadata), and exposes
   a canonical `digest()` over all of it. Two artifacts with the same
   digest are the same evidence; anything that cannot be hashed into the
   artifact must not influence the signal.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Mapping, Protocol, runtime_checkable

from .resources import ResourceUsage

# Version of the teacher-signal schema the artifact records. Bumping it
# changes request digests too (it is part of the canonical payload), so a
# schema change is always visible in every derived digest rather than
# silently mixing formats under one cache key.
SCHEMA_VERSION = 1


class SignalKind(str, Enum):
    """The brief's 10-item signal taxonomy, in its own order.

    `FULL_LOGITS` is a research-only mode and `HIDDEN_PROJECTION` an
    experimental one (see the module docstring's honesty rule 1 and
    docs/TEACHER_FABRIC.md's open questions); neither is commissionable
    through Slice A's negotiation.
    """

    SCALAR_REWARD = "scalar_reward"
    GENERATED_ANSWER = "generated_answer"
    CRITIQUE = "critique"
    REVISED_ANSWER = "revised_answer"
    CANDIDATE_RANKING = "candidate_ranking"
    SAMPLED_TOKEN_LOGPROBS = "sampled_token_logprobs"
    TOPK_LOGPROBS = "topk_logprobs"
    FULL_LOGITS = "full_logits"
    HIDDEN_PROJECTION = "hidden_projection"
    REMOTE_ADAPTER = "remote_adapter"


# Signal kinds that only make sense when teacher and student tokenize
# identically. Negotiation and direct queries refuse these unless both
# sides' tokenizer identities are known hashes and equal.
TOKEN_ALIGNED_SIGNALS = frozenset(
    {
        SignalKind.SAMPLED_TOKEN_LOGPROBS,
        SignalKind.TOPK_LOGPROBS,
        SignalKind.FULL_LOGITS,
        SignalKind.HIDDEN_PROJECTION,
    }
)


class TokenizerIncompatibilityError(ValueError):
    """A token-aligned request was refused because tokenizer compatibility
    is not provable. This is the fail-closed path: downgrade explicitly via
    `downgrade_request`, or fix the identities -- never approximate."""


class NegotiationError(ValueError):
    """No registered provider can serve the request. The message names
    every provider's reason, so the refusal is auditable."""


def _require_non_empty_str(value: object, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"teacher fabric {label} must be a non-empty string")


def _require_finite_non_negative(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"teacher fabric {label} must be a finite non-negative number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"teacher fabric {label} must be finite and non-negative")


def _sha256_hex(value: object, label: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"teacher fabric {label} must be a 64-character sha256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"teacher fabric {label} must be sha256 hex") from exc


def _canonical(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(payload: Any) -> str:
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TeacherCapabilities:
    """What a provider declares it can do.

    Exactly the brief's flag set; every flag is a strict boolean (no
    truthy strings, no None-meaning-maybe). Negotiation reads these via
    `supports()` and never infers beyond them.
    """

    generate: bool
    critique: bool
    revise: bool
    rank: bool
    scalar_reward: bool
    selected_token_logprobs: bool
    topk_logprobs: bool
    hidden_projection: bool
    remote_training: bool

    def __post_init__(self) -> None:
        for flag in (
            "generate",
            "critique",
            "revise",
            "rank",
            "scalar_reward",
            "selected_token_logprobs",
            "topk_logprobs",
            "hidden_projection",
            "remote_training",
        ):
            if not isinstance(getattr(self, flag), bool):
                raise ValueError(f"teacher capability {flag} must be a strict bool")

    _FLAG_BY_SIGNAL: ClassVar[Mapping[SignalKind, str]] = {
        SignalKind.SCALAR_REWARD: "scalar_reward",
        SignalKind.GENERATED_ANSWER: "generate",
        SignalKind.CRITIQUE: "critique",
        SignalKind.REVISED_ANSWER: "revise",
        SignalKind.CANDIDATE_RANKING: "rank",
        SignalKind.SAMPLED_TOKEN_LOGPROBS: "selected_token_logprobs",
        SignalKind.TOPK_LOGPROBS: "topk_logprobs",
        SignalKind.HIDDEN_PROJECTION: "hidden_projection",
        SignalKind.REMOTE_ADAPTER: "remote_training",
        # SignalKind.FULL_LOGITS is deliberately unmapped: the capability
        # vocabulary has no flag for the research-only full-logits mode,
        # and deriving one from another flag would be an inference. See
        # the module docstring's honesty rule 1.
    }

    def supports(self, signal_kind: SignalKind) -> bool:
        """Whether this block declares support for *signal_kind*.

        `FULL_LOGITS` is False for every capability block -- see
        `_FLAG_BY_SIGNAL` for why.
        """
        flag = self._FLAG_BY_SIGNAL.get(signal_kind)
        if flag is None:
            return False
        return bool(getattr(self, flag))


@dataclass(frozen=True)
class TeacherRequest:
    """One ask to one teacher: what signal, over what input.

    `prompt` plus `input_payload` carry the actual content (text prompt,
    candidate solutions, student answer to critique, ...).
    `student_tokenizer_identity` is the *student's* tokenizer identity
    hash -- required for token-aligned signal kinds, optional (and then
    unusable for those kinds) otherwise. `student_trajectory` is the
    student's sampled states/actions in text form, for on-policy scoring;
    its digest lands in the artifact either way.
    """

    teacher_id: str
    signal_kind: SignalKind
    prompt: str
    input_payload: Mapping[str, Any] = field(default_factory=dict)
    parameters: Mapping[str, Any] = field(default_factory=dict)
    student_tokenizer_identity: str | None = None
    student_trajectory: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        _require_non_empty_str(self.teacher_id, "teacher_id")
        if not isinstance(self.signal_kind, SignalKind):
            raise ValueError("teacher request signal_kind must be a SignalKind")
        _require_non_empty_str(self.prompt, "prompt")
        if not isinstance(self.input_payload, Mapping):
            raise ValueError("teacher request input_payload must be a mapping")
        if not isinstance(self.parameters, Mapping):
            raise ValueError("teacher request parameters must be a mapping")
        if self.student_tokenizer_identity is not None:
            _sha256_hex(self.student_tokenizer_identity, "student_tokenizer_identity")
        if self.student_trajectory is not None:
            if not isinstance(self.student_trajectory, tuple) or not all(
                isinstance(step, str) for step in self.student_trajectory
            ):
                raise ValueError(
                    "teacher request student_trajectory must be a tuple of strings"
                )

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "teacher_id": self.teacher_id,
            "signal_kind": self.signal_kind.value,
            "prompt": self.prompt,
            "input_payload": dict(self.input_payload),
            "parameters": dict(self.parameters),
            "student_tokenizer_identity": self.student_tokenizer_identity,
            "student_trajectory": (
                list(self.student_trajectory) if self.student_trajectory is not None else None
            ),
        }

    def canonical_json(self) -> str:
        return _canonical(self.canonical_payload())

    def digest(self) -> str:
        """Stable identity of the full request (the artifact's
        `request_digest`). Includes the schema version, so a schema change
        never collides with an older request's key."""
        return _digest(self.canonical_payload())

    def prompt_digest(self) -> str:
        """Stable identity of prompt + structured inputs (the artifact's
        `prompt_digest`), separate from the full request digest."""
        return _digest(
            {
                "schema_version": SCHEMA_VERSION,
                "prompt": self.prompt,
                "input_payload": dict(self.input_payload),
            }
        )

    def student_trajectory_digest(self) -> str | None:
        if self.student_trajectory is None:
            return None
        return _digest(
            {
                "schema_version": SCHEMA_VERSION,
                "student_trajectory": list(self.student_trajectory),
            }
        )


def downgrade_request(request: TeacherRequest, *, to: SignalKind) -> TeacherRequest:
    """Explicitly rebuild *request* as a non-token-aligned signal kind.

    The only sanctioned response to a `TokenizerIncompatibilityError` when
    the caller decides text/ranking/reward/critique supervision is
    acceptable instead. Deliberately requires a named destination kind
    (no default mapping -- "critique instead of logprobs" is a modelling
    decision, not a mechanical one) and refuses another token-aligned
    kind, which would just move the incompatibility.
    """
    if to in TOKEN_ALIGNED_SIGNALS:
        raise ValueError(
            f"downgrade target {to.value} is itself token-aligned; "
            "downgrading must land on text/ranking/reward/critique supervision"
        )
    return TeacherRequest(
        teacher_id=request.teacher_id,
        signal_kind=to,
        prompt=request.prompt,
        input_payload=dict(request.input_payload),
        parameters=dict(request.parameters),
        student_tokenizer_identity=request.student_tokenizer_identity,
        student_trajectory=request.student_trajectory,
    )


@dataclass(frozen=True)
class TeacherSignal:
    """What a provider returned for one request.

    `payload` is the signal itself; its per-kind schema is a later slice's
    concern (Slice E for logprob kinds, Slice C for black-box kinds) --
    Slice A only requires it to be a mapping so it is canonically
    hashable. `resource_usage` is real measured usage for GPU-backed
    teachers (rented inference reports through the same
    `ResourceUsage`/`accelerator_seconds` primitive as local training) and
    None for pure-API teachers. `monetary_cost_usd` and `token_counts`
    are provider-**reported** numbers, never Chowder's estimates; a
    provider that does not report them leaves them None/empty and that
    absence propagates to the artifact verbatim.
    """

    signal_kind: SignalKind
    payload: Mapping[str, Any]
    resource_usage: ResourceUsage | None = None
    monetary_cost_usd: float | None = None
    token_counts: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.signal_kind, SignalKind):
            raise ValueError("teacher signal signal_kind must be a SignalKind")
        if not isinstance(self.payload, Mapping):
            raise ValueError("teacher signal payload must be a mapping")
        if self.monetary_cost_usd is not None:
            _require_finite_non_negative(self.monetary_cost_usd, "monetary_cost_usd")
        if not isinstance(self.token_counts, Mapping):
            raise ValueError("teacher signal token_counts must be a mapping")
        for name, count in self.token_counts.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("teacher signal token count names must be non-empty strings")
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError(f"teacher signal token count {name!r} must be a non-negative int")

    def payload_digest(self) -> str:
        return _digest({"schema_version": SCHEMA_VERSION, "payload": dict(self.payload)})


@dataclass(frozen=True)
class TeacherSignalArtifact:
    """The durable, fully-provenanced record of one teacher signal.

    Construct via `from_signal`, which fills the digests and cost fields
    from the request/signal pair so there is exactly one source of truth.
    Every field the brief requires per artifact is present; the ones that
    can be genuinely absent (`tokenizer_identity_sha256` for a provider
    that does not report one, `parent_job_manifest_digest` before Slice G,
    provenance/licensing metadata where unknown) are None/empty rather
    than imputed -- the artifact records what is known, which is the
    provenance.
    """

    signal_id: str
    teacher_id: str
    provider_type: str
    model_revision: str
    tokenizer_identity_sha256: str | None
    request: TeacherRequest
    signal: TeacherSignal
    occurred_at: str
    latency_seconds: float
    generation_parameters: Mapping[str, Any]
    monetary_cost_usd: float | None
    token_counts: Mapping[str, int]
    gpu_hours: float | None
    parent_job_manifest_digest: str | None
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        _require_non_empty_str(self.signal_id, "signal_id")
        _require_non_empty_str(self.teacher_id, "teacher_id")
        _require_non_empty_str(self.provider_type, "provider_type")
        _require_non_empty_str(self.model_revision, "model_revision")
        if self.tokenizer_identity_sha256 is not None:
            _sha256_hex(self.tokenizer_identity_sha256, "tokenizer_identity_sha256")
        if not isinstance(self.request, TeacherRequest):
            raise ValueError("teacher signal artifact request must be a TeacherRequest")
        if not isinstance(self.signal, TeacherSignal):
            raise ValueError("teacher signal artifact signal must be a TeacherSignal")
        if self.signal.signal_kind is not self.request.signal_kind:
            raise ValueError(
                "teacher signal artifact answers a different signal kind than it was requested for"
            )
        _require_non_empty_str(self.occurred_at, "occurred_at")
        if isinstance(self.latency_seconds, bool) or not isinstance(self.latency_seconds, (int, float)):
            raise ValueError("teacher signal artifact latency_seconds must be a number")
        if not math.isfinite(float(self.latency_seconds)) or self.latency_seconds < 0:
            raise ValueError("teacher signal artifact latency_seconds must be finite and non-negative")
        if not isinstance(self.generation_parameters, Mapping):
            raise ValueError("teacher signal artifact generation_parameters must be a mapping")
        if self.monetary_cost_usd is not None:
            _require_finite_non_negative(self.monetary_cost_usd, "monetary_cost_usd")
        if not isinstance(self.token_counts, Mapping):
            raise ValueError("teacher signal artifact token_counts must be a mapping")
        if self.gpu_hours is not None:
            _require_finite_non_negative(self.gpu_hours, "gpu_hours")
        if self.parent_job_manifest_digest is not None:
            _sha256_hex(self.parent_job_manifest_digest, "parent_job_manifest_digest")
        if not isinstance(self.provenance, Mapping):
            raise ValueError("teacher signal artifact provenance must be a mapping")
        # Transcription guards: the artifact's copies of the signal's cost
        # evidence must agree with the signal itself (the same
        # single-source-of-truth discipline as TrainingArtifact vs
        # ResourceUsage).
        if self.signal.monetary_cost_usd is None:
            if self.monetary_cost_usd is not None:
                raise ValueError("artifact records a monetary cost the signal did not report")
        elif (
            self.monetary_cost_usd is None
            or not math.isclose(
                float(self.monetary_cost_usd),
                float(self.signal.monetary_cost_usd),
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("artifact monetary cost disagrees with the reported signal")
        usage_hours = self.signal.resource_usage.gpu_hours if self.signal.resource_usage else None
        if (self.gpu_hours is None) != (usage_hours is None) or (
            self.gpu_hours is not None
            and not math.isclose(self.gpu_hours, usage_hours, rel_tol=1e-9, abs_tol=1e-12)
        ):
            raise ValueError("artifact gpu_hours disagrees with the signal's resource usage")

    @classmethod
    def from_signal(
        cls,
        *,
        request: TeacherRequest,
        signal: TeacherSignal,
        signal_id: str,
        provider_type: str,
        model_revision: str,
        tokenizer_identity_sha256: str | None,
        occurred_at: str,
        latency_seconds: float,
        generation_parameters: Mapping[str, Any] | None = None,
        parent_job_manifest_digest: str | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> "TeacherSignalArtifact":
        usage_hours = signal.resource_usage.gpu_hours if signal.resource_usage else None
        return cls(
            signal_id=signal_id,
            teacher_id=request.teacher_id,
            provider_type=provider_type,
            model_revision=model_revision,
            tokenizer_identity_sha256=tokenizer_identity_sha256,
            request=request,
            signal=signal,
            occurred_at=occurred_at,
            latency_seconds=latency_seconds,
            generation_parameters=dict(generation_parameters or {}),
            monetary_cost_usd=signal.monetary_cost_usd,
            token_counts=dict(signal.token_counts),
            gpu_hours=usage_hours,
            parent_job_manifest_digest=parent_job_manifest_digest,
            provenance=dict(provenance or {}),
        )

    @property
    def request_digest(self) -> str:
        return self.request.digest()

    @property
    def prompt_digest(self) -> str:
        return self.request.prompt_digest()

    @property
    def student_trajectory_digest(self) -> str | None:
        return self.request.student_trajectory_digest()

    @property
    def payload_content_sha256(self) -> str:
        return self.signal.payload_digest()

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "signal_id": self.signal_id,
            "teacher_id": self.teacher_id,
            "provider_type": self.provider_type,
            "model_revision": self.model_revision,
            "tokenizer_identity_sha256": self.tokenizer_identity_sha256,
            "request": self.request.canonical_payload(),
            "signal": {
                "signal_kind": self.signal.signal_kind.value,
                "payload": dict(self.signal.payload),
                "monetary_cost_usd": self.signal.monetary_cost_usd,
                "token_counts": dict(self.signal.token_counts),
                "resource_usage": (
                    {
                        "wall_seconds": self.signal.resource_usage.wall_seconds,
                        "accelerator_seconds": self.signal.resource_usage.accelerator_seconds,
                        "active_accelerator_count": self.signal.resource_usage.active_accelerator_count,
                        "visible_accelerator_count": self.signal.resource_usage.visible_accelerator_count,
                    }
                    if self.signal.resource_usage is not None
                    else None
                ),
            },
            "occurred_at": self.occurred_at,
            "latency_seconds": self.latency_seconds,
            "generation_parameters": dict(self.generation_parameters),
            "monetary_cost_usd": self.monetary_cost_usd,
            "token_counts": dict(self.token_counts),
            "gpu_hours": self.gpu_hours,
            "parent_job_manifest_digest": self.parent_job_manifest_digest,
            "provenance": dict(self.provenance),
        }

    def canonical_json(self) -> str:
        return _canonical(self.canonical_payload())

    def digest(self) -> str:
        """Stable identity of the whole artifact. Two artifacts with the
        same digest are the same evidence -- the property Slice B's
        content addressing builds on."""
        return _digest(self.canonical_payload())


@dataclass(frozen=True)
class TeacherCostEstimate:
    """A provider's quoted cost for a request, before it is spent.

    Mirrors `executors.CostEstimate`'s role for the training executors,
    with the teacher cost vocabulary: monetary and token dimensions
    first (API teachers), GPU-hours where the teacher is GPU-backed.
    `gpu_hours=None` means the provider genuinely consumes no local-
    accountable GPU time -- a real answer, not a missing number.
    """

    monetary_cost_usd: float
    estimated_tokens: int
    gpu_hours: float | None = None
    confidence: float = 0.5

    def __post_init__(self) -> None:
        if isinstance(self.monetary_cost_usd, bool) or not isinstance(
            self.monetary_cost_usd, (int, float)
        ):
            raise ValueError("cost estimate monetary_cost_usd must be a finite non-negative number")
        if not math.isfinite(float(self.monetary_cost_usd)) or self.monetary_cost_usd < 0:
            raise ValueError("cost estimate monetary_cost_usd must be finite and non-negative")
        if isinstance(self.estimated_tokens, bool) or not isinstance(self.estimated_tokens, int):
            raise ValueError("cost estimate estimated_tokens must be an int")
        if self.estimated_tokens < 0:
            raise ValueError("cost estimate estimated_tokens must be non-negative")
        if self.gpu_hours is not None:
            _require_finite_non_negative(self.gpu_hours, "cost estimate gpu_hours")
        confidence = float(self.confidence)
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("cost estimate confidence must be finite and in [0, 1]")


@runtime_checkable
class TeacherProvider(Protocol):
    """A source of teacher signals.

    Same shape as `TrainingExecutor` (name / profile / cancel), with
    `run` renamed `query` -- the domain verb -- and `capabilities()` where
    negotiation lives. `tokenizer_identity_sha256` is the provider's own
    tokenizer identity hash; None means "unknown/not reported", which
    makes every token-aligned request fail closed against this provider.
    """

    name: str
    tokenizer_identity_sha256: str | None

    def capabilities(self) -> TeacherCapabilities:
        ...

    def profile(self, request: TeacherRequest) -> TeacherCostEstimate:
        ...

    def query(self, request: TeacherRequest) -> TeacherSignal:
        ...

    def cancel(self, request_id: str) -> None:
        ...


def ensure_tokenizer_compatible(
    provider_tokenizer_identity: str | None, request: TeacherRequest
) -> None:
    """Fail closed for token-aligned signals unless tokenizers provably match.

    A no-op for non-token-aligned kinds. For token-aligned kinds, both
    identities must be known 64-hex sha256 hashes and identical; anything
    else (unknown either side, or a mismatch) raises
    `TokenizerIncompatibilityError`. There is no path through this gate
    that silently approximates token correspondence.
    """
    if request.signal_kind not in TOKEN_ALIGNED_SIGNALS:
        return
    if provider_tokenizer_identity is None or request.student_tokenizer_identity is None:
        raise TokenizerIncompatibilityError(
            f"token-aligned signal {request.signal_kind.value} requires both teacher and "
            "student tokenizer identities; one is unknown, so compatibility is not "
            "provable and the request is refused (fail closed)"
        )
    if provider_tokenizer_identity != request.student_tokenizer_identity:
        raise TokenizerIncompatibilityError(
            f"teacher tokenizer {provider_tokenizer_identity} does not match student "
            f"tokenizer {request.student_tokenizer_identity}; token-level alignment is "
            "refused rather than approximated (fail closed). Use downgrade_request to "
            "explicitly select text/ranking/reward/critique supervision instead."
        )


class TeacherRegistry:
    """Registered providers plus deterministic capability negotiation.

    Negotiation walks providers in registration order and returns the
    first one that (a) declares support for the requested signal kind and
    (b) passes the tokenizer gate for it -- a provider failing either is
    skipped, with its reason recorded for the audit trail in the
    eventual error. No preference, reputation, or cost tie-breaking
    exists yet: that is `TeacherQueryController`'s job (Slice D), and
    inventing it here would pre-empt an unvalidated policy.
    """

    def __init__(self) -> None:
        self._providers: list[TeacherProvider] = []
        self._by_name: dict[str, TeacherProvider] = {}

    def register(self, provider: TeacherProvider) -> None:
        if not isinstance(provider, TeacherProvider):
            raise ValueError("registered object does not satisfy the TeacherProvider protocol")
        name = getattr(provider, "name", None)
        _require_non_empty_str(name, "provider name")
        if name in self._by_name:
            raise ValueError(f"a teacher provider named {name!r} is already registered")
        self._providers.append(provider)
        self._by_name[name] = provider

    def providers(self) -> tuple[TeacherProvider, ...]:
        return tuple(self._providers)

    def get(self, name: str) -> TeacherProvider:
        provider = self._by_name.get(name)
        if provider is None:
            raise KeyError(f"no teacher provider named {name!r} is registered")
        return provider

    def negotiate(self, request: TeacherRequest) -> TeacherProvider:
        """The first registered provider able to serve *request*, or
        `NegotiationError` naming every provider's disqualifying reason."""
        reasons: list[str] = []
        for provider in self._providers:
            if not provider.capabilities().supports(request.signal_kind):
                reasons.append(f"{provider.name}: does not declare {request.signal_kind.value}")
                continue
            try:
                ensure_tokenizer_compatible(provider.tokenizer_identity_sha256, request)
            except TokenizerIncompatibilityError as exc:
                reasons.append(f"{provider.name}: {exc}")
                continue
            return provider
        detail = "; ".join(reasons) if reasons else "no providers are registered"
        raise NegotiationError(f"no provider can serve {request.signal_kind.value} ({detail})")


class FakeTeacherProvider:
    """Deterministic, offline test double -- **not** a stub of a real
    provider and not a source of meaningful teacher signal.

    Payloads are pure functions of the request digest, so tests can
    assert end-to-end artifact construction (and later slices can assert
    dedup/caching) without network or model weight. Payload *content* is
    fixture semantics only and must never be read as evidence of teacher
    quality anywhere in Chowder.
    """

    def __init__(
        self,
        *,
        name: str = "fake",
        capabilities: TeacherCapabilities | None = None,
        tokenizer_identity_sha256: str | None = "f" * 64,
        model_revision: str = "fake-teacher-v1",
    ) -> None:
        self.name = name
        self.tokenizer_identity_sha256 = tokenizer_identity_sha256
        self.model_revision = model_revision
        self._capabilities = capabilities or TeacherCapabilities(
            generate=True,
            critique=True,
            revise=True,
            rank=True,
            scalar_reward=True,
            selected_token_logprobs=False,
            topk_logprobs=False,
            hidden_projection=False,
            remote_training=False,
        )

    def capabilities(self) -> TeacherCapabilities:
        return self._capabilities

    def profile(self, request: TeacherRequest) -> TeacherCostEstimate:
        # Deterministic token estimate from the prompt; a fake provider
        # spends no real money and no GPU time, and reports exactly that.
        return TeacherCostEstimate(
            monetary_cost_usd=0.0,
            estimated_tokens=max(1, len(request.prompt) // 4),
            gpu_hours=None,
            confidence=1.0,
        )

    def query(self, request: TeacherRequest) -> TeacherSignal:
        ensure_tokenizer_compatible(self.tokenizer_identity_sha256, request)
        if not self._capabilities.supports(request.signal_kind):
            raise NegotiationError(
                f"fake provider {self.name!r} does not declare {request.signal_kind.value}"
            )
        digest8 = request.digest()[:8]
        payload: dict[str, Any]
        if request.signal_kind is SignalKind.SCALAR_REWARD:
            payload = {"reward": int(digest8, 16) % 100 / 100.0}
        elif request.signal_kind is SignalKind.CANDIDATE_RANKING:
            candidates = request.input_payload.get("candidates")
            count = len(candidates) if isinstance(candidates, list) and candidates else 2
            payload = {"ranking": list(range(count)), "basis": f"fixture:{digest8}"}
        elif request.signal_kind in (
            SignalKind.SAMPLED_TOKEN_LOGPROBS,
            SignalKind.TOPK_LOGPROBS,
            SignalKind.FULL_LOGITS,
            SignalKind.HIDDEN_PROJECTION,
        ):
            payload = {"logprobs": [-0.1, -0.2], "basis": f"fixture:{digest8}"}
        else:
            kind_text = {
                SignalKind.GENERATED_ANSWER: "generated answer",
                SignalKind.CRITIQUE: "critique",
                SignalKind.REVISED_ANSWER: "revised answer",
                SignalKind.REMOTE_ADAPTER: "adapter reference",
            }.get(request.signal_kind, "signal")
            payload = {kind_text.replace(" ", "_"): f"fixture:{kind_text}:{digest8}"}
        return TeacherSignal(
            signal_kind=request.signal_kind,
            payload=payload,
            resource_usage=None,
            monetary_cost_usd=0.0,
            token_counts={
                "prompt_tokens": max(1, len(request.prompt) // 4),
                "completion_tokens": 8,
            },
        )

    def cancel(self, request_id: str) -> None:
        return None


def build_signal_artifact(
    *,
    provider: TeacherProvider,
    request: TeacherRequest,
    signal: TeacherSignal,
    signal_id: str,
    occurred_at: str,
    latency_seconds: float,
    parent_job_manifest_digest: str | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> TeacherSignalArtifact:
    """Construct the durable artifact for one served request.

    The single construction path so every artifact carries the same
    provenance shape. `provider_type` is the provider's declared
    identity (its `name` here; a real adapter may report a richer type),
    `model_revision` what the provider reports itself to be -- both are
    recorded evidence, not verified identity (verification is Slice G's
    signing problem, honestly out of scope here).
    """
    return TeacherSignalArtifact.from_signal(
        request=request,
        signal=signal,
        signal_id=signal_id,
        provider_type=getattr(provider, "name", "unknown"),
        model_revision=getattr(provider, "model_revision", "unknown"),
        tokenizer_identity_sha256=getattr(provider, "tokenizer_identity_sha256", None),
        occurred_at=occurred_at,
        latency_seconds=latency_seconds,
        generation_parameters=dict(request.parameters),
        parent_job_manifest_digest=parent_job_manifest_digest,
        provenance=provenance,
    )
