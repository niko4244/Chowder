import math

import pytest

from chowder.resources import ResourceUsage
from chowder.teacher_fabric import (
    SCHEMA_VERSION,
    FakeTeacherProvider,
    NegotiationError,
    SignalKind,
    TeacherCapabilities,
    TeacherCostEstimate,
    TeacherProvider,
    TeacherRegistry,
    TeacherRequest,
    TeacherSignal,
    TeacherSignalArtifact,
    TokenizerIncompatibilityError,
    build_signal_artifact,
    downgrade_request,
    ensure_tokenizer_compatible,
)

_STUDENT_TOKENS = "a" * 64
_TEACHER_TOKENS = "a" * 64
_OTHER_TOKENS = "b" * 64


def _request(kind=SignalKind.CRITIQUE, **overrides):
    fields = {
        "teacher_id": "teacher-frontier-x",
        "signal_kind": kind,
        "prompt": "Review this solution: 2+2=4",
        "input_payload": {"candidate": "2+2=4"},
        "parameters": {"temperature": 0.2},
        "student_tokenizer_identity": _STUDENT_TOKENS,
    }
    fields.update(overrides)
    return TeacherRequest(**fields)


# --- schema validation -------------------------------------------------------


def test_capabilities_reject_non_bool_flags():
    with pytest.raises(ValueError, match="strict bool"):
        TeacherCapabilities(
            generate=True,
            critique=1,  # type: ignore[arg-type]
            revise=True,
            rank=True,
            scalar_reward=True,
            selected_token_logprobs=False,
            topk_logprobs=False,
            hidden_projection=False,
            remote_training=False,
        )


def test_request_rejects_bad_tokenizer_identity_and_empty_fields():
    with pytest.raises(ValueError, match="sha256"):
        _request(student_tokenizer_identity="not-a-hash")
    with pytest.raises(ValueError, match="non-empty"):
        _request(teacher_id="")
    with pytest.raises(ValueError, match="SignalKind"):
        _request(kind="critique")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="mapping"):
        _request(input_payload=["not", "a", "mapping"])


def test_signal_rejects_non_finite_and_negative_costs():
    with pytest.raises(ValueError, match="finite"):
        TeacherSignal(
            signal_kind=SignalKind.SCALAR_REWARD,
            payload={"reward": 1.0},
            monetary_cost_usd=float("nan"),
        )
    with pytest.raises(ValueError, match="non-negative"):
        TeacherSignal(
            signal_kind=SignalKind.SCALAR_REWARD,
            payload={"reward": 1.0},
            token_counts={"prompt_tokens": -1},
        )


def test_full_logits_is_not_negotiable_through_the_capability_vocabulary():
    """Honesty rule 1: the research-only full-logits mode has no capability
    flag, so nothing can claim it -- and deriving one from topk_logprobs
    would be the forbidden inference."""
    caps = TeacherCapabilities(
        generate=True,
        critique=True,
        revise=True,
        rank=True,
        scalar_reward=True,
        selected_token_logprobs=True,
        topk_logprobs=True,
        hidden_projection=True,
        remote_training=True,
    )
    assert caps.supports(SignalKind.FULL_LOGITS) is False
    provider = FakeTeacherProvider(capabilities=caps, tokenizer_identity_sha256=_TEACHER_TOKENS)
    registry = TeacherRegistry()
    registry.register(provider)
    with pytest.raises(NegotiationError, match="full_logits"):
        registry.negotiate(_request(kind=SignalKind.FULL_LOGITS))


# --- digests -----------------------------------------------------------------


def test_request_digest_is_deterministic_and_content_sensitive():
    a = _request()
    b = _request()
    assert a.digest() == b.digest()
    assert a.prompt_digest() == b.prompt_digest()
    assert a.digest() != _request(parameters={"temperature": 0.3}).digest()
    assert a.prompt_digest() != _request(prompt="different").prompt_digest()
    assert a.digest() != _request(signal_kind=SignalKind.CANDIDATE_RANKING).digest()


def test_student_trajectory_digest_present_only_when_trajectory_exists():
    assert _request().student_trajectory_digest() is None
    with_traj = _request(student_trajectory=("step one", "step two"))
    assert with_traj.student_trajectory_digest() is not None
    # Distinct orderings are distinct digests.
    assert with_traj.student_trajectory_digest() != _request(
        student_trajectory=("step two", "step one")
    ).student_trajectory_digest()


def test_request_digest_includes_schema_version():
    """A schema change must never collide with an older request's key."""
    request = _request()
    payload = request.canonical_payload()
    assert payload["schema_version"] == SCHEMA_VERSION
    # Everything the digest covers carries the version, so bumping
    # SCHEMA_VERSION changes every derived digest.
    assert str(SCHEMA_VERSION) in request.canonical_json()


# --- tokenizer compatibility gate --------------------------------------------


def test_token_aligned_request_with_matching_tokenizer_passes_the_gate():
    request = _request(
        kind=SignalKind.SAMPLED_TOKEN_LOGPROBS, student_tokenizer_identity=_STUDENT_TOKENS
    )
    # Matching identities pass (no exception) and stay silent no-ops for
    # non-token-aligned kinds even with mismatched/unknown identities.
    ensure_tokenizer_compatible(_TEACHER_TOKENS, request)
    ensure_tokenizer_compatible(None, _request(kind=SignalKind.CRITIQUE))


def test_tokenizer_mismatch_fails_closed_never_approximates():
    request = _request(
        kind=SignalKind.SAMPLED_TOKEN_LOGPROBS, student_tokenizer_identity=_STUDENT_TOKENS
    )
    with pytest.raises(TokenizerIncompatibilityError, match="fail closed"):
        ensure_tokenizer_compatible(_OTHER_TOKENS, request)


def test_unknown_tokenizer_identity_fails_closed_for_token_aligned_kinds():
    request = _request(
        kind=SignalKind.TOPK_LOGPROBS, student_tokenizer_identity=_STUDENT_TOKENS
    )
    with pytest.raises(TokenizerIncompatibilityError, match="unknown"):
        ensure_tokenizer_compatible(None, request)
    with pytest.raises(TokenizerIncompatibilityError, match="unknown"):
        ensure_tokenizer_compatible(
            _TEACHER_TOKENS, _request(kind=SignalKind.TOPK_LOGPROBS, student_tokenizer_identity=None)
        )


def test_negotiation_refuses_token_aligned_request_on_tokenizer_mismatch():
    token_scorer_caps = TeacherCapabilities(
        generate=False,
        critique=False,
        revise=False,
        rank=False,
        scalar_reward=False,
        selected_token_logprobs=True,
        topk_logprobs=False,
        hidden_projection=False,
        remote_training=False,
    )
    provider = FakeTeacherProvider(
        name="token-scorer",
        capabilities=token_scorer_caps,
        tokenizer_identity_sha256=_OTHER_TOKENS,
    )
    registry = TeacherRegistry()
    registry.register(provider)
    request = _request(kind=SignalKind.SAMPLED_TOKEN_LOGPROBS)
    with pytest.raises(NegotiationError, match="tokenizer"):
        registry.negotiate(request)


def test_downgrade_is_explicit_and_never_token_aligned():
    request = _request(kind=SignalKind.SAMPLED_TOKEN_LOGPROBS)
    downgraded = downgrade_request(request, to=SignalKind.CRITIQUE)
    assert downgraded.signal_kind is SignalKind.CRITIQUE
    # Same teacher/prompt/inputs -- only the signal kind changed.
    assert downgraded.digest() != request.digest()
    assert downgraded.prompt == request.prompt
    with pytest.raises(ValueError, match="itself token-aligned"):
        downgrade_request(request, to=SignalKind.TOPK_LOGPROBS)


# --- registry + negotiation --------------------------------------------------


def test_registry_rejects_non_provider_and_duplicate_names():
    registry = TeacherRegistry()
    with pytest.raises(ValueError, match="TeacherProvider protocol"):
        registry.register(object())
    registry.register(FakeTeacherProvider(name="one"))
    with pytest.raises(ValueError, match="already registered"):
        registry.register(FakeTeacherProvider(name="one"))


def test_negotiation_picks_first_capable_provider_in_registration_order():
    registry = TeacherRegistry()
    critique_only = FakeTeacherProvider(
        name="critique-only",
        capabilities=TeacherCapabilities(
            generate=False,
            critique=True,
            revise=False,
            rank=False,
            scalar_reward=False,
            selected_token_logprobs=False,
            topk_logprobs=False,
            hidden_projection=False,
            remote_training=False,
        ),
    )
    full = FakeTeacherProvider(name="full", tokenizer_identity_sha256=_OTHER_TOKENS)
    registry.register(critique_only)
    registry.register(full)
    # The ranking request needs `rank`, which only the second provider declares.
    assert registry.negotiate(_request(kind=SignalKind.CANDIDATE_RANKING)) is full
    # The critique request is served by the first declarer, in order.
    assert registry.negotiate(_request(kind=SignalKind.CRITIQUE)) is critique_only


def test_negotiation_error_names_every_provider_reason():
    registry = TeacherRegistry()
    # "a" keeps the default capabilities (no token-logprob support).
    registry.register(FakeTeacherProvider(name="a"))
    # "b" declares the capability but carries a mismatched tokenizer.
    registry.register(
        FakeTeacherProvider(
            name="b",
            capabilities=TeacherCapabilities(
                generate=True,
                critique=True,
                revise=True,
                rank=True,
                scalar_reward=True,
                selected_token_logprobs=True,
                topk_logprobs=True,
                hidden_projection=False,
                remote_training=False,
            ),
            tokenizer_identity_sha256=_OTHER_TOKENS,
        )
    )
    with pytest.raises(NegotiationError) as excinfo:
        registry.negotiate(_request(kind=SignalKind.SAMPLED_TOKEN_LOGPROBS))
    message = str(excinfo.value)
    # Both disqualifying reasons are named, per provider: capability for
    # one, tokenizer gate for the other.
    assert "a: does not declare" in message
    assert "b:" in message and "tokenizer" in message


# --- fake provider + artifact construction -----------------------------------


def test_fake_provider_is_a_runtime_checkable_teacher_provider():
    assert isinstance(FakeTeacherProvider(), TeacherProvider)


def test_fake_provider_query_is_deterministic_and_capability_checked():
    provider = FakeTeacherProvider()
    request = _request(kind=SignalKind.CRITIQUE)
    first = provider.query(request)
    second = provider.query(request)
    assert first.payload == second.payload
    no_generate = FakeTeacherProvider(
        capabilities=TeacherCapabilities(
            generate=False,
            critique=False,
            revise=False,
            rank=False,
            scalar_reward=False,
            selected_token_logprobs=False,
            topk_logprobs=False,
            hidden_projection=False,
            remote_training=False,
        )
    )
    with pytest.raises(NegotiationError, match="does not declare"):
        no_generate.query(request)


def test_artifact_round_trip_carries_every_required_provenance_field():
    token_scorer_caps = TeacherCapabilities(
        generate=True,
        critique=True,
        revise=True,
        rank=True,
        scalar_reward=True,
        selected_token_logprobs=True,
        topk_logprobs=False,
        hidden_projection=False,
        remote_training=False,
    )
    provider = FakeTeacherProvider(
        name="fake",
        capabilities=token_scorer_caps,
        tokenizer_identity_sha256=_TEACHER_TOKENS,
    )
    request = _request(
        kind=SignalKind.SAMPLED_TOKEN_LOGPROBS,
        student_trajectory=("state one", "state two"),
    )
    signal = provider.query(request)
    artifact = build_signal_artifact(
        provider=provider,
        request=request,
        signal=signal,
        signal_id="sig-1",
        occurred_at="2026-09-06T00:00:00+00:00",
        latency_seconds=0.25,
        provenance={"license": "unknown"},
    )
    # The brief's required metadata, all present:
    assert artifact.teacher_id == "teacher-frontier-x"
    assert artifact.provider_type == "fake"
    assert artifact.model_revision == "fake-teacher-v1"
    assert artifact.tokenizer_identity_sha256 == _TEACHER_TOKENS
    assert artifact.request_digest == request.digest()
    assert artifact.prompt_digest == request.prompt_digest()
    assert artifact.student_trajectory_digest == request.student_trajectory_digest()
    assert artifact.generation_parameters == {"temperature": 0.2}
    assert artifact.occurred_at == "2026-09-06T00:00:00+00:00"
    assert artifact.latency_seconds == pytest.approx(0.25)
    assert artifact.monetary_cost_usd == 0.0
    assert artifact.token_counts["prompt_tokens"] > 0
    assert artifact.gpu_hours is None  # API teacher: no GPU cost to report
    assert artifact.parent_job_manifest_digest is None  # Slice G
    assert artifact.provenance == {"license": "unknown"}
    assert artifact.payload_content_sha256 == signal.payload_digest()


def test_artifact_digest_is_deterministic_and_content_sensitive():
    provider = FakeTeacherProvider()
    request = _request(kind=SignalKind.CRITIQUE)
    signal = provider.query(request)
    kwargs = dict(
        provider=provider,
        request=request,
        signal=signal,
        signal_id="sig-1",
        occurred_at="2026-09-06T00:00:00+00:00",
        latency_seconds=0.25,
    )
    a = build_signal_artifact(**kwargs)
    b = build_signal_artifact(**kwargs)
    assert a.digest() == b.digest()
    assert a.digest() != build_signal_artifact(**{**kwargs, "signal_id": "sig-2"}).digest()
    assert a.digest() != build_signal_artifact(**{**kwargs, "latency_seconds": 0.26}).digest()


def test_artifact_records_gpu_backed_teacher_cost_through_resource_usage():
    provider = FakeTeacherProvider()
    request = _request(kind=SignalKind.CRITIQUE)
    signal = TeacherSignal(
        signal_kind=SignalKind.CRITIQUE,
        payload=provider.query(request).payload,
        resource_usage=ResourceUsage.from_wall_time(
            wall_seconds=90, active_accelerator_count=1
        ),
        monetary_cost_usd=0.12,
        token_counts={"prompt_tokens": 40, "completion_tokens": 20},
    )
    artifact = build_signal_artifact(
        provider=provider,
        request=request,
        signal=signal,
        signal_id="sig-gpu",
        occurred_at="2026-09-06T00:00:00+00:00",
        latency_seconds=90.0,
    )
    assert artifact.gpu_hours == pytest.approx(90 / 3600.0)
    assert artifact.monetary_cost_usd == pytest.approx(0.12)


def test_artifact_rejects_disagreement_with_reported_signal_cost():
    provider = FakeTeacherProvider()
    request = _request(kind=SignalKind.CRITIQUE)
    signal = provider.query(request)
    # Direct construction with an artifact-side cost that disagrees with
    # the provider-reported signal cost is rejected (from_signal cannot
    # produce this: it copies the signal's reported values verbatim).
    with pytest.raises(ValueError, match="monetary cost disagrees"):
        TeacherSignalArtifact(
            signal_id="sig-x",
            teacher_id=request.teacher_id,
            provider_type="fake",
            model_revision="v1",
            tokenizer_identity_sha256=_TEACHER_TOKENS,
            request=request,
            signal=signal,
            occurred_at="2026-09-06T00:00:00+00:00",
            latency_seconds=0.1,
            generation_parameters={},
            monetary_cost_usd=0.99,
            token_counts={},
            gpu_hours=None,
            parent_job_manifest_digest=None,
            provenance={},
        )


def test_artifact_refuses_signal_kind_mismatch():
    provider = FakeTeacherProvider()
    request = _request(kind=SignalKind.CRITIQUE)
    signal = provider.query(_request(kind=SignalKind.CANDIDATE_RANKING))
    with pytest.raises(ValueError, match="different signal kind"):
        TeacherSignalArtifact.from_signal(
            request=request,
            signal=signal,
            signal_id="sig-y",
            provider_type="fake",
            model_revision="v1",
            tokenizer_identity_sha256=None,
            occurred_at="2026-09-06T00:00:00+00:00",
            latency_seconds=0.1,
        )


def test_cost_estimate_validation():
    assert TeacherCostEstimate(monetary_cost_usd=0.0, estimated_tokens=0).gpu_hours is None
    with pytest.raises(ValueError, match="estimated_tokens"):
        TeacherCostEstimate(monetary_cost_usd=0.1, estimated_tokens=-1)
    with pytest.raises(ValueError, match="confidence"):
        TeacherCostEstimate(monetary_cost_usd=0.1, estimated_tokens=1, confidence=1.5)
    with pytest.raises(ValueError, match="finite"):
        TeacherCostEstimate(
            monetary_cost_usd=float("inf"), estimated_tokens=1, gpu_hours=None
        )


def test_provider_declared_identity_is_recorded_not_verified():
    """provider_type/model_revision are recorded evidence of what the
    provider *claims*; verification is Slice G's signing problem."""
    provider = FakeTeacherProvider(model_revision="pretend-frontier")
    request = _request(kind=SignalKind.CRITIQUE)
    artifact = build_signal_artifact(
        provider=provider,
        request=request,
        signal=provider.query(request),
        signal_id="sig-z",
        occurred_at="2026-09-06T00:00:00+00:00",
        latency_seconds=0.1,
    )
    assert artifact.model_revision == "pretend-frontier"
