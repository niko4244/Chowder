"""CPU correctness checks for the isolated conditional-compute prototype."""
from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")
from torch import nn

from chowder.conditional_compute import (
    RoutingStats,
    AdaptiveEarlyExit,
    AlwaysOnRouter,
    BudgetTokenRouter,
    GateTaskBatch,
    ConditionalComputeError,
    ConditionalDepthLayer,
    ConditionalFFN,
    FixedLayerRouter,
    FixedSkipRouter,
    IntermediatePredictionHead,
    LearnedTokenRouter,
    SparseDepthResult,
    TokenwiseResidualAdapter,
    calibrate_exit_threshold,
    confidence_scores,
    freeze_backbone_train_gates,
    intermediate_head_loss,
    pareto_frontier,
    render_pareto_svg,
    routing_health,
    train_frozen_gate_control,
)


class _CountingFFN(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int = 8):
        super().__init__()
        self.up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.calls = 0
        with torch.no_grad():
            self.up.weight.fill_(0.25)
            self.down.weight.fill_(0.125)

    def forward(self, hidden_states):
        self.calls += 1
        return self.down(torch.relu(self.up(hidden_states)))


class _ResidualBlock(nn.Module):
    def __init__(self, hidden_size: int = 3):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(hidden_size))
        self.calls = 0

    def forward(self, hidden_states):
        self.calls += 1
        return hidden_states + self.proj(hidden_states)


class _SelectiveVerifier:
    supports_selective_decode = True
    supports_cache = False
    supports_selective_cache = False

    def __init__(self, logits: torch.Tensor):
        self.logits = logits
        self.calls: list[torch.Tensor] = []

    def __call__(self, hidden_states, selected_indices, **context):
        del context
        self.calls.append(selected_indices.clone())
        return self.logits.index_select(0, selected_indices)


def test_conditional_ffn_executes_only_selected_token_rows_and_preserves_residual():
    ffn = _CountingFFN(hidden_size=3)
    wrapper = ConditionalFFN(ffn, FixedSkipRouter(skip_every=2, skip_phase=0))
    hidden = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3) / 10
    positions = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])

    delta, stats = wrapper(
        hidden,
        position_ids=positions,
        return_routing_stats=True,
        token_mask=torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
    )
    output = hidden + delta

    assert ffn.calls == 1
    assert stats.tokens == 8
    assert stats.valid_tokens == 5
    assert stats.selected_tokens == 2
    assert stats.executed_tokens == 2
    assert stats.executed_fraction == pytest.approx(0.4)
    # Positions 1/5 are selected; skipped and padded positions keep the
    # ordinary residual exactly, not a zeroed hidden state.
    assert torch.equal(output[0, 0], hidden[0, 0])
    assert torch.equal(output[0, 2], hidden[0, 2])
    assert torch.equal(output[0, 3], hidden[0, 3])
    assert torch.equal(output[1, 1], hidden[1, 1] + ffn(hidden[1, 1].unsqueeze(0))[0])


def test_conditional_ffn_empty_route_does_not_execute_ffn():
    ffn = _CountingFFN(3)
    wrapper = ConditionalFFN(ffn, FixedSkipRouter(skip_every=1, skip_phase=0))
    hidden = torch.randn(2, 3, 3)
    delta, stats = wrapper(hidden, return_routing_stats=True)
    assert ffn.calls == 0
    assert torch.equal(delta, torch.zeros_like(hidden))
    assert stats.executed_tokens == 0


def test_always_on_is_a_dense_control_with_identical_ffn_delta():
    torch.manual_seed(20)
    dense = _CountingFFN(3)
    gated = _CountingFFN(3)
    gated.up.load_state_dict(dense.up.state_dict())
    gated.down.load_state_dict(dense.down.state_dict())
    hidden = torch.randn(2, 5, 3)
    reference = dense(hidden)
    result = ConditionalFFN(gated, AlwaysOnRouter())(hidden)
    assert torch.allclose(result, reference)
    assert gated.calls == 1


def test_budget_router_has_exact_deterministic_per_sequence_budget_and_masks_padding():
    router = BudgetTokenRouter(4, keep_rate=0.5)
    with torch.no_grad():
        router.projection.weight.zero_()
        router.projection.bias.zero_()
    hidden = torch.zeros(2, 5, 4)
    mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]])
    first = router(hidden, token_mask=mask)
    second = router(hidden, token_mask=mask)
    assert torch.equal(first.keep, second.keep)
    assert first.keep.sum(dim=1).tolist() == [2, 2]
    assert not bool(first.keep[~mask.bool()].any())


def test_budget_router_global_scope_applies_one_budget_across_the_batch():
    router = BudgetTokenRouter(4, keep_rate=0.5, budget_scope="global")
    with torch.no_grad():
        router.projection.weight.zero_()
        router.projection.bias.zero_()
    hidden = torch.zeros(2, 4, 4)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])

    plan = router(hidden, token_mask=mask)

    assert int(plan.keep.sum()) == 3
    # Equal scores tie by flattened row order under stable sorting.
    assert plan.keep[0].tolist() == [True, True, True, False]
    assert plan.keep[1].tolist() == [False, False, False, False]
    assert not bool(plan.keep[~mask.bool()].any())


def test_learned_router_budget_loss_trains_gate_while_backbone_is_frozen():
    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(4, 4)
            self.router = LearnedTokenRouter(4, target_keep_rate=0.7)

    model = _Model()
    trainable = freeze_backbone_train_gates(model, [model.router])
    hidden = torch.randn(4, 6, 4)
    loss = model.router.auxiliary_loss(hidden) + model.router.probabilities(hidden).mean()
    loss.backward()

    assert trainable
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.router.parameters())
    assert all(parameter.grad is not None for parameter in model.router.parameters())
    assert all(parameter.grad is None for parameter in model.backbone.parameters())


def test_frozen_gate_trainer_accepts_one_shot_iterable_and_calls_once_per_batch():
    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(4, 4)
            self.router = LearnedTokenRouter(4, target_keep_rate=0.6)

    model = _Model()
    calls = []
    hidden = torch.randn(1, 3, 4)

    def step_fn(batch):
        calls.append(batch)
        task_loss = model.router.probabilities(hidden).square().mean()
        return GateTaskBatch(task_loss, {model.router: hidden})

    def batches():
        yield "first"
        yield "second"

    report = train_frozen_gate_control(
        model, [model.router], batches(), step_fn=step_fn
    )

    assert calls == ["first", "second"]
    assert report.steps == 2


def test_frozen_gate_trainer_updates_only_router_parameters_and_restores_modes():
    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.ffn = _CountingFFN(4)
            self.router = LearnedTokenRouter(4, target_keep_rate=0.6, threshold=0.6)
            self.conditional_ffn = ConditionalFFN(self.ffn, self.router)

    model = _Model()
    hidden = torch.randn(2, 5, 4)
    backbone_before = [parameter.detach().clone() for parameter in model.ffn.parameters()]

    def step_fn(_batch):
        output, stats = model.conditional_ffn(hidden, return_routing_stats=True)
        task_loss = output.square().mean()
        return GateTaskBatch(
            task_loss=task_loss,
            router_inputs={model.router: hidden},
            routing_stats={model.router: stats},
        )

    report = train_frozen_gate_control(
        model,
        [model.router],
        [None, None],
        step_fn=step_fn,
        seed=11,
    )

    assert report.steps == 2
    assert report.backbone_frozen
    assert model.training
    assert all(not parameter.requires_grad for parameter in model.ffn.parameters())
    assert all(parameter.requires_grad for parameter in model.router.parameters())
    assert all(
        torch.equal(before, after)
        for before, after in zip(backbone_before, model.ffn.parameters())
    )
    assert report.router_health["gate_0"]["calls"] == 2


def test_gate_freeze_requires_registered_router_without_mutating_model():
    model = nn.Linear(4, 4)
    assert all(parameter.requires_grad for parameter in model.parameters())
    with pytest.raises(ValueError, match="already belong"):
        freeze_backbone_train_gates(model, [LearnedTokenRouter(4)])
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_gate_freeze_rejects_duplicate_router_without_mutating_model():
    model = nn.Module()
    model.router = LearnedTokenRouter(4)
    with pytest.raises(ValueError, match="duplicates"):
        freeze_backbone_train_gates(model, [model.router, model.router])
    assert all(parameter.requires_grad for parameter in model.router.parameters())


def test_fixed_router_uses_position_ids_for_stable_decode_routing():
    router = FixedSkipRouter(skip_every=3, skip_phase=1)
    hidden = torch.zeros(2, 1, 4)
    assert router(hidden, position_ids=torch.tensor([[1], [4]])).keep.tolist() == [[False], [False]]
    assert router(hidden, position_ids=torch.tensor([[2], [5]])).keep.tolist() == [[True], [True]]


def test_fixed_layer_router_supports_deterministic_static_depth_control():
    hidden = torch.randn(2, 3, 4)
    layer = _ResidualBlock(hidden_size=4)
    wrapped = ConditionalDepthLayer(
        layer,
        FixedLayerRouter(layer_index=1, skip_layers={1, 3}),
        adapter=TokenwiseResidualAdapter(layer.proj),
        fallback="error",
    )

    output, stats = wrapped(hidden, return_routing_stats=True)

    assert torch.equal(output, hidden)
    assert stats.selected_tokens == 0
    assert stats.executed_tokens == 0
    assert layer.calls == 0


def test_conditional_depth_gathers_scatters_selected_rows_and_preserves_skipped_rows():
    hidden = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    layer = _ResidualBlock()
    router = FixedSkipRouter(skip_every=2, skip_phase=0)
    adapter = TokenwiseResidualAdapter(layer.proj)
    wrapped = ConditionalDepthLayer(layer, router, adapter=adapter, fallback="error")
    output, stats = wrapped(hidden, return_routing_stats=True)

    expected = hidden.clone()
    expected[:, 1::2] *= 2
    assert torch.equal(output, expected)
    assert stats.executed_tokens == 4
    assert stats.selected_fraction == pytest.approx(0.5)
    # The adapter uses only its token-local transform, not the full layer.
    assert layer.calls == 0


def test_conditional_depth_stats_normalize_dense_fallback_work_by_valid_tokens():
    hidden = torch.randn(2, 4, 3)
    layer = _ResidualBlock()
    wrapped = ConditionalDepthLayer(
        layer,
        FixedSkipRouter(skip_every=2, skip_phase=1),
        adapter=None,
        fallback="dense",
    )
    valid = torch.tensor([[1, 1, 1, 0], [1, 0, 0, 0]])

    _, stats = wrapped(hidden, token_mask=valid, return_routing_stats=True)

    assert stats.tokens == 8
    assert stats.valid_tokens == 4
    assert stats.selected_fraction == pytest.approx(0.75)
    assert stats.executed_tokens == 8
    assert stats.executed_fraction == pytest.approx(2.0)
    summary = routing_health([stats])
    assert summary["executed_fraction"] == pytest.approx(2.0)
    assert summary["selected_fraction"] == pytest.approx(0.75)


def test_conditional_depth_defaults_to_dense_fallback_without_verified_adapter():
    hidden = torch.randn(2, 4, 3)
    layer = _ResidualBlock()
    wrapped = ConditionalDepthLayer(
        layer,
        FixedSkipRouter(skip_every=2),
        adapter=None,
        fallback="dense",
    )
    output, stats = wrapped(hidden, return_routing_stats=True)
    assert torch.equal(output, layer.proj(hidden) + hidden)
    assert layer.calls == 1
    assert stats.dense_fallback
    assert stats.executed_tokens == hidden.shape[0] * hidden.shape[1]


def test_conditional_depth_skip_preserves_differentiable_gate_gradients():
    hidden = torch.randn(2, 3, 4, requires_grad=True)
    layer = _ResidualBlock(hidden_size=4)
    router = LearnedTokenRouter(4, target_keep_rate=0.5, threshold=0.5)
    wrapper = ConditionalDepthLayer(
        layer,
        router,
        adapter=TokenwiseResidualAdapter(layer.proj),
        fallback="error",
    )

    output = wrapper(hidden)
    output.square().mean().backward()

    assert router.projection.weight.grad is not None
    assert torch.isfinite(router.projection.weight.grad).all()
    assert hidden.grad is not None


def test_conditional_depth_cache_uses_dense_fallback_even_when_all_tokens_skip():
    hidden = torch.randn(1, 1, 3)

    class _CachedResidualBlock(_ResidualBlock):
        def forward(self, hidden_states, **context):
            del context
            return super().forward(hidden_states)

    layer = _CachedResidualBlock()
    wrapped = ConditionalDepthLayer(
        layer,
        FixedSkipRouter(skip_every=2, skip_phase=0),
        adapter=None,
        fallback="dense",
    )
    token_mask = torch.tensor([[True]])
    output, stats = wrapped(
        hidden,
        position_ids=torch.tensor([[0]]),
        token_mask=token_mask,
        cache=object(),
        return_routing_stats=True,
    )
    assert torch.equal(output, hidden * 2)
    assert torch.equal(token_mask, torch.tensor([[True]]))
    assert stats.dense_fallback
    assert stats.executed_tokens == 1


def test_conditional_depth_with_unsupported_adapter_falls_back_even_when_every_row_skips():
    hidden = torch.randn(1, 2, 3)
    layer = _ResidualBlock()
    wrapped = ConditionalDepthLayer(
        layer,
        FixedSkipRouter(skip_every=1),
        adapter=None,
        fallback="dense",
    )

    output, stats = wrapped(hidden, return_routing_stats=True)

    assert torch.equal(output, hidden + layer.proj(hidden))
    assert stats.dense_fallback
    assert stats.selected_tokens == 0
    assert stats.executed_tokens == hidden.shape[0] * hidden.shape[1]


def test_conditional_depth_sparse_adapter_receives_position_ids_once():
    class _PositionAdapter(TokenwiseResidualAdapter):
        def __init__(self, transform):
            super().__init__(transform)
            self.position_ids = None

        def forward_selected_tokens(self, layer, hidden_states, selected_indices, **context):
            self.position_ids = context.pop("position_ids", None)
            return super().forward_selected_tokens(
                layer, hidden_states, selected_indices, **context
            )

    hidden = torch.randn(1, 4, 3)
    positions = torch.arange(4).unsqueeze(0)
    layer = _ResidualBlock()
    adapter = _PositionAdapter(layer.proj)
    wrapped = ConditionalDepthLayer(
        layer, FixedSkipRouter(skip_every=2), adapter=adapter, fallback="error"
    )

    wrapped(hidden, position_ids=positions)

    assert torch.equal(adapter.position_ids, positions)


def test_conditional_depth_validates_sparse_adapter_output_metadata():
    class _WrongDtypeAdapter(TokenwiseResidualAdapter):
        def forward_selected_tokens(self, *args, **kwargs):
            result = super().forward_selected_tokens(*args, **kwargs)
            return SparseDepthResult(result.hidden_states.double())

    hidden = torch.randn(1, 3, 3)
    layer = _ResidualBlock()
    wrapped = ConditionalDepthLayer(
        layer,
        FixedSkipRouter(skip_every=2),
        adapter=_WrongDtypeAdapter(layer.proj),
        fallback="error",
    )

    with pytest.raises(ConditionalComputeError, match="shape/device/dtype"):
        wrapped(hidden)


def test_conditional_depth_cache_requires_explicit_selective_cache_adapter():
    class _UnsafeAdapter(TokenwiseResidualAdapter):
        supports_selective_cache = False

    hidden = torch.randn(1, 3, 3)
    layer = _ResidualBlock()
    adapter = _UnsafeAdapter(layer.proj)
    wrapped = ConditionalDepthLayer(
        layer, FixedSkipRouter(skip_every=2), adapter=adapter, fallback="error"
    )
    with pytest.raises(ConditionalComputeError, match="cache contract"):
        wrapped(hidden, past_key_values=object())


def test_conditional_depth_refuses_adapter_that_does_not_confirm_cache_update():
    class _BadCacheAdapter(TokenwiseResidualAdapter):
        supports_selective_cache = True

        def forward_selected_tokens(self, *args, **kwargs):
            result = super().forward_selected_tokens(*args, **kwargs)
            return SparseDepthResult(result.hidden_states, cache_updated=False)

    hidden = torch.randn(1, 3, 3)
    layer = _ResidualBlock()
    wrapped = ConditionalDepthLayer(
        layer,
        FixedSkipRouter(skip_every=2),
        adapter=_BadCacheAdapter(layer.proj),
        fallback="error",
    )
    with pytest.raises(ConditionalComputeError, match="did not confirm cache/state update"):
        wrapped(hidden, cache=object())


def test_early_exit_uses_intermediate_logits_only_for_high_margin_rows():
    early_logits = torch.tensor([[4.0, 1.0, 0.0], [0.6, 0.5, 0.0], [0.7, 0.6, 0.0]])
    deep_logits = torch.tensor([[0.0, 4.0, 0.0], [0.0, 5.0, 0.0], [0.0, 6.0, 0.0]])
    hidden = torch.randn(3, 1, 8)
    verifier = _SelectiveVerifier(deep_logits)
    controller = AdaptiveEarlyExit(threshold=1.0)

    output, stats = controller(
        early_logits,
        hidden,
        verifier=verifier,
        single_token_decode=True,
        return_stats=True,
    )

    assert torch.equal(output[0], early_logits[0])
    assert torch.equal(output[1:], deep_logits[1:])
    assert stats.early_exit_tokens == 1
    assert stats.verification_tokens == 2
    assert stats.deep_tokens == 2
    assert verifier.calls[0].tolist() == [1, 2]


def test_early_exit_rejects_cached_decode_aliases_and_invalid_empty_rows():
    controller = AdaptiveEarlyExit(threshold=0.1)
    hidden = torch.ones(1, 1, 4)
    verifier = _SelectiveVerifier(torch.ones(1, 2))
    with pytest.raises(ConditionalComputeError, match="cache/Mamba state"):
        controller(
            torch.ones(1, 2),
            hidden,
            verifier=verifier,
            single_token_decode=True,
            cache_position=torch.tensor([3]),
        )
    with pytest.raises(ValueError, match="at least one row"):
        controller(
            torch.empty(0, 2),
            torch.empty(0, 4),
            verifier=verifier,
            single_token_decode=True,
        )


def test_early_exit_refuses_prefill_cache_or_unverified_verifier_paths():
    early = torch.tensor([[4.0, 0.0], [0.6, 0.5]])
    hidden = torch.randn(2, 1, 4)
    verifier = _SelectiveVerifier(torch.tensor([[0.0, 4.0], [0.0, 4.0]]))
    controller = AdaptiveEarlyExit(threshold=1.0)
    with pytest.raises(ConditionalComputeError, match="prefill/sequence"):
        controller(early, hidden, verifier=verifier, single_token_decode=False)

    class _CacheVerifier(_SelectiveVerifier):
        supports_cache = True

    with pytest.raises(ConditionalComputeError, match="cache/Mamba state"):
        controller(
            early,
            hidden,
            verifier=_CacheVerifier(torch.ones_like(early)),
            single_token_decode=True,
            cache=object(),
        )

    class _NoVerifier(_SelectiveVerifier):
        supports_selective_decode = False

    with pytest.raises(ConditionalComputeError, match="lacks selective-decode"):
        controller(
            early,
            hidden,
            verifier=_NoVerifier(torch.ones_like(early)),
            single_token_decode=True,
        )


def test_calibrate_exit_threshold_requires_finite_logits():
    with pytest.raises(ValueError, match="finite floating-point"):
        calibrate_exit_threshold(
            torch.tensor([[float("nan"), 0.0]]),
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([0]),
            early_compute_per_token=1.0,
            full_compute_per_token=2.0,
            minimum_accuracy=1.0,
        )


def test_calibrate_exit_threshold_for_dev_accuracy_floor_and_compute():
    early = torch.tensor([[3.0, 1.0], [2.0, 1.0], [0.0, 2.0]])
    deep = torch.tensor([[3.0, 1.0], [0.0, 2.0], [0.0, 2.0]])
    labels = torch.tensor([0, 1, 1])
    result = calibrate_exit_threshold(
        early,
        deep,
        labels,
        early_compute_per_token=1.0,
        full_compute_per_token=4.0,
        minimum_accuracy=1.0,
    )
    assert result.accuracy == 1.0
    assert result.full_accuracy == 1.0
    assert result.early_exit_fraction == pytest.approx(2 / 3)
    assert result.fallback_fraction == pytest.approx(1 / 3)
    assert result.mean_compute_per_token == pytest.approx(2.0)
    assert result.candidate_thresholds >= 2


def test_prediction_head_loss_gradients_and_all_ignored_labels_are_finite():
    head = IntermediatePredictionHead(hidden_size=4, vocab_size=7)
    hidden = torch.randn(2, 3, 4, requires_grad=True)
    logits = head(hidden)
    labels = torch.randint(0, 7, (2, 3))
    loss = intermediate_head_loss(logits, labels)
    loss.backward()
    assert head.projection.weight.grad is not None
    assert hidden.grad is not None

    ignored = intermediate_head_loss(logits, torch.full((2, 3), -100))
    assert torch.isfinite(ignored)
    assert ignored.item() == 0.0


def test_intermediate_head_loss_rejects_invalid_label_values():
    logits = torch.randn(2, 3, requires_grad=True)
    with pytest.raises(ValueError, match="integer class ids"):
        intermediate_head_loss(logits, torch.tensor([0.0, float("nan")]))
    with pytest.raises(ValueError, match="integer class ids"):
        intermediate_head_loss(logits, torch.tensor([True, False]))


def test_confidence_is_only_a_ranking_score_and_rejects_invalid_logits():
    logits = torch.tensor([[1.0, 0.0, -1.0]])
    assert confidence_scores(logits, kind="margin").item() == pytest.approx(1.0)
    assert 0.0 < confidence_scores(logits, kind="max_probability").item() < 1.0
    with pytest.raises(ValueError, match="finite"):
        confidence_scores(torch.tensor([[float("nan"), 0.0]]))


def test_early_exit_rejects_multi_token_rows_and_nonfinite_verifier_outputs():
    controller = AdaptiveEarlyExit(threshold=0.1)
    verifier = _SelectiveVerifier(torch.ones(2, 3))
    with pytest.raises(ConditionalComputeError, match="multi-token"):
        controller(
            torch.ones(1, 2, 3),
            torch.ones(1, 2, 4),
            verifier=verifier,
            single_token_decode=True,
        )

    class _BadVerifier(_SelectiveVerifier):
        def __call__(self, hidden_states, selected_indices, **context):
            return torch.full((len(selected_indices), 3), float("nan"))

    with pytest.raises(ConditionalComputeError, match="non-finite"):
        controller(
            torch.tensor([[0.05, 0.0, 0.0]]),
            torch.ones(1, 1, 4),
            verifier=_BadVerifier(torch.ones(1, 3)),
            single_token_decode=True,
        )


def test_routing_health_rejects_nonfinite_counts_fractions_and_collapse_floor():
    with pytest.raises(ValueError, match="collapse_floor"):
        routing_health([], collapse_floor=float("nan"))
    with pytest.raises(ValueError, match="counts"):
        routing_health([RoutingStats(1.5, 1, 1, 1.0, 1.0)])
    with pytest.raises(ValueError, match="fractions"):
        routing_health([RoutingStats(2, 1, 1, float("nan"), 0.5)])
    with pytest.raises(ValueError, match="fallback_reason"):
        routing_health([RoutingStats(1, 1, 1, 1.0, 1.0, fallback_reason=3)])


def test_routing_health_reports_always_on_and_off_collapse():
    always_on = routing_health([RoutingStats(100, 100, 100, 1.0, 1.0)])
    always_off = routing_health([RoutingStats(100, 0, 0, 0.0, 0.0)])
    assert always_on["gate_collapsed"]
    assert always_on["collapse_mode"] == "always_on"
    assert always_off["gate_collapsed"]
    assert always_off["collapse_mode"] == "always_off"
    invalid = RoutingStats(10, 2, 2, 0.2, 0.3, valid_tokens=10)
    with pytest.raises(ValueError, match="fractions"):
        routing_health([invalid])


def test_pareto_frontier_filters_dominated_points_and_renders_svg(tmp_path):
    points = [
        {"variant": "dense", "compute_per_token": 100.0, "quality": 0.95},
        {"variant": "fixed", "compute_per_token": 70.0, "quality": 0.91},
        {"variant": "learned", "compute_per_token": 60.0, "quality": 0.90},
        {"variant": "worse", "compute_per_token": 80.0, "quality": 0.88},
    ]
    frontier = pareto_frontier(points)
    assert [(point["compute_per_token"], point["quality"]) for point in frontier] == [
        (60.0, 0.90), (70.0, 0.91), (100.0, 0.95)
    ]
    path = tmp_path / "pareto.svg"
    render_pareto_svg(points, path)
    svg = path.read_text(encoding="utf-8")
    assert "<svg" in svg
    assert 'points="78.00,274.57 231.50,226.86 692.00,36.00"' in svg


def test_pareto_svg_escapes_axis_labels_and_rejects_invalid_dimensions(tmp_path):
    path = tmp_path / "escaped.svg"
    render_pareto_svg(
        [{"compute<&": 1.0, "quality<score": 0.5}],
        path,
        compute_key="compute<&",
        quality_key="quality<score",
    )
    svg = path.read_text(encoding="utf-8")
    assert "compute&lt;&amp;" in svg
    assert "quality&lt;score" in svg
    with pytest.raises(ValueError, match="integer SVG dimensions"):
        render_pareto_svg(
            [{"compute_per_token": 1.0, "quality": 0.5}],
            tmp_path / "invalid.svg",
            width=True,
        )
