"""Layer-bulk transient experts (``cpu-transient-grad-bulk``).

One host->device copy of each expert weight matrix per layer, instead of one per
expert, must produce exactly the per-expert transient forward's output and
gradients -- on the CPU, with tiny synthetic modules, so no CUDA is touched.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from chowder.backends import moe_transient_grad as mtg  # noqa: E402
from chowder.backends.transformers_peft import TransformersPeftRunSpec  # noqa: E402

NUM_EXPERTS, HIDDEN, INTER = 6, 8, 4
TOKENS, TOP_K = 10, 2


class Qwen3MoeExperts(nn.Module):
    """The name and the three 3-D attributes the install hook looks for."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.num_experts = NUM_EXPERTS
        self.gate_up_proj = nn.Parameter(
            torch.randn(NUM_EXPERTS, 2 * INTER, HIDDEN, generator=generator), requires_grad=False
        )
        self.down_proj = nn.Parameter(
            torch.randn(NUM_EXPERTS, HIDDEN, INTER, generator=generator), requires_grad=False
        )
        self.act_fn = F.silu

    def forward(self, hidden_states, top_k_index, top_k_weights):  # replaced by the install
        raise AssertionError("the test must install a transient forward before calling this")


class FakeMoeModel(nn.Module):
    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        self.experts = Qwen3MoeExperts(seed)


def _routing(seed: int = 11, experts=(0, 1, 2, 3, 4, 5)) -> torch.Tensor:
    choices = torch.tensor(experts)
    generator = torch.Generator().manual_seed(seed)
    rows = []
    for _ in range(TOKENS):
        pick = torch.randperm(len(choices), generator=generator)[:TOP_K]
        rows.append(choices[pick])
    return torch.stack(rows)


def _inputs(seed: int = 1, experts=(0, 1, 2, 3, 4, 5)):
    generator = torch.Generator().manual_seed(seed)
    hidden = torch.randn(TOKENS, HIDDEN, generator=generator, requires_grad=True)
    weights = (torch.randn(TOKENS, TOP_K, generator=generator) * 0.5).requires_grad_(True)
    return hidden, _routing(seed=seed + 10, experts=experts), weights


def _reference(hidden, index, weights, experts) -> torch.Tensor:
    """The plain loop: every expert, in index order, no autograd Function."""
    final = torch.zeros_like(hidden)
    mask = F.one_hot(index, num_classes=NUM_EXPERTS).permute(2, 1, 0)
    for e in range(NUM_EXPERTS):
        top_k_pos, token_idx = torch.where(mask[e])
        if token_idx.numel() == 0:
            continue
        gate, up = F.linear(hidden[token_idx], experts.gate_up_proj[e]).chunk(2, dim=-1)
        out = F.linear(experts.act_fn(gate) * up, experts.down_proj[e])
        out = out * weights[token_idx, top_k_pos, None]
        final.index_add_(0, token_idx, out.to(final.dtype))
    return final


def _per_expert_model():
    model = FakeMoeModel()
    assert mtg.install_transient_grad_expert_forward(model) == 1
    return model


def _bulk_model():
    model = FakeMoeModel()
    assert mtg.install_transient_grad_expert_forward(model, bulk=True) == 1
    return model


def test_bulk_forward_matches_the_per_expert_forward_and_the_reference():
    per_expert, bulk = _per_expert_model(), _bulk_model()
    hidden, index, weights = _inputs()
    reference = _reference(hidden, index, weights, per_expert.experts)
    transient = per_expert.experts(hidden, index, weights)
    layer_bulk = bulk.experts(hidden, index, weights)
    assert torch.allclose(reference, transient, atol=1e-6)
    assert torch.allclose(reference, layer_bulk, atol=1e-6)
    assert torch.allclose(transient, layer_bulk, atol=1e-6)


def _forward_and_grads(module) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden, index, weights = _inputs()
    out = module(hidden, index, weights)
    out.square().sum().backward()
    return out.detach(), hidden.grad.detach(), weights.grad.detach()


def _reference_forward_and_grads(experts) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden, index, weights = _inputs()
    out = _reference(hidden, index, weights, experts)
    out.square().sum().backward()
    return out.detach(), hidden.grad.detach(), weights.grad.detach()


def test_bulk_gradients_match_the_reference_for_hidden_states_and_routing():
    per_expert, bulk = _per_expert_model(), _bulk_model()
    out_r, grad_hidden_r, grad_weights_r = _reference_forward_and_grads(per_expert.experts)
    out_p, grad_hidden_p, grad_weights_p = _forward_and_grads(per_expert.experts)
    out_b, grad_hidden_b, grad_weights_b = _forward_and_grads(bulk.experts)

    assert torch.allclose(out_r, out_b, atol=1e-6)
    assert torch.allclose(grad_hidden_r, grad_hidden_b, atol=1e-5)
    assert torch.allclose(grad_hidden_p, grad_hidden_b, atol=1e-5)
    # The router sees the same gradient through top_k_weights.
    assert torch.allclose(grad_weights_r, grad_weights_b, atol=1e-5)
    assert torch.allclose(grad_weights_p, grad_weights_b, atol=1e-5)
    assert grad_weights_b.abs().sum() > 0
    # The experts stay frozen and get no gradient.
    for name, param in bulk.named_parameters():
        assert not param.requires_grad, name
        assert param.grad is None, name


def test_bulk_works_under_activation_checkpointing_with_identical_grads():
    per_expert, bulk = _per_expert_model(), _bulk_model()

    def grads(experts, *, checkpointed: bool):
        hidden, index, weights = _inputs()

        def forward(hidden_states):
            return experts(hidden_states, index, weights).square().sum()

        loss = (
            torch.utils.checkpoint.checkpoint(forward, hidden, use_reentrant=False)
            if checkpointed
            else forward(hidden)
        )
        grad_hidden, grad_weights = torch.autograd.grad(loss, (hidden, weights))
        return loss.detach(), grad_hidden, grad_weights

    loss_p, hidden_p, weights_p = grads(per_expert.experts, checkpointed=False)
    loss_b, hidden_b, weights_b = grads(bulk.experts, checkpointed=False)
    loss_bc, hidden_bc, weights_bc = grads(bulk.experts, checkpointed=True)

    assert torch.allclose(loss_p, loss_b, atol=1e-6)
    assert torch.allclose(loss_b, loss_bc, atol=1e-6)
    assert torch.allclose(hidden_p, hidden_bc, atol=1e-5)
    assert torch.allclose(weights_p, weights_bc, atol=1e-5)
    assert torch.allclose(hidden_b, hidden_bc, atol=1e-5)
    assert torch.allclose(weights_b, weights_bc, atol=1e-5)


def test_an_expert_with_no_routed_tokens_is_skipped_without_error():
    per_expert, bulk = _per_expert_model(), _bulk_model()
    hidden, index, weights = _inputs(experts=(0, 1, 2, 3, 4))
    assert not bool((index == 5).any()), "the fixture must leave expert 5 unrouted"
    reference = _reference(hidden, index, weights, per_expert.experts)
    assert torch.allclose(reference, per_expert.experts(hidden, index, weights), atol=1e-6)
    assert torch.allclose(reference, bulk.experts(hidden, index, weights), atol=1e-6)


def test_bulk_refuses_trainable_expert_weights_with_the_same_message():
    per_expert, bulk = FakeMoeModel(), FakeMoeModel()
    for model in (per_expert, bulk):
        model.experts.gate_up_proj.requires_grad_(True)
    with pytest.raises(RuntimeError, match="trainable expert weights") as default_error:
        mtg.install_transient_grad_expert_forward(per_expert)
    with pytest.raises(RuntimeError, match="trainable expert weights") as bulk_error:
        mtg.install_transient_grad_expert_forward(bulk, bulk=True)
    assert str(default_error.value) == str(bulk_error.value)


def test_pin_expert_memory_without_cuda_is_the_skipped_noop(monkeypatch):
    monkeypatch.delenv("CHOWDER_NO_GPU", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert mtg.pin_expert_memory(FakeMoeModel()) == {"pinned_bytes": 0, "skipped": "no cuda"}


def test_placement_validation_accepts_bulk_and_still_rejects_unknown():
    def spec(**overrides):
        fields = {"base_model": "base", "dataset": "data.jsonl", "output_dir": "out"}
        fields.update(overrides)
        return TransformersPeftRunSpec(**fields)

    assert "cpu-transient-grad-bulk" in mtg.EXPERT_PLACEMENTS
    ok = spec(expert_placement="cpu-transient-grad-bulk", target_preset="moe_frozen_experts")
    # Placement is value-transparent: it changes where frozen weights live.
    assert ok.recipe_digest() == spec(target_preset="moe_frozen_experts").recipe_digest()
    spec(expert_placement="cpu-transient-grad-bulk", target_preset="moe_frozen_experts", expert_pinning="registered")

    with pytest.raises(ValueError, match="unsupported expert_placement"):
        spec(expert_placement="gpu-magic")
    with pytest.raises(ValueError, match="moe_frozen_experts"):
        spec(expert_placement="cpu-transient-grad-bulk")
    with pytest.raises(ValueError, match="quantization='none'"):
        spec(
            expert_placement="cpu-transient-grad-bulk",
            target_preset="moe_frozen_experts",
            quantization="4bit",
        )
    with pytest.raises(ValueError, match="unsupported expert_pinning"):
        spec(expert_pinning="gpu-magic")
    with pytest.raises(ValueError, match="requires an expert_placement"):
        spec(expert_pinning="registered")


def test_layer_bulk_mode_does_not_exist_on_origin_main():
    """The one test that is red before this change: the mode and the pinning
    entry point are new, so ``origin/main`` has neither."""
    assert "cpu-transient-grad-bulk" in mtg.EXPERT_PLACEMENTS
    assert callable(mtg.pin_expert_memory)
    assert "cpu-transient-grad-bulk" in mtg.TRANSIENT_PLACEMENTS
    from chowder.backends.transformers_peft import _ALLOWED_EXPERT_PLACEMENTS

    assert "cpu-transient-grad-bulk" in _ALLOWED_EXPERT_PLACEMENTS
