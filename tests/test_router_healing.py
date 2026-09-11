"""Regression tests for `chowder.router_healing`.

Fast, CPU-only, no real model/checkpoint -- a tiny real `nn.Module` tree
with parameter names shaped like the real Qwen3_5MoE layout (`mlp.gate`,
`mlp.shared_expert_gate`, `mlp.experts.{gate_up_proj,down_proj}`, plus an
unrelated attention Linear) is enough to verify the freeze logic touches
exactly the right parameters and nothing else.
"""
from __future__ import annotations

import pytest

from chowder.router_healing import (
    RouterHealingError,
    freeze_for_router_healing,
    select_trainable_parameter_names,
)

torch = pytest.importorskip("torch")
nn = torch.nn


class _FakeMlp(nn.Module):
    def __init__(self, hidden: int, num_experts: int, moe_intermediate: int) -> None:
        super().__init__()
        self.gate = nn.Module()
        self.gate.weight = nn.Parameter(torch.zeros(num_experts, hidden))
        self.experts = nn.Module()
        self.experts.gate_up_proj = nn.Parameter(torch.zeros(num_experts, 2 * moe_intermediate, hidden))
        self.experts.down_proj = nn.Parameter(torch.zeros(num_experts, hidden, moe_intermediate))
        self.shared_expert_gate = nn.Linear(hidden, 1, bias=False)


class _FakeLayer(nn.Module):
    def __init__(self, hidden: int, num_experts: int, moe_intermediate: int) -> None:
        super().__init__()
        self.self_attn = nn.Linear(hidden, hidden, bias=False)
        self.mlp = _FakeMlp(hidden, num_experts, moe_intermediate)


class _FakeModel(nn.Module):
    def __init__(self, num_layers: int = 3, hidden: int = 8, num_experts: int = 4, moe_intermediate: int = 2) -> None:
        super().__init__()
        self.layers = nn.ModuleList(_FakeLayer(hidden, num_experts, moe_intermediate) for _ in range(num_layers))


def test_select_trainable_parameter_names_picks_only_gate_and_shared_gate():
    model = _FakeModel(num_layers=2)
    names = select_trainable_parameter_names(model.named_parameters())
    assert len(names) == 4  # 2 layers x (gate.weight + shared_expert_gate.weight)
    for name in names:
        assert name.endswith("mlp.gate.weight") or name.endswith("mlp.shared_expert_gate.weight")


def test_freeze_for_router_healing_sets_requires_grad_correctly():
    model = _FakeModel(num_layers=3)
    summary = freeze_for_router_healing(model)

    assert summary.layers_with_trainable_gate == 3
    assert summary.layers_with_trainable_shared_expert_gate == 3
    assert len(summary.trainable_param_names) == 6

    trainable_set = set(summary.trainable_param_names)
    seen_trainable = 0
    seen_frozen = 0
    for name, param in model.named_parameters():
        if name in trainable_set:
            assert param.requires_grad, f"{name} should be trainable"
            seen_trainable += 1
        else:
            assert not param.requires_grad, f"{name} should be frozen"
            seen_frozen += 1
    assert seen_trainable == 6
    # 3 layers x (self_attn + experts.gate_up_proj + experts.down_proj) = 9
    assert seen_frozen == 9

    # expert weights (the bulk of the model) are explicitly among the frozen set
    assert any(n.endswith("experts.gate_up_proj") for n, p in model.named_parameters() if not p.requires_grad)
    assert any(n.endswith("experts.down_proj") for n, p in model.named_parameters() if not p.requires_grad)


def test_freeze_for_router_healing_counts_match_real_parameter_numel():
    model = _FakeModel(num_layers=1, hidden=8, num_experts=4, moe_intermediate=2)
    summary = freeze_for_router_healing(model)
    # gate.weight: 4x8=32, shared_expert_gate.weight: 1x8=8 -> 40 trainable
    assert summary.trainable_param_count == 32 + 8
    # self_attn: 8x8=64, gate_up_proj: 4x(2x2)x8=128, down_proj: 4x8x2=64 -> 256 frozen
    assert summary.frozen_param_count == 64 + 128 + 64


def test_freeze_for_router_healing_hard_stops_on_no_match():
    class _EmptyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.unrelated = nn.Linear(4, 4)

    with pytest.raises(RouterHealingError):
        freeze_for_router_healing(_EmptyModel())
