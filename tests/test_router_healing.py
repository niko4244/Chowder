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
    ROUTER_ONLY_SUFFIXES,
    RouterHealingError,
    assert_trainable_gradients_reachable,
    freeze_for_router_healing,
    select_trainable_parameter_names,
)
from chowder.trainability import (
    GRAD_NONZERO,
    TrainabilityProbe,
    assert_router_only_scope,
    utilization_by_expert,
)

torch = pytest.importorskip("torch")


def _tiny_qwen3_moe():
    """A real Qwen3 MoE, E=4 / k=2, small enough for CPU in ~0.1s.

    A real model rather than a parameter-name stub, because the whole point of
    P5 is that `requires_grad` and structural checks are not evidence: a real
    forward/backward/update is required, and the plan asks for exactly this
    shape (E=4, k=2). Runs wherever transformers is installed (the real-CPU CI
    job and any dev box); there is no silent skip on a machine that has it.
    """
    pytest.importorskip("transformers")
    from transformers.models.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM

    torch.manual_seed(0)
    config = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=8,
        moe_intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=64,
    )
    return Qwen3MoeForCausalLM(config).float()


# --- P5: real autograd on a real MoE, not a shape stub -----------------------


def test_real_moe_router_only_training_shows_a_real_gradient_and_a_real_update():
    model = _tiny_qwen3_moe()
    summary = freeze_for_router_healing(model, suffixes=ROUTER_ONLY_SUFFIXES)
    gate_names = list(summary.trainable_param_names)
    assert gate_names == [
        "model.layers.0.mlp.gate.weight",
        "model.layers.1.mlp.gate.weight",
    ]

    frozen = [name for name, _ in model.named_parameters() if name not in set(gate_names)]
    probe = TrainabilityProbe(model, gate_names, window_steps=2, frozen_names=frozen)
    optimizer = torch.optim.SGD(
        [param for name, param in model.named_parameters() if name in set(gate_names)],
        lr=0.05,
    )

    for step in range(2):
        ids = torch.randint(0, 64, (1, 12))
        optimizer.zero_grad()
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        probe.record_gradients(step)  # after backward, before the step
        optimizer.step()
        probe.record_update(step)

    report = probe.assert_qualified()
    for name in gate_names:
        assert GRAD_NONZERO in report["components"][name]["gradient_states"]
        assert report["components"][name]["update_steps"]

    # the freeze policy is verified, not assumed: every expert and every frozen
    # attention weight is byte-identical after two real optimizer steps
    frozen_report = probe.assert_frozen_unchanged()
    assert frozen_report["frozen_parameters"] == len(frozen)
    assert frozen_report["digest_strategy"] == ["full"]


def test_real_moe_router_only_scope_is_exact_and_experts_are_refused():
    model = _tiny_qwen3_moe()
    gates = [
        name
        for name, _ in model.named_parameters()
        if name.endswith("mlp.gate.weight")
    ]
    report = assert_router_only_scope(gates, model)
    assert report["router_count"] == 2
    assert report["architecture_router_count"] == 2
    with pytest.raises(Exception, match="exactly the router gates"):
        assert_router_only_scope(gates + ["model.layers.0.mlp.experts.down_proj"], model)


def test_real_moe_expert_utilisation_is_reported_separately_from_reachability():
    """A routing table and a gradient are different evidence. This asserts the
    separation on a real router: utilisation is measured and reported, and says
    nothing about whether any tensor can learn."""
    model = _tiny_qwen3_moe()
    gate = model.model.layers[0].mlp.gate  # Qwen3MoeTopKRouter: logits = x @ weight.T
    hidden = torch.randn(8, gate.weight.shape[1])
    logits = torch.nn.functional.linear(hidden, gate.weight)
    top = logits.argmax(dim=-1)
    counts = [int((top == index).sum()) for index in range(4)]
    report = utilization_by_expert({"0": counts})
    assert report["status"] == "measured"
    assert sum(report["layers"]["0"]["counts"]) == 8
    assert "not evidence of per-tensor reachability" in report["note"]
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


class _ZeroSharedMlp(nn.Module):
    """The converted shape: shared expert written as zeros, gate zero too."""

    def __init__(self, hidden: int, num_experts: int, moe_intermediate: int, *, zero_shared: bool) -> None:
        super().__init__()
        self.gate = nn.Module()
        self.gate.weight = nn.Parameter(torch.zeros(num_experts, hidden))
        self.experts = nn.Module()
        self.experts.gate_up_proj = nn.Parameter(torch.zeros(num_experts, 2 * moe_intermediate, hidden))
        self.experts.down_proj = nn.Parameter(torch.zeros(num_experts, hidden, moe_intermediate))
        self.shared_expert = nn.Module()
        for leaf, shape in (
            ("gate_proj", (moe_intermediate, hidden)),
            ("up_proj", (moe_intermediate, hidden)),
            ("down_proj", (hidden, moe_intermediate)),
        ):
            lin = nn.Linear(shape[1], shape[0], bias=False)
            if zero_shared:
                nn.init.zeros_(lin.weight)
            setattr(self.shared_expert, leaf, lin)
        self.shared_expert_gate = nn.Linear(hidden, 1, bias=False)


class _ZeroSharedLayer(nn.Module):
    def __init__(self, **kw) -> None:
        super().__init__()
        self.mlp = _ZeroSharedMlp(**kw)


class _ZeroSharedModel(nn.Module):
    def __init__(self, *, zero_shared: bool, layers: int = 2) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            _ZeroSharedLayer(hidden=8, num_experts=4, moe_intermediate=2, zero_shared=zero_shared)
            for _ in range(layers)
        )


def test_zero_init_shared_expert_makes_its_gate_untrainable_and_is_refused():
    """The real defect: a gate multiplied by an all-zero frozen shared expert
    has exactly zero derivative, so training it is pure waste. Verified by CPU
    probe (grad 0.0 as converted vs 0.16 with a non-zero shared expert)."""
    model = _ZeroSharedModel(zero_shared=True)
    with pytest.raises(RouterHealingError) as excinfo:
        freeze_for_router_healing(model)
    assert "cannot receive gradient" in str(excinfo.value)
    assert "shared_expert" in str(excinfo.value)


def test_nonzero_shared_expert_passes_reachability():
    model = _ZeroSharedModel(zero_shared=False)
    summary = freeze_for_router_healing(model)
    assert summary.layers_with_trainable_shared_expert_gate == 2


def test_reachability_check_can_be_explicitly_bypassed():
    """Escape hatch for deliberately measuring the dead configuration, but it
    must be asked for -- the default refuses."""
    model = _ZeroSharedModel(zero_shared=True)
    summary = freeze_for_router_healing(model, require_reachable=False)
    assert summary.layers_with_trainable_shared_expert_gate == 2
