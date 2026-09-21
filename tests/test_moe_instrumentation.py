"""Fast, deterministic tests for real MoE router/expert instrumentation.

The fake router/experts modules below reproduce -- line for line -- the
forward math verified by reading the actual installed
transformers==5.16.1 source for Qwen3MoeTopKRouter/Qwen3MoeExperts (and the
identical Qwen3_5Moe/Olmoe implementations): a router returning
``(router_logits, router_scores, router_indices)`` and a fused expert module
whose forward loops ``gate, up = linear(x, gate_up_proj[e]).chunk(2, -1);
act_fn(gate) * up; linear(..., down_proj[e])`` per selected expert. Using the
same math (with an identity activation, for hand-checkable arithmetic) means
these tests validate the recorder's bookkeeping against a faithful replica of
the real shape, not an invented one.
"""

from __future__ import annotations

import json
import math

import pytest

from chowder.moe_instrumentation import (
    MoeArchitectureAuditError,
    audit_moe_architecture,
    run_calibration,
    write_expert_importance_jsonl,
)
from chowder.moe_planning import build_uniform_pruning_plan

torch = pytest.importorskip("torch")
nn = torch.nn


class _FakeRouter(nn.Module):
    def __init__(self, weight: torch.Tensor, top_k: int, norm_topk_prob: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.num_experts = weight.shape[0]
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.hidden_dim = weight.shape[1]

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = nn.functional.linear(hidden_states, self.weight)
        router_probs = nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
        router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            router_top_value = router_top_value / router_top_value.sum(dim=-1, keepdim=True)
        router_top_value = router_top_value.to(router_logits.dtype)
        return router_logits, router_top_value, router_indices


class _FakeExperts(nn.Module):
    def __init__(self, gate_up_proj: torch.Tensor, down_proj: torch.Tensor) -> None:
        super().__init__()
        self.gate_up_proj = nn.Parameter(gate_up_proj, requires_grad=False)
        self.down_proj = nn.Parameter(down_proj, requires_grad=False)
        self.num_experts = gate_up_proj.shape[0]
        self.act_fn = lambda x: x  # identity: keeps expected values hand-checkable

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
        for expert_idx in range(self.num_experts):
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            if token_idx.numel() == 0:
                continue
            current_state = hidden_states[token_idx]
            gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current = self.act_fn(gate) * up
            current = nn.functional.linear(current, self.down_proj[expert_idx])
            current = current * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current.to(final_hidden_states.dtype))
        return final_hidden_states


class _FakeSparseMoeBlock(nn.Module):
    def __init__(self, gate: _FakeRouter, experts: _FakeExperts) -> None:
        super().__init__()
        self.gate = gate
        self.experts = experts

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, seq, hidden_dim = hidden_states.shape
        flat = hidden_states.view(-1, hidden_dim)
        _, weights, indices = self.gate(flat)
        out = self.experts(flat, indices, weights)
        return out.reshape(batch, seq, hidden_dim)


class _FakeDenseMlp(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_states)


class _FakeLayer(nn.Module):
    def __init__(self, mlp: nn.Module) -> None:
        super().__init__()
        self.mlp = mlp

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.mlp(hidden_states)


class _FakeInnerModel(nn.Module):
    def __init__(self, layers: list[nn.Module]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)


class _FakeConfig:
    model_type = "fake_moe"


class _FakeModel(nn.Module):
    """A minimal stand-in for an HF causal LM: ``model.model.layers`` plus
    ``model.config``. ``forward`` treats each input scalar as a token id that
    directly *is* the hidden-state vector (via the test tokenizer below), so
    tests can hand-pick exact hidden states without a real embedding table.
    """

    def __init__(self, layers: list[nn.Module]) -> None:
        super().__init__()
        self.model = _FakeInnerModel(layers)
        self.config = _FakeConfig()

    def forward(self, input_ids: torch.Tensor):
        hidden_states = input_ids.unsqueeze(1)  # [batch=1, seq=1, hidden_dim]
        for layer in self.model.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class _HiddenStateTokenizer:
    """Maps a text label directly to the hidden-state vector a test wants."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors

    def __call__(self, text, return_tensors=None, truncation=None, max_length=None):
        return {"input_ids": torch.tensor([self._vectors[text]])}


def _two_expert_block() -> _FakeSparseMoeBlock:
    # gate.weight rows are one-hot so token [2,0] routes to expert 0 and
    # [0,3] routes to expert 1, unambiguously.
    gate = _FakeRouter(weight=torch.tensor([[1.0, 0.0], [0.0, 1.0]]), top_k=1)
    gate_up_proj = torch.stack(
        [
            torch.tensor([[1.0, 1.0], [1.0, -1.0]]),  # expert 0
            torch.tensor([[1.0, 1.0], [1.0, 1.0]]),  # expert 1
        ]
    )
    down_proj = torch.stack(
        [
            torch.tensor([[1.0], [1.0]]),  # expert 0
            torch.tensor([[2.0], [0.0]]),  # expert 1
        ]
    )
    experts = _FakeExperts(gate_up_proj, down_proj)
    return _FakeSparseMoeBlock(gate, experts)


def test_real_recorder_produces_exact_hand_checked_stats_for_one_moe_layer():
    model = _FakeModel([_FakeLayer(_two_expert_block())])
    tokenizer = _HiddenStateTokenizer({"A": [2.0, 0.0], "B": [0.0, 3.0]})

    audit, records = run_calibration(model, tokenizer, ["A", "B"])

    assert audit.model_type == "fake_moe"
    assert audit.num_hidden_layers == 1
    assert audit.dense_layer_indices == ()
    assert len(audit.moe_layers) == 1
    assert audit.moe_layers[0].num_experts == 2
    assert audit.moe_layers[0].num_experts_per_tok == 1

    by_expert = {(r.layer, r.expert): r for r in records}
    expert0 = by_expert[(0, 0)]
    assert expert0.selected_tokens == 1
    assert expert0.router_mass == pytest.approx(1.0)
    assert expert0.gated_activation == pytest.approx(4.0)
    assert expert0.output_norm == pytest.approx(math.sqrt(32.0))

    expert1 = by_expert[(0, 1)]
    assert expert1.selected_tokens == 1
    assert expert1.router_mass == pytest.approx(1.0)
    assert expert1.gated_activation == pytest.approx(9.0)
    assert expert1.output_norm == pytest.approx(18.0)


def test_stats_accumulate_across_multiple_calibration_examples_routed_to_same_expert():
    model = _FakeModel([_FakeLayer(_two_expert_block())])
    tokenizer = _HiddenStateTokenizer({"A": [2.0, 0.0]})

    _, records = run_calibration(model, tokenizer, ["A", "A", "A"])

    by_expert = {(r.layer, r.expert): r for r in records}
    assert by_expert[(0, 0)].selected_tokens == 3
    assert by_expert[(0, 0)].router_mass == pytest.approx(3.0)
    assert by_expert[(0, 0)].gated_activation == pytest.approx(12.0)
    # expert 1 never selected but must still appear, zeroed, for the pruning
    # planner's per-layer accounting to remain correct.
    assert by_expert[(0, 1)].selected_tokens == 0
    assert by_expert[(0, 1)].router_mass == 0.0


def test_dense_layers_are_skipped_not_misclassified():
    model = _FakeModel(
        [_FakeLayer(_FakeDenseMlp(2)), _FakeLayer(_two_expert_block()), _FakeLayer(_FakeDenseMlp(2))]
    )
    audit = audit_moe_architecture(model)
    assert audit.dense_layer_indices == (0, 2)
    assert [layer.layer for layer in audit.moe_layers] == [1]
    assert audit.is_uniform is False


def test_audit_hard_stops_when_no_layer_matches_the_verified_moe_shape():
    model = _FakeModel([_FakeLayer(_FakeDenseMlp(2))])
    with pytest.raises(MoeArchitectureAuditError, match="no decoder layer matched"):
        audit_moe_architecture(model)


def test_audit_hard_stops_when_model_has_no_recognizable_layer_list():
    class _Bare(nn.Module):
        config = _FakeConfig()

    with pytest.raises(MoeArchitectureAuditError, match="cannot locate model.model.layers"):
        audit_moe_architecture(_Bare())


def test_expert_importance_jsonl_round_trips_and_feeds_the_real_pruning_planner(tmp_path):
    model = _FakeModel([_FakeLayer(_two_expert_block())])
    tokenizer = _HiddenStateTokenizer({"A": [2.0, 0.0], "B": [0.0, 3.0]})
    audit, records = run_calibration(model, tokenizer, ["A", "B"])

    out_path = tmp_path / "expert_importance.jsonl"
    write_expert_importance_jsonl(
        out_path,
        audit=audit,
        records=records,
        model_source="fake://unit-test",
        calibration_texts=["A", "B"],
        transformers_version="0.0.0-test",
    )

    lines = out_path.read_text(encoding="utf-8").splitlines()
    provenance = json.loads(lines[0])["provenance"]
    assert provenance["schema_version"] == 1
    assert provenance["model_source"] == "fake://unit-test"
    assert provenance["moe_layer_count"] == 1
    assert provenance["calibration_example_count"] == 2

    parsed_records = [json.loads(line) for line in lines[1:]]
    assert len(parsed_records) == 2

    plan = build_uniform_pruning_plan(records, retention_fraction=0.5, minimum_survivors_per_layer=1)
    assert plan.layers[0].total_experts == 2
    assert len(plan.layers[0].keep_experts) == 1
    # expert 1 has higher gated_activation/output_norm evidence than expert 0
    assert plan.layers[0].keep_experts == (1,)
