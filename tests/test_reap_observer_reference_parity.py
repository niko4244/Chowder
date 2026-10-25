"""Bounded CPU numeric reference checks; NOT a claim of CUDA/production parity.

Only tiny randomly initialized Qwen3.5 MoE layers are materialized. The frozen
704-sequence BF16 checkpoint and original 71.9GB weights are never loaded.
"""

from __future__ import annotations

import copy
import math

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional
M = pytest.importorskip("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe")
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig

from chowder_batch.reap_observe import (
    _load_layer_without_duplicate_weights,
    _next_token_nll_chunked,
    observing_experts_forward,
)

# Predeclared tolerances: the reference and observer share BF16 kernels, so
# decoder activation equality must be bit exact on CPU. FP32 norm reductions
# may regroup, so expert statistic sums tolerate relative 2e-5, absolute 1e-4.
_STATS_RTOL, _STATS_ATOL = 2e-5, 1e-4
_LOSS_RTOL, _LOSS_ATOL = 1e-6, 2e-3


def _tiny_config(block_type: str = "full_attention") -> Qwen3_5MoeTextConfig:
    return Qwen3_5MoeTextConfig(
        vocab_size=71, hidden_size=32, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=2,
        moe_intermediate_size=16, shared_expert_intermediate_size=16,
        num_experts=4, num_experts_per_tok=2, layer_types=[block_type],
    )


def _stats_reference(experts, hidden, ids, weights):
    """Independent token/slot-wise REAP definition (not the observer loop)."""
    result = {name: torch.zeros(experts.num_experts, dtype=torch.float64)
              for name in ("count", "gate_sum", "norm_sum", "reap_sum")}
    with torch.inference_mode():
        for token in range(ids.shape[0]):
            for slot in range(ids.shape[1]):
                expert_id = int(ids[token, slot])
                gate, up = F.linear(
                    hidden[token: token + 1], experts.gate_up_proj[expert_id],
                ).chunk(2, dim=-1)
                expert_out = F.linear(
                    experts.act_fn(gate) * up, experts.down_proj[expert_id],
                )
                gate_weight = weights[token, slot].float()
                norm = expert_out.float().norm(dim=-1).squeeze(0)
                result["count"][expert_id] += 1
                result["gate_sum"][expert_id] += gate_weight
                result["norm_sum"][expert_id] += norm
                result["reap_sum"][expert_id] += gate_weight * norm
    return result


def test_observer_expert_outputs_and_saliency_match_reference_bf16_cpu():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261025)
        config = _tiny_config()
        reference = M.Qwen3_5MoeTextModel(config).layers[0].mlp.experts.to(torch.bfloat16).eval()
        instrumented = copy.deepcopy(reference)
        hidden = torch.randn(9, config.hidden_size).to(torch.bfloat16)
        ids = torch.tensor([[0, 1], [3, 2], [1, 3], [0, 2], [3, 1],
                            [2, 0], [1, 0], [3, 2], [0, 3]])
        weights = F.softmax(torch.randn(9, 2), dim=-1).to(torch.bfloat16)
    stats = {name: torch.zeros(config.num_experts, dtype=torch.float64)
             for name in ("count", "gate_sum", "norm_sum", "reap_sum")}
    instrumented.forward = observing_experts_forward(instrumented, stats)
    with torch.inference_mode():
        target = reference(hidden, ids, weights)
        observed = instrumented(hidden, ids, weights)
        independent = _stats_reference(reference, hidden, ids, weights)
    assert torch.isfinite(observed).all()
    torch.testing.assert_close(observed, target, rtol=0, atol=0)
    for name in stats:
        assert torch.isfinite(stats[name]).all(), name
        torch.testing.assert_close(
            stats[name], independent[name], rtol=_STATS_RTOL, atol=_STATS_ATOL,
        )
    assert stats["count"].sum() == hidden.shape[0] * 2


@pytest.mark.parametrize("block_type", ["linear_attention", "full_attention"])
def test_instrumented_reference_decoder_and_final_nll_bf16_cpu(block_type):
    config = _tiny_config(block_type)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261026)
        reference = M.Qwen3_5MoeTextModel(config).layers[0].to(torch.bfloat16).eval()
        assigned = _load_layer_without_duplicate_weights(
            M.Qwen3_5MoeDecoderLayer, config, 0, torch.bfloat16,
            {k: v.clone() for k, v in reference.state_dict().items()},
        ).eval()
        hidden = torch.randn(1, 11, config.hidden_size).to(torch.bfloat16)
        head = torch.randn(config.vocab_size, config.hidden_size).to(torch.bfloat16)
        targets = torch.randint(config.vocab_size, (11,))
    stats = {name: torch.zeros(config.num_experts, dtype=torch.float64)
             for name in ("count", "gate_sum", "norm_sum", "reap_sum")}
    assigned.mlp.experts.forward = observing_experts_forward(assigned.mlp.experts, stats)
    positions = torch.arange(hidden.shape[1]).view(1, -1)
    rotary = M.Qwen3_5MoeTextRotaryEmbedding(config)
    position_embeddings = rotary(hidden, positions[None].expand(3, 1, -1))
    if block_type == "full_attention":
        config._attn_implementation = "sdpa"
        mask = M.create_causal_mask(
            config=config, inputs_embeds=hidden, attention_mask=None,
            past_key_values=None, position_ids=positions,
        )
    else:
        mask = M.create_recurrent_attention_mask(
            config=config, inputs_embeds=hidden, attention_mask=None,
        )
    with torch.inference_mode():
        target = reference(
            hidden, position_embeddings=position_embeddings, attention_mask=mask,
            position_ids=positions,
        )
        observed = assigned(
            hidden, position_embeddings=position_embeddings, attention_mask=mask,
            position_ids=positions,
        )
    assert torch.isfinite(target).all()
    torch.testing.assert_close(observed, target, rtol=0, atol=0)
    assert int(stats["count"].sum()) == targets.numel() * config.num_experts_per_tok
    norm = M.Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(torch.bfloat16)
    with torch.inference_mode():
        expected = F.cross_entropy(
            F.linear(norm(target)[0, :-1], head).float(),
            targets[1:], reduction="sum",
        ).item()
    measured, positions_scored = _next_token_nll_chunked(
        norm, observed, targets, head, chunk_positions=3,
    )
    assert positions_scored == targets.numel() - 1
    assert math.isfinite(measured)
    assert measured == pytest.approx(expected, rel=_LOSS_RTOL, abs=_LOSS_ATOL)
