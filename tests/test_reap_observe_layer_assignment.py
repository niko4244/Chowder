"""CPU-only parity for assigning actual Qwen MoE decoder checkpoint storage."""

import weakref

import pytest

torch = pytest.importorskip("torch")
M = pytest.importorskip("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe")
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig

from chowder_batch.reap_observe import (
    _load_layer_without_duplicate_weights,
    observing_experts_forward,
)


def _assert_finite_exact_parity(reference, assigned):
    assert torch.isfinite(reference).all(), "reference decoder output is nonfinite"
    assert torch.isfinite(assigned).all(), "assigned decoder output is nonfinite"
    torch.testing.assert_close(assigned, reference, rtol=0, atol=0)


@pytest.mark.parametrize("block_type", ["linear_attention", "full_attention"])
def test_qwen_moe_bf16_forward_parity_and_shared_weight_storage(block_type):
    tc = Qwen3_5MoeTextConfig(
        vocab_size=64, hidden_size=32, num_hidden_layers=1, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, linear_key_head_dim=8,
        linear_value_head_dim=8, linear_num_key_heads=2,
        linear_num_value_heads=2, moe_intermediate_size=16,
        shared_expert_intermediate_size=16, num_experts=4,
        num_experts_per_tok=2, layer_types=[block_type],
    )
    # A bare decoder has torch.empty expert projections. Construct the tiny
    # model so its post_init() initializes those weights like a real checkpoint.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261023)
        original = M.Qwen3_5MoeTextModel(tc).layers[0].to(torch.bfloat16).eval()
    checkpoint = {key: value.clone() for key, value in original.state_dict().items()}
    assert checkpoint
    assert all(torch.isfinite(tensor).all() for tensor in checkpoint.values())
    assert checkpoint["mlp.experts.gate_up_proj"].abs().sum() > 0
    assigned = _load_layer_without_duplicate_weights(
        M.Qwen3_5MoeDecoderLayer, tc, 0, torch.bfloat16, checkpoint,
    ).eval()

    for key, tensor in assigned.state_dict().items():
        assert torch.equal(tensor, checkpoint[key]), key
        assert tensor.dtype == checkpoint[key].dtype
        assert tensor.data_ptr() == checkpoint[key].data_ptr(), key

    x = torch.randn(1, 7, tc.hidden_size, dtype=torch.bfloat16)
    pos = torch.arange(x.shape[1]).view(1, -1)
    rotary = M.Qwen3_5MoeTextRotaryEmbedding(tc)
    pe = rotary(x, pos[None].expand(3, 1, -1))
    if block_type == "full_attention":
        tc._attn_implementation = "sdpa"
        mask = M.create_causal_mask(config=tc, inputs_embeds=x, attention_mask=None,
                                    past_key_values=None, position_ids=pos)
    else:
        mask = M.create_recurrent_attention_mask(config=tc, inputs_embeds=x, attention_mask=None)
    with torch.inference_mode():
        expected = original(x, position_embeddings=pe, attention_mask=mask, position_ids=pos)
        got = assigned(x, position_embeddings=pe, attention_mask=mask, position_ids=pos)
    _assert_finite_exact_parity(expected, got)


def test_matching_nan_outputs_are_not_accepted_as_parity():
    poisoned = torch.full((1, 7, 32), float("nan"), dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match="reference decoder output is nonfinite"):
        _assert_finite_exact_parity(poisoned, poisoned.clone())


def test_assignment_refuses_implicit_checkpoint_dtype_change():
    class SmallLayer(torch.nn.Module):
        def __init__(self, config, layer_id):
            super().__init__()
            self.linear = torch.nn.Linear(3, 3, bias=False)

    with pytest.raises(TypeError, match="dtype mismatch"):
        _load_layer_without_duplicate_weights(
            SmallLayer, None, 0, torch.bfloat16,
            {"linear.weight": torch.randn(3, 3, dtype=torch.float32)},
        )


def test_observation_hook_does_not_retain_experts_after_release():
    class SmallExperts(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_experts = 2
            self.gate_up_proj = torch.nn.Parameter(torch.randn(2, 8, 4))
            self.down_proj = torch.nn.Parameter(torch.randn(2, 4, 4))
            self.act_fn = torch.nn.functional.silu

    experts = SmallExperts()
    stats = {key: torch.zeros(2, dtype=torch.float64)
             for key in ("count", "gate_sum", "norm_sum", "reap_sum")}
    experts.forward = observing_experts_forward(experts, stats)
    with torch.inference_mode():
        output = experts(torch.randn(3, 4),
                         torch.tensor([[0, 1], [1, 0], [0, 1]]),
                         torch.tensor([[0.6, 0.4], [0.7, 0.3], [0.5, 0.5]]))
    assert output.shape == (3, 4)
    assert torch.isfinite(output).all()
    assert torch.equal(stats["count"], torch.tensor([3, 3]))

    reference = weakref.ref(experts)
    del experts
    # No forced cyclic GC: the observer must release the layer at the point
    # the 40-layer loop runs ``del layer`` before loading the next layer.
    assert reference() is None
