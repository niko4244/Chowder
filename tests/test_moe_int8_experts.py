"""int8 frozen experts (``cpu-transient-grad-int8``): the option-B 16 GB path.

CPU only: the checkpoint is a tiny synthetic Qwen3-MoE written to ``tmp_path`` as
safetensors shards, so no real model is loaded and no GPU is touched. What is
proven here is the mechanism -- per-row int8 quantisation, a loader that never
holds the bf16 experts, and a forward/backward that tracks the bf16 bulk path --
not the GPU speed or memory of the placement, which its preregistration still
requires to be measured on the real artifact.
"""

from __future__ import annotations

import json
import os
import weakref
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import torch.nn.functional as F  # noqa: E402

pytest.importorskip("safetensors")
transformers = pytest.importorskip("transformers")

from chowder.backends import moe_int8_experts as mie  # noqa: E402
from chowder.backends import moe_transient_grad as mtg  # noqa: E402
from chowder.backends.transformers_peft import TransformersPeftRunSpec  # noqa: E402

HIDDEN, INTER, NUM_EXPERTS, LAYERS, TOKENS, TOP_K = 16, 4, 6, 2, 10, 2
TOLERANCE = 2e-2


@pytest.fixture(autouse=True)
def _no_gpu(monkeypatch):
    """Every loader call in this file resolves to the CPU."""
    monkeypatch.setenv("CHOWDER_NO_GPU", "1")


def _config(*, tie_word_embeddings: bool = False):
    from transformers import Qwen3MoeConfig

    return Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=HIDDEN,
        intermediate_size=8,
        moe_intermediate_size=INTER,
        num_hidden_layers=LAYERS,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=TOP_K,
        decoder_sparse_step=1,
        max_position_embeddings=32,
        head_dim=HIDDEN // 2,
        initializer_range=0.02,
        tie_word_embeddings=tie_word_embeddings,
    )


def _checkpoint(tmp_path: Path, *, tie_word_embeddings: bool = False, name: str = "tiny") -> Path:
    """A tiny sharded MoE checkpoint on disk: a config, a generation config and
    at least two safetensors shards."""
    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(_config(tie_word_embeddings=tie_word_embeddings))
    base = tmp_path / name
    model.save_pretrained(base, max_shard_size="20KB")
    shards = sorted(base.glob("*.safetensors"))
    assert len(shards) >= 2, f"the fixture must shard the checkpoint, got {[p.name for p in shards]}"
    return base


def _routing(seed: int = 11, experts: tuple[int, ...] = tuple(range(NUM_EXPERTS))) -> torch.Tensor:
    choices = torch.tensor(experts)
    generator = torch.Generator().manual_seed(seed)
    rows = []
    for _ in range(TOKENS):
        rows.append(choices[torch.randperm(len(choices), generator=generator)[:TOP_K]])
    return torch.stack(rows)


def _inputs(seed: int = 1, experts: tuple[int, ...] = tuple(range(NUM_EXPERTS))):
    generator = torch.Generator().manual_seed(seed)
    hidden = torch.randn(TOKENS, HIDDEN, generator=generator, requires_grad=True)
    weights = (torch.randn(TOKENS, TOP_K, generator=generator) * 0.5).requires_grad_(True)
    return hidden, _routing(seed=seed + 10, experts=experts), weights


def _experts_module(model, layer: int = 0):
    for name, module in model.named_modules():
        if name.endswith(f"layers.{layer}.mlp.experts"):
            return module
    raise AssertionError("no experts module on the model")


def _bf16_bulk_model(base: Path):
    """The bulk path on the CPU: bf16 (here fp32) experts in RAM, one copy per
    layer per pass."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(base)
    mtg.prepare_split_model(model, device_type="cpu", bulk=True)
    return model


def _int8_model(base: Path):
    model, census = mie.load_model_int8_experts(base)
    placement = mie.prepare_int8_placement(model, device_type="cpu")
    return model, census, placement


def _forward_and_grads(module):
    hidden, index, weights = _inputs()
    out = module(hidden, index, weights)
    out.square().sum().backward()
    return out.detach(), hidden.grad.detach(), weights.grad.detach()


def _relative(got: torch.Tensor, want: torch.Tensor) -> float:
    return ((got - want).abs().max() / want.abs().max().clamp_min(1e-12)).item()


# --- quantisation -------------------------------------------------------------


def test_quantize_round_trip_stays_within_half_a_step_of_the_row_absmax():
    generator = torch.Generator().manual_seed(3)
    # Rows of very different magnitudes: the scale must be per output row, not
    # per tensor, or the small rows would quantise to zero.
    weight = torch.randn(5, 32, generator=generator) * torch.linspace(0.01, 100.0, 5).unsqueeze(1)

    q, scale = mie.quantize_rows(weight)
    dequantized = mie.dequantize_rows(q, scale, torch.float32)

    assert q.dtype == torch.int8
    assert int(q.abs().max()) <= mie.INT8_MAX
    assert scale.shape == (5, 1) and scale.dtype == torch.float32
    row_absmax = weight.abs().amax(dim=-1, keepdim=True)
    assert torch.all((dequantized - weight).abs() <= row_absmax / mie.INT8_MAX)
    # Per-row, not per-tensor: each row's scale tracks its own absmax.
    assert torch.allclose(scale, row_absmax / mie.INT8_MAX, rtol=1e-6)
    # A tiny row survives at full precision instead of collapsing to zero.
    assert dequantized[0].abs().max() > 0


def test_an_all_zero_row_quantizes_to_zero_with_a_finite_scale():
    q, scale = mie.quantize_rows(torch.zeros(3, 8), scale_dtype=torch.bfloat16)
    assert (q == 0).all()
    assert torch.isfinite(scale).all() and scale.dtype == torch.bfloat16
    assert (mie.dequantize_rows(q, scale, torch.float32) == 0).all()


def test_dequantize_runs_wherever_the_int8_lives():
    q = torch.tensor([[-127, 0, 127], [1, -1, 0]], dtype=torch.int8)
    scale = torch.tensor([[2.0], [0.5]], dtype=torch.float32)
    out = mie.dequantize_rows(q, scale, torch.float32)
    assert torch.equal(out, torch.tensor([[-254.0, 0.0, 254.0], [0.5, -0.5, 0.0]]))
    assert mie.dequantize_rows(q, scale, torch.bfloat16).dtype == torch.bfloat16


# --- the loader ---------------------------------------------------------------


def test_loader_matches_from_pretrained_for_every_non_expert_parameter_and_buffer(tmp_path):
    from transformers import AutoModelForCausalLM

    base = _checkpoint(tmp_path)
    reference = AutoModelForCausalLM.from_pretrained(base)
    model, census = mie.load_model_int8_experts(base)

    reference_params = dict(reference.named_parameters())
    loaded_params = dict(model.named_parameters())
    missing = sorted(set(reference_params) - set(loaded_params))
    assert missing and all(mtg.EXPERT_PARAM_INFIX in name for name in missing), missing
    for name, parameter in loaded_params.items():
        assert torch.equal(parameter, reference_params[name]), name
        assert parameter.dtype == reference_params[name].dtype, name

    reference_buffers = {
        name: buffer for name, buffer in reference.named_buffers() if mtg.EXPERT_PARAM_INFIX not in name
    }
    loaded_buffers = {
        name: buffer for name, buffer in model.named_buffers() if mtg.EXPERT_PARAM_INFIX not in name
    }
    assert set(loaded_buffers) == set(reference_buffers)
    for name, buffer in loaded_buffers.items():
        assert torch.equal(buffer, reference_buffers[name]), name

    assert model.config.to_dict() == reference.config.to_dict()
    assert census["non_expert_params"] == len(loaded_params)
    assert census["non_expert_params_on_device"] == len(loaded_params)
    assert census["device"] == "cpu"


def test_loader_ties_the_head_exactly_as_from_pretrained_does(tmp_path):
    from transformers import AutoModelForCausalLM

    base = _checkpoint(tmp_path, tie_word_embeddings=True, name="tied")
    reference = AutoModelForCausalLM.from_pretrained(base)
    model, _ = mie.load_model_int8_experts(base)

    assert reference.lm_head.weight is reference.model.embed_tokens.weight
    assert model.lm_head.weight is model.model.embed_tokens.weight
    assert torch.equal(model.lm_head.weight, reference.lm_head.weight)


def test_no_bf16_or_fp32_expert_weight_survives_the_load(tmp_path):
    base = _checkpoint(tmp_path)
    model, census = mie.load_model_int8_experts(base)

    expert_params = [name for name, _ in model.named_parameters() if mtg.EXPERT_PARAM_INFIX in name]
    assert expert_params == []
    expert_buffers = [
        (name, buffer) for name, buffer in model.named_buffers() if mtg.EXPERT_PARAM_INFIX in name
    ]
    assert len(expert_buffers) == 4 * LAYERS
    for name, buffer in expert_buffers:
        leaf = name.rsplit(".", 1)[1]
        assert leaf in mie.INT8_EXPERT_BUFFERS, name
        expected_dtype = torch.int8 if leaf.endswith("_q") else torch.float32
        assert buffer.dtype is expected_dtype, name
        assert buffer.device.type == "cpu", name
        assert not buffer.requires_grad, name

    assert census["expert_int8_buffers"] == 4 * LAYERS
    assert census["expert_buffer_devices"] == ["cpu"]
    # The census is the actual bytes, not an estimate: int8 weights plus one
    # fp32 scale per output row.
    expected = 0
    for name, buffer in expert_buffers:
        expected += buffer.numel() * buffer.element_size()
    assert census["expert_bytes"] == expected
    assert census["expert_int8_bytes"] + census["expert_scale_bytes"] == census["expert_bytes"]
    # 4 bytes of int8-per-weight overhead comes only from the scales: one per
    # output row, so ~4.5 bits/weight, not 16.
    assert census["expert_scale_bytes"] > 0


def test_the_census_scales_with_the_quantized_experts_not_the_bf16_ones(tmp_path):
    from transformers import AutoModelForCausalLM

    base = _checkpoint(tmp_path)
    reference = AutoModelForCausalLM.from_pretrained(base)
    expert_elements = sum(
        parameter.numel()
        for name, parameter in reference.named_parameters()
        if mtg.EXPERT_PARAM_INFIX in name
    )
    bf16_expert_bytes = expert_elements * 2
    _, census = mie.load_model_int8_experts(base)
    # int8 is one byte per weight; the per-row fp32 scales add 4 bytes per output
    # row, so the resident total is half the bf16 bytes plus roughly one scale
    # per HIDDEN or INTER weights -- close to 8.5 bits/weight, never 16.
    assert census["expert_int8_bytes"] == expert_elements
    assert census["expert_bytes"] == census["expert_int8_bytes"] + census["expert_scale_bytes"]
    assert census["expert_bytes"] < bf16_expert_bytes
    # Every expert parameter is one output row per (expert, output feature).
    assert census["expert_scale_bytes"] == 4 * (LAYERS * NUM_EXPERTS * (2 * INTER + HIDDEN))


def test_streaming_holds_at_most_one_bf16_expert_tensor_at_a_time(tmp_path, monkeypatch):
    base = _checkpoint(tmp_path)
    observed: dict[str, int] = {"live": 0, "max": 0, "expert_tensors": 0}

    class _TrackedHandle:
        def __init__(self, handle):
            self._handle = handle

        def keys(self):
            return self._handle.keys()

        def get_tensor(self, key):
            tensor = self._handle.get_tensor(key)
            if mtg.EXPERT_PARAM_INFIX in key:
                observed["expert_tensors"] += 1
                observed["live"] += 1
                observed["max"] = max(observed["max"], observed["live"])
                weakref.finalize(tensor, _released)
            return tensor

        def __enter__(self):
            self._handle.__enter__()
            return self

        def __exit__(self, *args):
            return self._handle.__exit__(*args)

    def _released():
        observed["live"] -= 1

    real_open = mie.safe_open

    def _safe_open(*args, **kwargs):
        return _TrackedHandle(real_open(*args, **kwargs))

    monkeypatch.setattr(mie, "safe_open", _safe_open)
    mie.load_model_int8_experts(base)

    # More than one expert tensor was read (so this really streamed), and never
    # more than one bf16 expert tensor was alive at the same time.
    assert observed["expert_tensors"] > LAYERS
    assert observed["max"] == 1
    assert observed["live"] == 0


def test_a_fused_gate_up_checkpoint_loads_the_same_way_as_a_per_expert_one(tmp_path):
    from safetensors.torch import load_file, save_file

    from transformers import AutoModelForCausalLM

    per_expert = _checkpoint(tmp_path, name="per-expert")
    state = load_file(str(sorted(per_expert.glob("*.safetensors"))[0]))
    for shard in sorted(per_expert.glob("*.safetensors"))[1:]:
        state.update(load_file(str(shard)))

    fused: dict[str, torch.Tensor] = {}
    per_layer: dict[str, list[str]] = {}
    for key in state:
        if mtg.EXPERT_PARAM_INFIX in key:
            prefix, _, _tail = key.rpartition(mtg.EXPERT_PARAM_INFIX)
            per_layer.setdefault(prefix, []).append(key)
        else:
            fused[key] = state[key]
    for prefix, _keys in per_layer.items():
        layer = prefix + mtg.EXPERT_PARAM_INFIX
        gate = torch.stack([state[f"{layer}{index}.gate_proj.weight"] for index in range(NUM_EXPERTS)])
        up = torch.stack([state[f"{layer}{index}.up_proj.weight"] for index in range(NUM_EXPERTS)])
        down = torch.stack([state[f"{layer}{index}.down_proj.weight"] for index in range(NUM_EXPERTS)])
        fused[f"{layer}gate_up_proj"] = torch.cat([gate, up], dim=1)
        fused[f"{layer}down_proj"] = down

    fused_dir = tmp_path / "fused"
    fused_dir.mkdir()
    save_file(fused, str(fused_dir / "model.safetensors"))
    for name in ("config.json", "generation_config.json"):
        if (per_expert / name).is_file():
            (fused_dir / name).write_text((per_expert / name).read_text(encoding="utf-8"))

    reference = AutoModelForCausalLM.from_pretrained(fused_dir)
    model, _ = mie.load_model_int8_experts(fused_dir)
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, dict(reference.named_parameters())[name]), name
    for name, module in model.named_modules():
        if name.endswith("experts"):
            assert module.gate_up_q.shape == (NUM_EXPERTS, 2 * INTER, HIDDEN)
            assert module.down_q.shape == (NUM_EXPERTS, HIDDEN, INTER)


# --- the layout reap160 actually ships --------------------------------------
#
# chowder_batch/reap_prune.py keeps the source keys byte-for-byte and prunes the
# three expert tensors in place, so the artifact's shards carry FUSED expert keys
# under a `model.language_model.` prefix, a multimodal config (model_type
# qwen3_5_moe with a text_config), and the conditional-generation export's vision
# tower in the same shards. This fixture is that shape, tiny and synthetic.


def _artifact_checkpoint(tmp_path: Path, *, extra_towers: bool = False) -> Path:
    from safetensors.torch import save_file
    from transformers import AutoConfig, AutoModelForCausalLM

    text = {
        "vocab_size": 64,
        "hidden_size": HIDDEN,
        "intermediate_size": 8,
        "moe_intermediate_size": INTER,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "num_experts": NUM_EXPERTS,
        "num_experts_per_tok": TOP_K,
        "decoder_sparse_step": 1,
        "max_position_embeddings": 32,
        "head_dim": HIDDEN // 2,
        "tie_word_embeddings": False,
    }
    config = {
        "model_type": "qwen3_5_moe",
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "text_config": text,
        "tie_word_embeddings": False,
    }
    base = tmp_path / ("artifact-towers" if extra_towers else "artifact")
    base.mkdir()
    (base / "config.json").write_text(json.dumps(config), encoding="utf-8")

    with torch.device("meta"):
        skeleton = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(base))
    generator = torch.Generator().manual_seed(5)
    state = {
        key.replace("model.", "model.language_model.", 1) if key.startswith("model.") else key: (
            torch.randn(tensor.shape, generator=generator).to(tensor.dtype)
        )
        for key, tensor in skeleton.state_dict().items()
    }
    if extra_towers:
        state["model.visual.patch_embed.proj.weight"] = torch.zeros(4, 3, 2, 2)
        state["mtp.fc.weight"] = torch.zeros(2, 2)
    save_file(state, str(base / "model.safetensors"))
    return base, state


def test_the_artifact_layout_loads_and_matches_from_pretrained(tmp_path):
    from transformers import AutoModelForCausalLM

    base, state = _artifact_checkpoint(tmp_path)
    reference = AutoModelForCausalLM.from_pretrained(base)
    model, census = mie.load_model_int8_experts(base)

    # The shards really are the artifact's layout, not the plain one.
    assert any(
        key == "model.language_model.layers.0.mlp.experts.gate_up_proj" for key in state
    )
    assert not any(key.startswith("model.layers.") for key in state)
    # Every nested text-model key was rewritten; the head's name has no wrapper.
    assert census["keys_renamed"] == sum(1 for key in state if key.startswith("model."))
    assert census["keys_renamed"] == len(state) - 1

    reference_params = dict(reference.named_parameters())
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, reference_params[name]), name
    for name, buffer in model.named_buffers():
        if mtg.EXPERT_PARAM_INFIX in name:
            continue
        assert torch.equal(buffer, dict(reference.named_buffers())[name]), name
    assert model.config.to_dict() == reference.config.to_dict()

    # The fused expert tensors were quantized, not dropped.
    experts = _experts_module(model)
    for layer in range(LAYERS):
        module = _experts_module(model, layer)
        source = state[f"model.language_model.layers.{layer}.mlp.experts.gate_up_proj"]
        want = mie.dequantize_rows(*mie.quantize_rows(source), torch.float32)
        assert torch.allclose(module.gate_up_q.to(torch.float32) * module.gate_up_scale, want)
    assert experts.num_experts == NUM_EXPERTS


def test_a_vision_tower_in_the_shards_is_ignored_and_recorded(tmp_path):
    base, _state = _artifact_checkpoint(tmp_path, extra_towers=True)
    model, census = mie.load_model_int8_experts(base)

    assert census["ignored_key_count"] == 2
    assert census["ignored_keys"] == ["model.visual.patch_embed.proj.weight", "mtp.fc.weight"]
    assert census["unexpected_keys"] == []
    # The experts still quantized: the ignored towers did not derail the load.
    assert census["expert_modules"] == LAYERS


def test_the_artifact_layout_is_never_loaded_as_bf16_experts(tmp_path, monkeypatch):
    """The same one-live-tensor bound, on the layout the real artifact uses."""
    base, _state = _artifact_checkpoint(tmp_path)
    observed = {"live": 0, "max": 0, "seen": 0}

    class _TrackedHandle:
        def __init__(self, handle):
            self._handle = handle

        def keys(self):
            return self._handle.keys()

        def get_tensor(self, key):
            tensor = self._handle.get_tensor(key)
            if mtg.EXPERT_PARAM_INFIX in key:
                observed["seen"] += 1
                observed["live"] += 1
                observed["max"] = max(observed["max"], observed["live"])
                weakref.finalize(tensor, _released)
            return tensor

        def __enter__(self):
            self._handle.__enter__()
            return self

        def __exit__(self, *args):
            return self._handle.__exit__(*args)

    def _released():
        observed["live"] -= 1

    real_open = mie.safe_open
    monkeypatch.setattr(mie, "safe_open", lambda *a, **k: _TrackedHandle(real_open(*a, **k)))
    mie.load_model_int8_experts(base)

    assert observed["seen"] == 2 * LAYERS
    assert observed["max"] == 1
    assert observed["live"] == 0


# --- forward / backward -------------------------------------------------------


def test_int8_forward_matches_the_bf16_bulk_forward(tmp_path):
    base = _checkpoint(tmp_path)
    bulk = _experts_module(_bf16_bulk_model(base))
    int8, _, placement = _int8_model(base)
    assert placement["patched_expert_modules"] == LAYERS

    hidden, index, weights = _inputs()
    want = bulk(hidden, index, weights).detach()
    got = _experts_module(int8)(hidden, index, weights).detach()
    assert _relative(got, want) <= TOLERANCE


def test_int8_gradients_match_the_bf16_bulk_gradients(tmp_path):
    base = _checkpoint(tmp_path)
    bulk = _experts_module(_bf16_bulk_model(base))
    int8, _, _ = _int8_model(base)

    out_want, hidden_want, weights_want = _forward_and_grads(bulk)
    out_got, hidden_got, weights_got = _forward_and_grads(_experts_module(int8))

    assert _relative(out_got, out_want) <= TOLERANCE
    assert _relative(hidden_got, hidden_want) <= TOLERANCE
    assert _relative(weights_got, weights_want) <= TOLERANCE
    # The router really is trained by this path: the gradient is not a zero.
    assert weights_got.abs().sum() > 0


def test_int8_saves_only_activations_and_routing_for_backward(tmp_path):
    base = _checkpoint(tmp_path)
    int8, _, _ = _int8_model(base)
    module = _experts_module(int8)
    hidden, index, weights = _inputs()

    saved: list[tuple[torch.Size, torch.dtype]] = []

    def pack(tensor):
        saved.append((tuple(tensor.shape), tensor.dtype))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        module(hidden, index, weights)

    # hidden_states, top_k_index, top_k_weights only: no [experts, ...] matrix,
    # and no int8 buffer, was held for backward.
    assert saved
    assert all(len(shape) <= 2 for shape, _ in saved), saved
    assert not any(dtype is torch.int8 for _, dtype in saved), saved


def test_int8_forward_and_grads_survive_activation_checkpointing(tmp_path):
    base = _checkpoint(tmp_path)
    bulk = _experts_module(_bf16_bulk_model(base))
    int8, _, _ = _int8_model(base)

    def grads(module, *, checkpointed: bool):
        hidden, index, weights = _inputs()

        def forward(hidden_states):
            return module(hidden_states, index, weights).square().sum()

        loss = (
            torch.utils.checkpoint.checkpoint(forward, hidden, use_reentrant=False)
            if checkpointed
            else forward(hidden)
        )
        grad_hidden, grad_weights = torch.autograd.grad(loss, (hidden, weights))
        return loss.detach(), grad_hidden, grad_weights

    loss_bulk, hidden_bulk, weights_bulk = grads(bulk, checkpointed=False)
    loss_int8, hidden_int8, weights_int8 = grads(_experts_module(int8), checkpointed=False)
    loss_ckpt, hidden_ckpt, weights_ckpt = grads(_experts_module(int8), checkpointed=True)

    assert _relative(loss_int8, loss_bulk) <= TOLERANCE
    assert _relative(hidden_int8, hidden_bulk) <= TOLERANCE
    assert _relative(weights_int8, weights_bulk) <= TOLERANCE
    # Checkpointing recomputes the same function, so it must not drift further.
    assert torch.allclose(loss_ckpt, loss_int8, atol=1e-6)
    assert _relative(hidden_ckpt, hidden_bulk) <= TOLERANCE
    assert _relative(weights_ckpt, weights_bulk) <= TOLERANCE


def test_a_full_forward_pass_matches_the_bf16_bulk_model(tmp_path):
    base = _checkpoint(tmp_path)
    bulk = _bf16_bulk_model(base)
    int8, _, _ = _int8_model(base)

    generator = torch.Generator().manual_seed(7)
    tokens = torch.randint(0, 64, (1, 12), generator=generator)
    with torch.no_grad():
        want = bulk(tokens).logits
        got = int8(tokens).logits
    assert _relative(got, want) <= TOLERANCE


# --- placement admission ------------------------------------------------------


def test_expert_weights_are_never_trainable_under_the_int8_placement(tmp_path):
    base = _checkpoint(tmp_path)
    model, _, placement = _int8_model(base)
    assert placement["remaining_expert_params"] == 0
    for name, _parameter in model.named_parameters():
        assert mtg.EXPERT_PARAM_INFIX not in name, name
    for name, module in model.named_modules():
        if name.endswith("experts"):
            assert not any(parameter.requires_grad for parameter in module.parameters()), name


def test_installing_the_int8_forward_refuses_an_unquantized_module():
    class Qwen3MoeExperts(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_experts = NUM_EXPERTS
            self.gate_up_proj = torch.nn.Parameter(
                torch.zeros(NUM_EXPERTS, 2 * INTER, HIDDEN), requires_grad=False
            )
            self.down_proj = torch.nn.Parameter(torch.zeros(NUM_EXPERTS, HIDDEN, INTER), requires_grad=False)
            self.act_fn = F.silu

    model = torch.nn.Module()
    model.experts = Qwen3MoeExperts()
    with pytest.raises(RuntimeError, match="no int8 expert buffers"):
        mie.install_int8_expert_forward(model)


def test_placement_validation_accepts_int8_and_refuses_pinning():
    def spec(**overrides):
        fields = {"base_model": "base", "dataset": "data.jsonl", "output_dir": "out"}
        fields.update(overrides)
        return TransformersPeftRunSpec(**fields)

    assert mie.INT8_PLACEMENT in mtg.EXPERT_PLACEMENTS
    assert mie.INT8_PLACEMENT in mtg.INT8_PLACEMENTS
    from chowder.backends.transformers_peft import _ALLOWED_EXPERT_PLACEMENTS

    assert mie.INT8_PLACEMENT in _ALLOWED_EXPERT_PLACEMENTS

    accepted = spec(
        expert_placement=mie.INT8_PLACEMENT, target_preset="moe_frozen_experts", expert_pinning="none"
    )
    # Placement is value-transparent: it moves where frozen weights live.
    assert accepted.recipe_digest() == spec(target_preset="moe_frozen_experts").recipe_digest()

    with pytest.raises(ValueError, match="not supported with expert_placement"):
        spec(
            expert_placement=mie.INT8_PLACEMENT,
            target_preset="moe_frozen_experts",
            expert_pinning="registered",
        )
    with pytest.raises(ValueError, match="moe_frozen_experts"):
        spec(expert_placement=mie.INT8_PLACEMENT)
    with pytest.raises(ValueError, match="quantization='none'"):
        spec(expert_placement=mie.INT8_PLACEMENT, target_preset="moe_frozen_experts", quantization="4bit")
    with pytest.raises(ValueError, match="unsupported expert_placement"):
        spec(expert_placement="gpu-magic")


def test_the_gate_runner_self_check_produces_a_record_that_names_its_gaps(tmp_path):
    """The gate runner is the thing that measures the prereg; it has to be runnable
    evidence itself, on the CPU, with the gates it cannot answer said out loud."""
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "gates.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "chowder_batch" / "moe_int8_gates.py"),
            "--self-check",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        cwd=str(root),
        env={**os.environ, "CHOWDER_NO_GPU": "1", "PYTHONPATH": str(root / "src")},
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]

    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["schema"] == "chowder.moe-placement-gates.v1"
    assert record["placement"] == mie.INT8_PLACEMENT
    assert set(record["arms"]) == {"int8", "bulk"}
    for arm in ("int8", "bulk"):
        entry = record["arms"][arm]
        assert entry["census"], arm
        assert entry["s_per_step"] and entry["s_per_step"] > 0, arm
        assert entry["steps"] == 2, arm
        # It measures the placement, never claims to be a training step.
        assert entry["includes_lora"] is False, arm
        assert entry["includes_optimizer"] is False, arm
        assert entry["gates"]["load_census"] == "measured", arm
        assert entry["gates"]["host_rss"] == "measured", arm
        assert entry["gates"]["memory_vram"] == "not measured (CPU arm)", arm
        assert entry["gates"]["smoke"].startswith("not measured"), arm
        assert entry["gates"]["router_balance"].startswith("not measured"), arm
        assert entry["gates"]["wddm_spill"].startswith("not observable")

    int8_census = record["arms"]["int8"]["census"]
    assert int8_census["remaining_expert_params"] == 0
    assert int8_census["expert_bytes"] > 0
    assert int8_census["patched_expert_modules"] == LAYERS


def test_the_gate_runner_census_only_mode_steps_nothing_and_says_so(tmp_path):
    """``--steps 0`` is the mode a box with no free VRAM can still measure: the
    load census and host RSS, with the throughput gate reported as unmeasured."""
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "census.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "chowder_batch" / "moe_int8_gates.py"),
            "--self-check",
            "--steps",
            "0",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        cwd=str(root),
        env={**os.environ, "CHOWDER_NO_GPU": "1", "PYTHONPATH": str(root / "src")},
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]

    record = json.loads(out.read_text(encoding="utf-8"))
    for arm in ("int8", "bulk"):
        entry = record["arms"][arm]
        assert entry["steps"] == 0, arm
        assert entry["step_seconds"] == [], arm
        assert entry["s_per_step"] is None, arm
        assert entry["gates"]["throughput"] == "not measured (--steps 0: census only)", arm
        # The census itself is still a measurement, not a placeholder.
        assert entry["census"], arm
        assert entry["gates"]["load_census"] == "measured", arm


def test_the_int8_placement_does_not_exist_before_this_change():
    """The one test that is red on the base branch: there is no int8 placement,
    no streaming loader, and no int8 forward to install."""
    assert mie.INT8_PLACEMENT == "cpu-transient-grad-int8"
    assert "cpu-transient-grad-int8" in mtg.EXPERT_PLACEMENTS
    assert callable(mie.load_model_int8_experts)
    assert callable(mie.prepare_int8_placement)
    assert callable(mie.install_int8_expert_forward)
    from chowder.backends.transformers_peft import _ALLOWED_EXPERT_PLACEMENTS

    assert "cpu-transient-grad-int8" in _ALLOWED_EXPERT_PLACEMENTS
