"""The transient-grad experts forward: same loss and gradients as the resident
model, no expert weight saved for backward, frozen experts enforced, and the
opt-in MoE LoRA preset never reaching a routed expert."""

from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")
configuration = pytest.importorskip("transformers.models.qwen3_5_moe.configuration_qwen3_5_moe")

from chowder.backends.moe_transient_grad import install_transient_grad_expert_forward  # noqa: E402
from chowder.backends.transformers_worker import _QWEN3_5_TARGET_MODULES, _resolve_target_modules  # noqa: E402

LAYERS, HIDDEN, MOE_INT = 4, 32, 16


def _config():
    return configuration.Qwen3_5MoeTextConfig(
        vocab_size=64, hidden_size=HIDDEN, num_hidden_layers=LAYERS, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, moe_intermediate_size=MOE_INT,
        shared_expert_intermediate_size=16, num_experts=8, num_experts_per_tok=2,
        linear_num_value_heads=4, linear_num_key_heads=2, linear_key_head_dim=8,
        linear_value_head_dim=8, full_attention_interval=2, max_position_embeddings=64,
    )


def _tiny_moe():
    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(_config()).float()
    for name, param in model.named_parameters():
        if ".mlp.experts." in name:
            param.requires_grad_(False)
    return model


def _loss_and_grads(model, ids):
    model.zero_grad(set_to_none=True)
    loss = model(input_ids=ids, labels=ids).loss
    loss.backward()
    return loss.detach(), {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


def test_transient_grad_matches_the_resident_forward_and_backward():
    resident = _tiny_moe()
    transient = copy.deepcopy(resident)
    assert install_transient_grad_expert_forward(transient) == LAYERS
    ids = torch.randint(0, 64, (2, 16), generator=torch.Generator().manual_seed(1))

    loss_r, grads_r = _loss_and_grads(resident, ids)
    loss_t, grads_t = _loss_and_grads(transient, ids)

    assert torch.allclose(loss_r, loss_t, atol=1e-6)
    assert grads_r.keys() == grads_t.keys()
    # Gradients reach the router and the layers below the experts, not just the top.
    assert any(n.endswith("mlp.gate.weight") for n in grads_t)
    assert any("layers.0." in n for n in grads_t)
    for name in grads_r:
        assert torch.allclose(grads_r[name], grads_t[name], atol=1e-5), name
    assert not any(".mlp.experts." in n for n in grads_t)


def _saved_expert_slices(model, ids) -> int:
    """How many tensors autograd saved that live in an expert weight's storage
    (identity, not shape: attention and shared-expert weights can share shapes)."""
    expert_storage = {p.untyped_storage().data_ptr() for n, p in model.named_parameters() if ".mlp.experts." in n}
    count = 0

    def pack(t):
        nonlocal count
        count += t.untyped_storage().data_ptr() in expert_storage
        return t

    # Only what the forward saves is held until backward; the backward's own
    # one-expert recompute frees its copy as soon as that expert's grad is taken.
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        loss = model(input_ids=ids, labels=ids).loss
    loss.backward()
    return count


def test_no_expert_weight_is_saved_for_backward():
    ids = torch.randint(0, 64, (2, 16), generator=torch.Generator().manual_seed(2))
    resident = _tiny_moe()
    # Negative control: the stock forward saves expert slices for backward --
    # on the real model, every expert of every layer at once.
    assert _saved_expert_slices(resident, ids) > 0
    transient = copy.deepcopy(resident)
    install_transient_grad_expert_forward(transient)
    assert _saved_expert_slices(transient, ids) == 0


def test_trainable_experts_are_refused():
    model = _tiny_moe()
    model.model.layers[0].mlp.experts.gate_up_proj.requires_grad_(True)
    with pytest.raises(RuntimeError, match="trainable expert weights"):
        install_transient_grad_expert_forward(model)


def test_moe_frozen_experts_preset_never_reaches_a_routed_expert():
    modeling = pytest.importorskip("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe")
    with torch.device("meta"):
        model = modeling.Qwen3_5MoeForCausalLM(_config())
    resolved = _resolve_target_modules(model, explicit=(), preset="moe_frozen_experts")
    assert resolved == list(_QWEN3_5_TARGET_MODULES)
    matched = [
        name for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and name.rsplit(".", 1)[-1] in resolved
    ]
    assert matched and not any(".mlp.experts." in name for name in matched)
    # The shared expert is covered in every MoE layer.
    assert sum(".mlp.shared_expert.down_proj" in name for name in matched) == LAYERS
    # The dense preset still refuses the MoE stack.
    with pytest.raises(RuntimeError, match="no curated module list"):
        _resolve_target_modules(model, explicit=(), preset="attention_and_mlp")


def test_cpu_resident_experts_train_on_cuda_like_the_resident_model():
    """The real placement: experts on the CPU, everything else on the GPU."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    resident = _tiny_moe()
    ids = torch.randint(0, 64, (2, 16), generator=torch.Generator().manual_seed(3))
    loss_r, grads_r = _loss_and_grads(resident, ids)
    split = copy.deepcopy(resident)
    for name, param in split.named_parameters():
        if ".mlp.experts." not in name:
            param.data = param.data.cuda()
    for buffer in split.buffers():
        buffer.data = buffer.data.cuda()
    install_transient_grad_expert_forward(split)
    assert {p.device.type for n, p in split.named_parameters() if ".mlp.experts." in n} == {"cpu"}
    loss_s, grads_s = _loss_and_grads(split, ids.cuda())
    assert abs(float(loss_r) - float(loss_s)) < 1e-5
    for name in grads_r:
        assert torch.allclose(grads_r[name], grads_s[name].cpu(), atol=1e-5), name


# --- the worker's expert_placement ------------------------------------------------

from chowder.backends.transformers_peft import TransformersPeftRunSpec  # noqa: E402


def _spec(**overrides):
    fields = {"base_model": "base", "dataset": "data.jsonl", "output_dir": "out"}
    fields.update(overrides)
    return TransformersPeftRunSpec(**fields)


def test_cpu_transient_grad_placement_requires_the_frozen_experts_coverage():
    with pytest.raises(ValueError, match="moe_frozen_experts"):
        _spec(expert_placement="cpu-transient-grad")
    with pytest.raises(ValueError, match="moe_frozen_experts"):
        _spec(expert_placement="cpu-transient-grad", target_preset="moe_frozen_experts", target_modules=("q_proj",))
    with pytest.raises(ValueError, match="quantization='none'"):
        _spec(expert_placement="cpu-transient-grad", target_preset="moe_frozen_experts", quantization="4bit")
    with pytest.raises(ValueError, match="unsupported expert_placement"):
        _spec(expert_placement="gpu-magic")
    ok = _spec(expert_placement="cpu-transient-grad", target_preset="moe_frozen_experts")
    # Value-transparent: placement changes where experts live, never the recipe.
    assert ok.recipe_digest() == _spec(target_preset="moe_frozen_experts").recipe_digest()


def _saved_tiny_moe(tmp_path):
    path = tmp_path / "tiny-moe"
    _tiny_moe().save_pretrained(path)
    return str(path)


def test_split_device_map_puts_only_experts_on_the_cpu(tmp_path):
    from chowder.backends.moe_transient_grad import split_device_map

    device_map = split_device_map(_saved_tiny_moe(tmp_path), revision=None, local_files_only=True)
    experts = {n: d for n, d in device_map.items() if ".mlp.experts." in n}
    assert experts and set(experts.values()) == {"cpu"}
    assert {d for n, d in device_map.items() if n not in experts} == {0}


def test_the_placement_census_refuses_a_model_that_was_not_split():
    from chowder.backends.moe_transient_grad import prepare_split_model

    with pytest.raises(RuntimeError, match="census failed"):
        prepare_split_model(_tiny_moe(), device_type="cuda")


def test_split_load_trains_lora_through_cpu_resident_experts(tmp_path):
    """The worker's real path: from_pretrained with the split map, the census,
    the moe_frozen_experts LoRA, one backward."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM

    from chowder.backends.moe_transient_grad import prepare_split_model, split_device_map

    base_dir = _saved_tiny_moe(tmp_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_dir, dtype=torch.float32, device_map=split_device_map(base_dir, revision=None, local_files_only=True)
    )
    evidence = prepare_split_model(model)
    assert evidence["expert_params_on_cpu"] == evidence["expert_params"] > 0
    assert evidence["patched_expert_modules"] == LAYERS
    targets = _resolve_target_modules(model, explicit=(), preset="moe_frozen_experts")
    model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=targets, task_type=TaskType.CAUSAL_LM))
    ids = torch.randint(0, 64, (2, 16), device="cuda")
    model(input_ids=ids, labels=ids).loss.backward()
    lora = [(n, p) for n, p in model.named_parameters() if "lora_" in n]
    assert lora and all(p.grad is not None and torch.isfinite(p.grad).all() for _, p in lora)
    # Gradient reached the bottom layer, through every frozen-expert layer above it.
    assert any("layers.0." in n and float(p.grad.abs().sum()) > 0 for n, p in lora)


def test_parent_adapter_attach_keeps_experts_on_the_cpu(tmp_path):
    """A continued run attaches its parent with PeftModel.from_pretrained, which
    re-plans a model carrying hf_device_map onto the device (#247 measured all
    host experts moving). attach_preserving_split must keep them on the CPU."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM

    from chowder.backends.moe_transient_grad import attach_preserving_split, prepare_split_model, split_device_map

    base_dir = _saved_tiny_moe(tmp_path)
    adapter_dir = tmp_path / "parent"
    targets = list(_QWEN3_5_TARGET_MODULES)
    get_peft_model(_tiny_moe(), LoraConfig(r=4, lora_alpha=8, target_modules=targets,
                                           task_type=TaskType.CAUSAL_LM)).save_pretrained(adapter_dir)

    def load():
        model = AutoModelForCausalLM.from_pretrained(
            base_dir, dtype=torch.float32, device_map=split_device_map(base_dir, revision=None, local_files_only=True))
        prepare_split_model(model)
        return model

    model = attach_preserving_split(
        load(), lambda b: PeftModel.from_pretrained(b, str(adapter_dir), is_trainable=True))
    assert {p.device.type for n, p in model.named_parameters() if ".mlp.experts." in n} == {"cpu"}
    ids = torch.randint(0, 64, (1, 12), device="cuda")
    model(input_ids=ids, labels=ids).loss.backward()
    assert any(p.grad is not None for n, p in model.named_parameters() if "lora_" in n)
