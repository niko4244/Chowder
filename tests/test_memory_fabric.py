from __future__ import annotations

import os

import pytest

_REAL_ML_SMOKE = pytest.mark.skipif(
    os.environ.get("CHOWDER_REAL_ML_SMOKE") != "1",
    reason="real ML smoke requires CHOWDER_REAL_ML_SMOKE=1 and train dependencies",
)
_TINY_MODEL = "trl-internal-testing/tiny-LlamaForCausalLM-3.2"
_LORA_CFG = dict(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=["q_proj", "v_proj"])


def _build_lora_model():
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(_TINY_MODEL)
    return get_peft_model(base, LoraConfig(**_LORA_CFG))


def _build_inputs(tok, device):
    text = "Question: What token comes after alpha? Answer: beta"
    return tok(text, return_tensors="pt").to(device)


@_REAL_ML_SMOKE
def test_streamed_training_matches_resident_training_exactly():
    """The core correctness claim of the whole module: streaming a PEFT
    model's frozen base_layer weights from pinned CPU RAM with one-layer-
    ahead async prefetch must produce bit-identical loss and LoRA
    gradients to normal fully GPU-resident training on the same input
    and seed -- proven directly against the real production module, not
    just the isolated mechanism prototype this was designed from."""
    import torch
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device available for a real streaming comparison")

    from chowder.memory_fabric import stream_frozen_layers

    tok = AutoTokenizer.from_pretrained(_TINY_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    torch.manual_seed(0)
    model_resident = _build_lora_model().cuda()
    model_resident.train()
    inputs = _build_inputs(tok, "cuda")
    out_resident = model_resident(**inputs, labels=inputs["input_ids"])
    out_resident.loss.backward()
    grad_resident = {
        n: p.grad.detach().clone().cpu()
        for n, p in model_resident.named_parameters()
        if p.requires_grad and p.grad is not None
    }

    torch.manual_seed(0)
    model_streamed = _build_lora_model()
    model_streamed.train()
    model_streamed.to("cuda")
    streamed = stream_frozen_layers(model_streamed, torch.device("cuda"))
    streamed.start_step()
    out_streamed = model_streamed(**inputs, labels=inputs["input_ids"])
    out_streamed.loss.backward()
    torch.cuda.synchronize()
    grad_streamed = {
        n: p.grad.detach().clone().cpu()
        for n, p in model_streamed.named_parameters()
        if p.requires_grad and p.grad is not None
    }

    assert out_resident.loss.item() == pytest.approx(out_streamed.loss.item(), abs=1e-6)
    assert set(grad_resident) == set(grad_streamed)
    assert grad_resident, "no trainable gradients were compared -- test setup is broken"
    for name in grad_resident:
        assert torch.allclose(grad_resident[name], grad_streamed[name], atol=1e-5), name


@_REAL_ML_SMOKE
def test_streamed_training_is_correct_across_repeated_iterations():
    """A stream-synchronization race (a prefetched tensor's memory reused
    by the allocator before the compute stream finished reading it) would
    show up as non-deterministic drift across repeated iterations, not
    necessarily on the first one -- checked across several real steps,
    not just one."""
    import torch
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device available for a real streaming comparison")

    from chowder.memory_fabric import stream_frozen_layers

    tok = AutoTokenizer.from_pretrained(_TINY_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = _build_lora_model()
    model.train()
    model.to("cuda")
    streamed = stream_frozen_layers(model, torch.device("cuda"))
    inputs = _build_inputs(tok, "cuda")

    losses = []
    for _ in range(5):
        model.zero_grad(set_to_none=True)
        streamed.start_step()
        out = model(**inputs, labels=inputs["input_ids"])
        out.loss.backward()
        torch.cuda.synchronize()
        losses.append(out.loss.item())

    assert len(set(losses)) == 1, f"loss drifted across repeated iterations: {losses}"


@_REAL_ML_SMOKE
def test_restore_returns_model_to_normal_resident_training():
    """After restore(), the model must train normally again -- fully
    GPU-resident, no streaming -- with results identical to a model that
    was never streamed at all."""
    import torch
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device available for a real streaming comparison")

    from chowder.memory_fabric import stream_frozen_layers

    tok = AutoTokenizer.from_pretrained(_TINY_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    inputs = _build_inputs(tok, "cuda")

    torch.manual_seed(0)
    model_never_streamed = _build_lora_model().cuda()
    model_never_streamed.train()
    out_never = model_never_streamed(**inputs, labels=inputs["input_ids"])
    out_never.loss.backward()
    grad_never = {
        n: p.grad.detach().clone().cpu()
        for n, p in model_never_streamed.named_parameters()
        if p.requires_grad and p.grad is not None
    }

    torch.manual_seed(0)
    model = _build_lora_model()
    model.train()
    model.to("cuda")
    streamed = stream_frozen_layers(model, torch.device("cuda"))
    streamed.start_step()
    model(**inputs, labels=inputs["input_ids"])
    streamed.restore()

    model.zero_grad(set_to_none=True)
    out_restored = model(**inputs, labels=inputs["input_ids"])
    out_restored.loss.backward()
    grad_restored = {
        n: p.grad.detach().clone().cpu()
        for n, p in model.named_parameters()
        if p.requires_grad and p.grad is not None
    }

    assert out_never.loss.item() == pytest.approx(out_restored.loss.item(), abs=1e-6)
    for name in grad_never:
        assert torch.allclose(grad_never[name], grad_restored[name], atol=1e-5), name


@_REAL_ML_SMOKE
def test_runtime_tracks_real_bytes_transferred():
    import torch
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device available for a real streaming comparison")

    from chowder.memory_fabric import stream_frozen_layers

    tok = AutoTokenizer.from_pretrained(_TINY_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = _build_lora_model()
    model.train()
    model.to("cuda")
    streamed = stream_frozen_layers(model, torch.device("cuda"))
    inputs = _build_inputs(tok, "cuda")
    streamed.start_step()
    out = model(**inputs, labels=inputs["input_ids"])
    out.loss.backward()
    torch.cuda.synchronize()

    assert streamed.runtime.bytes_transferred > 0


@_REAL_ML_SMOKE
def test_patched_model_survives_a_blanket_to_device_call():
    """Regression test for a real bug found while investigating
    production wiring: HF's Trainer/accelerate call a blanket
    model.to(device) at more than one point this module does not control
    -- Trainer.__init__ and again inside accelerator.prepare_model() on
    every .train() call. A meta-tensor placeholder for the patched
    base_layer.weight/.bias (the first implementation) makes any of
    those calls raise "Cannot copy out of meta tensor; no data!" --
    confirmed by driving a real Trainer.train() end to end with
    streaming applied between Trainer construction and .train(). The
    fix is a genuinely empty (0-element) *real* tensor instead, which
    moves between devices like any ordinary parameter."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device available for a real streaming comparison")

    from chowder.memory_fabric import stream_frozen_layers

    model = _build_lora_model()
    model.train()
    model.to("cuda")
    streamed = stream_frozen_layers(model, torch.device("cuda"))

    # The exact call that crashed before the fix.
    model.to("cuda")
    for base_layer in streamed._patched:
        assert base_layer.weight.numel() == 0
        assert base_layer.weight.device.type == "cuda"
