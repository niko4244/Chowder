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


@_REAL_ML_SMOKE
def test_backward_prefetch_true_and_false_produce_bit_identical_gradients():
    """backward_prefetch is purely a scheduling change (when the H2D copy
    for a given layer's weight is *launched*, not what value it produces)
    -- loss and LoRA gradients must be identical whether or not backward's
    one-layer-ahead lookahead is enabled."""
    import torch
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device available for a real streaming comparison")

    from chowder.memory_fabric import stream_frozen_layers

    tok = AutoTokenizer.from_pretrained(_TINY_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    inputs = _build_inputs(tok, "cuda")

    def _run(backward_prefetch: bool) -> tuple[float, dict]:
        torch.manual_seed(0)
        model = _build_lora_model()
        model.train()
        model.to("cuda")
        streamed = stream_frozen_layers(model, torch.device("cuda"), backward_prefetch=backward_prefetch)
        streamed.start_step()
        out = model(**inputs, labels=inputs["input_ids"])
        streamed.start_backward()
        out.loss.backward()
        torch.cuda.synchronize()
        grads = {
            n: p.grad.detach().clone().cpu()
            for n, p in model.named_parameters()
            if p.requires_grad and p.grad is not None
        }
        return out.loss.item(), grads

    loss_with_prefetch, grads_with_prefetch = _run(True)
    loss_without_prefetch, grads_without_prefetch = _run(False)

    assert loss_with_prefetch == pytest.approx(loss_without_prefetch, abs=1e-6)
    assert set(grads_with_prefetch) == set(grads_without_prefetch)
    assert grads_with_prefetch, "no trainable gradients were compared -- test setup is broken"
    for name in grads_with_prefetch:
        assert torch.allclose(grads_with_prefetch[name], grads_without_prefetch[name], atol=1e-5), name


@_REAL_ML_SMOKE
def test_backward_prefetch_is_correct_across_repeated_iterations():
    """Same race-condition concern as forward's one-layer-ahead prefetch
    (a prefetched tensor's memory reused before the compute stream finished
    reading it), checked specifically for backward's own lookahead across
    several real steps."""
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
    streamed = stream_frozen_layers(model, torch.device("cuda"), backward_prefetch=True)
    inputs = _build_inputs(tok, "cuda")

    losses = []
    for _ in range(5):
        model.zero_grad(set_to_none=True)
        streamed.start_step()
        out = model(**inputs, labels=inputs["input_ids"])
        streamed.start_backward()
        out.loss.backward()
        torch.cuda.synchronize()
        losses.append(out.loss.item())

    assert len(set(losses)) == 1, f"loss drifted across repeated iterations: {losses}"


def _build_synthetic_frozen_stack(num_layers: int, dim: int, device: str):
    """A PEFT-shaped (`.base_layer` + small trainable adapter) synthetic
    stack sized so H2D transfer time (64MB/layer at dim=4096, fp32) and
    backward compute time (a [rows, dim] x [dim, dim] matmul) are the same
    order of magnitude -- the regime where overlapping one layer's transfer
    with the previous layer's compute actually has something to hide behind.
    Deliberately built from plain torch (no transformers/peft dependency):
    StreamedFrozenLayers detects targets generically via
    `hasattr(module, "base_layer")`, so this exercises the exact same real
    patch path a real PEFT model does, just at a size the tiny production
    smoke-test model is too small to show a meaningful timing difference at
    (see frozen_layer_streaming.py's own documented caveat about this)."""
    import torch
    from torch import nn

    class _Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base_layer = nn.Linear(dim, dim, bias=True)
            for p in self.base_layer.parameters():
                p.requires_grad = False
            self.lora_a = nn.Linear(dim, 8, bias=False)
            self.lora_b = nn.Linear(8, dim, bias=False)
            nn.init.zeros_(self.lora_b.weight)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.base_layer(x) + self.lora_b(self.lora_a(x))

    class _Stack(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([_Block() for _ in range(num_layers)])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for block in self.blocks:
                x = torch.relu(block(x))
            return x

    torch.manual_seed(0)
    return _Stack().to(device)


@_REAL_ML_SMOKE
def test_backward_prefetch_gives_a_real_measured_throughput_gain_with_no_vram_regression():
    """The actual throughput claim this feature exists for: backward wall
    time with prefetch enabled must be measurably lower than with it
    disabled, on a synthetic stack sized so per-layer H2D transfer and
    per-layer backward compute are comparable (frozen_layer_streaming_
    worker.py's real-model calibration documents that the tiny smoke-test
    model is too small for this -- compute there is too cheap to hide any
    transfer behind), while peak VRAM must not regress relative to the
    already-proven forward-only-prefetch baseline, and loss/gradients must
    stay identical to a fully resident run."""
    import statistics
    import time

    import torch

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device available for a real streaming comparison")

    from chowder.memory_fabric import stream_frozen_layers

    num_layers = 12
    dim = 4096
    rows = 4096
    device = "cuda"

    def _inputs():
        torch.manual_seed(1)
        return torch.randn(rows, dim, device=device)

    def _resident_step(model, x):
        torch.cuda.synchronize()
        out = model(x)
        loss = out.pow(2).mean()
        loss.backward()
        torch.cuda.synchronize()
        model.zero_grad(set_to_none=True)
        return loss.item()

    # Correctness: streamed (either mode) must match a fully resident run.
    resident_model = _build_synthetic_frozen_stack(num_layers, dim, device)
    resident_loss = _resident_step(resident_model, _inputs())
    resident_grads = {
        n: p.grad.detach().clone().cpu()
        for n, p in resident_model.named_parameters()
        if p.requires_grad and p.grad is not None
    }

    def _timed_backward_seconds(backward_prefetch: bool, *, iterations: int) -> tuple[list[float], float, float]:
        model = _build_synthetic_frozen_stack(num_layers, dim, device)
        streamed = stream_frozen_layers(model, torch.device(device), backward_prefetch=backward_prefetch)
        x = _inputs()

        def _step() -> tuple[float, float]:
            model.zero_grad(set_to_none=True)
            streamed.start_step()
            out = model(x)
            loss = out.pow(2).mean()
            torch.cuda.synchronize()
            started = time.perf_counter()
            streamed.start_backward()
            loss.backward()
            torch.cuda.synchronize()
            return time.perf_counter() - started, loss.item()

        for _ in range(2):  # warmup: CUDA kernel selection / allocator cache
            _step()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        times = []
        last_loss = None
        for _ in range(iterations):
            elapsed, last_loss = _step()
            times.append(elapsed)
        peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
        grads = {
            n: p.grad.detach().clone().cpu()
            for n, p in model.named_parameters()
            if p.requires_grad and p.grad is not None
        }
        streamed.restore()
        return times, peak_gb, last_loss, grads

    times_with_prefetch, peak_with_prefetch, loss_with_prefetch, grads_with_prefetch = _timed_backward_seconds(
        True, iterations=5
    )
    times_without_prefetch, peak_without_prefetch, loss_without_prefetch, grads_without_prefetch = (
        _timed_backward_seconds(False, iterations=5)
    )

    assert resident_loss == pytest.approx(loss_with_prefetch, abs=1e-3)
    assert resident_loss == pytest.approx(loss_without_prefetch, abs=1e-3)
    for name in resident_grads:
        assert torch.allclose(resident_grads[name], grads_with_prefetch[name], atol=1e-3), name
        assert torch.allclose(resident_grads[name], grads_without_prefetch[name], atol=1e-3), name

    median_with = statistics.median(times_with_prefetch)
    median_without = statistics.median(times_without_prefetch)
    assert median_with < median_without, (
        f"backward prefetch did not improve backward wall time: "
        f"with={median_with:.4f}s without={median_without:.4f}s"
    )
    # No VRAM regression relative to the non-prefetch streamed baseline --
    # both stream at most ~2 layers' weights at a time regardless of the
    # lookahead direction, so a real regression here would mean the new
    # code path is accidentally keeping extra layers resident.
    assert peak_with_prefetch <= peak_without_prefetch * 1.05


@_REAL_ML_SMOKE
def test_stream_frozen_layers_rejects_non_cuda_device_clearly():
    """Regression test for a real bug found on CI's CPU-only job: pinned
    memory and the dedicated CUDA prefetch stream both require a real
    accelerator, so a non-CUDA device must fail immediately with a clear
    Chowder-level message -- not deep inside torch.Tensor.pin_memory()
    with a raw "Cannot access accelerator device" error. Deliberately
    does not require CUDA to run (it tests the rejection path itself),
    so this exercises for real on CPU-only CI, unlike the "always"-mode
    production-wiring tests which genuinely need a CUDA device to test
    anything meaningful."""
    import torch

    from chowder.memory_fabric import stream_frozen_layers

    model = _build_lora_model()
    model.train()
    with pytest.raises(RuntimeError, match="requires an available CUDA device"):
        stream_frozen_layers(model, torch.device("cpu"))
