from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest

from chowder.activation_offload import (
    ActivationOffloadExperiment,
    _offload_calibration_key,
    _read_cache_entry,
    _write_cache_entry,
    run_activation_offload_experiment,
)
from chowder.backends.transformers_peft import TransformersPeftExecutor, TransformersPeftRunSpec
from chowder.executors import ExecutionContext
from chowder.memory import HardwareProfile
from chowder.models import Experiment, Hypothesis

_REAL_ML_SMOKE = pytest.mark.skipif(
    os.environ.get("CHOWDER_REAL_ML_SMOKE") != "1",
    reason="real ML smoke requires CHOWDER_REAL_ML_SMOKE=1 and train dependencies",
)
_TINY_MODEL = "trl-internal-testing/tiny-LlamaForCausalLM-3.2"


def _config(**backend_overrides):
    backend = {
        "type": "transformers-peft",
        "base_model": "org/model",
        "dataset": "train.jsonl",
        "max_length": 64,
        "quantization": "none",
        "precision": "fp32",
        "lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]},
    }
    backend.update(backend_overrides)
    return {"backend": backend}


def _hardware(vram_gb=16.0, pools=(16.0,)):
    return HardwareProfile(vram_gb, 64.0, 500.0, 12.0, 40.0, 3.0, accelerator_vram_gb=pools)


def _context(hardware=None):
    return ExecutionContext(hardware or _hardware(), ".", seed=1)


def _spec(**overrides):
    fields = dict(
        base_model="org/model",
        dataset="train.jsonl",
        output_dir="/tmp/out",
        max_length=64,
        quantization="none",
        precision="fp32",
        lora_r=4,
        lora_alpha=8,
        target_modules=("q_proj", "v_proj"),
        batch_size=1,
        gradient_checkpointing=False,
    )
    fields.update(overrides)
    return TransformersPeftRunSpec(**fields)


def _fake_available(**overrides):
    measured = {
        "available": True,
        "device": "cuda",
        "batch_size": 2,
        "max_length": 64,
        "baseline_peak_vram_gb": 1.0,
        "offload_peak_vram_gb": 0.8,
        "baseline_wall_seconds": 0.1,
        "offload_wall_seconds": 0.15,
    }
    measured.update(overrides)
    return measured


def _fake_unavailable(reason="activation offload requires an available CUDA device"):
    return {"available": False, "device": "cpu", "reason": reason}


# --- calibration key ----------------------------------------------------


def test_offload_calibration_key_is_stable_for_identical_inputs():
    spec = _spec()
    context = _context()
    a = _offload_calibration_key(spec=spec, context=context, batch_size=4)
    b = _offload_calibration_key(spec=spec, context=context, batch_size=4)
    assert a == b


def test_offload_calibration_key_changes_with_batch_size():
    """Unlike memory_preflight's key, this one must be batch-size-sensitive
    -- the offload experiment measures at one specific batch size, not a
    batch-size-independent pair extrapolated afterward."""
    spec = _spec()
    context = _context()
    a = _offload_calibration_key(spec=spec, context=context, batch_size=2)
    b = _offload_calibration_key(spec=spec, context=context, batch_size=8)
    assert a != b


def test_offload_calibration_key_changes_with_base_model():
    context = _context()
    a = _offload_calibration_key(spec=_spec(base_model="org/a"), context=context, batch_size=2)
    b = _offload_calibration_key(spec=_spec(base_model="org/b"), context=context, batch_size=2)
    assert a != b


# --- cache read/write ---------------------------------------------------


def test_offload_write_then_read_cache_entry_round_trips(tmp_path):
    _write_cache_entry(tmp_path, "key1", {"baseline_peak_vram_gb": 1.0})
    assert _read_cache_entry(tmp_path, "key1") == {"baseline_peak_vram_gb": 1.0}


def test_offload_read_cache_entry_returns_none_when_file_missing(tmp_path):
    assert _read_cache_entry(tmp_path, "key1") is None


def test_offload_cache_uses_its_own_file_not_memory_preflights(tmp_path):
    """activation_offload.json and memory_calibration.json must be
    distinct files -- their cached value shapes are different and a
    shared key namespace would risk one overwriting the other."""
    _write_cache_entry(tmp_path, "key1", {"baseline_peak_vram_gb": 1.0})
    assert not (tmp_path / ".chowder" / "memory_calibration.json").exists()
    assert (tmp_path / ".chowder" / "activation_offload.json").exists()


# --- run_activation_offload_experiment orchestration (mocked worker) ----


def test_experiment_reports_unavailable_on_cpu(tmp_path):
    with patch(
        "chowder.activation_offload._run_offload_worker", return_value=_fake_unavailable()
    ) as mock_worker:
        experiment = run_activation_offload_experiment(
            resolved_config=_config(), context=_context(), work_dir=tmp_path, use_cache=False
        )
    mock_worker.assert_called_once()
    assert experiment.available is False
    assert experiment.required is False
    assert experiment.recommended is False
    assert experiment.reason is not None


def test_experiment_computes_vram_saved_and_penalty_ratio(tmp_path):
    with patch(
        "chowder.activation_offload._run_offload_worker",
        return_value=_fake_available(baseline_peak_vram_gb=2.0, offload_peak_vram_gb=1.5,
                                      baseline_wall_seconds=0.1, offload_wall_seconds=0.3),
    ):
        experiment = run_activation_offload_experiment(
            resolved_config=_config(), context=_context(), work_dir=tmp_path, use_cache=False
        )
    assert experiment.vram_saved_gb == pytest.approx(0.5)
    assert experiment.wall_time_penalty_ratio == pytest.approx(3.0)


def test_experiment_not_recommended_when_not_required_and_penalty_too_high(tmp_path):
    with patch(
        "chowder.activation_offload._run_offload_worker",
        return_value=_fake_available(baseline_peak_vram_gb=1.0, offload_peak_vram_gb=0.8,
                                      baseline_wall_seconds=0.1, offload_wall_seconds=1.0),
    ):
        experiment = run_activation_offload_experiment(
            resolved_config=_config(),
            context=_context(_hardware(vram_gb=16.0, pools=(16.0,))),
            work_dir=tmp_path,
            use_cache=False,
        )
    assert experiment.required is False
    assert experiment.wall_time_penalty_ratio == pytest.approx(10.0)
    assert experiment.recommended is False


def test_experiment_recommended_when_not_required_but_penalty_acceptable(tmp_path):
    with patch(
        "chowder.activation_offload._run_offload_worker",
        return_value=_fake_available(baseline_peak_vram_gb=1.0, offload_peak_vram_gb=0.8,
                                      baseline_wall_seconds=1.0, offload_wall_seconds=1.1),
    ):
        experiment = run_activation_offload_experiment(
            resolved_config=_config(),
            context=_context(_hardware(vram_gb=16.0, pools=(16.0,))),
            work_dir=tmp_path,
            use_cache=False,
        )
    assert experiment.required is False
    assert experiment.wall_time_penalty_ratio == pytest.approx(1.1)
    assert experiment.recommended is True


def test_experiment_required_and_recommended_when_baseline_does_not_fit(tmp_path):
    """Required overrides the penalty ratio entirely: fitting at all beats
    not fitting, however slow."""
    with patch(
        "chowder.activation_offload._run_offload_worker",
        return_value=_fake_available(baseline_peak_vram_gb=8.0, offload_peak_vram_gb=6.0,
                                      baseline_wall_seconds=0.1, offload_wall_seconds=5.0),
    ):
        experiment = run_activation_offload_experiment(
            resolved_config=_config(),
            context=_context(_hardware(vram_gb=6.5, pools=(6.5,))),
            work_dir=tmp_path,
            use_cache=False,
        )
    assert experiment.required is True
    assert experiment.wall_time_penalty_ratio == pytest.approx(50.0)
    assert experiment.recommended is True


def test_experiment_never_recommended_with_zero_or_negative_savings_even_if_required(tmp_path):
    """Offload that doesn't actually save VRAM is never worth recommending
    -- 'required' only matters when offload would actually help fit."""
    with patch(
        "chowder.activation_offload._run_offload_worker",
        return_value=_fake_available(baseline_peak_vram_gb=8.0, offload_peak_vram_gb=8.2,
                                      baseline_wall_seconds=0.1, offload_wall_seconds=0.1),
    ):
        experiment = run_activation_offload_experiment(
            resolved_config=_config(),
            context=_context(_hardware(vram_gb=6.5, pools=(6.5,))),
            work_dir=tmp_path,
            use_cache=False,
        )
    assert experiment.required is True
    assert experiment.vram_saved_gb < 0
    assert experiment.recommended is False


def test_experiment_uses_the_cache_on_a_second_call(tmp_path):
    config = _config()
    context = _context()
    with patch(
        "chowder.activation_offload._run_offload_worker", return_value=_fake_available()
    ) as mock_worker:
        first = run_activation_offload_experiment(resolved_config=config, context=context, work_dir=tmp_path)
        second = run_activation_offload_experiment(resolved_config=config, context=context, work_dir=tmp_path)
    mock_worker.assert_called_once()
    assert first.from_cache is False
    assert second.from_cache is True
    assert second.vram_saved_gb == first.vram_saved_gb


def test_experiment_cache_key_differs_by_explicit_batch_size_override(tmp_path):
    config = _config(training={"batch_size": 2})
    context = _context()
    with patch(
        "chowder.activation_offload._run_offload_worker", return_value=_fake_available()
    ) as mock_worker:
        run_activation_offload_experiment(resolved_config=config, context=context, work_dir=tmp_path, batch_size=2)
        run_activation_offload_experiment(resolved_config=config, context=context, work_dir=tmp_path, batch_size=8)
    assert mock_worker.call_count == 2


def test_experiment_explicit_batch_size_overrides_configured_one(tmp_path):
    config = _config(training={"batch_size": 2})
    captured = {}

    def _fake_worker(spec, *, batch_size, work_dir, timeout_seconds):
        captured["batch_size"] = batch_size
        return _fake_available(batch_size=batch_size)

    with patch("chowder.activation_offload._run_offload_worker", side_effect=_fake_worker):
        experiment = run_activation_offload_experiment(
            resolved_config=config, context=_context(), work_dir=tmp_path, batch_size=16, use_cache=False
        )
    assert captured["batch_size"] == 16
    assert experiment.batch_size == 16


def test_experiment_use_cache_false_always_calls_the_worker(tmp_path):
    config = _config()
    context = _context()
    with patch(
        "chowder.activation_offload._run_offload_worker", return_value=_fake_available()
    ) as mock_worker:
        run_activation_offload_experiment(resolved_config=config, context=context, work_dir=tmp_path, use_cache=False)
        run_activation_offload_experiment(resolved_config=config, context=context, work_dir=tmp_path, use_cache=False)
    assert mock_worker.call_count == 2


# --- real end-to-end (actual model load + timed forward/backward with/without offload) --


@_REAL_ML_SMOKE
def test_real_activation_offload_worker_measures_a_genuine_wall_time_penalty():
    from chowder.backends.activation_offload_worker import run_experiment

    spec = _spec(base_model=_TINY_MODEL, dataset="unused.jsonl", max_length=64)
    result = run_experiment(spec, batch_size=4)

    if not result["available"]:
        pytest.skip("no CUDA device available in this environment for a real offload comparison")
    assert result["baseline_peak_vram_gb"] > 0
    assert result["offload_peak_vram_gb"] > 0
    assert result["baseline_wall_seconds"] > 0
    assert result["offload_wall_seconds"] > 0
    # Real, warmed-up measurement on a genuine model: CPU activation
    # offload is never faster than keeping activations on-device for a
    # single-GPU forward+backward.
    assert result["offload_wall_seconds"] > result["baseline_wall_seconds"]


@_REAL_ML_SMOKE
def test_real_run_activation_offload_experiment_end_to_end(tmp_path):
    from chowder.hardware import detect_hardware
    from chowder.project_runner import hardware_profile_from_snapshot

    hardware = hardware_profile_from_snapshot(detect_hardware(str(tmp_path)))
    context = ExecutionContext(hardware, str(tmp_path), seed=7)
    config = {
        "backend": {
            "type": "transformers-peft",
            "base_model": _TINY_MODEL,
            "dataset": "unused.jsonl",
            "max_length": 64,
            "quantization": "none",
            "precision": "fp32",
            "lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]},
            "training": {"batch_size": 4},
        }
    }

    experiment = run_activation_offload_experiment(resolved_config=config, context=context, work_dir=tmp_path)
    if not experiment.available:
        pytest.skip("no CUDA device available in this environment for a real offload comparison")
    assert experiment.batch_size == 4
    assert experiment.baseline_peak_vram_gb > 0
    assert experiment.wall_time_penalty_ratio > 0
    # A tiny smoke-test model has negligible real activation pressure --
    # the correct, honest conclusion for this workload is "not worth it",
    # exercising the same recommendation logic a real large model with
    # genuine memory pressure would also go through.
    assert isinstance(experiment.recommended, bool)


# --- real hardware regression: expanded attention-bias stride corruption --
#
# Found during the Memory Fabric OOM-acceptance hardware investigation:
# training a real Qwen2.5-1.5B model with
# backend.training.activation_offload: "always" at batch_size=96,
# max_length=1024, fp32, no quantization, LoRA r=8 crashed backward with
# `RuntimeError: attn_bias is not correctly aligned (strideM).
# attn_bias.stride(2) = 66, and should be a multiple of 4` -- confirmed a
# real bug in activation_offload itself (reproduces identically alone and
# combined with optimizer_tiering/frozen_layer_streaming), never caught by
# this file's existing tests because they all use batch sizes of 2-8,
# which never selects the memory-efficient/fused attention kernel that
# enforces this alignment requirement.
#
# Root cause: transformers.masking_utils.sdpa_mask builds the 4D
# causal+padding attention bias via
# `attention_mask.expand(batch_size, -1, q_length, kv_length)` -- a
# broadcast view (stride 0 on the head dimension), never materialized into
# real memory. saved_tensors_hooks intercepts *every* tensor autograd
# saves for backward during the wrapped forward, including this one, not
# just the model's own per-layer activations. A naive `tensor.to("cpu")` /
# `.to(device)` round trip does not reproduce the broadcast strides --
# `Tensor.to()` only preserves a handful of recognized memory formats
# (contiguous, channels-last, or a plain transpose) -- so PyTorch's
# memory-efficient SDPA backward kernel receives back a differently
# -strided tensor and rejects it outright.


@_REAL_ML_SMOKE
def test_real_activation_offload_hooks_preserve_stride_of_expanded_attention_bias():
    """Direct, real-hardware reproduction of the crash and confirmation of
    the fix, against torch.nn.functional.scaled_dot_product_attention and
    the real, shipped chowder.backends.activation_offload_hooks -- no full
    model needed, since the bug is in the pack/unpack hooks themselves,
    not anything Qwen2-specific. batch=96 (matching the real crash
    exactly) is large enough to select PyTorch's memory-efficient/fused
    attention backend; kv_length=66 is deliberately not a multiple of 4,
    matching the real crash -- a fixed round max_length is often
    coincidentally aligned, but a real dynamically-padded batch is not.

    Before the fix (activation_offload_hooks.offload_pack/offload_unpack
    with a naive `.to()` round trip): this test reproduced
    `RuntimeError: attn_bias is not correctly aligned (strideM).
    attn_bias.stride(2) = 66, and should be a multiple of 4` verbatim.
    """
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device available for a real attention-bias stride comparison")

    from chowder.backends.activation_offload_hooks import offload_pack, offload_unpack

    device = "cuda"
    batch, heads, kv_heads, seq, head_dim = 96, 12, 2, 66, 128

    def build_expanded_padding_mask(lengths):
        # Mirrors transformers.masking_utils.sdpa_mask's real construction
        # (see that module's sdpa_mask function): a per-position boolean
        # causal+padding function evaluated with a singleton head dim,
        # then broadcast across heads via .expand() -- never materialized,
        # exactly the stride-0 view this test is about.
        q_arange = torch.arange(seq, device=device).view(1, 1, seq, 1)
        kv_arange = torch.arange(seq, device=device).view(1, 1, 1, seq)
        lengths_t = torch.tensor(lengths, device=device).view(batch, 1, 1, 1)
        causal = kv_arange <= q_arange
        not_padding = kv_arange < lengths_t
        return (causal & not_padding).expand(batch, heads, seq, seq)

    def build_qkv():
        torch.manual_seed(0)
        q = torch.randn(batch, heads, seq, head_dim, device=device, requires_grad=True)
        k = torch.randn(batch, kv_heads, seq, head_dim, device=device, requires_grad=True)
        v = torch.randn(batch, kv_heads, seq, head_dim, device=device, requires_grad=True)
        # GQA repeat, matching how transformers expands KV heads for SDPA.
        k = k.repeat_interleave(heads // kv_heads, dim=1).detach().requires_grad_()
        v = v.repeat_interleave(heads // kv_heads, dim=1).detach().requires_grad_()
        return q, k, v

    torch.manual_seed(0)
    lengths = torch.randint(low=seq // 2, high=seq + 1, size=(batch,)).tolist()
    mask = build_expanded_padding_mask(lengths)
    assert mask.stride(1) == 0, "test setup must produce a real broadcast (stride-0) mask"
    assert mask.stride(2) % 4 != 0, "test setup must produce a kv-length not aligned to 4"

    # Baseline: real SDPA forward+backward, no offload hooks at all.
    q_b, k_b, v_b = build_qkv()
    out_b = torch.nn.functional.scaled_dot_product_attention(q_b, k_b, v_b, attn_mask=mask)
    out_b.sum().backward()

    # With the real, shipped activation_offload hooks wrapping the same call.
    q_o, k_o, v_o = build_qkv()
    with torch.autograd.graph.saved_tensors_hooks(
        lambda t: offload_pack(t)[0], lambda p: offload_unpack(p)[0]
    ):
        out_o = torch.nn.functional.scaled_dot_product_attention(q_o, k_o, v_o, attn_mask=mask)
        out_o.sum().backward()
    torch.cuda.synchronize()

    # Value-transparent: activation_offload changes only where saved
    # tensors physically live, never the computed result.
    assert torch.allclose(out_b, out_o, atol=1e-4)
    assert torch.allclose(q_b.grad, q_o.grad, atol=1e-3)
    assert torch.allclose(k_b.grad, k_o.grad, atol=1e-3)
    assert torch.allclose(v_b.grad, v_o.grad, atol=1e-3)


@_REAL_ML_SMOKE
def test_real_tiny_llama_trains_with_activation_offload_always_and_a_padded_batch(tmp_path):
    """Full production-path confirmation that the real training entry
    point (TransformersPeftExecutor.run -> transformers_worker.train)
    actually wires activation_offload through the fixed hooks: a batch of
    genuinely different-length rows forces the real dynamic-padding
    collator to build a real attention_mask with padding, the same
    real-world condition that produces the broadcast attention bias in
    test_real_activation_offload_hooks_preserve_stride_of_expanded_attention_bias
    above (which is the one that proves the alignment/kernel-scale
    mechanism itself; this test proves the production wiring around it)."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device available for a real activation-offload training run")

    data = tmp_path / "train.jsonl"
    rows = [
        {"text": "Question: What token comes after alpha? Answer: beta"},
        {"text": "Q: red? A: blue"},
        {"text": "Question: What token comes after gamma, delta, epsilon? Answer: zeta"},
        {"text": "Q: green? A: yellow"},
        {"text": "Question: What comes after one, two, three, four, five? Answer: six"},
        {"text": "Q: up? A: down"},
        {"text": "Question: What token comes after north, south, east? Answer: west"},
        {"text": "Q: hot? A: cold"},
    ]
    data.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    config = {
        "seed": 17,
        "backend": {
            "type": "transformers-peft",
            "base_model": _TINY_MODEL,
            "dataset": str(data),
            "max_length": 64,
            "quantization": "none",
            "precision": "fp32",
            "lora": {"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"]},
            "training": {
                "epochs": 1.0,
                "learning_rate": 1e-3,
                "batch_size": 8,
                "gradient_accumulation_steps": 1,
                "logging_steps": 1,
                "gradient_checkpointing": False,
                "activation_offload": "always",
            },
            "runtime": {"timeout_seconds": 180.0},
        },
    }
    hardware = HardwareProfile(16, 64, 500, 12, 40, 3)
    context = ExecutionContext(hardware, str(tmp_path), 17, resolved_config=config)
    experiment = Experiment("e1", None, Hypothesis("obs", "cause", "fix"), {}, 2.0)

    artifact = TransformersPeftExecutor().run(experiment, context)

    assert artifact.telemetry["global_step"] > 0
    assert artifact.evidence["hardware_aware_defaults"]["resolved_activation_offload"] is True
    assert artifact.telemetry["activation_offload_bytes_transferred"] > 0
