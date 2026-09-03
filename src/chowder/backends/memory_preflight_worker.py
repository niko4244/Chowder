from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .transformers_peft import TransformersPeftRunSpec
from .transformers_worker import _resolve_dtype, _resolve_target_modules


def _leaf_modules(model: Any):
    """Yield (name, module) for every named module with no children of its
    own. Under a PEFT-wrapped model this naturally lands one level below
    each LoRA wrapper -- base_layer, lora_A, lora_B, lora_dropout are each
    their own leaf -- which is finer-grained than "one entry per attention
    projection" but is an honest, unambiguous rule rather than guessing at
    which wrapping level is semantically "the layer". Any coarser
    aggregation (e.g. by common name prefix) is left to the caller.
    """
    for name, module in model.named_modules():
        if name and next(module.children(), None) is None:
            yield name, module


def load_dry_run_model(spec: TransformersPeftRunSpec) -> tuple[Any, Any, Any, int, int]:
    """Load the real model with the configured quantization/precision/LoRA
    recipe -- the exact same real loading path `dry_run` uses -- and return
    (model, tokenizer, device, frozen_params, trainable_params). Shared by
    every dry-run-style worker in this module (memory/telemetry dry_run,
    the activation-offload experiment) so each one measures against an
    identical real model construction rather than a slightly-diverged copy
    of the same loading logic. Deliberately NOT shared with the real
    training path in transformers_worker.py -- that code is battle-tested
    production logic and refactoring it to share with dry-run experiments
    would be a real regression risk for no benefit.

    Never trains anything and never touches the configured dataset --
    accepts a full TransformersPeftRunSpec (the same shape the real trainer
    uses) purely for convenience/consistency, but `dataset`/checkpoint/
    save-strategy fields on it are irrelevant here and ignored.
    """
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            "Transformers backend dependencies are missing; install chowder-ai[train] "
            "and chowder-ai[qlora] when using 4-bit quantization"
        ) from exc

    if spec.trust_remote_code:
        raise RuntimeError("trust_remote_code is disabled")

    dtype = _resolve_dtype(torch, spec.precision)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        spec.base_model,
        revision=spec.revision,
        trust_remote_code=False,
        local_files_only=spec.offline,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError(
                "tokenizer has neither pad_token nor eos_token; explicit tokenizer support is required"
            )
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": False,
        "dtype": dtype,
        "local_files_only": spec.offline,
    }
    if spec.revision is not None:
        model_kwargs["revision"] = spec.revision
    if spec.quantization == "4bit":
        if device.type != "cuda":
            raise RuntimeError("4-bit quantization requires an available CUDA device")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        model_kwargs["device_map"] = 0

    base_model = AutoModelForCausalLM.from_pretrained(spec.base_model, **model_kwargs)
    if spec.quantization == "4bit":
        base_model = prepare_model_for_kbit_training(
            base_model, use_gradient_checkpointing=spec.gradient_checkpointing
        )
    else:
        base_model = base_model.to(device)

    target_modules = _resolve_target_modules(
        base_model, explicit=spec.target_modules, preset=spec.target_preset
    )
    lora_config = LoraConfig(
        r=spec.lora_r,
        lora_alpha=spec.lora_alpha,
        lora_dropout=spec.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        use_rslora=spec.use_rslora,
    )
    model = get_peft_model(base_model, lora_config)
    if spec.gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
    model.train()

    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, tokenizer, device, frozen_params, trainable_params


def dry_run(spec: TransformersPeftRunSpec) -> dict[str, Any]:
    """Load the real model (with the configured quantization/precision/LoRA
    recipe) and run two tiny forward+backward steps -- batch size 1 and 2,
    both at the configured max_length -- to measure real peak VRAM twice.

    Two real measurements, not one, on purpose: the difference between them
    is the actual per-example activation cost this exact model/recipe/
    hardware combination produces, which the caller can then extrapolate to
    any configured batch size linearly. That's a measured, empirical slope,
    not a theoretical estimate of how attention/activation memory should
    scale -- deliberately avoiding having to model architecture-specific
    scaling behavior (flash attention vs. not, sequence-length quadratic
    terms, etc.) that would be a guess dressed up as a formula.

    Also collects, on the batch-size-2 pass, real per-leaf-module telemetry
    (real forward-hook-measured activation output size plus that module's
    own direct trainable/frozen parameter count) via
    `_leaf_modules`/forward hooks, and a real optimizer-state footprint: a
    genuine `torch.optim.AdamW` over the trainable parameters, stepped
    once, then the actual state tensors it allocated (exp_avg, exp_avg_sq,
    ...) are summed directly -- not inferred from a memory-delta
    measurement, which would report 0 on CPU (no CUDA allocation to
    measure a delta in) and would fold in CUDA-allocator block-rounding
    noise on GPU. Reading the real tensors PyTorch created is exact and
    device-agnostic.

    Never trains anything and never touches the configured dataset --
    accepts a full TransformersPeftRunSpec (the same shape the real trainer
    uses) purely for convenience/consistency, but `dataset`/checkpoint/
    save-strategy fields on it are irrelevant here and ignored.
    """
    model, tokenizer, device, frozen_params, trainable_params = load_dry_run_model(spec)
    # Safe to import bare here (no try/except needed): load_dry_run_model
    # already imported torch successfully -- if it hadn't, it would have
    # raised its own RuntimeError with a clear message before reaching
    # this line, and Python caches the successful import.
    import torch

    layer_telemetry: list[dict[str, Any]] = []

    def _make_layer_hook(name: str, module: Any):
        def _hook(mod: Any, _inputs: Any, output: Any) -> None:
            tensors = output if isinstance(output, tuple) else (output,)
            activation_bytes = sum(
                t.numel() * t.element_size() for t in tensors if torch.is_tensor(t)
            )
            layer_telemetry.append(
                {
                    "name": name,
                    "module_type": type(mod).__name__,
                    "trainable_params": sum(
                        p.numel() for p in mod.parameters(recurse=False) if p.requires_grad
                    ),
                    "frozen_params": sum(
                        p.numel() for p in mod.parameters(recurse=False) if not p.requires_grad
                    ),
                    "activation_bytes": activation_bytes,
                }
            )

        return _hook

    def _measure_peak_gb(batch_size: int, *, collect_layers: bool = False) -> float:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        input_ids = torch.randint(
            low=0,
            high=max(tokenizer.vocab_size, 2),
            size=(batch_size, spec.max_length),
            device=device,
        )
        attention_mask = torch.ones_like(input_ids)
        hooks = []
        if collect_layers:
            layer_telemetry.clear()
            hooks = [module.register_forward_hook(_make_layer_hook(name, module)) for name, module in _leaf_modules(model)]
        try:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            outputs.loss.backward()
        finally:
            for hook in hooks:
                hook.remove()
        model.zero_grad(set_to_none=True)
        if device.type != "cuda":
            return 0.0
        return float(torch.cuda.max_memory_allocated(device) / (1024**3))

    peak_gb_bs1 = _measure_peak_gb(1)
    peak_gb_bs2 = _measure_peak_gb(2, collect_layers=True)

    optimizer_state_bytes = 0
    trainable_param_list = [p for p in model.parameters() if p.requires_grad]
    if trainable_param_list:
        optimizer = torch.optim.AdamW(trainable_param_list, lr=1e-4)
        input_ids = torch.randint(
            low=0, high=max(tokenizer.vocab_size, 2), size=(1, spec.max_length), device=device
        )
        attention_mask = torch.ones_like(input_ids)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        outputs.loss.backward()
        optimizer.step()
        # Sum the actual state tensors PyTorch allocated (exp_avg,
        # exp_avg_sq, ...) rather than inferring a byte count from a
        # memory-delta measurement: this is exact and device-agnostic --
        # it works identically on CPU and CUDA, where a CUDA-allocated-
        # memory-delta trick would report 0 on CPU (there is no CUDA
        # allocation to measure a delta in).
        optimizer_state_bytes = sum(
            tensor.numel() * tensor.element_size()
            for state in optimizer.state.values()
            for tensor in state.values()
            if torch.is_tensor(tensor)
        )
        optimizer.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)

    return {
        "device": device.type,
        "frozen_params": frozen_params,
        "trainable_params": trainable_params,
        "max_length": spec.max_length,
        "peak_vram_gb_bs1": peak_gb_bs1,
        "peak_vram_gb_bs2": peak_gb_bs2,
        "layer_telemetry": layer_telemetry,
        "optimizer_state_bytes": optimizer_state_bytes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()

    spec_data = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    spec = TransformersPeftRunSpec(**spec_data)
    result = dry_run(spec)
    Path(args.result).write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
