from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .transformers_peft import TransformersPeftRunSpec
from .transformers_worker import _resolve_dtype, _resolve_target_modules


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

    def _measure_peak_gb(batch_size: int) -> float:
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
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        outputs.loss.backward()
        model.zero_grad(set_to_none=True)
        if device.type != "cuda":
            return 0.0
        return float(torch.cuda.max_memory_allocated(device) / (1024**3))

    peak_gb_bs1 = _measure_peak_gb(1)
    peak_gb_bs2 = _measure_peak_gb(2)

    return {
        "device": device.type,
        "frozen_params": frozen_params,
        "trainable_params": trainable_params,
        "max_length": spec.max_length,
        "peak_vram_gb_bs1": peak_gb_bs1,
        "peak_vram_gb_bs2": peak_gb_bs2,
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
