from __future__ import annotations

import argparse
import json
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .transformers_peft import TransformersPeftRunSpec


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _resolve_dtype(torch: Any, precision: str):
    if precision == "fp32":
        return torch.float32
    if precision == "bf16":
        if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 requested but the active CUDA device does not support bf16")
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def train(spec: TransformersPeftRunSpec) -> dict[str, Any]:
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Transformers backend dependencies are missing; install chowder-ai[train] "
            "and chowder-ai[qlora] when using 4-bit quantization"
        ) from exc

    if spec.trust_remote_code:
        raise RuntimeError("trust_remote_code is disabled")

    dtype = _resolve_dtype(torch, spec.precision)
    if spec.quantization == "4bit" and not torch.cuda.is_available():
        raise RuntimeError("initial 4-bit QLoRA backend requires an available CUDA device")

    set_seed(spec.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        spec.base_model, revision=spec.revision, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError(
                "tokenizer has neither pad_token nor eos_token; explicit tokenizer support is required"
            )
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {"trust_remote_code": False, "dtype": dtype}
    if spec.revision is not None:
        model_kwargs["revision"] = spec.revision
    if spec.quantization == "4bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        model_kwargs["device_map"] = 0

    model = AutoModelForCausalLM.from_pretrained(spec.base_model, **model_kwargs)
    if spec.quantization == "4bit":
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=spec.gradient_checkpointing,
        )

    lora_config = LoraConfig(
        r=spec.lora_r,
        lora_alpha=spec.lora_alpha,
        lora_dropout=spec.lora_dropout,
        target_modules=list(spec.target_modules),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        use_rslora=spec.use_rslora,
    )
    model = get_peft_model(model, lora_config)
    if spec.gradient_checkpointing:
        model.config.use_cache = False

    dataset = load_dataset("json", data_files=spec.dataset, split="train")
    if spec.text_field not in dataset.column_names:
        raise RuntimeError(
            f"dataset is missing text field {spec.text_field!r}; columns={dataset.column_names}"
        )

    def tokenize(batch):
        return tokenizer(
            batch[spec.text_field],
            truncation=True,
            max_length=spec.max_length,
            padding=False,
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    output_dir = Path(spec.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(output_dir / "trainer"),
        num_train_epochs=spec.epochs,
        per_device_train_batch_size=spec.batch_size,
        gradient_accumulation_steps=spec.gradient_accumulation_steps,
        learning_rate=spec.learning_rate,
        logging_steps=spec.logging_steps,
        save_strategy="no",
        report_to="none",
        gradient_checkpointing=spec.gradient_checkpointing,
        bf16=(dtype is torch.bfloat16),
        fp16=(dtype is torch.float16),
        seed=spec.seed,
        data_seed=spec.seed,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=collator,
    )

    started = time.perf_counter()
    train_output = trainer.train()
    runtime = time.perf_counter() - started
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    if torch.cuda.is_available():
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    else:
        peak_vram_gb = 0.0

    return {
        "telemetry": {
            "train_loss": float(train_output.training_loss),
            "global_step": int(trainer.state.global_step),
            "train_runtime_seconds": float(runtime),
            "peak_vram_gb": float(peak_vram_gb),
        },
        "provenance": {
            "requested_base_model": spec.base_model,
            "requested_revision": spec.revision,
            "resolved_model_commit": getattr(model.config, "_commit_hash", None),
        },
        "versions": {
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
            "peft": _package_version("peft"),
            "datasets": _package_version("datasets"),
            "accelerate": _package_version("accelerate"),
            "bitsandbytes": _package_version("bitsandbytes"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()

    spec_data = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    spec = TransformersPeftRunSpec(**spec_data)
    result = train(spec)
    result_path = Path(args.result)
    result_path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
