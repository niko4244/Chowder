from __future__ import annotations

import argparse
import hashlib
import json
import math
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


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_bound_input(path: str, expected_sha: str | None, *, label: str) -> str:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} dataset not found: {resolved}")
    actual = _sha256_file(resolved)
    if expected_sha is not None and actual != expected_sha:
        raise RuntimeError(f"{label} dataset digest changed before worker load")
    return actual


def _replay_sample_count(primary_rows: int, replay_rows: int, ratio: float) -> int:
    if primary_rows < 0 or replay_rows < 0:
        raise ValueError("dataset row counts cannot be negative")
    if replay_rows == 0 or primary_rows == 0:
        return 0
    if not math.isfinite(float(ratio)) or ratio <= 0:
        raise ValueError("replay ratio must be finite and positive")
    return min(replay_rows, max(1, math.ceil(primary_rows * float(ratio))))


def _text_digest(dataset: Any, text_field: str) -> str:
    digest = hashlib.sha256()
    for index in range(len(dataset)):
        text = str(dataset[index][text_field]).encode("utf-8")
        digest.update(len(text).to_bytes(8, "big"))
        digest.update(text)
    return digest.hexdigest()


def train(spec: TransformersPeftRunSpec) -> dict[str, Any]:
    # Verify immutable data bindings before importing/downloading heavyweight
    # model dependencies. The controller performs the same check immediately
    # before spawning us, closing the proposal→worker boundary on both sides.
    primary_sha = _verify_bound_input(
        spec.dataset, spec.dataset_sha256, label="training"
    )
    replay_sha: str | None = None
    if spec.replay_dataset is not None:
        replay_sha = _verify_bound_input(
            spec.replay_dataset, spec.replay_sha256, label="replay"
        )

    try:
        import torch
        from datasets import concatenate_datasets, load_dataset
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

    primary = load_dataset("json", data_files=spec.dataset, split="train")
    if spec.text_field not in primary.column_names:
        raise RuntimeError(
            f"dataset is missing text field {spec.text_field!r}; columns={primary.column_names}"
        )
    primary = primary.select_columns([spec.text_field])
    primary_rows = len(primary)
    if primary_rows == 0:
        raise RuntimeError("training dataset contains no rows")

    replay_available_rows = 0
    replay_selected_rows = 0
    dataset = primary
    if spec.replay_dataset is not None:
        replay = load_dataset("json", data_files=spec.replay_dataset, split="train")
        if spec.text_field not in replay.column_names:
            raise RuntimeError(
                f"replay dataset is missing text field {spec.text_field!r}; columns={replay.column_names}"
            )
        replay = replay.select_columns([spec.text_field])
        replay_available_rows = len(replay)
        replay_selected_rows = _replay_sample_count(
            primary_rows, replay_available_rows, spec.replay_ratio
        )
        if replay_selected_rows:
            selected_replay = replay.shuffle(seed=spec.seed).select(
                range(replay_selected_rows)
            )
            dataset = concatenate_datasets([primary, selected_replay]).shuffle(
                seed=spec.seed
            )

    mixed_text_sha = _text_digest(dataset, spec.text_field)

    def tokenize(batch):
        return tokenizer(
            batch[spec.text_field],
            truncation=True,
            max_length=spec.max_length,
            padding=False,
        )

    tokenized = dataset.map(
        tokenize, batched=True, remove_columns=dataset.column_names
    )
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

    # Reverify once more after training. A mutation during the run invalidates
    # the artifact instead of silently changing what its provenance means.
    if _verify_bound_input(spec.dataset, spec.dataset_sha256, label="training") != primary_sha:
        raise RuntimeError("training dataset changed during worker execution")
    if spec.replay_dataset is not None:
        if _verify_bound_input(
            spec.replay_dataset, spec.replay_sha256, label="replay"
        ) != replay_sha:
            raise RuntimeError("replay dataset changed during worker execution")

    return {
        "telemetry": {
            "train_loss": float(train_output.training_loss),
            "global_step": int(trainer.state.global_step),
            "train_runtime_seconds": float(runtime),
            "peak_vram_gb": float(peak_vram_gb),
            "primary_rows": primary_rows,
            "replay_selected_rows": replay_selected_rows,
            "training_rows": len(dataset),
        },
        "data_provenance": {
            "primary_dataset_sha256": primary_sha,
            "primary_rows": primary_rows,
            "replay_dataset_sha256": replay_sha,
            "replay_available_rows": replay_available_rows,
            "replay_selected_rows": replay_selected_rows,
            "replay_ratio": spec.replay_ratio,
            "selection_seed": spec.seed,
            "mixed_training_text_sha256": mixed_text_sha,
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
    result_path.write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
