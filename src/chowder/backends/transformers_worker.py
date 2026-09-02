from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from ..provenance import sha256_directory
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


def _verify_bound_adapter(path: str, expected_sha: str, *, label: str) -> str:
    resolved = Path(path).resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{label} adapter not found: {resolved}")
    actual = sha256_directory(resolved)
    if actual != expected_sha:
        raise RuntimeError(f"{label} adapter digest changed before worker load")
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


def _cuda_resource_snapshot(torch: Any, model: Any, trainer: Any) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "visible_accelerator_count": 0,
            "active_accelerator_count": 0,
            "active_accelerators": [],
            "peak_vram_gb_by_accelerator": {},
        }

    visible = int(torch.cuda.device_count())
    model_devices: set[int] = set()
    try:
        for parameter in model.parameters():
            device = getattr(parameter, "device", None)
            if getattr(device, "type", None) == "cuda" and device.index is not None:
                model_devices.add(int(device.index))
    except Exception:
        model_devices = set()

    world_size = int(getattr(getattr(trainer, "args", None), "world_size", 1) or 1)
    inferred = max(1, len(model_devices), world_size)
    active_count = min(visible, inferred) if visible else 0
    if model_devices:
        active_devices = sorted(model_devices)
    else:
        active_devices = list(range(active_count))

    peak_map = {
        f"cuda:{index}": float(torch.cuda.max_memory_allocated(index) / (1024 ** 3))
        for index in range(visible)
    }
    return {
        "visible_accelerator_count": visible,
        "active_accelerator_count": active_count,
        "active_accelerators": [f"cuda:{index}" for index in active_devices],
        "peak_vram_gb_by_accelerator": peak_map,
    }


def train(spec: TransformersPeftRunSpec) -> dict[str, Any] | None:
    """Returns None on non-main ranks under multi-GPU DDP -- only the main
    process (trainer.is_world_process_zero()) produces a result; see the
    rank-0 guard below for why."""
    primary_sha = _verify_bound_input(
        spec.dataset, spec.dataset_sha256, label="training"
    )
    replay_sha: str | None = None
    if spec.replay_dataset is not None:
        replay_sha = _verify_bound_input(
            spec.replay_dataset, spec.replay_sha256, label="replay"
        )
    parent_adapter_sha: str | None = None
    if spec.parent_adapter is not None:
        assert spec.parent_adapter_sha256 is not None
        parent_adapter_sha = _verify_bound_adapter(
            spec.parent_adapter,
            spec.parent_adapter_sha256,
            label="parent",
        )

    try:
        import torch
        from datasets import concatenate_datasets, load_dataset
        from peft import (
            LoraConfig,
            PeftModel,
            TaskType,
            get_peft_model,
            prepare_model_for_kbit_training,
        )
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

    base_model = AutoModelForCausalLM.from_pretrained(spec.base_model, **model_kwargs)
    if spec.quantization == "4bit":
        base_model = prepare_model_for_kbit_training(
            base_model,
            use_gradient_checkpointing=spec.gradient_checkpointing,
        )

    if spec.parent_adapter is not None:
        model = PeftModel.from_pretrained(
            base_model,
            spec.parent_adapter,
            is_trainable=True,
        )
    else:
        lora_config = LoraConfig(
            r=spec.lora_r,
            lora_alpha=spec.lora_alpha,
            lora_dropout=spec.lora_dropout,
            target_modules=list(spec.target_modules),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            use_rslora=spec.use_rslora,
        )
        model = get_peft_model(base_model, lora_config)

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
            dataset = concatenate_datasets([primary, selected_replay]).shuffle(seed=spec.seed)

    mixed_text_sha = _text_digest(dataset, spec.text_field)

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
    args_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir / "trainer"),
        "num_train_epochs": spec.epochs,
        "max_steps": spec.max_steps,
        "per_device_train_batch_size": spec.batch_size,
        "gradient_accumulation_steps": spec.gradient_accumulation_steps,
        "learning_rate": spec.learning_rate,
        "lr_scheduler_type": spec.lr_scheduler_type,
        "warmup_ratio": spec.warmup_ratio,
        "warmup_steps": spec.warmup_steps,
        "weight_decay": spec.weight_decay,
        "max_grad_norm": spec.max_grad_norm,
        "logging_steps": spec.logging_steps,
        "save_strategy": spec.save_strategy,
        "report_to": "none",
        "gradient_checkpointing": spec.gradient_checkpointing,
        "bf16": (dtype is torch.bfloat16),
        "fp16": (dtype is torch.float16),
        "seed": spec.seed,
        "data_seed": spec.seed,
    }
    if spec.save_strategy == "steps":
        args_kwargs["save_steps"] = spec.save_steps
    if spec.save_total_limit is not None:
        args_kwargs["save_total_limit"] = spec.save_total_limit
    args = TrainingArguments(**args_kwargs)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=collator,
    )

    started = time.perf_counter()
    train_output = trainer.train(resume_from_checkpoint=spec.resume_from_checkpoint)
    runtime = time.perf_counter() - started

    # Under multi-GPU DDP (accelerate launch spawns active_accelerator_count
    # real processes, one per device -- see TransformersPeftExecutor._worker_
    # command), every rank reaches this point, but only one process may write
    # the adapter/tokenizer/result files: N processes writing the same path
    # concurrently is a real corruption risk, not just wasted I/O. Trainer's
    # own train()/state are already synchronized across ranks by the time
    # .train() returns, so the non-main ranks have nothing left to do.
    if not trainer.is_world_process_zero():
        return None
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    # KNOWN LIMITATION: under multi-GPU DDP, torch.cuda.max_memory_allocated
    # is scoped to the CALLING process's own CUDA context per device -- this
    # process (rank 0) only ever allocated on its own device, so peak VRAM
    # for the other active devices reads as 0 here, not their real peak. The
    # accounting-critical fields (active/visible_accelerator_count, and
    # therefore accelerator_seconds/gpu_hours) are unaffected -- those come
    # from trainer.args.world_size, a distributed-environment property, not
    # a per-process memory query. Per-rank peak-VRAM aggregation (each rank
    # reporting its own device, rank 0 merging them) is a real follow-up,
    # not done here.
    resource_snapshot = _cuda_resource_snapshot(torch, model, trainer)

    if (
        _verify_bound_input(spec.dataset, spec.dataset_sha256, label="training")
        != primary_sha
    ):
        raise RuntimeError("training dataset changed during worker execution")
    if spec.replay_dataset is not None:
        if (
            _verify_bound_input(spec.replay_dataset, spec.replay_sha256, label="replay")
            != replay_sha
        ):
            raise RuntimeError("replay dataset changed during worker execution")
    if spec.parent_adapter is not None:
        assert spec.parent_adapter_sha256 is not None
        if (
            _verify_bound_adapter(
                spec.parent_adapter,
                spec.parent_adapter_sha256,
                label="parent",
            )
            != parent_adapter_sha
        ):
            raise RuntimeError("parent adapter changed during worker execution")

    peak_values = list(resource_snapshot["peak_vram_gb_by_accelerator"].values())
    peak_vram_gb = max(peak_values, default=0.0)
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
        "resource_usage": resource_snapshot,
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
            "continued_from_parent_adapter": parent_adapter_sha is not None,
            "parent_adapter_sha256": parent_adapter_sha,
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
    if result is not None:
        result_path = Path(args.result)
        result_path.write_text(
            json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
