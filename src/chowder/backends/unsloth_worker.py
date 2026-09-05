"""Isolated Unsloth training worker.

Deliberately self-contained: this script runs under the isolated
interpreter at .chowder/envs/unsloth/{Scripts,bin}/python, which does NOT
have the `chowder` package installed (see docs/UNSLOTH.md -- the isolated
environment intentionally shares no packages with Chowder's own tested
Transformers/PEFT/TRL stack). It is invoked by absolute file path
(`<isolated-python> unsloth_worker.py --spec ... --result ...`), never as
`-m chowder.backends.unsloth_worker`, and must not import anything from the
`chowder` package. This is why it cannot yet reuse
chowder.backends.training_data's chat-tokenization contract -- text-format
datasets only in this initial slice; chat-format support is a follow-up
once that cross-environment handoff is designed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Unlike plain PEFT's LoraConfig(target_modules=None), Unsloth's own
# FastLanguageModel.get_peft_model does not auto-detect target modules --
# confirmed directly on real hardware (RTX 5060 Ti): passing None raises
# `TypeError: 'NoneType' object is not iterable` inside unsloth's own
# get_peft_model (it iterates target_modules unconditionally, with no
# None-means-auto-detect path the way plain PEFT's LoraConfig has). This
# is Unsloth's own documented default target list for its supported
# Llama-family architectures (Llama/Mistral/Qwen/Gemma), used here only
# when the recipe does not specify one explicitly.
_DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(frozen=True)
class _Spec:
    base_model: str
    dataset: str
    output_dir: str
    dataset_sha256: str | None
    revision: str | None
    text_field: str
    max_length: int
    epochs: float
    max_steps: int
    learning_rate: float
    batch_size: int
    gradient_accumulation_steps: int
    logging_steps: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: list[str]
    quantization: str
    seed: int
    timeout_seconds: float | None
    offline: bool


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_bound_input(path: str, expected_sha: str | None) -> str:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"training dataset not found: {resolved}")
    actual = _sha256_file(resolved)
    if expected_sha is not None and actual != expected_sha:
        raise RuntimeError("training dataset digest changed before worker load")
    return actual


def train(spec: _Spec) -> dict[str, Any]:
    from unsloth import FastLanguageModel

    _verify_bound_input(spec.dataset, spec.dataset_sha256)

    import torch
    from datasets import load_dataset
    from transformers import (
        DataCollatorForLanguageModeling,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        set_seed,
    )

    set_seed(spec.seed)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=spec.base_model,
        revision=spec.revision,
        max_seq_length=spec.max_length,
        dtype=None,
        load_in_4bit=(spec.quantization == "4bit"),
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("tokenizer has neither pad_token nor eos_token")
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r=spec.lora_r,
        target_modules=list(spec.target_modules) or list(_DEFAULT_TARGET_MODULES),
        lora_alpha=spec.lora_alpha,
        lora_dropout=spec.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=spec.seed,
    )
    # Audit the actual trainable module names Unsloth/PEFT resolved for this
    # model, rather than assuming a preset -- recorded in evidence so a
    # config that silently matched zero real modules is visible, not silent.
    resolved_target_modules = sorted(model.peft_config[model.active_adapter].target_modules)

    dataset = load_dataset("json", data_files=spec.dataset, split="train")
    if spec.text_field not in dataset.column_names:
        raise RuntimeError(
            f"dataset is missing text field {spec.text_field!r}; "
            f"columns={dataset.column_names}"
        )
    dataset = dataset.select_columns([spec.text_field])
    if len(dataset) == 0:
        raise RuntimeError("training dataset contains no rows")

    def tokenize(batch: dict[str, Any]) -> dict[str, Any]:
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
    progress_path = output_dir / "progress.json"
    started = time.perf_counter()

    class _ProgressReportingCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not state.is_world_process_zero or not logs or "loss" not in logs:
                return
            payload = {
                "step": state.global_step,
                "max_steps": state.max_steps if state.max_steps and state.max_steps > 0 else None,
                "epoch": logs.get("epoch", state.epoch),
                "loss": logs.get("loss"),
                "learning_rate": logs.get("learning_rate"),
                "wall_seconds": time.perf_counter() - started,
            }
            tmp_path = progress_path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(payload), encoding="utf-8")
            tmp_path.replace(progress_path)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "trainer"),
        num_train_epochs=spec.epochs,
        max_steps=spec.max_steps,
        per_device_train_batch_size=spec.batch_size,
        gradient_accumulation_steps=spec.gradient_accumulation_steps,
        learning_rate=spec.learning_rate,
        logging_steps=spec.logging_steps,
        save_strategy="no",
        report_to="none",
        seed=spec.seed,
        data_seed=spec.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
        callbacks=[_ProgressReportingCallback()],
    )
    train_output = trainer.train()
    runtime = time.perf_counter() - started

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    peak_vram_gb = 0.0
    active_count = 0
    if torch.cuda.is_available():
        active_count = 1
        peak_vram_gb = float(torch.cuda.max_memory_allocated(0) / (1024**3))

    def _package_version(name: str) -> str:
        try:
            from importlib.metadata import version

            return version(name)
        except Exception:
            return "unknown"

    return {
        "telemetry": {
            "train_loss": float(train_output.training_loss),
            "global_step": int(trainer.state.global_step),
            "train_runtime_seconds": float(runtime),
            "peak_vram_gb": peak_vram_gb,
            "training_rows": len(dataset),
        },
        "resolved_target_modules": resolved_target_modules,
        "resource_usage": {
            "active_accelerator_count": active_count,
            "visible_accelerator_count": active_count,
            "peak_vram_gb_by_accelerator": (
                {"cuda:0": peak_vram_gb} if active_count else {}
            ),
        },
        "model_provenance": {
            "requested_base_model": spec.base_model,
            "requested_revision": spec.revision,
        },
        "versions": {
            "unsloth": _package_version("unsloth"),
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
            "peft": _package_version("peft"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()

    raw = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    spec = _Spec(
        base_model=raw["base_model"],
        dataset=raw["dataset"],
        output_dir=raw["output_dir"],
        dataset_sha256=raw.get("dataset_sha256"),
        revision=raw.get("revision"),
        text_field=raw.get("text_field", "text"),
        max_length=int(raw.get("max_length", 512)),
        epochs=float(raw.get("epochs", 1.0)),
        max_steps=int(raw.get("max_steps", -1)),
        learning_rate=float(raw.get("learning_rate", 2e-4)),
        batch_size=int(raw.get("batch_size", 1)),
        gradient_accumulation_steps=int(raw.get("gradient_accumulation_steps", 4)),
        logging_steps=int(raw.get("logging_steps", 10)),
        lora_r=int(raw.get("lora_r", 16)),
        lora_alpha=int(raw.get("lora_alpha", 32)),
        lora_dropout=float(raw.get("lora_dropout", 0.05)),
        target_modules=list(raw.get("target_modules", [])),
        quantization=raw.get("quantization", "none"),
        seed=int(raw.get("seed", 1)),
        timeout_seconds=raw.get("timeout_seconds"),
        offline=bool(raw.get("offline", False)),
    )
    result = train(spec)
    Path(args.result).write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
