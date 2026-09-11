"""Isolated Unsloth training worker.

Deliberately self-contained: this script runs under the isolated
interpreter at .chowder/envs/unsloth/{Scripts,bin}/python, which does NOT
have the `chowder` package installed (see docs/UNSLOTH.md -- the isolated
environment intentionally shares no packages with Chowder's own tested
Transformers/PEFT/TRL stack). It is invoked by absolute file path
(`<isolated-python> unsloth_worker.py --spec ... --result ...`), never as
`-m chowder.backends.unsloth_worker`, and must not import anything from the
`chowder` package. This is why chat-format datasets are never tokenized
here: `unsloth_peft.py` (controller-side, where `chowder.backends.
training_data`'s chat-tokenization contract IS importable) pre-renders
every row into `{input_ids, attention_mask, labels}` using that exact
shared contract before handoff, so this worker's chat path is just
"load already-tokenized rows" -- spec.pretokenized=True means the dataset
file has those three columns already and no chat template, message
validation, or assistant-masking logic exists in this file at all.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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
    parent_adapter: str | None
    parent_adapter_sha256: str | None
    replay_dataset: str | None
    replay_sha256: str | None
    replay_ratio: float
    text_field: str
    pretokenized: bool
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
    save_strategy: str
    save_steps: int
    save_total_limit: int | None
    resume_from_checkpoint: str | None


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_bound_input(path: str, expected_sha: str | None, *, label: str = "training") -> str:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} dataset not found: {resolved}")
    actual = _sha256_file(resolved)
    if expected_sha is not None and actual != expected_sha:
        raise RuntimeError(f"{label} dataset digest changed before worker load")
    return actual


def _replay_sample_count(primary_rows: int, replay_rows: int, ratio: float) -> int:
    """Local mirror of chowder.backends.training_data._replay_sample_count
    -- see that module for the exact rationale; this file cannot import it
    in the isolated environment."""
    if primary_rows < 0 or replay_rows < 0:
        raise ValueError("dataset row counts cannot be negative")
    if replay_rows == 0 or primary_rows == 0:
        return 0
    if not math.isfinite(float(ratio)) or ratio <= 0:
        raise ValueError("replay ratio must be finite and positive")
    return min(replay_rows, max(1, math.ceil(primary_rows * float(ratio))))


def _sha256_directory(path: str | Path) -> str:
    """Local mirror of chowder.provenance.sha256_directory -- this file is
    deliberately self-contained (see module docstring) and cannot import
    chowder in the isolated environment."""
    root = Path(path).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    digest = hashlib.sha256()
    entries = sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
    for entry in entries:
        if entry.is_symlink():
            raise ValueError(f"artifact directory contains unsupported symlink: {entry}")
        if not entry.is_file():
            continue
        relative = entry.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(b"\0")
        with entry.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _verify_bound_adapter(path: str, expected_sha: str) -> str:
    resolved = Path(path).resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"parent adapter not found: {resolved}")
    actual = _sha256_directory(resolved)
    if actual != expected_sha:
        raise RuntimeError("parent adapter digest changed before worker load")
    return actual


def _load_text_dataset_with_replay(dataset: Any, spec: _Spec) -> tuple[Any, int, int]:
    """Select spec.text_field from the already-loaded primary `dataset`,
    then mix in a sampled replay slice before tokenization -- exactly
    mirroring transformers_worker.py's text path (sample up to
    replay_ratio * primary_rows real replay rows, never more than are
    actually available, concatenate, then reshuffle so replay rows are not
    clustered at the end of an epoch).

    Extracted from train() as its own pure function (real `datasets`
    objects in, real `datasets` objects out -- no torch/unsloth/transformers
    needed) specifically so this real row-mixing logic is testable without
    driving the full Trainer lifecycle: `datasets` is an ordinary, directly
    importable package, unlike `transformers`, whose top-level `Trainer`
    resolution is guarded by a `_LazyModule` that caches its first real
    resolution process-wide -- confirmed for real to make monkeypatching
    `transformers.Trainer` order-dependent on whatever other tests already
    ran in the same process, which is not a foundation to build a test on.

    Returns (dataset, replay_available_rows, replay_selected_rows).
    """
    from datasets import concatenate_datasets, load_dataset

    if spec.text_field not in dataset.column_names:
        raise RuntimeError(
            f"dataset is missing text field {spec.text_field!r}; columns={dataset.column_names}"
        )
    primary = dataset.select_columns([spec.text_field])
    primary_rows = len(primary)

    merged = primary
    replay_available_rows = 0
    replay_selected_rows = 0
    if spec.replay_dataset is not None:
        replay = load_dataset("json", data_files=spec.replay_dataset, split="train")
        if spec.text_field not in replay.column_names:
            raise RuntimeError(
                f"replay dataset is missing text field {spec.text_field!r}; "
                f"columns={replay.column_names}"
            )
        replay = replay.select_columns([spec.text_field])
        replay_available_rows = len(replay)
        replay_selected_rows = _replay_sample_count(
            primary_rows, replay_available_rows, spec.replay_ratio
        )
        if replay_selected_rows:
            selected_replay = replay.shuffle(seed=spec.seed).select(range(replay_selected_rows))
            merged = concatenate_datasets([primary, selected_replay]).shuffle(seed=spec.seed)

    return merged, replay_available_rows, replay_selected_rows


def train(spec: _Spec) -> dict[str, Any]:
    from unsloth import FastLanguageModel

    _verify_bound_input(spec.dataset, spec.dataset_sha256)
    if spec.parent_adapter is not None:
        assert spec.parent_adapter_sha256 is not None
        _verify_bound_adapter(spec.parent_adapter, spec.parent_adapter_sha256)
    if spec.replay_dataset is not None:
        _verify_bound_input(spec.replay_dataset, spec.replay_sha256, label="replay")

    import torch
    from datasets import load_dataset
    from transformers import (
        DataCollatorForLanguageModeling,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        set_seed,
    )

    set_seed(spec.seed)

    # text_only=True makes Unsloth load the family's TEXT decoder class instead of
    # a *ForConditionalGeneration wrapper. That is what this backend wants on two
    # counts. It is a text trainer (text_field / max_length / chat templates; the
    # vision tower is never trained), and -- the reason this is a fix rather than a
    # preference -- the adapter it saves must be loadable by Chowder's evaluator,
    # which loads AutoModelForCausalLM. Without it, a VLM-wrapped model puts the
    # decoder under `model.language_model.layers` while the evaluator's model has
    # `model.layers`, so NO adapter key matches: PEFT warns, loads nothing, leaves
    # every LoRA B at zero, and the candidate silently scores as the base model.
    # Measured on Qwen3.8-9B: max logit delta 0.000000 versus 14.5 for the same
    # training under the Transformers engine.
    #
    # Unsloth does the remapping itself (_apply_text_only_key_mapping) and applies
    # it only when the text config belongs to the same family
    # (_is_family_text_decoder: "qwen3_5_text".startswith("qwen3_5")), keeping the
    # full model otherwise rather than loading random weights -- so this is safe to
    # pass unconditionally. Version-guarded because an Unsloth without the
    # parameter would raise on an unexpected kwarg; when it is absent the flag is
    # recorded in the result so an operator can explain a liveness refusal.
    _load_kwargs: dict[str, Any] = {
        "model_name": spec.base_model,
        "revision": spec.revision,
        "max_seq_length": spec.max_length,
        "dtype": None,
        "load_in_4bit": (spec.quantization == "4bit"),
    }
    import inspect as _inspect

    text_only_supported = "text_only" in _inspect.signature(
        FastLanguageModel.from_pretrained
    ).parameters
    if text_only_supported:
        _load_kwargs["text_only"] = True
    model, tokenizer = FastLanguageModel.from_pretrained(**_load_kwargs)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("tokenizer has neither pad_token nor eos_token")
        tokenizer.pad_token = tokenizer.eos_token

    if spec.parent_adapter is not None:
        # Continuation: load the exact verified parent adapter onto the
        # Unsloth-loaded base model instead of creating a fresh LoRA adapter.
        # An Unsloth-loaded model is a real transformers-compatible model
        # underneath, so plain PEFT's own loader works on it directly --
        # mirrors transformers_worker.py's identical
        # PeftModel.from_pretrained(base_model, spec.parent_adapter,
        # is_trainable=True) continuation path exactly, so both backends
        # continue a lineage the same way.
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, spec.parent_adapter, is_trainable=True)
        # Same guard as chowder.adapter_guard, inlined: this file must not
        # import from the chowder package (see the module docstring), and a
        # parent adapter that silently fails to load would turn a "continued"
        # run into a fresh one while provenance claimed continuity. PEFT only
        # warns on a total key mismatch and leaves every LoRA B at zero.
        _b = [q for n, q in model.named_parameters() if "lora_B" in n]
        if _b and not any(float(q.detach().float().abs().max()) > 0.0 for q in _b):
            raise RuntimeError(
                f"parent adapter {spec.parent_adapter} loaded but all "
                f"{len(_b)} LoRA B matrices are exactly zero, so it is an "
                "identity and this run would silently start from scratch"
            )
    else:
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
    # Populated identically by get_peft_model and PeftModel.from_pretrained.
    resolved_target_modules = sorted(model.peft_config[model.active_adapter].target_modules)
    # Counted the same way chowder.target_coverage does, inlined because this
    # file must not import from the chowder package (see the module docstring).
    # Keys off ".lora_A" so the adapter name does not matter. The controller
    # compares this against the requested list: Unsloth rewrites the list into a
    # regex, which on a hybrid model silently missed 72 linear_attn modules.
    _targets: set[str] = set()
    for _name, _ in model.named_modules():
        _i = _name.find(".lora_A")
        if _i > 0:
            _targets.add(_name[:_i])
    _adapted: dict[str, int] = {}
    for _t in _targets:
        _leaf = _t.rsplit(".", 1)[-1]
        if _leaf:
            _adapted[_leaf] = _adapted.get(_leaf, 0) + 1

    dataset = load_dataset("json", data_files=spec.dataset, split="train")
    if len(dataset) == 0:
        raise RuntimeError("training dataset contains no rows")

    # Chat-format replay is already merged into this file by unsloth_peft.py
    # before handoff; these stay 0 in that branch on purpose -- only the
    # text-format branch below performs its own replay merging.
    replay_available_rows = 0
    replay_selected_rows = 0

    if spec.pretokenized:
        # Chat-format handoff: unsloth_peft.py already rendered every row
        # through chowder.backends.training_data's shared chat-tokenization
        # contract (the same one transformers_worker.py uses), so these
        # columns are already real input_ids/attention_mask/completion-only
        # labels -- no text_field, no chat template, no masking logic here.
        required = {"input_ids", "attention_mask", "labels"}
        if not required.issubset(dataset.column_names):
            raise RuntimeError(
                f"pretokenized dataset is missing required columns; "
                f"need {sorted(required)}, have {dataset.column_names}"
            )
        tokenized = dataset.select_columns(sorted(required))
        collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=None, label_pad_token_id=-100, padding=True
        )
    else:
        if spec.text_field not in dataset.column_names:
            raise RuntimeError(
                f"dataset is missing text field {spec.text_field!r}; "
                f"columns={dataset.column_names}"
            )
        dataset, replay_available_rows, replay_selected_rows = _load_text_dataset_with_replay(
            dataset, spec
        )

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

    args_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir / "trainer"),
        "num_train_epochs": spec.epochs,
        "max_steps": spec.max_steps,
        "per_device_train_batch_size": spec.batch_size,
        "gradient_accumulation_steps": spec.gradient_accumulation_steps,
        "learning_rate": spec.learning_rate,
        "logging_steps": spec.logging_steps,
        "save_strategy": spec.save_strategy,
        "report_to": "none",
        "seed": spec.seed,
        "data_seed": spec.seed,
    }
    if spec.save_strategy == "steps":
        args_kwargs["save_steps"] = spec.save_steps
    if spec.save_total_limit is not None:
        args_kwargs["save_total_limit"] = spec.save_total_limit
    training_args = TrainingArguments(**args_kwargs)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
        callbacks=[_ProgressReportingCallback()],
    )
    train_output = trainer.train(resume_from_checkpoint=spec.resume_from_checkpoint)
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
            "replay_available_rows": replay_available_rows,
            "replay_selected_rows": replay_selected_rows,
        },
        "resolved_target_modules": resolved_target_modules,
        "adapted_modules_by_leaf": _adapted,
        # Whether the text-decoder class was requested. False means this
        # Unsloth build predates the parameter, and an adapter trained on a
        # VLM-wrapped model will not load into Chowder's evaluator.
        "text_only_requested": text_only_supported,
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
            "continued_from_parent_adapter": spec.parent_adapter is not None,
            "parent_adapter_sha256": spec.parent_adapter_sha256,
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
        parent_adapter=raw.get("parent_adapter"),
        parent_adapter_sha256=raw.get("parent_adapter_sha256"),
        replay_dataset=raw.get("replay_dataset"),
        replay_sha256=raw.get("replay_sha256"),
        replay_ratio=float(raw.get("replay_ratio", 0.0)),
        text_field=raw.get("text_field", "text"),
        pretokenized=bool(raw.get("pretokenized", False)),
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
        save_strategy=raw.get("save_strategy", "no"),
        save_steps=int(raw.get("save_steps", 0)),
        save_total_limit=raw.get("save_total_limit"),
        resume_from_checkpoint=raw.get("resume_from_checkpoint"),
    )
    result = train(spec)
    Path(args.result).write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
