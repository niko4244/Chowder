from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import os
import threading
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from ..hf_resilience import cache_status, with_hub_retries
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


# Only architectures whose attention (q/k/v/o_proj) AND MLP (gate/up/
# down_proj) naming has actually been verified against a real loaded model
# (directly, for llama; by well-documented, stable architectural convention
# shared with llama, for the rest) are listed here. PEFT silently trains
# only whatever subset of a target_modules list actually matches real module
# names on the model -- it does NOT error if some names don't match, only if
# NONE do -- so guessing wrong here would be a silent partial-coverage bug,
# not a loud one. When in doubt, leave an architecture out: "auto" (PEFT's
# own actively-maintained per-architecture mapping) or an explicit
# backend.lora.target_modules list are always available.
_ATTENTION_AND_MLP_MODEL_TYPES = {"llama", "mistral", "qwen2", "gemma", "gemma2"}
_ATTENTION_AND_MLP_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def _resolve_target_modules(
    model: Any, *, explicit: tuple[str, ...], preset: str
) -> list[str] | None:
    """None means "let PEFT's own per-architecture mapping decide" (its
    LoraConfig(target_modules=None) auto-detection, keyed off
    model.config.model_type) -- the safest, broadest-coverage default,
    actively maintained by PEFT itself rather than duplicated here.
    """
    if explicit:
        return list(explicit)
    if preset == "attention_and_mlp":
        model_type = getattr(model.config, "model_type", None)
        if model_type not in _ATTENTION_AND_MLP_MODEL_TYPES:
            raise RuntimeError(
                f"lora.target_preset='attention_and_mlp' has no curated module list for "
                f"model_type {model_type!r}; supported: {sorted(_ATTENTION_AND_MLP_MODEL_TYPES)}. "
                "Specify backend.lora.target_modules explicitly instead."
            )
        return list(_ATTENTION_AND_MLP_TARGET_MODULES)
    return None


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


_CHAT_ROLES = {"system", "user", "assistant"}


def _validate_chat_messages(raw: Any, *, row_index: int) -> list[dict[str, str]]:
    if not isinstance(raw, list) or not raw:
        raise RuntimeError(
            f"chat dataset row {row_index} has an empty or invalid messages list"
        )
    normalized: list[dict[str, str]] = []
    has_assistant = False
    for turn in raw:
        if not isinstance(turn, dict) or "role" not in turn or "content" not in turn:
            raise RuntimeError(
                f"chat dataset row {row_index} has a message missing role/content"
            )
        role = str(turn["role"])
        if role not in _CHAT_ROLES:
            raise RuntimeError(
                f"chat dataset row {row_index} has unsupported message role {role!r}; "
                f"supported roles are {sorted(_CHAT_ROLES)}"
            )
        if role == "assistant":
            has_assistant = True
        normalized.append({"role": role, "content": str(turn["content"])})
    if not has_assistant:
        raise RuntimeError(
            f"chat dataset row {row_index} has no assistant turn -- nothing to train on"
        )
    return normalized


def _render_chat_ids(
    tokenizer: Any, messages: list[dict[str, str]], *, add_generation_prompt: bool
) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=add_generation_prompt
    )
    return list(encoded["input_ids"] if hasattr(encoded, "keys") else encoded)


def _build_chat_example(
    tokenizer: Any, messages: list[dict[str, str]], *, max_length: int, row_index: int
) -> dict[str, list[int]]:
    """Tokenize one conversation with completion-only (assistant-turn) labels.

    Does not rely on the chat template defining a ``{% generation %}`` block
    -- most real templates (including the official Llama 3.1 template) don't.
    Instead, for each assistant turn, renders the conversation twice: once up
    to (not including) that turn with ``add_generation_prompt=True`` (the
    exact point the assistant's own tokens begin), and once through the end
    of that turn. The token-length difference is exactly the assistant's own
    generated span, verified to be a real prefix of the full sequence before
    being trusted -- a template that isn't prefix-consistent raises rather
    than silently mislabeling.
    """
    full_ids = _render_chat_ids(tokenizer, messages, add_generation_prompt=False)
    labels = [-100] * len(full_ids)
    for index, turn in enumerate(messages):
        if turn["role"] != "assistant":
            continue
        prefix_ids = _render_chat_ids(tokenizer, messages[:index], add_generation_prompt=True)
        through_ids = _render_chat_ids(
            tokenizer, messages[: index + 1], add_generation_prompt=False
        )
        if (
            len(prefix_ids) > len(full_ids)
            or len(through_ids) > len(full_ids)
            or full_ids[: len(prefix_ids)] != prefix_ids
            or full_ids[: len(through_ids)] != through_ids
        ):
            raise RuntimeError(
                f"chat dataset row {row_index}: chat template is not prefix-consistent "
                "across turns; cannot compute a reliable completion-only loss mask"
            )
        labels[len(prefix_ids) : len(through_ids)] = full_ids[len(prefix_ids) : len(through_ids)]

    full_ids = full_ids[:max_length]
    labels = labels[:max_length]
    if not any(label != -100 for label in labels):
        raise RuntimeError(
            f"chat dataset row {row_index}: no assistant tokens remain after truncating "
            f"to max_length={max_length} -- increase max_length or shorten this conversation"
        )
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def _chat_digest(dataset: Any, messages_field: str) -> str:
    digest = hashlib.sha256()
    for index in range(len(dataset)):
        payload = json.dumps(
            dataset[index][messages_field], sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
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
            DataCollatorForSeq2Seq,
            Trainer,
            TrainerCallback,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Transformers backend dependencies are missing; install chowder-ai[train] "
            "and chowder-ai[qlora] when using 4-bit quantization"
        ) from exc

    class _ProgressReportingCallback(TrainerCallback):
        """Writes the latest training progress to a fixed path after each
        log event, so the main process -- otherwise blind until this
        subprocess exits -- can poll it for live step/loss/lr instead of
        parsing stdout. Rank-0-only under DDP: every rank runs this
        callback, but only one may write the file without a real
        corruption risk, matching why only rank 0 writes the adapter/
        tokenizer/result files below.
        """

        def __init__(self, progress_path: Path, started: float) -> None:
            self._progress_path = progress_path
            self._started = started

        def on_log(self, args, state, control, logs=None, **kwargs):
            # Trainer also calls on_log once more at the very end of
            # train() with a summary dict (train_runtime, train_loss,
            # total_flos, ...) that has no "loss" key -- writing that over
            # the last real per-step snapshot would mean a poller whose
            # only read lands after training finishes (a real risk for a
            # fast run and a slow poll interval) sees loss=None instead of
            # the actual last measured loss. Real per-step logs always
            # carry "loss"; the summary one never does.
            if not state.is_world_process_zero or not logs or "loss" not in logs:
                return
            payload = {
                "step": state.global_step,
                "max_steps": state.max_steps if state.max_steps and state.max_steps > 0 else None,
                "epoch": logs.get("epoch", state.epoch),
                "loss": logs.get("loss"),
                "learning_rate": logs.get("learning_rate"),
                "wall_seconds": time.perf_counter() - self._started,
            }
            tmp_path = self._progress_path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(payload), encoding="utf-8")
            tmp_path.replace(self._progress_path)  # atomic on POSIX/NTFS, so a
            # concurrent poller in the main process never reads a half-written file.

    class _FrozenLayerStreamingCallback(TrainerCallback):
        """Kicks off each step's frozen-layer prefetch right before that
        step's forward pass. Constructed with streamed=None and passed
        into Trainer(callbacks=[...]) at construction time -- the actual
        StreamedFrozenLayers handle can only be created *after* Trainer's
        own __init__ finishes (it calls a blanket model.to(device) that
        raises if the model's frozen base_layer params have already been
        patched -- confirmed for real; see memory_fabric.py's own fix for
        the details), so this callback's real work starts only once
        `.streamed` is set from outside, after Trainer() returns.
        """

        def __init__(self) -> None:
            self.streamed: Any = None

        def on_step_begin(self, args, state, control, **kwargs):
            if self.streamed is not None:
                self.streamed.start_step()

    class _TrainingPhaseTimerCallback(TrainerCallback):
        """Real forward/backward/optimizer-step timing, by wrapping
        Trainer.compute_loss / Trainer.accelerator.backward / Trainer.
        optimizer.step on the instance (verified for real: Trainer.
        training_step calls compute_loss then accelerator.backward as
        two distinct steps; trainer.optimizer is None until Trainer's
        own create_optimizer_and_scheduler() runs, which -- confirmed
        directly -- has already happened by the time on_train_begin
        fires, so that is the right point to wrap optimizer.step too).

        Deliberately NOT installed unconditionally: torch.cuda.
        synchronize() around each of the three phases is required for
        the timing to mean anything (without it, async CUDA kernel
        queuing makes "time inside this Python call" nearly meaningless),
        and that synchronization has a real, measured cost -- ~17% wall-
        time overhead on a real training run. Only active when
        spec.detailed_timing_telemetry is explicitly requested.

        Also runs a background daemon thread sampling torch.cuda.
        utilization() (a real NVML query) at a low frequency for the
        run's average GPU utilization -- a coarse but genuine proxy for
        GPU idle/stall time, not a precise measurement.
        """

        def __init__(self) -> None:
            self.trainer_ref: Any = None
            self.forward_seconds = 0.0
            self.backward_seconds = 0.0
            self.optimizer_seconds = 0.0
            self._utilization_samples: list[float] = []
            self._stop_sampling = threading.Event()
            self._sampler_thread: threading.Thread | None = None

        def _sample_utilization(self) -> None:
            while not self._stop_sampling.is_set():
                try:
                    self._utilization_samples.append(float(torch.cuda.utilization()))
                except Exception:
                    # A transient NVML query failure -- including simply
                    # not having a CUDA device at all -- must never take
                    # down an otherwise-successful training run; this
                    # sample is skipped, and avg_gpu_utilization_percent
                    # correctly degrades to None if none ever succeed.
                    pass
                self._stop_sampling.wait(0.5)

        @staticmethod
        def _sync() -> None:
            # CPU tensor ops are already synchronous -- there is no async
            # kernel queue to drain, so synchronize() is only meaningful
            # (and only guaranteed safe to call) when a real CUDA device
            # is present.
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        def on_train_begin(self, args, state, control, **kwargs):
            trainer = self.trainer_ref

            original_compute_loss = trainer.compute_loss

            @functools.wraps(original_compute_loss)
            def timed_compute_loss(*call_args, **call_kwargs):
                self._sync()
                started = time.perf_counter()
                result = original_compute_loss(*call_args, **call_kwargs)
                self._sync()
                self.forward_seconds += time.perf_counter() - started
                return result

            trainer.compute_loss = timed_compute_loss

            original_backward = trainer.accelerator.backward

            @functools.wraps(original_backward)
            def timed_backward(*call_args, **call_kwargs):
                self._sync()
                started = time.perf_counter()
                result = original_backward(*call_args, **call_kwargs)
                self._sync()
                self.backward_seconds += time.perf_counter() - started
                return result

            trainer.accelerator.backward = timed_backward

            original_step = trainer.optimizer.step

            @functools.wraps(original_step)
            def timed_step(*call_args, **call_kwargs):
                self._sync()
                started = time.perf_counter()
                result = original_step(*call_args, **call_kwargs)
                self._sync()
                self.optimizer_seconds += time.perf_counter() - started
                return result

            trainer.optimizer.step = timed_step

            self._sampler_thread = threading.Thread(target=self._sample_utilization, daemon=True)
            self._sampler_thread.start()

        def on_train_end(self, args, state, control, **kwargs):
            self._stop_sampling.set()
            if self._sampler_thread is not None:
                self._sampler_thread.join(timeout=2)

        @property
        def avg_gpu_utilization_percent(self) -> float | None:
            if not self._utilization_samples:
                return None
            return sum(self._utilization_samples) / len(self._utilization_samples)

    if spec.trust_remote_code:
        raise RuntimeError("trust_remote_code is disabled")

    dtype = _resolve_dtype(torch, spec.precision)
    if spec.quantization == "4bit" and not torch.cuda.is_available():
        raise RuntimeError("initial 4-bit QLoRA backend requires an available CUDA device")

    set_seed(spec.seed)
    model_cache_status = cache_status(spec.base_model, spec.revision)
    tokenizer = with_hub_retries(
        lambda: AutoTokenizer.from_pretrained(
            spec.base_model,
            revision=spec.revision,
            trust_remote_code=False,
            local_files_only=spec.offline,
        ),
        label=f"tokenizer download for {spec.base_model}",
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
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        model_kwargs["device_map"] = 0

    base_model = with_hub_retries(
        lambda: AutoModelForCausalLM.from_pretrained(spec.base_model, **model_kwargs),
        label=f"model download for {spec.base_model}",
    )
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

    resolved_target_modules = sorted(model.peft_config[model.active_adapter].target_modules)

    if spec.gradient_checkpointing:
        model.config.use_cache = False

    is_chat = spec.dataset_format == "chat"
    field = spec.messages_field if is_chat else spec.text_field
    kind = "messages" if is_chat else "text"

    primary = load_dataset("json", data_files=spec.dataset, split="train")
    if field not in primary.column_names:
        raise RuntimeError(
            f"dataset is missing {kind} field {field!r}; columns={primary.column_names}"
        )
    primary = primary.select_columns([field])
    primary_rows = len(primary)
    if primary_rows == 0:
        raise RuntimeError("training dataset contains no rows")

    replay_available_rows = 0
    replay_selected_rows = 0
    dataset = primary
    if spec.replay_dataset is not None:
        replay = load_dataset("json", data_files=spec.replay_dataset, split="train")
        if field not in replay.column_names:
            raise RuntimeError(
                f"replay dataset is missing {kind} field {field!r}; columns={replay.column_names}"
            )
        replay = replay.select_columns([field])
        replay_available_rows = len(replay)
        replay_selected_rows = _replay_sample_count(
            primary_rows, replay_available_rows, spec.replay_ratio
        )
        if replay_selected_rows:
            selected_replay = replay.shuffle(seed=spec.seed).select(
                range(replay_selected_rows)
            )
            dataset = concatenate_datasets([primary, selected_replay]).shuffle(seed=spec.seed)

    assistant_token_count: int | None = None
    total_token_count: int | None = None
    if is_chat:
        mixed_text_sha = _chat_digest(dataset, field)

        def tokenize_chat(example, index):
            messages = _validate_chat_messages(example[field], row_index=index)
            return _build_chat_example(
                tokenizer, messages, max_length=spec.max_length, row_index=index
            )

        tokenized = dataset.map(
            tokenize_chat, with_indices=True, remove_columns=dataset.column_names
        )
        collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=None, label_pad_token_id=-100, padding=True
        )
        total_token_count = sum(len(row) for row in tokenized["input_ids"])
        assistant_token_count = sum(
            sum(1 for label in row if label != -100) for row in tokenized["labels"]
        )
    else:
        mixed_text_sha = _text_digest(dataset, field)

        def tokenize_text(batch):
            return tokenizer(
                batch[field],
                truncation=True,
                max_length=spec.max_length,
                padding=False,
            )

        tokenized = dataset.map(tokenize_text, batched=True, remove_columns=dataset.column_names)
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    output_dir = Path(spec.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # transformers>=5.2 merged the old two-field warmup_ratio/warmup_steps
    # split into a single overloaded warmup_steps: a float in [0, 1) means a
    # ratio of total steps, an int/float >= 1 means an absolute step count.
    # warmup_ratio itself is gone as a TrainingArguments kwarg entirely as of
    # 5.16. Chowder's own spec keeps the two separate, unambiguous fields as
    # its public schema -- only this translation, and only warmup_steps'
    # existing precedence over warmup_ratio, is HF-version plumbing.
    warmup_steps_arg = spec.warmup_steps if spec.warmup_steps > 0 else spec.warmup_ratio
    args_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir / "trainer"),
        "num_train_epochs": spec.epochs,
        "max_steps": spec.max_steps,
        "per_device_train_batch_size": spec.batch_size,
        "gradient_accumulation_steps": spec.gradient_accumulation_steps,
        "learning_rate": spec.learning_rate,
        "lr_scheduler_type": spec.lr_scheduler_type,
        "warmup_steps": warmup_steps_arg,
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
    if spec.optimizer_tiering:
        # HF's own built-in support (transformers.training_args.
        # OptimizerNames) for bitsandbytes' CUDA-unified-memory-paged
        # AdamW -- no custom optimizer construction needed, unlike
        # activation_offload's hand-rolled saved_tensors_hooks. Real,
        # proven library mechanism; state can page out to host RAM under
        # VRAM pressure without OOM-crashing.
        args_kwargs["optim"] = "paged_adamw_32bit"
    args = TrainingArguments(**args_kwargs)
    started = time.perf_counter()
    frozen_layer_streaming_callback = _FrozenLayerStreamingCallback()
    callbacks: list[Any] = [
        _ProgressReportingCallback(output_dir / "progress.json", started),
        frozen_layer_streaming_callback,
    ]
    timer_callback: Any = None
    if spec.detailed_timing_telemetry:
        timer_callback = _TrainingPhaseTimerCallback()
        callbacks.append(timer_callback)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=collator,
        callbacks=callbacks,
    )
    if timer_callback is not None:
        timer_callback.trainer_ref = trainer

    frozen_layer_streaming_bytes_transferred: int | None = None
    streamed_layers = None
    if spec.frozen_layer_streaming:
        # Applied *after* Trainer() construction, never before -- see
        # _FrozenLayerStreamingCallback's docstring for why: Trainer.
        # __init__ (already run by this point) and accelerator.
        # prepare_model() (called again inside trainer.train() itself,
        # on every call) both do a blanket model.to(device) this worker
        # does not control, which raises on the patched frozen base_layer
        # params if streaming were applied any earlier.
        from ..memory_fabric import stream_frozen_layers

        streamed_layers = stream_frozen_layers(model, trainer.args.device)
        frozen_layer_streaming_callback.streamed = streamed_layers

    activation_offload_bytes_transferred: int | None = None
    if spec.activation_offload:
        # Nested (not module-level) for the same reason _ProgressReporting
        # Callback is: torch is only importable after the lazy import at
        # the top of this function, so these close over the local `torch`
        # rather than needing a second import at module scope. Identical
        # mechanism to the one proven in
        # activation_offload_worker.run_experiment -- moves a tensor to
        # CPU when it's saved for backward, and back to its original
        # device when backward actually needs it. Value-transparent: the
        # computed values are identical either way, only where the
        # intermediate tensor physically lives changes (see
        # TransformersPeftRunSpec.recipe_digest's docstring for why this
        # is excluded from checkpoint-compatibility).
        activation_offload_bytes_transferred = 0

        def _activation_offload_pack(tensor: Any) -> Any:
            nonlocal activation_offload_bytes_transferred
            if not tensor.is_cuda:
                return tensor
            activation_offload_bytes_transferred += tensor.numel() * tensor.element_size()
            return (tensor.device, tensor.to("cpu", non_blocking=True))

        def _activation_offload_unpack(packed: Any) -> Any:
            nonlocal activation_offload_bytes_transferred
            if not isinstance(packed, tuple):
                return packed
            original_device, cpu_tensor = packed
            activation_offload_bytes_transferred += cpu_tensor.numel() * cpu_tensor.element_size()
            return cpu_tensor.to(original_device, non_blocking=True)

        with torch.autograd.graph.saved_tensors_hooks(
            _activation_offload_pack, _activation_offload_unpack
        ):
            train_output = trainer.train(resume_from_checkpoint=spec.resume_from_checkpoint)
    else:
        train_output = trainer.train(resume_from_checkpoint=spec.resume_from_checkpoint)
    runtime = time.perf_counter() - started

    if streamed_layers is not None:
        frozen_layer_streaming_bytes_transferred = streamed_layers.runtime.bytes_transferred
        # Restore before save_pretrained()/anything downstream touches the
        # model again -- save_pretrained() itself was verified to work
        # fine while still patched (PEFT's save only ever writes the
        # adapter, never the frozen base), but leaving the model in its
        # normal fully-resident shape afterward is the safer default for
        # whatever else might run against it.
        streamed_layers.restore()

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

    # Real, device-agnostic tensor introspection -- identical approach to
    # Phase 7A's optimizer_state_bytes and 7C's own experiment worker.
    # Works the same whether trainer.optimizer holds a plain torch.optim.
    # AdamW or bitsandbytes' paged variant: it reports what is actually
    # resident in the live optimizer's state, not a formula.
    optimizer_state_bytes = sum(
        value.numel() * value.element_size()
        for state in trainer.optimizer.state.values()
        for value in state.values()
        if torch.is_tensor(value)
    )

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
            "activation_offload_bytes_transferred": activation_offload_bytes_transferred,
            "optimizer_state_bytes": optimizer_state_bytes,
            "frozen_layer_streaming_bytes_transferred": frozen_layer_streaming_bytes_transferred,
            "forward_seconds": timer_callback.forward_seconds if timer_callback is not None else None,
            "backward_seconds": timer_callback.backward_seconds if timer_callback is not None else None,
            "optimizer_seconds": timer_callback.optimizer_seconds if timer_callback is not None else None,
            "avg_gpu_utilization_percent": (
                timer_callback.avg_gpu_utilization_percent if timer_callback is not None else None
            ),
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
            "dataset_format": spec.dataset_format,
            "total_token_count": total_token_count,
            "assistant_token_count": assistant_token_count,
        },
        "provenance": {
            "requested_base_model": spec.base_model,
            "requested_revision": spec.revision,
            "model_cache_status": model_cache_status,
            "resolved_model_commit": getattr(model.config, "_commit_hash", None),
            "model_type": getattr(model.config, "model_type", None),
            "resolved_target_modules": resolved_target_modules,
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


def _crash_rank_for_ddp_acceptance_test() -> None:
    """Test-support only, inert unless a specific env var is explicitly set
    (normal runs never set it): deterministically fails one rank under a
    real multi-process `accelerate launch`, so tests/test_ddp_acceptance.py
    can prove failure accounting is correct when a rank genuinely crashes.
    The crash itself is real -- this only makes it reliably reproducible
    instead of incidental, which is the only practical way to exercise that
    path under a real 2-GPU DDP launch rather than mocking it.
    """
    target_rank = os.environ.get("_CHOWDER_DDP_ACCEPTANCE_CRASH_RANK")
    if target_rank is not None and os.environ.get("RANK") == target_rank:
        raise RuntimeError(
            f"deliberate rank-{target_rank} crash for DDP failure-accounting acceptance test"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()

    _crash_rank_for_ddp_acceptance_test()

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
