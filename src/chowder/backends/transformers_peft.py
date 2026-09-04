from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping

if TYPE_CHECKING:
    from ..placement_policy import PlacementPlan
from uuid import uuid4

from ..cancellation import CancellationToken
from ..dependency_preflight import check_dependencies
from ..executors import CostEstimate, ExecutionContext, TrainingArtifact
from ..memory import HardwareProfile
from ..models import Experiment
from ..provenance import sha256_directory, sha256_file
from ..resources import ResourceUsage
from ..run_events import TrainingProgressEvent


_ALLOWED_QUANTIZATION = {"none", "4bit"}
_ALLOWED_ACTIVATION_OFFLOAD_MODES = {"auto", "always", "off"}
_ALLOWED_OPTIMIZER_TIERING_MODES = {"auto", "always", "off"}
_ALLOWED_FROZEN_LAYER_STREAMING_MODES = {"auto", "always", "off"}
_ALLOWED_PRECISION = {"auto", "bf16", "fp16", "fp32"}
_ALLOWED_DATASET_FORMATS = {"text", "chat"}
_ALLOWED_TARGET_PRESETS = {"auto", "attention_and_mlp"}
_ALLOWED_LR_SCHEDULER_TYPES = {
    "linear",
    "cosine",
    "cosine_with_restarts",
    "polynomial",
    "constant",
    "constant_with_warmup",
    "inverse_sqrt",
}

# Hardware-aware defaults apply only when the corresponding config key is
# absent entirely -- an explicit value, including one that happens to match
# what the heuristic would have chosen, always takes precedence and is never
# second-guessed. Both thresholds key off the smallest VRAM pool across
# active devices (not the largest, and not a sum): under multi-GPU DDP every
# device holds its own full model copy, so the worst-case device is what
# actually determines whether a step fits, not the best one.
_GRADIENT_CHECKPOINTING_VRAM_THRESHOLD_GB = 24.0
_LOW_VRAM_QUANTIZATION_THRESHOLD_GB = 16.0


def _min_device_vram_gb(hardware: HardwareProfile | None) -> float:
    if hardware is None:
        return 0.0
    if hardware.accelerator_vram_gb:
        return min(hardware.accelerator_vram_gb)
    return hardware.vram_gb


def _default_gradient_checkpointing(hardware: HardwareProfile | None) -> bool:
    """Memory-safe (True) below the threshold -- including when hardware
    info is unavailable, since "unknown" must not be treated as "plenty."
    Above it, off by default: forcing activation recomputation on hardware
    with real headroom only costs training speed for no benefit."""
    return _min_device_vram_gb(hardware) < _GRADIENT_CHECKPOINTING_VRAM_THRESHOLD_GB


def _resolve_activation_offload_flag(training: Mapping[str, Any]) -> bool:
    """Parse backend.training.activation_offload into the concrete boolean
    TransformersPeftRunSpec carries. "always"/True -> True, "off"/False/
    unset -> False. "auto" resolves to False *here* -- deciding "auto" for
    real requires running the real activation-offload experiment
    (chowder.activation_offload.run_activation_offload_experiment), which
    is expensive (a real subprocess + model load) and must never happen
    inside this cheap, pure config-parsing function: from_resolved_config
    is called from many places that must stay cheap and side-effect-free
    (checkpoint discovery building comparison specs, memory_preflight's
    own dry-run spec construction, the offload/tiering experiments'
    own spec construction -- an "auto" resolution triggering another real
    experiment from inside spec-parsing would risk real recursion).
    TransformersPeftExecutor.resolved_activation_offload is the one place
    that actually runs the experiment for "auto" and substitutes a
    concrete "always"/"off" into the config before spec construction, for
    the real training path specifically.
    """
    raw = training.get("activation_offload", "off")
    if isinstance(raw, bool):
        return raw
    value = str(raw).strip().lower()
    if value not in _ALLOWED_ACTIVATION_OFFLOAD_MODES:
        raise ValueError(
            f"backend.training.activation_offload must be one of "
            f"{sorted(_ALLOWED_ACTIVATION_OFFLOAD_MODES)} or a boolean, got {raw!r}"
        )
    return value == "always"


def _resolve_optimizer_tiering_flag(training: Mapping[str, Any]) -> bool:
    """Parse backend.training.optimizer_tiering into the concrete boolean
    TransformersPeftRunSpec carries. Same "auto resolves to False here,
    the real decision happens in TransformersPeftExecutor.
    resolved_optimizer_tiering" structure as
    _resolve_activation_offload_flag, and the same reason: this function
    must stay cheap and side-effect-free.

    Unlike activation_offload, optimizer_tiering is NOT value-transparent
    across a resume: bitsandbytes' paged optimizers keep their own
    internal state-dict keys (state1/state2), incompatible with
    torch.optim.AdamW's (exp_avg/exp_avg_sq) -- resuming a checkpoint
    trained under one optimizer implementation with the other fails with
    a real KeyError deep inside bitsandbytes, confirmed against real
    hardware before this was wired in. So, unlike activation_offload,
    this field is deliberately left IN both recipe_digest() and
    _bound_inputs() -- a changed optimizer_tiering setting must reject a
    resume, not silently allow one into a corrupted optimizer state.
    """
    raw = training.get("optimizer_tiering", "off")
    if isinstance(raw, bool):
        return raw
    value = str(raw).strip().lower()
    if value not in _ALLOWED_OPTIMIZER_TIERING_MODES:
        raise ValueError(
            f"backend.training.optimizer_tiering must be one of "
            f"{sorted(_ALLOWED_OPTIMIZER_TIERING_MODES)} or a boolean, got {raw!r}"
        )
    return value == "always"


def _resolve_frozen_layer_streaming_flag(training: Mapping[str, Any]) -> bool:
    """Parse backend.training.frozen_layer_streaming into the concrete
    boolean TransformersPeftRunSpec carries. Same "auto resolves to False
    here" structure as _resolve_activation_offload_flag/
    _resolve_optimizer_tiering_flag, and the same reason: this function
    must stay cheap and side-effect-free.

    Value-transparent across a resume, like activation_offload and
    unlike optimizer_tiering: chowder.memory_fabric.stream_frozen_layers
    only changes *where* a frozen layer's weight physically lives during
    compute (pinned CPU RAM, streamed just-in-time, vs. GPU-resident the
    whole time) via a custom autograd.Function proven to produce
    bit-identical loss/gradients either way -- never what gets computed,
    and never what a saved checkpoint contains (only the LoRA adapter and
    optimizer state are ever saved; the frozen base is never part of a
    checkpoint's own state). Verified directly against real hardware: a
    checkpoint trained without streaming resumes correctly with
    streaming turned on, and vice versa.
    """
    raw = training.get("frozen_layer_streaming", "off")
    if isinstance(raw, bool):
        return raw
    value = str(raw).strip().lower()
    if value not in _ALLOWED_FROZEN_LAYER_STREAMING_MODES:
        raise ValueError(
            f"backend.training.frozen_layer_streaming must be one of "
            f"{sorted(_ALLOWED_FROZEN_LAYER_STREAMING_MODES)} or a boolean, got {raw!r}"
        )
    return value == "always"


def _resolve_detailed_timing_telemetry_flag(training: Mapping[str, Any]) -> bool:
    """Parse backend.training.detailed_timing_telemetry into a plain
    boolean -- unlike activation_offload/optimizer_tiering/
    frozen_layer_streaming, this has no "auto" mode: it is pure
    diagnostic instrumentation with no placement decision to make, so
    there is nothing for an experiment to recommend.

    Defaults to False for a real, measured reason, not just consistency
    with this codebase's other "off by default" flags: forcing
    torch.cuda.synchronize() around each of forward/backward/optimizer-
    step to get accurate per-phase timing is not free -- measured at a
    real ~17% wall-time overhead on a real training run (tiny model,
    where per-step compute is small enough that synchronization
    overhead is a meaningful fraction of it). Unlike Phase 7A's dry-run
    telemetry (which runs in an isolated, separate subprocess that never
    touches real production training), this instrumentation runs
    *inside* the real training loop and has a real cost -- so it must be
    an explicit opt-in, never silently always-on.
    """
    raw = training.get("detailed_timing_telemetry", False)
    if isinstance(raw, bool):
        return raw
    value = str(raw).strip().lower()
    if value not in ("true", "false"):
        raise ValueError(
            f"backend.training.detailed_timing_telemetry must be a boolean, got {raw!r}"
        )
    return value == "true"


def _default_quantization(hardware: HardwareProfile | None) -> str:
    """"none" unless there's a real, VRAM-constrained CUDA device (0 means
    either no GPU or unknown hardware -- never treated as "small") and the
    qlora extra is actually importable in this environment; defaulting to
    "4bit" when bitsandbytes isn't installed would trade a likely OOM for a
    guaranteed ImportError, not fix anything.
    """
    vram = _min_device_vram_gb(hardware)
    if vram <= 0 or vram >= _LOW_VRAM_QUANTIZATION_THRESHOLD_GB:
        return "none"
    if importlib.util.find_spec("bitsandbytes") is None:
        return "none"
    return "4bit"


@dataclass(frozen=True)
class TransformersPeftRunSpec:
    base_model: str
    dataset: str
    output_dir: str
    dataset_sha256: str | None = None
    replay_dataset: str | None = None
    replay_sha256: str | None = None
    replay_ratio: float = 0.0
    parent_adapter: str | None = None
    parent_adapter_sha256: str | None = None
    revision: str | None = None
    dataset_format: str = "text"
    text_field: str = "text"
    messages_field: str = "messages"
    max_length: int = 512
    epochs: float = 1.0
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "linear"
    warmup_ratio: float = 0.0
    warmup_steps: int = 0
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    max_steps: int = -1
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    logging_steps: int = 10
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ()
    target_preset: str = "auto"
    use_rslora: bool = False
    quantization: str = "none"
    precision: str = "auto"
    gradient_checkpointing: bool = True
    activation_offload: bool = False
    optimizer_tiering: bool = False
    frozen_layer_streaming: bool = False
    detailed_timing_telemetry: bool = False
    seed: int = 1
    timeout_seconds: float | None = None
    trust_remote_code: bool = False
    offline: bool = False
    save_strategy: str = "no"
    save_steps: int = 0
    save_total_limit: int | None = None
    resume_from_checkpoint: str | None = None

    def __post_init__(self) -> None:
        if not self.base_model.strip():
            raise ValueError("backend.base_model is required")
        if not self.dataset.strip():
            raise ValueError("backend.dataset is required")
        if self.dataset_sha256 is not None and len(self.dataset_sha256) != 64:
            raise ValueError("backend.dataset_sha256 must be a SHA-256 digest")

        has_replay_dataset = self.replay_dataset is not None
        has_replay_sha = self.replay_sha256 is not None
        if has_replay_dataset != has_replay_sha:
            raise ValueError("backend replay dataset and SHA must be supplied together")
        ratio = float(self.replay_ratio)
        if has_replay_dataset:
            assert self.replay_dataset is not None
            assert self.replay_sha256 is not None
            if not self.replay_dataset.strip():
                raise ValueError("backend replay dataset cannot be empty")
            if len(self.replay_sha256) != 64:
                raise ValueError("backend replay SHA must be a SHA-256 digest")
            if not math.isfinite(ratio) or ratio <= 0 or ratio > 10:
                raise ValueError("backend replay ratio must be finite and in (0, 10]")
        elif ratio != 0.0:
            raise ValueError("backend replay ratio requires a replay dataset")

        has_parent_path = self.parent_adapter is not None
        has_parent_sha = self.parent_adapter_sha256 is not None
        if has_parent_path != has_parent_sha:
            raise ValueError(
                "backend parent adapter path and SHA must be supplied together"
            )
        if has_parent_path:
            assert self.parent_adapter is not None
            assert self.parent_adapter_sha256 is not None
            if not self.parent_adapter.strip():
                raise ValueError("backend parent adapter path cannot be empty")
            if len(self.parent_adapter_sha256) != 64:
                raise ValueError("backend parent adapter SHA must be a SHA-256 digest")

        if self.dataset_format not in _ALLOWED_DATASET_FORMATS:
            raise ValueError(f"unsupported dataset_format: {self.dataset_format}")
        if self.dataset_format == "chat" and not self.messages_field.strip():
            raise ValueError("backend.messages_field cannot be empty")
        if self.max_length <= 0:
            raise ValueError("backend.max_length must be positive")
        if self.epochs <= 0 or self.learning_rate <= 0:
            raise ValueError("training epochs and learning_rate must be positive")
        if self.lr_scheduler_type not in _ALLOWED_LR_SCHEDULER_TYPES:
            raise ValueError(f"unsupported lr_scheduler_type: {self.lr_scheduler_type}")
        if not math.isfinite(self.warmup_ratio) or not 0 <= self.warmup_ratio < 1:
            # Strictly < 1: the worker forwards this as HF TrainingArguments'
            # overloaded warmup_steps, which treats a value >= 1 as an
            # absolute step count, not a ratio. Accepting 1.0 here would
            # silently become "warm up for 1 step" instead of "warm up for
            # the entire run" once it reaches the worker.
            raise ValueError("warmup_ratio must be finite and in [0, 1)")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps cannot be negative")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        if not math.isfinite(self.max_grad_norm) or self.max_grad_norm < 0:
            raise ValueError("max_grad_norm must be finite and non-negative")
        if self.max_steps != -1 and self.max_steps <= 0:
            raise ValueError("max_steps must be -1 (disabled) or a positive integer")
        if self.batch_size <= 0 or self.gradient_accumulation_steps <= 0:
            raise ValueError("batch sizes must be positive")
        if self.lora_r <= 0 or self.lora_alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError("LoRA dropout must be in [0, 1)")
        if self.target_modules and any(
            not isinstance(module, str) or not module.strip() for module in self.target_modules
        ):
            raise ValueError("LoRA target module names must be non-empty strings")
        if self.target_preset not in _ALLOWED_TARGET_PRESETS:
            raise ValueError(f"unsupported lora.target_preset: {self.target_preset}")
        if self.quantization not in _ALLOWED_QUANTIZATION:
            raise ValueError(f"unsupported quantization: {self.quantization}")
        if self.precision not in _ALLOWED_PRECISION:
            raise ValueError(f"unsupported precision: {self.precision}")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.trust_remote_code:
            raise ValueError("trust_remote_code is disabled for autonomous Chowder execution")
        if self.save_strategy not in {"no", "steps", "epoch"}:
            raise ValueError(f"unsupported save_strategy: {self.save_strategy}")
        if self.save_strategy == "steps" and self.save_steps <= 0:
            raise ValueError("save_steps must be positive when save_strategy='steps'")
        if self.save_total_limit is not None and self.save_total_limit <= 0:
            raise ValueError("save_total_limit must be positive")
        if self.resume_from_checkpoint is not None and not self.resume_from_checkpoint.strip():
            raise ValueError("resume_from_checkpoint cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def recipe_digest(self) -> str:
        """Digest of the training *recipe* -- hyperparameters that determine
        what model resuming/replaying would produce. Paths, timeouts,
        checkpoint cadence (save_strategy/save_steps/save_total_limit,
        resume_from_checkpoint), and offline mode are operational, not
        recipe: none of them change what the training run is trying to do
        -- how often you checkpoint, which checkpoint you resume from, and
        whether the model was fetched from cache or network all produce the
        exact same trained result.

        activation_offload is excluded for the same reason, but on a
        stronger footing than "operational": torch.autograd.graph.
        saved_tensors_hooks is explicitly designed to be value-transparent
        -- it only changes *where* an intermediate tensor physically lives
        between forward and backward, never what gets computed. Unlike
        gradient_checkpointing (which changes the actual computation --
        recomputation instead of caching -- and stays part of the recipe),
        resuming a checkpoint with a different activation_offload setting
        than it was saved under produces the identical result.

        optimizer_tiering is deliberately NOT excluded, unlike
        activation_offload -- bitsandbytes' paged optimizers use their own
        internal state-dict keys (state1/state2), incompatible with
        torch.optim.AdamW's (exp_avg/exp_avg_sq): switching between them
        across a resume is a real, confirmed KeyError, not a
        value-transparent change. It stays part of the recipe on purpose.

        frozen_layer_streaming is excluded for the same reason as
        activation_offload: chowder.memory_fabric.stream_frozen_layers is
        a custom torch.autograd.Function proven to produce bit-identical
        loss/gradients to normal resident training -- it only changes
        where a frozen layer's weight physically lives during compute,
        never what gets computed, and the checkpoint itself never
        contains the frozen base weights in the first place (only the
        LoRA adapter and optimizer state are saved). Verified directly:
        a checkpoint trained without streaming resumes correctly with it
        turned on, and vice versa.

        detailed_timing_telemetry is excluded for the most clear-cut
        reason of all: it is pure observation, wrapping timing around
        calls that already happen -- it changes nothing about what gets
        computed, ever.
        """
        recipe = self.to_dict()
        recipe.pop("output_dir", None)
        recipe.pop("dataset", None)
        recipe.pop("replay_dataset", None)
        recipe.pop("parent_adapter", None)
        recipe.pop("timeout_seconds", None)
        recipe.pop("offline", None)
        recipe.pop("save_strategy", None)
        recipe.pop("save_steps", None)
        recipe.pop("save_total_limit", None)
        recipe.pop("resume_from_checkpoint", None)
        recipe.pop("activation_offload", None)
        recipe.pop("frozen_layer_streaming", None)
        recipe.pop("detailed_timing_telemetry", None)
        payload = json.dumps(recipe, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_resolved_config(
        cls,
        config: Mapping[str, Any],
        *,
        work_dir: str | Path,
        output_dir: str | Path,
        seed: int,
        hardware: HardwareProfile | None = None,
    ) -> "TransformersPeftRunSpec":
        backend = config.get("backend")
        if not isinstance(backend, Mapping):
            raise ValueError("resolved config must contain a backend mapping")
        if backend.get("type", "transformers-peft") != "transformers-peft":
            raise ValueError("resolved config backend.type is not transformers-peft")

        training = backend.get("training", {})
        lora = backend.get("lora", {})
        runtime = backend.get("runtime", {})
        replay = backend.get("replay", {})
        parent_adapter = backend.get("parent_adapter", {})
        if (
            not isinstance(training, Mapping)
            or not isinstance(lora, Mapping)
            or not isinstance(runtime, Mapping)
        ):
            raise ValueError("backend training/lora/runtime sections must be mappings")
        if replay is None:
            replay = {}
        if parent_adapter is None:
            parent_adapter = {}
        if not isinstance(replay, Mapping):
            raise ValueError("backend replay section must be a mapping")
        if not isinstance(parent_adapter, Mapping):
            raise ValueError("backend parent_adapter section must be a mapping")

        dataset = Path(str(backend.get("dataset", "")))
        if not dataset.is_absolute():
            dataset = Path(work_dir) / dataset
        dataset = dataset.resolve()

        replay_dataset: Path | None = None
        if replay.get("dataset") is not None:
            replay_dataset = Path(str(replay.get("dataset")))
            if not replay_dataset.is_absolute():
                replay_dataset = Path(work_dir) / replay_dataset
            replay_dataset = replay_dataset.resolve()

        parent_adapter_path: Path | None = None
        if parent_adapter.get("path") is not None:
            parent_adapter_path = Path(str(parent_adapter.get("path")))
            if not parent_adapter_path.is_absolute():
                parent_adapter_path = Path(work_dir) / parent_adapter_path
            parent_adapter_path = parent_adapter_path.resolve()

        dataset_sha = backend.get("dataset_sha256")
        replay_sha = replay.get("sha256")
        parent_sha = parent_adapter.get("sha256")

        resume_raw = backend.get("resume_from_checkpoint")
        resume_from_checkpoint: Path | None = None
        if resume_raw is not None:
            resume_from_checkpoint = Path(str(resume_raw))
            if not resume_from_checkpoint.is_absolute():
                resume_from_checkpoint = Path(work_dir) / resume_from_checkpoint
            resume_from_checkpoint = resume_from_checkpoint.resolve()

        save_total_limit_raw = training.get("save_total_limit")
        return cls(
            base_model=str(backend.get("base_model", "")),
            dataset=str(dataset),
            output_dir=str(Path(output_dir).resolve()),
            dataset_sha256=(str(dataset_sha) if dataset_sha is not None else None),
            replay_dataset=(str(replay_dataset) if replay_dataset is not None else None),
            replay_sha256=(str(replay_sha) if replay_sha is not None else None),
            replay_ratio=(
                float(replay.get("ratio", 1.0)) if replay_dataset is not None else 0.0
            ),
            parent_adapter=(
                str(parent_adapter_path) if parent_adapter_path is not None else None
            ),
            parent_adapter_sha256=(str(parent_sha) if parent_sha is not None else None),
            revision=(str(backend["revision"]) if backend.get("revision") is not None else None),
            dataset_format=str(backend.get("dataset_format", "text")),
            text_field=str(backend.get("text_field", "text")),
            messages_field=str(backend.get("messages_field", "messages")),
            max_length=int(backend.get("max_length", 512)),
            epochs=float(training.get("epochs", 1.0)),
            learning_rate=float(training.get("learning_rate", 2e-4)),
            lr_scheduler_type=str(training.get("lr_scheduler_type", "linear")),
            warmup_ratio=float(training.get("warmup_ratio", 0.0)),
            warmup_steps=int(training.get("warmup_steps", 0)),
            weight_decay=float(training.get("weight_decay", 0.0)),
            max_grad_norm=float(training.get("max_grad_norm", 1.0)),
            max_steps=int(training.get("max_steps", -1)),
            batch_size=int(training.get("batch_size", 1)),
            gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 4)),
            logging_steps=int(training.get("logging_steps", 10)),
            lora_r=int(lora.get("r", 16)),
            lora_alpha=int(lora.get("alpha", 32)),
            lora_dropout=float(lora.get("dropout", 0.05)),
            target_modules=tuple(str(x) for x in lora.get("target_modules", ())),
            target_preset=str(lora.get("target_preset", "auto")),
            use_rslora=bool(lora.get("use_rslora", False)),
            quantization=(
                str(backend["quantization"]).lower()
                if "quantization" in backend
                else _default_quantization(hardware)
            ),
            precision=str(backend.get("precision", "auto")).lower(),
            gradient_checkpointing=(
                bool(training["gradient_checkpointing"])
                if "gradient_checkpointing" in training
                else _default_gradient_checkpointing(hardware)
            ),
            activation_offload=_resolve_activation_offload_flag(training),
            optimizer_tiering=_resolve_optimizer_tiering_flag(training),
            frozen_layer_streaming=_resolve_frozen_layer_streaming_flag(training),
            detailed_timing_telemetry=_resolve_detailed_timing_telemetry_flag(training),
            seed=int(config.get("seed", seed)),
            timeout_seconds=(
                float(runtime["timeout_seconds"])
                if runtime.get("timeout_seconds") is not None
                else None
            ),
            trust_remote_code=bool(backend.get("trust_remote_code", False)),
            offline=bool(backend.get("offline", False)),
            save_strategy=str(training.get("save_strategy", "no")),
            save_steps=int(training.get("save_steps", 0)),
            save_total_limit=(
                int(save_total_limit_raw) if save_total_limit_raw is not None else None
            ),
            resume_from_checkpoint=(
                str(resume_from_checkpoint) if resume_from_checkpoint is not None else None
            ),
        )


_CHECKPOINT_MANIFEST_NAME = "chowder-checkpoint-manifest.json"


class TransformersPeftExecutor:
    name = "transformers-peft"

    def __init__(self) -> None:
        self._processes: dict[str, subprocess.Popen[Any]] = {}
        self._cancellation: CancellationToken | None = None
        self._progress_callback: Callable[[TrainingProgressEvent], None] | None = None

    def bind_cancellation(self, token: CancellationToken | None) -> None:
        """Optional capability: a CancellationToken to register the
        in-flight run against, so token.request() can terminate a training
        subprocess that is already running, not just prevent a future one
        from starting."""
        self._cancellation = token

    def bind_progress_callback(
        self, callback: Callable[[TrainingProgressEvent], None] | None
    ) -> None:
        """Optional capability: called with a TrainingProgressEvent each
        time the worker subprocess reports new step/loss/lr progress,
        polled from the same progress file the worker writes -- the only
        way to see inside an otherwise-opaque, isolated subprocess before
        it exits."""
        self._progress_callback = callback

    def _poll_progress(
        self, progress_path: Path, experiment_id: str, stop: threading.Event
    ) -> None:
        last_step: int | None = None
        while not stop.is_set():
            last_step = self._poll_progress_once(progress_path, experiment_id, last_step)
            stop.wait(1.0)

    def _poll_progress_once(
        self, progress_path: Path, experiment_id: str, last_step: int | None
    ) -> int | None:
        """One read-and-maybe-report cycle, factored out of the polling
        loop above so it's directly testable without waiting on real
        thread timing: read the current progress file, and if its step
        has moved on from `last_step`, report it and return the new step;
        otherwise return `last_step` unchanged. Any read/parse failure
        (the worker hasn't written the file yet, or is mid-write despite
        the atomic rename) is treated as "nothing new yet", not an error.
        """
        if not progress_path.is_file():
            return last_step
        try:
            data = json.loads(progress_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return last_step
        if not isinstance(data, Mapping) or data.get("step") == last_step:
            return last_step
        callback = self._progress_callback
        if callback is not None:
            try:
                callback(
                    TrainingProgressEvent(
                        experiment_id=experiment_id,
                        step=int(data.get("step", 0)),
                        max_steps=data.get("max_steps"),
                        epoch=data.get("epoch"),
                        loss=data.get("loss"),
                        learning_rate=data.get("learning_rate"),
                        wall_seconds=float(data.get("wall_seconds", 0.0)),
                    )
                )
            except Exception:
                # A caller's callback (writing to a UI, the registry) must
                # never take down the poller -- the worst case is a missed
                # progress update, not a crashed training run.
                pass
        return data.get("step")

    @staticmethod
    def _bound_inputs(spec: TransformersPeftRunSpec) -> dict[str, Any]:
        """The training inputs a checkpoint is bound to.

        Everything a resumed run must match exactly before its optimizer/
        scheduler state can be trusted: the exact base model and revision,
        the exact dataset/replay/parent-adapter content this checkpoint was
        actually produced from, and a recipe digest of the hyperparameters
        that determine optimizer-state validity (learning rate, batch size,
        LoRA config, quantization/precision, seed, sequence length).

        Deliberately its own digest, not ``spec.recipe_digest()`` (which
        serves a different purpose elsewhere -- comparing whether two
        experiments used "the same recipe" for repair-tracking -- and
        includes ``epochs``/``max_steps``). ``epochs`` and ``max_steps`` are
        excluded here on purpose: training for more total epochs/steps than
        originally planned is the entire point of resuming, and Trainer
        legitimately recomputes the remaining LR-scheduler trajectory for a
        new total when resuming -- that is not a hazard to the optimizer
        state the way a changed learning rate, batch size, weight decay, LR
        schedule shape, or dataset would be. ``activation_offload`` is
        excluded for the same value-transparency reason as in
        recipe_digest() -- resuming with a different offload setting (or a
        different real "auto" recommendation on re-run) never invalidates
        the optimizer state. ``optimizer_tiering`` is deliberately kept
        IN, the opposite choice: it changes which optimizer implementation
        actually holds the state (torch.optim.AdamW vs. bitsandbytes'
        paged variant), and their state-dict keys are incompatible --
        resuming across a changed setting must be rejected here, not
        allowed to reach a real KeyError deep inside bitsandbytes.
        ``frozen_layer_streaming`` is excluded for the same
        value-transparency reason as ``activation_offload`` -- the
        checkpoint never contains the frozen base weights either way,
        only the LoRA adapter and optimizer state, and the custom
        autograd.Function it uses is proven to produce bit-identical
        results to normal resident computation. ``detailed_timing_
        telemetry`` is excluded for the same reason -- it is pure
        observation and never changes what gets computed.
        """
        recipe = spec.to_dict()
        for key in (
            "output_dir",
            "dataset",
            "replay_dataset",
            "parent_adapter",
            "timeout_seconds",
            "offline",
            "save_strategy",
            "save_steps",
            "save_total_limit",
            "resume_from_checkpoint",
            "epochs",
            "max_steps",
            "activation_offload",
            "frozen_layer_streaming",
            "detailed_timing_telemetry",
        ):
            recipe.pop(key, None)
        recipe_payload = json.dumps(recipe, sort_keys=True, separators=(",", ":"))
        return {
            "checkpoint_recipe_sha256": hashlib.sha256(
                recipe_payload.encode("utf-8")
            ).hexdigest(),
            "base_model": spec.base_model,
            "revision": spec.revision,
            "dataset_sha256": spec.dataset_sha256,
            "replay_dataset_sha256": spec.replay_sha256,
            "parent_adapter_sha256": spec.parent_adapter_sha256,
        }

    @staticmethod
    def _write_checkpoint_manifest(trainer_dir: Path, bound_inputs: Mapping[str, Any]) -> None:
        trainer_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = trainer_dir / _CHECKPOINT_MANIFEST_NAME
        payload = json.dumps(dict(bound_inputs), sort_keys=True, indent=2) + "\n"
        existing = manifest_path.read_text(encoding="utf-8") if manifest_path.is_file() else None
        if existing is not None and existing != payload:
            raise RuntimeError(
                f"checkpoint manifest {manifest_path} already exists with different bound "
                "inputs -- this run directory was not produced by the current spec"
            )
        manifest_path.write_text(payload, encoding="utf-8")

    @classmethod
    def _verify_resume_checkpoint(
        cls, spec: TransformersPeftRunSpec, bound_inputs: Mapping[str, Any]
    ) -> None:
        """Reject a resume if any bound training input has changed.

        A checkpoint's optimizer/scheduler state is only meaningful for the
        exact recipe, model, and data it was produced under -- resuming
        into it after any of those changed would silently continue
        optimizing toward a different objective with stale momentum/LR
        schedule state. Refusing is the safe default; the caller can always
        start a fresh (non-resuming) run instead.
        """
        checkpoint_dir = Path(spec.resume_from_checkpoint).resolve()  # type: ignore[arg-type]
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(f"resume_from_checkpoint not found: {checkpoint_dir}")
        manifest_path = checkpoint_dir.parent / _CHECKPOINT_MANIFEST_NAME
        if not manifest_path.is_file():
            raise RuntimeError(
                f"no checkpoint manifest found at {manifest_path} -- refusing to resume "
                "from a checkpoint with no recorded bound inputs to verify against"
            )
        recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(recorded, Mapping):
            raise RuntimeError(f"checkpoint manifest {manifest_path} is not a JSON object")
        changed = {
            key: {"checkpoint": recorded.get(key), "requested": value}
            for key, value in bound_inputs.items()
            if recorded.get(key) != value
        }
        if changed:
            raise ValueError(
                f"refusing to resume from {checkpoint_dir}: bound training input(s) changed "
                f"since this checkpoint was produced: {json.dumps(changed, sort_keys=True)}"
            )

    @staticmethod
    def _json_digest(value: Mapping[str, Any]) -> str:
        payload = json.dumps(dict(value), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _verify_input(path: str, expected_sha: str | None, *, label: str) -> str:
        resolved = Path(path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"{label} dataset not found: {resolved}")
        actual = sha256_file(resolved)
        if expected_sha is not None and actual != expected_sha:
            raise ValueError(f"{label} dataset content changed after proposal")
        return actual

    @staticmethod
    def _verify_adapter(path: str, expected_sha: str, *, label: str) -> str:
        resolved = Path(path).resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"{label} adapter not found: {resolved}")
        actual = sha256_directory(resolved)
        if actual != expected_sha:
            raise ValueError(f"{label} adapter content changed after proposal")
        return actual

    def _spec_for(
        self,
        experiment: Experiment,
        context: ExecutionContext,
        *,
        run_dir: Path,
    ) -> TransformersPeftRunSpec:
        if not context.resolved_config:
            raise ValueError(
                "TransformersPeftExecutor requires ExecutionContext.resolved_config"
            )
        artifact_dir = run_dir / "adapter"
        spec = TransformersPeftRunSpec.from_resolved_config(
            context.resolved_config,
            work_dir=context.work_dir,
            output_dir=artifact_dir,
            seed=context.seed,
            hardware=context.hardware,
        )
        primary_sha = self._verify_input(spec.dataset, spec.dataset_sha256, label="training")
        if spec.dataset_sha256 is None:
            spec = replace(spec, dataset_sha256=primary_sha)

        resolved_offload = self.resolved_activation_offload(context)
        if resolved_offload != spec.activation_offload:
            spec = replace(spec, activation_offload=resolved_offload)

        resolved_tiering = self.resolved_optimizer_tiering(context)
        if resolved_tiering != spec.optimizer_tiering:
            spec = replace(spec, optimizer_tiering=resolved_tiering)

        resolved_streaming = self.resolved_frozen_layer_streaming(context)
        if resolved_streaming != spec.frozen_layer_streaming:
            spec = replace(spec, frozen_layer_streaming=resolved_streaming)

        if spec.replay_dataset is not None:
            self._verify_input(spec.replay_dataset, spec.replay_sha256, label="replay")
            if Path(spec.replay_dataset).resolve() == Path(spec.dataset).resolve():
                raise ValueError("training and replay datasets must be different files")

        if spec.parent_adapter is not None:
            assert spec.parent_adapter_sha256 is not None
            self._verify_adapter(
                spec.parent_adapter,
                spec.parent_adapter_sha256,
                label="parent",
            )
        return spec

    @staticmethod
    def _profile_accelerator_count(backend: Mapping[str, Any], profile: Mapping[str, Any]) -> int:
        raw = profile.get("active_accelerator_count")
        if raw is None:
            runtime = backend.get("runtime", {})
            if isinstance(runtime, Mapping):
                raw = runtime.get("active_accelerator_count")
        if raw is None:
            return 1
        count = int(raw)
        if count < 0:
            raise ValueError("active_accelerator_count cannot be negative")
        return count

    @staticmethod
    def resolved_quantization(context: ExecutionContext) -> str:
        """The quantization value this executor would actually train with,
        accounting for the hardware-aware default -- used by preflight
        dependency checking, which needs to know whether bitsandbytes will
        actually be required before any config-resolution/spec-building has
        happened yet.
        """
        config = context.resolved_config
        backend = config.get("backend", {}) if isinstance(config, Mapping) else {}
        if isinstance(backend, Mapping) and "quantization" in backend:
            return str(backend["quantization"]).lower()
        return _default_quantization(context.hardware)

    @staticmethod
    def _placement_plan(context: ExecutionContext) -> "PlacementPlan":
        """The shared Memory Fabric placement decision every "auto"-valued
        mechanism defers to (see resolved_activation_offload/
        resolved_optimizer_tiering/resolved_frozen_layer_streaming).
        Reuses `placement_policy.build_placement_plan`, which itself only
        ever recommends a 2+-mechanism combination when a REAL,
        empirically-validated `combined_mechanism_experiment.
        CombinedMechanismExperiment` exists for it -- "do not auto-apply
        an unvalidated combination" -- and returns every `enable_*=False`
        for multi-GPU DDP (each mechanism's own DDP-rejection stays the
        real fail-closed backstop regardless).

        Known limitation: this plan is computed as if all three mechanisms
        were free to choose from scratch, even if the caller has some of
        them explicitly pinned ("always"/"off") rather than "auto" -- a
        pinned mechanism's own resolved_* method never consults this
        (explicit values resolve directly, before this is ever called),
        but a DIFFERENT mechanism left on "auto" alongside it still gets
        a plan computed without knowledge of that pin. This matches
        today's existing behavior for a lone "auto" mechanism (always
        decided in isolation) and only improves the case this slice
        targets: multiple mechanisms simultaneously left on "auto".
        """
        from ..placement_policy import build_placement_plan

        config = context.resolved_config
        return build_placement_plan(
            resolved_config=config, context=context, work_dir=context.work_dir
        )

    @staticmethod
    def resolved_activation_offload(context: ExecutionContext) -> bool:
        """The activation_offload value this executor will actually train
        with. "always"/"off"/an explicit boolean resolve directly, with no
        extra cost, matching _resolve_activation_offload_flag exactly.
        "auto" is resolved for real here (unlike inside
        TransformersPeftRunSpec.from_resolved_config, which must stay
        cheap and side-effect-free -- see _resolve_activation_offload_flag)
        by deferring to the shared Memory Fabric placement plan (see
        _placement_plan) when the recipe actually needs an intervention to
        fit. When the plan reports fits_without_intervention (the recipe
        already fits resident), the plan itself recommends nothing for
        ANY mechanism -- see PlacementPlan's own docstring, this was
        always scoped to "recipe does not fit" -- so this falls back to
        activation_offload's own single-mechanism experiment and its
        `recommended` verdict, exactly as "auto" resolved before this
        combination-search wiring existed: real, measured, worthwhile-
        even-though-not-required savings (e.g. free insurance under an
        acceptable wall-time penalty) must not be silently lost just
        because the recipe would technically fit without it.
        """
        config = context.resolved_config
        backend = config.get("backend", {}) if isinstance(config, Mapping) else {}
        training = backend.get("training", {}) if isinstance(backend, Mapping) else {}
        raw = training.get("activation_offload", "off") if isinstance(training, Mapping) else "off"
        if isinstance(raw, bool):
            return raw
        value = str(raw).strip().lower()
        if value == "always":
            return True
        if value == "off":
            return False
        if value != "auto":
            raise ValueError(
                f"backend.training.activation_offload must be one of "
                f"{sorted(_ALLOWED_ACTIVATION_OFFLOAD_MODES)} or a boolean, got {raw!r}"
            )
        plan = TransformersPeftExecutor._placement_plan(context)
        if not plan.fits_without_intervention:
            return plan.enable_activation_offload

        from ..activation_offload import run_activation_offload_experiment

        experiment = run_activation_offload_experiment(
            resolved_config=config, context=context, work_dir=context.work_dir
        )
        return experiment.recommended

    @staticmethod
    def resolved_optimizer_tiering(context: ExecutionContext) -> bool:
        """The optimizer_tiering value this executor will actually train
        with. Structurally identical to resolved_activation_offload:
        "always"/"off"/an explicit boolean resolve directly; "auto" defers
        to the shared Memory Fabric placement plan's enable_optimizer_
        tiering verdict when the recipe needs intervention to fit, falling
        back to this mechanism's own single-mechanism `recommended`
        verdict when it already fits -- see resolved_activation_offload's
        docstring for why (the plan itself recommends nothing when
        fits_without_intervention, but a mechanism can still be real,
        measured, worthwhile insurance even when not strictly required).
        """
        config = context.resolved_config
        backend = config.get("backend", {}) if isinstance(config, Mapping) else {}
        training = backend.get("training", {}) if isinstance(backend, Mapping) else {}
        raw = training.get("optimizer_tiering", "off") if isinstance(training, Mapping) else "off"
        if isinstance(raw, bool):
            return raw
        value = str(raw).strip().lower()
        if value == "always":
            return True
        if value == "off":
            return False
        if value != "auto":
            raise ValueError(
                f"backend.training.optimizer_tiering must be one of "
                f"{sorted(_ALLOWED_OPTIMIZER_TIERING_MODES)} or a boolean, got {raw!r}"
            )
        plan = TransformersPeftExecutor._placement_plan(context)
        if not plan.fits_without_intervention:
            return plan.enable_optimizer_tiering

        from ..optimizer_tiering import run_optimizer_tiering_experiment

        experiment = run_optimizer_tiering_experiment(
            resolved_config=config, context=context, work_dir=context.work_dir
        )
        return experiment.recommended

    @staticmethod
    def resolved_frozen_layer_streaming(context: ExecutionContext) -> bool:
        """The frozen_layer_streaming value this executor will actually
        train with. Structurally identical to resolved_activation_offload
        /resolved_optimizer_tiering: "always"/"off"/an explicit boolean
        resolve directly; "auto" defers to the shared Memory Fabric
        placement plan's enable_frozen_layer_streaming verdict when the
        recipe needs intervention to fit, falling back to this
        mechanism's own single-mechanism `recommended` verdict when it
        already fits -- see resolved_activation_offload's docstring for
        why (the plan itself recommends nothing when
        fits_without_intervention, but a mechanism can still be real,
        measured, worthwhile insurance even when not strictly required).
        """
        config = context.resolved_config
        backend = config.get("backend", {}) if isinstance(config, Mapping) else {}
        training = backend.get("training", {}) if isinstance(backend, Mapping) else {}
        raw = (
            training.get("frozen_layer_streaming", "off")
            if isinstance(training, Mapping)
            else "off"
        )
        if isinstance(raw, bool):
            return raw
        value = str(raw).strip().lower()
        if value == "always":
            return True
        if value == "off":
            return False
        if value != "auto":
            raise ValueError(
                f"backend.training.frozen_layer_streaming must be one of "
                f"{sorted(_ALLOWED_FROZEN_LAYER_STREAMING_MODES)} or a boolean, got {raw!r}"
            )
        plan = TransformersPeftExecutor._placement_plan(context)
        if not plan.fits_without_intervention:
            return plan.enable_frozen_layer_streaming

        from ..frozen_layer_streaming import run_frozen_layer_streaming_experiment

        experiment = run_frozen_layer_streaming_experiment(
            resolved_config=config, context=context, work_dir=context.work_dir
        )
        return experiment.recommended

    def profile(self, experiment: Experiment, context: ExecutionContext) -> CostEstimate:
        config = context.resolved_config
        backend = config.get("backend", {}) if isinstance(config, Mapping) else {}
        profile = backend.get("profile", {}) if isinstance(backend, Mapping) else {}
        if not isinstance(profile, Mapping):
            profile = {}

        steps = profile.get("estimated_steps")
        seconds_per_step = profile.get("seconds_per_step")
        peak_vram = profile.get("peak_vram_gb")
        if steps is not None and seconds_per_step is not None:
            wall_hours = max(0.0, float(steps) * float(seconds_per_step) / 3600.0)
            active_count = self._profile_accelerator_count(backend, profile)
            hours = wall_hours * active_count
            confidence = 0.75 if profile.get("source") == "measured" else 0.5
            notes = (
                f"derived from backend step-time profile across {active_count} active accelerator(s)",
            )
        else:
            hours = max(0.0, experiment.estimated_gpu_hours)
            confidence = 0.25
            notes = (
                "using experiment-declared GPU-hour estimate; no measured step profile",
            )
        return CostEstimate(
            gpu_hours=hours,
            peak_vram_gb=float(peak_vram) if peak_vram is not None else None,
            confidence=confidence,
            notes=notes,
        )

    @staticmethod
    def _worker_module_args(spec_path: Path, result_path: Path) -> list[str]:
        return [
            "-m",
            "chowder.backends.transformers_worker",
            "--spec",
            str(spec_path),
            "--result",
            str(result_path),
        ]

    @classmethod
    def _worker_command(
        cls, spec_path: Path, result_path: Path, *, active_accelerator_count: int
    ) -> list[str]:
        module_args = cls._worker_module_args(spec_path, result_path)
        if active_accelerator_count <= 1:
            return [sys.executable, *module_args]
        # DDP, not FSDP, for a first multi-GPU launcher: accelerate launch
        # spawns active_accelerator_count real worker processes, each on its
        # own device, each with the full model -- HF's Trainer detects the
        # resulting WORLD_SIZE/RANK env vars and wraps the model in DDP
        # automatically, no worker-side launch code required. This is why
        # DDP does NOT let a model "use" combined VRAM across devices the
        # way naive sharding might suggest: every process still needs the
        # full model on its own single device, which is exactly what
        # HardwareTopology/HardwareProfile already refuse to pool (see
        # hardware_bridge.py) -- multi-GPU here means more parallel
        # replicas, not more room for one bigger model.
        return [
            sys.executable,
            "-m",
            "accelerate.commands.launch",
            "--multi_gpu",
            f"--num_processes={active_accelerator_count}",
            "--num_machines=1",
            *module_args,
        ]

    @staticmethod
    def _active_accelerator_count(context: ExecutionContext) -> int:
        config = context.resolved_config
        backend = config.get("backend", {}) if isinstance(config, Mapping) else {}
        runtime = backend.get("runtime", {}) if isinstance(backend, Mapping) else {}
        raw = runtime.get("active_accelerator_count") if isinstance(runtime, Mapping) else None
        count = 1 if raw is None else int(raw)
        if count < 0:
            raise ValueError("backend.runtime.active_accelerator_count cannot be negative")
        # accelerator_vram_gb is the accurate multi-device topology when
        # populated (always true for real hardware via hardware_bridge.py).
        # Callers that only set the legacy single-pool vram_gb (most tests,
        # and any pre-topology construction) leave it empty -- that still
        # means exactly one visible device, not zero.
        visible = len(context.hardware.accelerator_vram_gb) or (
            1 if context.hardware.vram_gb > 0 else 0
        )
        if count > visible:
            raise ValueError(
                f"backend.runtime.active_accelerator_count={count} requests more "
                f"accelerators than are visible ({visible})"
            )
        return count

    @staticmethod
    def _tail(path: Path, lines: int = 200) -> str:
        # 30 was enough for a single-process worker's own traceback, but
        # under accelerate-launch DDP a crashed rank's real traceback is
        # followed by torchrun's own ChildFailedError summary (its
        # boilerplate alone runs ~20-25 lines per failed rank) -- a short
        # tail can show only that summary and crowd out the one thing a
        # developer actually needs: what the worker itself raised.
        if not path.exists():
            return ""
        return "\n".join(
            path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
        )

    @staticmethod
    def _resource_usage_from_worker(
        worker_result: Mapping[str, Any],
        *,
        wall_seconds: float,
        fallback_gpu: bool,
    ) -> ResourceUsage:
        raw = worker_result.get("resource_usage", {})
        if not isinstance(raw, Mapping):
            raise RuntimeError("worker result resource_usage must be a mapping")
        if raw:
            active_count = int(raw.get("active_accelerator_count", 0))
            visible_count = int(raw.get("visible_accelerator_count", active_count))
            peak_raw = raw.get("peak_vram_gb_by_accelerator", {})
            if not isinstance(peak_raw, Mapping):
                raise RuntimeError("worker peak_vram_gb_by_accelerator must be a mapping")
            peaks = {str(key): float(value) for key, value in peak_raw.items()}
        else:
            # Backward-compatible fallback for old/mock worker manifests. New real
            # workers always emit resource_usage.
            active_count = 1 if fallback_gpu else 0
            visible_count = active_count
            peaks = {}
        return ResourceUsage.from_wall_time(
            wall_seconds=wall_seconds,
            active_accelerator_count=active_count,
            visible_accelerator_count=visible_count,
            peak_vram_gb_by_accelerator=peaks,
        )

    @staticmethod
    def _hardware_aware_defaults_report(context: ExecutionContext) -> dict[str, Any]:
        backend = context.resolved_config.get("backend", {})
        backend = backend if isinstance(backend, Mapping) else {}
        training = backend.get("training", {})
        training = training if isinstance(training, Mapping) else {}
        return {
            "min_device_vram_gb": _min_device_vram_gb(context.hardware),
            "quantization_defaulted": "quantization" not in backend,
            "gradient_checkpointing_defaulted": "gradient_checkpointing" not in training,
            "activation_offload_mode": str(training.get("activation_offload", "off")),
            "optimizer_tiering_mode": str(training.get("optimizer_tiering", "off")),
            "frozen_layer_streaming_mode": str(training.get("frozen_layer_streaming", "off")),
            "detailed_timing_telemetry_enabled": bool(
                training.get("detailed_timing_telemetry", False)
            ),
        }

    @staticmethod
    def _activation_offload_evidence(
        *, spec: "TransformersPeftRunSpec", context: ExecutionContext, telemetry: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Predicted vs. actual, when there is a real prediction to compare
        against: only when the config said "auto" did a real experiment
        actually run a decision through -- for "always"/"off" there is no
        prediction, just the explicit choice. Re-fetches the experiment
        (a cache hit, since resolved_activation_offload already ran it
        once for this same model+recipe+hardware+batch_size during
        _spec_for) rather than threading the object through -- cheap, and
        keeps resolved_activation_offload's own return type a plain bool.

        "actual" is intentionally not a single like-for-like number against
        the experiment's single-step prediction: a full training run's
        wall-clock includes data loading, checkpointing, and logging
        overhead the experiment's isolated forward+backward+step does not.
        Reporting both the real measured peak VRAM and the real average
        per-step time lets a caller judge the comparison themselves rather
        than this method collapsing it into one possibly-misleading ratio.
        """
        backend = context.resolved_config.get("backend", {})
        backend = backend if isinstance(backend, Mapping) else {}
        training = backend.get("training", {})
        training = training if isinstance(training, Mapping) else {}
        mode = str(training.get("activation_offload", "off")).strip().lower()

        global_step = telemetry.get("global_step")
        train_runtime_seconds = telemetry.get("train_runtime_seconds")
        actual: dict[str, Any] = {
            "actual_peak_vram_gb": telemetry.get("peak_vram_gb"),
            "actual_avg_step_seconds": (
                float(train_runtime_seconds) / global_step
                if isinstance(global_step, int) and global_step > 0
                and isinstance(train_runtime_seconds, (int, float))
                else None
            ),
            # Real transfer pressure: every byte the pack/unpack hooks
            # actually moved between device and host during this real run
            # (see transformers_worker.train's activation_offload_bytes_
            # transferred) -- None when offload wasn't active, since there
            # is nothing to report, not because it was zero.
            "actual_bytes_transferred": telemetry.get("activation_offload_bytes_transferred"),
        }
        if mode != "auto" or isinstance(training.get("activation_offload"), bool):
            return {"mode": mode, "resolved": spec.activation_offload, **actual}

        from ..activation_offload import run_activation_offload_experiment

        try:
            experiment = run_activation_offload_experiment(
                resolved_config=context.resolved_config, context=context, work_dir=context.work_dir
            )
        except Exception:
            # Evidence is best-effort: a cache-read hiccup here must never
            # fail an otherwise-successful training run.
            return {"mode": mode, "resolved": spec.activation_offload, **actual}
        return {
            "mode": mode,
            "resolved": spec.activation_offload,
            "predicted_available": experiment.available,
            "predicted_required": experiment.required,
            "predicted_recommended": experiment.recommended,
            "predicted_baseline_peak_vram_gb": experiment.baseline_peak_vram_gb,
            "predicted_offload_peak_vram_gb": experiment.offload_peak_vram_gb,
            "predicted_vram_saved_gb": experiment.vram_saved_gb,
            "predicted_wall_time_penalty_ratio": experiment.wall_time_penalty_ratio,
            **actual,
        }

    @staticmethod
    def _optimizer_tiering_evidence(
        *, spec: "TransformersPeftRunSpec", context: ExecutionContext, telemetry: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Predicted vs. actual for optimizer_tiering, structurally
        identical to _activation_offload_evidence -- see that method's
        docstring for why "actual" stays separate real numbers rather
        than one collapsed ratio, and why the experiment is re-fetched
        (cache hit) instead of threaded through.

        There is no actual-PCIe-bytes-transferred counterpart to
        activation_offload's actual_bytes_transferred here: bitsandbytes'
        CUDA-unified-memory paging happens inside the driver/kernel, not
        through a Python-visible tensor copy this code can hook and
        count. actual_optimizer_state_bytes (real tensor introspection
        of the live optimizer's state, identical approach to Phase 7A/7C's
        own measurement) is what's actually observable from here.
        """
        backend = context.resolved_config.get("backend", {})
        backend = backend if isinstance(backend, Mapping) else {}
        training = backend.get("training", {})
        training = training if isinstance(training, Mapping) else {}
        mode = str(training.get("optimizer_tiering", "off")).strip().lower()

        global_step = telemetry.get("global_step")
        train_runtime_seconds = telemetry.get("train_runtime_seconds")
        actual: dict[str, Any] = {
            "actual_avg_step_seconds": (
                float(train_runtime_seconds) / global_step
                if isinstance(global_step, int) and global_step > 0
                and isinstance(train_runtime_seconds, (int, float))
                else None
            ),
            "actual_optimizer_state_bytes": telemetry.get("optimizer_state_bytes"),
        }
        if mode != "auto" or isinstance(training.get("optimizer_tiering"), bool):
            return {"mode": mode, "resolved": spec.optimizer_tiering, **actual}

        from ..optimizer_tiering import run_optimizer_tiering_experiment

        try:
            experiment = run_optimizer_tiering_experiment(
                resolved_config=context.resolved_config, context=context, work_dir=context.work_dir
            )
        except Exception:
            # Evidence is best-effort: a cache-read hiccup here must never
            # fail an otherwise-successful training run.
            return {"mode": mode, "resolved": spec.optimizer_tiering, **actual}
        baseline = experiment.variant("adamw")
        paged = experiment.variant("paged_adamw")
        return {
            "mode": mode,
            "resolved": spec.optimizer_tiering,
            "predicted_available": experiment.available,
            "predicted_required": experiment.required,
            "predicted_recommended": experiment.recommended,
            "predicted_model_peak_vram_gb": experiment.model_peak_vram_gb,
            "predicted_baseline_state_bytes": baseline.state_bytes if baseline else None,
            "predicted_paged_state_bytes": paged.state_bytes if paged else None,
            "predicted_wall_time_penalty_ratio": experiment.wall_time_penalty_ratio,
            **actual,
        }

    @staticmethod
    def _frozen_layer_streaming_evidence(
        *, spec: "TransformersPeftRunSpec", context: ExecutionContext, telemetry: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Predicted vs. actual for frozen_layer_streaming, structurally
        identical to _activation_offload_evidence -- see that method's
        docstring for why "actual" stays separate real numbers rather
        than one collapsed ratio, and why the experiment is re-fetched
        (cache hit) instead of threaded through.

        actual_bytes_transferred here IS directly observable (unlike
        optimizer_tiering's PCIe traffic): memory_fabric.py's
        FrozenLayerPrefetchRuntime tracks every real H2D copy it issues
        in Python, the same way activation_offload's pack/unpack hooks
        do -- see transformers_worker.train's frozen_layer_streaming_
        bytes_transferred.
        """
        backend = context.resolved_config.get("backend", {})
        backend = backend if isinstance(backend, Mapping) else {}
        training = backend.get("training", {})
        training = training if isinstance(training, Mapping) else {}
        mode = str(training.get("frozen_layer_streaming", "off")).strip().lower()

        global_step = telemetry.get("global_step")
        train_runtime_seconds = telemetry.get("train_runtime_seconds")
        actual: dict[str, Any] = {
            "actual_avg_step_seconds": (
                float(train_runtime_seconds) / global_step
                if isinstance(global_step, int) and global_step > 0
                and isinstance(train_runtime_seconds, (int, float))
                else None
            ),
            "actual_bytes_transferred": telemetry.get("frozen_layer_streaming_bytes_transferred"),
        }
        if mode != "auto" or isinstance(training.get("frozen_layer_streaming"), bool):
            return {"mode": mode, "resolved": spec.frozen_layer_streaming, **actual}

        from ..frozen_layer_streaming import run_frozen_layer_streaming_experiment

        try:
            experiment = run_frozen_layer_streaming_experiment(
                resolved_config=context.resolved_config, context=context, work_dir=context.work_dir
            )
        except Exception:
            # Evidence is best-effort: a cache-read hiccup here must never
            # fail an otherwise-successful training run.
            return {"mode": mode, "resolved": spec.frozen_layer_streaming, **actual}
        return {
            "mode": mode,
            "resolved": spec.frozen_layer_streaming,
            "predicted_available": experiment.available,
            "predicted_required": experiment.required,
            "predicted_recommended": experiment.recommended,
            "predicted_baseline_peak_vram_gb": experiment.baseline_peak_vram_gb,
            "predicted_streamed_peak_vram_gb": experiment.streamed_peak_vram_gb,
            "predicted_vram_saved_gb": experiment.vram_saved_gb,
            "predicted_wall_time_penalty_ratio": experiment.wall_time_penalty_ratio,
            **actual,
        }

    @staticmethod
    def _production_timing_evidence(
        *, spec: "TransformersPeftRunSpec", telemetry: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Real per-phase timing breakdown, when detailed_timing_telemetry
        was enabled for this run -- no predicted/auto path here, unlike
        activation_offload/optimizer_tiering/frozen_layer_streaming, since
        there is no placement decision to make: this is pure observation.

        forward_seconds/backward_seconds/optimizer_seconds come from
        real torch.cuda.synchronize()-bracketed timing around Trainer's
        own compute_loss/accelerator.backward/optimizer.step calls (see
        transformers_worker.train's _TrainingPhaseTimerCallback).
        data_loading_and_overhead_seconds is an honest residual --
        total wall time minus the three measured phases -- not a
        separately measured quantity; it is dominated by DataLoader
        fetch time but also includes ordinary Python/callback overhead,
        and this method does not claim to separate those.
        avg_gpu_utilization_percent is a real, NVML-sampled average
        across training (see the same callback's background sampler
        thread), a coarse but genuine proxy for "GPU idle/stall time"
        -- not a precise idle-time measurement.

        Deliberately does NOT report a separate all-reduce time: under
        multi-GPU DDP, gradient synchronization happens inside
        accelerator.backward() itself, so it is already folded into
        backward_seconds rather than broken out -- doing so correctly
        would need verification against real multi-GPU hardware this
        instrumentation has not had.
        """
        enabled = spec.detailed_timing_telemetry
        if not enabled:
            return {"enabled": False}
        train_runtime_seconds = telemetry.get("train_runtime_seconds")
        forward_seconds = telemetry.get("forward_seconds")
        backward_seconds = telemetry.get("backward_seconds")
        optimizer_seconds = telemetry.get("optimizer_seconds")
        other_seconds = None
        if (
            isinstance(train_runtime_seconds, (int, float))
            and isinstance(forward_seconds, (int, float))
            and isinstance(backward_seconds, (int, float))
            and isinstance(optimizer_seconds, (int, float))
        ):
            other_seconds = max(
                0.0,
                float(train_runtime_seconds)
                - float(forward_seconds)
                - float(backward_seconds)
                - float(optimizer_seconds),
            )
        return {
            "enabled": True,
            "forward_seconds": forward_seconds,
            "backward_seconds": backward_seconds,
            "optimizer_seconds": optimizer_seconds,
            "data_loading_and_overhead_seconds": other_seconds,
            "avg_gpu_utilization_percent": telemetry.get("avg_gpu_utilization_percent"),
        }

    def run(self, experiment: Experiment, context: ExecutionContext) -> TrainingArtifact:
        run_id = f"{experiment.experiment_id}-{uuid4().hex[:12]}"
        run_dir = (Path(context.work_dir) / ".chowder" / "runs" / run_id).resolve()
        run_dir.mkdir(parents=True, exist_ok=False)
        hardware_defaults = self._hardware_aware_defaults_report(context)
        spec = self._spec_for(experiment, context, run_dir=run_dir)
        active_accelerator_count = self._active_accelerator_count(context)
        if spec.activation_offload and active_accelerator_count > 1:
            # Explicit safe rejection, not silent best-effort: saved_tensors_
            # hooks is a per-process autograd context, and wrapping each
            # accelerate-launch rank's own trainer.train() call in it
            # *should* apply independently with no cross-rank interaction --
            # but that reasoning has not been proven on real multi-GPU
            # hardware (this codebase's own established discipline, per the
            # Phase 5 DDP acceptance work, is that unverified multi-GPU
            # claims do not ship as if proven). Reject clearly rather than
            # risk a real, currently-unverified DDP interaction.
            raise ValueError(
                "activation_offload is not yet verified safe under multi-GPU DDP "
                f"(active_accelerator_count={active_accelerator_count}); set "
                "backend.training.activation_offload to 'off' for multi-GPU runs"
            )
        if spec.optimizer_tiering and active_accelerator_count > 1:
            # Same explicit-rejection principle as activation_offload above:
            # bitsandbytes' paged optimizers have not been proven safe
            # under this project's own multi-GPU DDP launch path (each
            # accelerate-launch rank constructing its own paged optimizer
            # independently is plausible, but unverified on real hardware
            # here), so reject clearly rather than ship an unverified
            # multi-GPU claim.
            raise ValueError(
                "optimizer_tiering is not yet verified safe under multi-GPU DDP "
                f"(active_accelerator_count={active_accelerator_count}); set "
                "backend.training.optimizer_tiering to 'off' for multi-GPU runs"
            )
        if spec.frozen_layer_streaming and active_accelerator_count > 1:
            # Same explicit-rejection principle as activation_offload/
            # optimizer_tiering above: the custom autograd.Function and
            # dedicated CUDA prefetch stream in memory_fabric.py have
            # only been verified on single-GPU hardware here -- whether
            # per-rank CUDA streams interact safely under accelerate-
            # launch DDP is plausible but unproven, so reject clearly
            # rather than ship an unverified multi-GPU claim.
            raise ValueError(
                "frozen_layer_streaming is not yet verified safe under multi-GPU DDP "
                f"(active_accelerator_count={active_accelerator_count}); set "
                "backend.training.frozen_layer_streaming to 'off' for multi-GPU runs"
            )

        bound_inputs = self._bound_inputs(spec)
        if spec.resume_from_checkpoint is not None:
            self._verify_resume_checkpoint(spec, bound_inputs)
        if spec.save_strategy != "no":
            self._write_checkpoint_manifest(Path(spec.output_dir) / "trainer", bound_inputs)

        spec_path = run_dir / "run-spec.json"
        result_path = run_dir / "worker-result.json"
        stdout_path = run_dir / "stdout.log"
        stderr_path = run_dir / "stderr.log"
        spec_path.write_text(spec.canonical_json() + "\n", encoding="utf-8")

        command = self._worker_command(
            spec_path, result_path, active_accelerator_count=active_accelerator_count
        )
        started = time.perf_counter()
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr, text=True)
            self._processes[run_id] = process
            if self._cancellation is not None:
                self._cancellation._register_active(self, run_id)
            # A separate thread polling the worker's progress file, running
            # concurrently with (not instead of) the existing wait/timeout/
            # cancellation handling below -- this is purely additive, so
            # that well-tested lifecycle logic keeps working exactly as it
            # did before, with no callback bound.
            stop_polling = threading.Event()
            poll_thread: threading.Thread | None = None
            if self._progress_callback is not None:
                poll_thread = threading.Thread(
                    target=self._poll_progress,
                    args=(
                        Path(spec.output_dir) / "progress.json",
                        experiment.experiment_id,
                        stop_polling,
                    ),
                    daemon=True,
                )
                poll_thread.start()
            try:
                process.wait(timeout=spec.timeout_seconds)
            except subprocess.TimeoutExpired as exc:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise TimeoutError(f"training run {run_id} exceeded timeout") from exc
            finally:
                stop_polling.set()
                if poll_thread is not None:
                    poll_thread.join(timeout=5)
                self._processes.pop(run_id, None)
                if self._cancellation is not None:
                    self._cancellation._clear_active()

        elapsed = time.perf_counter() - started
        if process.returncode != 0:
            tail = self._tail(stderr_path)
            raise RuntimeError(
                f"transformers worker failed with exit code {process.returncode}:\n{tail}"
            )
        if not result_path.is_file():
            raise RuntimeError(
                "transformers worker exited successfully without a result manifest"
            )
        if not Path(spec.output_dir).is_dir():
            raise RuntimeError(
                "transformers worker exited successfully without an adapter artifact"
            )

        primary_sha = self._verify_input(spec.dataset, spec.dataset_sha256, label="training")
        replay_sha: str | None = None
        if spec.replay_dataset is not None:
            replay_sha = self._verify_input(
                spec.replay_dataset, spec.replay_sha256, label="replay"
            )
        parent_adapter_sha: str | None = None
        if spec.parent_adapter is not None:
            assert spec.parent_adapter_sha256 is not None
            parent_adapter_sha = self._verify_adapter(
                spec.parent_adapter,
                spec.parent_adapter_sha256,
                label="parent",
            )

        worker_result = json.loads(result_path.read_text(encoding="utf-8"))
        telemetry = worker_result.get("telemetry", {})
        versions = worker_result.get("versions", {})
        worker_provenance = worker_result.get("provenance", {})
        data_provenance = worker_result.get("data_provenance", {})
        if (
            not isinstance(telemetry, Mapping)
            or not isinstance(versions, Mapping)
            or not isinstance(worker_provenance, Mapping)
            or not isinstance(data_provenance, Mapping)
        ):
            raise RuntimeError(
                "worker result contains invalid telemetry/version/provenance payload"
            )

        usage = self._resource_usage_from_worker(
            worker_result,
            wall_seconds=elapsed,
            fallback_gpu=context.hardware.vram_gb > 0,
        )
        if active_accelerator_count > 1 and usage.active_accelerator_count != active_accelerator_count:
            raise RuntimeError(
                f"requested {active_accelerator_count} active accelerators via "
                f"accelerate launch, but the worker's own resource snapshot reports "
                f"{usage.active_accelerator_count} -- the multi-GPU launch did not "
                "actually engage every requested device; do not trust this run's "
                "GPU-hour accounting"
            )
        activation_offload_evidence = self._activation_offload_evidence(
            spec=spec, context=context, telemetry=telemetry
        )
        optimizer_tiering_evidence = self._optimizer_tiering_evidence(
            spec=spec, context=context, telemetry=telemetry
        )
        frozen_layer_streaming_evidence = self._frozen_layer_streaming_evidence(
            spec=spec, context=context, telemetry=telemetry
        )
        production_timing_evidence = self._production_timing_evidence(
            spec=spec, telemetry=telemetry
        )
        return TrainingArtifact(
            run_id=run_id,
            experiment_id=experiment.experiment_id,
            artifact_ref=spec.output_dir,
            gpu_hours=usage.gpu_hours,
            telemetry=dict(telemetry),
            resource_usage=usage,
            evidence={
                "backend": self.name,
                "execution_spec_sha256": spec.digest(),
                "recipe_sha256": spec.recipe_digest(),
                "dataset_sha256": primary_sha,
                "replay_dataset_sha256": replay_sha,
                "replay_ratio": spec.replay_ratio,
                "parent_adapter_sha256": parent_adapter_sha,
                "continued_from_parent_adapter": parent_adapter_sha is not None,
                "data_provenance": dict(data_provenance),
                "artifact_sha256": sha256_directory(spec.output_dir),
                "resolved_config_sha256": self._json_digest(context.resolved_config),
                "worker_result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
                "stdout_log": str(stdout_path),
                "stderr_log": str(stderr_path),
                "versions": dict(versions),
                "model_provenance": dict(worker_provenance),
                "resource_usage": {
                    "wall_seconds": usage.wall_seconds,
                    "accelerator_seconds": usage.accelerator_seconds,
                    "active_accelerator_count": usage.active_accelerator_count,
                    "visible_accelerator_count": usage.visible_accelerator_count,
                    "peak_vram_gb_by_accelerator": dict(
                        usage.peak_vram_gb_by_accelerator
                    ),
                },
                "seed": spec.seed,
                "requested_active_accelerator_count": active_accelerator_count,
                "hardware_aware_defaults": {
                    **hardware_defaults,
                    "resolved_quantization": spec.quantization,
                    "resolved_gradient_checkpointing": spec.gradient_checkpointing,
                    "resolved_activation_offload": spec.activation_offload,
                    "resolved_optimizer_tiering": spec.optimizer_tiering,
                    "resolved_frozen_layer_streaming": spec.frozen_layer_streaming,
                    "detailed_timing_telemetry_enabled": spec.detailed_timing_telemetry,
                },
                "activation_offload": activation_offload_evidence,
                "optimizer_tiering": optimizer_tiering_evidence,
                "frozen_layer_streaming": frozen_layer_streaming_evidence,
                "production_timing": production_timing_evidence,
                "checkpoint": {
                    "save_strategy": spec.save_strategy,
                    "save_steps": spec.save_steps,
                    "save_total_limit": spec.save_total_limit,
                    "resumed_from_checkpoint": spec.resume_from_checkpoint,
                    "trainer_dir": (
                        str(Path(spec.output_dir) / "trainer")
                        if spec.save_strategy != "no"
                        else None
                    ),
                },
            },
        )

    def cancel(self, run_id: str) -> None:
        process = self._processes.get(run_id)
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
