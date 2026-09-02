from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from ..dependency_preflight import check_dependencies
from ..executors import CostEstimate, ExecutionContext, TrainingArtifact
from ..memory import HardwareProfile
from ..models import Experiment
from ..provenance import sha256_directory, sha256_file
from ..resources import ResourceUsage


_ALLOWED_QUANTIZATION = {"none", "4bit"}
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
        schedule shape, or dataset would be.
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
    def _tail(path: Path, lines: int = 30) -> str:
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
        }

    def run(self, experiment: Experiment, context: ExecutionContext) -> TrainingArtifact:
        run_id = f"{experiment.experiment_id}-{uuid4().hex[:12]}"
        run_dir = (Path(context.work_dir) / ".chowder" / "runs" / run_id).resolve()
        run_dir.mkdir(parents=True, exist_ok=False)
        hardware_defaults = self._hardware_aware_defaults_report(context)
        spec = self._spec_for(experiment, context, run_dir=run_dir)

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

        active_accelerator_count = self._active_accelerator_count(context)
        command = self._worker_command(
            spec_path, result_path, active_accelerator_count=active_accelerator_count
        )
        started = time.perf_counter()
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr, text=True)
            self._processes[run_id] = process
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
                self._processes.pop(run_id, None)

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
                },
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
