from __future__ import annotations

import hashlib
import json
import math
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping
from uuid import uuid4

from ..cancellation import CancellationToken
from ..executors import CostEstimate, ExecutionContext, TrainingArtifact
from ..models import Experiment
from ..provenance import sha256_directory, sha256_file
from ..resources import ResourceUsage
from ..run_events import TrainingProgressEvent
from ..unsloth_env import unsloth_env_dir, unsloth_python
from .training_data import (
    _build_chat_example,
    _chat_digest,
    _replay_sample_count,
    _validate_chat_messages,
    _verify_bound_adapter,
    _verify_bound_input,
)

# Initial, minimal scope (see docs -- the isolated Unsloth executor plan):
# one NVIDIA GPU, PEFT LoRA/QLoRA, standard PEFT adapter output. Continuing
# from a parent adapter is deliberately out of scope here and lands in a
# follow-up slice. Chat-format datasets are supported via a deterministic
# controller-side handoff (see _materialize_pretokenized_chat_dataset):
# the isolated Unsloth worker cannot import chowder.backends.training_data
# directly, so this module renders every row through that exact shared
# contract *before* handoff and hands the worker already-tokenized
# {input_ids, attention_mask, labels} rows instead of raw messages -- the
# worker never sees a chat template or masking decision, so there is no
# code path where it could drift from the Transformers backend's semantics.
# Chowder's own
# activation_offload/optimizer_tiering/frozen_layer_streaming are refused
# outright under this engine -- none of them have been verified against
# Unsloth's own patched model/attention implementation, and a silent no-op
# would misrepresent what actually ran.
#
# Checkpoint/resume (this slice): a distinct manifest filename from
# Transformers' own _CHECKPOINT_MANIFEST_NAME in transformers_peft.py is
# the whole mechanism for "reject a Transformers checkpoint resumed under
# engine='unsloth' or vice versa" -- a Transformers checkpoint directory
# has no chowder-unsloth-checkpoint-manifest.json file (and an Unsloth one
# has no chowder-checkpoint-manifest.json), so each engine's own resume
# check already fails closed on the other engine's checkpoint with no
# extra cross-engine detection code needed.

_ALLOWED_QUANTIZATION = {"none", "4bit"}
_CHECKPOINT_MANIFEST_NAME = "chowder-unsloth-checkpoint-manifest.json"


class UnslothConfigError(ValueError):
    """Raised when an unsloth-engine recipe requests something this
    executor cannot safely honor -- fails at config-resolution time,
    never as a silent no-op or a confusing mid-training crash."""


@dataclass(frozen=True)
class UnslothPeftRunSpec:
    base_model: str
    dataset: str
    output_dir: str
    dataset_sha256: str | None = None
    revision: str | None = None
    parent_adapter: str | None = None
    parent_adapter_sha256: str | None = None
    replay_dataset: str | None = None
    replay_sha256: str | None = None
    replay_ratio: float = 0.0
    dataset_format: str = "text"
    text_field: str = "text"
    messages_field: str = "messages"
    # Internal, never user-configured: True once _spec_for has replaced
    # dataset/dataset_sha256 with the materialized pretokenized handoff
    # file for a dataset_format="chat" run. Tells the worker the dataset
    # already has {input_ids, attention_mask, labels} columns.
    pretokenized: bool = False
    chat_total_token_count: int | None = None
    chat_assistant_token_count: int | None = None
    chat_replay_available_rows: int | None = None
    chat_replay_selected_rows: int | None = None
    max_length: int = 512
    epochs: float = 1.0
    max_steps: int = -1
    learning_rate: float = 2e-4
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    logging_steps: int = 10
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ()
    quantization: str = "none"
    seed: int = 1
    timeout_seconds: float | None = None
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
        has_parent_path = self.parent_adapter is not None
        has_parent_sha = self.parent_adapter_sha256 is not None
        if has_parent_path != has_parent_sha:
            raise ValueError("backend parent adapter path and SHA must be supplied together")
        if has_parent_path:
            assert self.parent_adapter is not None
            assert self.parent_adapter_sha256 is not None
            if not self.parent_adapter.strip():
                raise ValueError("backend parent adapter path cannot be empty")
            if len(self.parent_adapter_sha256) != 64:
                raise ValueError("backend parent adapter SHA must be a SHA-256 digest")
        has_replay_dataset = self.replay_dataset is not None
        has_replay_sha = self.replay_sha256 is not None
        if has_replay_dataset != has_replay_sha:
            raise ValueError("backend replay dataset and SHA must be supplied together")
        replay_ratio = float(self.replay_ratio)
        if has_replay_dataset:
            assert self.replay_dataset is not None
            assert self.replay_sha256 is not None
            if not self.replay_dataset.strip():
                raise ValueError("backend replay dataset cannot be empty")
            if len(self.replay_sha256) != 64:
                raise ValueError("backend replay SHA must be a SHA-256 digest")
            if not math.isfinite(replay_ratio) or replay_ratio <= 0 or replay_ratio > 10:
                raise ValueError("backend replay ratio must be finite and in (0, 10]")
        elif replay_ratio != 0.0:
            raise ValueError("backend replay ratio requires a replay dataset")
        if self.dataset_format not in {"text", "chat"}:
            raise ValueError(f"unsupported dataset_format: {self.dataset_format}")
        if not self.text_field.strip():
            raise ValueError("backend.text_field cannot be empty")
        if self.dataset_format == "chat" and not self.messages_field.strip():
            raise ValueError("backend.messages_field cannot be empty")
        if self.max_length <= 0:
            raise ValueError("backend.max_length must be positive")
        if self.epochs <= 0 or self.learning_rate <= 0:
            raise ValueError("training epochs and learning_rate must be positive")
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
        if self.quantization not in _ALLOWED_QUANTIZATION:
            raise ValueError(f"unsupported quantization: {self.quantization}")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
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

    @classmethod
    def from_resolved_config(
        cls,
        resolved_config: Mapping[str, Any],
        *,
        work_dir: str | Path,
        output_dir: str | Path,
        seed: int,
    ) -> "UnslothPeftRunSpec":
        backend = resolved_config.get("backend", {})
        backend = backend if isinstance(backend, Mapping) else {}
        training = backend.get("training", {})
        training = training if isinstance(training, Mapping) else {}
        lora = backend.get("lora", {})
        lora = lora if isinstance(lora, Mapping) else {}

        for unsupported, label in (
            ("activation_offload", "activation_offload"),
            ("optimizer_tiering", "optimizer_tiering"),
            ("frozen_layer_streaming", "frozen_layer_streaming"),
        ):
            raw = training.get(unsupported, "off")
            requested = bool(raw) if isinstance(raw, bool) else str(raw).strip().lower() != "off"
            if requested:
                raise UnslothConfigError(
                    f"backend.training.{label} is not supported under engine='unsloth' "
                    f"(unverified against Unsloth's own patched model/attention "
                    f"implementation); set it to 'off' or use engine='transformers'"
                )

        dataset_raw = Path(str(backend.get("dataset", "")))
        if not dataset_raw.is_absolute():
            dataset_raw = Path(work_dir) / dataset_raw
        dataset = str(dataset_raw.resolve())
        target_modules = tuple(lora.get("target_modules", ()) or ())

        resume_raw = backend.get("resume_from_checkpoint")
        resume_from_checkpoint: str | None = None
        if resume_raw is not None:
            resume_path = Path(str(resume_raw))
            if not resume_path.is_absolute():
                resume_path = Path(work_dir) / resume_path
            resume_from_checkpoint = str(resume_path.resolve())

        parent_adapter_cfg = backend.get("parent_adapter", {})
        parent_adapter_cfg = parent_adapter_cfg if isinstance(parent_adapter_cfg, Mapping) else {}
        parent_adapter_path: str | None = None
        if parent_adapter_cfg.get("path") is not None:
            resolved_parent = Path(str(parent_adapter_cfg.get("path")))
            if not resolved_parent.is_absolute():
                resolved_parent = Path(work_dir) / resolved_parent
            parent_adapter_path = str(resolved_parent.resolve())
        parent_adapter_sha = parent_adapter_cfg.get("sha256")

        replay_cfg = backend.get("replay", {})
        replay_cfg = replay_cfg if isinstance(replay_cfg, Mapping) else {}
        replay_dataset_path: str | None = None
        if replay_cfg.get("dataset") is not None:
            resolved_replay = Path(str(replay_cfg.get("dataset")))
            if not resolved_replay.is_absolute():
                resolved_replay = Path(work_dir) / resolved_replay
            replay_dataset_path = str(resolved_replay.resolve())
        replay_sha = replay_cfg.get("sha256")

        return cls(
            base_model=str(backend.get("base_model", "")),
            dataset=dataset,
            output_dir=str(output_dir),
            dataset_sha256=backend.get("dataset_sha256"),
            revision=backend.get("revision"),
            parent_adapter=parent_adapter_path,
            parent_adapter_sha256=(
                str(parent_adapter_sha) if parent_adapter_sha is not None else None
            ),
            replay_dataset=replay_dataset_path,
            replay_sha256=(str(replay_sha) if replay_sha is not None else None),
            replay_ratio=(float(replay_cfg.get("ratio", 1.0)) if replay_dataset_path is not None else 0.0),
            dataset_format=str(backend.get("dataset_format", "text")),
            text_field=str(backend.get("text_field", "text")),
            messages_field=str(backend.get("messages_field", "messages")),
            max_length=int(backend.get("max_length", 512)),
            epochs=float(training.get("epochs", 1.0)),
            max_steps=int(training.get("max_steps", -1)),
            learning_rate=float(training.get("learning_rate", 2e-4)),
            batch_size=int(training.get("batch_size", 1)),
            gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 4)),
            logging_steps=int(training.get("logging_steps", 10)),
            lora_r=int(lora.get("r", 16)),
            lora_alpha=int(lora.get("alpha", 32)),
            lora_dropout=float(lora.get("dropout", 0.05)),
            target_modules=target_modules,
            quantization=str(backend.get("quantization", "none")),
            seed=seed,
            timeout_seconds=(backend.get("runtime", {}) or {}).get("timeout_seconds"),
            offline=bool(backend.get("offline", False)),
            save_strategy=str(training.get("save_strategy", "no")),
            save_steps=int(training.get("save_steps", 0)),
            save_total_limit=(
                int(training["save_total_limit"])
                if training.get("save_total_limit") is not None
                else None
            ),
            resume_from_checkpoint=resume_from_checkpoint,
        )


def _materialize_pretokenized_chat_dataset(
    spec: UnslothPeftRunSpec, *, work_dir: str | Path
) -> tuple[str, str, int, int, int, int]:
    """Render every row of a dataset_format="chat" dataset (primary, plus
    a sampled replay slice when spec.replay_dataset is set) into
    {input_ids, attention_mask, labels} via the exact same shared contract
    transformers_worker.py uses (chowder.backends.training_data's
    _validate_chat_messages/_build_chat_example/_replay_sample_count), then
    write it out as a plain JSONL file the isolated worker can load with
    zero chat-template, masking, or replay-mixing logic of its own.

    Replay merging happens on raw rows *before* tokenization -- the same
    order transformers_worker.py uses -- so a token that came from a replay
    row is indistinguishable from a primary row's token by the time the
    worker sees it; only this function's returned counts, and the caller's
    evidence, know the split.

    Returns (pretokenized_path, pretokenized_sha256, total_token_count,
    assistant_token_count, replay_available_rows, replay_selected_rows).
    Content-addressed by (primary content, replay content + ratio, base
    model + revision, max_length): re-running with identical inputs reuses
    the cached file rather than re-tokenizing, but any real change to any
    of those inputs -- including a replay dataset edit or ratio change --
    produces a different cache key, never a stale hit.

    Requires `transformers`/`datasets` to be importable in the *controller*
    process (not the isolated Unsloth env) -- this is the deliberate
    design this cross-environment handoff calls for: pre-render in the
    environment that can import the shared tokenization contract, hand the
    isolated environment already-tokenized rows it needs no chat-aware code
    to consume.
    """
    from datasets import concatenate_datasets, load_dataset
    from transformers import AutoTokenizer

    dataset_sha256 = _verify_bound_input(spec.dataset, spec.dataset_sha256, label="training")
    replay_sha256 = (
        _verify_bound_input(spec.replay_dataset, spec.replay_sha256, label="replay")
        if spec.replay_dataset is not None
        else None
    )
    cache_key = hashlib.sha256(
        json.dumps(
            {
                "dataset_sha256": dataset_sha256,
                "replay_sha256": replay_sha256,
                "replay_ratio": spec.replay_ratio,
                "base_model": spec.base_model,
                "revision": spec.revision,
                "max_length": spec.max_length,
                "messages_field": spec.messages_field,
                "seed": spec.seed,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    cache_dir = Path(work_dir) / ".chowder" / "_unsloth_chat_handoff"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{cache_key}.jsonl"
    meta_path = cache_dir / f"{cache_key}.meta.json"
    if cache_path.is_file() and meta_path.is_file():
        cached_sha = sha256_file(cache_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("pretokenized_sha256") == cached_sha:
            return (
                str(cache_path),
                cached_sha,
                int(meta["total_token_count"]),
                int(meta["assistant_token_count"]),
                int(meta["replay_available_rows"]),
                int(meta["replay_selected_rows"]),
            )

    tokenizer = AutoTokenizer.from_pretrained(
        spec.base_model, revision=spec.revision, local_files_only=spec.offline
    )
    primary = load_dataset("json", data_files=spec.dataset, split="train")
    if spec.messages_field not in primary.column_names:
        raise UnslothConfigError(
            f"chat dataset is missing messages field {spec.messages_field!r}; "
            f"columns={primary.column_names}"
        )
    primary = primary.select_columns([spec.messages_field])
    primary_rows = len(primary)

    replay_available_rows = 0
    replay_selected_rows = 0
    rows = primary
    if spec.replay_dataset is not None:
        replay = load_dataset("json", data_files=spec.replay_dataset, split="train")
        if spec.messages_field not in replay.column_names:
            raise UnslothConfigError(
                f"chat replay dataset is missing messages field {spec.messages_field!r}; "
                f"columns={replay.column_names}"
            )
        replay = replay.select_columns([spec.messages_field])
        replay_available_rows = len(replay)
        replay_selected_rows = _replay_sample_count(
            primary_rows, replay_available_rows, spec.replay_ratio
        )
        if replay_selected_rows:
            selected_replay = replay.shuffle(seed=spec.seed).select(range(replay_selected_rows))
            rows = concatenate_datasets([primary, selected_replay]).shuffle(seed=spec.seed)

    _chat_digest(rows, spec.messages_field)  # validated as a real digest input; not persisted here

    total_token_count = 0
    assistant_token_count = 0
    tmp_path = cache_path.with_suffix(".jsonl.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for index in range(len(rows)):
            messages = _validate_chat_messages(rows[index][spec.messages_field], row_index=index)
            example = _build_chat_example(
                tokenizer, messages, max_length=spec.max_length, row_index=index
            )
            total_token_count += len(example["input_ids"])
            assistant_token_count += sum(1 for label in example["labels"] if label != -100)
            handle.write(json.dumps(example) + "\n")
    tmp_path.replace(cache_path)

    pretokenized_sha256 = sha256_file(cache_path)
    meta_path.write_text(
        json.dumps(
            {
                "pretokenized_sha256": pretokenized_sha256,
                "total_token_count": total_token_count,
                "assistant_token_count": assistant_token_count,
                "replay_available_rows": replay_available_rows,
                "replay_selected_rows": replay_selected_rows,
            }
        ),
        encoding="utf-8",
    )
    return (
        str(cache_path),
        pretokenized_sha256,
        total_token_count,
        assistant_token_count,
        replay_available_rows,
        replay_selected_rows,
    )


class UnslothPeftExecutor:
    name = "unsloth-peft"

    def __init__(self) -> None:
        self._processes: dict[str, subprocess.Popen[Any]] = {}
        self._cancellation: CancellationToken | None = None
        self._progress_callback: Callable[[TrainingProgressEvent], None] | None = None

    def bind_cancellation(self, token: CancellationToken | None) -> None:
        self._cancellation = token

    def bind_progress_callback(
        self, callback: Callable[[TrainingProgressEvent], None] | None
    ) -> None:
        self._progress_callback = callback

    def _poll_progress_once(
        self, progress_path: Path, experiment_id: str, last_step: int | None
    ) -> int | None:
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
                pass
        return data.get("step")

    def _poll_progress(
        self, progress_path: Path, experiment_id: str, stop: threading.Event
    ) -> None:
        last_step: int | None = None
        while not stop.is_set():
            last_step = self._poll_progress_once(progress_path, experiment_id, last_step)
            stop.wait(1.0)

    def profile(self, experiment: Experiment, context: ExecutionContext) -> CostEstimate:
        return CostEstimate(
            gpu_hours=max(0.0, experiment.estimated_gpu_hours),
            confidence=0.25,
            notes=("unsloth engine: using experiment-declared GPU-hour estimate",),
        )

    @staticmethod
    def _isolated_python(work_dir: str | Path) -> Path:
        env_dir = unsloth_env_dir(work_dir)
        python_executable = unsloth_python(env_dir)
        if not python_executable.is_file():
            raise UnslothConfigError(
                f"no isolated Unsloth environment found at {env_dir}; run "
                "`chowder setup unsloth` before training with engine='unsloth'"
            )
        return python_executable

    @staticmethod
    def _worker_script_path() -> Path:
        return Path(__file__).with_name("unsloth_worker.py")

    @staticmethod
    def _environment_manifest_sha256(work_dir: str | Path) -> str | None:
        manifest_path = unsloth_env_dir(work_dir) / "chowder-unsloth-manifest.json"
        if not manifest_path.is_file():
            return None
        return sha256_file(manifest_path)

    @staticmethod
    def _bound_inputs(spec: UnslothPeftRunSpec, *, environment_manifest_sha256: str | None) -> dict[str, Any]:
        """The training inputs an Unsloth checkpoint is bound to -- same
        principle as TransformersPeftExecutor._bound_inputs, plus the
        isolated environment's own manifest digest, since an Unsloth
        checkpoint's optimizer/scheduler state is only meaningful for the
        exact Unsloth/Torch/PEFT/TRL versions that produced it. epochs and
        max_steps are excluded on purpose (extending training length is the
        point of resuming); everything else that could invalidate optimizer
        state is included.
        """
        recipe = spec.to_dict()
        for key in (
            "output_dir",
            "dataset",
            "parent_adapter",
            "replay_dataset",
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
            "checkpoint_recipe_sha256": hashlib.sha256(recipe_payload.encode("utf-8")).hexdigest(),
            "base_model": spec.base_model,
            "revision": spec.revision,
            "dataset_sha256": spec.dataset_sha256,
            "parent_adapter_sha256": spec.parent_adapter_sha256,
            "replay_dataset_sha256": spec.replay_sha256,
            "environment_manifest_sha256": environment_manifest_sha256,
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
        cls, spec: UnslothPeftRunSpec, bound_inputs: Mapping[str, Any]
    ) -> None:
        """Reject a resume if any bound training input (including the
        isolated environment itself) has changed since this checkpoint was
        produced -- an Unsloth checkpoint's optimizer/scheduler state is
        only trustworthy for the exact recipe, model, data, and Unsloth
        environment it came from. A checkpoint directory with no manifest
        at all (e.g. a Transformers checkpoint pointed at under
        engine='unsloth' by mistake) is refused the same way -- there is
        no recorded bound inputs to verify against, so it cannot be
        trusted rather than assumed compatible.
        """
        assert spec.resume_from_checkpoint is not None
        checkpoint_dir = Path(spec.resume_from_checkpoint).resolve()
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(f"resume_from_checkpoint not found: {checkpoint_dir}")
        manifest_path = checkpoint_dir.parent / _CHECKPOINT_MANIFEST_NAME
        if not manifest_path.is_file():
            raise RuntimeError(
                f"no Unsloth checkpoint manifest found at {manifest_path} -- refusing to "
                "resume from a checkpoint with no recorded bound inputs to verify against "
                "(this may not be an Unsloth-produced checkpoint)"
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

    def _spec_for(
        self, experiment: Experiment, context: ExecutionContext, *, run_dir: Path
    ) -> UnslothPeftRunSpec:
        spec = UnslothPeftRunSpec.from_resolved_config(
            context.resolved_config,
            work_dir=context.work_dir,
            output_dir=run_dir / "adapter",
            seed=context.seed,
        )
        # Same convention as TransformersPeftExecutor._spec_for: bind the
        # spec to the dataset's real, actually-measured digest immediately
        # when the caller didn't already pin one, so every downstream use
        # of spec.dataset_sha256 (the checkpoint manifest, evidence, the
        # spec JSON sent to the worker) carries a concrete value rather
        # than staying None.
        primary_sha = _verify_bound_input(spec.dataset, spec.dataset_sha256, label="training")
        if spec.dataset_sha256 is None:
            spec = replace(spec, dataset_sha256=primary_sha)

        if spec.replay_dataset is not None:
            # Same convention as TransformersPeftExecutor._spec_for: verify
            # the replay dataset's own digest up front, and refuse a replay
            # file that's literally the same file as the primary dataset --
            # that would double-count every row rather than genuinely
            # rehearsing prior capability.
            _verify_bound_input(spec.replay_dataset, spec.replay_sha256, label="replay")
            if Path(spec.replay_dataset).resolve() == Path(spec.dataset).resolve():
                raise ValueError("training and replay datasets must be different files")

        if spec.dataset_format == "chat":
            # Pre-render every row (primary + a sampled replay slice, if
            # configured) into {input_ids, attention_mask, labels} in this
            # (controller) process, then repoint the spec at that
            # materialized file -- every downstream consumer (checkpoint
            # manifest binding, bound_inputs, the worker's own re-verify,
            # evidence) now sees the pretokenized file's own real digest,
            # with no special-casing needed anywhere else in this class.
            (
                path,
                sha,
                total_tokens,
                assistant_tokens,
                replay_available,
                replay_selected,
            ) = _materialize_pretokenized_chat_dataset(spec, work_dir=context.work_dir)
            spec = replace(
                spec,
                dataset=path,
                dataset_sha256=sha,
                pretokenized=True,
                chat_total_token_count=total_tokens,
                chat_assistant_token_count=assistant_tokens,
                chat_replay_available_rows=replay_available,
                chat_replay_selected_rows=replay_selected,
            )
        if spec.parent_adapter is not None:
            assert spec.parent_adapter_sha256 is not None
            _verify_bound_adapter(spec.parent_adapter, spec.parent_adapter_sha256, label="parent")
        return spec

    def run(self, experiment: Experiment, context: ExecutionContext) -> TrainingArtifact:
        run_id = f"{experiment.experiment_id}-{uuid4().hex[:12]}"
        run_dir = (Path(context.work_dir) / ".chowder" / "runs" / run_id).resolve()
        run_dir.mkdir(parents=True, exist_ok=False)
        spec = self._spec_for(experiment, context, run_dir=run_dir)
        python_executable = self._isolated_python(context.work_dir)

        environment_manifest_sha256 = self._environment_manifest_sha256(context.work_dir)
        bound_inputs = self._bound_inputs(spec, environment_manifest_sha256=environment_manifest_sha256)
        if spec.resume_from_checkpoint is not None:
            self._verify_resume_checkpoint(spec, bound_inputs)
        if spec.save_strategy != "no":
            self._write_checkpoint_manifest(Path(spec.output_dir) / "trainer", bound_inputs)

        spec_path = run_dir / "run-spec.json"
        result_path = run_dir / "worker-result.json"
        stdout_path = run_dir / "stdout.log"
        stderr_path = run_dir / "stderr.log"
        spec_path.write_text(spec.canonical_json() + "\n", encoding="utf-8")

        command = [
            str(python_executable),
            str(self._worker_script_path()),
            "--spec",
            str(spec_path),
            "--result",
            str(result_path),
        ]
        started = time.perf_counter()
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr, text=True)
            self._processes[run_id] = process
            if self._cancellation is not None:
                self._cancellation._register_active(self, run_id)
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
                raise TimeoutError(f"unsloth training run {run_id} exceeded timeout") from exc
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
                f"unsloth worker failed with exit code {process.returncode}:\n{tail}"
            )
        if not result_path.is_file():
            raise RuntimeError("unsloth worker exited successfully without a result manifest")
        if not Path(spec.output_dir).is_dir():
            raise RuntimeError("unsloth worker exited successfully without an adapter artifact")

        # Re-verify after the run too: the same real-input-tampering hazard
        # every other Chowder training executor guards against.
        primary_sha = _verify_bound_input(spec.dataset, spec.dataset_sha256, label="training")

        worker_result = json.loads(result_path.read_text(encoding="utf-8"))
        telemetry = worker_result.get("telemetry", {})
        versions = worker_result.get("versions", {})
        model_provenance = worker_result.get("model_provenance", {})
        if (
            not isinstance(telemetry, Mapping)
            or not isinstance(versions, Mapping)
            or not isinstance(model_provenance, Mapping)
        ):
            raise RuntimeError("worker result contains invalid telemetry/version/provenance payload")

        usage = self._resource_usage_from_worker(worker_result, wall_seconds=elapsed)

        # Chat-format replay was merged (and counted) by this controller
        # before handoff; text-format replay is merged by the worker
        # itself, which reports its own counts in telemetry. Exactly one
        # of these is the real source for a given run.
        replay_available_rows = spec.chat_replay_available_rows
        replay_selected_rows = spec.chat_replay_selected_rows
        if replay_available_rows is None:
            replay_available_rows = telemetry.get("replay_available_rows")
        if replay_selected_rows is None:
            replay_selected_rows = telemetry.get("replay_selected_rows")

        return TrainingArtifact(
            run_id=run_id,
            experiment_id=experiment.experiment_id,
            artifact_ref=spec.output_dir,
            gpu_hours=usage.gpu_hours,
            telemetry=dict(telemetry),
            resource_usage=usage,
            evidence={
                "backend": self.name,
                "engine": "unsloth",
                "execution_spec_sha256": spec.digest(),
                "dataset_format": spec.dataset_format,
                "dataset_sha256": primary_sha,
                "pretokenized": spec.pretokenized,
                "chat_total_token_count": spec.chat_total_token_count,
                "chat_assistant_token_count": spec.chat_assistant_token_count,
                "parent_adapter_sha256": spec.parent_adapter_sha256,
                "continued_from_parent_adapter": spec.parent_adapter_sha256 is not None,
                "replay_dataset_sha256": spec.replay_sha256,
                "replay_ratio": spec.replay_ratio,
                "replay_available_rows": replay_available_rows,
                "replay_selected_rows": replay_selected_rows,
                "artifact_sha256": sha256_directory(spec.output_dir),
                "resolved_config_sha256": hashlib.sha256(
                    json.dumps(context.resolved_config, sort_keys=True, default=str).encode(
                        "utf-8"
                    )
                ).hexdigest(),
                "worker_result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
                "stdout_log": str(stdout_path),
                "stderr_log": str(stderr_path),
                "versions": dict(versions),
                "model_provenance": dict(model_provenance),
                "resolved_target_modules": worker_result.get("resolved_target_modules"),
                "resource_usage": {
                    "wall_seconds": usage.wall_seconds,
                    "accelerator_seconds": usage.accelerator_seconds,
                    "active_accelerator_count": usage.active_accelerator_count,
                    "visible_accelerator_count": usage.visible_accelerator_count,
                    "peak_vram_gb_by_accelerator": dict(usage.peak_vram_gb_by_accelerator),
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
                    "environment_manifest_sha256": environment_manifest_sha256,
                },
            },
        )

    @staticmethod
    def _tail(path: Path, lines: int = 200) -> str:
        if not path.exists():
            return ""
        return "\n".join(
            path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
        )

    @staticmethod
    def _resource_usage_from_worker(
        worker_result: Mapping[str, Any], *, wall_seconds: float
    ) -> ResourceUsage:
        raw = worker_result.get("resource_usage", {})
        if not isinstance(raw, Mapping):
            raise RuntimeError("worker result resource_usage must be a mapping")
        active_count = int(raw.get("active_accelerator_count", 0))
        visible_count = int(raw.get("visible_accelerator_count", active_count))
        peak_raw = raw.get("peak_vram_gb_by_accelerator", {})
        if not isinstance(peak_raw, Mapping):
            raise RuntimeError("worker peak_vram_gb_by_accelerator must be a mapping")
        peaks = {str(key): float(value) for key, value in peak_raw.items()}
        return ResourceUsage.from_wall_time(
            wall_seconds=wall_seconds,
            active_accelerator_count=active_count,
            visible_accelerator_count=visible_count,
            peak_vram_gb_by_accelerator=peaks,
        )

    def cancel(self, run_id: str) -> None:
        """Terminate the tracked worker process, matching
        TransformersPeftExecutor.cancel exactly. Confirmed sufficient on
        real hardware (a real, mid-flight Unsloth training run, cancelled
        after 8 real seconds): the worker's PID was fully gone afterward
        (verified directly via the OS process table, not nvidia-smi's
        --query-compute-apps, which was observed to report a stale,
        unchanging process list on this Windows/WDDM machine and cannot be
        trusted for this check here) and run() returned promptly with a
        real RuntimeError, no hang. This worker never forks additional
        child processes (HF Trainer's default dataloader_num_workers=0,
        no other subprocess spawning in unsloth_worker.py), so there is no
        process *tree* to kill in the current design -- if a future
        real-hardware run is found to leave orphans (e.g. from a changed
        worker that does spawn children), that would need real
        process-tree termination (e.g. a Windows job object or
        `taskkill /T /F`), not a preemptive, unverified addition here.
        """
        process = self._processes.get(run_id)
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
