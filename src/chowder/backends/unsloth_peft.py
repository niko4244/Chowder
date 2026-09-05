from __future__ import annotations

import hashlib
import json
import math
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping
from uuid import uuid4

from ..cancellation import CancellationToken
from ..executors import CostEstimate, ExecutionContext, TrainingArtifact
from ..models import Experiment
from ..provenance import sha256_directory
from ..resources import ResourceUsage
from ..run_events import TrainingProgressEvent
from ..unsloth_env import unsloth_env_dir, unsloth_python
from .training_data import _verify_bound_input

# Initial, minimal scope (see docs -- the isolated Unsloth executor plan):
# one NVIDIA GPU, PEFT LoRA/QLoRA, standard PEFT adapter output, text-format
# datasets only. Chat-format datasets, checkpoint/resume, replay, and
# continuing from a parent adapter are deliberately out of scope here and
# land in a follow-up slice once the cross-environment data-handoff question
# (the isolated env cannot import chowder.backends.training_data directly)
# is resolved. Chowder's own activation_offload/optimizer_tiering/
# frozen_layer_streaming are refused outright under this engine -- none of
# them have been verified against Unsloth's own patched model/attention
# implementation, and a silent no-op would misrepresent what actually ran.

_ALLOWED_QUANTIZATION = {"none", "4bit"}


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
    text_field: str = "text"
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

    def __post_init__(self) -> None:
        if not self.base_model.strip():
            raise ValueError("backend.base_model is required")
        if not self.dataset.strip():
            raise ValueError("backend.dataset is required")
        if self.dataset_sha256 is not None and len(self.dataset_sha256) != 64:
            raise ValueError("backend.dataset_sha256 must be a SHA-256 digest")
        if not self.text_field.strip():
            raise ValueError("backend.text_field cannot be empty")
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

        return cls(
            base_model=str(backend.get("base_model", "")),
            dataset=dataset,
            output_dir=str(output_dir),
            dataset_sha256=backend.get("dataset_sha256"),
            revision=backend.get("revision"),
            text_field=str(backend.get("text_field", "text")),
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

    def _spec_for(
        self, experiment: Experiment, context: ExecutionContext, *, run_dir: Path
    ) -> UnslothPeftRunSpec:
        spec = UnslothPeftRunSpec.from_resolved_config(
            context.resolved_config,
            work_dir=context.work_dir,
            output_dir=run_dir / "adapter",
            seed=context.seed,
        )
        return spec

    def run(self, experiment: Experiment, context: ExecutionContext) -> TrainingArtifact:
        run_id = f"{experiment.experiment_id}-{uuid4().hex[:12]}"
        run_dir = (Path(context.work_dir) / ".chowder" / "runs" / run_id).resolve()
        run_dir.mkdir(parents=True, exist_ok=False)
        spec = self._spec_for(experiment, context, run_dir=run_dir)
        python_executable = self._isolated_python(context.work_dir)

        primary_sha = _verify_bound_input(spec.dataset, spec.dataset_sha256, label="training")

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
                "dataset_sha256": primary_sha,
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
        process = self._processes.get(run_id)
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
