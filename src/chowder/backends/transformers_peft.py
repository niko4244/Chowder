from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from ..executors import CostEstimate, ExecutionContext, TrainingArtifact
from ..models import Experiment
from ..provenance import sha256_directory, sha256_file


_ALLOWED_QUANTIZATION = {"none", "4bit"}
_ALLOWED_PRECISION = {"auto", "bf16", "fp16", "fp32"}


@dataclass(frozen=True)
class TransformersPeftRunSpec:
    base_model: str
    dataset: str
    output_dir: str
    dataset_sha256: str | None = None
    replay_dataset: str | None = None
    replay_sha256: str | None = None
    replay_ratio: float = 0.0
    revision: str | None = None
    text_field: str = "text"
    max_length: int = 512
    epochs: float = 1.0
    learning_rate: float = 2e-4
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    logging_steps: int = 10
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    use_rslora: bool = False
    quantization: str = "none"
    precision: str = "auto"
    gradient_checkpointing: bool = True
    seed: int = 1
    timeout_seconds: float | None = None
    trust_remote_code: bool = False

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

        if self.max_length <= 0:
            raise ValueError("backend.max_length must be positive")
        if self.epochs <= 0 or self.learning_rate <= 0:
            raise ValueError("training epochs and learning_rate must be positive")
        if self.batch_size <= 0 or self.gradient_accumulation_steps <= 0:
            raise ValueError("batch sizes must be positive")
        if self.lora_r <= 0 or self.lora_alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError("LoRA dropout must be in [0, 1)")
        if not self.target_modules:
            raise ValueError("at least one LoRA target module is required")
        if self.quantization not in _ALLOWED_QUANTIZATION:
            raise ValueError(f"unsupported quantization: {self.quantization}")
        if self.precision not in _ALLOWED_PRECISION:
            raise ValueError(f"unsupported precision: {self.precision}")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.trust_remote_code:
            raise ValueError("trust_remote_code is disabled for autonomous Chowder execution")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def recipe_digest(self) -> str:
        """Hash the reproducible training recipe, excluding machine/run paths."""
        recipe = self.to_dict()
        recipe.pop("output_dir", None)
        recipe.pop("dataset", None)
        recipe.pop("replay_dataset", None)
        recipe.pop("timeout_seconds", None)
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
        if not isinstance(training, Mapping) or not isinstance(lora, Mapping) or not isinstance(runtime, Mapping):
            raise ValueError("backend training/lora/runtime sections must be mappings")
        if replay is None:
            replay = {}
        if not isinstance(replay, Mapping):
            raise ValueError("backend replay section must be a mapping")

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

        dataset_sha = backend.get("dataset_sha256")
        replay_sha = replay.get("sha256")
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
            revision=(str(backend["revision"]) if backend.get("revision") is not None else None),
            text_field=str(backend.get("text_field", "text")),
            max_length=int(backend.get("max_length", 512)),
            epochs=float(training.get("epochs", 1.0)),
            learning_rate=float(training.get("learning_rate", 2e-4)),
            batch_size=int(training.get("batch_size", 1)),
            gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 4)),
            logging_steps=int(training.get("logging_steps", 10)),
            lora_r=int(lora.get("r", 16)),
            lora_alpha=int(lora.get("alpha", 32)),
            lora_dropout=float(lora.get("dropout", 0.05)),
            target_modules=tuple(
                str(x)
                for x in lora.get(
                    "target_modules", ("q_proj", "k_proj", "v_proj", "o_proj")
                )
            ),
            use_rslora=bool(lora.get("use_rslora", False)),
            quantization=str(backend.get("quantization", "none")).lower(),
            precision=str(backend.get("precision", "auto")).lower(),
            gradient_checkpointing=bool(training.get("gradient_checkpointing", True)),
            seed=int(config.get("seed", seed)),
            timeout_seconds=(
                float(runtime["timeout_seconds"])
                if runtime.get("timeout_seconds") is not None
                else None
            ),
            trust_remote_code=bool(backend.get("trust_remote_code", False)),
        )


class TransformersPeftExecutor:
    """Isolated Transformers + PEFT SFT backend.

    Heavy ML dependencies are imported only inside the worker subprocess. The
    controller remains importable and testable without torch/transformers.
    """

    name = "transformers-peft"

    def __init__(self) -> None:
        self._processes: dict[str, subprocess.Popen[Any]] = {}

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

    def _spec_for(
        self,
        experiment: Experiment,
        context: ExecutionContext,
        *,
        run_dir: Path,
    ) -> TransformersPeftRunSpec:
        if not context.resolved_config:
            raise ValueError("TransformersPeftExecutor requires ExecutionContext.resolved_config")
        artifact_dir = run_dir / "adapter"
        spec = TransformersPeftRunSpec.from_resolved_config(
            context.resolved_config,
            work_dir=context.work_dir,
            output_dir=artifact_dir,
            seed=context.seed,
        )
        primary_sha = self._verify_input(
            spec.dataset, spec.dataset_sha256, label="training"
        )
        if spec.dataset_sha256 is None:
            # Bind every run to the bytes observed immediately before launch,
            # even when a non-repair caller did not predeclare a digest.
            spec = replace(spec, dataset_sha256=primary_sha)

        if spec.replay_dataset is not None:
            replay_sha = self._verify_input(
                spec.replay_dataset, spec.replay_sha256, label="replay"
            )
            if Path(spec.replay_dataset).resolve() == Path(spec.dataset).resolve():
                raise ValueError("training and replay datasets must be different files")
            if replay_sha != spec.replay_sha256:
                raise ValueError("replay dataset content changed after proposal")
        return spec

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
            hours = max(0.0, float(steps) * float(seconds_per_step) / 3600.0)
            confidence = 0.75 if profile.get("source") == "measured" else 0.5
            notes = ("derived from backend step-time profile",)
        else:
            hours = max(0.0, experiment.estimated_gpu_hours)
            confidence = 0.25
            notes = ("using experiment-declared GPU-hour estimate; no measured step profile",)
        return CostEstimate(
            gpu_hours=hours,
            peak_vram_gb=float(peak_vram) if peak_vram is not None else None,
            confidence=confidence,
            notes=notes,
        )

    @staticmethod
    def _worker_command(spec_path: Path, result_path: Path) -> list[str]:
        return [
            sys.executable,
            "-m",
            "chowder.backends.transformers_worker",
            "--spec",
            str(spec_path),
            "--result",
            str(result_path),
        ]

    @staticmethod
    def _tail(path: Path, lines: int = 30) -> str:
        if not path.exists():
            return ""
        return "\n".join(
            path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
        )

    def run(self, experiment: Experiment, context: ExecutionContext) -> TrainingArtifact:
        run_id = f"{experiment.experiment_id}-{uuid4().hex[:12]}"
        run_dir = (Path(context.work_dir) / ".chowder" / "runs" / run_id).resolve()
        run_dir.mkdir(parents=True, exist_ok=False)
        spec = self._spec_for(experiment, context, run_dir=run_dir)

        spec_path = run_dir / "run-spec.json"
        result_path = run_dir / "worker-result.json"
        stdout_path = run_dir / "stdout.log"
        stderr_path = run_dir / "stderr.log"
        spec_path.write_text(spec.canonical_json() + "\n", encoding="utf-8")

        command = self._worker_command(spec_path, result_path)
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

        # Recheck after the worker exits so a concurrent mutation during training
        # cannot silently produce an artifact attributed to different input bytes.
        primary_sha = self._verify_input(
            spec.dataset, spec.dataset_sha256, label="training"
        )
        replay_sha: str | None = None
        if spec.replay_dataset is not None:
            replay_sha = self._verify_input(
                spec.replay_dataset, spec.replay_sha256, label="replay"
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

        return TrainingArtifact(
            run_id=run_id,
            experiment_id=experiment.experiment_id,
            artifact_ref=spec.output_dir,
            gpu_hours=elapsed / 3600.0,
            telemetry=dict(telemetry),
            evidence={
                "backend": self.name,
                "execution_spec_sha256": spec.digest(),
                "recipe_sha256": spec.recipe_digest(),
                "dataset_sha256": primary_sha,
                "replay_dataset_sha256": replay_sha,
                "replay_ratio": spec.replay_ratio,
                "data_provenance": dict(data_provenance),
                "artifact_sha256": sha256_directory(spec.output_dir),
                "resolved_config_sha256": self._json_digest(context.resolved_config),
                "worker_result_sha256": hashlib.sha256(
                    result_path.read_bytes()
                ).hexdigest(),
                "stdout_log": str(stdout_path),
                "stderr_log": str(stderr_path),
                "versions": dict(versions),
                "model_provenance": dict(worker_provenance),
                "seed": spec.seed,
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
