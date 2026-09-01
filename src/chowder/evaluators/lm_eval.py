from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from ..executors import EvaluationOutcome, ExecutionContext, TrainingArtifact
from ..models import Experiment
from ..provenance import sha256_directory

_ALLOWED_PRECISION = {"auto", "bf16", "fp16", "fp32"}
_ALLOWED_QUANTIZATION = {"none", "4bit"}


@dataclass(frozen=True)
class LmEvalSpec:
    base_model: str
    adapter_dir: str
    output_dir: str
    tasks: tuple[str, ...]
    metric_map: Mapping[str, str]
    revision: str | None = None
    device: str = "auto"
    batch_size: int | str = "auto"
    num_fewshot: int | None = None
    limit: int | float | None = None
    precision: str = "auto"
    quantization: str = "none"
    apply_chat_template: bool = False
    fewshot_as_multiturn: bool = True
    seed: int = 1
    timeout_seconds: float | None = None
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        if not self.base_model.strip():
            raise ValueError("lm-eval base_model is required")
        if not self.adapter_dir.strip():
            raise ValueError("lm-eval adapter_dir is required")
        if not self.tasks or any(not task.strip() for task in self.tasks):
            raise ValueError("lm-eval tasks must be non-empty")
        if len(self.tasks) != len(set(self.tasks)):
            raise ValueError("lm-eval tasks must be unique")
        if not self.metric_map:
            raise ValueError("lm-eval metric_map must explicitly select promotion metrics")
        for target, source in self.metric_map.items():
            if not str(target).strip() or ":" not in str(source):
                raise ValueError("metric_map entries must use target -> task:metric syntax")
        if self.precision not in _ALLOWED_PRECISION:
            raise ValueError(f"unsupported lm-eval precision: {self.precision}")
        if self.quantization not in _ALLOWED_QUANTIZATION:
            raise ValueError(f"unsupported lm-eval quantization: {self.quantization}")
        if isinstance(self.batch_size, int) and self.batch_size <= 0:
            raise ValueError("lm-eval batch_size must be positive")
        if self.num_fewshot is not None and self.num_fewshot < 0:
            raise ValueError("lm-eval num_fewshot cannot be negative")
        if self.limit is not None and self.limit <= 0:
            raise ValueError("lm-eval limit must be positive")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("lm-eval timeout_seconds must be positive")
        if self.trust_remote_code:
            raise ValueError("trust_remote_code is disabled for autonomous Chowder evaluation")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tasks"] = list(self.tasks)
        payload["metric_map"] = dict(self.metric_map)
        return payload

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_context(
        cls,
        *,
        config: Mapping[str, Any],
        artifact: TrainingArtifact,
        work_dir: str | Path,
        output_dir: str | Path,
        seed: int,
    ) -> "LmEvalSpec":
        backend = config.get("backend")
        evaluation = config.get("evaluation")
        if not isinstance(backend, Mapping) or not isinstance(evaluation, Mapping):
            raise ValueError("resolved config must contain backend and evaluation mappings")
        if evaluation.get("type") != "lm-eval":
            raise ValueError("resolved config evaluation.type is not lm-eval")

        raw_tasks = evaluation.get("tasks")
        raw_metric_map = evaluation.get("metric_map")
        if not isinstance(raw_tasks, (list, tuple)):
            raise ValueError("evaluation.tasks must be a list")
        if not isinstance(raw_metric_map, Mapping):
            raise ValueError("evaluation.metric_map must be a mapping")

        model_provenance = artifact.evidence.get("model_provenance", {})
        resolved_commit = (
            model_provenance.get("resolved_model_commit")
            if isinstance(model_provenance, Mapping)
            else None
        )
        revision = str(resolved_commit) if resolved_commit else (
            str(backend["revision"]) if backend.get("revision") is not None else None
        )

        quantization = str(evaluation.get("quantization", "inherit")).lower()
        if quantization == "inherit":
            quantization = str(backend.get("quantization", "none")).lower()
        precision = str(evaluation.get("precision", "inherit")).lower()
        if precision == "inherit":
            precision = str(backend.get("precision", "auto")).lower()

        runtime = evaluation.get("runtime", {})
        if not isinstance(runtime, Mapping):
            raise ValueError("evaluation.runtime must be a mapping")

        batch_size = evaluation.get("batch_size", "auto")
        if isinstance(batch_size, str):
            batch_size = batch_size.strip()
        elif batch_size is not None:
            batch_size = int(batch_size)

        return cls(
            base_model=str(backend.get("base_model", "")),
            adapter_dir=str(Path(artifact.artifact_ref).resolve()),
            output_dir=str(Path(output_dir).resolve()),
            tasks=tuple(str(task) for task in raw_tasks),
            metric_map={str(k): str(v) for k, v in raw_metric_map.items()},
            revision=revision,
            device=str(evaluation.get("device", "auto")),
            batch_size=batch_size,
            num_fewshot=(int(evaluation["num_fewshot"]) if evaluation.get("num_fewshot") is not None else None),
            limit=evaluation.get("limit"),
            precision=precision,
            quantization=quantization,
            apply_chat_template=bool(evaluation.get("apply_chat_template", False)),
            fewshot_as_multiturn=bool(evaluation.get("fewshot_as_multiturn", True)),
            seed=int(config.get("seed", seed)),
            timeout_seconds=(float(runtime["timeout_seconds"]) if runtime.get("timeout_seconds") is not None else None),
            trust_remote_code=bool(evaluation.get("trust_remote_code", False)),
        )


class LmEvalEvaluator:
    """Isolated adapter for EleutherAI lm-evaluation-harness."""

    name = "lm-eval"

    @staticmethod
    def _worker_command(spec_path: Path, result_path: Path) -> list[str]:
        return [
            sys.executable,
            "-m",
            "chowder.evaluators.lm_eval_worker",
            "--spec",
            str(spec_path),
            "--result",
            str(result_path),
        ]

    @staticmethod
    def _tail(path: Path, lines: int = 30) -> str:
        if not path.exists():
            return ""
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])

    def evaluate(
        self,
        *,
        experiment: Experiment,
        artifact: TrainingArtifact,
        context: ExecutionContext,
    ) -> EvaluationOutcome:
        if experiment.experiment_id != artifact.experiment_id:
            raise ValueError("artifact experiment_id does not match experiment")
        expected_digest = artifact.evidence.get("artifact_sha256")
        if not isinstance(expected_digest, str) or len(expected_digest) != 64:
            raise ValueError("training artifact is missing artifact_sha256 provenance")
        actual_digest = sha256_directory(artifact.artifact_ref)
        if actual_digest != expected_digest:
            raise ValueError("training artifact content digest changed before evaluation")
        if not context.resolved_config:
            raise ValueError("LmEvalEvaluator requires ExecutionContext.resolved_config")

        eval_id = f"{artifact.run_id}-lm-eval"
        eval_dir = (Path(context.work_dir) / ".chowder" / "evals" / eval_id).resolve()
        eval_dir.mkdir(parents=True, exist_ok=False)
        spec = LmEvalSpec.from_context(
            config=context.resolved_config,
            artifact=artifact,
            work_dir=context.work_dir,
            output_dir=eval_dir,
            seed=context.seed,
        )

        spec_path = eval_dir / "eval-spec.json"
        result_path = eval_dir / "eval-result.json"
        stdout_path = eval_dir / "stdout.log"
        stderr_path = eval_dir / "stderr.log"
        spec_path.write_text(spec.canonical_json() + "\n", encoding="utf-8")

        started = time.perf_counter()
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            process = subprocess.Popen(
                self._worker_command(spec_path, result_path),
                stdout=stdout,
                stderr=stderr,
                text=True,
            )
            try:
                process.wait(timeout=spec.timeout_seconds)
            except subprocess.TimeoutExpired as exc:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise TimeoutError(f"evaluation {eval_id} exceeded timeout") from exc

        elapsed = time.perf_counter() - started
        if process.returncode != 0:
            raise RuntimeError(
                f"lm-eval worker failed with exit code {process.returncode}:\n{self._tail(stderr_path)}"
            )
        if not result_path.is_file():
            raise RuntimeError("lm-eval worker exited successfully without a result manifest")

        payload = json.loads(result_path.read_text(encoding="utf-8"))
        metrics = payload.get("metrics")
        runtime = payload.get("runtime", {})
        versions = payload.get("versions", {})
        raw_digest = payload.get("raw_results_sha256")
        if not isinstance(metrics, Mapping) or not isinstance(runtime, Mapping) or not isinstance(versions, Mapping):
            raise RuntimeError("lm-eval result contains invalid evidence")
        if set(metrics) != set(spec.metric_map):
            raise RuntimeError("lm-eval result metric names do not match metric_map")
        gpu_count = int(runtime.get("gpu_count", 0))
        if gpu_count < 0:
            raise RuntimeError("lm-eval runtime reported a negative gpu_count")

        return EvaluationOutcome(
            run_id=eval_id,
            experiment_id=experiment.experiment_id,
            source_artifact_ref=artifact.artifact_ref,
            metrics={name: float(value) for name, value in metrics.items()},
            gpu_hours=(elapsed / 3600.0) * gpu_count,
            evidence={
                "evaluator": self.name,
                "training_run_id": artifact.run_id,
                "artifact_sha256": actual_digest,
                "evaluation_spec_sha256": spec.digest(),
                "evaluation_result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
                "raw_results_sha256": raw_digest,
                "tasks": list(spec.tasks),
                "metric_map": dict(spec.metric_map),
                "runtime": dict(runtime),
                "versions": dict(versions),
                "wall_time_seconds": elapsed,
                "stdout_log": str(stdout_path),
                "stderr_log": str(stderr_path),
                "seed": spec.seed,
            },
        )
