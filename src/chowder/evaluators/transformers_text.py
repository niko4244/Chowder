from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from ..executors import CostEstimate, EvaluationOutcome, ExecutionContext, TrainingArtifact
from ..models import Experiment
from ..protocol import protocol_fingerprint
from ..provenance import sha256_directory, sha256_file

_ALLOWED_SCORING = {"exact_match", "normalized_exact_match"}
_ALLOWED_PRECISION = {"auto", "bf16", "fp16", "fp32"}
_ALLOWED_QUANTIZATION = {"none", "4bit"}


@dataclass(frozen=True)
class EvalSuiteSpec:
    name: str
    dataset: str
    prompt_field: str = "prompt"
    expected_field: str = "expected"
    scoring: str = "normalized_exact_match"
    max_new_tokens: int = 64
    use_chat_template: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("evaluation suite name is required")
        if not self.dataset.strip():
            raise ValueError(f"evaluation suite {self.name!r} dataset is required")
        if self.scoring not in _ALLOWED_SCORING:
            raise ValueError(f"unsupported scoring method: {self.scoring}")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")


@dataclass(frozen=True)
class TransformersTextEvalSpec:
    base_model: str
    adapter_dir: str
    output_dir: str
    suites: tuple[EvalSuiteSpec, ...]
    revision: str | None = None
    precision: str = "auto"
    quantization: str = "none"
    device: str = "auto"
    seed: int = 1
    timeout_seconds: float | None = None
    trust_remote_code: bool = False
    offline: bool = False

    def __post_init__(self) -> None:
        if not self.base_model.strip():
            raise ValueError("evaluation base_model is required")
        if not self.adapter_dir.strip():
            raise ValueError("evaluation adapter_dir is required")
        if not self.suites:
            raise ValueError("at least one evaluation suite is required")
        names = [suite.name for suite in self.suites]
        if len(names) != len(set(names)):
            raise ValueError("evaluation suite names must be unique")
        if self.precision not in _ALLOWED_PRECISION:
            raise ValueError(f"unsupported evaluation precision: {self.precision}")
        if self.quantization not in _ALLOWED_QUANTIZATION:
            raise ValueError(f"unsupported evaluation quantization: {self.quantization}")
        if not self.device.strip():
            raise ValueError("evaluation device is required")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("evaluation timeout_seconds must be positive")
        if self.trust_remote_code:
            raise ValueError("trust_remote_code is disabled for autonomous Chowder evaluation")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["suites"] = [asdict(suite) for suite in self.suites]
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
    ) -> "TransformersTextEvalSpec":
        backend = config.get("backend")
        evaluation = config.get("evaluation")
        if not isinstance(backend, Mapping) or not isinstance(evaluation, Mapping):
            raise ValueError("resolved config must contain backend and evaluation mappings")
        if evaluation.get("type", "transformers-text") != "transformers-text":
            raise ValueError("resolved config evaluation.type is not transformers-text")

        raw_suites = evaluation.get("suites")
        if not isinstance(raw_suites, (list, tuple)) or not raw_suites:
            raise ValueError("evaluation.suites must be a non-empty list")
        suites: list[EvalSuiteSpec] = []
        for raw in raw_suites:
            if not isinstance(raw, Mapping):
                raise ValueError("each evaluation suite must be a mapping")
            dataset = Path(str(raw.get("dataset", "")))
            if not dataset.is_absolute():
                dataset = Path(work_dir) / dataset
            suites.append(
                EvalSuiteSpec(
                    name=str(raw.get("name", "")),
                    dataset=str(dataset.resolve()),
                    prompt_field=str(raw.get("prompt_field", "prompt")),
                    expected_field=str(raw.get("expected_field", "expected")),
                    scoring=str(raw.get("scoring", "normalized_exact_match")),
                    max_new_tokens=int(raw.get("max_new_tokens", evaluation.get("max_new_tokens", 64))),
                    use_chat_template=bool(raw.get("use_chat_template", evaluation.get("use_chat_template", False))),
                )
            )

        model_provenance = artifact.evidence.get("model_provenance", {})
        resolved_commit = (
            model_provenance.get("resolved_model_commit")
            if isinstance(model_provenance, Mapping)
            else None
        )
        revision = str(resolved_commit) if resolved_commit else (
            str(backend["revision"]) if backend.get("revision") is not None else None
        )

        inherited_quant = str(backend.get("quantization", "none")).lower()
        quantization = str(evaluation.get("quantization", "inherit")).lower()
        if quantization == "inherit":
            quantization = inherited_quant

        inherited_precision = str(backend.get("precision", "auto")).lower()
        precision = str(evaluation.get("precision", "inherit")).lower()
        if precision == "inherit":
            precision = inherited_precision

        runtime = evaluation.get("runtime", {})
        if not isinstance(runtime, Mapping):
            raise ValueError("evaluation.runtime must be a mapping")

        return cls(
            base_model=str(backend.get("base_model", "")),
            adapter_dir=str(Path(artifact.artifact_ref).resolve()),
            output_dir=str(Path(output_dir).resolve()),
            suites=tuple(suites),
            revision=revision,
            precision=precision,
            quantization=quantization,
            device=str(evaluation.get("device", "auto")),
            seed=int(config.get("seed", seed)),
            timeout_seconds=(float(runtime["timeout_seconds"]) if runtime.get("timeout_seconds") is not None else None),
            trust_remote_code=bool(evaluation.get("trust_remote_code", False)),
            offline=bool(evaluation.get("offline", backend.get("offline", False))),
        )


class TransformersTextEvaluator:
    """Independent deterministic text-generation evaluator."""

    name = "transformers-text"

    def __init__(self) -> None:
        self._processes: dict[str, subprocess.Popen[Any]] = {}

    def profile(self, experiment: Experiment, context: ExecutionContext) -> CostEstimate:
        config = context.resolved_config
        evaluation = config.get("evaluation", {}) if isinstance(config, Mapping) else {}
        hours = 0.0
        if isinstance(evaluation, Mapping) and evaluation.get("estimated_gpu_hours") is not None:
            hours = max(0.0, float(evaluation["estimated_gpu_hours"]))
        return CostEstimate(
            gpu_hours=hours,
            confidence=0.25,
            notes=(
                "using evaluation-declared GPU-hour estimate; no measured evaluation profile",
            ),
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

    @staticmethod
    def _worker_command(spec_path: Path, result_path: Path) -> list[str]:
        return [
            sys.executable,
            "-m",
            "chowder.evaluators.transformers_text_worker",
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
            raise ValueError("TransformersTextEvaluator requires ExecutionContext.resolved_config")

        eval_id = f"{artifact.run_id}-eval"
        eval_dir = (Path(context.work_dir) / ".chowder" / "evals" / eval_id).resolve()
        eval_dir.mkdir(parents=True, exist_ok=False)
        spec = TransformersTextEvalSpec.from_context(
            config=context.resolved_config,
            artifact=artifact,
            work_dir=context.work_dir,
            output_dir=eval_dir,
            seed=context.seed,
        )
        for suite in spec.suites:
            if not Path(suite.dataset).is_file():
                raise FileNotFoundError(f"evaluation dataset not found: {suite.dataset}")

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
            self._processes[eval_id] = process
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
            finally:
                self._processes.pop(eval_id, None)

        elapsed = time.perf_counter() - started
        if process.returncode != 0:
            raise RuntimeError(
                f"transformers evaluator failed with exit code {process.returncode}:\n{self._tail(stderr_path)}"
            )
        if not result_path.is_file():
            raise RuntimeError("transformers evaluator exited successfully without a result manifest")

        payload = json.loads(result_path.read_text(encoding="utf-8"))
        metrics = payload.get("metrics")
        suite_evidence = payload.get("suites")
        versions = payload.get("versions", {})
        runtime = payload.get("runtime", {})
        model_provenance = payload.get("model_provenance", {})
        if (
            not isinstance(metrics, Mapping)
            or not isinstance(suite_evidence, Mapping)
            or not isinstance(versions, Mapping)
            or not isinstance(runtime, Mapping)
            or not isinstance(model_provenance, Mapping)
        ):
            raise RuntimeError("evaluation result contains invalid metrics/evidence payload")
        gpu_count = int(runtime.get("gpu_count", 0))
        if gpu_count < 0:
            raise RuntimeError("evaluation runtime reported a negative gpu_count")

        expected_names = {suite.name for suite in spec.suites}
        if set(metrics) != expected_names:
            raise RuntimeError("evaluation result metric names do not match configured suites")
        if set(suite_evidence) != expected_names:
            raise RuntimeError("evaluation suite evidence names do not match configured suites")

        fingerprint_hashes: dict[str, str] = {}
        for suite_name, suite_payload in suite_evidence.items():
            if not isinstance(suite_payload, Mapping):
                raise RuntimeError(f"suite evidence for {suite_name!r} is invalid")
            fingerprint_ref = suite_payload.get("holdout_fingerprints_file")
            declared_digest = suite_payload.get("holdout_fingerprints_sha256")
            if not isinstance(fingerprint_ref, str) or not isinstance(declared_digest, str):
                raise RuntimeError(f"suite {suite_name!r} is missing holdout fingerprint evidence")
            fingerprint_path = Path(fingerprint_ref).resolve()
            if not fingerprint_path.is_relative_to(eval_dir):
                raise RuntimeError("holdout fingerprint evidence escaped evaluation directory")
            if not fingerprint_path.is_file():
                raise RuntimeError(f"holdout fingerprint evidence not found: {fingerprint_path}")
            actual_fingerprint_digest = sha256_file(fingerprint_path)
            if actual_fingerprint_digest != declared_digest:
                raise RuntimeError("holdout fingerprint evidence digest mismatch")
            fingerprint_hashes[str(suite_name)] = actual_fingerprint_digest

        dataset_hashes = {suite.name: sha256_file(suite.dataset) for suite in spec.suites}
        protocol = {
            "evaluator": self.name,
            "base_model": spec.base_model,
            "revision": spec.revision,
            "precision": spec.precision,
            "quantization": spec.quantization,
            "device": runtime.get("device"),
            "seed": spec.seed,
            "versions": dict(versions),
            "suites": [
                {
                    "name": suite.name,
                    "dataset_sha256": dataset_hashes[suite.name],
                    "prompt_field": suite.prompt_field,
                    "expected_field": suite.expected_field,
                    "scoring": suite.scoring,
                    "max_new_tokens": suite.max_new_tokens,
                    "use_chat_template": suite.use_chat_template,
                }
                for suite in spec.suites
            ],
        }
        protocol_sha = protocol_fingerprint(protocol)
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
                "evaluation_dataset_sha256": dataset_hashes,
                "evaluation_result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
                "protocol": protocol,
                "protocol_sha256": protocol_sha,
                "holdout_fingerprint_sha256": fingerprint_hashes,
                "suite_evidence": dict(suite_evidence),
                "versions": dict(versions),
                "runtime": dict(runtime),
                "model_provenance": dict(model_provenance),
                "wall_time_seconds": elapsed,
                "stdout_log": str(stdout_path),
                "stderr_log": str(stderr_path),
                "seed": spec.seed,
            },
        )
