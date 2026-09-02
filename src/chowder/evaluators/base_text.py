from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from ..executors import EvaluationOutcome, ExecutionContext
from ..protocol import protocol_fingerprint
from ..provenance import sha256_file
from .transformers_text import EvalSuiteSpec


@dataclass(frozen=True)
class BaseTextEvalSpec:
    base_model: str
    output_dir: str
    suites: tuple[EvalSuiteSpec, ...]
    revision: str | None = None
    precision: str = "auto"
    quantization: str = "none"
    device: str = "auto"
    seed: int = 1
    timeout_seconds: float | None = None
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        if not self.base_model.strip():
            raise ValueError("baseline evaluation base_model is required")
        if not self.suites:
            raise ValueError("baseline evaluation requires at least one suite")
        if len({suite.name for suite in self.suites}) != len(self.suites):
            raise ValueError("baseline evaluation suite names must be unique")
        if self.precision not in {"auto", "bf16", "fp16", "fp32"}:
            raise ValueError(f"unsupported baseline precision: {self.precision}")
        if self.quantization not in {"none", "4bit"}:
            raise ValueError(f"unsupported baseline quantization: {self.quantization}")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("baseline timeout_seconds must be positive")
        if self.trust_remote_code:
            raise ValueError("trust_remote_code is disabled for baseline evaluation")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["suites"] = [asdict(suite) for suite in self.suites]
        return payload

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        work_dir: str | Path,
        output_dir: str | Path,
        seed: int,
    ) -> "BaseTextEvalSpec":
        backend = config.get("backend")
        evaluation = config.get("evaluation")
        if not isinstance(backend, Mapping) or not isinstance(evaluation, Mapping):
            raise ValueError("resolved config must contain backend and evaluation mappings")
        if evaluation.get("type", "transformers-text") != "transformers-text":
            raise ValueError("baseline evaluator supports evaluation.type='transformers-text'")

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
                    max_new_tokens=int(
                        raw.get("max_new_tokens", evaluation.get("max_new_tokens", 64))
                    ),
                    use_chat_template=bool(
                        raw.get(
                            "use_chat_template",
                            evaluation.get("use_chat_template", False),
                        )
                    ),
                )
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

        return cls(
            base_model=str(backend.get("base_model", "")),
            output_dir=str(Path(output_dir).resolve()),
            suites=tuple(suites),
            revision=(
                str(backend["revision"])
                if backend.get("revision") is not None
                else None
            ),
            precision=precision,
            quantization=quantization,
            device=str(evaluation.get("device", "auto")),
            seed=int(config.get("seed", seed)),
            timeout_seconds=(
                float(runtime["timeout_seconds"])
                if runtime.get("timeout_seconds") is not None
                else None
            ),
            trust_remote_code=bool(evaluation.get("trust_remote_code", False)),
        )


class BaseModelTextEvaluator:
    """Evaluate the untouched base model using the same protocol shape as adapters."""

    name = "transformers-text"

    @staticmethod
    def _worker_command(spec_path: Path, result_path: Path) -> list[str]:
        return [
            sys.executable,
            "-m",
            "chowder.evaluators.base_text_worker",
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

    def evaluate(
        self,
        *,
        config: Mapping[str, Any],
        context: ExecutionContext,
        baseline_id: str = "baseline",
    ) -> EvaluationOutcome:
        eval_id = f"baseline-{uuid4().hex[:12]}"
        eval_dir = (Path(context.work_dir) / ".chowder" / "evals" / eval_id).resolve()
        eval_dir.mkdir(parents=True, exist_ok=False)
        spec = BaseTextEvalSpec.from_config(
            config,
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
        spec_path.write_bytes((spec.canonical_json() + "\n").encode("utf-8"))

        started = time.perf_counter()
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr:
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
                raise TimeoutError(f"baseline evaluation {eval_id} exceeded timeout") from exc

        elapsed = time.perf_counter() - started
        if process.returncode != 0:
            raise RuntimeError(
                f"base-model evaluator failed with exit code {process.returncode}:\n"
                f"{self._tail(stderr_path)}"
            )
        if not result_path.is_file():
            raise RuntimeError("base-model evaluator exited without a result manifest")

        payload = json.loads(result_path.read_text(encoding="utf-8"))
        metrics = payload.get("metrics")
        suite_evidence = payload.get("suites")
        versions = payload.get("versions", {})
        runtime = payload.get("runtime", {})
        provenance = payload.get("model_provenance", {})
        if not all(
            isinstance(value, Mapping)
            for value in (metrics, suite_evidence, versions, runtime, provenance)
        ):
            raise RuntimeError("baseline evaluation result has invalid evidence payload")

        gpu_count = int(runtime.get("gpu_count", 0))
        if gpu_count < 0:
            raise RuntimeError("baseline evaluation reported negative gpu_count")
        expected_names = {suite.name for suite in spec.suites}
        if set(metrics) != expected_names or set(suite_evidence) != expected_names:
            raise RuntimeError("baseline evaluation metrics do not match configured suites")

        fingerprint_hashes: dict[str, str] = {}
        for suite_name, suite_payload in suite_evidence.items():
            if not isinstance(suite_payload, Mapping):
                raise RuntimeError(f"suite evidence for {suite_name!r} is invalid")
            ref = suite_payload.get("holdout_fingerprints_file")
            declared = suite_payload.get("holdout_fingerprints_sha256")
            if not isinstance(ref, str) or not isinstance(declared, str):
                raise RuntimeError("baseline holdout fingerprint evidence is incomplete")
            path = Path(ref).resolve()
            if not path.is_relative_to(eval_dir) or not path.is_file():
                raise RuntimeError("baseline holdout fingerprint path is invalid")
            actual = sha256_file(path)
            if actual != declared:
                raise RuntimeError("baseline holdout fingerprint digest mismatch")
            fingerprint_hashes[str(suite_name)] = actual

        dataset_hashes = {suite.name: sha256_file(suite.dataset) for suite in spec.suites}
        resolved_commit = provenance.get("resolved_model_commit")
        revision = str(resolved_commit) if resolved_commit else spec.revision
        protocol = {
            "evaluator": self.name,
            "base_model": spec.base_model,
            "revision": revision,
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
        source_ref = f"base-model:{spec.base_model}@{revision or 'unresolved'}"
        return EvaluationOutcome(
            run_id=eval_id,
            experiment_id=baseline_id,
            source_artifact_ref=source_ref,
            metrics={name: float(value) for name, value in metrics.items()},
            gpu_hours=(elapsed / 3600.0) * gpu_count,
            evidence={
                "evaluator": self.name,
                "baseline": True,
                "evaluation_spec_sha256": spec.digest(),
                "evaluation_dataset_sha256": dataset_hashes,
                "evaluation_result_sha256": hashlib.sha256(
                    result_path.read_bytes()
                ).hexdigest(),
                "protocol": protocol,
                "protocol_sha256": protocol_sha,
                "holdout_fingerprint_sha256": fingerprint_hashes,
                "suite_evidence": dict(suite_evidence),
                "versions": dict(versions),
                "runtime": dict(runtime),
                "model_provenance": dict(provenance),
                "wall_time_seconds": elapsed,
                "stdout_log": str(stdout_path),
                "stderr_log": str(stderr_path),
                "seed": spec.seed,
            },
        )
