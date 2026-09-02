from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .backends.transformers_peft import TransformersPeftRunSpec
from .config_validation import validate_transformers_backend_config
from .evaluators.base_text import BaseTextEvalSpec
from .models import (
    Experiment,
    ExperimentResult,
    Goal,
    Hypothesis,
    MetricTarget,
    OptimizationDirection,
)


PROJECT_SCHEMA_VERSION = 1


class ProjectValidationError(ValueError):
    """Raised when a user project cannot be executed safely."""


def _mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProjectValidationError(f"{path} must be a mapping")
    return value


def _finite(value: Any, *, path: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ProjectValidationError(f"{path} must be numeric") from exc
    if not math.isfinite(number):
        raise ProjectValidationError(f"{path} must be finite")
    return number


def _resolve_path(value: str, *, base: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


@dataclass(frozen=True)
class ProjectSpec:
    name: str
    work_dir: Path
    registry_path: Path
    seed: int
    goal: Goal
    baseline_mode: str
    baseline: ExperimentResult | None
    experiment: Experiment
    config: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ProjectValidationError("project.name is required")
        if self.seed < 0:
            raise ProjectValidationError("project.seed cannot be negative")
        if self.baseline_mode not in {"auto", "fixed"}:
            raise ProjectValidationError("baseline.mode must be 'auto' or 'fixed'")

        metric_names = {target.name for target in self.goal.metrics}
        if self.baseline_mode == "fixed":
            if self.baseline is None:
                raise ProjectValidationError("fixed baseline requires baseline metrics")
            if set(self.baseline.metrics) != metric_names:
                raise ProjectValidationError(
                    "baseline metric names must exactly match goal metric names"
                )
        elif self.baseline is not None:
            raise ProjectValidationError("automatic baseline must not include fixed metrics")

        evaluation = _mapping(self.config.get("evaluation"), path="config.evaluation")
        if evaluation.get("type", "transformers-text") != "transformers-text":
            raise ProjectValidationError(
                "project runner currently supports evaluation.type='transformers-text'"
            )
        suites = evaluation.get("suites")
        if not isinstance(suites, (list, tuple)) or not suites:
            raise ProjectValidationError("config.evaluation.suites must be a non-empty list")
        suite_names: list[str] = []
        for index, row in enumerate(suites):
            suite = _mapping(row, path=f"config.evaluation.suites[{index}]")
            name = str(suite.get("name", "")).strip()
            if not name:
                raise ProjectValidationError(
                    f"config.evaluation.suites[{index}].name is required"
                )
            suite_names.append(name)
        if len(suite_names) != len(set(suite_names)):
            raise ProjectValidationError("evaluation suite names must be unique")
        if set(suite_names) != metric_names:
            raise ProjectValidationError(
                "evaluation suite names must exactly match goal metric names"
            )

        # Validate both the strict namespace and the actual executable spec at
        # project-load time. This catches semantically invalid precision,
        # quantization, LoRA, timeout, and evaluation settings before compute.
        validate_transformers_backend_config(self.config)
        try:
            TransformersPeftRunSpec.from_resolved_config(
                self.config,
                work_dir=self.work_dir,
                output_dir=self.work_dir / ".chowder" / "validation-adapter",
                seed=self.seed,
            )
            BaseTextEvalSpec.from_config(
                self.config,
                work_dir=self.work_dir,
                output_dir=self.work_dir / ".chowder" / "validation-eval",
                seed=self.seed,
            )
        except (TypeError, ValueError) as exc:
            raise ProjectValidationError(str(exc)) from exc

    def validate_files(self) -> None:
        backend = _mapping(self.config.get("backend"), path="config.backend")
        training_dataset = _resolve_path(
            str(backend.get("dataset", "")), base=self.work_dir
        )
        if not training_dataset.is_file():
            raise ProjectValidationError(
                f"training dataset not found: {training_dataset}"
            )
        evaluation = _mapping(self.config.get("evaluation"), path="config.evaluation")
        suites = evaluation.get("suites", ())
        for index, row in enumerate(suites):
            suite = _mapping(row, path=f"config.evaluation.suites[{index}]")
            dataset = _resolve_path(str(suite.get("dataset", "")), base=self.work_dir)
            if not dataset.is_file():
                raise ProjectValidationError(
                    f"evaluation dataset not found: {dataset}"
                )
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)


def _metric_from_mapping(raw: Mapping[str, Any]) -> MetricTarget:
    try:
        direction = OptimizationDirection(str(raw.get("direction", "maximize")))
    except ValueError as exc:
        raise ProjectValidationError(
            "goal metric direction must be maximize or minimize"
        ) from exc
    return MetricTarget(
        name=str(raw.get("name", "")),
        minimum=(
            None
            if raw.get("minimum") is None
            else _finite(raw["minimum"], path="metric.minimum")
        ),
        maximum=(
            None
            if raw.get("maximum") is None
            else _finite(raw["maximum"], path="metric.maximum")
        ),
        weight=_finite(raw.get("weight", 1.0), path="metric.weight"),
        regression_tolerance=_finite(
            raw.get("regression_tolerance", 0.0),
            path="metric.regression_tolerance",
        ),
        direction=direction,
    )


def _fixed_baseline(raw: Mapping[str, Any]) -> ExperimentResult:
    metrics_raw = _mapping(raw.get("metrics"), path="baseline.metrics")
    metrics = {
        str(name): _finite(value, path=f"baseline.metrics.{name}")
        for name, value in metrics_raw.items()
    }
    evidence: dict[str, Any] = {}
    protocol_sha = raw.get("evaluation_protocol_sha256")
    if protocol_sha is not None:
        protocol = str(protocol_sha)
        if len(protocol) != 64:
            raise ProjectValidationError(
                "baseline.evaluation_protocol_sha256 must be a SHA-256 digest"
            )
        evidence["evaluation_protocol_sha256"] = protocol
    return ExperimentResult(
        experiment_id=str(raw.get("experiment_id", "baseline")),
        metrics=metrics,
        gpu_hours=_finite(raw.get("gpu_hours", 0.0), path="baseline.gpu_hours"),
        artifact_ref=(
            str(raw["artifact_ref"]) if raw.get("artifact_ref") is not None else None
        ),
        evidence=evidence,
    )


def project_from_mapping(
    raw: Mapping[str, Any],
    *,
    source_dir: str | Path = ".",
) -> ProjectSpec:
    schema_version = raw.get("schema_version", PROJECT_SCHEMA_VERSION)
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise ProjectValidationError("project.schema_version must be an integer")
    if schema_version != PROJECT_SCHEMA_VERSION:
        raise ProjectValidationError(
            f"unsupported project.schema_version {schema_version}; expected {PROJECT_SCHEMA_VERSION}"
        )

    source = Path(source_dir).expanduser().resolve()
    work_dir = _resolve_path(str(raw.get("work_dir", ".")), base=source)
    registry_path = _resolve_path(
        str(raw.get("registry_path", ".chowder/runs.db")), base=work_dir
    )

    seed_raw = raw.get("seed", 1)
    if isinstance(seed_raw, bool) or not isinstance(seed_raw, int):
        raise ProjectValidationError("project.seed must be an integer")

    goal_raw = _mapping(raw.get("goal"), path="goal")
    metric_rows = goal_raw.get("metrics")
    if not isinstance(metric_rows, (list, tuple)) or not metric_rows:
        raise ProjectValidationError("goal.metrics must be a non-empty list")
    metrics = tuple(
        _metric_from_mapping(_mapping(row, path=f"goal.metrics[{index}]"))
        for index, row in enumerate(metric_rows)
    )
    budget = _finite(goal_raw.get("gpu_hour_budget"), path="goal.gpu_hour_budget")
    max_parallel = goal_raw.get("max_parallel_candidates", 1)
    if isinstance(max_parallel, bool) or not isinstance(max_parallel, int):
        raise ProjectValidationError("goal.max_parallel_candidates must be an integer")
    goal = Goal(
        metrics=metrics,
        gpu_hour_budget=budget,
        max_parallel_candidates=max_parallel,
        minimum_promotion_gain=_finite(
            goal_raw.get("minimum_promotion_gain", 0.0),
            path="goal.minimum_promotion_gain",
        ),
        require_protocol_match=bool(goal_raw.get("require_protocol_match", False)),
    )

    baseline_raw = _mapping(raw.get("baseline", {"mode": "auto"}), path="baseline")
    baseline_mode = str(baseline_raw.get("mode", "fixed")).strip().lower()
    if baseline_mode not in {"auto", "fixed"}:
        raise ProjectValidationError("baseline.mode must be 'auto' or 'fixed'")
    baseline = None if baseline_mode == "auto" else _fixed_baseline(baseline_raw)

    experiment_raw = _mapping(raw.get("experiment"), path="experiment")
    hypothesis_raw = _mapping(
        experiment_raw.get("hypothesis", {}), path="experiment.hypothesis"
    )
    hypothesis = Hypothesis(
        observation=str(
            hypothesis_raw.get("observation", "Initial post-training run")
        ),
        suspected_cause=str(
            hypothesis_raw.get(
                "suspected_cause",
                "Base model has not been adapted to the target data",
            )
        ),
        intervention=str(
            hypothesis_raw.get("intervention", "Supervised LoRA fine-tuning")
        ),
        expected_deltas=dict(hypothesis_raw.get("expected_deltas", {})),
    )
    experiment = Experiment(
        experiment_id=str(experiment_raw.get("experiment_id", "initial-sft")),
        parent_id=None,
        hypothesis=hypothesis,
        config_patch=dict(experiment_raw.get("config_patch", {})),
        estimated_gpu_hours=_finite(
            experiment_raw.get("estimated_gpu_hours", 0.25),
            path="experiment.estimated_gpu_hours",
        ),
        tags=tuple(str(tag) for tag in experiment_raw.get("tags", ())),
    )

    config_raw = _mapping(raw.get("config"), path="config")
    config = dict(config_raw)
    config.setdefault("seed", seed_raw)

    return ProjectSpec(
        name=str(raw.get("name", "Chowder Project")),
        work_dir=work_dir,
        registry_path=registry_path,
        seed=seed_raw,
        goal=goal,
        baseline_mode=baseline_mode,
        baseline=baseline,
        experiment=experiment,
        config=config,
    )


def load_project(path: str | Path, *, validate_files: bool = True) -> ProjectSpec:
    source_path = Path(path).expanduser().resolve()
    raw = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ProjectValidationError("project file root must be a JSON object")
    project = project_from_mapping(raw, source_dir=source_path.parent)
    if validate_files:
        project.validate_files()
    return project


def write_project(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    normalized = dict(payload)
    normalized["schema_version"] = PROJECT_SCHEMA_VERSION
    project_from_mapping(normalized, source_dir=target.parent)
    target.write_bytes(
        (
            json.dumps(normalized, sort_keys=True, indent=2, allow_nan=False) + "\n"
        ).encode("utf-8")
    )
    return target
