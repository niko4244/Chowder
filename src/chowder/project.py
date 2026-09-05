from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .backend_selection import (
    BackendSelectionError,
    UNSLOTH_ENGINE,
    normalize_training_config_for_executor,
    resolve_training_engine,
)
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
from .recursive_repair import RecursiveRepairPolicy
from .repair_candidates import RepairVariant


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
class RepairSpec:
    """Config-driven autonomous repair: how to source repair examples and what
    hyperparameter variants to try when the initial candidate is rejected.

    Repair variants may only patch training config, never LoRA topology --
    ``run_bounded_autonomous_repair`` hard-rejects any variant with a
    ``lora_patch`` (continuation repair cannot change adapter shape), so the
    schema does not expose that field at all.
    """

    corpus_files: tuple[str, ...]
    variants: tuple[RepairVariant, ...]
    policy: RecursiveRepairPolicy = field(default_factory=RecursiveRepairPolicy)
    provider_max_examples: int = 32
    provider_min_examples: int = 1
    provider_examples_per_failure: int = 2

    def __post_init__(self) -> None:
        if not self.corpus_files:
            raise ProjectValidationError("repair.corpus_files must be a non-empty list")
        if not self.variants:
            raise ProjectValidationError("repair.variants must be a non-empty list")


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
    repair: RepairSpec | None = None

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

        try:
            training_engine = resolve_training_engine(self.config)
        except BackendSelectionError as exc:
            raise ProjectValidationError(str(exc)) from exc
        if training_engine == UNSLOTH_ENGINE:
            raise ProjectValidationError(
                "backend.engine='unsloth' is recognized but its isolated executor "
                "is not available yet"
            )
        training_config = normalize_training_config_for_executor(self.config)

        # Validate both the strict namespace and the actual executable spec at
        # project-load time. This catches semantically invalid precision,
        # quantization, LoRA, timeout, and evaluation settings before compute.
        validate_transformers_backend_config(training_config)
        try:
            TransformersPeftRunSpec.from_resolved_config(
                training_config,
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
        if self.repair is not None:
            for index, corpus in enumerate(self.repair.corpus_files):
                if not Path(corpus).is_file():
                    raise ProjectValidationError(
                        f"repair.corpus_files[{index}] not found: {corpus}"
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


def _repair_variant_from_mapping(raw: Mapping[str, Any], *, path: str) -> RepairVariant:
    if "lora_patch" in raw:
        raise ProjectValidationError(
            f"{path}.lora_patch is not supported; repair variants may only patch training config"
        )
    name = str(raw.get("name", "")).strip()
    if not name:
        raise ProjectValidationError(f"{path}.name is required")
    training_patch = _mapping(raw.get("training_patch", {}), path=f"{path}.training_patch")
    expected_deltas_raw = _mapping(raw.get("expected_deltas", {}), path=f"{path}.expected_deltas")
    expected_deltas = {
        str(name): _finite(value, path=f"{path}.expected_deltas.{name}")
        for name, value in expected_deltas_raw.items()
    }
    try:
        return RepairVariant(
            name=name,
            estimated_gpu_hours=_finite(
                raw.get("estimated_gpu_hours"), path=f"{path}.estimated_gpu_hours"
            ),
            training_patch=dict(training_patch),
            expected_deltas=expected_deltas,
        )
    except ValueError as exc:
        raise ProjectValidationError(str(exc)) from exc


def _repair_policy_from_mapping(raw: Mapping[str, Any], *, path: str) -> RecursiveRepairPolicy:
    max_depth = raw.get("max_depth", 3)
    if isinstance(max_depth, bool) or not isinstance(max_depth, int):
        raise ProjectValidationError(f"{path}.max_depth must be an integer")
    max_occurrences = raw.get("max_failure_signature_occurrences", 1)
    if isinstance(max_occurrences, bool) or not isinstance(max_occurrences, int):
        raise ProjectValidationError(
            f"{path}.max_failure_signature_occurrences must be an integer"
        )
    replay_ratio_raw = raw.get("replay_ratio", 1.0)
    replay_ratio = (
        None
        if replay_ratio_raw is None
        else _finite(replay_ratio_raw, path=f"{path}.replay_ratio")
    )
    try:
        return RecursiveRepairPolicy(
            max_depth=max_depth,
            min_score_improvement=_finite(
                raw.get("min_score_improvement", 1e-4), path=f"{path}.min_score_improvement"
            ),
            max_failure_signature_occurrences=max_occurrences,
            replay_ratio=replay_ratio,
        )
    except ValueError as exc:
        raise ProjectValidationError(str(exc)) from exc


def _repair_from_mapping(raw: Any, *, base: Path, path: str = "repair") -> RepairSpec | None:
    if raw is None:
        return None
    section = _mapping(raw, path=path)

    corpus_raw = section.get("corpus_files")
    if not isinstance(corpus_raw, (list, tuple)) or not corpus_raw:
        raise ProjectValidationError(f"{path}.corpus_files must be a non-empty list")
    corpus_files = tuple(
        str(_resolve_path(str(item), base=base)) for item in corpus_raw
    )

    variant_rows = section.get("variants")
    if not isinstance(variant_rows, (list, tuple)) or not variant_rows:
        raise ProjectValidationError(f"{path}.variants must be a non-empty list")
    variants = tuple(
        _repair_variant_from_mapping(
            _mapping(row, path=f"{path}.variants[{index}]"),
            path=f"{path}.variants[{index}]",
        )
        for index, row in enumerate(variant_rows)
    )

    def _int_field(name: str, default: int) -> int:
        value = section.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ProjectValidationError(f"{path}.{name} must be an integer")
        return value

    policy = _repair_policy_from_mapping(
        _mapping(section.get("policy", {}), path=f"{path}.policy"), path=f"{path}.policy"
    )

    return RepairSpec(
        corpus_files=corpus_files,
        variants=variants,
        policy=policy,
        provider_max_examples=_int_field("max_examples", 32),
        provider_min_examples=_int_field("min_examples", 1),
        provider_examples_per_failure=_int_field("examples_per_failure", 2),
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

    repair = _repair_from_mapping(raw.get("repair"), base=work_dir)

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
        repair=repair,
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
