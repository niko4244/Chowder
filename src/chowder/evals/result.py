"""Normalized evaluation results for the growth system.

Every adapter (Inspect, lm-eval-harness, native agent suites, Chowder custom)
normalizes into ``EvalResult`` so the scoreboard, capability profile, and
promotion rules see one schema while raw benchmark artifacts stay on disk
untouched.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

# Score support levels, from the benchmark registry's availability status:
# some benchmarks cannot run on a text-only model, and scoring them as zero
# would poison every aggregate.
SUPPORTED = "SUPPORTED"
NOT_APPLICABLE_MODALITY = "NOT_APPLICABLE_MODALITY"
UNSUPPORTED_HARNESS = "UNSUPPORTED_HARNESS"

# Per-benchmark raw-vs-harness distinction: a raw model score and an
# agent-harness score are different measurements and are never merged.
RAW_MODEL = "raw_model"
AGENT_HARNESS = "agent_harness"


@dataclass(frozen=True)
class BenchmarkRun:
    """One executed (or honestly unavailable) benchmark measurement."""

    benchmark_qualified_id: str  # benchmark_id@version -- never vague "latest"
    adapter: str  # inspect | lm_eval | native | chowder_custom
    generation_version: str
    score: float | None  # normalized 0..1; None when not applicable
    support: str = SUPPORTED
    measurement_kind: str = RAW_MODEL
    n_samples: int = 0
    per_sample_scores: tuple[float, ...] = ()
    metric: str = "accuracy"
    tool_setting: str = "none"
    reasoning_setting: str = "direct"
    raw_artifact_ref: str = ""  # on-disk raw output, preserved as evidence
    notes: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_qualified_id": self.benchmark_qualified_id,
            "adapter": self.adapter,
            "generation_version": self.generation_version,
            "score": self.score,
            "support": self.support,
            "measurement_kind": self.measurement_kind,
            "n_samples": self.n_samples,
            "per_sample_scores": list(self.per_sample_scores),
            "metric": self.metric,
            "tool_setting": self.tool_setting,
            "reasoning_setting": self.reasoning_setting,
            "raw_artifact_ref": self.raw_artifact_ref,
            "notes": self.notes,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class EvalReport:
    """All runs from one evaluation pass over one generation."""

    generation_version: str
    runs: tuple[BenchmarkRun, ...]
    hardware: Mapping[str, Any] = field(default_factory=dict)
    date: str = ""

    def supported_runs(self) -> tuple[BenchmarkRun, ...]:
        return tuple(run for run in self.runs if run.support == SUPPORTED)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation_version": self.generation_version,
            "runs": [run.to_dict() for run in self.runs],
            "hardware": dict(self.hardware),
            "date": self.date,
        }

    def save(self, path: Path | str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path | str) -> "EvalReport":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            generation_version=data["generation_version"],
            runs=tuple(
                BenchmarkRun(
                    benchmark_qualified_id=run["benchmark_qualified_id"],
                    adapter=run["adapter"],
                    generation_version=run["generation_version"],
                    score=run.get("score"),
                    support=run.get("support", SUPPORTED),
                    measurement_kind=run.get("measurement_kind", RAW_MODEL),
                    n_samples=run.get("n_samples", 0),
                    per_sample_scores=tuple(run.get("per_sample_scores", ())),
                    metric=run.get("metric", "accuracy"),
                    tool_setting=run.get("tool_setting", "none"),
                    reasoning_setting=run.get("reasoning_setting", "direct"),
                    raw_artifact_ref=run.get("raw_artifact_ref", ""),
                    notes=run.get("notes", ""),
                    metadata=run.get("metadata", {}),
                )
                for run in data["runs"]
            ),
            hardware=data.get("hardware", {}),
            date=data.get("date", ""),
        )


class EvalAdapter:
    """Common adapter interface. Adapters normalize, never invent.

    Implementations translate a benchmark's native output into
    ``BenchmarkRun``; an adapter that cannot execute a benchmark returns a
    run with ``support`` set honestly rather than a fabricated score.
    """

    name: str = "abstract"

    def run(self, benchmark_qualified_id: str, generation_version: str) -> BenchmarkRun:
        raise NotImplementedError
