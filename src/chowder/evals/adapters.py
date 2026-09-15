"""Evaluation adapters: one common interface over external harnesses.

Chowder does not rewrite every benchmark. The adapters translate external
harness outputs into the normalized ``BenchmarkRun`` schema:

- ``InspectAdapter``: Inspect AI / Inspect Evals.
- ``LMEvalAdapter``: EleutherAI lm-evaluation-harness.
- ``NativeAgentBenchmarkAdapter``: benchmark-native agent suites.
- ``ChowderCustomEvalAdapter``: Chowder-native diagnostics.

Adapters require the external harness package to be importable and fail
honestly (``UNSUPPORTED_HARNESS``) when it is absent -- they never
substitute a fake score. Raw harness artifacts are preserved on disk and
referenced from the run; the normalized result is a view, not a replacement.
"""

from __future__ import annotations

import importlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from .result import (
    AGENT_HARNESS,
    RAW_MODEL,
    SUPPORTED,
    UNSUPPORTED_HARNESS,
    BenchmarkRun,
)


class AdapterUnavailable(RuntimeError):
    """The external harness package or CLI is not importable on this host."""


def _require_module(module_name: str) -> None:
    try:
        importlib.import_module(module_name)
    except ImportError as error:
        raise AdapterUnavailable(
            f"harness module '{module_name}' is not importable; refusing to "
            "fabricate a score"
        ) from error


class InspectAdapter:
    """Inspect AI (``inspect_ai``) -- log-dir based, one eval per task."""

    name = "inspect"

    def __init__(self, *, logs_dir: Path | str | None = None) -> None:
        self._logs_dir = Path(logs_dir) if logs_dir else None

    def run(
        self,
        benchmark_qualified_id: str,
        generation_version: str,
        *,
        task: str,
        model: str,
        metric: str = "accuracy",
    ) -> BenchmarkRun:
        _require_module("inspect_ai")
        from inspect_ai import eval as inspect_eval  # noqa: PLC0415

        logs = inspect_eval(task, model=model)
        scores: list[float] = []
        for log in logs:
            results = getattr(log, "results", None)
            if results is None:
                continue
            for score in results.scores:
                if score.name == metric and score.value is not None:
                    value = score.value
                    scores.append(float(value if isinstance(value, (int, float)) else value.get("value", 0.0)))
        score_value = (sum(scores) / len(scores)) if scores else None
        artifact = str(self._logs_dir) if self._logs_dir else ""
        return BenchmarkRun(
            benchmark_qualified_id=benchmark_qualified_id,
            adapter=self.name,
            generation_version=generation_version,
            score=score_value,
            support=SUPPORTED,
            measurement_kind=RAW_MODEL,
            n_samples=len(scores),
            per_sample_scores=tuple(scores),
            metric=metric,
            raw_artifact_ref=artifact,
        )


class LMEvalAdapter:
    """EleutherAI lm-evaluation-harness -- task-name based."""

    name = "lm_eval"

    def __init__(self, *, output_dir: Path | str | None = None) -> None:
        self._output_dir = Path(output_dir) if output_dir else None

    def run(
        self,
        benchmark_qualified_id: str,
        generation_version: str,
        *,
        task: str,
        model: str,
        model_args: str = "",
        batch_size: int = 1,
    ) -> BenchmarkRun:
        _require_module("lm_eval")
        import lm_eval  # noqa: PLC0415

        results = lm_eval.simple_validate(
            model=model,
            model_args=model_args,
            tasks=[task],
            batch_size=batch_size,
        )
        task_results = results.get("results", {}).get(task, {})
        # lm_eval reports metrics like "acc,none"; take the first metric.
        metric_value: float | None = None
        metric_name = ""
        for key, value in task_results.items():
            if key in {"alias", "alias "}:
                continue
            if isinstance(value, (int, float)):
                metric_value = float(value)
                metric_name = key.split(",")[0]
                break
        artifact = ""
        if self._output_dir:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = self._output_dir / f"{task}.json"
            artifact_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
            artifact = str(artifact_path)
        return BenchmarkRun(
            benchmark_qualified_id=benchmark_qualified_id,
            adapter=self.name,
            generation_version=generation_version,
            score=metric_value,
            support=SUPPORTED,
            measurement_kind=RAW_MODEL,
            n_samples=1,
            metric=metric_name or "acc",
            raw_artifact_ref=artifact,
        )


class NativeAgentBenchmarkAdapter:
    """Benchmark-native agent suites run in their own processes/environments.

    The adapter shells out to a provided command and reads a JSON results
    file -- it does not pretend to understand each suite's internals.
    ``measurement_kind`` is AGENT_HARNESS: these scores are never mixed with
    raw-model numbers without labeling.
    """

    name = "native_agent"

    def __init__(self, *, results_dir: Path | str | None = None) -> None:
        self._results_dir = Path(results_dir) if results_dir else None

    def run(
        self,
        benchmark_qualified_id: str,
        generation_version: str,
        *,
        command: Sequence[str],
        score_key: str = "score",
        timeout_seconds: int = 3600,
    ) -> BenchmarkRun:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        if completed.returncode != 0:
            raise AdapterUnavailable(
                f"native agent suite failed (exit {completed.returncode}): "
                f"{completed.stderr[-500:]}"
            )
        payload: Mapping[str, Any] = json.loads(completed.stdout)
        score = payload.get(score_key)
        artifact = ""
        if self._results_dir:
            self._results_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = self._results_dir / f"{benchmark_qualified_id.replace('@', '_')}.json"
            artifact_path.write_text(completed.stdout, encoding="utf-8")
            artifact = str(artifact_path)
        return BenchmarkRun(
            benchmark_qualified_id=benchmark_qualified_id,
            adapter=self.name,
            generation_version=generation_version,
            score=float(score) if score is not None else None,
            support=SUPPORTED,
            measurement_kind=AGENT_HARNESS,
            n_samples=int(payload.get("n_samples", 0)),
            metric=payload.get("metric", "resolve_rate"),
            raw_artifact_ref=artifact,
            metadata={"exit_code": completed.returncode},
        )


class ChowderCustomEvalAdapter:
    """Chowder-native diagnostics (function-calling stress, contamination
    canaries, smoke tiers). Executed in-process against a callable scorer."""

    name = "chowder_custom"

    def run(
        self,
        benchmark_qualified_id: str,
        generation_version: str,
        *,
        scorer: Any,
        n_samples: int = 0,
        metric: str = "accuracy",
    ) -> BenchmarkRun:
        outcome = scorer()
        if isinstance(outcome, tuple):
            score, per_sample = float(outcome[0]), tuple(float(s) for s in outcome[1])
        else:
            score, per_sample = float(outcome), ()
        return BenchmarkRun(
            benchmark_qualified_id=benchmark_qualified_id,
            adapter=self.name,
            generation_version=generation_version,
            score=score,
            support=SUPPORTED,
            measurement_kind=RAW_MODEL,
            n_samples=n_samples or len(per_sample),
            per_sample_scores=per_sample,
            metric=metric,
        )


def unavailable_run(
    benchmark_qualified_id: str,
    generation_version: str,
    adapter: str,
    *,
    reason: str,
) -> BenchmarkRun:
    """An honest non-measurement: no score, explicit support level."""
    return BenchmarkRun(
        benchmark_qualified_id=benchmark_qualified_id,
        adapter=adapter,
        generation_version=generation_version,
        score=None,
        support=UNSUPPORTED_HARNESS,
        notes=reason,
    )


ADAPTERS = {
    InspectAdapter.name: InspectAdapter,
    LMEvalAdapter.name: LMEvalAdapter,
    NativeAgentBenchmarkAdapter.name: NativeAgentBenchmarkAdapter,
    ChowderCustomEvalAdapter.name: ChowderCustomEvalAdapter,
}

__all__ = [
    "ADAPTERS",
    "AdapterUnavailable",
    "ChowderCustomEvalAdapter",
    "InspectAdapter",
    "LMEvalAdapter",
    "NativeAgentBenchmarkAdapter",
    "unavailable_run",
]
