"""Shared fixture vocabulary for the promotion-gate test files.

One owner for the synthetic measurement shapes the gate tests share: the
per-sample pattern behind every ``BenchmarkResult``, and the canonical
result builder. Domain-specific helpers (a candidate-search attempt, a
training-binding record, a metric-binding run) stay in the test files that
own them -- only genuinely identical helpers live here.
"""

from __future__ import annotations

from chowder.growth.promotion import BenchmarkResult


def _samples(mean: float, spread: float = 0.02, blocks: int = 5) -> tuple[float, ...]:
    """Per-sample scores centered on ``mean`` with honest spread."""
    pattern = (-1.5, -0.5, 0.0, 0.5, 1.5)
    return tuple(mean + spread * p for p in pattern * blocks)


def _result(benchmark: str, score: float, *, origin: str) -> BenchmarkResult:
    """A clean, measured-on-``origin`` result for one benchmark."""
    return BenchmarkResult(
        benchmark_qualified_id=benchmark,
        score=score,
        samples=_samples(score),
        contamination="CLEAN",
        measurement_origin=origin,
    )
