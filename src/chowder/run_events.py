from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Union


@dataclass(frozen=True)
class RunEvent:
    """A generic lifecycle/stage transition -- project loaded, hardware
    detected, baseline established, a candidate starting -- that doesn't
    carry the more specific structure of one of the event types below.
    Replaces the old stage/message-only ProjectRunEvent: every previous
    caller matched on `.stage`, which this keeps, so existing string-based
    handling (a CLI printing `[{stage}] {message}`, for instance) still
    works unchanged.
    """

    stage: str
    message: str
    experiment_id: str | None = None


@dataclass(frozen=True)
class TrainingProgressEvent:
    """Live progress from an in-flight training run. Emitted as the worker
    reports it (via its logging_steps cadence) -- fields the worker hasn't
    reported yet at a given point (e.g. loss before the first log step) are
    None rather than a fabricated placeholder.
    """

    experiment_id: str
    step: int
    max_steps: int | None
    epoch: float | None
    loss: float | None
    learning_rate: float | None
    wall_seconds: float


@dataclass(frozen=True)
class EvaluationProgressEvent:
    """Live progress from an in-flight evaluation run."""

    experiment_id: str
    suite: str
    rows_scored: int
    rows_total: int | None
    wall_seconds: float


@dataclass(frozen=True)
class CheckpointEvent:
    """A Trainer checkpoint was written (or would be resumed from)."""

    experiment_id: str
    checkpoint_dir: str
    step: int | None
    resumed: bool = False


@dataclass(frozen=True)
class RepairEvent:
    """One step of the autonomous repair loop: a hop starting, or the loop
    concluding. `stop_reason`/`stop_detail` are set only for the concluding
    event; `depth`/`failure_signature` describe the hop being attempted for
    a starting event.
    """

    target_experiment_id: str
    depth: int
    failure_signature: str | None = None
    stop_reason: str | None = None
    stop_detail: str | None = None


@dataclass(frozen=True)
class FailureEvent:
    """Failures harvested from a candidate's evaluation and clustered into
    repair plans."""

    experiment_id: str
    failure_count: int
    repair_plan_count: int


@dataclass(frozen=True)
class PromotionEvent:
    """A candidate cleared the regression gate and became the new
    baseline."""

    experiment_id: str
    metrics: Mapping[str, float] = field(default_factory=dict)


RunEventPayload = Union[
    RunEvent,
    TrainingProgressEvent,
    EvaluationProgressEvent,
    CheckpointEvent,
    RepairEvent,
    FailureEvent,
    PromotionEvent,
]


def event_type_name(event: RunEventPayload) -> str:
    return type(event).__name__


def event_experiment_id(event: RunEventPayload) -> str | None:
    return getattr(event, "experiment_id", None) or getattr(
        event, "target_experiment_id", None
    )


def event_payload(event: RunEventPayload) -> dict[str, Any]:
    return asdict(event)


def format_event(event: RunEventPayload) -> str:
    """One human-readable line for any event in the union -- the single
    formatter both the CLI and the TUI use, so the two surfaces reading the
    same event stream actually render it the same way rather than each
    hand-rolling (and inevitably drifting from) their own per-type text.
    """
    if isinstance(event, RunEvent):
        return f"[{event.stage}] {event.message}"
    if isinstance(event, TrainingProgressEvent):
        parts = [f"step {event.step}" + (f"/{event.max_steps}" if event.max_steps else "")]
        if event.epoch is not None:
            parts.append(f"epoch {event.epoch:.2f}")
        if event.loss is not None:
            parts.append(f"loss {event.loss:.4f}")
        if event.learning_rate is not None:
            parts.append(f"lr {event.learning_rate:.2e}")
        return f"[training:{event.experiment_id}] " + ", ".join(parts)
    if isinstance(event, EvaluationProgressEvent):
        total = f"/{event.rows_total}" if event.rows_total is not None else ""
        return f"[evaluate:{event.experiment_id}] {event.suite}: {event.rows_scored}{total} rows"
    if isinstance(event, CheckpointEvent):
        verb = "resumed from" if event.resumed else "checkpoint written to"
        return f"[checkpoint:{event.experiment_id}] {verb} {event.checkpoint_dir}"
    if isinstance(event, RepairEvent):
        if event.stop_reason is not None:
            return (
                f"[repair:{event.target_experiment_id}] stopped: {event.stop_reason} "
                f"at depth {event.depth} ({event.stop_detail})"
            )
        return f"[repair:{event.target_experiment_id}] hop {event.depth} starting"
    if isinstance(event, FailureEvent):
        return (
            f"[failures:{event.experiment_id}] {event.failure_count} failure(s), "
            f"{event.repair_plan_count} repair plan(s)"
        )
    if isinstance(event, PromotionEvent):
        metrics = ", ".join(f"{name}={value:.4f}" for name, value in sorted(event.metrics.items()))
        return f"[promoted:{event.experiment_id}] {metrics}"
    return f"[{event_type_name(event)}] {event}"
