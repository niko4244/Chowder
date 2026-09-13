"""Identify which learning-rate schedule a run *actually* followed.

Chowder validates `lr_scheduler_type` and both workers now pass it to the trainer,
and `tests/test_lr_scheduler_honoured.py` guards that by source. A source check
cannot catch everything it needs to:

* the key reaching `TrainingArguments` does not prove the optimiser followed that
  curve -- the Unsloth worker accepted the key for months and dropped it, and
  nothing in the config, the telemetry or the result said so;
* a transformers upgrade could change its cosine definition, its `num_cycles`
  default, or when it logs the rate, and every source assertion would still pass.

This closes that gap from the other end: given the learning rates a run actually
logged, say which schedule they came from. Training workers publish
`learning_rate` every `logging_steps`, so the evidence is already there.

Honesty properties that matter more than the classification
-------------------------------------------------------------
1. **It refuses when it cannot tell.** Near step 0 every schedule sits at the peak
   rate, so a short noisy trajectory carries no information. The verdict is then
   ``INCONCLUSIVE`` rather than a confident guess -- if a run dies at step 10 the
   answer must be "unknown", not "cosine".
2. **It refuses when the truth is not among the candidates.** Only the schedules
   passed in are modelled; anything else has to fail the absolute-fit gate rather
   than be reported as the nearest candidate.

The logging offset is real, not a fudge
----------------------------------------
HF's Trainer logs `get_last_lr()`, the rate set *before* this step's
`scheduler.step()`, so the rate logged at step *s* tracks lambda(*s*-1). Offsets
-1, 0 and +1 are all scored and the best kept. This cannot turn one schedule into
another: an offset moves a curve by well under 1% of peak while linear and cosine
differ by up to 10.4% of peak (at progress 0.25 and 0.75).

Verified against a real 500-step run of the pruned 9B, which fits HF cosine
*exactly* at offset -1 (`tests/test_schedule_audit.py`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

#: The winner must actually fit, not merely fit best: guards against naming the
#: nearest candidate when the real schedule was not offered at all.
MAX_RESIDUAL_FRACTION = 0.01
#: And it must beat the runner-up clearly, or the trajectory is uninformative.
MIN_SEPARATION = 3.0

#: Offsets searched, covering HF's log-before-step convention either way.
_OFFSETS = (-1, 0, 1)

INCONCLUSIVE = "INCONCLUSIVE"


def _warmup_factor(step: int, warmup_steps: int) -> float:
    return step / max(1, warmup_steps)


def _linear(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return _warmup_factor(step, warmup)
    return max(0.0, (total - step) / max(1, total - warmup))


def _cosine(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return _warmup_factor(step, warmup)
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def _constant(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return _warmup_factor(step, warmup)
    return 1.0


#: Only what Chowder's recipes actually use. A schedule absent from here is not
#: misreported as its nearest neighbour -- it fails MAX_RESIDUAL_FRACTION.
SCHEDULES: Mapping[str, Callable[[int, int, int], float]] = {
    "linear": _linear,
    "cosine": _cosine,
    "constant": _constant,
}


@dataclass(frozen=True)
class ScheduleVerdict:
    """What the logged rates say. `schedule` is INCONCLUSIVE when unknowable."""

    schedule: str
    best_fit: str
    residual_fraction_of_peak: float
    separation: float
    offset: int
    samples: int
    candidates: tuple[str, ...]

    @property
    def conclusive(self) -> bool:
        return self.schedule != INCONCLUSIVE

    def matches(self, expected: str) -> bool:
        """True only when the evidence positively supports `expected`."""
        return self.conclusive and self.schedule == expected

    def as_dict(self) -> dict[str, object]:
        return {
            "schedule": self.schedule,
            "best_fit": self.best_fit,
            "residual_fraction_of_peak": self.residual_fraction_of_peak,
            "separation": self.separation,
            "offset": self.offset,
            "samples": self.samples,
            "candidates": list(self.candidates),
        }


def _rmse(
    observations: Sequence[tuple[int, float]],
    lam: Callable[[int, int, int], float],
    *,
    peak_lr: float,
    total_steps: int,
    warmup_steps: int,
    offset: int,
) -> float:
    total = 0.0
    for step, observed in observations:
        predicted = peak_lr * lam(step + offset, total_steps, warmup_steps)
        total += (observed - predicted) ** 2
    return math.sqrt(total / len(observations))


def identify_schedule(
    observations: Iterable[tuple[int, float]],
    *,
    peak_lr: float,
    total_steps: int,
    warmup_steps: int = 0,
    candidates: Sequence[str] | None = None,
) -> ScheduleVerdict:
    """Name the schedule behind `observations` ((step, learning_rate) pairs).

    Raises ValueError on unusable input rather than returning a verdict that looks
    like a measurement.
    """
    pairs = [(int(step), float(lr)) for step, lr in observations]
    if not pairs:
        raise ValueError("no learning-rate observations to identify a schedule from")
    if peak_lr <= 0 or not math.isfinite(peak_lr):
        raise ValueError(f"peak_lr must be positive and finite, got {peak_lr!r}")
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps!r}")
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps cannot be negative, got {warmup_steps!r}")

    names = tuple(candidates) if candidates is not None else tuple(SCHEDULES)
    unknown = [name for name in names if name not in SCHEDULES]
    if unknown:
        raise ValueError(f"unmodelled schedule(s): {sorted(unknown)}")
    if len(names) < 2:
        raise ValueError("at least two candidates are needed to discriminate")

    scored: list[tuple[float, int, str]] = []
    for name in names:
        best = min(
            (
                (
                    _rmse(
                        pairs,
                        SCHEDULES[name],
                        peak_lr=peak_lr,
                        total_steps=total_steps,
                        warmup_steps=warmup_steps,
                        offset=offset,
                    ),
                    offset,
                )
                for offset in _OFFSETS
            ),
            key=lambda pair: pair[0],
        )
        scored.append((best[0], best[1], name))
    scored.sort(key=lambda row: row[0])

    (winner_rmse, winner_offset, winner), (runner_rmse, _, _) = scored[0], scored[1]
    residual = winner_rmse / peak_lr
    separation = runner_rmse / winner_rmse if winner_rmse > 0 else math.inf
    conclusive = residual < MAX_RESIDUAL_FRACTION and separation >= MIN_SEPARATION
    return ScheduleVerdict(
        schedule=winner if conclusive else INCONCLUSIVE,
        best_fit=winner,
        residual_fraction_of_peak=residual,
        separation=separation,
        offset=winner_offset,
        samples=len(pairs),
        candidates=names,
    )
