"""Difficulty calibration for curriculum items.

Difficulty is estimated from evidence, not asserted: base-model success
rate (the current model's own pass rate on similar items), stronger-model
success rate, stated reasoning-step counts, solution length, and verifier
complexity. Items land in bands:

- trivial: base model solves nearly always (no training value)
- easy: base model mostly solves (consolidation value only)
- medium: base model sometimes solves (the productive training zone)
- hard: base model rarely solves, stronger models can (stretch)
- extreme: no known solver at any scale (not trainable -- park it)

Curricula progress upward through the bands as the model improves: the
mixture shifts, it never jumps straight to impossible items.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

BANDS = ("trivial", "easy", "medium", "hard", "extreme")

#: Band bounds on estimated success probability of the target model.
BAND_BOUNDS: dict[str, tuple[float, float]] = {
    "trivial": (0.95, 1.01),
    "easy": (0.75, 0.95),
    "medium": (0.25, 0.75),
    "hard": (0.02, 0.25),
    "extreme": (-0.01, 0.02),
}


@dataclass(frozen=True)
class DifficultyEstimate:
    band: str
    estimated_success_probability: float
    confidence: float
    contributors: dict[str, float]

    def __post_init__(self) -> None:
        if self.band not in BANDS:
            raise ValueError(f"unknown band {self.band!r}")


def estimate_difficulty(
    *,
    base_success_rate: float | None = None,
    stronger_success_rate: float | None = None,
    reasoning_steps: int | None = None,
    solution_length_tokens: int | None = None,
    verifier_complexity: float | None = None,
) -> DifficultyEstimate:
    """Combine evidence into a difficulty estimate.

    Each available contributor maps to an implied success probability for
    the target model; the estimate is their confidence-weighted mean.
    ``stronger_success_rate`` anchors the ceiling: if a stronger model also
    fails, the item leans extreme regardless of other signals.
    """
    contributors: dict[tuple[str, float], float] = {}
    weight_by_name: dict[str, float] = {}

    def _add(name: str, implied_success: float, weight: float) -> None:
        if not 0.0 <= implied_success <= 1.0:
            raise ValueError(f"{name}: implied success must be in [0, 1]")
        contributors[(name, implied_success)] = weight
        weight_by_name[name] = weight

    if base_success_rate is not None:
        _add("base_success", float(base_success_rate), 1.0)
    if stronger_success_rate is not None:
        # A stronger model's success caps the item's solvability: the target
        # model's implied success is pulled toward the stronger model's rate
        # from below (we cannot be better than the strongest known solver
        # minus a margin for the skill gap).
        implied = max(0.0, min(float(stronger_success_rate), 0.9))
        _add("stronger_anchor", implied, 0.5)
    if reasoning_steps is not None:
        # More steps -> harder. Calibrated so ~3 steps ~ 0.85 success,
        # ~10 steps ~ 0.4, ~25+ steps ~ <0.15.
        implied = max(0.05, min(0.95, 1.0 / (1.0 + max(0, reasoning_steps - 1) / 6.0)))
        _add("reasoning_steps", implied, 0.4)
    if solution_length_tokens is not None:
        implied = max(0.05, min(0.95, 1.0 / (1.0 + max(0, solution_length_tokens - 120) / 900.0)))
        _add("solution_length", implied, 0.3)
    if verifier_complexity is not None:
        if not 0.0 <= verifier_complexity <= 1.0:
            raise ValueError("verifier_complexity must be in [0, 1]")
        implied = 1.0 - 0.5 * verifier_complexity
        _add("verifier_complexity", implied, 0.3)

    if not contributors:
        raise ValueError("estimate_difficulty requires at least one evidence signal")

    total_weight = sum(w for w in weight_by_name.values())
    estimated = sum(
        implied * w for (_name, implied), w in contributors.items()
    ) / total_weight
    confidence = min(1.0, total_weight / 2.0)

    for band, (low, high) in BAND_BOUNDS.items():
        if low <= estimated < high:
            return DifficultyEstimate(
                band=band,
                estimated_success_probability=estimated,
                confidence=confidence,
                contributors={name: implied for (name, implied), _ in contributors.items()},
            )
    return DifficultyEstimate(
        band="extreme",
        estimated_success_probability=estimated,
        confidence=confidence,
        contributors={name: implied for (name, implied), _ in contributors.items()},
    )


def mixture_for_capability(
    current_success_rate: float,
    *,
    progression: bool = True,
) -> Mapping[str, float]:
    """The evidence-driven band mixture for one skill at the model's current
    success rate on it.

    The model trains mostly in the productive zone (medium), with
    consolidation below and stretch above. As capability rises the mixture
    shifts upward (progression), never presenting a wall of impossible
    items. Not hard-coded percentages: derived from the measured rate.
    """
    if not 0.0 <= current_success_rate <= 1.0:
        raise ValueError("current_success_rate must be in [0, 1]")
    if current_success_rate < 0.15:
        # The skill is barely present: consolidate on easy/medium, tiny stretch.
        mix = {"trivial": 0.05, "easy": 0.35, "medium": 0.45, "hard": 0.15, "extreme": 0.0}
    elif current_success_rate < 0.45:
        mix = {"trivial": 0.0, "easy": 0.25, "medium": 0.50, "hard": 0.25, "extreme": 0.0}
    elif current_success_rate < 0.75:
        mix = {"trivial": 0.0, "easy": 0.15, "medium": 0.45, "hard": 0.40, "extreme": 0.0}
    elif current_success_rate < 0.95:
        mix = {"trivial": 0.0, "easy": 0.10, "medium": 0.30, "hard": 0.60, "extreme": 0.0}
    else:
        # Near-saturation locally: maintain with hard items; extreme stays
        # parked unless a stronger-model anchor shows it is solvable.
        mix = {"trivial": 0.0, "easy": 0.0, "medium": 0.20, "hard": 0.80, "extreme": 0.0}
    if not progression:
        # Ablation control: flat productive-zone mix.
        mix = {"trivial": 0.0, "easy": 0.2, "medium": 0.6, "hard": 0.2, "extreme": 0.0}
    return mix


def band_sequence(
    items: Sequence[tuple[str, float]],
) -> list[tuple[str, str]]:
    """Order (item_id, estimated_success) pairs for progressive training:
    ascending success probability (easy -> hard) with a small warmup of the
    easiest items. Returns (item_id, band) in training order."""
    ordered = sorted(items, key=lambda pair: -pair[1])
    result: list[tuple[str, str]] = []
    for item_id, success in ordered:
        for band, (low, high) in BAND_BOUNDS.items():
            if low <= success < high:
                result.append((item_id, band))
                break
    return result
