from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class ResourceUsage:
    """Measured execution resource usage.

    ``accelerator_seconds`` is the accounting primitive. It naturally
    distinguishes one accelerator from two accelerators running for the same
    wall time, while avoiding the false assumption that their VRAM is one
    contiguous pool.
    """

    wall_seconds: float
    accelerator_seconds: float
    active_accelerator_count: int = 0
    visible_accelerator_count: int = 0
    peak_vram_gb_by_accelerator: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for label, value in (
            ("wall_seconds", self.wall_seconds),
            ("accelerator_seconds", self.accelerator_seconds),
        ):
            number = float(value)
            if not math.isfinite(number) or number < 0:
                raise ValueError(f"{label} must be finite and non-negative")
        for label, value in (
            ("active_accelerator_count", self.active_accelerator_count),
            ("visible_accelerator_count", self.visible_accelerator_count),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{label} must be a non-negative integer")
        if self.visible_accelerator_count and (
            self.active_accelerator_count > self.visible_accelerator_count
        ):
            raise ValueError("active accelerator count cannot exceed visible accelerator count")
        for key, value in self.peak_vram_gb_by_accelerator.items():
            if not isinstance(key, str) or not key:
                raise ValueError("accelerator resource keys must be non-empty strings")
            number = float(value)
            if not math.isfinite(number) or number < 0:
                raise ValueError("peak VRAM values must be finite and non-negative")

    @property
    def gpu_hours(self) -> float:
        return self.accelerator_seconds / 3600.0

    @classmethod
    def from_wall_time(
        cls,
        *,
        wall_seconds: float,
        active_accelerator_count: int,
        visible_accelerator_count: int | None = None,
        peak_vram_gb_by_accelerator: Mapping[str, float] | None = None,
    ) -> "ResourceUsage":
        visible = (
            active_accelerator_count
            if visible_accelerator_count is None
            else visible_accelerator_count
        )
        return cls(
            wall_seconds=wall_seconds,
            accelerator_seconds=wall_seconds * active_accelerator_count,
            active_accelerator_count=active_accelerator_count,
            visible_accelerator_count=visible,
            peak_vram_gb_by_accelerator=dict(peak_vram_gb_by_accelerator or {}),
        )
