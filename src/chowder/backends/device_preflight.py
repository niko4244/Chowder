"""The device preflight arithmetic for router-healing runs.

P11 rung 2: the guard admits `cuda` only behind a measured preflight. The
arithmetic lives here so it is unit-testable without a device and identical in
both workers. Two projections, both refuse-shaped:

* memory — a step peak measured before training must fit the free memory
  measured before training, with headroom stated rather than implied;
* wall — a measured per-step cost must fit the declared horizon inside the
  declared wall budget.

Both projections label themselves `measured: true` only when the inputs they
consume were actually measured; a projection over assumed inputs is not a
preflight, it is a guess wearing one.
"""

from __future__ import annotations

from typing import Any

GIB = 1024.0**3


def project_device_memory(*, free_bytes: int, peak_bytes: int) -> dict[str, Any]:
    """Refuse a run whose measured step peak cannot fit measured free memory."""
    free = int(free_bytes)
    peak = int(peak_bytes)
    return {
        "measured": True,
        "free_memory_bytes": free,
        "peak_step_bytes": peak,
        "headroom_bytes": free - peak,
        "projected_oom": peak > free,
    }


def project_step_cost(
    *, step_seconds: float, max_steps: int, max_seconds: float | None
) -> dict[str, Any]:
    """Refuse a run whose measured step cost cannot fit the declared horizon."""
    per_step = float(step_seconds)
    steps = int(max_steps)
    projected = per_step * steps
    if max_seconds is None:
        return {
            "measured": True,
            "step_seconds": per_step,
            "max_steps": steps,
            "projected_wall_seconds": projected,
            "max_seconds": None,
            "would_exceed_budget": False,
        }
    budget = float(max_seconds)
    return {
        "measured": True,
        "step_seconds": per_step,
        "max_steps": steps,
        "projected_wall_seconds": projected,
        "max_seconds": budget,
        "would_exceed_budget": projected > budget,
    }
