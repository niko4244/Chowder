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


def project_device_memory(
    *, free_bytes: int, peak_bytes: int, resident_before_step_bytes: int = 0
) -> dict[str, Any]:
    """Refuse a run whose measured step demand cannot fit measured free memory.

    ``peak_bytes`` is sampled with ``torch.cuda.memory_allocated`` *during* a
    real step, so it already contains the resident model; ``free_bytes`` is
    measured after the model is resident too. Comparing them directly demands
    the model fit twice — the rung-3b CUDA run caught this on the 9B artifact
    (11.15 GB "peak" vs 5.49 GB free, when diagnostic D had measured the same
    workload fitting at an 11.37 GB absolute peak). The honest comparison is
    the *incremental* step demand — allocated minus the resident bytes before
    the step — against free memory. Pass ``resident_before_step_bytes=0`` to
    keep the old absolute-peak semantics (valid when the sampler baseline is
    empty).
    """
    free = int(free_bytes)
    peak = int(peak_bytes)
    resident = int(resident_before_step_bytes)
    incremental = peak - resident
    return {
        "measured": True,
        "free_memory_bytes": free,
        "peak_step_bytes": peak,
        "resident_before_step_bytes": resident,
        "incremental_step_bytes": incremental,
        "headroom_bytes": free - incremental,
        "projected_oom": incremental > free,
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
