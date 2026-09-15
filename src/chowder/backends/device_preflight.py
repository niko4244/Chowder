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

import math
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


def project_load_cost(
    *, load_seconds: float, max_load_seconds: float | None, accelerator_count: int
) -> dict[str, Any]:
    """The third preflight projection: the model load, budgeted like the steps.

    The rung-3b CUDA run recorded a GPU-hour exceedance nobody preregistered
    for: the ceiling was derived from workload-only time, and three on-device
    model loads at 12.8 s each doubled the measured total. A load is a real
    device phase with a measurable cost; it belongs in the same refused-when-
    overrun arithmetic as memory and step cost, not in a post-hoc footnote.

    ``max_load_seconds`` is the *declared* ceiling, frozen into the spec before
    the run; ``load_seconds`` is the measurement. ``load_gpu_hours`` reports
    the load's attributable accelerator hours -- the unit a preregistration's
    ceiling is written in -- so budgeting loads is arithmetic, not archaeology.
    An undeclared ceiling cannot be exceeded and says so rather than guessing.
    """
    load = float(load_seconds)
    if not math.isfinite(load) or load < 0:
        raise ValueError("load_seconds must be a finite non-negative number")
    accelerators = int(accelerator_count)
    if accelerators < 0:
        raise ValueError("accelerator_count must be non-negative")
    if max_load_seconds is None:
        return {
            "measured": True,
            "load_seconds": load,
            "max_load_seconds": None,
            "max_load_gpu_hours": None,
            "load_gpu_hours": (load * accelerators) / 3600.0,
            "would_exceed_load_budget": False,
        }
    ceiling = float(max_load_seconds)
    if not math.isfinite(ceiling) or ceiling <= 0:
        raise ValueError("max_load_seconds must be a finite positive number when set")
    return {
        "measured": True,
        "load_seconds": load,
        "max_load_seconds": ceiling,
        "max_load_gpu_hours": (ceiling * accelerators) / 3600.0,
        "load_gpu_hours": (load * accelerators) / 3600.0,
        "would_exceed_load_budget": load > ceiling,
    }
