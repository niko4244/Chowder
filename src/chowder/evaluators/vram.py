"""How much VRAM did *this process* use? Reported so a run can answer it.

The training workers have always reported `peak_vram_gb`. The evaluation workers
did not, and that gap cost a verdict.

`PRUNED_9B_REAL_TRAINING_PREREG.md` pre-registered "peak VRAM under 15.93 GiB" and
listed oversubscription as a FAIL "judged by headroom and step-time blowup". For the
evaluation leg neither quantity existed in the run's artifacts, so the only available
proxy was `nvidia-smi` -- which reports the whole **machine**: every browser, every
service, every other process. It read 559 MiB free during the candidate eval and the
run was recorded as FAIL on oversubscription.

Controlled re-measurement afterwards put the evaluation process's own footprint at
~6.6 GiB, flat across token budgets from 1 to 768 and across 12 sequential problems,
with ~8.8 GiB of card free throughout. Five candidate mechanisms were eliminated
(adapter weights, token count, accumulation, cache class, instrument disagreement),
and the remaining ~9 GiB was held by something outside the experiment that could not
be identified. A run has to be able to answer "how much did *I* use" from its own
evidence; otherwise a busy desktop can fail an experiment.

Both numbers are reported because today's investigation turned on the difference:

* `peak_vram_gb` (allocated) -- what the model actually needed.
* `peak_vram_reserved_gb` -- what torch took from the driver, which is the figure
  that decides whether the run fits *alongside anything else*.

Never raises. Telemetry must not be able to kill a run -- the same rule
`progress_write.py` exists to enforce, learned the same way.
"""
from __future__ import annotations

from typing import Any

#: Explicit "not measured" rather than 0.0, so a CPU run or a torch-less
#: environment cannot be read as "used no memory". Unknown is not zero -- the same
#: distinction `target_coverage` makes with `status: "not_reported"`.
_UNMEASURED: dict[str, Any] = {"peak_vram_gb": None, "peak_vram_reserved_gb": None}


def peak_vram(device_name: str) -> dict[str, Any]:
    """Peak VRAM this process allocated and reserved on `device_name`, in GiB."""
    if not device_name.startswith("cuda"):
        return dict(_UNMEASURED)
    try:
        import torch

        index = int(device_name.split(":", 1)[1]) if ":" in device_name else 0
        return {
            "peak_vram_gb": float(torch.cuda.max_memory_allocated(index) / (1024**3)),
            "peak_vram_reserved_gb": float(torch.cuda.max_memory_reserved(index) / (1024**3)),
        }
    except Exception:  # pragma: no cover - defensive: telemetry is never fatal
        return dict(_UNMEASURED)
