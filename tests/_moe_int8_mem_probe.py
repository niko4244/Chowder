"""The load-memory probe: one process that does nothing but load the checkpoint.

``test_loading_a_mid_size_checkpoint_fits_the_int8_budget`` runs this in a child
process on purpose. The parent has to *write* the synthetic checkpoint (2.3 GB of
shards), and on Windows the host heap keeps that high-water mark committed after
the tensors are freed -- so a load measured in the parent would be reading the
generator's footprint, not the loader's. Here the process starts empty, the
checkpoint is already on disk, and the only thing between the baseline reading
and the load is the import of torch/transformers the loader itself needs.

Reports one JSON object on stdout, prefixed so a stray print cannot be mistaken
for it:

    ``PROBE {json}``

Fields are bytes except ``census`` keys. ``baseline`` is this process's private
bytes after the imports and before ``load_model_int8_experts`` is called, so
``peak - baseline`` is the loader's own transient footprint.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import threading
import time
from pathlib import Path

#: How often the sampler thread reads the process's commit charge.
SAMPLE_INTERVAL_S = 0.005

PREFIX = "PROBE "


def private_bytes() -> int:
    """This process's private (committed) bytes.

    Windows reports ``private`` directly. POSIX has no such counter; unique set
    size is the closest portable answer to "memory only this process holds".
    """
    import psutil

    process = psutil.Process()
    info = process.memory_info()
    private = getattr(info, "private", None)
    if private is not None:
        return int(private)
    return int(process.memory_full_info().uss)


class _Sampler(threading.Thread):
    """Samples private bytes off-thread while the load runs."""

    def __init__(self, interval: float = SAMPLE_INTERVAL_S) -> None:
        super().__init__(daemon=True)
        self.interval = interval
        self.peak = 0
        self.samples = 0
        # Not ``_stop``: Thread already has a private method by that name.
        self._done = threading.Event()

    def run(self) -> None:
        while not self._done.is_set():
            self.peak = max(self.peak, private_bytes())
            self.samples += 1
            time.sleep(self.interval)

    def stop(self) -> None:
        self.peak = max(self.peak, private_bytes())
        self._done.set()
        self.join(timeout=30)


def main(argv: list[str]) -> int:
    base = Path(argv[0])

    import torch  # noqa: F401  (the loader imports it; charge it to the baseline)
    from transformers import AutoModelForCausalLM  # noqa: F401

    from chowder.backends import moe_int8_experts as mie

    gc.collect()
    baseline = private_bytes()
    sampler = _Sampler()
    sampler.start()
    try:
        model, census = mie.load_model_int8_experts(base, device="cpu", dtype=torch.bfloat16)
    finally:
        sampler.stop()
    gc.collect()
    final = private_bytes()

    payload = {
        "baseline": baseline,
        "peak": sampler.peak,
        "final": final,
        "samples": sampler.samples,
        "expert_int8_bytes": census["expert_int8_bytes"],
        "expert_scale_bytes": census["expert_scale_bytes"],
        "non_expert_bytes": census["non_expert_bytes"],
        "expert_modules": census["expert_modules"],
        "expert_buffer_devices": census["expert_buffer_devices"],
        "remaining_expert_params": sum(
            1 for name, _ in model.named_parameters() if ".mlp.experts." in name
        ),
    }
    print(PREFIX + json.dumps(payload), flush=True)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("CHOWDER_NO_GPU", "1")
    raise SystemExit(main(sys.argv[1:]))
