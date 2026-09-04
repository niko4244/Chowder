# Memory Fabric OOM → success acceptance

Chowder's Memory Fabric (`combined_mechanism_experiment.py`, `placement_policy.py`,
`transformers_peft.py`'s `resolved_activation_offload`/`resolved_optimizer_tiering`/
`resolved_frozen_layer_streaming`) can, since PRs #81-#82, actually drive real
training settings in `"auto"` mode. Every piece of that machinery has its own
real-hardware test coverage (`tests/test_combined_mechanism_experiment.py`,
`tests/test_placement_policy.py`, `tests/test_transformers_backend.py`), but
none of those tests prove the end-to-end claim the roadmap's own Priority 1
milestone asks for:

> Same model, same recipe, same GPU — resident training genuinely CUDA-OOMs,
> Memory Fabric's real placement decision makes the identical recipe succeed.

## Status: **attempted for real, not yet passed**

This is not a stub or a placeholder — real hardware time was spent on this
(RTX 5060 Ti, 16 GB, real Qwen2.5-1.5B and Qwen2.5-3B models, fp32, no
quantization, batch sizes from 8 through 96) — but a clean, reproducible
resident-OOM → Memory-Fabric-success pair has not been demonstrated yet, for
real, evidence-backed reasons documented below. `docs/ROADMAP.md` keeps
Memory Fabric in **IN PRODUCTION HARDENING**, not PROVEN, until this passes.

## What was tried, and what it found

### Attempt 1: Qwen2.5-1.5B, batch=96, max_length=1024, all mechanisms forced "always"

Resident training at this scale does not raise a clean CUDA OOM on this
machine. Instead, Windows' driver-level VRAM-to-system-RAM paging fallback
silently absorbs the overflow: a resident run "succeeded" while its own
`torch.cuda.max_memory_allocated()` reported **21.3 GB against a 17.1 GB
card** — a real, measured value that is only possible because the excess was
quietly paged to system RAM rather than raising an error. Pushing further
(batch=128+) did not crash either; it exceeded the configured timeout (a
different, real failure mode: catastrophic slowdown, not a crash).

Forcing `activation_offload`+`optimizer_tiering`+`frozen_layer_streaming` all
`"always"` simultaneously on this same workload crashed with a real, previously
undiscovered bug:

```
RuntimeError: attn_bias is not correctly aligned (strideM). attn_bias.stride(2) = 66,
and should be a multiple of 4.
```

Bisected: this reproduces with `activation_offload` alone, at this batch/seq
scale, regardless of what else is enabled — a real bug in that mechanism's own
`saved_tensors_hooks`-based CPU offload (most likely: the CPU round-trip
changes a non-contiguous attention-bias tensor's strides in a way a fused
attention kernel rejects), not a multi-mechanism interaction. It never
reproduces at the small batch sizes (2-8) this project's existing test suite
uses, which is why it was never caught before. **Flagged as a separate,
scoped follow-up** rather than fixed inline here (see the spawned background
task on `activation_offload` stride alignment).

### A real, separate production bug found and fixed along the way

`build_placement_plan()`'s internal single-mechanism calibration calls used a
hardcoded 300-second timeout regardless of the recipe's own batch size. A
real batch=96 calibration run legitimately exceeds that on this hardware.
Fixed: the calibration timeout now derives from the recipe's own
`backend.runtime.timeout_seconds` (never shrinking below the existing 300s
default) in both `placement_policy.py` and `combined_mechanism_experiment.py`.
This is a real, merged fix, independent of whether the acceptance test below
ever passes.

### Attempt 2: real `"auto"` placement mode (not forced) on the same workload

With the timeout fix in place, real `"auto"` mode was exercised directly.
`activation_offload`'s own calibration subprocess genuinely raised
`torch.OutOfMemoryError: CUDA out of memory` at this batch size — the first
*clean*, unambiguous CUDA OOM seen in this whole investigation — confirming
this workload really does sit at a genuine memory boundary, just not one that
always manifests as a crash (see the paging-fallback finding above; whether a
given run pages or crashes appears to depend on real-time system state, not
purely on the recipe).

### Attempt 3: Qwen2.5-3B, batch=2, max_length=256 (small, weight-heavy)

Chosen to make frozen-weight residency (not activations) the dominant memory
cost. Real result: peak VRAM was nearly identical between resident (12.33 GB)
and `frozen_layer_streaming` (12.35 GB) — no real memory pressure at this
scale at all; both fit comfortably. The apparent 15x wall-time gap (465s vs
30s) in the first pass at this config was **not** a Memory Fabric effect — it
was a cold-vs-warm model-download-cache artifact in the test methodology
(caught and corrected, not reported as a false positive).

### Attempt 4: Qwen2.5-3B, batch=16, max_length=512, warm cache (methodology-corrected)

A clean, apples-to-apples, warm-cache comparison. Real result: resident
peak VRAM 13.81 GB, `frozen_layer_streaming` peak VRAM 13.88 GB — **nearly
identical**, no meaningful Memory Fabric benefit measured in this specific
full multi-step production run, despite `frozen_layer_streaming`'s own
isolated forward+backward experiment (a different, narrower measurement —
see `frozen_layer_streaming.py`) independently proving real savings.

This is the third time in this project's history that this exact pattern has
shown up (Phase 7D/7E's own real measurements, then `combined_mechanism_
experiment.py`'s PR #81 finding, now this): **a mechanism's isolated
single-forward+backward experiment measuring real savings does not reliably
predict what a full, multi-step `Trainer.train()` run's overall peak VRAM
will show.** The likely cause, consistent across all three occurrences: a
full run's peak can be set at a point (very first step, model load, allocator
warm-up) unrelated to the steady-state per-step savings an isolated
single-shot experiment measures. This is a real, now well-evidenced
limitation of the current placement approach, not a one-off fluke.

### A real, separate confound: shared desktop GPU, not a dedicated card

This development machine's GPU is the same one driving the live desktop
session, not an isolated/dedicated card. The *identical* config (Qwen2.5-3B,
batch=16) failed during model loading in one run and succeeded cleanly in
another, minutes apart — real-time contention for the same physical VRAM
budget from ordinary desktop use (browser tabs, compositing, etc.)
introduces genuine non-determinism this investigation could not control for
locally.

## What this means for next steps

Two real, independent blockers, either of which being resolved makes this
tractable again:

1. **A dedicated/isolated GPU** (removing desktop-contention noise) — the
   same class of access constraint Phase 5's DDP acceptance needed real
   Kaggle 2×T4 hardware for, rather than simulating.
2. **The `activation_offload` stride-alignment bug**, once fixed, reopens an
   activation-heavy workload (large batch, moderate model) as a candidate —
   activation memory is a more controllable, more reliably-scaled lever than
   frozen-weight residency turned out to be at the scales tested here.

Separately, worth investigating on its own merits (not blocking this
specific acceptance test): *why* a full training run's peak VRAM doesn't
reflect an isolated mechanism experiment's real measured savings, now that
this has been observed three times. Understanding that gap could improve
`build_placement_plan()`'s prediction accuracy regardless of which hardware
this acceptance test eventually runs on.

## How to retry

```bash
pip install -e ".[train,dev]"
# On a real, ideally dedicated CUDA GPU:
CHOWDER_REAL_ML_SMOKE=1 python -m pytest -q tests/test_memory_fabric_acceptance.py -v
```

(`tests/test_memory_fabric_acceptance.py` does not exist yet — it will be
added once a workload that reliably reproduces both a clean resident OOM and
a clean Memory Fabric rescue is found. Until then, this document itself is
the acceptance record: an honest "attempted, not yet passed," per this
project's own no-faking discipline.)
