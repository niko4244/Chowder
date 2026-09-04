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

## Follow-up investigation: the calibration-vs-production peak mismatch

A dedicated real-hardware investigation into *why* a full training run's
peak VRAM doesn't reflect an isolated mechanism experiment's measured
savings (observed three times above). Two competing hypotheses were tested:

**Hypothesis A (extrapolation error) — tested and disproven.**
`memory_preflight_worker.py`'s dry-run measured peak at batch_size=1 and
batch_size=2 only, then extrapolated linearly to the configured batch size.
A real Qwen2.5-1.5B `Trainer.train()` run at batch_size=8/max_length=256
measured a real peak of **18.6 GB**, reached entirely within the first
training step (confirmed via per-step logging with a real `TrainerCallback`
— not a gradual multi-step climb), while the linear extrapolation from
batch=1/2 predicted only ~10.8 GB. The natural hypothesis: a 4×-48× scale-up
extrapolation from two tiny points is unreliable, and directly measuring at
the real configured batch size would close the gap. **Implemented and
tested directly against real hardware: it did not.** A real, direct
forward+backward measurement at batch_size=8 (bypassing extrapolation
entirely) measured ~10.75 GB — nearly identical to what the old
extrapolation already predicted, and still ~7.9 GB short of the real
`Trainer.train()` peak. This conclusively rules out extrapolation error as
the (or at least the sole) cause.

**Hypothesis B (Trainer-specific overhead) — plausible, not yet confirmed.**
Since a bare `model(...); loss.backward()` measurement — at the *same*
batch size, *same* sequence length, *same* model — lands nowhere near the
real `Trainer.train()` peak, whatever accounts for the ~8 GB gap must come
from something specific to the real `Trainer`/`accelerate` machinery that a
hand-rolled forward+backward loop never exercises: candidates include
gradient-clipping's norm computation, `accelerate`'s own gradient/backward
wrapping, `TrainerState`/callback bookkeeping, or the real `DataLoader`'s
collation path (as opposed to the dry-run's synthetic `torch.randint`
input). None of these were isolated or confirmed as the specific cause —
this would need its own dedicated investigation (e.g. bisecting by
progressively wrapping the bare loop with each piece of real `Trainer`
machinery until the gap reproduces) which was not completed here.

**What was still shipped**: `memory_preflight_worker.py`/`memory_preflight.py`
now take one real, direct measurement at the actual configured batch size
instead of only ever extrapolating from batch=1/2 (see
`MemoryEstimate.measured_peak_gb_at_configured_batch_size`), and a genuine
CUDA OOM during that direct measurement is now treated as a confirmed
non-fit rather than crashing the whole preflight check. This is a real,
defensible improvement in its own right (a direct measurement is always at
least as trustworthy as an extrapolation, and it now catches real
per-batch-size OOMs directly) — but, per the finding above, it does **not**
close the calibration-vs-production gap, and must not be described as
having fixed it. The calibration-timeout derivation fix (see above) was
also applied to this module's own dry-run calls for consistency.

Honest status: the root cause of the ~8 GB Trainer-specific gap remains
unresolved. Any future placement decision built on `estimate_memory_
requirements()` should treat its `estimated_peak_gb` as a real, measured
lower bound for a bare forward+backward pass at the configured batch size,
not a reliable prediction of a full multi-step `Trainer.train()` run's
actual peak.

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
