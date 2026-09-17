# Generation 1 Preregistration — Amendment 3 (2026-09-17)

Written after real hardware execution refuted the training-vehicle arithmetic
of the prereg as composed. Nothing about the *science* changes: target
weakness, instrument, thresholds (EOS ≥ 0.90; unclosed-think ≤ 0.250; loops =
0; trigram ≥ 0.900), promotion rule, secondary gates, seeds, curriculum,
contamination firewall, and stopping rules all stand frozen. What changes is
the execution vehicle and the budget units it is actually charged in, both
corrected from measurement before any candidate training.

## C1 — the frozen training config does not fit this hardware

A 3-step probe of the exact frozen recipe shape (batch 4 × grad-accum 4 ×
seq 1024, bf16, `frozen_layer_streaming: always`, `quantization: none`) on
the RTX 5060 Ti measured:

| quantity | measured |
| --- | --- |
| steady-state optimizer step | **204.7 s = 0.0569 GPU-h per step** |
| peak device memory | **19.90 GB** (exceeds the 17.1 GB card → WDDM spill) |
| frozen-layer streaming | 130.1 GB transferred (mechanism works) |
| model load | 2.7 s |

The frozen `max_steps: 200` therefore projects **≈ 11.4 GPU-h** — outside
any affordable envelope and not executable within the prereg's aggregate.

A second probe at the reduced micro-batch (batch 1 × grad-accum 1 × seq 512,
same dtype/policy/mechanisms) measured:

| quantity | measured |
| --- | --- |
| steady-state optimizer step | **28.9 s = 0.0080 GPU-h per step** (57.7 s / 2 steps) |
| peak device memory | **15.93 GB** (fits the card) |
| optimizer behavior | 2 steps, loss 3.01 → 1.80, finite; adapters saved |

The streamed-LoRA mechanism is sound; only the token throughput was
overestimated (streaming moves ~31 GB/step, which dominates step cost).

## C2 — budget units: the engine charges wall, not device

Attempts 07/08 of the first protocol-compliance campaign exposed a unit
error in the composed projects: the prereg's 0.25 GPU-h training ceiling was
written into `goal.gpu_hour_budget`, but the engine seeds
`spent_gpu_hours` with the automatic baseline's **wall-charged** row and
refuses when `spent + reserved > goal` (attempt-07: baseline measured
cleanly at 0.373 GPU-h wall > 0.25 → mechanical refusal after a fully
successful baseline measurement). The refusal was the enforcement working
correctly against a mis-specified envelope. All budgets below are stated in
the engine's actual charging units (wall), with the device/wall distinction
recorded.

## C3 — baseline carriage, not re-measurement

`baseline.mode: auto` re-pays a full base-model measurement per recipe
(0.373 GPU-h wall measured at attempt-07) for a quantity the frozen
evidence already holds: attempt-07's baseline IS the Gen-0 parent measured
under the identical protocol (16 diagnostics, 32 tokens, offload placement,
bf16, quality 0.0625, protocol sha `5adb2f61204082533fa555976d1976671cd1975b92a927a58317a767ab3417a3`).
Amendment: each composed project carries that measurement as
`baseline.mode: fixed` (metrics `{quality: 0.0625}`, the protocol digest,
`gpu_hours: 0.0` — a measurement pointer, not a re-paid cost). The candidate
side is still measured by the independent evaluator against the fresh
artifact; only the *parent arm* is carried.

## C4 — amended training recipes (from measurement, frozen here)

| | recipe-a | recipe-b |
| --- | --- | --- |
| max_steps | **30** | **30** |
| batch × grad-accum | **1 × 1** | same |
| seq_len (`max_length`) | **512** | same |
| warmup | **2** (scaled to the 30-step horizon) | same |
| lr / scheduler | 1e-4 / cosine (unchanged) | 2e-4 / cosine (unchanged) |
| LoRA / dtype / streaming / quantization | unchanged (16/32, bf16, always, none) | same |

30 steps × 28.9 s = 867 s ≈ **0.241 GPU-h** projected training cost per
recipe; the 48-row corpus receives ~1.3 epochs of exposure — thin but real,
and the target behavior (closing a think block, emitting EOS) is the class
SFT teaches fastest. The project goal budget becomes **0.30 GPU-h per
recipe (wall)**; total campaign envelope 0.60 (train, both recipes) + 1.00
(evaluation aggregate) = **1.60 GPU-h**, replacing the prereg's 1.50 whose
arithmetic mixed device and wall units.

## C5 — WDDM/first-GPU-process operational rule

Across this campaign, every native 0xC0000005 crash of the training worker
occurred when a prior CUDA-heavy process (automatic baseline eval or a
previously crashed attempt) had already run in the same driver session;
three consecutive clean-session runs (fixed-baseline ordering, no preceding
eval) completed model load, training steps, and adapter publication without
fault. bitsandbytes quantized paths remain dead on this card (Amendment 1,
B1 — reproducibly, independently of session state). Operational rule for
every remaining Gen-1 run: training is the **first GPU workload** of its
session; the fixed-baseline ordering (C3) guarantees it by construction.

## Unchanged

Target, instrument, thresholds, promotion rule, secondary gates, seeds,
curriculum, firewall, stopping rules, and the evaluation aggregate (1.00)
stand as frozen. A REJECTED verdict remains an acceptable, durable outcome.
