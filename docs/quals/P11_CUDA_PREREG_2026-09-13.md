# P11 rung 2 — small-CUDA router qualification preregistration

Frozen **2026-09-13, before the implementation commit and before any run**.
This file will be committed to `niko4244/Chowder` (`codex/p7-evidence-guards`)
before the code change that lifts the guard, so the GitHub commit order is the
honest order: preregister first, run second.

## What this rung is

Lift the router backend's CPU-only `QUALIFIED_DEVICES` guard to admit `cuda`
**only behind a real, preregistered qualification on a small CUDA workload**,
measured by the same worker that will later run bigger workloads. Success
qualifies the *device path* (loading, training, freezing, payload publication,
evaluation, accounting on a real accelerator). It does not qualify a model, a
quality result, or the 9B pilot.

## Workload (fixed before the run)

| Field | Value |
|---|---|
| Base model | tiny Qwen3 MoE (E=4, k=2), builder `build_tiny_router_pilot.py` SHA-256 `bfa954eec0f838fc70ae66d07fc67c529a552b8af0e7a6e67329a36eca2a22e6` |
| Training corpus | `router-corpus.txt` (400 sentences, distinct from holdout) |
| Holdout corpus | `router-holdout.txt` (400 sentences) |
| Steps / horizon | 12 steps, horizon 12, stop_reason `max_steps` expected |
| seq_len / batch_size | 16 / 2 |
| learning_rate / seed | 0.05 / 1 |
| Device | `cuda` (single GPU, device 0) |
| Route | `chowder project-validate` then `chowder train` on a CUDA project file |

## Budget (refusal, not aspiration)

- Wall: **10 minutes** for training; **5 minutes** for the baseline arm.
- Memory: the measured preflight **must refuse before optimizer step 1** if
  projected peak exceeds the device's free memory. There is no "close enough".
- GPU-hours: ≤ **0.1** attributed to the qualification (expected far below).

## Thresholds (fixed before the run)

1. **Preflight is measured, not configured**: the run's device preflight
   records measured free memory, a measured first forward/backward/update
   (the step-cost probe), and a projected peak before training begins. A
   projected overflow refuses before step 1.
2. **Trainability on device**: `trainability.ok` is true with the same
   intended-component evidence as CPU (finite non-zero gradients, real update,
   exact gate-path scope). Frozen-tensor before/after digests on CUDA match
   what a host re-read of the same weights reports (full strategy for these
   small tensors — no sampled-vs-full ambiguity).
3. **Frozen unchanged**: frozen digest strategy reported, zero changed.
4. **Limits honoured**: 12/12 steps, 384 tokens, `max_steps`.
5. **Gate**: candidate holdout loss vs base, minimum_promotion_gain 0.0 — the
   accept/reject is whatever the measurements produce; the only requirement is
   that it is recorded with its ledger.
6. **Accounting**: `active_accelerator_count` = 1, peak VRAM measured (not
   `{}`), phases present: model_load, steady_state_steps, payload publication;
   candidate evaluation ledger includes baseline + candidate generation.
7. **Identity chain intact**: base content identity, spec digest, worker source
   identity all verified as on CPU.

## Interpretation rules

- Any threshold failure ends the rung as a measured negative with artifacts
  preserved; the guard re-tightens rather than loosening silently.
- A preflight refusal before any training is a **valid qualification outcome**
  of the budget mechanism (it must be demonstrated in tests regardless), but
  the rung itself is only qualified by a completed run.
- No champion promotion, no 9B work, no release permission is claimed from this
  rung. CPU remains a qualified device; this adds `cuda` behind evidence.
