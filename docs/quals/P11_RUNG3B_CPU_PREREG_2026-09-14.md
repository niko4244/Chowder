# P11 rung 3b — bounded tiny-CPU router pilot (the amendment's CPU arm)

Frozen **2026-09-14, before the run**. Committed to `niko4244/Chowder` and
pushed **before** `chowder train` is invoked, so the commit order is the
honest order. This is the amendment's explicitly-provided fallback path: the
9B-derived artifact stays deferred to the amendment-licensed CUDA rung, and
**this run does not qualify the 9B or any large artifact** — it proves the
router-training loop end to end through the normal CLI on the tiny real MoE,
with the new load-policy seam in the code path (policy `fp32-resident`, its
default).

## Workload (fixed before the run; identical recipe to the rung-2 CPU pilot)

| Field | Value |
|---|---|
| Base model | `tiny-qwen3-moe-e4-k2` from `build_tiny_router_pilot.py`, SHA-256 prefix `bfa954eec0f838fc` (re-verified before the run) |
| Training corpus | `router-corpus.txt` (400 sentences, distinct from holdout) |
| Holdout corpus | `router-holdout.txt` (400 sentences) |
| Steps / horizon | 12 steps, `max_steps` stop expected |
| seq_len / batch_size | 16 / 2 |
| learning_rate / seed / probe_window | 0.05 / 1 / 2 |
| eval_batches | 2 |
| Device / load policy | `cpu` / `fp32-resident` (default) |
| minimum_promotion_gain | 0.0 — the gate accepts or rejects on measurement alone |
| Route | `chowder project-validate`, then `chowder train`, in a fresh evidence directory with a fresh registry |

## Budget (refusal, not aspiration)

- Wall: **10 minutes** for the entire validate+train cycle.
- Registry GPU-hour budget 1.0; expected attributable cost ~0 on CPU.

## Thresholds (fixed before the run)

1. **Validate before train**: `chowder project-validate` passes on the fresh
   project file; `chowder train` is only then invoked.
2. **Registry truth**: the run ends with a terminal experiment row; the
   stranded-result audit reports no findings at closeout.
3. **Trainability**: finite non-zero gradients on the intended gate path, a
   real optimizer update, exact gate-path scope; frozen-tensor digests report
   zero changed.
4. **Horizon**: 12/12 steps, `stop_reason = max_steps`.
5. **Gate**: baseline and candidate holdout losses both measured; the gate
   decision is recorded with its ledger, whatever the measurement says. A
   rejected gate is a valid outcome; an unmeasured one is not.
6. **Accounting**: phase ledger carries model load, steady-state steps, and
   both evaluation generations; measured wall time and attributable cost are
   recorded; no outstanding reservations remain.
7. **Identity chain**: base content identity, spec digest, and worker source
   identity all verified as in every qualified run.

## Interpretation rules

Any refusal or failure is recorded with its measured reason and stands as the
rung's outcome — no threshold is renegotiated after seeing the data. A passing
run completes the tiny-CPU arm of the ladder; the 9B arm remains governed by
the amendment and its own future rung.
