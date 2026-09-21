# P11 rung 3b (CUDA arm) — the 9B-derived router pilot under the amendment

Frozen **2026-09-14, before the run**. Committed to `niko4244/Chowder` and
pushed **before** `chowder train` is invoked. This rung executes the fallback
the rung-3 amendment licensed: the 9B-derived MoE artifact, trained as a
router pilot on the RTX 5060 Ti under the **`bf16-offload-transient` load
policy**, with budgets derived from the amendment's measured basis ×1.5.

## Governing documents

- Amendment: `docs/quals/P11_RUNG3_AMENDMENT_2026-09-14.md` (`264c03d`)
- Measured basis: `P11_RUNG3_RESULT_2026-09-14.md` (`79b963b`) and
  `Chowder-Protected/runs/2026-09-14-router-healing-p11-rung3/preflight_probe.py`
  (probe SHA-256 `5beee3c48fded712…`): diagnostic D measured 11.368 GB peak,
  1.997 s/step (batch 2, seq 64), loss 3.036 finite, 32/32 gate gradients
  finite and non-zero.

## Workload (fixed before the run)

| Field | Value |
|---|---|
| Base model | `F:\llm-models\Qwen3.8-9B-HotCore-CW-E16-k2-h2176`, manifest SHA-256 `77520edadb9a94f4ed70636328c4bbbaafa49c75e3b47c51418851c5ad4869c4` (re-verified by the full-mode content identity at spec build and by every worker) |
| Training corpus | `router-corpus-9b-pilot.txt` (512 deterministic sentences), SHA-256 `15d5f5f51a739ceee2712fe5b7b550982aba7272f633781f06d5a7ea64f47941` |
| Holdout corpus | `router-holdout-9b-pilot.txt` (64 deterministic sentences, disjoint from training), SHA-256 `97e87c223894d8fbd2af2b79728d324993e5dee28535ef1872c16e466a5718fc` |
| Steps / horizon | 12 steps, `max_steps` stop expected |
| seq_len / batch_size | 64 / 2 — the amendment's measured workload |
| learning_rate / seed / probe_window | 0.05 / 1 / 2 |
| eval_batches | 2 |
| Device / load policy | `cuda` (device 0) / **`bf16-offload-transient`** |
| minimum_promotion_gain | 0.0 — the gate accepts or rejects on measurement alone |
| Route | `chowder project-validate`, then `chowder train`, fresh evidence directory, fresh registry |

## Budget (measured ×1.5, refusal not aspiration)

Derived exactly as the amendment required, from D's measured numbers:

- **Memory**: refuse-before-step-1 line **14.0 GB** (measured 11.368 GB). The
  worker's existing CUDA preflight enforces this from its own probe.
- **Step cost**: ceiling **3.0 s/step** (measured 1.997 s); the preflight's
  projection refuses a run that cannot fit the wall budget.
- **Wall**: **40 minutes** for the whole cycle. Dominated by identity, not
  compute: `resolve_base_identity` is full-mode, so the 18.8 GB artifact is
  re-hashed per call — ~233 s per pass across roughly 5 passes (train spec,
  train worker, baseline eval, candidate eval) ≈ 19 minutes of hashing, plus
  the 12 × ~2 s steps and two model loads (~5 s each under the policy).
- **GPU-hours**: ≤ **0.01** attributable (measured basis: ~30 s of device
  time).

## Thresholds (fixed before the run)

1. **Validate before train**: `chowder project-validate` passes; `chowder
   train` is only then invoked.
2. **Policy contract**: every worker result carries
   `load_policy_report.policy = "bf16-offload-transient"`,
   `dtype = torch.bfloat16`, a `placement_census` with all expert parameters
   on CPU and all gates on `cuda:0`, and `verified: true`.
3. **Measured preflight**: the CUDA preflight records free memory and a real
   step-cost probe, and refuses before step 1 if projections exceed the
   memory or wall budget.
4. **Trainability**: finite non-zero gate gradients on the exact gate path,
   real update, 32 router gates in scope; frozen-tensor digests (`full`
   strategy on-device) report zero changed.
5. **Horizon**: 12/12 steps, `stop_reason = max_steps`.
6. **Gate**: base and candidate holdout losses both measured; the decision is
   recorded with its ledger, whatever it is. Rejected is valid; unmeasured is
   not.
7. **Accounting**: phase ledger with model load, steady-state steps, both
   evaluation generations; measured peak VRAM; attributable GPU-hours within
   the ceiling; reservation settled, no outstanding; stranded-result audit
   clean at closeout.
8. **Identity chain**: full-mode base content identity matches the pinned
   manifest hash; spec digest and worker source identity verified.

## Interpretation rules

Any refusal or failure is the rung's outcome, recorded with its measured
reason — no threshold is renegotiated after seeing the data. A passing run
completes the amendment's CUDA arm at pilot scale; it does not qualify model
quality, and the strict target remains the lowest per-model active-parameter
floor, not this artifact's quality.
