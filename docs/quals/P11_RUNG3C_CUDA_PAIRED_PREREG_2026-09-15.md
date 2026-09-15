# P11 rung 3c (CUDA arm, paired) — the 9B-derived router pilot with the baseline amortized into one resident evaluation

Frozen **2026-09-15, before the run**. Committed to `niko4244/Chowder` and
pushed **before** `chowder train` is invoked. This rung repeats the rung-3b
CUDA arm with the two changes merged in PR #161: the automatic baseline is
**measured inside the candidate's resident evaluation** (`paired_arms`), and
**model load is an explicitly budgeted preflight phase** instead of an
unanticipated cost. The rung-3b exceedance this prereg exists to fix: total
measured device time 0.0206648 GPU-h across three workers exceeded the 0.01
workload-only ceiling by exactly the third load plus its duplicate base
generation (0.0045231 GPU-h). Nothing about that exceedance is renegotiated
here; the successor budget simply prices loads from measurement.

## Governing documents

- Rung-3b preregistration: `docs/quals/P11_RUNG3B_CUDA_PREREG_2026-09-14.md`
  (`4456e75`) and its result
  `Chowder-Protected/runs/2026-09-14-router-healing-rung3b-cuda/P11_RUNG3B_CUDA_RESULT_2026-09-14.md`.
- Amendment: `docs/quals/P11_RUNG3_AMENDMENT_2026-09-14.md` (`264c03d`) — the
  `bf16-offload-transient` policy and ×1.5 derivation rule, unchanged.
- Implementation basis: PR #161 (`a6573f0`, squash-merged as `a495926`) —
  `paired_arms` spec/knob resolution, the resident-pair eval worker
  (baseline-before-apply, single load, `arm="paired"` evidence), engine
  baseline deferral with gate-time refusal and cost reconciliation, and the
  deferred-baseline registry completion. Full suite 1959 passed / 81 skipped;
  all six protected checks green, including the real Transformers/PEFT CPU
  smoke.

## Measured basis (durable phase ledgers from the rung-3b run)

Per-worker `lifecycle` ledgers from `work/.chowder/` (run directory
`router-pilot-9b-ed626dc17192`):

| Worker | model_load | workload phases | measured total |
|---|---|---|---|
| Train | 12.875 s = 0.0035764 | steady_state 24.99 s = 0.0069419; publication+closeout ≈ 0.0000113 | **0.0105295** |
| Baseline eval (separate) | 12.857 s = 0.0035714 | baseline_generation 3.426 s = 0.0009517 | **0.0045231** |
| Candidate eval (already resident-pair) | 12.769 s = 0.0035469 | generations 3.430 + 4.005 s = 0.0020653 | **0.0056122** |

Rung-3b total: **0.0206648 GPU-h, three on-device loads**. The rung-3b
candidate eval already scored base-then-candidate in one process — the
redundancy was the separate baseline worker. Under paired arms the baseline
worker is skipped entirely, so the projected total is train + one paired eval:

> **Projected paired total: 0.0105295 + 0.0056122 = 0.0161417 GPU-h
> ≈ 58.1 s on device, two loads, workload ≈ 0.0090.**

The candidate's holdout result must reproduce rung-3b's measured arithmetic
(same seed, corpus, horizon, policy): base 2.9327, candidate 2.8369,
428/512 experts alive. Equality within the evaluator's determinism is the
expected outcome; a materially different candidate loss at identical inputs
is a defect to investigate, not a threshold.

## Workload (fixed before the run; unchanged from rung-3b)

| Field | Value |
|---|---|
| Base model | `F:\llm-models\Qwen3.8-9B-HotCore-CW-E16-k2-h2176`, manifest SHA-256 `77520edadb9a94f4ed70636328c4bbbaafa49c75e3b47c51418851c5ad4869c4` (re-verified by full-mode content identity at spec build and by every worker) |
| Training corpus | `router-corpus-9b-pilot.txt` (512 deterministic sentences), SHA-256 `15d5f5f51a739ceee2712fe5b7b550982aba7272f633781f06d5a7ea64f47941` |
| Holdout corpus | `router-holdout-9b-pilot.txt` (64 deterministic sentences, disjoint), SHA-256 `97e87c223894d8fbd2af2b79728d324993e5dee28535ef1872c16e466a5718fc` |
| Steps / horizon | 12 steps, `max_steps` stop expected |
| seq_len / batch_size | 64 / 2 |
| learning_rate / seed / probe_window | 0.05 / 1 / 2 |
| eval_batches | 2 |
| Device / load policy | `cuda` (device 0) / **`bf16-offload-transient`** |
| **`paired_arms`** | **`true`** — in both the backend knobs (`config.backend.router_healing`) and the experiment's research patch, so every resolver (spec builder, evaluator, project runner deferral decision) reads the same value from the same precedence order |
| minimum_promotion_gain | 0.0 — the gate accepts or rejects on measurement alone |
| Route | `chowder project-validate`, then `chowder train`, fresh evidence directory (`2026-09-15-router-healing-rung3c-paired-cuda`), fresh registry |

## Budget (measured ×1.5, refusal not aspiration)

- **Attributable GPU-hours ceiling: 0.025** (= 0.0161417 measured × 1.5 =
  0.0242, rounded up). Decomposed, and the decomposition is the enforcement
  basis, not a narrative:
  - model loads: ≤ 0.0107 (two loads × 12.87 s × 1.5 = 38.6 s);
  - steady-state steps: ≤ 0.0100 (12 × 3.0 s/step);
  - generations + publication + closeout: ≤ 0.0043 (7.43 s + 0.04 s × 1.5).
- **Model load as a budgeted phase**: the preflight measures the load and the
  step cost **before step 1** and refuses if load + steps + evaluation
  projection exceeds the ceiling, the step-cost line, or the memory line. A
  run whose loads push it over 0.025 is refused, not excused.
- **Memory**: refusal line **14.0 GB** peak, unchanged. Rung-3b measured
  13.503 GB (train) / 9.48 GB (eval) — a 0.5 GB margin on the 17.1 GB card,
  recorded here as narrow. The policy and workload are identical, so the
  projection basis is identical; pairing removes a load, it does not add
  resident memory.
- **Step cost**: ceiling **3.0 s/step** (measured 1.997–2.804 s/step across
  probe and run).
- **Wall**: **40 minutes** for the whole cycle. Pairing removes one identity
  pass (~4 min of the rung-3b wall) and one eval spawn; full-mode re-hashing
  of the 18.8 GB artifact still dominates.
- **Goal envelope**: `gpu_hour_budget` 0.05. The results table charges
  conservative wall-on-device cost (~2.4× the measured device time in
  rung-3b); projected charges ≈ 0.041, inside the envelope. If actual charges
  exceed it, the exceedance is recorded per the interpretation rules.

## Thresholds (fixed before the run)

1. **Validate before train**: `chowder project-validate` passes; `chowder
   train` is only then invoked.
2. **Policy contract**: every worker result carries
   `load_policy_report.policy = "bf16-offload-transient"`,
   `dtype = torch.bfloat16`, a `placement_census` with all expert parameters
   on CPU and all gates on `cuda:0`, and `verified: true`.
3. **Measured preflight with an explicit load budget**: the CUDA preflight
   records free memory, a real step-cost probe, and the **model-load cost as
   a measured phase**, and refuses before step 1 if the projection (load +
   steps + evaluation) exceeds the 0.025 ceiling, 3.0 s/step, or 14.0 GB.
4. **Trainability**: finite non-zero gate gradients on the exact gate path,
   real update, 32 router gates in scope; frozen-tensor digests report zero
   changed.
5. **Horizon**: 12/12 steps, `stop_reason = max_steps`.
6. **Paired gate contract**: exactly one evaluation worker exists for the
   candidate; its result carries `arm = "paired"` with the base measured
   **before** any payload is read, a single measured `model_load` phase, and
   both generations measured; the separate automatic-baseline spawn is
   skipped (no second eval directory); the baseline registry row is completed
   from the paired evidence with its reconciled measured cost; the gate
   decision is recorded with its ledger, whatever it is. Rejected is valid;
   unmeasured is not.
7. **Accounting**: total attributable GPU-hours on the phase-ledger basis
   ≤ **0.025**, with loads explicitly budgeted; reservations checked at
   proposal and settled; no outstanding; stranded-result audit clean at
   closeout; conservative wall charges recorded, within the 0.05 envelope or
   the exceedance honestly recorded.
8. **Identity chain**: full-mode base content identity matches the pinned
   manifest hash; the payload is bound to base content; spec digest and
   worker source identity verified.

## Interpretation rules

Any refusal or failure is the rung's outcome, recorded with its measured
reason — no threshold is renegotiated after seeing the data. A passing run
completes the paired CUDA arm at pilot scale; it does not qualify model
quality, and the strict target remains the lowest per-model active-parameter
floor, not this artifact's quality.
