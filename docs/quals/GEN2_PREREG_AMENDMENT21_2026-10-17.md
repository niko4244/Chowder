# GEN2 prereg amendment 21 — 2026-10-17: the budget is re-derived from the campaign's measured model

Adds to the 2026-09-17 prereg and amendments 1–20 and supersedes none of
them. **This is a compute envelope, not a quality gate.** No slice,
threshold, retention bound, comparison or decoding parameter moves, and no
Gen-2 candidate evaluation exists. The same admissibility holds as for
amendment 20: the state root has 0 `candidate_evaluation.json` and 0
`chosen_candidate.json`.

## A. The defect: no real attempt could ever settle

Settlement (`compute_cost.settle_cost`, called from `training_binding`) charges
each attempt **training wall + in-run evaluation wall**. It then refuses the
attempt on any of these:

- the charge exceeds the per-recipe wall ceiling (`ACTUAL_WALL_GPU_HOURS_EXCEEDED`);
- it exceeds the project budget (`RESOURCE_OVERRUN`);
- it exceeds the recipe's own projection by more than 25%
  (`ACTUAL_EXCEEDS_PROJECTION`).

The projection came from `RecipePlanner.project_cost` over the device probe:

| | value | source |
|---|---:|---|
| synthetic step at seq 2048 | 0.0023 s | `prepared-v11/hardware-budget.json` (its `measurement_method`: "not a measurement of the campaign's model") |
| projected wall per recipe | ~0.00085 GPU-h | `campaign_projection` in readiness |
| **measured** training, a12 (344 steps, seq 2048) | **1,318.27 s** (3.83 s/step) | a12 `worker-result.json` → `telemetry.lifecycle` |
| **measured** in-run evaluation at the declared instrument | **3,687.94 s** | parent arm `ancestor-eval-result.json` → `runtime.lifecycle` |
| measured attempt | **1.39 GPU-h** | the two rows above |
| per-recipe ceiling (prepared-v11) | 0.75 | `docs/gen2/gen2_campaign.json` |

A real attempt is about 1,600× its projection and about 1.9× its ceiling, so
every attempt would refuse after spending ~1.4 GPU-h. Amendment 20's batch-size
fix was necessary, but on its own it could not produce a measured candidate.

## B. The correction

- **Code (PR #231):** `calibrate_model_lifecycle` reads the two measured
  artifacts above (both sha256-digested), and `prepare` writes them into the
  hardware budget as `model_lifecycle`. `project_cost` now charges
  `steps × measured s/step + measured in-run eval`, and the executor template
  reserves the measured evaluation cost (1.0244 GPU-h) instead of 0.05.
- **Declaration (this amendment):** the inputs name `prepared-v12`, produced
  with `--calibrate-training <a12 worker-result.json> --calibrate-eval <parent-arm ancestor-eval-result.json>`.
  The ceilings are re-derived from the calibrated projection:

| ceiling | was | now | basis |
|---|---:|---:|---|
| per-recipe device / wall | 0.30 / 0.75 | **2.0 / 2.0** | one recipe projects 1.39 GPU-h. Settlement accepts actual ≤ 1.74 (×1.25) and ≤ 2.0 |
| campaign device / wall | 0.60 / 1.50 | **5.0 / 5.0** | two recipes project 2.78, plus the independent candidate arm (~1.02) ≈ 3.8 |

Under the old ceilings the calibrated planner refuses every recipe before
compute: "smallest candidate projects 1.0372 … vs ceilings 0.3000/0.7500".
That is the right place for the defect to surface, and it is how it was
confirmed.

## C. Evidence

- `prepare` → `prepared-v12` with recipe ids unchanged (`recipe-00-lr5e-05`,
  `recipe-01-lr0.0001`). `model_lifecycle`: 3.832186 s/step at seq 2048,
  in-run evaluation 3,687.944 s. Source digests: `364d2e1b…` (training) and
  `3827dd30…` (evaluation).
- `project-template.json`:
  - every suite `batch_size: 16` (amendment 20);
  - evaluation reserves 1.024429 GPU-h, training reserves 0.975571, budget 2.0;
  - the 7,200 s attempt process timeout covers training + evaluation
    (5,006 s), and the 5,400 s in-run timeout covers 3,688 s with ×1.46 margin.
- Readiness on `prepared-v12/gen2_campaign.prepared.json`: **READY**,
  nineteen of nineteen checks `ok`, `campaign_projection` 2.781231 device /
  2.781231 wall GPU-h within the declared ceilings.

## D. What this amendment does not do

- It does not start the run. That needs an exclusive GPU window, because the
  calibration was measured under the conditions recorded in `model_lifecycle.conditions`.
- It does not claim the calibration transfers to other hardware or another
  base model. A new campaign calibrates its own.
- It does not change the a11 access-violation finding (amendment 20 §E).
