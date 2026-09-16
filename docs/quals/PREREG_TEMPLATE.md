# Preregistration template — router-healing qualification runs

Copy this file to `docs/quals/<RUN>_PREREG_<date>.md`, fill every field, and
commit it **before** implementation or execution. The run is judged by a
mechanical judge built from this document (`docs/quals/HARNESS.md`).

## 1. Identity

- Run name / ladder rung:
- Base artifact + content manifest pin (path-independent digest):
- Device + card memory:
- Branch/PR carrying this prereg (merged before the run):

## 2. Workload (frozen)

- Model, steps, tokens/step, corpus files with SHA-256 pins:
- Policy (e.g. `bf16-offload-transient`) and dtype:

## 3. Budgets (all enforced by preflight; see `project_run_ceiling`)

- Aggregate ceiling (GPU-h, device basis):
- Sub-budgets: loads ____ / steps ____ / generations ____ (device basis)
- `max_load_seconds` per model load:
- Wall-charge envelope (goal accounting, NOT a refusal threshold): ______

### Wall-charge reconciliation (mandatory since rung 3c)

Wall charges (registry `gpu_hours`) are goal accounting billed on wall time,
while preflight ceilings govern the device basis. The two are related by the
run class's measured wall multiplier M = wall GPU-h / device GPU-h
(rung 3c measured **M = 3.5**: 0.0585 wall / 0.0167 device).

Every prereg must therefore predict the wall charge from measurement:

    predicted_wall_gpu_h = device_budget_gpu_h × M,
    with M named from a prior run of the same class,

and set the envelope **≥ predicted_wall_gpu_h**. Asserting an envelope below
the measurement-derived prediction is a prereg defect, not a run failure. With
M = 3.5, any device ceiling needs an envelope of at least 3.5× it. The
envelope stays a recorded goal (the judge surfaces exceedance; a human records
it; refusal thresholds stay on the device basis).

Ledger note (fixed 2026-09-15): a paired baseline row no longer re-charges
the shared resident wall — it records gpu_hours 0.0 with the shared charge
disclosed in `compute.shared_wall_gpu_hours` and `compute.charged_to`. Wall
predictions must use the post-fix model: candidate row only.

## 4. Thresholds

T1 validate-before-train; T2 policy contract; T3 measured preflight (no
projection exceeds); T4 trainability; T5 horizon; T6 paired-gate contract;
T7 accounting (device basis); T8 identity chain. Define each check's exact
artifact fields.

### T4a — routing-collapse threshold (explicit, replaces the binary contract)

The binary "every gate grad-nonzero on every step" contract **overstates**
collapsed-topology runs (rung 3b/3c: base routers arrive maximally collapsed;
saturated softmax yields exact-zero gradients that honest workers must record).
Future preregs judge collapse explicitly:

- **Metric**: per-layer `grad_zero_steps` (steps with exact-zero gate
  gradients) and `dead_experts` (experts with zero routing mass), from the
  train worker's per-layer trainability record and the eval routing table.
- **Baseline reference**: record the base model's dead-expert count (measured
  in a baseline eval or pinned from a prior run's artifact).
- **Thresholds** (set per rung; defaults below):
  - `dead_experts_after <= dead_experts_base` — training must not increase
    collapse (equality allowed: a fully saturated run that changed nothing
    else may still qualify as a mechanical control).
  - `layers_with_grad_zero <= collapse_allowance` where
    `collapse_allowance` is preregistered (rung 3c value: 3 of 32).
  - Any layer with `grad_zero_steps == horizon` must be named and explained
    in the result doc (topology cause vs optimizer cause), citing the
    baseline routing evidence.
- **Unknown handling**: missing per-layer records ⇒ T4a = UNKNOWN (refuses
  certification), never an assumed pass.

## 5. Interpretation rules

- Judge exit 0 only when every threshold is PASS; UNKNOWN refuses to certify
  anything.
- Wall-charge exceedance of the envelope: recorded in the result doc by a
  human, never auto-failed (device basis governs refusal).
- All attempts preserved: budget-refused attempts are evidence, and any
  surviving run must pass every threshold.

## 6. Judge

Judge script name (committed on the same PR, dry-run against the prior run's
artifacts before the run), plus the verbatim-capture convention for the
initial verdict.
