# Growth Integrity Assessment — measurement provenance, budget settlement, and accounting (2026-09-17)

Scope note: this note documents **traced code paths** on `origin/main`
(252c0e06), not doc comments. Every claim below was read out of the named
file/line ranges before any behavior change.

## Data flow today

```
driver (docs/gen1/run_gen1_cycle.py)
  ├─ _candidate_runs()      -> BenchmarkRun rows labeled generation_version="gen1"
  ├─ _parent_runs()         -> BenchmarkRun rows labeled generation_version="gen0"
  ├─ MetricBinder.from_manifest(registry, manifest)
  ├─ binder.promotion_input(candidate_runs, parent_runs, ...)
  │     └─ bind()/bind_all()   -> BenchmarkResult(score, samples, contamination)
  │     └─ evaluate_promotion(PromotionInput)   [growth/promotion.py]
  │           └─ compare()                      [growth/statistics.py]
  └─ GenerationLedger.record()                  [growth/lineage.py]
```

## Defect 1 — no measurement provenance; parent rows relabeled as candidate

- `BenchmarkResult` (promotion.py:26) carries only
  `benchmark_qualified_id / score / samples / contamination /
  measurement_confidence`. There is **no field** distinguishing a measurement
  produced by this generation from one copied from the parent.
- `MetricBinder.bind()` (metric_binding.py:233) checks only
  `run.generation_version == generation_version` — a *label*, and the label is
  set by the caller. The driver's `_candidate_runs()` (run_gen1_cycle.py:912)
  takes the parent's frozen math500/mgsm rows and re-emits them with
  `generation_version=CANDIDATE`, `notes="carried from the frozen Gen-0
  attempt-2 measurement"`. The binder cannot see the difference: the parent's
  48 mgsm per-sample values become "candidate samples".
- Consequence in `evaluate_promotion` (promotion.py:140-166): a carried pair
  (identical arrays, delta exactly 0) lands in the aggregate-only branch and
  is certified `protected:<id> = "not-regressed"`; with mgsm's 48 real parent
  samples on *both* sides the paired branch runs Welch's t-test on a
  degenerate all-zero pair and certifies `"ok"`. Either way, **no
  candidate-side measurement existed**, yet the promotion record reads
  `protected_regression: ok` and `broad_battery: ok` (same carried rows feed
  the broad-battery means).

## Defect 2 — no settlement; actual cost never checked

- Admission: `SubprocessTrainingFn._check_cost` (training_binding.py:542)
  refuses *before* compute when `recipe.projected_device_gpu_hours` /
  `projected_wall_gpu_hours` exceed the envelope, and the project's own
  `goal.gpu_hour_budget` is enforced preflight by the engine
  (engine.py:72: `spent + reserved > goal.gpu_hour_budget`).
- Settlement: `_settle` (training_binding.py:757) verifies registry rows,
  source identity, artifact, stranded results — and then returns
  `STATUS_SUCCEEDED` **without ever comparing `evidence["measured_gpu_hours"]`
  (the trainer's actual reported GPU-h) against the recipe's projection, the
  envelope, or the project budget**. A run that trains successfully at any
  actual cost settles as a success.
- `evaluate_promotion`'s resource check (promotion.py:232) sees a single
  `device_gpu_hours` number against a single `device_gpu_hours_ceiling` —
  no wall unit, no projected-vs-actual pair, no campaign total, and the
  caller (driver) passes the number by hand.

## Defect 3 — incomplete accounting

- The driver's `evidence_gpu_hours()` sums per-attempt
  `measured_gpu_hours` for the *chosen* recipe only; losing recipes
  (attempt-11: 0.397 GPU-h), the automatic-baseline measurement inside each
  attempt, and any failed attempts are omitted from what the judge sees.
- Wall vs device units: attempts record wall-charged GPU-h
  (`training-evidence.json["measured_gpu_hours"]` = 0.4325 wall for
  attempt-10) while the prereg's ceilings were device units; nothing in code
  forces the distinction (this exact confusion caused the attempt-07/08
  refusals documented in GEN1_PREREG_AMENDMENT3).
- Baseline: with `baseline.mode: fixed`, the parent measurement is referenced
  at zero incremental cost — correct — but nothing records the *reference*
  explicitly; with `auto` it is re-paid silently.

## Defect 4 — no correction path for lineage

`GenerationLedger.record` refuses a duplicate version (lineage.py:108) and
`_flush` rewrites `generations.json` wholesale — append-only per *record*,
but there is no way to say "this adjudication was later found unsound" without
mutating the historical bytes or inventing a new generation version.

## What must change (design)

1. `MeasurementOrigin` enum on `BenchmarkResult` +
   `BenchmarkRun.measurement_origin` (schema field, defaulting to
   `UNMEASURED` for legacy rows — legacy is never silently trusted on the
   candidate side). Values: `MEASURED_THIS_GENERATION`,
   `MEASURED_PARENT`, `CARRIED_REFERENCE`, `UNMEASURED`.
2. Binder: candidate-side binding requires
   `MEASURED_THIS_GENERATION` (or an explicit `UNMEASURED` row, bound as an
   honest non-measurement that promotion treats as inconclusive). Parent-side
   binding accepts `MEASURED_THIS_GENERATION`/`MEASURED_PARENT`. A row whose
   origin contradicts its generation label refuses with a named reason.
   `evaluate_promotion`: protected/broad/target gates count only rows with
   candidate-measured provenance; carried rows render as evidence but gate
   as inconclusive. Identical copied arrays cannot manufacture paired
   significance because they can no longer enter as candidate samples.
3. Settlement: after a successful train, compare actual measured cost against
   the recipe's own projection, the envelope's wall and device ceilings, and
   the project budget; overrun => `STATUS_REFUSED`-style failure with
   machine-readable `ACTUAL_*_EXCEEDED`, artifact and measurements preserved.
   *Corrected 2026-09-18:* the device ceiling may only be settled against a
   measured device figure. `ACTUAL_DEVICE_GPU_HOURS_UNMEASURED` refuses the
   unmeasured case; a declared ceiling the run cannot measure stays an
   admission constraint on the projected plan and is recorded as such by the
   binding. See `DEVICE_CEILING_FAIL_CLOSED_ADDENDUM_2026-09-18.md`.
4. `ComputeCost` (device/wall pair, validated finite/non-negative) +
   `CycleCostLedger` producing `cycle_compute_accounting.json`: per-recipe
   device/wall, evaluation cost, failed attempts, baseline references at zero
   incremental cost, campaign totals; deterministic digest.
5. `GenerationLedger.adjudication_revisions`: append-only superseding
   adjudications referencing the original record's digest; effective verdict
   = latest revision, resolved deterministically. Original bytes unchanged.
6. Candidate selection: selection may consume target-smoke/protocol evidence
   only; a policy test pins that protected/broad final scores cannot influence
   selection.
7. Scoped verdicts: prefer the existing verdicts. `INCONCLUSIVE` +
   `target_repair_validated: true` in the decision payload expresses
   "real repair, insufficient full-promotion evidence" without a new status.

Legacy compat: rows without `measurement_origin` bind on the **parent** side
(as before) but are refused on the candidate side of a promotion input — the
one place where a copied row could change a verdict. Refusal names the fix
(record provenance), so nothing silently degrades.
