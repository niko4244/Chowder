# Integrity pass report — candidate-measured evidence + actual-cost settlement (2026-09-17)

Mission: make promotion reachable only from **candidate-measured evidence** and
**actual measured compute**. This is the closing report for that pass; the
per-defect traces live in `docs/growth/INTEGRITY_ASSESSMENT_2026-09-17.md` and
the re-adjudication in `docs/growth/GEN1_READJUDICATION_ADDENDUM_2026-09-17.md`.

## Verdicts

| | |
| --- | --- |
| Original Gen-1 verdict | **PROMOTED** — preserved verbatim, never rewritten |
| Corrected effective Gen-1 verdict | **INCONCLUSIVE** with `target_repair_validated = true` |
| Original ledger record | untouched (`adjudication_revisions.json` appends, never mutates) |
| Re-adjudication | revision `gen1-adjudication-001`, digest-bound to the original |
| Gen-2 scientifically safe to start? | **Yes, and only on candidate-measured evidence** — the run path now refuses anything else. Gen-2's target and thresholds were preregistered before compute (PR #175) |

The correction is not a punishment for a narrow result. Gen-1's target repair is
real and candidate-measured: EOS termination 0.000 → 1.000 on the candidate,
cap-hit 1.000 → 0.000, unclosed-think 0.000, no loops, trigram ratio 0.9732,
diagnostics 3/16 vs the parent's 1/16. What could not stand was certifying
**protected** and **broad** gates from rows the candidate never produced.

## Defects fixed

1. **Carried parent evidence satisfied candidate gates.** Gen-0's frozen
   math500/mgsm rows were re-emitted with `generation_version="gen1"` and
   `support="SUPPORTED"`, so `protected_regression: ok` and
   `broad_battery: ok` were certified from rows that never measured the
   candidate — including a Welch t-test over identical copied arrays and a
   0.0-parent-floor "cannot regress" shortcut.
2. **Budget settlement did not exist.** `_check_cost` ran pre-launch on
   *projections* only; `_settle` verified registry/source/artifact and returned
   SUCCEEDED without ever comparing actual `measured_gpu_hours` to any ceiling.
   The judge saw one recipe's cost (losing recipe and both evaluations omitted).
3. **Zero-variance target pairs were permanently `inconclusive`.** A hard 0.0 →
   1.0 lift past a 0.90 threshold could not return `improved` because the
   no-variance branch of the comparison never handled the symmetric case.
4. **Aggregate-only protected rows were unverifiable.** A real aggregate
   measurement with no per-sample vector had no adjudication path at all.
5. **`MetricBinder.from_manifest` failed open on section-shaped input**
   (found by the repeatability probe, PR #173): feeding the `benchmarks`
   section instead of the whole manifest silently degraded to an empty
   contamination map → every row `UNKNOWN` → a plausible-but-wrong
   `inconclusive`. Now refused at the call site.

## What the implementation enforces

- **Measurement provenance** (`measurement_origin`: `MEASURED_THIS_GENERATION`
  / `MEASURED_PARENT` / `CARRIED_REFERENCE` / `UNMEASURED`) on every run and
  result. The binder refuses non-candidate-measured rows on the candidate side
  and carried rows on both sides; legacy rows bind on the parent side and
  refuse on the candidate side with the fix named. Missing origin refuses
  rather than guessing.
- **Promotion gates count only gate-eligible rows.** A relabeled parent row
  reads `inconclusive` — never `not-regressed` — and identical copied arrays can
  no longer pair, so no fake statistical confidence.
- **Actual-cost settlement** (`ComputeCost` with explicit device and wall
  fields, validated finite/non-negative; `settle_cost`): every successful train
  settles against the frozen per-recipe ceilings, the project budget and its own
  projection. Overruns refuse with `ACTUAL_DEVICE_GPU_HOURS_EXCEEDED` /
  `ACTUAL_WALL_GPU_HOURS_EXCEEDED` / `RESOURCE_OVERRUN` /
  `ACTUAL_EXCEEDS_PROJECTION`, artifact and measurements preserved, and the same
  fields ride into promotion so a blown ceiling vetoes **after** execution.
  *Corrected 2026-09-18:* the device ceiling was settleable against an
  unmeasured device figure, so it could pass without measuring anything; it now
  fails closed with `ACTUAL_DEVICE_GPU_HOURS_UNMEASURED`. See
  `DEVICE_CEILING_FAIL_CLOSED_ADDENDUM_2026-09-18.md`.
- **Cycle compute accounting** (`CycleCostLedger` → `cycle_compute_accounting.json`):
  winning and losing recipes, per-attempt evaluations, failed attempts, and
  zero-incremental historical references, with a deterministic digest.
- **Append-only corrections** (`GenerationLedger.append_adjudication_revision`):
  original bytes untouched, digest-bound, chained, deterministic
  `effective_verdict`.
- **Candidate selection** from training-side evidence only
  (`select_candidate`; no parameter accepts protected scores).
- **Campaign manifests** (`chowder growth campaign validate|plan|run|settle`):
  preregistration as configuration — paths in config, unit-named ceilings,
  unknown-field / unpinned-benchmark refusal, projected admission plus actual
  settlement. `run` composes the existing engine (`GrowthCycle`,
  `SubprocessTrainingFn`, `MetricBinder`, `settle_cost`) rather than a second
  cycle implementation, and every accepted field is listed in
  `campaign_runner.FIELD_ENFORCEMENT` (checked against the schema by
  `assert_every_field_enforced`), so an unimplemented key refuses instead of
  being ignored. Example: `docs/gen2_campaign_manifest.example.json`.
- **Scoped repair semantics**: `INCONCLUSIVE` + `target_repair_validated=true`.
  Deliberately **not** a fifth verdict: `PROMOTED` must keep meaning "complete
  evidence".

## Tests run (exact)

| Command | Result |
| --- | --- |
| `pytest -q` (full suite, final head) | **2203 passed, 77 skipped in 457.01s** |
| `ruff check src tests` | **All checks passed** |
| Build (`python -m build`, package) | succeeds |

The 77 skips are the environment's CPU-only matrix (GPU-gated tests); they are
**not** reported as passes. Targeted suites added or extended:

| New suite | Tests |
| --- | --- |
| `test_growth_measurement_provenance.py` | 12 |
| `test_growth_budget_settlement.py` | 12 |
| `test_growth_ledger_corrections.py` | 6 |
| `test_growth_campaign.py` | 8 |
| `test_growth_candidate_selection.py` | 5 |
| `test_growth_promotion_settlement.py` | 4 |

Plus updates to `test_growth_decisions.py`, `test_growth_metric_binding.py` and
`test_growth_training_binding.py`.

## Adversarial self-review

| Question | Answer |
| --- | --- |
| Can parent evidence still masquerade as candidate evidence? | No. Provenance is explicit and the candidate side refuses anything that is not `MEASURED_THIS_GENERATION`; `generation_version` alone decides nothing. |
| Can missing candidate measurements satisfy promotion? | No. Unmeasured is not zero and not passing; the row reads `inconclusive`. |
| Can a candidate exceed a frozen budget and still promote? | No. Settlement compares the actual cost against every ceiling it can measure -- wall and the project budget in wall units, plus the device ceiling when the run separated device time -- and vetoes promotion after execution while preserving the artifact. A declared device ceiling that cannot be measured is no longer *certified*: it is enforced against the projected plan at admission and recorded as such. |
| Can losing-candidate compute disappear from campaign accounting? | No. `CycleCostLedger` totals every recipe, every evaluation and every failed attempt. |
| Can device and wall GPU-hours be confused? | Not in the growth layer: `ComputeCost` carries both explicitly and validates them, and a wall figure can no longer pass a device ceiling. **Corrected 2026-09-18:** the original answer here was wrong -- a 0.0 device placeholder *was* compared against a device ceiling and passed it, because nothing recorded whether that zero was measured. `ComputeCost.device_measured` now carries that fact and `settle_cost` refuses the comparison when it is false. |
| Can a historical ledger verdict be silently rewritten? | No. Revisions append; the original record's bytes are unchanged and the effective verdict is a deterministic function of the file. |
| Can candidate selection peek at protected final scores? | No. `select_candidate` takes training-side evidence only — no parameter accepts protected or broad scores. |
| Can an unmeasured row become zero? | No. Absence stays `UNMEASURED`/`inconclusive` through the binder and the gates. |
| Can a copied array produce fake statistical confidence? | No. Identical copied arms were refused as provenance-invalid before the comparison runs. |

## Changed files (27)

- **Source**: `src/chowder/evals/result.py`; `src/chowder/growth/`:
  `compute_cost.py` (new), `campaign.py` (new), `promotion.py`,
  `metric_binding.py`, `training_binding.py`, `cycle.py`, `lineage.py`, `cli.py`.
- **Tooling/evidence**: `docs/gen1/re_adjudicate_gen1.py`,
  `docs/growth/INTEGRITY_ASSESSMENT_2026-09-17.md`,
  `docs/growth/GEN1_READJUDICATION_ADDENDUM_2026-09-17.md`,
  `docs/gen2_campaign_manifest.example.json`.
- **Docs truth**: `docs/HANDOFF.md`, `docs/ROADMAP.md`,
  `docs/MODEL_GROWTH_SYSTEM.md`, `docs/FIRST_MODEL_GROWTH_CAMPAIGN.md`,
  `docs/quals/GEN1_RESULT_2026-09-17.md` (addendum, original text preserved).
- **Tests**: `test_growth_measurement_provenance.py`,
  `test_growth_budget_settlement.py`, `test_growth_promotion_settlement.py`,
  `test_growth_ledger_corrections.py`, `test_growth_campaign.py`,
  `test_growth_candidate_selection.py`, plus three existing suites updated.

## Unresolved limitations

1. **Gen-1's protected/broad capability remains unmeasured.** The corrected
   verdict says so instead of implying otherwise. Measuring it now would be a
   *new* evaluation (regression qualification), not a retrofit of the original
   record — the distinction is deliberate.
2. **`math500`/`mgsm` Gen-0 zeros are protocol artifacts, not a capability
   floor** (28-item and 24-item lm-eval limits; see the frontier seed addendum).
   They must never be used as "cannot regress below 0.0" protection.
3. **The frozen Gen-0 mgsm row is internally inconsistent** — `n_samples: 24`
   with 48 per-sample scores, two passing, against an aggregate of `0.0`. The
   row cannot serve as an MGSM regression floor until reconciled. Recorded, not
   rewritten; no verdict depends on it.
4. **Production successive-halving / search-controller integration is still not
   wired** and remains explicitly non-blocking for v1: `select_candidate` is a
   deterministic, declared policy, and the library-level search controller has
   not been qualified for production.
5. **Frontier values still decide nothing**: every imported reference is
   `MEDIUM`/`LOW`, so 0 gate-eligible gap rows exist. Frontier gap work needs a
   first-party protocol measurement, not more imports.

## Gen-2 readiness

Safe to start, bounded, and only through the hardened path: the run must carry
candidate-measured target/protected/broad rows, settle its actual cost against
a preregistered envelope, and record its verdict append-only. Gen-2's
preregistration, campaign manifest and frozen judge are prepared in PR #175
(target: response-surface compliance, chosen from fresh Gen-1 measurement —
EOS is already repaired, so the earlier non-termination recommendation is
recorded as rejected by evidence).
