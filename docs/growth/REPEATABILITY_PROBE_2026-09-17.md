# GrowthCycle Repeatability Probe — Result (2026-09-17)

Question: can a second cycle start **strictly from the durable gen1 records**
(GenerationLedger entry, Gen-0 freeze, attempt evidence, evaluation report)
with no manual reconstruction — and if anything breaks, what exactly?

Method: `docs/growth/repeatability_probe.py`. It reconstructs, in order:
parent identity → adapter artifact (re-hashed) → capability profile →
curriculum → recipes from measured physics → contamination manifest →
MetricBinder → full promotion replay → probe-local ledger record. Every
input path is read from a durable record; no value was hand-carried.

**Verdict: PASS — the cycle is repeatable. The replay reproduced the
recorded PROMOTED verdict check-for-check from durable rows alone.**

## What broke on the way (the actual deliverable)

| # | Seam | What happened | Class | Disposition |
|---|---|---|---|---|
| R3 | CapabilityProfile → target selection | gen1's behavioral target (EOS 1.0) lives in the diagnostics instrument, which the catalog excludes from skill aggregation; the rebuilt profile carries only the carried-floor math/mgsm rows. A real gen2 target needs a fresh behavioral measurement. | GAP-ACCEPTED | Recorded honestly; recorded, not faked |
| R5 | HardwareBudget → RecipePlanner | Durable evidence holds a measured step cost at seq 512; the planner's projection grid hardcodes seq_len=2048. Interpolating would fabricate physics. | GAP-ACCEPTED | Probe registers the measured point under the planner's key and says so; prereg template should record the seq of measured points |
| R7 | Resource envelope input | `training-evidence.json['measured_gpu_hours']` already aggregates train + independent eval (0.4325 = 0.173 + 0.260). No reconstruction needed. | PASS | Confirmed durable |
| R8 | Generation pinning | Probe's first run fed gen0/gen1-labeled rows into a `gen2-probe` cycle: the binder **refused every row wrong-generation** (correct behavior — the labels were the probe's bug). Replay pins gen0→gen1 and reproduces the verdict. | PASS | Confirmed fail-closed |
| R9 | Instrument metric identity | The row's declared metric (`accuracy`) is not the promotion metric (`eos_termination_rate` from row metadata); the prereg disambiguates, so replay is possible — but a naive bind yields delta 0.1875 → REJECTED. | GAP-ACCEPTED | Prereg template should pin the primary metric on the row itself |
| R10 | Metric-name mapping | Freeze rows record the harness name (`exact_match`); the registry declares `accuracy`. The binder refuses undeclared names, so verbatim replay fails; the mapping convention was recoverable only from the original driver source. | GAP-ACCEPTED | Mirrored with mapping note; same fix family as R9 |
| R11 | `MetricBinder.from_manifest` | **The one genuine fail-open defect.** Passing the manifest's `benchmarks` *section* (instead of the whole file) silently built a binder with no contamination verdicts → every row UNKNOWN → honest-but-misleading `inconclusive`, with no error. | **GAP → FIXED** | Production guard added (fail-closed) + 2 regression tests |

## The R11 fix (fail-closed)

`from_manifest` now refuses section-shaped input (keys look like
`name@version`) with a message naming the contract, and its docstring states
the whole-file requirement. Regression tests:
`test_from_manifest_refuses_benchmarks_section_instead_of_whole_manifest`,
`test_from_manifest_accepts_whole_manifest`.

Why it matters: the degraded binder does not crash — it produces a plausible
inconclusive verdict that a human would be tempted to read as "insufficient
evidence" rather than "caller bug". A fail-open path into the promotion rule
is exactly the class of defect the repeatability probe exists to surface.

## Verification

- Probe end-to-end: RESULT PASS — 5 findings, 0 blockers; replay verdict
  PROMOTED with `target=met`, `evidence_integrity=ok`, `protected_regression=ok`,
  `broad_battery=ok`, `resource_envelope=ok` — identical to the ledger record.
- Adapter artifact re-hashed from the ledger record matches
  `ca8769c5e7e06575…` without touching the training machine state.
- Focused: binding/decisions/training-binding/contamination 114 passed.
- Full suite: 2156 passed, 77 skipped. Ruff clean.

## Residuals (recorded, non-blocking)

1. Prereg template: pin the promotion metric name on the eval row itself (R9/R10).
2. Prereg template: record the sequence length of measured step-cost points (R5).
3. Gen-2 target selection requires a fresh behavioral measurement; the profile
   aggregation intentionally excludes the diagnostics instrument (R3).
