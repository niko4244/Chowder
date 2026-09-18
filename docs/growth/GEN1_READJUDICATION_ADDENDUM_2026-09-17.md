# Gen-1 Integrity Addendum — re-adjudication under the corrected policy (2026-09-17)

## What was originally recorded

Generation 1 (PR #171, `66be8ae`) was recorded **PROMOTED**: the candidate
 materially improved EOS termination (0.0 -> 1.0), max-token-cap behavior
(1.0 -> 0.0), thinking-block closure, and looping behavior under a
protocol-identical 16-prompt diagnostics instrument. That original ledger
record remains byte-identical on disk and in history; nothing in this
addendum rewrites it.

## What the integrity audit found

Post-merge audit (this branch) traced the actual code paths and found the
original adjudication had been fed two classes of defective evidence:

1. **Carried parent evidence satisfied candidate gates.** The Gen-0 frozen
   math500/mgsm rows were copied into the candidate side of the promotion
   input with the candidate's generation label. The promotion rule then
   certified `protected_regression: ok` and `broad_battery: ok` from rows
   that never measured the candidate — including a Welch t-test over
   identical copied arrays (manufactured statistical confidence) and a
   parent-floor shortcut where 0.0 meant "cannot regress".
2. **Budget settlement was incomplete.** Only the winning recipe's cost
   reached the resource gate; the losing recipe (0.397 wall GPU-h), both
   attempts' independent-evaluation cost, and the failed-attempt ledger were
   omitted. No actual-vs-ceiling comparison ran after training.

## What was implemented (this branch)

- `measurement_origin` provenance on every benchmark run/result
  (MEASURED_THIS_GENERATION / MEASURED_PARENT / CARRIED_REFERENCE /
  UNMEASURED); candidate-side binding refuses anything not actually
  measured on the candidate; promotion gates count only candidate-measured
  rows; identical copied arrays can no longer pair.
- Actual-cost settlement: post-run comparison of measured cost against the
  frozen device/wall ceilings and project budget, in the training binding
  and again at promotion; overruns refuse with machine-readable reasons
  (`ACTUAL_*_EXCEEDED`, `RESOURCE_OVERRUN`, `ACTUAL_EXCEEDS_PROJECTION`).
- `ComputeCost` (explicit device/wall units, validated) and the
  `CycleCostLedger` -> `cycle_compute_accounting.json` (deterministic
  digest; winning + losing recipes, evaluations, failed attempts,
  zero-incremental references).
- Append-only `GenerationAdjudicationRevision`s in the ledger (original
  bytes untouched; digest-bound supersession; deterministic effective
  verdict).
- Candidate selection from allowed evidence only; campaign manifests with
  admission (projected) and settlement (actual) controls;
  `chowder growth campaign validate|settle`.

## Re-adjudication result

`docs/gen1/re_adjudicate_gen1.py` was executed against the immutable
artifacts (adapter digest verified `ca8769c5...`, original record digest
`5622759a...` bound into the revision):

| | |
|---|---|
| Original verdict | PROMOTED (preserved, untouched) |
| Corrected effective verdict | **INCONCLUSIVE** |
| target_repair_validated | **true** (EOS delta +1.0, candidate-measured) |
| protected:math500 / protected:mgsm | inconclusive (never candidate-measured) |
| broad_battery | inconclusive (same evidence) |
| evidence_integrity | ok (contamination CLEAN) |
| resource_envelope | ok (actual 1.3526 wall GPU-h vs frozen 1.50 aggregate) |
| Revision | gen1-adjudication-001, appended to adjudication_revisions.json |

**Corrected status: Gen-1's protocol repair is real and validated, but
full-generation promotion is INCONCLUSIVE pending genuine candidate-side
protected evaluation.** Gen-1 remains a live candidate (not a promoted
generation); a future full evaluation of the same artifact under the same
policy can promote it if the protected gates hold on real measurements.

## Scientific path forward

1. Run the affordable candidate-side regression mini-battery (small fixed
   math500/MGSM-EN slices + termination diagnostics, frozen before
   execution) on the gen1 artifact.
2. If it passes without regression, a superseding adjudication (revision
   002) may promote gen1 — or the gen2 campaign may simply treat gen1 as
   its parent-branch point with the manifest declaring the evidence state.
3. The corrected policy guarantees whichever path is taken will be decided
   by candidate-measured evidence and settled costs only.
