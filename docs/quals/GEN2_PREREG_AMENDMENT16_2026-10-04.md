# GEN2 prereg amendment 16 — 2026-10-04: the settlement artifact pin (judge gate T23)

Additive to the 2026-09-17 prereg and its amendments 1–15. Admissible under the
freeze on the same grounds amendment 15 stated: **no Gen-2 candidate evaluation
exists** (`runs/2026-09-17-gen2-response-surface/attempts/*` hold training
evidence only; the state root has no `candidate_evaluation.json` or
`chosen_candidate.json`, and the run refused at the `candidate_evaluation`
phase), **and no threshold moves** — the gate compares two artifacts the run
already wrote and adds no numeric policy.

## A. The defect, measured

`tests/test_growth_settlement_adversarial.py` runs one real `run_campaign` root
and then judges it twice over. The campaign's evaluation spend pushes the run
past its frozen wall envelope, and the run records **REJECTED** with

```
ACTUAL_WALL_GPU_HOURS_EXCEEDED: actual wall 0.512000 > ceiling 0.200000
```

On the untouched root the frozen judge returns **REJECTED** (exit 1) with T13
FAIL — the judge and the run agree, which is the coupling T13 was built for.
Then exactly one file is edited, `cycle_compute_accounting.json`, whose
`totals.incremental.wall_gpu_hours` is set to `0.0`. The run's own record is
left alone: it still says `"verdict": "REJECTED"`, still pins the digest of the
bytes it wrote, and still records a non-compliant settlement. The judge now
returns **PROMOTED**, with every row PASSing and exit code 0:

```
T13  actual cost settled within the declared ceilings         PASS  device 0.0000 (unmeasured) / wall 0.0000 against the campaign envelope
T21  the run's promotion decision agrees with the declared profile  PASS  the run refused this candidate without a declared-gate breach: actual wall GPU-h 0.5120 exceeds preregistered wall ceiling 0.2000
```

This is the split-brain condition amendment 15 closed for declared gates, one
artifact over. T21's PASS branch for a rule-refused run is deliberate — the
judge audits the predeclared rule itself (T11/T16/T17) and may legitimately
disagree with it — but the resource envelope is **not** unaudited: T13
recomputes it. Two authorities over one number, with the recomputation reading
bytes nothing ties to the run that settled them.

Root cause: T13 settles the accounting artifact from whatever is in the run
root, and T21 collects only `RETENTION_`-prefixed reasons, so a resource
refusal falls into "refused without a declared-gate breach" — a PASS whose
detail names the refusal it is passing. The anchor existed and was unread:
`CycleCostLedger.write` stamps `digest_sha256` over the document it writes, and
`CampaignRun.to_dict` pins that digest at `cost.accounting_digest`.

## B. The gate: T23

One function, `_settlement_artifact_gate`, over the two artifacts T13 and T21
already read — `<run_root>/campaign-run.json` and
`<run_root>/cycle_compute_accounting.json`. It emits one row, fail-closed in
every branch and with no arithmetic of its own:

| state | T23 |
| --- | --- |
| record absent, artifact absent/unreadable, no pin (`cost.accounting_digest` empty), no recorded settlement, unreadable campaign or totals | `UNKNOWN` |
| artifact digest ≠ pinned digest | `FAIL` — `ACCOUNTING_ARTIFACT_MOVED` |
| recorded `budget_compliant` ≠ the artifact's recomputed settlement | `FAIL` — `SETTLEMENT_DISAGREES_WITH_RECORD` |
| artifact is the pinned one and settles as the record says | `PASS` |

The digest is recomputed through **production's own writer**, not a second
implementation: `compute_cost.ledger_digest(document)` is extracted from
`CycleCostLedger.render`, which now calls it. A reader that re-derived the
canonical form could drift from the writer, and then every correctly pinned
artifact would look moved. The settlement is production's `settle_campaign`
over `ComputeCost.from_dict(totals.incremental)`, the same call T13 makes.

`ACCOUNTING_UNPINNED` and `ACCOUNTING_UNSETTLED_BY_RUN` cover the run that
refused before settlement: it pins nothing, so nothing can be read out of its
root — UNKNOWN, never a pass, exactly as `RUN_DECISION_ABSENT` treats a run
refused before adjudication.

## C. What did not move

`branch_verdict`, the exit code, T1–T22 and every threshold are unchanged. No
campaign budget, protocol, recipe, protection constant or declared value moves;
`docs/gen2/gen2_campaign.json` gains a paragraph in `notes` recording this
amendment and nothing else. The gate adds a failure mode for an artifact that
does not match the run that settled it; a clean run's rows are unchanged
(asserted as a control, so the gate is not a blanket refusal).

## D. Proof

- `tests/test_growth_settlement_adversarial.py` (25 tests) is the audit that
  measured the defect: the settlement rule one mutation at a time, the campaign
  contract and the ledger, then whole `run_campaign` roots under the frozen
  judge. The attack above is
  `test_the_judge_never_certifies_a_root_whose_accounting_artifact_moved`; the
  re-pinned variant (artifact *and* digest edited together, record's settlement
  left saying non-compliant) is
  `test_the_judge_never_certifies_a_settlement_the_record_disagrees_with`; the
  controls are `test_the_judge_settles_the_same_artifact_the_run_did` and
  `test_the_judge_still_certifies_the_settlement_of_a_clean_run`.
- **Revert proofs, both measured.** With the gate's wiring removed, those three
  judge tests fail on `assert final != "PROMOTED"` (the judge promoted a root
  the run refused). With only the digest clause disabled, the tamper test fails
  on `ACCOUNTING_ARTIFACT_MOVED` because the agreement clause refuses the same
  root one step later — the two clauses are independent, which is why both are
  kept.
- `tests/test_growth_gen2_judge.py`'s run-root fixture now attests every record
  it writes (`_attest_record`): the pin and the settlement claim are computed
  from the artifact the fixture wrote, through production's own functions, so a
  fixture root cannot carry a record a real run would not write. 13 tests in
  that file would otherwise have become INCONCLUSIVE on roots they intend to be
  certifiable.
- The write→pin→judge path is asserted end to end by
  `test_the_record_is_the_settlement_of_the_ledger_the_run_wrote`: the
  record's `cost.accounting_digest`, the artifact's `digest_sha256` and a
  recomputation through `ledger_digest` are one number.

## E. What this amendment does not do

- It does not re-sum the ledger's entries in the judge. The pin makes any edit
  to either the entries or the totals detectable, which is the stronger
  property; re-summing would be a second implementation of
  `CycleCostLedger.total`.
- It does not detect a tamperer who rewrites the artifact, re-pins the digest
  *and* edits the record's settlement, verdict and reasons to agree. A fully
  consistent forgery of every file in the run root is out of reach of any
  check that does not sign the root, and the same limit applies to the
  declared-gate mutations amendment 15 added.
- It does not make `_ceiling_enforcement` a judged artifact: the record's
  claim about *which unit* each ceiling was settled in stays reporting. What
  gates is the settlement (T13) and the identity of the artifact it was
  computed from (T23).
- It does not change what settlement itself certifies. The device ceiling is
  still admission-only when the declaration says `device_time_measured=false`,
  and is still settled against measured device time when it says true.
