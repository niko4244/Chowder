# Proposal: amendment 16 to the frozen Gen-2 judge (gate T23) — **implemented, 2026-10-04**

**Status: PROPOSED AND IMPLEMENTED** (branch `feature/settlement-audit`, prereg
`GEN2_PREREG_AMENDMENT16_2026-10-04.md`). The proposal below is kept as the
record of what was asked for and why; acceptance of the amendment series is the
repo owner's call, and the change is isolated in its own commit so it can be
dropped without touching the audit that measured it.

## The defect, as measured

`tests/test_growth_settlement_adversarial.py` measures it on a real
`run_campaign` root. A campaign whose evaluation pushes it past its frozen wall
envelope records **REJECTED** with the machine reason
`ACTUAL_WALL_GPU_HOURS_EXCEEDED: actual wall 0.512000 > ceiling 0.200000`, and
the frozen judge, pointed at that same root, returns **REJECTED** with T13 FAIL
— the two authorities agree.

Then one file is edited: `cycle_compute_accounting.json`, whose
`totals.incremental.wall_gpu_hours` is set to `0.0`. The run's own record is
untouched — it still says `"verdict": "REJECTED"` and still records a
non-compliant settlement — and the frozen judge now returns **PROMOTED**, with
**every row PASSing and exit code 0**, including:

```
T13  actual cost settled within the declared ceilings  PASS  device 0.0000 (unmeasured) / wall 0.0000 against the campaign envelope
T21  the run's promotion decision agrees with the declared profile  PASS  the run refused this candidate without a declared-gate breach: actual wall GPU-h 0.5120 exceeds preregistered wall ceiling 0.2000
```

That is the split-brain condition amendment 15 closed for declared gates, one
artifact over: the judge certifies a candidate the branch had already refused,
on a root whose own record says so.

Root cause, statically pinned: T13 settles the accounting artifact *from
whatever bytes are in the run root*, and nothing compares those bytes with the
artifact the run settled. T21 reads the record but collects only
`RETENTION_`-prefixed reasons, so a resource refusal is the "refused without a
declared-gate breach" branch — a PASS whose detail names the refusal it is
passing. The anchor against tampering already exists and is simply unread:
`CycleCostLedger.write` stamps `digest_sha256` over the document it writes, and
`CampaignRun.to_dict` pins that digest at `cost.accounting_digest`.

## What "minimal" means here

The judge owns the frozen *policy* and delegates every mechanism to production.
This amendment therefore adds no arithmetic, no threshold and no verdict class:
it recomputes one digest through production's own function, calls the same
`settle_campaign` T13 already calls, and compares two artifacts the run already
wrote. `branch_verdict` already maps FAIL to REJECTED and UNKNOWN to
INCONCLUSIVE, and the exit code already treats anything but all-PASS as a
refusal, so **no change to the verdict algebra, the thresholds, or the exit
code is required**.

## The amendment

### 1. Four named reasons (next to the existing `RUN_RECORD_*` block)

```python
ACCOUNTING_ARTIFACT_MOVED = "ACCOUNTING_ARTIFACT_MOVED"
ACCOUNTING_UNPINNED = "ACCOUNTING_UNPINNED"
ACCOUNTING_UNSETTLED_BY_RUN = "ACCOUNTING_UNSETTLED_BY_RUN"
SETTLEMENT_DISAGREES_WITH_RECORD = "SETTLEMENT_DISAGREES_WITH_RECORD"
```

### 2. One production helper, in `chowder/growth/compute_cost.py`

The ledger's canonical form gets one owner instead of two. `CycleCostLedger.render`
already computed this digest; it now calls

```python
def ledger_digest(document: Mapping[str, Any]) -> str: ...
```

so the judge recomputes the writer's number rather than re-deriving it. A reader
that reimplemented the canonical form could drift from the writer, and then every
correctly pinned artifact would look moved.

### 3. One gate function (beside `_settlement_gates`)

`_settlement_artifact_gate` reads the same two artifacts T13 and T21 already
read — `campaign-run.json` and `cycle_compute_accounting.json` — and emits one
row, fail-closed in every branch:

| state | T23 |
| --- | --- |
| record absent, artifact absent/unreadable, no pin (`cost.accounting_digest` empty), no recorded settlement, unreadable campaign or totals | UNKNOWN |
| artifact digest ≠ pinned digest (`ACCOUNTING_ARTIFACT_MOVED`) | FAIL |
| recorded `budget_compliant` ≠ the artifact's recomputed settlement (`SETTLEMENT_DISAGREES_WITH_RECORD`) | FAIL |
| artifact is the pinned one and settles as the record says | PASS |

### 4. One wiring line (in `judge()`, after `_settlement_gates`)

```python
    _settlement_artifact_gate(verdict, run_root, campaign)
```

### 5. Three documentation lines

- the module docstring's amendment list gains amendment 16 and T23;
- the docstring's run-record paragraph gains the T23 paragraph;
- the `INFO` frozen-policy row gains `+ GEN2_PREREG_AMENDMENT16_2026-10-04`.

## Governance: what amending the frozen judge required

1. **A named prereg amendment**, in the same series as amendments 1–15:
   `docs/quals/GEN2_PREREG_AMENDMENT16_2026-10-04.md`. The judge is frozen
   *with* the prereg; it is amended by amending the prereg, in writing, before
   the change. **Done.**
2. **The no-visible-candidate-results rule.** The freeze says thresholds may not
   change after candidate results are visible. Both admissibility conditions
   hold, and the reviewer should confirm both:
   - it changes **no threshold** — the gate compares a digest and a settlement
     verdict, both of which the run already wrote, and adds no numeric policy;
   - no Gen-2 candidate evaluation exists: the run refused at the
     `candidate_evaluation` phase, and the state root holds no
     `candidate_evaluation.json` or `chosen_candidate.json`.
3. **A test that fails without it.** `tests/test_growth_settlement_adversarial.py`
   contains the measured attack as a passing test, and the amendment's own
   ignore-source proof was run in both directions:
   - **gate unwired**: the tamper test, the re-pinned-artifact test and the
     T13-agreement control all fail on `assert final != "PROMOTED"`;
   - **digest clause disabled**: the tamper test fails on
     `ACCOUNTING_ARTIFACT_MOVED`, because the agreement clause refuses the same
     root for the same reason one step later — the two clauses are independent,
     which is the point of having both.

## What the amendment deliberately does NOT do

- It does not recompute the campaign's spend from the accounting *entries*.
  The digest pin makes any edit to either the entries or the totals detectable,
  which is the stronger property, and re-summing in the judge would be a second
  implementation of `CycleCostLedger.total`.
- It does not stop a tamperer who rewrites the artifact, re-pins the digest in
  the record *and* edits the record's settlement, verdict and reasons to agree.
  A fully consistent forgery of every artifact in the run root is beyond any
  check that does not sign the root; the same limit applies to the declared-gate
  mutations amendment 15 added.
- It does not make `_ceiling_enforcement` a judged artifact. The record's claim
  about which unit each ceiling was settled in stays reporting; what gates is
  the settlement itself (T13) and its identity (T23).
