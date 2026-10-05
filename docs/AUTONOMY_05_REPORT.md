# AUTONOMY_05 — the 0.5 promotion-gate architecture: current report

Merged into `main` at ef857da (PR #206; branch
`feature/autonomous-05-architecture`). This file is the single current
report; it replaces the three overlapping documents (implementation
report, limitations, next-action). Every claim is backed by an executable
test named alongside it.

## The enforced contract (what the architecture guarantees today)

1. **A survivor earns a larger budget only by continuing its own lineage.**
   Checkpoint-resuming progressive halving (`candidate_search.run_search`):
   survivors carry checkpoints, missing checkpoints stop the lineage,
   executor-reported non-resumes disqualify it, and an interrupted search
   resumes only onto a validated rounds-prefix.
2. **A declared search axis is consumed by a real reader or refused.** The
   search-axis contract (`recipe_planner.classify_recipe_fields`,
   `assert_search_axes_consumed`) is checked against the backend readers'
   actual source.
3. **Families train only under a maturity their evidence earned.** Evidence
   is scoped and hash-chained (`evidence.EvidenceStore`); priors starve
   falsified directions without ever auto-promoting; a candidate begins as a
   falsifiable hypothesis (`hypotheses.py`).
4. **The search cannot see the promotion gate.** Three evidence tiers; a
   reserved protected-suite name refuses demotion at declaration and read
   time; `assert_search_isolation` refuses tier-3 names in any
   search-readable surface.
5. **Declared retention constraints are enforced fail-closed on both
   promotion paths.** `GrowthCycle._apply_promotion_gates` runs in
   `decide_promotion` and `decide_promotion_from_runs`: an unmeasured
   constraint is a violation, never a pass, and a PROMOTED verdict over a
   violation is downgraded to REJECTED with the machine codes
   `RETENTION_REGRESSION` / `RETENTION_FLOOR` / `RETENTION_UNMEASURED`
   (`RetentionViolation.code`, owned in `retention.py`). Every violation's
   reason reaches the record whatever the incoming verdict: a declared gate
   that fires on a candidate the predeclared protected arithmetic already
   rejected is still a fact about that candidate, and only the verdict is
   tightened -- PROMOTED becomes REJECTED, while REJECTED, TAINTED and
   INCONCLUSIVE keep the verdict the predeclared rule earned, because a
   declared breach must not manufacture a strong verdict over evidence the
   rule found too thin to decide.
6. **Settlement refusals stop advancing.** `compute_cost.settlement_refusal`
   reads exactly the two shapes production writes — the `budget_settlement`
   verdict (`training_binding.py`) and the `refused_by`/`refusal_reason`
   stamp (`_finish`) — `run_search` ends the lineage, `advanced` and
   `cycle.select_candidate` refuse, and `attempt_failure.classify_failure`
   derives from the same predicate, so classifier and runner cannot
   disagree.
7. **The manifest binds the gates.** A campaign manifest may declare
   `retention_profile` and `eval_tier_policy`; both parse fail-closed at
   load (`campaign.py`: unknown fields, empty profiles, malformed
   constraints, unmeasurable constraints, unknown tiers, reserved-name and
   search-view demotions refuse as `CampaignManifestError` naming the
   source) and reach `CycleConfig` as the same objects programmatic
   construction would pass.
8. **Both sides of a declared constraint demand earned provenance.** The
   candidate side reads only rows measured on this generation
   (`gate_eligible`); the parent side only rows measured on the parent arm
   (`parent_measured` in `promotion.py`). A carried reference is a quotation
   from history: the binder refuses it parent-side, and the gate's wall
   makes the caller-passed `decide_promotion` seam agree — an unearned row
   reads as unmeasured, never a silent compare against a borrowed number.
9. **Budget adaptivity reorders inside the declared envelope.** The ladder
   (`budget_ladder.py`) reorders survivors; ceilings, budgets and the
   per-round schedule are never the ladder's to change.

## The corrections the audits forced (each owned by its commit)

- **7e1b63e — the gates were wired into the runner.** Wiring caught the
  predicted second instance of the disease: the campaign fixtures' "clean"
  runner reported 8x its projection and had only ever promoted because
  selection ignored settlement refusals. The fixtures now report settleable
  costs, the overruns are explicit, and the scenarios that relied on the
  hole assert the honest refusal path.
- **3e3f9cb + 6e67427 — one owner of the refusal vocabulary, honestly
  claimed.** The phantom `settle_refusal` arm was deleted; the predicate
  reads exactly the two shapes production writes. 3e3f9cb's own "three real
  vocabulary sources" claim was false for one of three (a `settlement_failed`
  key only its test wrote); 6e67427 owned the correction explicitly, and the
  `settle_refusal` local and both docs now speak the real vocabulary.
- **df08a7f — the parent side earns its baseline.** `_retention_values`
  trusted any non-`UNMEASURED` parent row; now `parent_measured` (owned
  beside `gate_eligible`) demands earned provenance on the caller-passed
  seam too. The pin test caught a latent mislabel: a missing parent
  measurement was coded `RETENTION_REGRESSION` (the candidate's score rode
  along as `measured`); it is now `RETENTION_UNMEASURED`, as the vocabulary
  defines.

## Verification

- Gate suites: `tests/test_growth_runner_gates.py` (22 — predicate
  vocabulary, advance/selection/search refusals, classifier agreement,
  retention/tier gates on both paths, carried-baseline and binder-path
  pins), `tests/test_growth_manifest_promotion_gates.py` (9 — loader
  refusals, ownership proof, unchanged-default control),
  `tests/test_growth_attempt_failure.py`.
- Growth regression: `python -m pytest tests/test_growth_*.py -q` — 700
  passed at df08a7f. Full suite: 2,790 passed / 77 skipped / 0 failed.
  Ruff clean on `src/chowder/growth/` and `tests/`.
- Driven directly (not only via the suite): the loader wrap with source
  context, a real `_decide` retention rejection, the carried-parent
  scenario through `decide_promotion`, and the predicate's two arms against
  the production shapes.

## Not wired yet

1. **The budget ladder's adaptive rung has never seen a real search.** The
   UCB1 ordering is deterministic and tested; its reward (training-side
   efficiency per GPU-hour) has no production definition, so campaigns start
   DETERMINISTIC by construction and the ladder escalates only when the
   evidence store says so.

## Proven only at small scale

2. **Checkpoint-resuming progressive halving has never run against a real
   GPU.** The E2E test composes a round-1 project that genuinely carries the
   round-0 checkpoint, but the composed loop has no Kaggle/GPU execution
   behind it; the continuation projection is a floor (restore cost
   unmodeled).
3. **Measured costs come from tiny screening runs** (0.008 device-GPU-h at
   300 steps on 2xT4); extrapolation to 1200-step rounds assumes linear step
   cost. The campaign design's falsification clause pauses on 2x overrun.
4. **The evidence store holds one real record** (Run 4's falsified
   replay-decay). Every prior multiplier is a documented starting point, not
   a fitted constant; the value is the policy, not the numbers.
5. **The families registry is declarative.** None beyond the SFT trio has
   in-scope evidence; the RESEARCH six are refused by default.
6. **EI is deliberately not a search-time policy** (its reward would breach
   the tier wall); it remains a campaign-layer selector.
7. **The Kaggle backend bar (R1-R6) is a contract, not code.** PR #200's
   dispatch is green for smoke work; the production bar gates campaign use.
8. **The full local suite exercises no GPU.** Real-ML tests skip locally;
   cpu-smoke proof lives in CI.
9. **The ambient editable install points at a stale clone**
   (`C:\Users\nikma\Chowder`); every command must pin the worktree.
10. **PR #201's CI is failing and untriaged; #202 conflicts with main.**
    Outside this arc's scope; they block the teacher-free lane and are a
    merge-order prerequisite for it.

## The judge gap, measured (2026-10-04)

The frozen `docs/gen2/judge_gen2.py` was measured, not changed, against a
real `run_campaign` root whose manifest declares the protection and whose
candidate breaches only that gate (an absolute floor, the one declared shape
the predeclared rule has no check for). `tests/test_growth_gen2_judge_agreement.py`
records the fact:

- The run **REJECTS** with `RETENTION_FLOOR: candidate 0.5 is below the
  absolute floor 0.5625` -- the declared gate, enforced through the seam the
  architecture wired.
- The frozen judge, pointed at the same declaration and reading the same run
  root, returns **INCONCLUSIVE** with **every gate it owns PASSING on the
  very evidence the run rejected**: the protected slice, the
  candidate-vs-parent regression, the trusted-ancestor protection, the
  identity chain (T19), settlement (T13), the recipe accounting (T14), and
  the judged contamination evidence. The declared gate the run enforced does
  not appear anywhere in the judge's record; the judge reads no run decision
  and no `retention_profile` (pinned statically).
- The INCONCLUSIVE comes only from the instrument gates (T1-T10 plus the
  fixture's sourceless contamination row) being UNKNOWN on a synthetic
  candidate arm. A real Gen-2 arm carrying the instrument metadata would
  leave certification **reachable on a root the run rejected** -- that is
  the gap, stated at its sharpest.
- Attribution: judged against the real frozen manifest instead of the
  fixture's, the root additionally fails bookkeeping the fixture cannot
  carry (the real base/adapter digests, recipe ids, contamination pin) --
  deployment mismatch, not a fact about the judge's gates.

### The minimal amendment: proposed, then implemented (amendment 15)

`docs/gen2/JUDGE_AMENDMENT_PROPOSAL_T21.md` states the proposal with its exact
insertion points; **implemented** on `feature/judge-retention-coupling` as prereg
`GEN2_PREREG_AMENDMENT15_2026-10-04.md`, which shipped T21 *and* T22 and, in
doing so, caught a defect in amendment 14.

**T21 -- the run's recorded decision.** The judge opens one more artifact from
the directory it already reads: `campaign-run.json`, the run's own record. A
candidate the run refused on a declared gate is `FAIL`
(`DECLARED_GATE_REJECTED_RUN`) -- this judge audits no declared gate, so its
table cannot overturn a refusal the branch already recorded. Also `FAIL`: a
record promoting a candidate it simultaneously recorded breaching, and a breach
on a constraint the declaration does not name. `UNKNOWN` for a record that is
absent, belongs to another cycle, or carries no decision (a run refused before
adjudicating). Reasons are classified by production's own codes
(`RetentionViolation.code`, read off the owner via `RETENTION_CODES`), never by
parsing prose.

**T22 -- the judge's own recomputation.** Reading a record still leaves two
answers to compare by eye, so the judge also *recomputes* the declared profile
through production's own evaluator (`evaluate_retention`) and production's own
provenance filter (`retention_values`, renamed from the private
`_retention_values` in `cycle.py` so the judge and the promotion path call one
function), on the arms it already audited. Recomputed codes must equal recorded
codes, or the gate is `FAIL` (`RETENTION_RECOMPUTATION_DISAGREES`). Nothing is
reimplemented: the declaration owns the constraint, production owns the
comparison, the judge owns the agreement. `branch_verdict`, the exit code and
T1-T20 are unchanged; a run root with no record can no longer certify, which is
the intended fail-closed cost.

**What the coupling immediately found: a sign error in amendment 14.** A
`max-regression` constraint's `value` is the *minimum acceptable
candidate-vs-parent delta*, so a permitted dip is declared **negative** (pinned
by production's `test_small_declared_dip_within_budget_passes`, `-0.02` permits
a -0.015 dip). Amendment 14 wrote `+0.0625` and called it the frozen
`slice_regression_max`; that requires the candidate to **improve** by one
sixteenth on both protected benchmarks rather than permitting a one-sixteenth
regression -- stricter than the frozen protection rule the judge enforces, and
with Gen-1 at 0.0 on both slices it would have rejected a candidate the frozen
branch-protection rule accepts. Both constraints are now `-0.0625`, and
`test_the_declared_retention_profile_states_the_frozen_tolerance_with_the_right_sign`
pins both the value and its meaning (a dip of exactly the tolerance passes; one
hundredth past it is `RETENTION_REGRESSION`).

The sharpest proof is a root whose instrument gates T1-T10 are **all decided**
-- the case the original gap predicted would certify -- on which T21 is the only
failing row. And reverting only the judge makes
`tests/test_growth_gen2_judge_agreement.py` fail with the original measurement
restated: `the judge's verdict on a run-rejected root changed: INCONCLUSIVE`.

## The Gen-2 pre-compute state (2026-10-04)

`docs/gen2/gen2_campaign.json` declares its preregistered protection
mechanically (`GEN2_PREREG_AMENDMENT14_2026-10-04.md`; the tag first written
into `notes` collided with the real `GEN2_PREREG_AMENDMENT7_2026-09-18` and is
corrected): `retention_profile` (`gen2-protection`: max-regression **-0.0625**
-- a *permitted* dip of one 16th of a 16-item mini-slice, exactly
`-(protection.slice_regression_max)`, signed per amendment 15 -- on
`math500@2024-04` and `mgsm@2022-11`) and `eval_tier_policy` (both protected
benchmarks are promotion-evidence).

**The seven previously-undeclared inputs now exist, produced by production
code, and readiness is fully green.** `chowder growth campaign prepare` wrote
them into one directory (`<state_root>/prepared-v10`, never over `prepared-v2`,
which holds the preserved measurement), and the declaration names them at
`prepared_input_paths`' predicted paths plus the planner's own recipe ids
(`recipe-00-lr5e-05`, `recipe-01-lr0.0001`) in place of the placeholders the
planner had never proposed. Measured, not asserted --
`check_campaign_readiness` reports `READY`, `reason_codes: []`, and **all
seventeen checks `ok`** (321 s, most of it the base model-content digest over
ten payload files):

| check | detail |
| --- | --- |
| `schema`, `declared_inputs` | FIELD_ENFORCEMENT; every run-phase input declared |
| `base_identity`, `parent_adapter_identity` | `59e767aab1da` over 10 payload files; `ca8769c5e7e0` verified over base |
| `contamination`, `project_template`, `training_material`, `data_registry`, `hardware_budget` | each parses from the prepared bundle |
| `parent_profile` | gen1 `SkillProfile` in the attributed `estimates` shape (the stale legacy shape that blocked `prepared-v2` is not reproduced) |
| `parent_arm`, `ancestor_arm` | 3 gen1 rows, 2 gen0 rows |
| `protection_policy` | gen0 branch protection, tolerance 0.0625, 16 items |
| `plan`, `recipe_set`, `candidate_search`, `campaign_projection` | 5 items -> 2 recipes; 0.000312 device / 0.001091 wall GPU-h within the ceilings |
| `evaluator`, `evaluator_coverage` | `SubprocessEvaluationFn` available; covers every declared benchmark |

Where the evidence came from, since "readiness is green" is only worth what its
inputs are: the hardware budget is a real CUDA device probe on this machine
(RTX 5060 Ti; bounded synthetic step timings, and its own `measurement_method`
says it is not a measurement of the campaign's model); the evaluation material
is the pinned offline `HuggingFaceH4/MATH-500` and `juletxara/mgsm` caches plus
the in-repo instrument prompts; the corpus (10,340 examples, 79,904 tokens,
`verifier_pass_rate` 1.0, `duplicate_rate` 0.0) and its registry and project
template come from production planning; and the parent arm is the **preserved**
2026-09-19 Gen-1 measurement carried forward through `--parent-measurement`,
row-for-row identical including `artifact_sha256` and `slice_sha256` -- not
re-measured, not a carried quotation. No path, digest or measurement was
invented; the two inputs that would have needed new GPU measurement already had
their measurements on disk.

Starting the run is still a separate decision, and two facts are unchanged: no
Gen-2 candidate evaluation exists (the run refused at the `candidate_evaluation`
phase), and the target instrument's diagnostic metadata -- the judge's T1-T10 --
still lives only in the historical Gen-1 driver.

## The settlement audit (2026-10-04)

The promotion matrix had no analogue for the resource envelope, so the audit now
covers the settlement and compute-cost paths too:
`tests/test_growth_settlement_adversarial.py` (25 tests) in the same
three-layer shape as the promotion audit, one mutation at a time.

**The rule (`settle_cost`).** An unmeasured device figure never settles a
declared device ceiling while an *observed* zero does -- the attempts 07/08
unit-confusion, one layer down; each ceiling is settled only in its own unit;
every breach is reported and none is downgraded to a warning; a `0.0` ceiling is
a ceiling and only `None` removes the control; the project budget is wall-charged
and is its own control; the projection tolerance is directional, and a zero
projection is an absent basis rather than a licence past a declared ceiling.

**The campaign contract and its ledger.** The declared device ceiling settles
only when the declaration says device time is measured, and the wall envelope is
never demoted; `_ceiling_enforcement` must not name a device settlement the
ledger cannot support; the total is the sum of its incremental entries, so one
unmeasured contributor makes the whole total an estimate (and the declared
device ceiling unsettleable) while a zero-cost reference neither costs nor
demotes the measurement; and the ledger digest is recomputable from the bytes.
Then whole `run_campaign` roots under the frozen judge.

**What the audit found: the settlement analogue of the judge gap.** A run
refused on its own frozen envelope (`ACTUAL_WALL_GPU_HOURS_EXCEEDED`, 0.5120
wall against the declared 0.2000) is certified **PROMOTED** -- every row
PASSing, exit code 0 -- after one file is edited:
`cycle_compute_accounting.json`'s incremental totals set to zero. The run's
record is untouched and still says `REJECTED`; T21 passes it under "refused
without a declared-gate breach" (it collects only `RETENTION_` reasons) while
T13 recomputes compliance from the edited bytes. The anchor existed and was
unread: the ledger stamps `digest_sha256` and `CampaignRun.to_dict` pins it at
`cost.accounting_digest`.

**Amendment 16 (T23).** `docs/gen2/JUDGE_AMENDMENT_PROPOSAL_T23.md` plus prereg
`GEN2_PREREG_AMENDMENT16_2026-10-04.md`. T23 reads the same two artifacts T13
and T21 already read and emits one row: the artifact's digest, recomputed
through production's `ledger_digest` (extracted from `CycleCostLedger.render`,
so the writer keeps the canonical form), must equal the digest the record
pinned, and the artifact's own settlement must agree with the recorded one. An
absent pin, absent settlement, unreadable artifact or unreadable campaign is
UNKNOWN; a moved artifact or a disagreement is FAIL. No threshold, no verdict
class and no exit-code rule moves. Both revert proofs were measured: with the
gate unwired the attack certifies, and with only the digest clause disabled the
agreement clause refuses the same root.
