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
   (`RetentionViolation.code`, owned in `retention.py`).
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
  deployment mismatch, not a fact about the judge's gates. Whether to amend
  the frozen judge (it predates the declared profiles) or ship Gen-2 with
  this documented is a product decision, deliberately not taken here.

## The Gen-2 pre-compute state (2026-10-04)

`docs/gen2/gen2_campaign.json` now declares its preregistered protection
mechanically (GEN2_PREREG_AMENDMENT7_2026-10-04): `retention_profile`
(`gen2-protection`: max-regression 0.0625 -- one 16th of a 16-item
mini-slice -- on `math500@2024-04` and `mgsm@2022-11`) and
`eval_tier_policy` (both protected benchmarks are promotion-evidence). The
loader accepts it, and `check_campaign_readiness` reports `schema: ok`,
`base_identity: ok`, `parent_adapter_identity: ok`, and
`protection_policy: ok`.

**Exactly what still blocks `run_campaign` -- 7 declared inputs
(`READINESS_DECLARED_INPUT`):**

1. `project_template_path` -- the executor has no project to compose.
2. `training_material_path` -- the executor writes the corpus this run
   trains on.
3. `data_registry_path` -- nothing may train on an unadmitted source.
4. `hardware_budget_path` -- recipes are projected against measured
   hardware, never guesses.
5. `parent_profile_path` -- a curriculum cannot be planned from nothing
   (needs a Gen-1 measurement; the Gen-0 freeze profile is not the Gen-1
   profile).
6. `evaluation_material_path` -- the production evaluator has no data to
   measure the selected candidate on (the run would spend its training
   compute and refuse afterwards).
7. `parent_eval_report_path` -- the promotion rule compares the candidate
   against its parent; without that arm the run can only reach
   INCONCLUSIVE.

Every downstream check (contamination, evaluator, plan, projection) reports
`skipped` until these exist. Producing them from production code, plus
measuring the declared Gen-0 baseline arm, is the remaining pre-compute
build; starting the run itself is a separate decision.
