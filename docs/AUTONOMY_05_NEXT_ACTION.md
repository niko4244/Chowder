# The one next action

**Status: done.** Wired on top of the 0.5 architecture branch, one change,
two halves — plus the fixture repair the wiring forced.

## 1. Settlement-refused attempts stop advancing

`compute_cost.settlement_refusal(evidence)` is the single owner of the
settlement-refusal vocabulary: it reads the production `budget_settlement`
verdict, the classifier-facing `settle_refusal` field, and the
`refused_by: budget_settlement` stamp, and returns the machine-readable
identifier (`ACTUAL_EXCEEDS_PROJECTION`, ...) or `None`.

- `run_search` ends the lineage of an attempt that settled over budget: the
  spend stays in the accounting, the attempt stays in the round's record,
  and the stop lands in `lineage_stops` — an over-budget run cannot earn a
  larger budget, however well it trained.
- `candidate_search.advanced` and `cycle.select_candidate` both refuse such
  rows. `candidate_succeeded` is set *before* settlement runs, so the
  success flag alone was never a safe advance rule.
- `attempt_failure.classify_failure` derives its settlement branch from the
  same predicate, so the classifier and the runner cannot disagree. (Its
  previous `settle_refusal`-only read could never fire on a real production
  record.)

## 2. The promotion path consults the preregistered gates

`GrowthCycle._apply_promotion_gates` runs in both promotion paths
(`decide_promotion` and `decide_promotion_from_runs`):

- a declared `RetentionProfile` (`CycleConfig.retention_profile`) is
  evaluated fail-closed: an unmeasured constraint is a violation, not a
  pass; a PROMOTED verdict over any violation is downgraded to REJECTED with
  `RETENTION_REGRESSION` / `RETENTION_FLOOR` / `RETENTION_UNMEASURED`
  reasons. The candidate side reads only rows measured on this generation;
  the parent side only rows a measurement exists for.
- a declared `EvalTierPolicy` (`CycleConfig.eval_tier_policy`) must classify
  every constraint's benchmark as promotion evidence; a constraint measured
  on search-readable evidence refuses outright (`SearchIsolationRefusal`) —
  that is campaign wiring, not a measured outcome.

## 3. What the wiring caught

Exactly what the audit predicted: the campaign fixtures' "clean" runner
reported a fixed 0.05 wall against a ~0.006 projection, and the clean
promotions in the campaign-runner, certification-coupling, evaluation-binding
and dry-run-matrix suites had only ever promoted because selection ignored
settlement refusals. The fixtures now report a settleable cost
(`ATTEMPT_WALL_GPU_HOURS = 0.006`), the deliberate overruns are explicit
(`OVERRUN_WALL_GPU_HOURS = 0.05`), and the three scenarios that relied on
the hole assert the honest refusal path: the overrunning attempt refuses
itself, the campaign stops or refuses at selection, the spend stays in the
accounting, and nothing promotes. The REFUSED-at-selection record also now
carries the `stopping` phase, which it previously dropped.

## Verification

`tests/test_growth_runner_gates.py` (20 tests) covers the predicate's
vocabulary, the advance/selection/search refusals, the classifier agreement,
and both promotion paths' gates. Full growth regression: 690 passed. Ruff
clean.
