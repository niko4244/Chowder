# GEN2 prereg amendment 23 — 2026-10-19: a declared baseline carries its protocol identity

Adds to the 2026-09-17 prereg and amendments 1–22 and supersedes none of
them. **This is a lifecycle-plumbing fix, not a quality gate.** No slice,
threshold, retention bound, comparison, decoding parameter, recipe, step count,
ceiling or prepared input moves. The declaration still names `prepared-v16`,
unchanged.

## A. What run 9 did and why it was refused

Run 9 launched 2026-10-19 05:53 from main `16200cb` (prepared-v16; readiness
READY 19/19).
- Both admitted attempts started and stopped within about 0.6 s each:
  `attempt-28` and `attempt-29` ended `STOP_UNCERTAIN` with no experiment.
- The run was then refused at `candidate_evaluation`
  (`CANDIDATE_EVALUATION_NOT_PRODUCED`) with **no GPU time spent** and no
  candidate measured.

Bisected to `61e83bc` (PR #199, "derive run status from the goal lifecycle
and harden identity evidence"), merged after run 8.
- Since then, an attempt's `baseline_mode: fixed` baseline carries no
  evaluation-protocol evidence.
- The lifecycle therefore assesses it as `MISSING_EVALUATION_PROTOCOL` /
  `INCOMPLETE_EVALUATION_IDENTITY` and ends the objective before its first
  experiment.
- Run 8 predates #199, which is why it trained.

## B. The fix (PR #256)

- **The declared baseline.** A fixed baseline without measured protocol
  evidence carries `protocol_fingerprint({"evaluation": config.evaluation})`.
  That is the exact digest the objective is frozen under for such a baseline,
  and the evidence states where it came from. Measured evidence is never
  overwritten.
- **The attempt's lifecycle mode.** Every Gen-2 attempt goal metric is
  direction-only, with no bounds; pass/fail is this campaign's certification
  and judge. So the composed attempt declares
  `goal_lifecycle.mode: legacy_unbounded`, the lifecycle contract's sanctioned
  opt-in. Without it, the candidate's measured protocol fingerprint (versions,
  device, base identity) never equals the declared baseline's config
  fingerprint.
- **Verification.** The run-9 attempt, reproduced with the GPU hidden, now
  reaches its experiment; before the fix it stopped at the lifecycle.

## C. Admissibility

**No candidate was produced or measured in run 9**, so no result exists that
this amendment could be steered by. Run 8's in-run scores remain the only
visible Gen-2 candidate measurements, already disclosed in amendment 22.

The rerun (run 10) repeats prepared-v16 exactly: same recipes, steps, slices,
gates and judge. Run 9's `campaign-run.json` and
`cycle_compute_accounting.json` are archived under
`<state_root>/superseded-2026-10-19-run9/`. Its two empty attempts stay
under `attempts/`, and nothing is deleted.
