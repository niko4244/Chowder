# 0.5 architecture preparation — limitations

Stated plainly, because an optimistic limitation list is just marketing.

## Not wired yet (the most important section)

1. **The budget ladder's adaptive rung has never seen a real search.** The
   UCB1 ordering is deterministic and tested, but its reward (training-side
   efficiency per GPU-hour) has no production definition yet; campaigns
   start DETERMINISTIC by construction, and the ladder escalates only when
   the evidence store says so.

## Wired since the original list (was items 1–2 here)

The runner's advance/promote path now consults the gates. A
settlement-refused attempt ends its lineage (`run_search` records the stop;
`advanced` and `cycle.select_candidate` both refuse it), so an unpriced
attempt can neither earn a larger budget nor become the candidate; and
`GrowthCycle.decide_promotion` / `decide_promotion_from_runs` evaluate the
campaign's preregistered retention profile fail-closed and refuse a
constraint measured on search-readable evidence outright. Wiring it caught
a second instance of the same disease: the campaign fixtures' "clean" run
reported 8x its projection and only promoted because selection ignored
settlement refusals — the fixtures now report settleable costs and the
deliberate-overrun scenarios assert the honest refusal path. Tests:
`tests/test_growth_runner_gates.py`; the campaign-runner, coupling,
evaluation-binding and dry-run-matrix suites re-pointed.

The loader seam is closed too: a campaign manifest can declare
`retention_profile` and `eval_tier_policy` (`campaign.py` parses both
fail-closed — unknown fields, unmeasurable constraints, unknown tiers and a
gate demoted into the search's view refuse at load), `_build_cycle` passes
them into `CycleConfig`, and `FIELD_ENFORCEMENT` names what each drives.
Tests: `tests/test_growth_manifest_promotion_gates.py` (8) — parse-valid
binding with an observed gate downgrade, twelve malformed-declaration
refusals, and unchanged promotion when neither section is declared.

## Proven only at small scale

2. **Checkpoint-resuming progressive halving has never run against a real
   GPU.** The E2E test composes a round-1 project whose config genuinely
   carries the round-0 checkpoint, and the peft backend's resume path is
   separately qualified (manifest binding, resume witness), but the
   composed loop — resume → train delta → settle → next round — has no
   Kaggle/GPU execution behind it. The continuation projection is a floor
   (restore cost unmodeled), as stated in `plan_search`'s docstring.
3. **Measured costs come from tiny screening runs** (0.008 device-GPU-h at
   300 steps on 2×T4). Extrapolating to 1200-step rounds assumes step cost
   scales linearly at fixed seq_len — plausible, unproven at this scale.
   The campaign design's falsification clause (§7) pauses on 2× overrun.
4. **The evidence store holds one real record** (Run 4's falsified
   replay-decay). Every prior number (0.35^n, 1.25×, 1.5×) is a heuristic
   multiplier chosen by argument, not fitted to data; they are documented
   starting points, and the store's value is the *policy* (starve
   falsified, never auto-promote), not the constants.

## Declared but not earned

5. **The families registry is declarative.** Nine families exist with
   basis notes; none beyond the SFT trio has in-scope evidence, and the
   RESEARCH six are refused by default. Their parameter ranges and risks
   are design intent, not measured envelopes.
6. **EI is deliberately not a search-time policy.** The repository's own
   backtest says EI does not beat UCB1 here (and its reward reads gate
   scores, which would breach the tier wall). It remains a campaign-layer
   selector with an honest docstring.
7. **The Kaggle backend bar is a contract, not code.** PR #200's dispatch
   is green and useful for smoke work; R1–R6 (source binding, artifact
   hashes, settlement, resume vocabulary, classified failures, pinned env)
   are unimplemented and gate campaign use.

## Environmental

8. **The ambient editable install points at a stale clone**
    (`C:\Users\nikma\Chowder`). Every command in this branch pins the
    worktree. Two name collisions (eval_tiers, failure_taxonomy) were
    caught by this — a third may exist where this branch's base diverged
    from main.
9. **PR #201's CI is failing and untriaged; #202 conflicts with main.**
    They are outside this branch's scope but block the "teacher-free"
    experiment lane; triage is a merge-order prerequisite, not an
    afterthought.
10. **The full local suite exercises no GPU.** Real-ML tests skip locally;
    the cpu-smoke proof for the drift fix lives in PR #205's CI (green),
    and any GPU-qualified claim in the campaign plan is contingent on
    that environment.
