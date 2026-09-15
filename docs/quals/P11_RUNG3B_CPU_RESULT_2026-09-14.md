# P11 rung 3b — tiny-CPU router pilot result (2026-09-14)

Judged against the committed preregistration
`docs/quals/P11_RUNG3B_CPU_PREREG_2026-09-14.md` (pushed as PR #161 commit
`410696f`, **before** `chowder project-validate`/`chowder train` ran). Evidence
root: this directory (fresh registry `work/runs.db`, run
`router-pilot-a0649170b84b`, both evaluation arms under `work/.chowder/evals/`).

## Verdict

**Rung 3b (CPU arm): COMPLETE — all seven preregistered thresholds pass on
measured evidence.** The gate decision is **rejected**, which the prereg fixed
as a valid outcome: both arms were measured, the decision and its ledger are
recorded, and the routing genuinely changed — the trained gates simply did not
improve holdout loss in 12 steps.

## The run, through the normal interface

`chowder project-validate` (exit 0) → `chowder train` (exit 0), project
`router-healing-tiny-rung3b-cpu`, recipe byte-identical to the rung-2 CPU pilot
(base `tiny-qwen3-moe-e4-k2` from the hash-verified builder `bfa954ee…`,
corpora copied unmodified, 12 steps, lr 0.05, seq 16, batch 2, seed 1,
`minimum_promotion_gain` 0.0). The load-policy seam introduced by the
amendment implementation is in the code path and reports
`policy: fp32-resident` in every worker result.

| Threshold | Evidence | Result |
|---|---|---|
| 1. validate before train | `project-validate-stdout.log` exit 0; `train-stdout.log` exit 0 | PASS |
| 2. registry truth | rows `baseline: passed`, `router-pilot: rejected` — both terminal; closeout stranded-result audit reported no findings | PASS |
| 3. trainability | `trainability.ok=true`; frozen digests `full` strategy, **23/23 frozen tensors, `changed: {}`**; exact gate-path scope (2 router gates, `scope.ok=true`) | PASS |
| 4. horizon | `global_step=12`, `stop_reason=max_steps` | PASS |
| 5. gate | base 4.174509 / candidate 4.174548 (**delta +3.886e-05**), both measured; decision **rejected** with ledger; payload is a real `replacement` (params changed, not identity), routing measurably changed (`top1_equal=false`, max weight delta 0.732) | PASS |
| 6. accounting | train wall 7.05 s; eval arms 7.3 s / 7.07 s; phase ledger: model_load 7.18 s, baseline_generation 0.017 s, candidate_generation 0.010 s; attributable GPU-hours 0.0; reservation settled, no outstanding | PASS |
| 7. identity chain | base content `903331fb…`, worker source identity verified, spec digest `943e82e7…` | PASS |

## Honest observations

- **The candidate was worse, not better.** 12 steps at lr 0.05 moved the
  routers measurably (weight delta 0.732, top-1 decisions changed) but
  holdout loss rose by 3.9e-05, and `dead_experts` went 1 → 3 on the holdout
  distribution: the untrained-then-lightly-trained router collapsed routing
  toward fewer experts for this corpus. That is a measured property of the
  recipe at this horizon, not a defect in the loop — the rung proves the
  machinery, and the gate did exactly its job.
- The 9B-derived arm remains governed by the amendment
  (`P11_RUNG3_AMENDMENT_2026-09-14.md`); nothing here qualifies any large
  artifact.

## Reproduction

```bash
cd /c/Users/nikma/Chowder-Protected/runs/2026-09-14-router-healing-rung3b-cpu
PYTHONPATH=<chowder>/src python -m <chowder-cli> project-validate router-project-rung3b-cpu.json
PYTHONPATH=<chowder>/src python -m <chowder-cli> train router-project-rung3b-cpu.json
```

Durable artifacts: `work/runs.db` (registry + audit), `work/.chowder/runs/…`
(training worker result, spec, payload, identity), `work/.chowder/evals/…`
(base and candidate eval results), both CLI stdout logs, and the project file.
