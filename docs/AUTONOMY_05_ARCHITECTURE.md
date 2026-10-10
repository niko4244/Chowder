# The 0.5 model-improvement architecture

One loop, nine gates. Every arrow is a refusal path as well as a data path:
each stage refuses rather than degrades, and every measured outcome —
including every falsification — feeds the next iteration.

```
                      ┌─────────────────────────────────────────────┐
                      │ EvidenceStore (append-only JSONL, sha256    │
                      │ hash chain, scoped by family × model ×      │
                      │ architecture)                               │
                      └──────┬──────────────────────▲───────────────┘
              prior_for_family│                     │ record()
              maturity gate   │                     │ (classified,
                              ▼                     │  fail-closed)
┌──────────────┐   ┌────────────────────┐   ┌────────┴─────────┐
│ Observation: │   │ Hypothesis         │   │ Campaign runner  │
│ metric weak- │──▶│ generation         │──▶│ attempt outcomes │
│ ness vs      │   │ (weakness-only,    │   │ classified by    │
│ threshold    │   │ prior-weighted,    │   │ attempt_failure  │
└──────────────┘   │ maturity-gated,    │   │ (infra / noise / │
                   │ anti-repeat)       │   │ budget / falsi-  │
                   └────────────────────┘   │ fied / arch /    │
                          │                 │ data)            │
                          ▼                 └────────▲─────────┘
        ┌──────────────────────────────┐                │
        │ Intervention families        │                │
        │ (group.mechanism, parameters │                │
        │ with ranges, maturity:       │                │
        │ PRODUCTION / QUALIFIED_EXP / │                │
        │ RESEARCH / REJECTED)         │                │
        └──────────────┬───────────────┘                │
                       ▼                                │
        ┌──────────────────────────────┐   ┌────────────┴───────────┐
        │ Candidate recipes            │   │ 3-tier evaluation      │
        │ (TrainingRecipe; every field │   │ search-evidence ◀──    │
        │ classified consumed / unsafe │   │ survivor-evidence ◀─   │
        │ / planned-unmapped; contract │   │ promotion-evidence     │
        │ tests prove reader paths)    │   │ (wall: isolation       │
        └──────────────┬───────────────┘   │ refuses tier-3 in any  │
                       ▼                   │ search surface)        │
        ┌──────────────────────────────┐   └────────────▲───────────┘
        │ Budget ladder                │                │
        │ deterministic → prior-       │                │
        │ weighted → adaptive (UCB1    │                │
        │ over training-side          │                │
        │ efficiency); reorder-only,   │                │
        │ ceilings immutable           │                │
        └──────────────┬───────────────┘                │
                       ▼                                │
        ┌──────────────────────────────┐                │
        │ plan_search (ceilings,       │                │
        │ worst-case projection) →     │                │
        │ run_search:                  │                │
        │   round 0: candidates train  │                │
        │   round r: survivors RESUME  │─── checkpoints ┘
        │   their own checkpoint;      │
        │   no-checkpoint ⇒ lineage    │
        │   stops; not-a-resume ⇒      │
        │   lineage stops, spend kept  │
        └──────────────┬───────────────┘
                       ▼
        ┌──────────────────────────────┐     ┌─────────────────────────┐
        │ Survivor evaluation (tier 2) │────▶│ Retention profile       │
        │ paired parent-vs-candidate   │     │ (preregistered max-     │
        │ per task, protocol-keyed;    │     │ regression + absolute   │
        │ bootstrap CI over task       │     │ floors; unmeasured =    │
        │ deltas; win/loss/ties        │     │ violation)              │
        └──────────────────────────────┘     └────────────┬────────────┘
                                                          ▼
                                           ┌─────────────────────────────┐
                                           │ Promotion only if:          │
                                           │  no violations AND CI       │
                                           │  excludes 0 AND tier-3      │
                                           │  gates pass                 │
                                           │  → PRODUCTION_QUALIFIED     │
                                           │  evidence (never auto-      │
                                           │  promoted again)            │
                                           └─────────────────────────────┘
```

## Enforcement points (where the guarantees live)

| Guarantee | Enforced by |
| --- | --- |
| No protected-benchmark optimization | `eval_isolation.classify_benchmarks` + `assert_search_isolation` (refuses at plan time; reserved names cannot be demoted) |
| No silent restart winning allocation | `run_search` lineage stops on `not-a-resume` and on missing checkpoints |
| No unbounded compute | `HalvingSchedule` (one owner) + `plan_search` ceilings + `should_stop` mid-round + Kaggle quota (R8) |
| No wasted GPU on disproven ideas | evidence priors (FAILED→0.35^n, incompatible→excluded) + `record_refuses_disproven_repeat` + maturity gate |
| No fabricated evidence | settlement refusals record nothing; store is append-only + hash-chained; `record()` refuses duplicate ids |
| No post-hoc thresholds | preregistered declaration/profile (docs/CAMPAIGN_DESIGN_05.md §6); mutation checks executable |
| No unpriced remote spend | Kaggle bar R3: settle through `compute_cost` like local spend |
| No threshold-drift in provenance | `hf_resilience.resolve_model_commit` (PR #205) |

## What is wired vs. library-only

Wired into the runnable path today: candidate search (continuation,
lineage, resume-progress), the axis contract, the maturity gate at
hypothesis generation, the evidence store + priors, the budget ladder's
`order_survivors` seam, attempt classification, retention evaluation,
paired deltas.

Library-only until the next wiring PR (see AUTONOMY_05_LIMITATIONS.md):
eval-isolation and retention are not yet consulted by the campaign
runner's promote path; the Kaggle bar (R1–R8) is a contract, not code.
