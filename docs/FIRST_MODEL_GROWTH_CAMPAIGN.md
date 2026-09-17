# First Model Growth Campaign

This is the dry-run plan for the first autonomous cycle. It does **not**
start the 9B campaign on this branch; it is the template every future
generation follows.

## Prerequisites

1. Platform closeout merged to `main`; this branch rebased onto it.
2. The router/training backend qualification evidence in place
   (P11 rung 2/3 artifacts).
3. The parent checkpoint pinned: exact revision, tokenizer, quantization,
   inference engine.

## Step 1 — Freeze Generation 0 — DONE (2026-09-17)

The dense Qwen3.8-9B abliterated parent is frozen as Generation 0,
evaluation-only, per `docs/gen0/GEN0_EVAL_RESULT_2026-09-17.md`: exact
identity manifest (content digest `59e767aa…7555f`), protocol
`gen0-freeze-protocol-v1` (chat template, greedy, seed 1234), measured
battery (`math500@2024-04` 0.0, `mgsm@2022-11` 0.0, generation
diagnostics: EOS rate 0.000 / cap-hit 1.000 / trigram 0.988), honest
UNMEASURED rows preserved, contamination manifest all-UNKNOWN (no training
pool exists), freeze digest `5c8b18ab…`, immutable snapshot
`gen0-frontier`, ledger record `gen0`.

The first cycle is preregistered (before any training compute) at
`docs/quals/GEN1_PREREG_2026-09-17.md`: target = chat-protocol compliance
(turn termination + thinking-block closure), chosen from the measured
evidence above.

## Step 2 — First cycle

```text
1.  freeze Generation 0                       (done in step 1)
2.  evaluate                                  → EvalReport + profile
3.  derive capability profile                 → skills vs frontier ladder
4.  select 1–3 highest-confidence trainable weaknesses
5.  build a small curriculum                  (chowder growth curriculum)
6.  propose multiple recipes                  (recipe_planner, hardware-bounded)
7.  train candidates                          (bounded; tiny/local smoke first)
8.  successive-halving selection              (growth planner budget envelope)
9.  protected regression evaluation           (tier 2 battery + probes)
10. broad evaluation                           (tier 3)
11. promote or reject                          (the predeclared rule)
```

Every phase writes machine-readable provenance into the cycle ledger
(`growth/cycle.py`). The verdict is whatever the evidence says — a REJECTED
first cycle is a legitimate, recorded outcome.

> Step 8 note: Chowder's `run_successive_halving()` /
> `prioritize_candidates()` are proven library capabilities, but no
> production caller invokes them and `docs/ROADMAP.md` records that wiring
> them into `project_runner.py` is still open. Until it lands, step 8 means
> the growth planner's bounded envelope plus whatever the supplied
> `TrainingFn` actually executes — not a search controller this cycle can
> address by config.

## Decision gates

| Gate | Rule |
| --- | --- |
| Budget | GPU-hours preregistered before the cycle; overruns refuse, not apologize |
| Contamination | Any KNOWN/POSSIBLE on target or protected evidence taints the cycle |
| Statistics | Target improvement must clear the declared minimum, significantly |
| Protected | No regression past tolerance; unmeasured ≠ passing |
| Frontier | Recorded, never decisive |

## Cost expectations

The first cycle is deliberately small: tiny smoke runs locally, a bounded
LoRA recipe on the order of hours, not days. The measured P11 wall-charge
model (workload + one model load, ×1.5 safety) is the budget template.

## Readiness verdict

**Infrastructure: READY. Campaign: NOT YET STARTED — awaiting the frozen
Generation-0 evaluation of the parent 9B on the qualified device path.**

The blocking dependency is platform closeout landing and the rebase of this
branch; the growth system itself is complete and tested.

## The 12 questions this system must answer

1. How smart is this model? → scoreboard + category aggregates
2. Where specifically is it weak? → skill profile + failure bank
3. How does it compare (parent/peers/open/absolute frontier)? → gap rows, protocol-honest
4. What should it learn next? → curriculum priority traces
5. What data can legally and scientifically teach that? → data registry (license + verification)
6. How do we know the data is good? → trust class + quality gate + verifier evidence
7. How do we know it did not contain our protected tests? → contamination manifest
8. Which recipe gets the GPU budget? → successive halving under envelope
9. Did the candidate actually become better? → predeclared promotion statistics
10. Did it forget anything already taught? → regression probes + replay
11. Should it become the next generation? → the verdict, from (9)–(10)
12. Can every answer be reproduced from durable evidence? → lineage ledger + frozen snapshots

When those are mechanically answerable — and they now are — Chowder has
moved from an experiment runner to an evidence-driven autonomous
model-development laboratory.
