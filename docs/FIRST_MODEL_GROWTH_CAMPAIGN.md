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

## Step 1 — Freeze Generation 0

Run the broadest affordable evaluation battery **before** any training and
record exactly: model revision, tokenizer, quantization, inference engine,
prompt templates, reasoning settings, sampling parameters, hardware, and
benchmark versions. Without a frozen Generation 0, "improvement" is
meaningless.

Artifacts: `eval-report.json`, `capability profile`,
`contamination_manifest.json`, frontier snapshot `gen0-frontier`.

## Step 2 — First cycle

```text
1.  freeze Generation 0                       (done in step 1)
2.  evaluate                                  → EvalReport + profile
3.  derive capability profile                 → skills vs frontier ladder
4.  select 1–3 highest-confidence trainable weaknesses
5.  build a small curriculum                  (chowder growth curriculum)
6.  propose multiple recipes                  (recipe_planner, hardware-bounded)
7.  train candidates                          (bounded; tiny/local smoke first)
8.  successive-halving selection              (Chowder search controller)
9.  protected regression evaluation           (tier 2 battery + probes)
10. broad evaluation                           (tier 3)
11. promote or reject                          (the predeclared rule)
```

Every phase writes machine-readable provenance into the cycle ledger
(`growth/cycle.py`). The verdict is whatever the evidence says — a REJECTED
first cycle is a legitimate, recorded outcome.

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
