# GEN2 preregistration — Amendment 2 (2026-09-18)

Written before any gen2 compute. It declares an **input** the frozen judge
already reads; it changes no threshold, no benchmark set, no budget, no
stopping rule, no verdict class and no gate.

## A. What was missing

Amendment 1 (section C8) added gate **T16, trusted-ancestor protection**: a gen2
candidate may not be promoted on its protected mini-slices unless it also fails
to regress against the **trusted ancestor gen0** — because its immediate parent
gen1 is `INCONCLUSIVE` with `target_repair_validated=true`, and an unresolved
parent must not become the protection baseline by default.

The judge reads that arm from `baseline_evaluation.json` in the run root. The
preregistration, however, never declared where a gen0 arm comes from, and
neither did the campaign manifest: §3 names only two arms ("parent gen1 adapter,
gen2 candidate"), and `docs/gen2/gen2_campaign.json` declared no ancestor
report. Section 3 of amendment 1 describes the three arms as required files
without saying what produces the third one.

Since PR #181 the campaign runner materialises the judged evidence set from the
manifest's declared inputs, and its rule is that **an undeclared input produces
no file rather than a placeholder** — which is correct, but it means that with
the manifest as frozen, a gen2 run would write no `baseline_evaluation.json`,
T16 would be `UNKNOWN`, and gen2 could never be `PROMOTED` at all: only
`INCONCLUSIVE`. The gate would be permanently undecidable, which is a structural
block, not a measurement outcome.

## B. Why it was caught before compute

No gen2 model load, no gen2 training, no gen2 evaluation has happened: there is
no `2026-09-17-gen2-response-surface` run directory, and no gen2 result exists to
condition anything below. The finding came from reading the frozen manifest's
declared inputs against the artifacts the frozen judge consumes and the runner
now writes — the same audit that found the writer gap — not from any result.

## C. Corrected rule (now the only rule)

**The gen0 trusted-ancestor arm is a required declared input of the gen2
campaign.** It is declared in `docs/gen2/gen2_campaign.json` as:

```json
"baseline_eval_report_path":
  "C:/Users/nikma/Chowder-Protected/runs/2026-09-18-gen2-gen0-arm/gen0-baseline-evaluation.json"
```

and the runner copies it, verbatim, to `<state_root>/baseline_evaluation.json`
for the judge. Requirements on that artifact:

1. **Fresh measurement, under the identical frozen protocol.** Both mini-slices
   of §3 exactly: 16 items, dataset order, no shuffle, seed 1234, greedy,
   `max_new_tokens=512`, chat-template prompt, and rows carrying
   `n_samples=16`, `sample_indices=0..15`. Measured on the **untouched dense
   gen0 parent** (`F:/llm-models/Qwen3.8-9B-abliterated-25-bf16`, digest
   `59e767aa…`), not on the gen1 adapter.
2. **Provenance.** Every gen0 row carries `measurement_origin=MEASURED_PARENT`
   and `generation_version=gen0`. Gen0 is a *different object* from gen1: an arm
   whose rows are relabelled gen1 (or gen2) measurements is not a gen0
   measurement, and the candidate side may never satisfy this arm.
3. **The prior gen0 freeze report is not this arm.** `2026-09-16-gen0-eval-freeze/
   freeze/eval-report.json` was measured by a different protocol (not the frozen
   16-item mini-slices), so it cannot be declared here; using it would make T16
   undecidable again and is the carried-evidence substitution the promotion
   path refuses.
4. **Absence is fail-closed.** A missing declared file refuses the run
   (`baseline_eval_report_path` named) before any arm is written; a written arm
   whose rows are missing, duplicated, wrongly originated or off-protocol is
   `UNKNOWN`/`FAIL` at T11/T16, never a pass. Nothing about T16 may be
   approximated from the gen1 parent arm.
5. **Cost.** The arm is measured **once, before the gen2 run**, and referenced
   by the campaign at zero incremental cost — the same treatment §5 already
   gives the parent baseline (`baseline.mode: fixed`). Its measurement is a real
   prior evaluation job with its own recorded evidence; the arm's rows must name
   the artifact that produced them (`raw_artifact_ref`), so the cost of the arm
   is inspectable even though it is not charged to this campaign's recipe
   envelope. It may not be used to claim budget compliance for this campaign.

The judge is **not modified** by this amendment: it already reads
`baseline_evaluation.json` with `MEASURED_PARENT` rows and applies T16.
What is enforced by the judge is provenance and protocol; the generation label
and `raw_artifact_ref` are declarations the arm records honestly, and a reader
verifies them from the named artifact.

## D. What did NOT change

- Thresholds (`duplication ≤ 0.125`, `echo ≤ 0.062`, format 8/8, correctness
  ≥ 15/16, EOS ≥ 0.900, cap < 0.100, loops ≤ 0, trigram ≥ 0.900, unclosed think
  ≤ 0.250, slice regression tolerance `0.0625`).
- The benchmark sets, the mini-slice protocol (16 items, seed 1234, greedy,
  512 tokens, chat template), the budgets (0.30/0.75 per recipe, 0.60/1.50 per
  campaign), the stopping rules, the verdict classes and the scoped-repair
  convention.
- Amendment 1's device-ceiling policy: `device_time_measured: false`, so device
  ceilings are admission constraints on the projected plan and wall remains the
  post-run settlement gate.

## E. Consequence for the cycle

T16 becomes decidable. With the arm declared and measured, a gen2 candidate is
promoted only if it holds against **both** the gen1 parent and the gen0 trusted
ancestor, plus the target and format gates, exact contamination coverage,
settlement within the declared envelope and a recomputed winner-artifact digest.
A gen2 that merely matches a gen1 that had already regressed against gen0 is
`REJECTED` (hard), consistent with §7. A gen2 whose gen0 arm is absent remains
`INCONCLUSIVE` — the honest answer, and the reason this amendment had to be
written before compute rather than after.

## F. Operator checklist before the gen2 run

1. Measure the gen0 arm under the §3 protocol; write it to the declared path.
2. `chowder growth campaign validate docs/gen2/gen2_campaign.json`.
3. Confirm the arm's rows are `MEASURED_PARENT` / `generation_version=gen0`, 16
   items, indices 0–15, seed 1234, no shuffle, greedy, 512 tokens, chat-template
   prompt, each naming the artifact that produced it.
4. Only then start the campaign: the run writes
   `<state_root>/baseline_evaluation.json` from this declaration, and the frozen
   judge reads it as the trusted-ancestor arm.
