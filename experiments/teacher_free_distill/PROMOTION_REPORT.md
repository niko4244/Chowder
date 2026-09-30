# Promotion-rule report — Condition A vs Condition B (repair continuation)

Status: **CONDITION B TRAINED (2026-09-30) — paired capability eval in flight;
all decision slots still PENDING.** This report is the template the promotion
rule requires, pre-filled with every number that is already measured. Nothing
here promotes anything: the recipe's rule is *"never auto-promoted; repair
behavior metrics plus general-capability regression check against the
condition-A student."* B's lineage: 2 optimizer steps, LoRA continuation from
the Condition A adapter (parent digest `68b6df86…` worker-verified), 23 repair
rows + 6 replayed general rows (ratio 0.25), loss 2.2219 → 2.2218,
`checkpoints/cond_b/run_record.json`.

## The rule, applied

| check | threshold / comparison | evidence slot | status |
|---|---|---|---|
| Repair behavior: student repair-metrics on held-out replayed tasks | compare against Condition A student's repair-metrics | `eval_cond_b_repair.json` (to be produced by `eval_student.py` + `evaluate.py repair-metrics`) | **PENDING B** |
| General capability: MMLU (paired, 200 items) | no regression vs **Condition A student** (0.235), base 0.340 for reference | `eval_cond_b_mmlu.json` | **PENDING B** |
| General capability: GSM8K (paired, 200 items, answer-stop) | no regression vs Condition A student (0.105), base 0.580 for reference | `eval_cond_b_gsm8k.json` | **PENDING B** |
| Specialization fit: dev perplexity (197 rows) | compare vs Condition A student (5.49), base 13.70 for reference | `eval_cond_b_ppl.json` | **PENDING B** |
| Repair-training effect: did the 20 % general mix + continuation *reduce* forgetting vs A? | delta of (B − A) on MMLU/GSM8K; the whole point of Condition B | computed from the two rows above | **PENDING B** |
| Provenance: repair rows all `verification.method == sandbox_replay` with observed green events | prepare.py gate refuses the rest; 55 examples from 13 verified rows, 24 accepted after dedup | `prepared_repair/manifest.json` | **DONE** |
| Adapter lineage: B continues from A's adapter, digest-pinned | parent `sha256_directory = 68b6df8671a6692d…`, verified by the worker before load | `checkpoints/cond_b/run_record.json` | **DONE** |
| Loss history exists | recipe `loss_history_required: true` | `checkpoints/cond_b/loss_history.json` — 2 entries (per-step logging; 2-step run) | **DONE** |

## Measured inputs (Condition A side)

General-capability paired table (same items, same prompts, same scorer;
`eval_base*.json`, `eval_cond_a_*.json`):

| axis | base | Condition A | delta |
|---|---|---|---|
| MMLU bare-letter (200) | 0.340 | 0.235 | −0.105 (paired exact p = 0.0025) |
| GSM8K answer-stop (200) | 0.580 | 0.105 | −0.475 |
| dev perplexity (197) | 13.70 | 5.49 | −8.21 |

Repair-behavior baseline of the rows Condition B trains on — the A repair
slice scored by `evaluate.py repair-metrics` (`repair_metrics_a_slice.json`,
22 usable replayed trajectories of which 13 are the verified ones):

| metric | value |
|---|---|
| green_completion_rate | 0.5909 |
| nonexistent_read_rate | 0.0000 |
| premature_success_rate | 0.0000 |
| recovery_rate | 0.5909 |
| mean_tool_calls | 5.18 |

Slice-size note: the 41-row recoverable-path replay (patch-derived tests,
`--recover-mispaired-derived-tests`, `replayed_traces_recover.jsonl`) certified
**0 additional rows** (28 `recovery_evidence_insufficient` — zero tests ran in
both phases — plus 7 `not_green`, 6 `patch_did_not_fit`), so this repair slice
cannot grow beyond the 24 examples prepared from the 13 certified rows without
new verified source data.

Note on scope: this is a **training-slice** behavior profile, not a held-out
capability score. After B trains, the same scorer runs over B's *held-out*
repair tasks (replayed trajectories not in B's training rows) — that number,
not this one, feeds the promotion comparison.

## What would pass, what would fail

- B passes the regression check only if MMLU/GSM8K deltas vs the **A student**
  are non-negative (the probe exists because A forgot; B without replay
  improvement is a failed hypothesis, honestly reportable).
- B failing repair metrics on held-out tasks vs the A student means the
  55-example slice was too thin to add repair skill — also a real answer.
- Any regression flag from `evaluate.py compare` is **advisory**; the decision
  stays `requires_operator_review` by construction.

## Pending steps to fill this report

1. ~~Gate + training~~ **DONE 2026-09-30**: the gate fired when GPU0 cleared
   the floor; B trained after two launch defects were fixed and
   regression-tested (multi-turn mask alignment; per-step loss logging).
2. **Paired capability eval — in flight**: `eval_student.py --label cond_b
   --mmlu-style bare` on the same 200/200/197 slices (MMLU bare so it pairs
   with A's 0.235; GSM8K answer-stop pairs with A's 0.105; PPL pairs with
   5.49). Output: `eval_cond_b.json`; the paired table is produced by
   `eval_student.py --compare eval_cond_a_merged.json eval_cond_b.json`.
3. ~~Held-out repair scoring~~ **DONE with a structural caveat**: the only
   usable replayed trajectories outside B's 8 trained instances are 9 rows,
   all from `not_green` instances (`repair_metrics_heldout.json`:
   `green_completion` 0.0, `mean_tool_calls` 5.11). The held-out slice cannot
   produce a positive repair score **by construction** — every certifiable
   instance is inside B's training set — so the repair-behavior comparison
   vs A's 0.591 training-slice baseline is not meaningful at this corpus
   size and is reported as a known limitation, not a result.
4. Fill the table, run `evaluate.py compare` where applicable, record the
   operator's decision here.
