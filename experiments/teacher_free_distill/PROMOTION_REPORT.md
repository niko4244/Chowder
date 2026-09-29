# Promotion-rule report — Condition A vs Condition B (repair continuation)

Status: **PENDING — Condition B has not trained yet.** This report is the
template the promotion rule requires, pre-filled with every number that is
already measured. Nothing here promotes anything: the recipe's rule is
*"never auto-promoted; repair behavior metrics plus general-capability
regression check against the condition-A student."*

## The rule, applied

| check | threshold / comparison | evidence slot | status |
|---|---|---|---|
| Repair behavior: student repair-metrics on held-out replayed tasks | compare against Condition A student's repair-metrics | `eval_cond_b_repair.json` (to be produced by `eval_student.py` + `evaluate.py repair-metrics`) | **PENDING B** |
| General capability: MMLU (paired, 200 items) | no regression vs **Condition A student** (0.235), base 0.340 for reference | `eval_cond_b_mmlu.json` | **PENDING B** |
| General capability: GSM8K (paired, 200 items, answer-stop) | no regression vs Condition A student (0.105), base 0.580 for reference | `eval_cond_b_gsm8k.json` | **PENDING B** |
| Specialization fit: dev perplexity (197 rows) | compare vs Condition A student (5.49), base 13.70 for reference | `eval_cond_b_ppl.json` | **PENDING B** |
| Repair-training effect: did the 20 % general mix + continuation *reduce* forgetting vs A? | delta of (B − A) on MMLU/GSM8K; the whole point of Condition B | computed from the two rows above | **PENDING B** |
| Provenance: repair rows all `verification.method == sandbox_replay` with observed green events | prepare.py gate refuses the rest; 55 examples from 13 verified rows, 24 accepted after dedup | `prepared_repair/manifest.json` | **DONE** |
| Adapter lineage: B continues from A's adapter, digest-pinned | parent `sha256_directory = 68b6df8671a6692d…`, verified by the worker before load | `checkpoints/cond_b/run_record.json` (after launch) | **PENDING B** |
| Loss history exists | recipe `loss_history_required: true` | `checkpoints/cond_b/loss_history.json` (after training) | **PENDING B** |

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

1. `_launch_cond_b_gated.py` launches training when GPU0 clears the 8 GiB
   floor (armed; `cond_b_gate.log`).
2. After training: `eval_student.py --label cond_b --adapter checkpoints/cond_b/adapter`
   for MMLU/GSM8K/PPL (same 200/200/197 paired slices as A and base).
3. Replay B's held-out repair tasks and score with `evaluate.py repair-metrics`.
4. Fill the table, run `evaluate.py compare` where applicable, record the
   operator's decision here.
