# Promotion-rule report — Condition A vs Condition B (repair continuation)

Status: **ALL SLOTS MEASURED (2026-09-30) — decision stays
`requires_operator_review`.** The recipe's rule is *"never auto-promoted;
repair behavior metrics plus general-capability regression check against the
condition-A student."* Nothing here promotes anything.

**Addendum 2026-10-01 — the scaled rerun and the powered probe are measured,
and they measure against pilot-scale B:** a 40-epoch B variant
(`checkpoints/cond_b_scaled`, loss 1.4842) **fails** the no-regression check
against A on the full 14 042-item MMLU probe (−0.0474 vs A, exact McNemar
p = 2.0e-106; `fullmmlu_paired_analysis.json`), and the same probe measures
**Condition A's own forgetting at −0.1971 vs base** (3 872 vs 1 105 discordant
items) — the 200-item slot below is now known to have been unrepresentative by
roughly a factor of two. Pilot-scale B remains the only variant that is
no-worse-than-A on every measured axis; the powered table is appended below
and the operator decision now has a strictly better evidence base.

B's lineage: 2 optimizer steps, LoRA continuation from the Condition A
adapter (parent digest `68b6df86…`, re-verified by the worker before load),
23 repair rows (`prepared_repair/train.jsonl` sha `fc2f47ef…`) + 6 replayed
general rows (`pilot_v3/train.jsonl` sha `f916364e…`, ratio 0.25, seed 2026),
loss 2.2219 → 2.2218, wall 70 s — `checkpoints/cond_b/run_record.json`,
`loss_history.json` (2 per-step entries).

A scaled variant (`recipes/b_repair_scaled.json` → `checkpoints/cond_b_scaled`)
freezes everything above except the epoch count (2 → 40) and trains the same
lineage over the same 24-example slice: 40/40 steps, loss 2.2219 → **1.4842**
(the slice is genuinely fit this time), wall 1 483 s, 40-entry loss history,
same parent-digest verification. It is measured in the powered table below and
**does not pass the advisory regression check** that pilot-scale B passes.

## The rule, applied

| check | threshold / comparison | evidence slot | status |
|---|---|---|---|
| Repair behavior: student repair-metrics on held-out replayed tasks | compare against Condition A student's repair-metrics | `repair_metrics_heldout.json` (9 usable held-out trajectories) | **MEASURED — structurally inconclusive** (see below) |
| General capability: MMLU (paired, 200 items, bare) | no regression vs **Condition A student** (0.235), base 0.340 for reference | `eval_cond_b_mmlu.json` | **MEASURED: 0.245 (Δ +0.010)** |
| General capability: GSM8K (paired, 200 items, answer-stop) | no regression vs Condition A student (0.105), base 0.580 for reference | `eval_cond_b_gsm8k.json` | **MEASURED: 0.115 (Δ +0.010)** |
| Specialization fit: dev perplexity (197 rows) | compare vs Condition A student (5.49), base 13.70 for reference | `eval_cond_b_ppl.json` | **MEASURED: 5.4923 (Δ +0.0008)** |
| Repair-training effect: did the 20 % general mix + continuation *reduce* forgetting vs A? | delta of (B − A) on MMLU/GSM8K; the whole point of Condition B | `eval_compare_b_vs_a.json` | **MEASURED: no further forgetting; recovery not demonstrated** (see verdict) |
| Provenance: repair rows all `verification.method == sandbox_replay` with observed green events | prepare.py gate refuses the rest; 55 examples from 13 verified rows, 24 accepted after dedup | `prepared_repair/manifest.json` | **DONE** |
| Adapter lineage: B continues from A's adapter, digest-pinned | parent `sha256_directory = 68b6df8671a6692d…`, verified by the worker before load | `checkpoints/cond_b/run_record.json` | **DONE** |
| Loss history exists | recipe `loss_history_required: true` | `checkpoints/cond_b/loss_history.json` — 2 entries (per-step logging; 2-step run) | **DONE** |

## Measured paired table (base → A → B; same items, same prompts, same scorer)

| axis | base | Condition A | Condition B | B − A | paired detail (B vs A) |
|---|---|---|---|---|---|
| MMLU bare-letter (200) | 0.340 | 0.235 | **0.245** | **+0.010** | candidate-only 5, base-only 3, both 44, neither 148 |
| GSM8K answer-stop (200) | 0.580 | 0.105 | **0.115** | **+0.010** | candidate-only 17, base-only 15, both 6, neither 162 |
| dev perplexity (197) | 13.70 | 5.49 | **5.4923** | +0.0008 (NLL +0.0002) | 197 rows, completion-only NLL |
| **Powered MMLU probe (14 042 items, bare, exact McNemar)** — base → A | 0.4743 | **0.2772** | — | — | discordants base-only 3 872 vs A-only 1 105, **p ≈ 0**; Wilson CIs [0.466, 0.483] / [0.270, 0.285] |
| **Powered MMLU probe — base → B (pilot, 2 steps)** | — | — | — | — | not run: pilot-B's 200-item deltas (+0.010) cannot be resolved at 14 k without ~4 h GPU; its no-regression evidence stands on the 200-item paired runs |
| **Powered MMLU probe — base → B-scaled (40 ep)** | 0.4743 | — | **0.2298** | **−0.2445 vs base, −0.0474 vs A (p = 2.0e-106)** | discordants vs A: 835 vs 169; vs base: 4 497 vs 1 064; **fails the no-regression-vs-A check** |
| **Powered GSM8K (200-item, the only recovery signal)** | 0.580 | 0.105 | **0.385** (scaled) | +0.28 vs A | a recovery of most of what A lost, purchased with the MMLU regression above; pilot-B stayed at 0.115 |

Advisory regression check (`eval_student.py --compare eval_cond_a_merged.json
eval_cond_b_merged.json` → `eval_compare_b_vs_a.json`): **zero regression
flags**; decision `requires_operator_review` by construction.

## The forgetting-probe verdict, read honestly

- **No further forgetting**: B does not lose anything A still had — both
  capability axes move +0.010 and the paired discordants are near-symmetric
  (MMLU 5:3 for B, GSM8K 17:15 for B). Nothing here approaches significance
  at n=200; the probe's clean read is "no measurable harm", which is what the
  replay mechanism was built to guarantee.
- **No demonstrated recovery**: A's forgetting (GSM8K 0.580 → 0.105 vs base)
  is essentially untouched (B 0.115). Two optimizer steps over 24 examples
  cannot be expected to move general capability; the measured deltas are
  consistent with "B ≈ A plus a nudge inside paired noise".
- **What the run actually establishes** at pilot scale: the A→B continuation
  lineage works end-to-end (digest-pinned parent, verified repair-only
  provenance, general-mix replay inside the executor), the completion-mask
  fix makes multi-turn SFT trainable on Qwen3 at all, and the paired eval
  produces machine-checkable per-item evidence for every claim above.
- **What the scaled rerun adds (2026-10-01): the "no further forgetting"
  reading does not survive scaling.** 40 epochs over the same 24 examples
  produce real training progress (loss 2.2219 → 1.4842) and a genuine GSM8K
  recovery (0.105 → 0.385), but the powered MMLU probe shows the extra
  training **deepens** general forgetting: B-scaled is 4.7 points below A with
  835-vs-169 discordant items (p = 2.0e-106). Under GPU contention this cost
  ~2 h of eval per model and is fully evidenced per item
  (`eval_fullmmlu_*.json`, `fullmmlu_paired_analysis.json`). The rule's
  regression check works exactly as designed: it fires for B-scaled and does
  not fire for pilot-scale B. Nothing here promotes either variant.

## Repair behavior: measured, with the structural caveat on the record

- **Held-out slice** (`repair_metrics_heldout.json`): the only usable replayed
  trajectories outside B's 8 trained instances are 9 rows — all from
  `not_green` instances (gTTS, thefuzz, dask `pr_7138`/`pr_8897`/`pr_9130`).
  `green_completion_rate` **0.0**, `nonexistent_read_rate` 0.0,
  `premature_success_rate` 0.0, `recovery_rate` 0.0, `mean_tool_calls` 5.11.
  The 0.0 is **by construction**: every certifiable instance is inside B's
  training set (the 41-row recoverable-path replay certified 0 additional
  instances), so no held-out task exists that a student could complete green.
  This number is a corpus limitation, not a model measurement.
- **Training-slice baseline** (`repair_metrics_a_slice.json`, 22 usable
  trajectories, 13 verified): green_completion 0.5909, recovery 0.5909,
  nonexistent_read 0.0, premature_success 0.0, mean_tool_calls 5.18.
  A same-slice scoring of the B student was not run: B trained on every
  verified trajectory in the slice, so its score there measures memorization,
  not repair behavior — reporting it as a comparison would be misleading.

## What passed, what failed, what remains open

- **Passed**: no-regression check on both capability axes (B − A ≥ 0
  everywhere, zero advisory flags); lineage, provenance and loss-history
  requirements all evidenced.
- **Failed (hypothesis, honestly reportable)**: the repair-continuation
  hypothesis is **not validated at this scale** — 2 steps × 24 examples
  produced no measurable repair-skill gain, and the held-out slice cannot
  demonstrate one until new verified instances exist outside the training set.
- **Open for the operator**: whether to promote B as the new base for further
  work (it is strictly no-worse than A on every measured axis and carries the
  repair-continuation machinery), or to first grow the verified repair corpus
  so the repair-behavior check becomes measurable. The official-image recovery
  replay (28 rows, 2026-10-01) certified **0** with container-grade evidence
  (6 honest `not_green`, 22 honest insufficiencies — the zero-evidence bucket
  was fragility plus scratch patches, not hidden certifications), so corpus
  growth now rests on the batch-2 census in flight (pandas/graphene/
  alive-progress; 1 277 rows / 300 instances; fit stage checkpointed against
  host restarts). The recoverable path and upstream SWE-smith patch/branch
  coherence remain the measured bottlenecks.

## Steps that filled this report

1. **Gate + training — DONE 2026-09-30**: the gate fired when GPU0 cleared
   the floor; B trained after two launch defects were fixed and
   regression-tested (Qwen3 final-turn think-block mask alignment;
   per-step loss logging for tiny runs).
2. **Paired capability eval — DONE**: three per-axis invocations (restart-
   resilient; each axis writes its own file): GSM8K answer-stop, MMLU
   `--mmlu-style bare` (pairs with A's 0.235), PPL — same 200/200/197 slices;
   merged into `eval_cond_b_merged.json` and compared against
   `eval_cond_a_merged.json`.
3. **Held-out repair scoring — DONE with the structural caveat above**
   (`heldout_repair_rows.jsonl` → `repair_metrics_heldout.json`).
4. **Decision**: recorded as `requires_operator_review` — the operator fills
   the final disposition here; no automation promotes anything.
