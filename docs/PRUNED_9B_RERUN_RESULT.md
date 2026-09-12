# Result — re-run of the training arm on the pruned 9B

> **SKELETON, written 2026-09-12 ~01:0x while the candidate eval was still running.**
> Every `«TBD»` is a number not yet measured. The verdict rules below were fixed
> before those numbers existed, so filling them in produces the verdict mechanically.
> **No `«TBD»` may survive into the final commit** — if one does, the doc is a draft
> and must not be read as a result.

Pre-registration: `PRUNED_9B_REAL_TRAINING_PREREG.md` (`9f66960`) and
`PRUNED_9B_REAL_TRAINING_PREREG_ADDENDUM.md` (`5de632d`), both committed before this
run started. First attempt and its failure: `PRUNED_9B_REAL_TRAINING_RESULT.md`.

Ran 2026-09-11 22:27:39 → «TBD». Work dir `F:\llm-models\_a4b\realtrain-gsm8k-2`.

## Verdict

| arm | verdict |
|---|---|
| **Engineering** (the primary claim) | **«TBD»** |
| **Capability** (secondary) | **«TBD»** |

### Engineering, condition by condition

The pre-registered PASS required all six. FAIL is any one of them missing, or
OOM/oversubscription judged by headroom and step-time blowup.

| condition | required | measured | |
|---|---|---:|:--|
| run completes | 500 optimiser steps | **500** | ✓ |
| target coverage | 200/200 modules | **200/200** | ✓ |
| live adapter | not inert | `adapter_model.safetensors` written, liveness guard passed | ✓ |
| loss finite and decreasing | yes | 2.8503 (step 1) → 1.0676 (step 500), mean 1.2107 | ✓ |
| peak VRAM, training | < 15.93 GiB | **6.14 GiB** | ✓ |
| GSM8K both arms, same protocol | yes | baseline 0.00 (50/50); candidate «TBD» | «TBD» |

Coverage reproduced the hybrid's real structure from a live model rather than from an
assumption: `q/k/v/o_proj` 8 each (the 8 full-attention layers), `in_proj_qkv` /
`in_proj_z` / `out_proj` 24 each (the 24 linear-attn layers), `gate/up/down_proj` 32
each (all layers) = 200.

### Capability

Baseline **0.00**, candidate **«TBD»**, delta **«TBD»**.

* **RECOVERS** — delta ≥ +0.02
* **FLAT** — |delta| ≤ 0.02
* **DEGRADES** — delta < −0.02

**This arm is one-sided and cannot produce a negative result.** The baseline is at
the floor, so DEGRADES is unreachable: the candidate cannot score below 0.00. A FLAT
here therefore means "no measurable movement off zero", not "training did nothing
harmful" — the experiment has no power to detect harm. Recorded now because it is a
limitation of the design, not of the outcome.

**What the gate needs, precisely.** `gate.py:67` accepts on `score >
minimum_promotion_gain`, a strict `>`. With `minimum_promotion_gain: 0.02` and a
0.00 baseline, promotion requires **2 of 50 correct (0.04)**, not 1. The original
pre-registration glossed 0.02 as "one problem in fifty"; one problem gives score
exactly 0.02, which is not `> 0.02` and is rejected. This is a precision correction
to the gloss, not a change to the threshold.

Gate verdict: promoted = **«TBD»**.

## The schedule the pre-registration asked for actually ran

The first attempt trained on linear while the recipe said cosine, because
`unsloth_worker` never read `lr_scheduler_type` (`5de632d`). Verified this time from
the realised curve rather than the config, because the config said cosine on the
first attempt's successor too:

| candidate schedule | residual (fraction of 2e-4 peak) |
|---|---:|
| **cosine** | **0.0000 — exact, all 500 steps** |
| linear | 0.0754 |
| constant | 0.6116 |

All 500 logged rates equal `2e-4 · ½(1 + cos(π(s−1)/500))` to the last float bit, at
the offset −1 that HF's `get_last_lr()` logging implies. The final rate is
**1.97e-9** against a 2e-4 peak — nine orders of magnitude down, which no other
candidate schedule produces and which needs no fitting to read. Promoted into
`chowder/schedule_audit.py` with the trajectory committed as a test fixture
(`6a488b9`).

## The defect that killed the first attempt did not recur

Step **323** — where the first attempt died on an unguarded `progress.tmp` →
`progress.json` rename — passed without incident, and the worker reported
`progress_write_failures: 0`, so the retry path was never even needed. That makes
the original `WinError 5` a genuinely transient handle, which is the case the
retry-then-continue design targets.

The two runs also agree where they overlap, which independently validates the
forensic recovery from the crash:

| | first attempt (recovered) | this run |
|---|---:|---:|
| step 322 loss | 1.1002 | 1.0984 |
| step 323 loss | 1.0737 | 1.0750 |

## Degeneration readout (pre-registered in the addendum as descriptive, not a gate)

Computed post-hoc from each arm's durable `predictions-gsm8k.jsonl`; no evaluation
code was touched and the protocol is byte-identical between arms.

| measure | baseline | candidate «TBD» |
|---|---:|---:|
| closed `</think>` (produced an answer span) | 0/50 | «TBD» |
| flagged degenerate | 50/50 at n=31 checked mid-run; «TBD» final | «TBD» |
| mean distinct-trigram ratio | 0.098 (at n=31) → «TBD» | «TBD» |
| mean compression ratio | 0.087 (at n=31) → «TBD» | «TBD» |
| hit the 768-token cap | «TBD» (needs tokenizer, not chars) | «TBD» |

**Pre-registered reading:** a move here is evidence that 500 steps changed
generation behaviour even if GSM8K does not move. It is **not** a substitute for the
capability gate and is not reported as one. No threshold is set on it, before or
after.

The mid-run baseline numbers at n=31 replicate the dense-vs-pruned control (0.094
distinct-trigram, 0.084 compression at n=8) on an independent, larger sample.

## Open question: the candidate arm ran 2× slower on a nearly full card

| | baseline arm | candidate arm |
|---|---:|---:|
| seconds per problem | 83 | **166** |
| card free during the arm | ~10.5 GiB | **0.54 GiB** |
| resident process | eval worker only | eval worker only (no leaked trainer) |

**My pre-registration gives two tests that disagree here, and this run found the
gap.** The PASS clause is "peak VRAM under 15.93 GiB", and ~15.1 GiB is inside it.
The FAIL clause is oversubscription "judged by headroom and step-time blowup", and
3.4% headroom with a 2× blowup satisfies that. Naming the ambiguity rather than
picking the reading I prefer.

Two explanations, distinguished by a controlled A/B (`ab_adapter_overhead.py`, rule
fixed before running):

* **H1 adapter compute** — 200 unfused r=16 modules add two matmuls each per token,
  and batch-1 decode is launch-latency bound. Nothing wrong; the arms are unequal
  work. Predicts the slowdown persists with the whole card free.
* **H2 VRAM pressure** — WDDM pages instead of raising, so the model silently
  crawls. This is the pre-registered FAIL.

Primary measure is **seconds per generated token**, not per problem, because a
trained adapter can change where generation stops and per-problem time would then
compare unequal work. Rule: ratio ≥ 1.5 with ≥ 4 GiB free in both arms → H1;
≤ 1.2 → H2; between → inconclusive, no verdict claimed.

A/B result: **«TBD»**. Engineering verdict consequence: **«TBD»**.

A third question is likely to open regardless: an r=16 adapter over 200 modules is
~50 MB, which **cannot** explain a ~9 GiB difference in resident memory. If the A/B
shows the adapter costing under 1 GiB, then the 15.1 GiB during the run has a
separate cause still to be found.

## What is NOT concluded

* **Not** that the pruned checkpoint is usable. The control already showed it
  degenerates on 8/8 prompts and cannot terminate; 500 LoRA steps at r=16 were never
  expected to restore that, and the addendum said so in advance.
* **Not** that a capability gain, if one appears, generalises. n=50, single seed, one
  run per arm, ~2,000 of 7,473 problems seen (about a quarter epoch).
* **Not** that the engineering PASS, if reached, clears the architecture for
  deployment. It clears *Chowder's ability to train this checkpoint*, which is the
  claim the pre-registration makes and the only one it can support.

## Corrections recorded during this run

1. My ETA used the first attempt's 1.20 h baseline figure; the live rate was 1.71 h,
   matching the control. Projected total 3.87 h against the 4.5 h ceiling — had I
   raised the ceiling only to 4.0 h as first considered, this run would have been
   refused mid-eval.
2. I predicted early termination from an accelerating eval rate; it was system
   variance. 0/31 closed `</think>`, 31/31 degenerate. Checking beat speculating.
3. My schedule verifier searched offsets 0 and +1 only, reporting cosine at 0.0397%
   residual. HF logs `get_last_lr()`, so the true offset is −1; adding it turned a
   close fit into an exact one. The verdict was right, my confidence in it was
   under-justified.
4. I flagged a step-1 loss "anomaly" (2.85 vs ~4.8). There was none: the ~4.8 came
   from the level-2 invented-facts runs and was never measured on GSM8K. Retracted
   in `69e27c0`.
5. Both workers recorded `resolved_target_modules` as `sorted(...)` of a regex
   **string**, i.e. its characters — so this run's own provenance recorded a sorted
   list of 99 characters. Fixed and guarded in `69e27c0`.
