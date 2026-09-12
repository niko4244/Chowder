# Result — re-run of the training arm on the pruned 9B

Pre-registration: `PRUNED_9B_REAL_TRAINING_PREREG.md` (`9f66960`) and
`PRUNED_9B_REAL_TRAINING_PREREG_ADDENDUM.md` (`5de632d`), both committed before this
run started. This document's verdict rules were written as a skeleton (`5df9193`)
while the candidate eval was still running, so the numbers were filled into rules
that already existed. First attempt and its failure:
`PRUNED_9B_REAL_TRAINING_RESULT.md`.

Ran 2026-09-11 22:27:39 → 2026-09-12 01:55:39, **208.0 min** wall (3.47 h against the
4.5 h ceiling). Work dir `F:\llm-models\_a4b\realtrain-gsm8k-2`.

## Verdict

| arm | verdict |
|---|---|
| **Engineering** (the primary claim) | **FAIL** — oversubscription, on a reproducible cause |
| **Capability** (secondary) | **RECOVERS** — +0.12, and the gain is not a scoring artifact |

That pair is uncomfortable and it is the honest reading. Every one of the six PASS
conditions was met, training worked, and arithmetic measurably improved — and the
pre-registered FAIL list says "**any of**: … OOM/oversubscription (judged by headroom
and step-time blowup, never by an OOM exception — this platform pages instead of
raising)". The candidate eval ran at **0.56 GiB free with a 1.58–2.0× step-time
blowup**, from a cause reproduced on a clean card. Relabelling that as a PASS because
the rest went well is precisely the post-hoc move the pre-registration exists to
prevent.

### Engineering, condition by condition

| condition | required | measured | |
|---|---|---:|:--|
| run completes | 500 optimiser steps | **500** | ✓ |
| target coverage | 200/200 modules | **200/200** | ✓ |
| live adapter | not inert | written, liveness guard passed | ✓ |
| loss finite and decreasing | yes | 2.8503 → 1.0676, mean 1.2107 | ✓ |
| peak VRAM, training | < 15.93 GiB | **6.14 GiB** | ✓ |
| GSM8K both arms, same protocol | yes | 0.00 / 0.12, identical protocol | ✓ |
| **no oversubscription** | headroom + no blowup | **0.56 GiB free, 1.58–2.0× blowup** | **✗** |

Coverage reproduced the hybrid's real structure from a live model rather than from an
assumption: `q/k/v/o_proj` 8 each (the 8 full-attention layers), `in_proj_qkv` /
`in_proj_z` / `out_proj` 24 each (the 24 linear-attn layers), `gate/up/down_proj` 32
each (all layers) = 200.

**On the pre-registration's internal conflict.** Its PASS list says "peak VRAM under
15.93 GiB", which ~15.1 GiB satisfies; its FAIL list says oversubscription judged by
headroom and blowup, which 0.56 GiB and 1.58× satisfy. I flagged this as ambiguous
before the deciding measurement existed. The plain text settles it without needing a
new rule: the FAIL list is "any of", so a FAIL trigger is sufficient on its own. The
numeric budget was also met *on the leg it describes* — training peaked at 6.14 GiB.

### Capability

| arm | GSM8K (`final_number_match`, pre-registered) |
|---|---:|
| baseline | **0.00** |
| candidate | **0.12** |
| delta | **+0.12** |

Rule: RECOVERS ≥ +0.02; FLAT within ±0.02; DEGRADES < −0.02. → **RECOVERS**.
Gate promoted **`realtrain-unsloth`** (0.12 > the 0.02 minimum gain).

**My pre-registered prediction was wrong.** The addendum said I expected 0.00 on both
sides and an uninformative FLAT, explicitly against the original pre-registration's
expectation of a rise. The original was right. I recorded the worse prediction in
advance precisely so this could be said plainly.

**The arm remained one-sided**, as recorded in advance: with the baseline at the floor
the candidate could not score lower, so DEGRADES was unreachable and this design had
no power to detect harm.

**The gate needed two problems, not one.** `gate.py:67` accepts on `score >
minimum_promotion_gain`, a strict `>`, so one correct problem (0.02) would have been
rejected. The original pre-registration's gloss "one problem in fifty" understates
it. A correction to the gloss, not the threshold.

## The +0.12 is real arithmetic, not the scorer being unlocked

The obvious suspicion: training removed the `<think>` wrapper, the strict rule
discards an unclosed `<think>`, so the baseline was unscoreable while the candidate
was scoreable — a format change masquerading as capability. Tested against two rules
that treat both arms identically:

| arm | strict (pre-registered) | lenient (ignores the wrapper) | pre-loop head |
|---|---:|---:|---:|
| baseline | 0.00 | 0.02 | 0.06 |
| candidate | **0.12** | **0.12** | **0.14** |

So of the +0.12 strict delta, **0.02 is the wrapper effect and +0.10 is genuine**.
"Pre-loop head" takes each response up to its first repeated line — the part a
terminating model would have emitted — and agrees independently: 0.06 → 0.14.

A detail that cuts against the score rather than for it: **degeneration cost a
problem.** Pre-loop 0.14 against final 0.12 means that on one problem the model
reached the right answer and then drifted onto a wrong number before the cap. The
0.12 is slightly deflated by looping, not inflated by lucky repetition.

## And the checkpoint still cannot terminate

| measure | baseline | candidate |
|---|---:|---:|
| opened `<think>` | 50/50 | **0/50** |
| closed `</think>` | **0/50** | **0/50** |
| no think markers at all | 0/50 | 50/50 |
| flagged degenerate | **50/50** | **50/50** |
| mean distinct-trigram ratio | 0.113 | 0.118 |
| mean compression ratio | 0.089 | 0.089 |
| generated tokens (median) | 768 | 768 |
| **hit the 768-token cap** | **50/50** | **50/50** |

**0 of 100 responses across both arms ever terminated.** Every single one ran to the
cap. The degeneration measures are statistically indistinguishable between arms.

What 500 steps bought was the GSM8K *answer format*: the candidate dropped `<think>`
entirely and emits `#### N`. All six correct answers are loops —
`#### 3 | #### 3 | #### 3 …` to the cap — whose first one or two lines are genuinely
correct work, e.g. *"The white fiber takes 2/2=1 bolt / So the total number of bolts
is 2+1=3 / #### 3"*, followed by unbounded repetition.

So training taught this checkpoint to **solve** more problems, not to **stop**. That
is the same conclusion the dense-vs-pruned control reached, now at n=50 rather than
n=8, and the baseline's mid-run numbers (0.098 / 0.087 at n=31) replicate the
control's pruned arm (0.094 / 0.084 at n=8) on an independent sample.

This readout was pre-registered in the addendum as descriptive, not a gate, and no
threshold is placed on it before or after.

## Why the candidate eval oversubscribed: the adapter path, not the adapter

Controlled A/B (`evidence/pruned-real-training/ab_adapter_overhead.py`), 5 problems ×
768 tokens, greedy, from a clean card, decision rule fixed before running:

| arm | ms/token | tokens | cap | torch reserved | card used | outside torch |
|---|---:|---:|---:|---:|---:|---:|
| no-adapter | 112.1 | 3840 | 5/5 | 5.64 GiB | 6.47 GiB | ~0 |
| adapter | **177.6** | 3840 | 5/5 | **5.84 GiB** | **15.37 GiB** | **~8.45 GiB** |

Both arms generated exactly the same number of tokens and hit the cap 5/5, so the
1.58× is per equal work — the seconds-per-token precaution proved unnecessary here,
but it is verified rather than assumed.

**The adapter weights cost +0.20 GiB and do not explain the memory.** ~8.45 GiB
appears *outside torch's accounting* during generation with the adapter attached:
9.31 GiB free immediately after load, 0.56 GiB free while generating. That reproduces
the real run's 15.1 GiB from a clean start, which rules out external contention — it
is the configuration under test.

**My hypothesis framing was wrong and the A/B could not have confirmed H1.** I posed
adapter compute (H1) and VRAM pressure (H2) as exclusive alternatives. They are
coupled: the adapter path *causes* the pressure. My rule required ≥4 GiB headroom in
the adapter arm to credit H1, which is unsatisfiable by construction when the adapter
is what consumes the headroom, so the H1 branch could never fire. The script's
verdict message was additionally wrong on its face — it printed "ratio 1.58 falls
between 1.2 and 1.5" for a ratio of 1.58; the logic was right, the stated reason was
not. Both corrected in `39bee30`.

Reproduction is partial: 1.58× here against 2.0× in the 50-problem run, with the
no-adapter arm (86.1 s/problem) matching the real baseline (83). The longer run was
worse, plausibly fragmentation accumulating over 50 problems — untested.

**Still open:** what allocates the ~8.45 GiB. cuBLAS or bitsandbytes workspaces
outside the caching allocator are the obvious suspects for 400 extra small matmuls
per token, but that is a hypothesis and has not been measured. Until it is, the
practical finding stands on its own: **evaluating with an unfused 200-module adapter
nearly exhausts a 16 GiB card for reasons unrelated to adapter size.**

## The schedule the pre-registration asked for actually ran

The first attempt trained on linear while the recipe said cosine, because
`unsloth_worker` never read `lr_scheduler_type` (`5de632d`). Verified this time from
the realised curve rather than the config, because the config said cosine either way:

| candidate schedule | residual (fraction of 2e-4 peak) |
|---|---:|
| **cosine** | **0.0000 — exact, all 500 steps** |
| linear | 0.0754 |
| constant | 0.6116 |

All 500 logged rates equal `2e-4 · ½(1 + cos(π(s−1)/500))` to the last float bit, at
the offset −1 that HF's `get_last_lr()` logging implies. The final rate is
**1.97e-9** against a 2e-4 peak — nine orders of magnitude down, which no other
candidate schedule produces and which needs no fitting to read. Promoted into
`chowder/schedule_audit.py` with this trajectory committed as a fixture (`6a488b9`).

## The defect that killed the first attempt did not recur

Step **323** — where the first attempt died on an unguarded `progress.tmp` →
`progress.json` rename — passed without incident, and the worker reported
`progress_write_failures: 0`, so the retry path was never needed. That makes the
original `WinError 5` a genuinely transient handle, which is the case the
retry-then-continue design targets.

The two runs agree where they overlap, which independently validates the forensic
recovery from that crash:

| | first attempt (recovered) | this run |
|---|---:|---:|
| step 322 loss | 1.1002 | 1.0984 |
| step 323 loss | 1.0737 | 1.0750 |

## What is NOT concluded

* **Not** that the pruned checkpoint is usable. 0/100 responses terminated. A
  RECOVERS verdict here is about arithmetic, not usability.
* **Not** that the capability gain generalises. n=50, single seed, one run per arm,
  ~2,000 of 7,473 problems seen (about a quarter epoch). One problem is 0.02.
* **Not** that the engineering FAIL means Chowder cannot train this checkpoint. It
  trained it correctly end to end; the FAIL is a resource finding about evaluating
  with an unfused adapter on a 16 GiB card.
* **Not** that the ~8.45 GiB has an identified cause. It has a reproduction, not an
  explanation.

## Corrections recorded during this run

1. My ETA used the first attempt's 1.20 h baseline figure; the live rate was 1.71 h,
   matching the control. Had I raised the GPU-hour ceiling only to 4.0 h as first
   considered, this run would have been refused mid-eval.
2. I predicted early termination from an accelerating eval rate; it was system
   variance — 0/31 closed `</think>`, 31/31 degenerate. Checking beat speculating.
3. My schedule verifier searched offsets 0 and +1 only, reporting cosine at 0.0397%
   residual. HF logs `get_last_lr()`, so the true offset is −1; adding it turned a
   close fit into an exact one. The verdict was right, my confidence under-justified.
4. I flagged a step-1 loss "anomaly" (2.85 vs ~4.8). There was none: the ~4.8 came
   from the level-2 invented-facts runs and was never measured on GSM8K. Retracted in
   `69e27c0`.
5. Both workers recorded `resolved_target_modules` as `sorted(...)` of a regex
   **string**, i.e. its characters — so this run's own provenance recorded a sorted
   list of 99 characters. Fixed and guarded in `69e27c0`.
6. I reported `progress_write_failures` and coverage as absent from the worker result;
   they are top-level fields, not under `telemetry`/`provenance`. I looked in the
   wrong place — the same mistake shape as the earlier coverage-nesting bug.
7. The A/B's hypotheses were not exclusive and its H1 branch was unreachable; its
   verdict message misstated the reason. Both corrected in `39bee30`.
