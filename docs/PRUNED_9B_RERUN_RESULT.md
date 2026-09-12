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
| **Engineering** (the primary claim) | **PASS, qualified** — see the withdrawal below |
| **Capability** (secondary) | **RECOVERS** — +0.12, and the gain is not a scoring artifact |

> **This was first recorded as Engineering FAIL, and the FAIL is withdrawn.** It is
> the only verdict revised in this document, the reasons are below in full, and the
> reasoning deserves scepticism because flipping a FAIL to a PASS is exactly the
> post-hoc move a pre-registration exists to prevent.
>
> The test I applied to myself: *would I have accepted the opposite outcome?* Had the
> controlled probes shown the adapter genuinely consuming 15 GiB, I would have
> recorded the FAIL as confirmed and said so. The procedure was symmetric before the
> data arrived, which is the only thing that distinguishes a correction from a
> rationalisation.
>
> **PASS is "qualified" and not clean.** All six pre-registered conditions were met,
> but the oversubscription clause turned out to be *undecidable from the run's own
> artifacts*, so its withdrawal rests on post-hoc controlled measurement of the same
> configuration rather than on evidence the run itself produced. An unqualified PASS
> would require re-running the evaluation leg on a verified-idle card with the
> instrumentation now added (`52cba56`). That has not been done.

### Engineering, condition by condition

| condition | required | measured | |
|---|---|---:|:--|
| run completes | 500 optimiser steps | **500** | ✓ |
| target coverage | 200/200 modules | **200/200** | ✓ |
| live adapter | not inert | written, liveness guard passed | ✓ |
| loss finite and decreasing | yes | 2.8503 → 1.0676, mean 1.2107 | ✓ |
| peak VRAM, training | < 15.93 GiB | **6.14 GiB** | ✓ |
| GSM8K both arms, same protocol | yes | 0.00 / 0.12, identical protocol | ✓ |
| no oversubscription | headroom + no blowup | **undecidable from run artifacts** | — |

Coverage reproduced the hybrid's real structure from a live model rather than from an
assumption: `q/k/v/o_proj` 8 each (the 8 full-attention layers), `in_proj_qkv` /
`in_proj_z` / `out_proj` 24 each (the 24 linear-attn layers), `gate/up/down_proj` 32
each (all layers) = 200.

**On the pre-registration's internal conflict.** Its PASS list says "peak VRAM under
15.93 GiB"; its FAIL list says oversubscription judged by headroom and blowup. I
flagged the tension before the deciding measurement existed. The resolution turned out
to be more basic than choosing between them: **neither quantity existed in the
evaluation leg's artifacts.** The evaluator workers recorded no VRAM at all, so both
clauses were undecidable for that leg and I substituted `nvidia-smi`, which measures
the whole machine. Fixed in `52cba56`; see the withdrawal below.

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

## Withdrawn: the candidate eval did not oversubscribe

The trigger was that the candidate arm ran at 166 s/problem against the baseline's 83
while `nvidia-smi` showed 559 MiB of card free. Two things turned out to be true: the
slowdown is real and explained, and the headroom figure was never a measurement of
this run.

### The slowdown is adapter compute

Probed with `torch.cuda.mem_get_info()` (driver-level free memory from inside the
process) and **one arm per process**, at five token budgets:

| budget | no-adapter | adapter | ratio | adapter non-torch | driver free |
|---:|---:|---:|---:|---:|---:|
| 1 | 887.0 | 923.2 | 1.04 | 1.15 GiB | 9.01 GiB |
| 16 | 124.2 | 191.7 | **1.54** | 1.15 GiB | 8.96 GiB |
| 64 | 111.3 | 172.8 | **1.55** | 1.15 GiB | 8.90 GiB |
| 256 | 108.8 | 167.9 | **1.54** | 1.15 GiB | 8.90 GiB |
| 768 | 105.9 | 193.5 | **1.83** | 1.15 GiB | **8.88 GiB** |

A stable 1.54–1.83× **with ~8.9 GiB free throughout**. 200 unfused LoRA modules add
two matmuls each per token, and batch-1 decode is launch-latency bound, so the arms
are simply unequal work. The 1-token row is prefill-dominated and uninformative.

### The ~8.45 GiB has no reproduction, and five mechanisms are eliminated

| candidate mechanism | verdict |
|---|---|
| adapter weights | **no** — +0.15 GiB at 768 tokens |
| token count | **no** — non-torch flat 1 → 768 tokens |
| accumulation across problems | **no** — reserved plateaus at 5.99 GiB by problem 4 and is identical for the next eight; free plateaus at 8.79 GiB; ms/tok shows no progressive slowing |
| PEFT forcing a non-specialised cache | **no** — `DynamicCache` in both arms |
| nvidia-smi vs mem_get_info disagreeing | **no** — 26 interleaved samples agree within 0.33 GiB, with nvidia-smi reading slightly *more* free |

The evaluation process's own footprint is **~6.6 GiB**, measured four ways, against a
15.93 GiB budget. The remaining ~9 GiB during the run was held by something outside
the experiment which I **could not identify**. The tidiest candidate — the user's
Ollama, which runs with `KEEP_ALIVE 5m` and would have unloaded by morning — fits the
facts but has **no supporting evidence**: no log entries on either date, newest server
log from June 30. Dropped rather than left standing on plausibility.

### The actual defect, now fixed

**The evaluator workers recorded no VRAM at all.** The training workers always have.
So a pre-registered peak-VRAM condition was undecidable for the evaluation leg, and
the only available proxy measured the whole machine — every browser and service on it.
A busy desktop was able to fail an experiment.

`chowder/evaluators/vram.py` (`52cba56`) now reports both `peak_vram_gb` (allocated —
what the model needed) and `peak_vram_reserved_gb` (what torch took from the driver —
what decides whether a run fits alongside anything else), from both arms, with
unknown reported as `None` rather than 0.0, and never raising. Mutation-verified.

### What I got wrong along the way

* The A/B ran both arms in **one process**, so its adapter arm could never hold the
  4 GiB of headroom its own decision rule demanded for H1. I declared H1 unreachable;
  it was reachable, just not by that design. One arm per process showed it plainly.
* I framed adapter compute and VRAM pressure as exclusive alternatives. They were
  neither exclusive nor both real.
* The A/B's verdict message said "ratio 1.58 falls between 1.2 and 1.5" for a ratio of
  1.58 — the logic was right, the stated reason false. Fixed in `39bee30`.
* I did not notice until late that **instrument and condition had changed together**:
  every reading showing a full card used nvidia-smi, every reading showing free memory
  used mem_get_info. That confound was mine to catch at design time.

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
* **Not** that the qualified PASS is a clean one. The oversubscription clause was
  undecidable from the run's artifacts, and its withdrawal rests on post-hoc
  controlled measurement of the same configuration, not on the run's own evidence.
* **Not** that the ~9 GiB has an identified cause. It has five *eliminated* causes and
  no positive explanation, and it did not come from the experiment.

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
7. The A/B's hypotheses were not exclusive and its H1 branch was unreachable by
   construction; its verdict message misstated the reason. Corrected in `39bee30`.
8. **The Engineering FAIL was withdrawn.** It rested on a whole-machine `nvidia-smi`
   reading standing in for a per-process figure the evaluators never recorded. Five
   mechanisms eliminated, footprint ~6.6 GiB measured four ways, root defect fixed in
   `52cba56`. This is the one verdict revised here, and it is marked qualified because
   the withdrawal is post-hoc rather than from the run's own evidence.
9. A test of mine was order-dependent: it asserted `peak_vram("cuda:99")` returns
   `None`, which passed in the full suite and failed standalone, because torch rejects
   an out-of-range ordinal only once CUDA has been initialised by an earlier test. It
   now drives the defensive path by making torch raise.
