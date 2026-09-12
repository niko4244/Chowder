# Dense→MoE vs static prune, measured to a negative result — plus ten defects it exposed in Chowder's training path

**Base** `main` (`41a2913`) · **Head** `feat/hot-core-upcycling` (`384a34e`) · **34 commits, 114 files, +14890/-90**

> The previous title (`feat(upcycle): hot-core dense->MoE init, validated against a prior
> prediction`) describes `f948cc1` only — the first of 34. The branch went on to
> measure that init against a simpler alternative, conclude *against* it, withdraw the
> deployability recommendation that followed, and then spend its second half fixing the
> training and evaluation defects it hit while trying to train the resulting checkpoint.

---

## What this branch is

Three arcs, in the order they happened. Each ends in a written verdict, and two of those
verdicts go against the thing the branch was built to support.

1. **Build two dense→MoE/prune converters and rank channels properly.**
   `src/chowder/channel_importance.py`, `src/chowder/hot_core_upcycle.py`,
   `src/chowder/static_prune.py`.
2. **Measure them against each other at equal active compute, and take the loser off the
   table.** `docs/HOT_CORE_VS_STATIC_PRUNE.md`, `docs/HOT_CORE_UPCYCLING.md`,
   `docs/HOT_CORE_HEALING_PILOT_RESULT.md`.
3. **Try to actually train the recommended checkpoint through Chowder's own lifecycle.**
   This is where the defects are: `docs/CAN_CHOWDER_TRAIN_THIS_MODEL.md`,
   `docs/PRUNED_9B_REAL_TRAINING_*`, `docs/PRUNED_9B_RERUN_RESULT.md`.

**Suggested review order:** the three arc-closing docs first
(`HOT_CORE_VS_STATIC_PRUNE.md` → `CAN_CHOWDER_TRAIN_THIS_MODEL.md` →
`PRUNED_9B_RERUN_RESULT.md`), then the defect table below, then the converters. Every number
in this description is quoted from a doc or an `evidence/` artifact in the branch.

---

## Arc 1 — the converters

### `channel_importance.py`

Sums `|h|` per FFN intermediate channel (`h = silu(gate(x)) * up(x)`, the vector `down_proj`
consumes) over a calibration split, via a forward hook on the unmodified parent. The ranking
is an artifact with its own digest, bound to one checkpoint by `source_manifest_sha256`.

Why it exists: at 28% of channels kept on the real 9B, an **index-ordered** subset scores
**614.7×** baseline perplexity; an **activation-ranked** subset of the same size scores
**1.69×** (`src/chowder/channel_importance.py` docstring,
`docs/HOT_CORE_UPCYCLING.md`). Ranking — not sparsity — is what collapsed the earlier 27B
ladder.

`spread_across()` (`channel_importance.py:309`) exists because *where* the calibration split
comes from matters more than routing did: a 32-prompt **contiguous** split cost 1.691× near
it and 3.646× far away; 32 prompts **spread** over the same corpus cost 1.859× and 3.011×
(17.4% better out of distribution), sharing only **73.5%** of their top-3,440 channels
(`docs/HOT_CORE_HEALING_PILOT_RESULT.md` Addendum 2).

### `hot_core_upcycle.py`

Hot core → `shared_expert` (computed once, so active cost equals coverage), cold channels →
routed experts dealt round-robin by rank, cold `down_proj` pre-scaled ×`top_k`, shared
`down_proj` ×2 because a zero-init gate gives exactly `sigmoid(0) = 0.5`. `top_k` must be a
power of two. Provenance records `exactness_contract: "NONE"` — this init is explicitly not
exact, and says so.

Validated on `Qwen3.8-9B-abliterated-25-bf16` → E=16, `top_k`=2, core 2176, cold 632/expert,
3,440 of 12,288 channels active (f = 0.2799, 63.3% core):

| | |
|---|---|
| dense parent, nf4, 32 held-out prompts | ppl **5.3137** |
| converted, same load and prompts | ppl **9.8444** |
| ratio | **1.853×** |
| pre-existing mask-sweep prediction for f=0.28 at 50–75% core | **1.81×–2.09×** |
| conversion time / output | 917 s / 18.82 GB, loads in stock transformers |
| stored FFN / active FFN | 4.832B (**1.000×** dense) / **1.353B** |

The prediction was made before this converter existed, from a different code path. Byte-level
check: all **664** non-consumed source tensors byte-identical, including all **108**
vision-tower `.mlp.` tensors; the 96 dense decoder MLP tensors consumed and absent.

Two things it does **not** buy, recorded in the doc rather than omitted: **no memory saving**
(peak nf4 allocation dense 5.72 GiB vs converted **15.07 GiB**, because the raw-`nn.Parameter`
expert bank never quantises) and **no demonstrated capability**.

### `static_prune.py`

Deletes the unranked channels instead of routing them. No scaling anywhere, no architecture
change — still `qwen3_5` with a smaller `intermediate_size`. Built artifact
`Qwen3.8-9B-Pruned-CW-3440`: 5.931B total = active, **11.86 GB** on disk, **5.45 GiB** peak
nf4 — less than the dense parent. Byte check: 760 tensors in, 760 out, 664 non-MLP
byte-identical.

---

## Arc 2 — the conclusion, and the withdrawal

### Static pruning wins on perplexity

`docs/HOT_CORE_VS_STATIC_PRUNE.md`. Design held identical across both checkpoints, so the
ranking is the only variable.

| | eval A | eval B |
|---|---:|---:|
| dense parent | 5.2707 | 4.3044 |
| static prune, corpus-wide ranking (built checkpoint) | **9.9235** | **13.0258** |
| hot-core MoE init, corpus-wide ranking | 13.4224 | 15.2936 |
| hot-core MoE, contiguous, after 150 healing steps | 9.2324 | 14.9314 |

Geometric mean across both splits: static **2.387×** dense (real checkpoint; 2.366×
mask-emulated) against the MoE's **3.008×** — static is **26.0%** better at identical active
compute, with no router, no healing, and a smaller checkpoint.

Why, and it was in the data all along: the MoE spends **2,176 of its 3,440** active channels
on a fixed core and only 1,264 on routed choice, so at init it trades ranks 2,176–3,439 for a
scattered sample of cold ones. The core-share sweep had already said this (f=0.28: 100% core
1.69×, 75% 1.81×, 25% 3.06×, 0% 214×) — **100% core *is* static pruning**, and the doc
records that the monotone result was read as a starting point instead of a conclusion.

A second self-correction in the same arc: the router-healing pilot's headline
"beats static by **4.85%** on eval B" is **withdrawn** — against a corpus-wide-ranked static
baseline the same trained MoE **loses by 13.20%** (`HOT_CORE_HEALING_PILOT_RESULT.md`
Addendum 2). The pilot's own pre-registered verdict was **ATTRIBUTION FAIL**: on the
pre-registered primary split the router contributed **−0.03** and all the gain came from the
shared-expert gate. The doc also records that making eval A primary was a
pre-registration *design* error, and declines to promote eval B after seeing the numbers.

### …and the deployability recommendation is withdrawn too

A paired control on the real models — same 8 GSM8K prompts, same settings, same scorer, same
nf4 load, only the weights differing — shows the pruned checkpoint **cannot terminate**:

| | dense | pruned (f=0.28) |
|---|---:|---:|
| GSM8K (`final_number_match`) | **0.375** (3/8) | 0.125 (1/8) |
| mean distinct-trigram ratio | 0.641 | **0.094** |
| mean compression ratio | 0.344 | **0.084** |
| degenerate | 1/8 | **8/8** |
| hit the 768-token cap | 37.5% | **100%** |

Evidence: `evidence/pruned-real-training/control-dense-vs-pruned-generation.json`. **Pruning
is the cause, not the harness** — the dense parent runs the identical path and terminates on
most prompts. The perplexity result above stands *as perplexity*; perplexity simply does not
predict whether a checkpoint can stop. The pruned 0.125 is itself an artifact: one degenerate
loop that happened to end on the right number.

**Net reviewable claim:** static pruning beats the hot-core MoE on perplexity at equal active
compute, **and neither artifact has been shown usable for generation** (n=8, and the MoE
checkpoint was never measured for degeneration at all).

---

## Arc 3 — ten defects in Chowder's training/eval path, each with tests

Opening the gated real-ML suites is what found most of these. Those tests sit among the
**77 skipped** on a normal run, so "1448 passed" had never shown that Chowder trains
(`docs/CAN_CHOWDER_TRAIN_THIS_MODEL.md`).

| # | defect | fix | commit | tests |
|---|---|---|---|---|
| 1 | A worker subprocess imports whatever `.pth` the editable install points at, so parent and child can run **different Chowders** — a worktree run trained and evaluated against main's code | `src/chowder/worker_env.py` | `b7c1710` | `tests/test_worker_env.py` (8) |
| 2 | Scoring an adapter that **cannot change the model**. `PeftModel.from_pretrained` succeeds with zero key overlap; `adapter_loaded` in provenance was literally `spec.adapter_dir is not None` | `src/chowder/adapter_guard.py`, called at all four load sites | `3621965` | `tests/test_adapter_guard.py` (8) |
| 3 | **Root cause** of #2 on this architecture: Unsloth loads the full `Qwen3_5ForConditionalGeneration` (layers under `model.language_model.`) while the evaluator loads the text-only CausalLM, so every adapter key mismatched — **max logit delta 0.000000** vs 14.5 for a Transformers adapter | `FastLanguageModel.from_pretrained(text_only=True)`, version-guarded in provenance | `bf190e6` | `tests/test_unsloth_peft_real.py` |
| 4 | **Unverified target coverage.** PEFT raises only when *nothing* matches; a ten-name list silently adapted **128 modules instead of 200** (all 72 `linear_attn` modules skipped) and the gate promoted it | `src/chowder/target_coverage.py` | `81cfe05` | `tests/test_target_coverage.py` (16) |
| 5 | Unsloth turned an explicit target list into a regex that missed the Mamba-style layers | pass a suffix-match regex so the full list is honoured | `0bd62ed` | `tests/test_project_runner_repair_unsloth.py` |
| 6 | A **telemetry rename killed a 500-step run at step 323**: `PermissionError: [WinError 5]` on `progress.tmp → progress.json` inside `TrainerCallback.on_log` propagates out of `Trainer.train()`, discarding 323 steps | `src/chowder/progress_write.py` — retry briefly, then keep training; failures surfaced as `progress_write_failures` | `e2e6df1` | `tests/test_progress_write.py` (7) |
| 7 | **Two text scorers had silently diverged.** Baseline ran through `base_text_worker` (discards an unclosed `<think>`), candidate through `transformers_text_worker` (scores raw text) — so the two sides of one comparison used different rules under one name | `src/chowder/evaluators/scoring.py`; both workers hold `_score = score`, the same function object | `a9cd0ad` | `tests/test_scorer_agreement.py` (23), `tests/test_final_number_scoring.py` (16) |
| 8 | `unsloth_worker` **never read `lr_scheduler_type`** (nor `warmup_ratio`/`warmup_steps`) — the key was validated by the spec and honoured by `transformers_worker`, so a pre-registered **cosine** recipe trained on **linear** and nothing said so | read and pass it in both spec and worker | `5de632d` | `tests/test_lr_scheduler_honoured.py` (10), mutation-verified source check |
| 9 | Both workers recorded `resolved_target_modules` as `sorted(...)` of a regex **string** — i.e. a sorted list of its 99 characters — in the run's own provenance | keep the regex out of `sorted()`; also retracts an unmeasured loss figure | `69e27c0` | guarded in the same commit |
| 10 | **Evaluators recorded no VRAM at all.** A pre-registered peak-VRAM condition was therefore undecidable for the evaluation leg, and the only proxy (`nvidia-smi`) measures the whole machine — a busy desktop could fail an experiment | `src/chowder/evaluators/vram.py` reports `peak_vram_gb` and `peak_vram_reserved_gb`, `None` for unknown, never raises | `52cba56` | `tests/test_evaluator_vram_reporting.py` (5) |

Supporting test-infrastructure work: `61c9409` makes both real Unsloth tests **runnable and
re-runnable** (`tests/unsloth_env_link.py`); the gated Transformers smoke went 1/4 → **4/4**.

### What those fixes bought, measured

`docs/CAN_CHOWDER_TRAIN_THIS_MODEL.md`, on the real 5.937B pruned hybrid via
`project_runner.run_project` (production lifecycle: registry, automatic baseline, protocol
binding, worker, gate):

| engine | loss | peak VRAM | quality (baseline 0.30) | gate |
|---|---|---:|---:|---|
| Transformers, 200/200 modules | 4.8354 → 0.3767 (50 steps) | 11.66 GB | 0.45 | **not promoted** — +0.15 below the `minimum_promotion_gain: 0.2` set before the run |
| Unsloth, before the `text_only` fix | 4.4014 → 0.3760 | 6.24 GB | 0.30 — **exactly baseline**, predictions byte-identical | rejected |
| Unsloth, after | — | 5.84 GB | **0.60** | promoted |

The 0.60-vs-0.45 gap is **not** a controlled comparison (n=20, different engines, one run
each) and the doc says so.

### `schedule_audit.py` — verify the schedule from the curve, not the config

A source check cannot prove the optimiser followed a curve (defect #8 passed every config
assertion for months). `src/chowder/schedule_audit.py` identifies the schedule from the rates
a run actually logged, and returns `INCONCLUSIVE` rather than guessing when the trajectory
carries no information. The real 500-step trajectory is committed as a fixture:
`tests/data/lr-trajectory-pruned9b-cosine-500.jsonl` (500 rows), with
`tests/test_schedule_audit.py` (21 tests).

On the re-run: **cosine residual 0.0000 — exact, all 500 steps**, against linear 0.0754 and
constant 0.6116; final logged rate **1.97e-9** against a 2e-4 peak.

---

## The two pre-registered experiments, with their results

### Attempt 1 — `PRUNED_9B_REAL_TRAINING_PREREG.md` (`9f66960`, before the run)

Ran 89.5 min. **Engineering FAIL** (did not complete — defect #6 at step 323/500) and
**Capability VOID** (defect #7 meant the two arms were not scored by the same rule, which the
pre-registration required). Both verdicts are the pre-registered labels, not post-hoc ones;
the doc states plainly that the run "failed **by finding a real defect in Chowder**, which is
a useful outcome but not a PASS".

Forensics survived the crash: `progress.json` step 322 loss **1.1002**, `progress.tmp` step
323 loss **1.0737** — payload correct, only the rename failed. Baseline degeneration at n=50:
median distinct-trigram **0.079**, zlib **0.080**, **48/50** flagged, unclosed `<think>`
**50/50**.

`PRUNED_9B_REAL_TRAINING_CORRECTION.md` (`785ce94`) was written **mid-run**, prompted by an
independent review, and records that the author's own earlier claim ("0.00 on the first 28")
fused two things of which only one survives — the repetition is real, the 0.00 was a scorer
artifact. It also records that a metric of the author's own (duplicate-**line** ratio)
undercounted degeneration at 41/50 where line-agnostic measures flag 48/50, and that the
control script was corrected **before** running rather than after.

### Attempt 2 — `PRUNED_9B_REAL_TRAINING_PREREG_ADDENDUM.md` (`5de632d`, before the re-run)

No outcome threshold altered; the GPU-hour ceiling was raised 2.0 → 4.5 from measured leg
costs, with the reasoning for why a resource ceiling is not an outcome threshold. Result
skeleton committed at `5df9193` *while the candidate eval was still running*, so the verdict
rules predate the numbers — guarded by `tests/test_result_docs_are_not_drafts.py` (5 tests),
which forbids a placeholder surviving the removal of the SKELETON banner.

`PRUNED_9B_RERUN_RESULT.md`: 208.0 min (3.47 h against the 4.5 h ceiling). 500 steps,
coverage 200/200, loss **2.8503 → 1.0676**, peak training VRAM **6.14 GiB**,
`progress_write_failures: 0`. The two runs agree where they overlap (step 323: 1.0737
recovered vs 1.0750), which independently validates the forensic recovery.

**Capability RECOVERS, +0.12** (0.00 → 0.12), and tested against the obvious objection that
it is the scorer being unlocked rather than arithmetic:

| arm | strict (pre-registered) | lenient | pre-loop head |
|---|---:|---:|---:|
| baseline | 0.00 | 0.02 | 0.06 |
| candidate | **0.12** | **0.12** | **0.14** |

0.02 of the delta is the wrapper effect, **+0.10 is genuine**; pre-loop 0.14 vs final 0.12
means looping *cost* a problem rather than manufacturing one. And the checkpoint still cannot
stop: **0 of 100 responses across both arms terminated**, all 100 hit the 768-token cap.
Training taught it to *solve* more, not to *stop*.

### The two reversals a reviewer should look at hardest

1. **A pre-registered prediction of the author's was wrong and is recorded as wrong.** The
   addendum predicted "0.00 on both sides — FLAT — uninformative", explicitly against the
   original pre-registration's expectation of a rise. The original was right.
2. **An Engineering FAIL was withdrawn to PASS, qualified.** The FAIL rested on a
   whole-machine `nvidia-smi` reading standing in for a per-process figure the evaluators
   never recorded (defect #10). Controlled re-measurement — one arm per process,
   `torch.cuda.mem_get_info()` — showed the slowdown is adapter compute (**1.54–1.83×** with
   **~8.9 GiB free throughout**) and put the eval process's own footprint at **~6.6 GiB**,
   measured four ways. Five mechanisms eliminated; the remaining ~9 GiB was held by something
   outside the experiment that **could not be identified**, and the tidiest candidate was
   dropped for lack of evidence rather than left standing on plausibility.

   The doc marks the PASS **qualified** because the withdrawal rests on post-hoc measurement
   of the same configuration, not on the run's own artifacts, and states the symmetry test
   applied beforehand: *would the opposite outcome have been accepted?* It also lists nine
   corrections made during the run, four of which are errors of the author's own (including an
   order-dependent test of theirs, and an A/B whose verdict message misstated its own reason).

---

## What is explicitly **not** claimed

- **Not** that MoE routing is worthless — one architecture, one model, one active budget
  (f=0.28), perplexity rather than GSM8K, ~25k healing tokens.
- **Not** that the static→oracle gap is closed or refuted: the oracle at f=0.28 is 1.03×
  against static's 1.859×, and nothing here captures that.
- **Not** that either artifact is usable. 0/100 generations terminated.
- **Not** that the +0.12 generalises: n=50, single seed, one run per arm, ~2,000 of 7,473
  problems (a quarter epoch). One problem is 0.02.
- **Not** that the qualified PASS is a clean one, and **not** that the ~9 GiB has an
  identified cause — it has five eliminated causes and no positive explanation.

## Risk and blast radius

- **Behaviour change to review closely:** the unified strict scoring rule
  (`evaluators/scoring.py`) scores reasoning models **lower** than the old lenient
  `transformers_text_worker` path — an unclosed `<think>` now means no answer span. It was
  declared in the addendum in advance, in the direction that disfavours the candidate. Any
  historical number produced through the candidate path is on the old rule.
- **Historical results to re-check, per `CAN_CHOWDER_TRAIN_THIS_MODEL.md`:** any previous
  Unsloth result on a model with the `language_model.` wrapper shape may have been scored as
  the base model (defect #3), and any run whose target list partially matched may have trained
  less than it asked for (defect #4). Both failures were silent.
- **Additive surface:** `worker_env`, `adapter_guard`, `target_coverage`, `progress_write`,
  `scoring`, `vram`, `schedule_audit`, `channel_importance`, `hot_core_upcycle`,
  `static_prune` are all new modules; the touched existing files are the two training workers,
  the text evaluators and their backends, plus `kaggle_equivalence` (re-pointed at `scoring`
  so the extraction helpers have one owner).
- `allow_unmatched_target_modules: true` remains the deliberate opt-in for partial coverage;
  it is no longer needed for this architecture.

## Reviewer note on commit count

`git log main..HEAD` shows **49** commits against a local `main` that is 19 commits behind the
remote. Against the PR's actual base (`origin/main` = `41a2913`, which already carries
PRs #154–#157) the range is **34 commits, 114 files, +14890/-90** — GitHub reports the same
count. The router-healing modules (`router_healing*.py`, `parent_freeze.py`, `dense_to_moe.py`
changes) are **already on `main`** and are not part of this review.

---

## Closing result, added after the draft

`docs/PRUNE_FRACTION_GENERATION_SWEEP.md` — swept five prune fractions for
**generation**, masking the channels a prune would delete with the same corpus-wide
ranking the built checkpoints use:

| f | keep/layer | ×dense ppl | GSM8K | terminated | degenerate | verdict |
|---:|---:|---:|---:|---:|---:|---|
| 1.00 | 12288 | 1.00 | 0.375 | 5/8 | 2/8 | **SURVIVES** |
| 0.75 | 9216 | 1.03 | 0.250 | 4/8 | 0/8 | **SURVIVES** |
| 0.5625 | 6912 | 1.09–1.14 | 0.250 | 1/8 | 7/8 | FAILS |
| 0.40 | 4915 | 1.19 | 0.000 | 0/8 | 8/8 | FAILS |
| 0.28 | 3441 | 1.48–1.69 | 0.000 | 0/8 | 8/8 | FAILS |

Perplexity moves **1.03× → 1.14×** across the interval where termination collapses
**4/8 → 1/8**. So on this model, choosing a prune fraction by perplexity selects a
checkpoint that cannot stop — the metric is not merely uninformative about termination,
it is actively misleading. Both pre-registered anchors behaved: f=1.00 reproduced the
independent dense control (0.375/5-of-8/0.641 against 0.375/5-of-8/0.6406) and f=0.28
reproduced the built checkpoint's degeneration, which is the reason to believe the
fractions between them. n=8 per arm, so the cliff's existence is robust and its exact
position between 0.75 and 0.5625 is not.
