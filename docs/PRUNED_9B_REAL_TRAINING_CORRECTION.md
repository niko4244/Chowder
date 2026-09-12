# Correction — the capability arm of the real training run is void

Written 2026-09-11 ~20:40, while the run is still in flight. Prompted by an
independent review (Codex) that found a scorer asymmetry in code I wrote; I
verified every claim myself before accepting it, and the verification found the
problem to be **worse** than reported.

Pre-registration: `docs/PRUNED_9B_REAL_TRAINING_PREREG.md`, committed `9f66960`
before the run started. It required that GSM8K be "measured on both base and
trained through the identical protocol."

## The defect: the two sides of the run score by different rules

`final_number_match` was added to **both** text-evaluator workers. Their
pre-existing `_score` bodies already differed, and I did not check that before
adding a branch to each:

* `base_text_worker._score` calls `_final_answer()` first, which returns `""`
  when a `<think>` block is opened and never closed.
* `transformers_text_worker._score` extracts from the **raw** prediction.

Verified on CPU, same inputs, expected answer `18`:

| prediction | base rule | candidate rule |
|---|---:|---:|
| `<think>18` (unfinished) | **0** | **1** |
| `<think>work</think>18` | 1 | 1 |
| `18` | 1 | 1 |

The automatic baseline runs through the base-model evaluator; the candidate runs
through the adapter evaluator. So this run's two sides are scored by different
rules under one name, which violates the pre-registered protocol requirement.

## Why it is worse than an edge case here

**All 41 baseline predictions so far contain an unfinished `<think>` block.** The
model never closes it. So `_final_answer()` returns `""` for every row and the
baseline scores 0.00 *by construction*, independent of content:

| | value |
|---|---:|
| baseline rows scored | 41 |
| recorded score (base rule, as run) | **0** |
| same rows re-scored with the candidate rule | **1** |
| predictions containing `<think>` | 41 (all unfinished) |

The baseline is therefore **not a measurement**. It is a structurally fixed zero.
Comparing a candidate scored on raw text against a structurally-zero baseline is
biased toward showing improvement — the dangerous direction, because it could
manufacture a PASS against a threshold of one problem in fifty.

## Correcting what I told you earlier

I reported: *"the pruned model degenerates into repetition loops — 0.00 on the
first 28 problems."*

Two separable claims were fused there, and only one survives:

* **The repetition is real.** It is visible in the raw generations and it is why
  every problem burns the full 768-token budget.
* **The 0.00 is an artifact.** It is the scorer discarding unclosed reasoning, not
  the scorer judging the content wrong. I presented a scorer artifact as a model
  measurement.

## A second flaw the real data exposed, in the scorer I designed

Row 4 of the baseline disagrees between the rules because the degenerate loop
happens to end on the correct number:

```
expected 20 | base 0 | candidate 1
tail: " 20 = 20.\nThe equation is x * 20 = 20.\nThe equation is x * 20"
```

**"Last number wins" gives credit to a degenerate loop by luck.** The finding that
motivated the rule (`FINDINGS-GSM8K-EVAL-BUG.md`) warned about *truncation* —
reasoning cut off before the answer. This is the opposite failure: output that is
not truncated but never terminates, where last-number-wins rewards repetition.
Stripping unclosed `<think>` fixes the case seen here; a degenerate loop with no
`<think>` wrapper would still be credited, and that residual weakness is recorded
rather than patched speculatively.

## Verdict on this run

* **Capability arm: VOID, not FAIL.** The pre-registered protocol-identity
  requirement was not met, so the comparison cannot be interpreted either way.
  "Void" is the honest label: a FAIL would imply the model was measured and found
  wanting, which is precisely what did not happen.
* **Engineering arm: still live and still meaningful.** Loss trajectory, 200-module
  coverage, adapter liveness, peak VRAM, registry rows, actual GPU-hours and the
  gate's own behaviour do not depend on the scorer. Those were the primary claim,
  and they will be judged against the pre-registered PASS/FAIL conditions.

## Two further pre-registration deviations found in the same review

1. **Scheduler.** The recipe says "lr 2e-4 cosine". The project config omits
   `lr_scheduler_type`, so the trainer default (linear) applies. Recorded as a
   deviation; not changed mid-run.
2. **Budget.** `gpu_hour_budget` is 2.0, with 0.5 reserved for the experiment and
   0.1 for its evaluation. The baseline eval alone is tracking toward ~1.4 h at
   ~1.7 min/problem, so the remaining reservation may not fit and the ledger may
   refuse before training. **If it refuses, that is correct behaviour and will be
   reported as such** — the budget will not be raised, the eval will not be
   shortened, and the run will not be restarted to dodge it.

Also corrected: the session handoff named `level2-report.json` as the output. The
driver writes **`realtrain-report.json`**.

## What is deliberately NOT being done now

**No evaluation code is being changed while an evaluation is pending.** The
candidate eval has not started; editing `transformers_text_worker.py` now would
silently score the candidate with code that did not exist when the baseline ran.
That would trade one protocol violation for a worse one. The fix waits for the run
to terminate.

## Planned fix, after the run ends

1. Make both workers share one scoring implementation, so a divergence of this
   kind cannot recur by editing one file.
2. The base worker's behaviour is the correct one: an unclosed `<think>` means no
   answer was produced, which is **incorrect**, not "extract whatever number is
   lying around." The candidate worker moves to that rule.
3. A test asserting the two workers agree on a shared case table, so the next
   scoring mode added to one of them cannot quietly diverge.
4. Record the residual degenerate-loop weakness in the scorer's own docstring.

## Done (2026-09-11, after the run terminated and the control completed)

All four, in `src/chowder/evaluators/scoring.py`. No evaluation was pending: the
training run had already terminated FAILED and the dense-vs-pruned control had
finished, so nothing was re-scored mid-flight.

Both workers now hold `_score = score` against that one module -- the same function
object, not two copies that agree today. `tests/test_scorer_agreement.py` guards it
three ways: **identity** (both names resolve to the shared function), **behaviour**
(an 18-case table asserted through both workers, including the real degenerate
generation), and **source** (neither worker file may re-declare `_score`,
`_final_number`, `_final_answer`, `_normalize`, or its own `_FINAL_NUMBER` regex).
The source guard is mutation-verified: pasting a local lenient `_score` back into
`transformers_text_worker` fails 13 of the file's tests.

`kaggle_equivalence` imported the extraction helpers *from the base worker* to avoid
drift; it now imports them from `scoring`, which is the module that actually owns
the rule.

**The effect on the real recorded run, measured rather than assumed.** Re-scoring
all 50 baseline predictions under the unified rule:

| | |
|---|---:|
| responses with no answer span at all (unclosed `<think>`) | **50/50** |
| recorded score | 0/50 |
| unified-rule score | 0/50 |
| what the old lenient rule would have scored correct | **1/50** |

So the baseline stays 0.00 — but now for the right reason, and symmetrically. The
interesting number is the last row: the lenient rule would have manufactured **one
correct answer out of 50 responses that contain no answer**. That is the same
artifact, at the same small-n rate, as the pruned checkpoint's 0.125 in the control
while degenerate on 8 of 8 prompts.
