# Addendum to the pre-registration — re-run of the training arm

Written and committed **before** the re-run starts. The original
`PRUNED_9B_REAL_TRAINING_PREREG.md` stands; this records only what changed since the
first attempt FAILED at step 323, and why. **No outcome threshold is altered.** The
engineering PASS/FAIL conditions and the ±0.02 capability bands are exactly as
pre-registered, including the one that makes me call it a failure.

## What changed in the code under test, and why each is a fix not a tuning

1. **Progress publishing can no longer kill the run** (`chowder/progress_write.py`).
   This is the defect that ended the first attempt: a telemetry rename raised
   `PermissionError: [WinError 5]` inside `TrainerCallback.on_log`, which propagates
   out of `Trainer.train()`, discarding 323 steps of working training.

2. **One scoring rule for both workers** (`chowder/evaluators/scoring.py`). The
   baseline ran through `base_text_worker` and the candidate through
   `transformers_text_worker`, and the two disagreed on an unclosed `<think>`. Both
   now use the strict rule: no closing marker means no answer span, which scores as
   incorrect. **This is a protocol change relative to the first attempt** and is
   declared here rather than discovered afterwards. It makes the two sides
   comparable, which is the precondition for the capability arm meaning anything.

   Direction of the effect, stated in advance so it cannot be spun later: the strict
   rule scores this model **lower**, not higher. On the first attempt's 50 baseline
   predictions the lenient rule would have scored 1 correct out of 50 responses that
   contain no answer at all.

3. **The cosine schedule the original pre-registration specified now actually
   applies.** The first attempt trained on linear. I recorded that as "the config
   omits `lr_scheduler_type`", which was wrong about the cause: the key *was*
   validated by the shared spec and honoured by `transformers_worker`, but
   `unsloth_worker` never read it and never passed it to `TrainingArguments`. On the
   Unsloth engine the key would have been accepted and silently dropped. Fixed in
   both the spec and the worker, with `warmup_ratio` / `warmup_steps`, which were
   dropped the same way. Guarded by `tests/test_lr_scheduler_honoured.py`, whose
   source check is mutation-verified.

   So the first attempt's 323 steps were on a schedule the recipe did not ask for.
   That does not change its FAIL verdict, which was about not completing.

## The GPU-hour ceiling is raised 2.0 → 4.5, and this is not threshold-moving

The original 2.0 h was a session-bounding number attached to an estimate that was
wrong by ~18× on the evaluation leg. From **measured** numbers:

| leg | cost | source |
|---|---:|---|
| automatic baseline, 50 × 768 tokens | 1.20 h | measured, first attempt |
| training, 500 steps @ 3.04 s/step | 0.42 h | measured, first attempt |
| candidate eval, 50 × 768 tokens | 1.75 h | control: 126 s/problem, cap-hit 100% |
| **total** | **3.37 h** | |

The first attempt never tested the budget because it died before the candidate eval.
At 2.0 h this run would be refused partway through, which would produce another
non-result. A **resource ceiling** is not an outcome threshold: raising it cannot
make a negative result look positive, and the per-leg estimates in the project are
corrected to match measurement (evaluation 0.1 → 1.8 h) so the ledger stops lying.

Expect **~3.5 h wall clock**.

## Stated expectation, given what the control now shows

The original pre-registration expected GSM8K to **rise**, written before the
dense-vs-pruned control existed. The control has since shown this checkpoint
degenerates on 8/8 prompts and hits the token cap every time, and that all 50
baseline responses contain no answer span. So I now expect:

> **GSM8K 0.00 on both sides — FLAT — and the capability arm to be uninformative
> rather than negative.** 500 steps of LoRA at r=16 is very unlikely to restore the
> ability to terminate a generation, which is the binding failure.

This is a worse-looking prediction than the original and is recorded anyway. If
GSM8K does rise, the original expectation was right and mine is wrong.

## Added measurement, so the run is not a guaranteed 0-vs-0

Because a 0-vs-0 capability result says nothing about whether training did anything,
the following are **pre-registered now** as a secondary readout. They are computed
post-hoc from the durable `predictions-gsm8k.jsonl` of both arms — they change no
evaluation code, and the eval protocol stays byte-identical between the arms:

* fraction of responses that **close** `</think>` (i.e. produce an answer span)
* fraction that hit the 768-token cap
* mean distinct-trigram ratio and mean zlib compression ratio
* count flagged degenerate (either measure below its threshold)

**Pre-registered reading:** these are descriptive, not a gate. A move in them is
evidence that 500 steps changed the model's generation behaviour even if GSM8K does
not move. It is **not** a substitute for the capability gate and will not be
reported as one, and no threshold on them is being set after the fact because none
is being set at all.

## Unchanged

Model, data, engine, LoRA shape and the ten-name hybrid target list, `max_length`
1024, batch 1 × grad-accum 4, lr 2e-4, 500 steps, seed 123, 50 GSM8K test problems
at `max_new_tokens` 768, `minimum_promotion_gain` 0.02, `require_protocol_match`
true, automatic baseline. All limitations in the original document still apply,
n=50 and single-seed included.
