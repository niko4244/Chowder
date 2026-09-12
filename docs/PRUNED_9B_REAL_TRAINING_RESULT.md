# Result — first real training run on the pruned 9B

Pre-registration: `docs/PRUNED_9B_REAL_TRAINING_PREREG.md`, committed `9f66960`
**before** the run. Correction issued mid-run: `..._CORRECTION.md` (`785ce94`).
Ran 2026-09-11 19:23 → 20:52, 89.5 min wall.

## Verdict against the pre-registered conditions

| arm | verdict |
|---|---|
| **Engineering** (the primary claim) | **FAIL** — the run did not complete |
| **Capability** (secondary) | **VOID** — protocol not identical, and no candidate eval ran |

The engineering FAIL is the honest label: the pre-registered PASS required the run
to complete with coverage, a live adapter, a falling loss, VRAM in budget, and GSM8K
measured on both sides. It did not complete. It failed **by finding a real defect in
Chowder**, which is a useful outcome but not a PASS, and relabelling it as one would
be exactly the post-hoc move the pre-registration exists to prevent.

## What killed it: telemetry, not training

`unsloth_worker.py` published progress by writing a temp file and renaming it over
the live one, unguarded. At **step 323 of 500**, 16.4 min in:

```
PermissionError: [WinError 5] Access is denied:
  '...\adapter\progress.tmp' -> '...\adapter\progress.json'
```

An exception inside `TrainerCallback.on_log` propagates out of `Trainer.train()`, so
a *telemetry* write destroyed the run. Both progress files survived the crash and
settle where the fault was:

| file | step | loss | elapsed |
|---|---:|---:|---:|
| `progress.json` (last published) | 322 | **1.1002** | 978.6 s |
| `progress.tmp` (rename refused) | 323 | **1.0737** | 981.9 s |

The payload was written correctly; only the rename failed. **Training was working**:
loss had reached 1.07 by step 323 at 3.04 s/step. The adapter directory was left
empty, so 323 steps of real training were discarded.

> **Correction (2026-09-12).** This paragraph and the bullet below originally said
> loss fell "from ~4.8". That figure was never measured on this run. The driver
> crashed on printing `CANDIDATE ERROR` through a cp1252 console *before* reaching
> its loss-trajectory print, so `realtrain4.log` contains no trajectory at all, and
> `progress.json` was overwritten every step so only 322 and 323 survived. The
> 4.8354 came from the **level-2 invented-facts runs** — a different task, 20 items,
> 50 steps — and I carried it across. What is actually supported for this run is
> steps 322 (1.1002) and 323 (1.0737) and nothing earlier.
>
> The re-run under the same config puts the start at **2.8503** (step 1) and reaches
> **1.0750** at step 323, against this run's 1.0737 — agreement to ~0.1% at the same
> step, which is independent evidence the recovered progress files were read
> correctly. Treat 2.85 as the re-run's measurement, not a retrofit of this one.

`transformers_worker.py` carried the identical pattern, with a comment asserting the
rename is "atomic on POSIX/NTFS" — true of the semantics when it succeeds, silent on
whether it can fail, which is the assumption that broke.

**Fixed** in `chowder/progress_write.py`: retry briefly (a transient antivirus or
indexer handle clears in milliseconds), then give up and keep training. Failures are
counted and surfaced as `progress_write_failures` in telemetry rather than swallowed,
because a path that is *always* unwritable is worth seeing — just not worth a run.
Applied to both workers; `unsloth_worker` inlines it because it must not import from
the chowder package. 7 tests, including the exact `WinError 5`, a transient failure
that retries into success, and a source guard that is mutation-verified to fail if
either worker goes back to an unguarded rename.

## What the run did establish

* Chowder's lifecycle behaved correctly throughout. The automatic baseline completed
  (50/50), the registry recorded `realtrain-unsloth` as **FAILED** rather than
  leaving it RUNNING, and the durable artifacts were complete enough to reconstruct
  the entire failure after my driver also crashed (on printing the error, through a
  cp1252 console — a second, separate encoding bug of mine).
* Training on a real corpus works at this scale: 7,473 GSM8K rows, 1024-token
  sequences, 3.04 s/step, loss down to 1.07 by step 323 (see the correction above:
  the starting value was not recorded for this run).
* **The budget did not refuse.** Baseline ~1.2 h + training 0.27 h = ~1.47 h of the
  2.0 h allowance. The risk was correctly identified in review before the run; it
  simply did not materialise.

## The degeneration finding, now measured properly

The interim claim was *"the pruned model degenerates — 0.00 on the first 28."* The
correction separated those. The final numbers on all 50 baseline problems:

| measure | median | flagged degenerate |
|---|---:|---:|
| distinct-trigram ratio | **0.079** (92% of trigrams are repeats) | 46/50 |
| zlib compression ratio | **0.080** (compresses to 8%) | 45/50 |
| either measure | | **48/50** |
| unclosed `<think>` | | **50/50** |

Even the single most varied response cycles `20*20=200 / 200/20=20` forever, its
surface variety coming only from line numbering. **Degeneration is effectively
universal and scorer-independent** — two measures agree, and neither involves the
scorer. That is the part of the interim claim that survives.

The 0.00 does not survive: with `<think>` unclosed in all 50, the base scorer
discarded every response, so the baseline was 0.00 *by construction*.

### A metric of mine also undercounted

The first degeneration measure was a duplicate-**line** ratio. It flagged 41/50 and
scored the worst offender at 0.00, because that response is one unbroken line
repeating to the token cap. Line-agnostic measures flag 48/50. The control script
prepared for the dense-vs-pruned comparison used the same blind metric and has been
corrected before running — had it not been, it could have undercounted degeneration
in the dense model too and produced a wrong causal verdict.

## Practical consequence for the pruned checkpoint

The capability arm being void matters less than it would have. **Even a perfectly
symmetric scorer would be scoring a model that cannot terminate.** At f=0.28 the
pruned checkpoint is not usable for generation, and that conclusion now rests on
scorer-independent degeneration measures rather than on a score.

What remains genuinely unresolved is *cause*: pruning, or something shared by both
checkpoints (prompt format, chat template, generation config). The corrected control
answers that and has not been run.

## The control ran: pruning is the cause

Same 8 prompts, same settings, same scorer, same nf4 load; only the weights differ.

| | dense | pruned |
|---|---:|---:|
| GSM8K (`final_number_match`) | **0.375** (3/8) | 0.125 (1/8) |
| mean distinct-trigram ratio | 0.641 | **0.094** |
| mean compression ratio | 0.344 | **0.084** |
| degenerate | 1/8 | **8/8** |
| mean generated tokens | 492.6 | **768.0** |
| hit the 768-token cap | 37.5% | **100%** |
| wall seconds | 465 | 1009 |

The dense parent reasons, answers 3 of 8, and terminates on most prompts. The pruned
checkpoint degenerates on every prompt and never terminates. **The harness is
exonerated and pruning is the cause.**

Note the pruned 0.125 is not capability: it is one degenerate loop that happened to
end on the correct number, which is the `final_number_match` weakness recorded above.

Two limits on the conclusion, stated rather than glossed: **n=8**, so these rates are
coarse; and **the MoE checkpoint was never measured for degeneration**, so "static
pruning beats the hot-core MoE" remains a perplexity claim — neither artifact has been
shown usable for generation.

## Next, in order

1. ~~Run the corrected control.~~ **Done — see above.** Pruning is the cause.
2. ~~Unify the two scorers.~~ **Done** — one `evaluators/scoring.py`, the strict
   rule in both workers, guarded by identity + an 18-case agreement table + a
   mutation-verified source check. Re-scored, the 50 baseline predictions contain
   **no answer span at all**, and the old lenient rule would have scored 1 of them
   correct. See `..._CORRECTION.md`.
3. ~~Re-run the training arm.~~ **Done — `PRUNED_9B_RERUN_RESULT.md`.** It completed
   500 steps with `progress_write_failures: 0`, and the cosine schedule was verified
   against the realised LR curve. The cause of the missing schedule was not "the
   config omits `lr_scheduler_type`" as written here: the key was validated and
   honoured by `transformers_worker`, but `unsloth_worker` never read it, so setting
   it would have changed nothing. Engineering **FAIL** on oversubscription during the
   candidate eval; capability **RECOVERS** at +0.12, though 0/100 responses across
   both arms ever terminated.
4. ~~Revisit whether a milder prune (f=0.56 cost just 1.14× perplexity) produces a
   checkpoint that terminates.~~ **Done — no. See
   `PRUNE_FRACTION_GENERATION_SWEEP.md`.** At f=0.56 generation is already broken:
   1/8 terminating, 7/8 degenerate. The cliff sits between f=0.75 and f=0.56, and
   perplexity is nearly flat across it (1.03× → 1.14×), so at that operating point
   perplexity is actively misleading rather than merely uninformative.
