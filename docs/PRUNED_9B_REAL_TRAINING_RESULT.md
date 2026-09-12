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
loss falling steadily from ~4.8 to 1.07, 3.04 s/step. The adapter directory was left
empty, so 323 steps of real training were discarded.

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
  sequences, 3.04 s/step, loss 4.8 → 1.07 over 323 steps.
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

## Next, in order

1. **Run the corrected control** — dense vs pruned, same prompts, same settings, now
   with line-agnostic degeneration. If the dense parent also degenerates, the finding
   is about the harness, not pruning.
2. **Unify the two scorers.** The base worker's stricter rule is the correct one: an
   unclosed `<think>` means no answer was produced, which is incorrect — not "extract
   whatever number is lying around." Add a test that both workers agree on a shared
   case table so a future mode cannot quietly diverge.
3. **Re-run the training arm** with the progress fix in place. The recipe's cosine
   scheduler was also never applied (the config omits `lr_scheduler_type`, so the
   trainer default linear was used); fix that in the same pass and record it.
4. Only then revisit whether a milder prune (f=0.56 cost just 1.14× perplexity)
   produces a checkpoint that terminates.
