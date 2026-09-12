# Session handoff — 2026-09-12

Branch `feat/hot-core-upcycling`, [PR #158](https://github.com/niko4244/Chowder/pull/158),
**32 commits against `origin/main`** (`git log main..HEAD` reports 49 from a local
`main` that is 19 commits stale — use `origin/main..HEAD`). Suite **1577 passed, 77
skipped**. Working tree clean, everything pushed.

## §1 READ FIRST — work was in flight when this was written

A **prune-fraction generation sweep** was running on the GPU. If it finished, its
result is at `F:\llm-models\_a4b\prune-fraction-generation-sweep.json` (written
incrementally after each arm, so a partial file is still valid). Driver:
`scratchpad/prune_fraction_generation_sweep.py`, log `scratchpad/frac_sweep.log`.

**Results landed before this handoff:**

| f | keep/layer | GSM8K | terminated | degenerate | trigram | verdict |
|---:|---:|---:|---:|---:|---:|---|
| 1.0 (dense) | 12288 | 0.375 | 5/8 | 2/8 | 0.641 | **SURVIVES** |
| 0.75 | 9216 | 0.250 | 4/8 | 0/8 | 0.765 | **SURVIVES** |
| 0.5625 | 6912 | — | — | — | — | *running* |
| 0.40 | 4915 | — | — | — | — | queued |
| 0.28 | 3441 | — | — | — | — | queued |

The f=1.0 anchor reproduced the independent dense control almost exactly (0.375
GSM8K, 5/8 terminated, 0.641 trigram vs the control's 0.375 / 5/8 / 0.6406), which
validates the masking harness. **Caveat: degenerate count has ±1 jitter at n=8** — the
control scored 1/8 where this scored 2/8 on identical trigram means. Read the
termination column, which matched exactly; treat the degeneration column as coarse.

**Pre-registered survival rule, fixed before any arm ran:** survives iff ≥4/8
terminate AND ≤2/8 degenerate. Anchored at both ends: f=1.0 must pass, f=0.28 must
fail (it degenerated 8/8 with 0/8 terminating in the re-run). If f=0.28 *passes*, the
masking emulation does not reproduce the built checkpoint's behaviour and the sweep
should not be believed.

## §2 The scientific state, plainly

**Chowder trains this 9B hybrid correctly.** 500 steps, 200/200 module coverage,
6.14 GiB peak, cosine schedule verified from the realised LR curve, on a real corpus.

**No prune of this model has been shown to produce a usable checkpoint.** At f=0.28,
0 of 100 responses across two arms ever terminated — every one ran to the 768-token
cap. Training recovered **+0.10 of real arithmetic** (verified under three scoring
rules, so it is not a scorer artifact) while leaving degeneration statistically
unchanged. It learned to **solve** more problems, not to **stop**.

`HOT_CORE_VS_STATIC_PRUNE.md` is marked **WITHDRAWN as a deployable recommendation**.
Its perplexity results stand; perplexity simply does not predict termination. The
hot-core MoE checkpoint was never measured for degeneration at all, so "static
pruning beats hot-core by 26%" remains a perplexity claim with **neither artifact
shown usable**.

## §3 Ten defects fixed this branch, each with tests

| defect | fix | commit |
|---|---|---|
| worker subprocess PYTHONPATH | `worker_env.py` | earlier |
| Unsloth loaded the **VLM wrapper**, so every adapter key mismatched (logit delta 0.0) | `text_only=True` | `bf190e6` |
| scoring an adapter that cannot change the model | `adapter_guard.py` | `3621965` |
| unverified target coverage | `target_coverage.py` | `81cfe05` |
| Unsloth's regex conversion dropped the Mamba-style layers | suffix-match regex | `0bd62ed` |
| a telemetry rename killed a 500-step run at step 323 | `progress_write.py` | `e2e6df1` |
| two text scorers had silently diverged | `evaluators/scoring.py` | `a9cd0ad` |
| Unsloth silently dropped `lr_scheduler_type` (prereg said cosine, ran linear) | spec + worker | `5de632d` |
| `sorted()` over a regex **string** recorded 99 characters as provenance | both workers | `69e27c0` |
| **evaluators recorded no VRAM at all** | `evaluators/vram.py` | `52cba56` |

The last one caused a spurious verdict: a pre-registered "peak VRAM under budget"
condition was undecidable for the evaluation leg, so a whole-machine `nvidia-smi`
proxy stood in and **a busy desktop failed an experiment**. See §5.

## §4 Open items, in priority order

1. **Finish the fraction sweep** (§1) and record where generation survives. This is
   the question that decides whether the pruning line has a deployable endpoint.
2. **Unqualified engineering PASS** — re-run the evaluation leg on a verified-idle
   card with the new VRAM instrumentation (~1.4 h). The verdict is "PASS, qualified"
   only because the run could not measure its own footprint.
3. **Apply the PR #158 title/description.** Drafted and verified against the
   30-commit range at `docs/PR158-DESCRIPTION.md`. **Nothing has been pushed to
   GitHub** — retitling is a visible change on a public repo and was left for the
   user to approve.
4. **`qwen3_5_moe` and `qwen3_5_text` preset entries.** Deliberately not added: the
   MoE FFN replaces one gate/up/down triple per layer with one per expert plus a
   router, so the same ten names resolve to a very different set and it needs its own
   real-model leaf count first. A test pins `qwen3_5_moe` as unsupported so adding it
   forces a deliberate edit.
5. **Two stale coverage artifacts.** `level2-unsloth-report.json` and
   `level2-unsloth-fixed-report.json` record `resolved_target_modules` as
   character-exploded garbage (the `69e27c0` bug), so the 128 and 200 module counts
   live only in prose. Regenerating needs two short GPU runs.
6. **The ~9 GiB is unexplained.** Five mechanisms eliminated (adapter weights at
   +0.15 GiB, token count, accumulation across problems, cache class, instrument
   disagreement); the footprint is ~6.6 GiB measured four ways; the remainder was
   external and unidentified. The instrumentation fix means it can no longer silently
   fail a run, so this is safe to leave as documented-unknown.

## §5 Two verdicts were revised. Both are marked, neither was quietly changed

**Engineering FAIL → PASS, qualified** (`78e2615`). The FAIL rested on 0.56 GiB
headroom plus a 2× blowup. The blowup is adapter compute (1.54–1.83× measured with
~8.9 GiB free), and the headroom figure measured the whole machine because the
evaluators recorded no per-process VRAM. Marked **qualified** because the withdrawal
rests on post-hoc measurement, not the run's own evidence. The symmetry test is
recorded in the doc: had the probes shown the adapter genuinely consuming 15 GiB, the
FAIL would have been confirmed.

**My pre-registered capability prediction was wrong.** I predicted 0.00/FLAT against
the original pre-registration's expectation of a rise. The original was right
(+0.12). Recorded as wrong rather than reframed.

## §6 Standing constraints, carried forward

* Never stage the 5 unrelated frontier v4-scoring files.
* Never kill the user's Ollama / llama-server to free GPU.
* Never use bare `git stash` — the stack is shared across ~11 worktrees.
* Never lower a pre-registered threshold after seeing results.
* Never change evaluation code while an evaluation is pending.
* The GPU is a **single serialized resource** — one job at a time. Parallel work must
  be off-GPU.

## §7 Mistakes of mine worth not repeating

1. **Numbers carried through prose without a source.** "loss fell from ~4.8" was
   imported from a *different experiment* (level-2 invented-facts) into a GSM8K
   write-up. Retracted in `69e27c0`. I then nearly "corrected" the f=0.56 **1.14×**
   figure the same way — it turned out to be properly measured. Check the source
   before trusting *or* retracting.
2. **A near-miss wrong-model error.** A probe pointed at `Qwen3.5-9B` when the
   ranking, control and both checkpoints are bound to
   `Qwen3.8-9B-abliterated-25-bf16`. Both are 32 layers × 12288 intermediate with the
   same architecture, so it would have produced plausible numbers about the wrong
   weights. The script now asserts `ranking.source_dir == DENSE` and refuses.
3. **Four silent `str.replace` anchor misses.** Switched to editing by verified line
   number with a content assertion.
4. **An experiment design whose positive branch was unreachable.** The adapter A/B
   required ≥4 GiB headroom to credit "adapter compute", which is unsatisfiable when
   the adapter is what consumes the headroom; and it ran both arms in one process.
   One arm per process settled it immediately.
5. **An order-dependent test** that passed in suite order and failed standalone.
6. **Looking in the wrong place and reporting absence as fact** — `coverage` and
   `progress_write_failures` are top-level in `worker-result.json`, not under
   `telemetry`/`provenance`.
