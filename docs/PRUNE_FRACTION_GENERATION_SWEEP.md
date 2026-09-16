# Where does generation survive pruning on this 9B? Not where perplexity says.

Ran 2026-09-12 09:23 → 10:16. Driver
`evidence/pruned-real-training/prune_fraction_generation_sweep.py`, result
`evidence/pruned-real-training/prune-fraction-generation-sweep.json`.

## The result

Same 8 GSM8K prompts, 768-token cap, greedy, one nf4 load of the dense parent, masking
the channels a prune would delete using the **same corpus-wide ranking the built
checkpoints use** (digest `304fffa858ea`). No scaling, matching `static_prune`.

| f | keep/layer | ×dense ppl | GSM8K | terminated | degenerate | trigram | compression | verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1.00 | 12288 | 1.00 | 0.375 | 5/8 | 2/8 | 0.641 | 0.344 | **SURVIVES** |
| 0.75 | 9216 | 1.03 | 0.250 | 4/8 | 0/8 | 0.765 | 0.347 | **SURVIVES** |
| 0.5625 | 6912 | 1.09–1.14 | 0.250 | **1/8** | **7/8** | 0.384 | 0.200 | FAILS |
| 0.40 | 4915 | 1.19 | 0.000 | 0/8 | 8/8 | 0.227 | 0.131 | FAILS |
| 0.28 | 3441 | 1.48–1.69 | 0.000 | 0/8 | 8/8 | 0.108 | 0.084 | FAILS |

**Lowest surviving fraction: 0.75.** The generation cliff is between f=0.75 and
f=0.5625.

## The finding: perplexity is nearly flat across the cliff

Across the interval where termination collapses from 4/8 to 1/8, perplexity moves from
**1.03× to 1.14×** — eleven points, on a metric that then keeps climbing gently to
1.48× while generation is already dead at 0/8.

| | f=0.75 → f=0.5625 |
|---|---|
| perplexity | 1.03× → 1.14× (+11 points, looks nearly free) |
| terminated | 4/8 → 1/8 |
| degenerate | 0/8 → 7/8 |

So this is stronger than "perplexity does not predict termination", which is what
`PRUNED_9B_RERUN_RESULT.md` could already say. **At 1.14× perplexity the metric is
actively misleading**: it reports a near-free prune of a model that can no longer stop.
Anyone selecting a prune fraction on perplexity alone would have picked f=0.56 — 44% of
channels deleted for a claimed 14% perplexity cost — and shipped a checkpoint that
terminates on 1 prompt in 8.

## Both pre-registered anchors behaved, so the emulation is trustworthy

The survival rule (≥4/8 terminate **and** ≤2/8 degenerate) and both anchors were fixed
before any arm ran:

* **f=1.00 had to pass**, and reproduced the independent dense control almost exactly —
  GSM8K 0.375 vs 0.375, terminated 5/8 vs 5/8, trigram 0.641 vs 0.6406. With `mask=None`
  this *is* the same measurement, and it agrees.
* **f=0.28 had to fail**, and did: 0/8 terminated, 8/8 degenerate, trigram 0.108 against
  the built checkpoint's measured 0.094 (control, n=8) and 0.113–0.118 (re-run arms,
  n=50). The mask reproduces the real artifact's generation behaviour, which is the
  reason to believe the fractions in between.

Had f=0.28 *survived*, the emulation would not reproduce the built checkpoint and none
of these numbers would be usable. That was stated in advance.

## Limits, stated rather than glossed

* **n=8 per arm.** One problem is 0.125 of the termination count. The cliff's *location*
  is robust (4/8 → 1/8 → 0/8 is three arms moving together with the degeneration and
  trigram columns); its exact position between 0.75 and 0.5625 is not resolved.
* **Degeneration count has ±1 jitter at n=8** — f=1.00 scored 2/8 here where the control
  scored 1/8 on identical trigram means. The termination column matched exactly and is
  the one to lean on.
* **Masking, not built checkpoints.** nf4 quantises a narrowed matrix slightly
  differently; four prior mask-vs-checkpoint comparisons drifted +1.3% to +2.5% on
  perplexity. Immaterial for a qualitative termination question, and the f=0.28 anchor
  confirms it empirically here.
* **One model, one ranking, one task.** Nothing here says where the cliff sits on
  another architecture, or that a *trained* prune cannot move it — the re-run showed 500
  LoRA steps recovered arithmetic at f=0.28 without restoring termination, but no
  healing was attempted at f=0.75.
* **f=0.75 is not a free lunch either.** Its GSM8K is 0.250 against the dense 0.375 —
  one problem below, at n=8. It terminates and does not degenerate; it is not shown to
  be as capable.

## Consequence for the roadmap

The open item "revisit whether a milder prune produces a checkpoint that terminates"
(`PRUNED_9B_REAL_TRAINING_RESULT.md`) is **closed with a negative answer at f=0.56**.
The surviving window is narrow and expensive: f=0.75 keeps 9,216 of 12,288 channels, so
the FFN saving is 25% rather than the 72% that f=0.28 promised. Whether a 25% FFN prune
is worth building and healing is a different, smaller question than the one this line
started with.
