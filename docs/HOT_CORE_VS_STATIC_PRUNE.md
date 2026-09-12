# Hot-core MoE vs static pruning at equal active compute — the conclusion

> **WITHDRAWN AS A DEPLOYABLE RECOMMENDATION (2026-09-11).** A paired control on the
> real models — same 8 GSM8K prompts, same settings, same scorer, same nf4 load, only
> the weights differing — shows the pruned checkpoint **degenerates on 8/8 prompts**
> and hits the 768-token cap every time, while the dense parent degenerates on 1/8 and
> terminates 62.5% of the time:
>
> | | dense | pruned (f=0.28) |
> |---|---:|---:|
> | GSM8K (`final_number_match`) | **0.375** | 0.125 |
> | distinct-trigram ratio | 0.641 | **0.094** |
> | compression ratio | 0.344 | **0.084** |
> | degenerate | 1/8 | **8/8** |
> | hit the token cap | 37.5% | **100%** |
>
> **Pruning is the cause, not the harness.** The pruned 0.125 is itself an artifact —
> `final_number_match` takes the last number, and a degenerate loop sometimes ends on
> the right one. Everything below remains correct **as perplexity**; perplexity simply
> does not predict whether a checkpoint can terminate. See
> `PRUNED_9B_REAL_TRAINING_RESULT.md`.
>
> Two limits on how far this generalises: the MoE checkpoint has **not** been measured
> for degeneration, so "static pruning beats the hot-core MoE" stays a perplexity
> claim and neither artifact has been shown usable; and f=0.28 is the only fraction
> tested this way — f=0.56 cost just 1.14× perplexity and is untested for generation.

Evidence: `evidence/hot-core-upcycling/reconvert-corpuswide.json`,
`corpus-wide-ranking.json`, `static-prune-evalb.json`, `pilot-result.json`.
Design held identical across both checkpoints (E=16, `top_k`=2, core 2176, cold
632/expert, 3,440 of 12,288 active) so the **ranking is the only variable**.

## The result

**At f=0.28 on this 9B, static hot pruning beats the hot-core MoE at identical
active compute, and a better ranking made the MoE worse rather than better.**

| | eval A | eval B |
|---|---:|---:|
| dense parent | 5.2707 | 4.3044 |
| static prune, contiguous ranking | **8.9111** | 15.6918 |
| static prune, corpus-wide ranking | 9.7968 | **12.9605** |
| hot-core MoE init, contiguous ranking | 9.8444 | 19.4255 |
| hot-core MoE init, **corpus-wide ranking** | **13.4224** | **15.2936** |
| hot-core MoE, contiguous, after 150 healing steps | 9.2324 | 14.9314 |

Re-ranking corpus-wide moved the MoE init **+36.3% worse on eval A** and −21.3%
better on eval B. It did not rescue the design.

A deployable artifact has to commit to one ranking, and the general choice is the
corpus-wide one. Geometric-mean cost across both splits:

| | ×dense |
|---|---:|
| static prune, corpus-wide | **2.366×** |
| hot-core MoE init, corpus-wide | 3.008× |

**Static pruning is 27% better at identical active compute**, with no training, no
router, and a smaller checkpoint.

## Why — and it was in the data the whole time

The MoE spends **2,176 of its 3,440 active channels on a fixed core** and only
1,264 on routed choice. Static pruning spends all 3,440 on the best fixed set. So
at init the MoE trades away ranks 2,176–3,439 — the next-best channels — in
exchange for a scattered sample of cold ones. That is a *strictly worse static
choice*, and routing has to earn the difference back:

| gap the MoE must close just to MATCH its own static reference | eval A | eval B |
|---|---:|---:|
| contiguous ranking | +10.5% | +23.8% |
| corpus-wide ranking | +37.0% | +18.0% |

What 150 healing steps actually delivered: **−6.2% on eval A, −23.1% on eval B.**
Enough to approach parity on one split, nowhere near it on the other.

**The core-share sweep already said this and I misread it.** It found more hot core
better at *every* active budget — f=0.28: 100% core 1.69×, 75% 1.81×, 50% 2.09%,
25% 3.06×, 0% 214×. I recorded "at init the optimum is always core_share=1.0" and
then argued routing would earn its keep through training. But **100% core *is*
static pruning**. The monotone result was the answer; I treated it as a starting
point instead of a conclusion, and it took two training runs and a second
conversion to arrive where the sweep already pointed.

## Why a better ranking hurt the MoE specifically

The design bets its active budget on the core being right. A parochial ranking
matched to eval A put most of what eval A needs inside the top 2,176, so the
scattered cold channels cost little. A corpus-wide ranking makes the core a
compromise, so more of what any *particular* split needs falls below rank 2,176 —
into the cold region, where only 2 of 16 slices are available per token. Hence the
init-vs-static gap widened on eval A (+10.5% → +37.0%) even though the static
reference itself improved.

Put plainly: **the hot-core design is helped by an overfit ranking and punished by
a general one**, which is the opposite of what a deployable artifact wants.

## What is NOT concluded

* **Not** that MoE routing is worthless in general. This is one architecture, one
  model, one active budget (f=0.28), perplexity rather than GSM8K, and ~25k
  training tokens of healing.
* **Not** that the static→oracle gap is closed or refuted. The oracle at f=0.28 is
  1.03× against static's 1.859× (corpus-wide, eval A), so a very large gap remains
  available in principle. Nothing tested here captures it — including the router.
* **Not** that the converter is wrong. It is validated twice: predicted 1.81–2.09×
  vs measured 1.853× on the first checkpoint, and predicted 13.2087/15.0482 vs
  measured 13.4224/15.2936 here — **+1.6% drift on both splits**, same direction as
  the +2.5% seen earlier, consistent with the two tensor layouts quantising
  differently under nf4 rather than with a conversion error.

## The built artifact

`F:\llm-models\Qwen3.8-9B-Pruned-CW-3440`, built by `chowder.static_prune`
(`evidence/hot-core-upcycling/static-prune-checkpoint.json`). Corpus-wide ranking
`304fffa858ea`, top 3,440 of 12,288 channels kept, emitted in rank order.

| | dense parent | hot-core MoE (CW) | **static prune (CW)** |
|---|---:|---:|---:|
| architecture | `qwen3_5` | `qwen3_5_moe` | **`qwen3_5`, unchanged** |
| total params | 9.410B | 9.410B | **5.931B** |
| active params | 9.410B | 5.931B | **5.931B** |
| checkpoint on disk | 18.82 GB | 18.82 GB | **11.86 GB** |
| peak VRAM, nf4 load | 5.72 GiB | 15.07 GiB | **5.45 GiB** |
| eval A ppl | 5.2707 | 13.4224 | **9.9235** |
| eval B ppl | 4.3044 | 15.2936 | **13.0258** |
| geo-mean × dense | 1.000× | 3.008× | **2.387×** |

It beats the MoE on every axis simultaneously: smaller on disk, **less VRAM than
even the dense parent** (the MoE's raw-`nn.Parameter` expert bank stays BF16 under
nf4; a pruned `nn.Linear` quantises like any other), lower perplexity, and no
change of architecture.

**Framing, so the number is not misread as shrinking:** the earlier "27% better"
was `MoE / static − 1` using the mask-emulated static cost (2.366×). With the real
pruned checkpoint it is `3.008 / 2.387 − 1` = **26.0%** in that same framing — a
one-point move from the static side's real-checkpoint drift. The build log's "21%"
is the other framing, `1 − static / MoE`. Same result.

**Prediction held.** The mask emulation predicted 9.7968 / 12.9605; the real
checkpoint scored 9.9235 / 13.0258 — drift **+1.3% / +0.5%**, same positive
direction as the +1.6% and +2.5% seen at both previous mask-vs-checkpoint
comparisons. Four independent comparisons now drift the same way, which makes
nf4 layout differences the consistent explanation rather than a conversion error.

Byte-level check on the real artifact: 760 tensors in, 760 out; all **664**
non-MLP tensors byte-identical, including all **108** vision-tower `.mlp.`
tensors; the 96 decoder MLP tensors narrowed to (3440, 4096) / (4096, 3440).

**One deployment wart, predicted and confirmed.** bitsandbytes warns *"inner
dimension (3440) is not aligned for fast kernel with blocksize=64, falling back to
slower implementation."* 3,440 is not a multiple of 64, which is exactly why the
build predicted a drift band rather than a point. It is correct, just slower. For
a deployment build, **keep 3,456 (= 54 × 64)** — 16 more channels, +0.47%, fast
kernel. Because output is in rank order, that is also a strict superset of this
checkpoint's channels.

## Recommendation

**Stop spending on the hot-core/routing path at f=0.28 and take the static prune.**
It is 27% better, free, and simpler. Concretely, for the ≤10B-total goal: a
corpus-wide-ranked static prune to 3,440 channels gives active FFN 1.353B at
2.366× dense perplexity, with no router, no healing, and no MoE-specific
deployment risk.

If the routing thesis is to be revisited, the measurements say to change the thing
that actually binds — the **core:routed split**, not the ranking. Every data point
so far says a bigger core wins, and the limit of that argument is no router at all.
A design where routing gets a *majority* of the active budget would be a genuinely
different bet, and it would start from the 25%-core arm measured at 3.06× — i.e.
far behind. That is the honest prior to carry in.
