# Hot-core MoE vs static pruning at equal active compute — the conclusion

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
