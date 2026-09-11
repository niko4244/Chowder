# Hot-core router-healing pilot — result

Spec: `docs/HOT_CORE_HEALING_PILOT_PREREG.md` (written before the run).
Evidence: `evidence/hot-core-upcycling/pilot-result.json`, `healing_pilot.py`.
Ran 2026-09-11, 150 steps in 25.7 min on one RTX 5060 Ti.

## Pre-registered verdict: **ATTRIBUTION FAIL**

The routing thesis is **not supported** by this pilot. Perplexity improved, but on
the pre-registered primary split the improvement came entirely from the
shared-expert gate — a scalar recalibration — and the router contributed nothing.

Step 0 reproduced the recorded init exactly (eval A 9.8444, drift 0.0000), so the
eval path is identical to the one that produced the converted model's baseline.

## Trajectory

| step | eval A | vs init | eval B | vs init | ‖W‖ router | ‖W‖ gate |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 9.8444 | — | 19.4255 | — | 0.00 | 0.00 |
| 25 | 9.4103 | −4.4% | 15.5215 | −20.1% | 4.84 | 1.63 |
| 50 | **8.9060** | **−9.5%** | 15.7526 | −18.9% | 7.62 | 2.38 |
| 75 | 10.4343 | **+6.0%** | 17.8155 | −8.3% | 9.02 | 2.81 |
| 100 | 9.3555 | −5.0% | 15.2613 | −21.4% | 9.46 | 2.95 |
| 125 | 9.1989 | −6.6% | **14.7491** | **−24.1%** | 9.57 | 2.99 |
| 150 | 9.2324 | −6.2% | 14.9314 | −23.1% | 9.58 | 2.99 |

lr 1e-3 was too high: step 75 is 6% **worse** than the untrained init. With 33
blocks of 768 tokens (24,849 training tokens) and 150 steps, this is ~4.5 epochs
over a tiny corpus, and it shows.

## Attribution — the part that matters

Both trainable groups start at **exactly** zero, so resetting either one to its
init is a clean ablation rather than an approximation.

| variant | eval A | gain | eval B | gain |
|---|---:|---:|---:|---:|
| init (step 0) | 9.8444 | — | 19.4255 | — |
| both trained | 9.2324 | +0.61 | 14.9314 | +4.49 |
| **router only** (gate reset to 0) | **9.8742** | **−0.03** | **15.5713** | **+3.85** |
| **gate only** (router reset to 0) | **9.0863** | **+0.76** | **17.4243** | **+2.00** |

**On eval A the router is worthless — slightly harmful.** All of the gain is the
gate. That triggers the pre-registered ATTRIBUTION FAIL, which was defined in
advance as "does NOT support the thesis regardless of the perplexity number."

**On eval B the opposite holds**: the router delivers +3.85 and the gate +2.00.
Eval B is the cleaner split — never touched by the channel ranking *or* training,
whereas eval A sits in the same contiguous region of the file as the ranking
split. So the cleaner measurement favours the router and the pre-registered one
does not.

I will not promote eval B to primary after seeing the numbers. The pre-registered
verdict stands. But the prereg itself flagged this exact weakness — *"eval A's
prompts are from the same contiguous region as the ranking split, and a gain that
appears only on eval A would be weak evidence"* — and then made eval A primary
anyway. **That was a pre-registration design error**, and the honest summary is
that this pilot is **inconclusive** on routing, with its pre-registered reading
negative.

## The milestone was a tie, not a crossing

Best eval A was 8.9060 against a static-hot-prune reference of 8.9096 at equal
active compute — a margin of **+0.04%**. The pre-registered rule technically fires,
but a 0.04% margin is a tie. **Routing did not beat static pruning here.** And
since the eval-A gain is attributable to the gate, what tied the static prune was
a scalar recalibration, not token-conditional routing.

## What the gate actually learned, and why it is useful anyway

`shared_expert_gate` moved from 0 to ‖W‖ 2.99, pushing `sigmoid(gate(x))` away
from 0.5 and therefore **upweighting the hot core** beyond the unit scale the ×2
`down_proj` pre-scale calibrates it to.

That is precisely what the design sweep predicted: more hot core was better at
*every* active budget (f=0.28: 100% core 1.69× vs 75% core 1.81× vs 50% core
2.09×). The gate is doing by gradient descent what the sweep said to do by
construction. The actionable reading is that **the core is too small / the
cold-channel differentiation too expensive at f=0.28**, and the next converter
should try a larger core.

## Honest status of the program thesis

The static→oracle gap (up to 4.30×) remains the justification for building an MoE
here, and it remains **unclaimed**. This pilot does not close it:

* the pre-registered measurement says the router bought nothing;
* the cleaner held-out measurement says it bought most of a 24% improvement;
* the two disagree, and one pilot at lr 1e-3 over 24.8k tokens cannot settle it.

## Addendum — the eval-B static reference, measured

Follow-up #2 is done (`evidence/hot-core-upcycling/static-prune-evalb.json`). The
same saved ranking artifact that built the checkpoint (digest `df207741fdcc`) was
used, masking the dense parent in activation space on eval B's 64 prompts at the
same `max_length` 384 and nf4 load.

| arm on eval B | active | ppl | ×dense |
|---|---:|---:|---:|
| dense | 12,288 | **4.3044** | 1.000× |
| **static hot prune — THE REFERENCE** | 3,440 | **15.6918** | 3.646× |
| converted init channel set (as a mask) | 3,440 | 18.9474 | 4.402× |
| hot core alone | 2,176 | 31.6221 | 7.347× |
| arbitrary subset (control) | 3,440 | 2879.3025 | 668.9× |

### The ranking does not generalise, and that is the finding

Static hot pruning at f=0.28 costs **1.677×** on eval A but **3.646×** on eval B —
**2.17× worse** out of distribution. The ranking was measured on the even-index
half of the first 64 prompts, i.e. adjacent to eval A; eval B comes from a
different region of the corpus. One static channel set is simply a worse fit
there.

That reframes the whole pilot. Routing's value is **conditional on how badly the
static set fits the data**, which is exactly the static→oracle gap argument: the
gap is largest where the single best static choice is weakest.

### Against each split's own reference, at equal active compute

Negative = beats the static prune.

| variant | eval A | eval B |
|---|---:|---:|
| converted init | +10.49% | +23.79% |
| **trained, step 150** | **+3.62%** | **−4.85%** |
| router only | +10.83% | **−0.77%** |
| gate only | +1.98% | +11.04% |

**On eval B the trained model beats the best static choice at equal active compute
by 4.85%** — the first time anything in this program has done so. Step 150 is the
predetermined end of the run, so that is the unbiased estimate; the step-125 "best"
(14.7491, −6.39%) was selected by looking at eval B and should not be quoted as
the headline.

And **neither component alone suffices**: gate-only *loses* to static by 11.04%,
router-only beats it by only 0.77%. Together they beat it by 4.85%. So the router
is necessary but not sufficient on its own — which is a different claim from
either "routing works" or the eval-A verdict's "routing bought nothing".

### Revised status

The pre-registered verdict on eval A stands and is not retracted: there, routing
bought nothing and a scalar gate recalibration tied the static prune. But with the
eval-B reference now in hand, the coherent reading across both splits is:

* where the static ranking fits the data, static pruning is strong (1.677×) and
  routing adds nothing;
* where it fits poorly (3.646×), router + gate together beat it by 4.85%.

This is support for the thesis, but narrow and conditional: one pilot, one split,
lr too high, 24.8k training tokens, and perplexity rather than GSM8K. It is not
the 4.30× static→oracle gap being closed — it is ~5% of a much larger available
gap, under distribution shift.

One loose end: the converted init channel set masked onto the dense model scores
18.9474 against the converted checkpoint's own measured 19.4255 (drift 0.478,
2.5%). Most likely the two checkpoints' different tensor layouts quantise
differently under nf4, but it could also mean the zero-router tie-break does not
select experts 0 and 1. **Unverified either way** — it does not affect the
reference above, which is measured entirely on the dense model.

## Designated follow-ups, in order

1. ~~Measure the static-prune reference on eval B.~~ **Done — see the addendum.**
   It produced the program's first equal-active-compute win (−4.85%) and the
   ranking-generalisation finding.
2. **Rank channels on a corpus-wide split, not a contiguous one.** This is now the
   highest-value fix: the ranking costs 1.677× in distribution and 3.646× out of
   it, so a ranking measured across the whole corpus should cut the out-of-
   distribution penalty directly — and it would also shrink the very gap routing
   is currently being credited with closing. Both effects matter and they pull in
   opposite directions, which is exactly why it should be measured.
3. **Re-run with eval B pre-registered as primary** plus a third fresh split, at
   lr 1e-4–3e-4 with fewer epochs. The step-75 instability and the 24.8k-token
   corpus are both fixable.
4. **Router-only arm** (`ROUTER_ONLY_SUFFIXES`), so the gate cannot launder the
   result either way. Router-only already sits at −0.77% on eval B; a clean arm
   would say whether that is real.
5. **A larger hot core**, since the gate spent its capacity asking for one — and
   the core alone at 2,176 channels costs 7.347× on eval B, so the routed half is
   doing substantial work there.
