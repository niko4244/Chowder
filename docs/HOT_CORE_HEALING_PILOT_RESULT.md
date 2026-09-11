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
by 4.85%.** **[SUPERSEDED — see Addendum 2: against a corpus-wide-ranked static
baseline it loses by 13.20%.]** Step 150 is the
predetermined end of the run, so that is the unbiased estimate; the step-125 "best"
(14.7491, −6.39%) was selected by looking at eval B and should not be quoted as
the headline.

And **neither component alone suffices**: gate-only *loses* to static by 11.04%,
router-only beats it by only 0.77%. Together they beat it by 4.85%. So the router
is necessary but not sufficient on its own — which is a different claim from
either "routing works" or the eval-A verdict's "routing bought nothing".

### Status as of this addendum — itself later superseded

At this point the reading was: where the static ranking fits the data, static
pruning is strong and routing adds nothing; where it fits poorly, router + gate
together beat it by 4.85%. That looked like narrow, conditional support for the
thesis.

**Addendum 2 removes it.** The "where it fits poorly" clause was the weakness of a
parochial ranking, not a property of static pruning, and fixing the ranking beats
the trained model outright. The paragraph is kept here so the sequence of
reasoning stays legible, not because it still holds.

One loose end: the converted init channel set masked onto the dense model scores
18.9474 against the converted checkpoint's own measured 19.4255 (drift 0.478,
2.5%). Most likely the two checkpoints' different tensor layouts quantise
differently under nf4, but it could also mean the zero-router tie-break does not
select experts 0 and 1. **Unverified either way** — it does not affect the
reference above, which is measured entirely on the dense model.

## Addendum 2 — the 4.85% win does not survive a better baseline

Follow-up #2 above is now also done
(`evidence/hot-core-upcycling/corpus-wide-ranking.json`). It was run specifically
because the eval-B reference looked suspiciously weak, and the suspicion was
correct.

The ranking split was rebuilt at **the same size** (32 prompts) but spread evenly
across the whole corpus instead of taken from `rows[0:64:2]`, excluding both eval
splits. Only the *location* of the ranking data changed.

| static hot prune, 3,440 active | eval A | eval B | A→B spread |
|---|---:|---:|---:|
| contiguous ranking | 8.9111 (1.691×) | 15.6918 (3.646×) | 2.16× |
| **corpus-wide ranking** | 9.7968 (1.859×) | **12.9605 (3.011×)** | **1.62×** |
| change | +9.9% worse | **−17.4% better** | |

The two rankings share only **73.5%** of their top-3,440 channels (min 62.1%, max
92.9%), so this is a real change in what gets kept. Concentration is essentially
unchanged (0.226 → 0.209), so it is not a sharper ranking — just a less parochial
one. Note also that eval A's contiguous number reproduces the independently
measured 8.9096 to within 0.0015.

### The consequence

| | eval B ppl |
|---|---:|
| pilot's trained MoE, step 150 (built on the **contiguous** ranking) | 14.9314 |
| contiguous static reference | 15.6918 → **beat by 4.85%** |
| **corpus-wide static reference** | **12.9605 → LOSES by 13.20%** |

**Fixing the ranking beats adding routing.** The pilot's equal-active-compute win
was an artifact of a weak baseline, and it is withdrawn as a headline result. A
better static channel set, obtained from a 40-second ranking run with **no
training at all**, beats the trained MoE by 13.2% on the same split at the same
active compute.

### What this does and does not establish

It does **not** show routing is worthless. The comparison is across checkpoints:
the trained MoE was built on the contiguous ranking, and a MoE converted from the
corpus-wide ranking would start from a better place and has never been tried. What
it establishes is an ordering of effort:

1. **Ranking quality is the larger and far cheaper lever** — 17.4% out of
   distribution, for one forward pass over 32 prompts.
2. Any future routing claim must be measured against a **corpus-wide-ranked**
   static baseline, not a parochial one. Measured against the right baseline, this
   pilot's routing contribution is negative.
3. The A→B spread narrowed from 2.16× to 1.62× but did not close, so some of the
   difficulty difference between the splits is real rather than an artifact of
   where the ranking came from. (Worth noting the dense model finds eval B
   *easier* — 4.3044 vs 5.2707 — yet it is much harder to prune. Whatever explains
   the residual spread has to be consistent with that, and nothing here does.)

Acted on in code: `channel_importance.spread_across` now exists to pick a
corpus-wide ranking split with the eval splits excluded structurally, and the
module docstring carries these numbers so the next caller does not repeat it.

## Designated follow-ups, in order

1. ~~Measure the static-prune reference on eval B.~~ **Done — see the addendum.**
   It produced the program's first equal-active-compute win (−4.85%) and the
   ranking-generalisation finding.
2. ~~Rank channels on a corpus-wide split, not a contiguous one.~~ **Done — see
   Addendum 2.** Both predicted effects occurred, and the second dominated: the
   OOD penalty fell 17.4%, and the routing win it was credited with vanished.
   **Re-convert from the corpus-wide ranking** and re-run the pilot against the
   corrected baseline — that is now the only way a routing claim can mean
   anything here.
3. **Re-run with eval B pre-registered as primary** plus a third fresh split, at
   lr 1e-4–3e-4 with fewer epochs. The step-75 instability and the 24.8k-token
   corpus are both fixable.
4. **Router-only arm** (`ROUTER_ONLY_SUFFIXES`), so the gate cannot launder the
   result either way. Router-only already sits at −0.77% on eval B; a clean arm
   would say whether that is real.
5. **A larger hot core**, since the gate spent its capacity asking for one — and
   the core alone at 2,176 channels costs 7.347× on eval B, so the routed half is
   doing substantial work there.
