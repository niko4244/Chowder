# Amendment 4: exact paired confidence-bound enclosure (research only)

**October 29, 2026.** This is a CPU-only numerical certificate for the
conservative iid exact-conditional method investigated in
`reap-v2-amendment4-exact-conditional-ni-research-20261029.md`. It does **not**
modify the frozen Section 4 Wald method, resolve the preregistered McNemar
ambiguity, authenticate model outputs, or qualify any Gen-0 gate.

## Why a confidence endpoint matters

The earlier exact-conditional module correctly computed a strict upper-tail
noninferiority test at the frozen `-2%` margin, but did not compute the
one-sided lower confidence endpoint discussed in its mathematical proof.
This independent companion computes a **certified rational interval** for
that endpoint, rather than reporting an unverified floating-point root.

For `m=B+C>0` discordances and `B>0` candidate-only successes, the one-sided
95% Clopper–Pearson lower endpoint `p_L` is the unique solution of

```
P[Binomial(m, p_L) >= B] = 1/20.
```

For `B=0` or `m=0`, define `p_L=0`. A binary search over adjacent dyadic
rationals `lo/2^bits` and `hi/2^bits` produces a rigorous enclosure
`lo/2^bits < p_L <= hi/2^bits`, using **integer-only** binomial sums and
strict inequality checks. The default 48-bit grid yields endpoint width at
most `2^-48` for `p_L`, without normal approximations.

Let `q=P(discordance)` and `p=P(candidate-only | discordance)` under iid
paired sampling. The population paired risk difference is
`delta=q*(2*p-1)`. Conditional on each observed `m`, Clopper–Pearson
guarantees `P(p_L <= p | m) >= 0.95` (for `q=0`, `m=0` and the reported
bound is trivially valid). On this coverage event,

```
delta >= min(0, 2*p_L-1) >= min(0, 2*lo/2^bits-1).
```

Thus the **reported lower dyadic enclosure** is itself a conservative
one-sided paired risk-difference lower confidence bound with coverage at
least 95% under the iid model. The module also reports an upper numerical
enclosure, which is **not** an upper confidence bound on the population
parameter. Both endpoints are rational strings; the difference between them
is at most `2/2^bits`.

The exact margin decision is separately computed from the integer upper
tail at `p=49/100`: `20 * tail_numerator < tail_denominator`. A dyadic
interval may straddle the margin; if so, the interval is labeled unresolved
but the exact integer predicate remains definitive. Neither result is a
frozen scientific-gate approval.

For the extreme hypothetical `M=n` (every item discordant), independent
integer enumeration finds the first strictly rejecting candidate-only count
is `B=92` for `n=164` and `B=677` for `n=1319`. At the latter boundary,
`B=676` has null upper-tail probability approximately `0.05395513767487557`
(nonreject), while `B=677` has probability approximately
`0.04818338311919551` (reject). The certified 48-bit risk-difference lower
enclosures at these two GSM8K counts are approximately `-0.02102363373610`
and `-0.01950807177126`, respectively. These are **hypothetical integer
boundary tables**, not observed model results or evidence of iid sampling.

## Independent tests and adversarial safeguards

The companion tests independently evaluate binomial tails using
`math.comb` and `fractions.Fraction`, check the dyadic root brackets at both
canonical cohort scales, enumerate exact small-sample paired-multinomial
coverage probabilities, reject malformed counts and precision settings,
and verify the completed Q8-vs-Q4 HumanEval research pair remains blocked.

```
python -B -m pytest -q -p no:cacheprovider tests/test_reap_amendment4_exact_ci_enclosure_20261029.py
```

This certificate requires **iid paired sampling** and cannot turn a fixed
benchmark into an iid experiment. It does not adjudicate whether the
preregistered nonzero-margin phrase "paired McNemar" should mean Wald,
score, exact conditional, or another interval. Independent prospective
statistical authority remains mandatory. In particular, the completed
research-only Q4/Q8 HumanEval164 arms are not original frozen Q8-vs-dense
Gate 2, and this method cannot select a model.

**Invariant:** `research_only=true`,
`frozen_amendment4_method_adjudicated=false`,
`eligible_for_optional_v2_replacement=false`,
`gen0_status=BLOCKED`, `chosen=null`, `release_approved=false`.
