# Amendment 4: exact conditional noninferiority method candidate (research only)

**2026-10-29.** This is an independent finite-sample method investigation,
not a retrospective replacement for the frozen preregistration, a benchmark
execution, or a release decision. The frozen text ambiguously refers to a
"paired McNemar" one-sided 95% lower confidence bound at the **nonzero**
margin `−0.02`, while §4 specifies a sample-SD Wald lower bound. Neither the
historical research-only Q4/Q8 results nor this method can substitute for the
original frozen Q8-vs-dense Gate 2 evidence or prospective independent
statistical-method adjudication.

## A finite-sample exact, uniformly valid alternative

For an iid paired binary experiment define `B` as candidate-only correct,
`C` as reference-only correct, `M=B+C`, `n` as total paired tasks,
`q=P(discordance)`, and `δ=P(candidate only)−P(reference only)`.
Under the composite noninferiority null `H0: δ≤−1/50`, the conditional
candidate-only probability given discordance satisfies

```
p = P(candidate only | discordance) = (q+δ)/(2q)
  ≤ (q−1/50)/(2q) = 1/2−1/(100q) ≤ 49/100,
```

because `1/50≤q≤1`. Conditional on `M=m`, the count `B` is
`Binomial(m,p)` under iid sampling. Thus the upper-tail p-value

```
P_{Bin(m,49/100)}(B'≥B)
  = [sum(k=B..m) choose(m,k) 49^k 51^(m−k)] / 100^m
```

is superuniform under every composite-null parameter **conditional on each
possible `M=m`**. The binomial family is stochastically increasing in `p`,
so the worst case for each `m` is `p=49/100`; the ordinary discrete upper-tail
p-value at that parameter rejects with probability at most `1/20`. Averaging
over any distribution of `M` preserves the bound. This proves exact
finite-sample Type I error **≤5% over the entire iid composite null** for
both `n=164` and `n=1319`, without normal approximations or numerical
optimization. The implementation uses a **strict** `p<1/20` integer
comparison, making the discrete rule slightly more conservative.

The same test has a conservative **one-sided paired risk-difference lower
confidence bound with at least 95% coverage**. Let `p_L` be the one-sided 95%
Clopper–Pearson lower bound for `p` conditional on the observed `M=m`
(`p_L=0` if `m=0` or `B=0`; otherwise its defining binomial upper tail is
`1/20`). Define `L=min(0,2*p_L−1)`. On the event `p_L≤p`, if `p≤1/2`
then `δ=q(2p−1)≥2p−1≥L`; if `p>1/2`, then `δ≥0≥L`.
Thus `P(L≤δ)≥0.95` under iid paired sampling, including conditioning on
`M`. Inverting the exact binomial upper tail gives `L>−1/50` exactly when
the module's strict `p-value<1/20` predicate holds. The implementation
certifies this comparison using exact integers rather than numerically
approximating `p_L` or asserting an uncomputed confidence endpoint.

The proof is deliberately conditional; no independence between `B` and `M`
is claimed. At `q<1`, the bound `p≤49/100` is loose and the procedure may be
substantially conservative. It is **not** the standard exact conditional
McNemar *equality* test (which uses `p=1/2` under `δ=0`), nor the Wald or
Tango/Nam method already analyzed in separate certificates. It is an
alternative exact noninferiority test requiring a **prospective** independent
method decision before it could have any scientific-gate relevance.
When `M=0`, its lower bound is `−1` even if the candidate and reference
agree on every task, illustrating potentially very low power.

## Source and arithmetic safeguards

`chowder_batch/reap_amendment4_exact_conditional_ni_20261029.py` accepts only
complete canonical populations of 164 or 1,319 paired items. It evaluates
binomial weights with exact Python integers, returns the exact numerator and
denominator, refuses malformed counts, and never reads original F: archives,
changes a scorer, calls a model, or touches the frozen selector. Decimal
display is informational; the strict decision uses integer arithmetic.

The companion tests independently reconstruct exact tails with `math.comb`,
check every HumanEval-sized discordance table, exercise multiple composite-null
parameters, test the strict 5% boundary and malformed inputs, and verify that
even a hypothetical positive research test cannot authorize the selector.

For illustration only, the **previously completed research-only** HumanEval
Q8-vs-Q4 pairing (`both=147`, `Q8-only=5`, `Q4-only=4`, `neither=8`) has
`M=9` and is not a positive result under this conservative exact method.
This is **not** Q8-vs-dense and cannot be used as Gate 2.

The existing GSM8K boundary-null counterexample has `B=0,C=18,n=1319`.
The Wald diagnostic can falsely approve at this null point, whereas the
conditional exact p-value is one and does not reject. This is an illustration
of a different rejection rule, not a claim about observed model outcomes.

## Reproduction and scientific limits

```
python -B -m pytest -q -p no:cacheprovider tests/test_reap_amendment4_exact_conditional_ni_20261029.py
```

This is an iid paired-multinomial *sampling* result. A fixed canonical
benchmark does not automatically justify iid generalization or model
execution authenticity. The method is not an independently approved frozen
Amendment 4 confidence interval. The frozen fallback order, historical
REAP128-v1 Gate 1 FAIL, unknown REAP160 native speed, and all mandatory
original scientific requirements remain unchanged.

**Always:** `research_only=true`,
`frozen_amendment4_method_adjudicated=false`,
`eligible_for_optional_v2_replacement=false`,
`gen0_status=BLOCKED`, `chosen=null`, `release_approved=false`.
