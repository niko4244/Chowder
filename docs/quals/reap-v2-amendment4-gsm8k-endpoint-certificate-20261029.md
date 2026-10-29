# Amendment 4: exact GSM8K endpoint-null certificate (research only)

**2026-10-29 — mathematical method audit, not model inference or gate approval.**
The frozen Amendment 4 optional replacement clause in
`docs/REAP_QWEN36_PREREG.md` lines 171–180 requires a strict one-sided 95%
paired lower bound above `−0.02` on both GSM8K (`n=1,319`) and HumanEval
(`n=164`). The clause calls the method "paired McNemar" while original §4
(lines 55–59) explicitly defines a sample-SD Wald bound with multiplier
`1.645`. Its nonzero-margin confidence method remains unresolved. The
existing complete exact-rational and Bernstein analyses concern `n=164`.

The independent CPU-only module
`chowder_batch/reap_amendment4_gsm8k_endpoint_certificate_20261029.py`
extends the audit to **one rigorously specified null point for `n=1,319`**.
It is a lower-bound counterexample to nominal 5% type-I-error control for
each *research diagnostic*, not an assertion about a global supremum, a
genuine score, or the interpretation that the prospective independent
statistical authority must ultimately choose.

## Exact endpoint distribution

For hypothetical independent, identically distributed paired Bernoulli
observations, let `B` count candidate-only correct responses, `C` count
reference-only correct responses, and `K` collect concordant pairs. At the
noninferiority boundary `delta=p10−p01=−1/50`, choose the admissible nuisance
endpoint `q=p10+p01=1/50`:

```
p10=0, p01=1/50, p_concordant=49/50.
B=0 with probability 1; C~Binomial(1319,1/50).
P(C=c)=choose(1319,c)*2**c*98**(1319-c)/100**1319.
```

The **frozen §4 Wald diagnostic** rejects the true boundary null (an
"approval" under optional replacement) exactly when

```
S=1319-50*c > 0
and 16*S**2*1318 > 329**2*(1319*c-c**2).
```

That is the strict comparison `LB>−0.02` after squaring the original
`mean(d)−1.645*sd_sample(d)/sqrt(1319)` inequality; the positive `S`
condition is checked *before* squaring. Exhaustive integer checks for every
`c=0…1319` show rejection iff `c<=19`. The integer difference of the two
sides is `+197810468` at `c=19`, then `−666165212` at `c=20`.

The **illustrative Tango/Nam efficient-score diagnostic** profiles the
discordance nuisance at the fixed null `delta=−1/50`. If its observed-minus-
null difference is positive, then `c<=26`, and the log-likelihood's derivative
at the smallest admissible discordance `q=1/50` is nonpositive because
`51*c<=2*1319`. Concavity proves the constrained MLE is `q_hat=1/50`, giving

```
Z=(1319-50*c)/(7*sqrt(1319)).
```

For `c>=27` the numerator is nonpositive, so the one-sided test cannot
reject. For `c<=26`, compare the exact rational squared statistic with the
two rational cutoffs enclosing `z_0.95`: `1.6448<z_0.95<1.6449`. The source
proof of that bracket is
`chowder_batch/reap_amendment4_rational_score_certificate_20261029.py`
(Machin's identity, alternating rational Taylor bounds). Any result between
the cutoffs fails closed. Every `c=0…1319` is decidable, with rejection iff
`c<=18` (`c=18` is above the upper cutoff; `c=19` below the lower cutoff).

## Exact false-approval probabilities at this null point

| Research predicate | Rejecting `C` values | Exact rational probability | Decimal probability |
| --- | --- | --- | ---: |
| Frozen §4 Wald | 0–19 | `sum(c=0..19) choose(1319,c)*2**c*98**(1319-c) / 100**1319` | **0.0830519314943311840387941593516958484130062404684650620036574…** |
| Tango/Nam score | 0–18 | `sum(c=0..18) choose(1319,c)*2**c*98**(1319-c) / 100**1319` | **0.0544459938351120922216792829800297970935316335823060556639997…** |

The numerator and denominator are represented by exact Python integers, and
each numerator is verified against `denominator/20` by integer comparison.
The module reports complete exact integer strings, decimal expansions,
decision-vector SHA-256 digests and research-only qualification flags. The
binomial weights are independently verified to sum to `100**1319`.
Thus the **Wald endpoint size is 8.3051931494…%** and the **score endpoint
size is 5.4445993835…%**, both strictly above the nominal one-sided 5%.
Their global worst-case false-approval probabilities over the entire
`n=1319` composite null are at least these values; this certificate makes no
claim that the displayed endpoint is globally least favorable.

## Reproduction and boundaries

Run the pure CPU test suite from the repository root:

```
python -B -m pytest -q -p no:cacheprovider tests/test_reap_amendment4_gsm8k_endpoint_certificate_20261029.py
```

The tests exhaustively cross-check the **1,320** possible endpoint tables
against the separately implemented Wald and paired-score diagnostics, prove
both exact decision transitions, recalculate binomial numerators by an
independent integer recurrence, and exercise invalid inputs, ambiguous
quantile bounds, nonmonotone decisions, and forged probabilities.

**Scientific scope:** These are hypothetical iid sampling null-rejection
probabilities, not observed REAP128_v2-versus-v1 outcomes, a generalization
certificate for the fixed canonical GSM8K benchmark, or a confidence-interval
procedure prospectively approved by an independent authority. No original
models, F: evidence, selector, preregistration, or gate outputs are read or
mutated. No inference, GPU work, or benchmark evaluation occurs. The
original fallback order and all production requirements remain unchanged.
The result is always `research_only=true`,
`frozen_amendment4_method_adjudicated=false`,
`eligible_for_optional_v2_replacement=false`, `gen0_status=BLOCKED`,
`chosen=null`, `release_approved=false`.
