# Amendment 4: exact-integer continuum certificate (research only)

**Date:** 2026-10-29. **Scope:** HumanEval's frozen 164 paired items; the
boundary-null candidate-minus-reference difference is exactly `-0.02`.
This is a statistical-method audit, not an observed Chowder score or a
prospective selection of the frozen Amendment 4 confidence interval.

## Why a second certificate is needed

The prior Bernstein analysis encloses the entire nuisance interval
`q = p10 + p01 in [0.02, 1]`, but computes its coefficients with binary
floating-point arithmetic and uses a non-formal numerical slack. The new
`chowder_batch.reap_amendment4_integer_null_certificate_20261029` module
constructs all 165 coefficients as **exact integers**, with common denominator
`100**164`, and checks endpoint dominance using only integer comparisons.

At `q=0.02`, a trial's `(p10,p01,pconcordant)` is `(0,2,98)/100`;
at `q=1`, it is `(49,51,0)/100`. With `t=(q-0.02)/0.98`, independently
select the second endpoint for each of 164 trials with probability `t`.
Conditional on `j` selections, the candidate-only count from those trials
is `Bin(j,49/100)`, and the additional reference-only count from the other
trials is `Bin(164-j,2/100)`. This gives exact integer Bernstein coefficients
`C_j` for each Boolean rejection predicate. Thus

`P(reject | q) = sum(j=0..164) choose(164,j) t**j (1-t)**(164-j) C_j/100**164`.

All weights are nonnegative and sum to one. For **every** coefficient,
the exact integer comparison establishes `C_j <= C_164` for the Tango
score research predicate, and `C_j <= C_0` for the frozen Section 4 Wald
diagnostic. Because those endpoint coefficients are themselves attained,
they are the **global** maxima throughout the continuum, with no numerical
optimizer or floating-point enclosure needed for this step.

| Predicate | Attained maximum | At q | Coefficient-vector SHA-256 |
|---|---:|---:|---|
| Tango efficient-score research predicate | 0.056557711909011184409219801286834145262821305687887 | 1 | `81cc0ffa828ffc16a2781c7b2b1e7d73518650dc9aaff3e206f7cd9887fa857b` |
| Frozen Section 4 Wald diagnostic | 0.15821947422228511136236118986400860133971492129124 | 0.02 | `079c9c99a3f4813d7d4979c925a93805cb71491e590421e62bd81acce29e8958` |

The digests describe ordered integer coefficient vectors. They are
reproducibility checks, **not** independent original native-execution
attestations. Exact maximum numerators and the common denominator are
available in the module's returned research object.

## Independent adversarial checks and limitations

The tests independently reconstruct exact multinomial rejection numerators
at `t=1/2` and `t=1/4` and compare them **as integers** with the Bernstein
representation. All 13,695 score decisions are recomputed using 80-digit
Decimal arithmetic rather than the production research `_score_z` function;
the minimum observed separation from the approximate 95% normal cutoff is
greater than `0.0008`. All Wald decisions are independently rederived using
an exact integer inequality for `1.645 = 329/200`. Deliberately altered
Wald decisions and interior coefficients exceeding endpoints are refused.

**Precise scope:** The polynomial coefficient and endpoint-dominance proof
is exact **conditional on the Boolean research decision table**. The
normal-quantile cutoff is independently high-precision cross-checked, but
not a formally directed-rounded proof of the inverse normal CDF. The
Tango method is not uniformly exact at nominal one-sided alpha 0.05:
its maximum boundary-null false-approval probability exceeds 5%.
This calculation does not cover the full composite null by itself, does
not adjudicate Amendment 4's ambiguous method, and does not authenticate
any original v2-v1 paired predictions.

No original evidence, model weights, frozen selector, F: journals, or
benchmark scores are changed. **Gen-0 remains `BLOCKED`, `chosen=null`,
`release_approved=false`** pending every mandatory original-path gate.
