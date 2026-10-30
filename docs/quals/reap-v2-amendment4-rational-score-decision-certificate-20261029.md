# Amendment 4: rational certificate of the paired score decision table

**Date:** 2026-10-29. **Research only.** This is a mathematical audit of the
HumanEval `n=164` paired-score candidate method, **not** a measured Chowder
benchmark, an approved interpretation of Amendment 4, or a release gate.

The earlier exact-integer Bernstein certificate proved its nuisance-continuum
maxima conditional on 13,695 Boolean score decisions. It independently checked
those decisions at high decimal precision but did not formally bracket the
normal quantile. The new
`chowder_batch.reap_amendment4_rational_score_certificate_20261029` module
closes that **specific mathematical conditionality** without invoking a
floating-point inverse normal CDF.

For the one-sided 95% standard-normal cutoff, Machin's identity
`pi = 16*atan(1/5) - 4*atan(1/239)` and exact rational alternating-series
bounds enclose `pi`. Squared rational inequalities then enclose
`1/sqrt(2*pi)`. An integrated alternating Taylor series for
`integral_0^x exp(-t*t/2) dt` proves the strict inequalities
`Phi(1.6448) < 0.95 < Phi(1.6449)` using only rational arithmetic.

For every possible discordant table `(b,c)`, the constrained nuisance MLE
at null difference `-1/50` is enclosed using scaled **integer square roots**.
The positive squared efficient-score statistic is compared against both
rational cutoff endpoints. Cells too close to resolve would fail closed;
none of the 13,695 original research cells is ambiguous. The resulting
decisions match the earlier independent research score table exactly.

The rational certificate yields **5,955 rejecting tables out of 13,695**;
its ordered `(b,c,reject)` vector SHA-256 is
`0eb2615268c596f3cb74ecfb3def9beab467dbe4f5101eb7fb04d00378ef990c`.
The earlier integer Bernstein coefficient-vector SHA-256 remains
`81cc0ffa828ffc16a2781c7b2b1e7d73518650dc9aaff3e206f7cd9887fa857b`.
Together, the two research implementations certify the score method's
boundary-null maximum of approximately **5.655771%**, attained at discordance
probability `q=1`, without a floating-point quantile assumption. The Wald
diagnostic's independently exact integer maximum remains **15.821947%** at
`q=0.02`. Both exceed nominal one-sided 5% at their respective worst cases.

**Limitations:** The mathematical certificate does not authenticate any
original native model output, establish the full composite-null behavior of
other interval methods, or choose an Amendment 4 method prospectively.
The Tango score research predicate is demonstrably not uniformly exact at
nominal alpha 0.05. Frozen original Gate 1/Gate 2, dense comparison, BF16
chained numerics, REAP160, and frontier suites remain unqualified.

No original F: evidence, weights, source certificates, or frozen selector are
changed. Gen-0 remains `BLOCKED`, `chosen=null`, `release_approved=false`.
