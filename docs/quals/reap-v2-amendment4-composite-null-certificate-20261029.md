# Amendment 4: full composite-null extension of the exact statistical audit

**2026-10-29 — research only; no original inference or gate approval.**

The earlier exact-rational decision and exact-integer Bernstein certificates
established the maximum rejection probability for the *boundary* null
`delta = p10-p01 = -0.02`, allowing all discordance probabilities
`q = p10+p01` in `[0.02, 1]`. A boundary-only result does not automatically
prove the maximum for the full composite null `delta <= -0.02`.

The new `chowder_batch.reap_amendment4_composite_null_certificate_20261029`
closes that specific mathematical gap under the **iid paired-multinomial
sampling assumption**. It independently verifies all 13,695 exact-rational
score and exact-integer Wald decisions and checks that, for each of the 165
possible discordance counts `M=m`, the rejection predicate is an upper tail
in candidate-only discordances `B` (equivalently, it never reverses from
reject to accept as `B` increases at fixed `m`).

For fixed `q`, the total discordances `M=B+C` follow `Bin(n,q)` regardless
of `delta`. Conditional on `M=m`, `B` follows
`Bin(m, (q+delta)/(2q))`. As `delta` increases, this conditional binomial
success probability increases. Since each rejection set is an upper tail,
the rejection probability is nondecreasing in `delta` **at fixed q**.
Every feasible composite-null point has `q >= |delta| >= 0.02`, so the
boundary point `delta=-0.02` at that same `q` is feasible. Therefore the
full composite-null supremum equals the previously certified boundary-null
supremum, attained at its existing boundary endpoint.

Consequently the exact research maxima remain approximately **5.655771%**
for the Tango paired score predicate (`q=1`) and **15.821947%** for the
frozen Section 4 Wald diagnostic (`q=0.02`). Both exceed nominal one-sided
5%; these are **null false-approval probabilities**, not model benchmark
scores. The exact numerators, denominators, coefficient SHA-256 hashes,
upper-tail cutoff hashes, and score decision SHA-256 are emitted by the
research module for independent inspection.

**Scope:** The statistical claim presumes iid paired-multinomial sampling.
The fixed canonical HumanEval164 benchmark is not itself a random sample
certificate, and this proof does not establish generalization to other
task populations. It does not select an Amendment 4 interpretation,
authenticate any native execution, change a frozen test, or approve release.
Independent **prospective** statistical-method adjudication remains required.
Gen-0 remains `BLOCKED`, `chosen=null`, `release_approved=false`.
