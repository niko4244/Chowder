# Original GSM8K1319 cross-host pair: finite-sample paired-NI research

**Date:** 2026-10-30. **Scope:** CPU-only, source-bound, research-only.
**Frozen Gen-0:** `{"status":"BLOCKED","chosen":null,"release_approved":false}`.

## Genuine evidence, without new inference

The new module `chowder_batch/reap_original_gsm8k_crosshost_exact_ni_20261030.py`
reopens the already completed Windows Q4 and separate Linux Q8 GSM8K1319
files through the committed, independently regrading cross-host pair verifier.
It checks that verifier's exact source SHA-256 and the exact source of the
preexisting integer binomial-tail implementation. The original two row SHA
pins are `66d0f39545dd8769e837ed5772a265871cfe148bb9a529ea2188454d72c1243f`
and `9ff2b418fecabdc419d94b2a06f6d9b115bbbd4bfba59ccae2d0d9f81e8ef749`.
The canonical 1,319-item pairing is unchanged:

| Original item outcomes | Count |
| --- | ---: |
| Both correct | 1,199 |
| Windows Q4 only correct | 25 |
| Linux Q8 only correct | 21 |
| Neither correct | 74 |
| Windows Q4 correct | 1,224 / 1,319 |
| Linux Q8 correct | 1,220 / 1,319 |

The observed paired difference (Q4 minus Q8) is **4/1319 = 0.30326
percentage points**. The preexisting two-sided McNemar *equality* test gives
`p = 0.658738077`. This equality test is **not** a test of a -2% noninferiority
margin. A large equality p-value and a small noninferiority p-value can coexist.

## A different, prospective exact noninferiority method

The frozen Amendment 4 method remains unresolved. This module does **not**
reinterpret, replace, or silently change it. It investigates a finite-sample
alternative under an explicit **iid paired-observation** model, which a fixed
canonical benchmark does **not** establish.

Let `B` be candidate-only correct, `C` reference-only correct, `M=B+C`,
`n=1319`, `q=P(discordant)`, `p=P(candidate-only | discordant)`, and
`delta=P(candidate correct)-P(reference correct)=q(2p-1)`.
Under the composite null `delta <= -1/50`, necessarily `q >= 1/50` and
`p <= (q-1/50)/(2q)`. The null contains unknown nuisance `q`; treating the
observed `M/n` as its exact value would be an invalid anti-conservative test.

Instead, allocate `alpha_q=1/40` to a one-sided Clopper-Pearson upper
confidence bound `U(M)` for `q`, computed by integer-only binomial CDF
inversion. The exact root satisfies
`P_{Bin(n,U)}(X<=M)=1/40` when `M<n`; the implementation rounds `U` **up**
to the next 40-bit dyadic. When `M=n`, `U=1`. Consequently
`P_q(q>U(M))<=1/40` for every `q`.

On `q<=U(M)`, the null implies
`p <= p_upper(M)=max(0,(U(M)-1/50)/(2U(M)))`.
Conditional on `M=m`, `B~Bin(m,p)` under iid pairing. Reject when the
**exact** upper binomial tail
`P_{Bin(m,p_upper(m))}(X>=B)` is **strictly less** than `1/40`.
For each `m` satisfying `q<=U(m)`, this has conditional Type I error
at most `1/40` by stochastic ordering. The remaining `m` belong to the
Clopper-Pearson failure event. By the union bound, for every point in the
composite null:

`P(reject) <= P(q>U(M)) + P(reject AND q<=U(M)) <= 1/40+1/40 = 1/20`.

This is a mathematical **finite-sample 5% Type I upper bound under iid**,
not an assertion of independence for the fixed original benchmark. It is
not the frozen Wald/McNemar Amendment 4 procedure and requires independent
statistical authority before any prospective method adoption.

## Exact outcome on the real cross-host pair

For `M=46`, `B=25`, the upward-rounded upper discordance bound is
`U=25424111401/549755813888` (approximately `0.04624618923`).
The corresponding conservative null conditional success upper bound is
`360724878081/1271205570050` (approximately `0.28376596719`).

The exact conditional upper-tail p-value is
**0.000191533192786468027831074563694606235061504502382**,
strictly below `1/40=0.025`. Therefore this *research method* rejects the
`delta<=-2%` null **conditional on its iid assumptions**. Its reduced exact
rational p-value has SHA-256:
`df86f214dfc914be8bd7c1e8f26c12c3f3f3a51473725f0dd15d42dd54f4de0b`.

The previous ultra-conservative conditional method used the worst-case
`q<=1` and gave approximately `p=0.28162449` on these same counts. The
new CP split spends half of alpha to estimate a conservative upper bound on
discordance, then tests at the remaining half. The different outcomes do
not invalidate either method; they demonstrate different power.

## Strict limitations and reproduction

These original Q4 and Q8 responses were produced in **different serving
sessions and on different hosts**. Neither native HTTP request/response
lineage nor physical same-session comparability is independently certified.
The pair is **not** the mandatory original Q8-vs-dense frozen Gate 1/Gate 2
cohort. No synthetic responses, original score edits, reruns, new source
approval pins, or release decision were created.

Reproduce the deterministic research certificate with:

`python -m chowder_batch.reap_original_gsm8k_crosshost_exact_ni_20261030`

Adversarial tests include independent combinatorial binomial CDF checks,
dyadic CP enclosure inequalities, independent exact p-value calculation,
forged/malformed paired counts, source/module drift, forged pair receipts,
optimized-Python rejection, and explicit nonrelease assertions.

The original v3 AutomationBench checkpoint independently verifies at
**88/600 receipt-backed tasks, 3 strict passes**. A fresh source-pinned
read-only continuation preflight refused native admission due to untrusted
process/GPU-owner inventory and the original 72k conservative resource
floor. Protected Ollama/WSL/Podman workloads were not interrupted.

**No mandatory frozen Gen-0 gate has newly passed.** The original v1 Gate 1
FAIL, REAP160 native speed UNKNOWN, BF16 chained full numerical requirement,
same-host dense comparisons, and full frontier suite requirements persist.
