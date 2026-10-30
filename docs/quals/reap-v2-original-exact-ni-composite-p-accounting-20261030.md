# REAP128_v2 original research: conditional tail versus composite-null p-value

**Scope:** CPU-only statistical audit of previously completed original research
certificates. This document is **not** a frozen-gate qualification, not an
Amendment 4 method authorization, and not an iid claim about the fixed
HumanEval164 or GSM8K1319 benchmark populations. No inference was repeated.

The previous prospective research reports contain a **conditional exact upper
tail** computed at a data-dependent 97.5% Clopper–Pearson upper bound on the
discordance probability. Calling that number an unconditional p-value for the
composite paired risk-difference null would overstate the evidence. The
conditional tail is only one part of the prespecified split-alpha test.

For paired iid outcomes, let `q` be the discordance probability, `p` the
candidate-only fraction conditional on discordance, and
`delta = q(2p-1)`. Under `H0: delta <= -0.02`, use a one-sided
`1 - alpha1 = 0.975` Clopper–Pearson bound `U(M)` for `q`, where
`M` is the observed discordance count. The original method sets
`p_null(M) = max(0, (U(M)-0.02)/(2U(M)))` and computes
`T = Pr[Bin(M,p_null(M)) >= candidate_only]` exactly.

On the event `q <= U(M)`, which fails with probability at most `alpha1=0.025`,
the conditional upper-tail is superuniform under the composite null.
Therefore for any threshold `t`, `Pr[T <= t] <= alpha1 + t`. The
conservative **unconditional composite-null p-value**
`p_composite = min(1, alpha1 + T)` is superuniform. The previously registered
decision `T < alpha2=0.025` is equivalent to `p_composite < 0.05`, so
correcting the interpretation **does not change either research decision**.
This bound is conservative and not asserted to be a sharp or uniquely optimal
noninferiority procedure.

| Previously certified paired original research | Conditional tail `T` | Conservative composite-null `p_composite` | Research-only 5% split-alpha decision |
|---|---:|---:|---|
| Windows Q4 vs Linux Q8, GSM8K1319; candidate-only 25, reference-only 21 | 0.0001915331927864680 | **0.0251915331927864680** | Reject `delta <= -0.02` **under iid assumption only** |
| Q8 vs Q4, HumanEval164; candidate-only 5, reference-only 4 | 0.2699031226504845 | **0.2949031226504845** | Do not reject |

The new verifier
`chowder_batch/reap_original_exact_ni_composite_p_witness_20261030.py`
authenticates the two source-locked existing research verifiers, recomputes
their exact tails independently using integer binomial coefficients and
`Fraction`, and verifies the canonical reduced-rational tail hashes. It reads
only the already sealed paired HumanEval certificate and the previously
completed original GSM8K research evidence through their existing verifiers.
Adversarial tests refuse forged original-tail hashes, changed source hashes,
outside-worktree imports, forged release states, invalid populations and
malformed count types.

**Qualification remains blocked.** These pairs are Q4/Q8, not the original
frozen Q8-versus-dense Gate 2; GSM8K was served across different hosts;
native HTTP lineage is not authenticated; and iid sampling has not been
established for these fixed benchmarks. The historical REAP128-v1 Gate 1
failure and unknown REAP160 native speed remain unchanged. No frozen
Amendment 4 ambiguity was resolved or retrospectively reinterpreted.

Gen-0 remains exactly
`{"status":"BLOCKED","chosen":null,"release_approved":false}`.
