# Original paired research: conservative exact one-sided lower confidence bounds

**Date:** 2026-10-30. **Scope:** Previously completed, immutable original
research-only Q4/Q8 GSM8K and HumanEval164 pairs. **Decision:** Gen-0 remains
`BLOCKED`, `chosen:null`, `release_approved:false`. No frozen gate is qualified.

The previous source-locked paired noninferiority research reported conditional
tail probabilities and conservative composite-null p-values at the fixed
−2 percentage point margin. This separate CPU-only audit inverts the same
predeclared test over nonpositive risk differences to produce an actual
one-sided lower confidence bound, rather than calling a p-value a bound.

## Original results (not new inference)

| Source-locked pair | n | Candidate-only | Reference-only | Conservative 95% lower bound on candidate−reference |
| --- | ---: | ---: | ---: | ---: |
| Windows Q4 vs separate Linux Q8, GSM8K | 1,319 | 25 | 21 | −0.010161817073822021 |
| Original Q8 vs Q4, HumanEval | 164 | 5 | 4 | −0.05852985382080078 |

The dyadic grid has 24 fractional bits. **For these two interior cases**,
each reported lower endpoint is the largest **rejected** null on that grid,
rounded downward to avoid overstating the bound. Its immediately adjacent,
less negative null is not rejected. For the degenerate no-rejection case,
the code returns the trivial lower bound −1 with no adjacent bracket;
when the null at zero is rejected, the result is clipped to zero, again
without an adjacent nonrejected bracket. The first pair's research bound
is above −0.02; the second is below −0.02. The composite p-values at −0.02 remain approximately
0.025191533 and 0.294903123, respectively.

## Statistical method and limitations

Under *iid paired sampling*, write `q=P(discordant)` and
`delta=P(candidate-only)−P(reference-only)`. At any null `delta<=d<=0`,
`q>=−d` and `p=P(candidate-only|discordant)<=(q+d)/(2q)`. Let `U` be the
one-sided 97.5% Clopper–Pearson **upper** bound on `q`. On the event
`q<=U`, the upper null success probability is at most
`max(0,(U+d)/(2U))`. An exact one-sided binomial upper-tail test at 2.5%
then has unconditional size at most 5% by a union bound. This rejection
criterion is monotone in `d`, so inversion yields a conservative 95%
one-sided lower confidence bound. A dyadic upper enclosure of `U` is used,
never a liberal downward approximation. For `d>0`, the monotonic argument
changes direction; this research deliberately clips the bound at zero.

This coverage statement is **conditional on iid paired observations**.
The fixed GSM8K and HumanEval task sets do not establish iid sampling.
The GSM8K serving sessions used different operating systems/hardware,
and neither result is the frozen original Q8-versus-dense Gate 2. The
construction is **not** the frozen Amendment 4's independently authorized
one-sided paired McNemar-style interval. It must not be substituted into
that preregistered decision after observing these outcomes.

The implementation `chowder_batch.reap_original_exact_ni_lower_bound_research_20261030`
requires the exact source SHA of the prior composite-null witness, which
itself authenticates its upstream research modules and previously completed
certificates. The binomial tail is independently recomputed by a finite
integer-coefficient sum; no inference, original journal replay, native
HTTP capture, model-weight change, or F: write is performed.

## Verification

Run `python -B -m pytest -q -p no:cacheprovider
tests/test_reap_original_exact_ni_lower_bound_research_20261030.py`.
Adversarial tests cover source drift, foreign imports, invalid populations,
exact binomial arithmetic, adjacent rejection boundaries, degenerate
discordance, and refusal to promote research into release authority.

The remaining mandatory work is original same-session dense controls,
native request/response attestation, chained BF16 numerical evidence,
REAP160 speed/quality, remaining registered suites, and independent
prospective Amendment 4 statistical adjudication. Historical REAP128-v1
Gate 1 remains **FAIL**; REAP160 native speed remains **UNKNOWN**.
