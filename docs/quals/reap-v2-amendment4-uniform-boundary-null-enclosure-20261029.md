# Amendment 4: continuous-nuisance null-size enclosure (research only)

**Scope:** Frozen HumanEval population `n=164`, paired candidate-minus-reference
noninferiority boundary `d=-0.02`. This is a mathematical method audit, not a
model result, an official gate, or a prospective statistical-method decision.
The frozen selector and all original evidence remain unchanged.

The previous exact-null calibration sampled a fixed set of discordance
probabilities `q=p10+p01`. Sampling does not establish worst-case type-I error
over the full nuisance range `q in [0.02, 1]`. This independent extension
constructs the **entire degree-164 rejection-probability polynomial** in the
Bernstein basis, then bounds it using the convex-hull property and adaptive
de Casteljau subdivision.

Define `t=(q-0.02)/0.98`. The null multinomial trial distribution is a mixture
of two endpoint distributions: `A=(p10=0,p01=.02,p_concordant=.98)` with weight
`1-t`, and `B=(p10=.49,p01=.51,p_concordant=0)` with weight `t`. Conditional on
exactly `j` B trials among 164, `candidate_only~Bin(j,.49)` and independent
`extra_reference_only~Bin(164-j,.02)`. The conditional rejection probability
is the Bernstein coefficient `a_j`. Thus

`P(reject | q) = sum_j a_j * C(164,j) * t**j * (1-t)**(164-j)`.

Every Bernstein polynomial value lies between its smallest and largest
coefficients; the same property applies after subdivision on any subinterval.
Unlike a fixed grid, this bounds **all** nuisance values, subject to ordinary
floating-point arithmetic. An explicit `1e-10` numerical slack is added, but
this is **not** a formally directed-rounded interval-arithmetic certificate.

At `tolerance=0.0001`, the endpoint-mixture Bernstein coefficients already
establish the following continuum-wide numerical enclosures without needing
subdivision:

| One-sided 95% lower-bound rule | Numerical worst-case false-approval enclosure | Nuisance endpoint attaining observed lower bound |
| --- | ---: | ---: |
| Tango paired efficient-score | 0.056557711809–0.056557712009 | `q=1.00` |
| Frozen Section 4 Wald diagnostic | 0.158219474122–0.158219474322 | `q=0.02` |

Both methods exceed nominal 5% at some valid boundary-null nuisance value.
The Wald diagnostic reaches approximately **15.82%**, while the paired score
diagnostic reaches approximately **5.66%**. The score procedure improves the
low-discordance behavior substantially but is not an exact uniformly 5%-level
procedure. These results strengthen the requirement for independent,
**prospective** adjudication of the statistical method; they do not themselves
select a replacement or modify the original preregistration.

Reproduction (CPU-only, no original model data):

`python -B -m pytest -q -p no:cacheprovider tests/test_reap_amendment4_uniform_null_enclosure_20261029.py tests/test_reap_amendment4_exact_null_calibration_20261029.py tests/test_reap_amendment4_paired_score_research_20261029.py`

The tests independently compare the Bernstein polynomial against the exact
multinomial enumeration at eight nuisance values, exercise a synthetic
interior supremum and fail-closed subdivision budget, and assert that all
results remain `BLOCKED`, `chosen=null`, `release_approved=false`.

**Scientific limits:** This covers the boundary `d=-0.02` for HumanEval only;
it is not a full composite-null size proof for every `d<=-0.02`, a confidence
interval coverage theorem, an official observed Q8-vs-dense result, or
independent statistical-authority approval. Original frozen Gate 1/Gate 2 and
all remaining mandatory scientific qualifications remain blocked.
