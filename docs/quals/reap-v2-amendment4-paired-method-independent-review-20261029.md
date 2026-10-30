# Amendment 4: prospective paired noninferiority method review (2026-10-29)

**Status: RESEARCH ONLY / METHOD NOT ADJUDICATED / Gen-0 BLOCKED.** This
document is an independent statistical-method review, not a change to the
2026-10-19 frozen preregistration, `reap_select.py`, or any recorded score.

The preregistration says `LB(v2 - v1) > -0.02` for both GSM8K (1,319 paired
items) and HumanEval (164 paired items), using a one-sided 95% **paired
McNemar** lower confidence bound. Classical McNemar's test concerns the
equality null `p10 = p01` (risk difference zero). It does **not** uniquely
specify a confidence limit for the nonzero margin `-0.02`. Section 4 instead
explicitly defines a sample-standard-deviation Wald lower bound, with
multiplier `1.645`. Neither statistic can silently replace the other.

Independent statistical authorities:

- Tango, T. (1998), *Equivalence test and confidence interval for the
  difference in proportions for the paired-sample design*, Statistics in
  Medicine **17**, 891–908, DOI `10.1002/(SICI)1097-0258(19980430)17:8<891::AID-SIM780>3.0.CO;2-B`.
  The paper derives an efficient-score one-sided equivalence test and paired
  confidence interval; ordinary McNemar equality testing is a special case.
- Fagerland, M. W., Lydersen, S. and Laake, P. (2014), *Recommended tests and
  confidence intervals for paired binomial proportions*, Statistics in
  Medicine **33**, 2850–2875, DOI `10.1002/sim.6148`. They evaluate 24
  methods and recommend score intervals among the methods for paired risk
  differences, distinguishing these from McNemar tests of association.
- `statsmodels.stats.proportion.confint_proportions_2indep` is explicitly
  for **independent** binomial samples; it is not a valid paired-data shortcut.

The CPU-only `reap_amendment4_paired_score_research_20261029.py` independently
profiles the multinomial nuisance discordance probability under a fixed
paired risk difference and inverts the one-sided efficient-score test. It
also reports the exact conditional two-sided McNemar **equality** p-value
and the original §4 Wald statistic. Every result is labeled diagnostic,
and the API always reports `eligible_for_optional_v2_replacement: false`,
`gen0_status: BLOCKED`, and `release_approved: false`.

For a synthetic HumanEval-size table with 164 concordant correct pairs and
zero discordance, the original Wald lower bound is exactly zero, while the
paired score lower bound is `-z^2/(164+z^2)` (about `-0.01623`). This is a
method discrepancy, **not** a measured model result. No research-only Q4/Q8
HumanEval comparison is the original v2-v1 frozen Gate 2.

More sharply, **the methods can disagree about the strict `-0.02` boundary**
on legitimate synthetic paired tables of exactly the frozen sizes:

| Synthetic cells (both right, v2 only, v1 only, both wrong) | Frozen §4 Wald LB | Tango score LB |
| --- | ---: | ---: |
| HumanEval-size `(163, 0, 1, 0)` | `-0.016128` | `-0.026865` |
| GSM8K-size `(1300, 0, 19, 0)` | `-0.019804` | `-0.020881` |

The Wald diagnostic exceeds `-0.02` in both examples, whereas the paired
score diagnostic does not. This establishes a **material methodological
ambiguity** without looking at any actual v2-v1 predictions or retrofitting
a gate after seeing outcomes. Neither diagnostic is a qualification verdict.

**Required before optional v2 replacement can be certified:** an independently
approved prospective method statement identifying the paired marginal-risk
difference estimand, interval construction (e.g. Tango/Nam score inversion),
one-sided 95% convention, strict comparison with `-0.02`, zero-discordance
behavior, and independent implementation cross-check. Only after that can
genuine source-locked, canonical, paired v2-v1 official GSM8K/HumanEval
predictions be evaluated. The mandatory v1 Gate 0/1 failure fallback order
remains distinct and unchanged. No selector change, threshold movement,
model inference, or evidence mutation is authorized by this review.
