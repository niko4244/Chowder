# Amendment 4: exact finite-sample null calibration (2026-10-29)

**RESEARCH ONLY — METHOD UNADJUDICATED — Gen-0 BLOCKED.** This calculation
does not read original model predictions, alter the frozen preregistration,
or authorize any v2 replacement. It is an independent follow-up to
`reap-v2-amendment4-paired-method-independent-review-20261029.md`.

For the **frozen HumanEval population size of 164 pairs**, enumerate all
13,695 possible triples `(candidate-only, reference-only, concordant)` under
the *boundary null* `p10 - p01 = -0.02`. Vary the nuisance discordance
probability `q = p10 + p01` over a fixed, disclosed grid. Each table has its
exact multinomial probability. Sum probabilities of the strict event that
the diagnostic one-sided lower confidence bound exceeds `-0.02`.

| Null discordance q | Tango/Nam score rejection probability | Frozen §4 Wald rejection probability |
| ---: | ---: | ---: |
| 0.02 | 0.03639791 | **0.15821947** |
| 0.03 | 0.03092116 | 0.08933648 |
| 0.05 | 0.04118666 | 0.06092371 |
| 0.10 | 0.05126896 | 0.05452483 |
| 0.20 | 0.04917101 | 0.05148490 |
| 0.50 | 0.05102788 | 0.05102788 |
| 0.80 | 0.05024517 | 0.05082375 |
| 1.00 | **0.05655771** | **0.05655771** |

At `q=0.02`, the original §4 Wald rule rejects a *true* `-0.02` boundary
null with probability approximately **15.82%**, compared with **3.64%** for
the paired efficient-score rule. This is a mathematically material difference
in finite-sample false noninferiority certification, not a model score.
The score rule is **not** an exact uniform 5% test either: at `q=1` its
boundary-null rejection probability is about **5.66%**. The grid is not an
optimization over the full nuisance interval and cannot establish uniform
coverage or select a method prospectively by itself.

Implementation: `chowder_batch/reap_amendment4_exact_null_calibration_20261029.py`.
The CPU-only adversarial tests independently check the score-inversion
rejection predicate on **every** possible table and cross-check both `q`
endpoints against separate one-dimensional exact binomial calculations.
The entire multinomial mass is checked to sum to one at every grid point.

These findings reinforce the requirement for an independent, *prospective*
choice of the noninferiority confidence construction, finite-sample coverage
policy, and strict `> -0.02` decision rule before inspecting original paired
v2-v1 outcomes. They do **not** revise historical REAP128-v1 Gate 1 FAIL,
establish REAP160 speed, complete frozen Gate 1/Gate 2, or affect the Gen-0
state `{status: BLOCKED, chosen: null, release_approved: false}`.
