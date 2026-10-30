# Original HumanEval164 Q8/Q4: exact paired noninferiority research

**2026-10-30 · Research-only · Gen-0 BLOCKED · No frozen-gate promotion.**

This is an independent, CPU-only application of the *prospective* exact
paired noninferiority construction to the **previously completed** original
Q8-versus-Q4 HumanEval164 research pairing. It is **not** the frozen Q8-vs-dense
Gate2, not an independently approved interpretation of Amendment 4, and not a
statistical guarantee for a fixed, nonrandom benchmark population.

## Source-bound, complete original paired evidence

The exact existing certificate at
`F:\reap-v2-original-human164-q4q8-COMPLETE-paired-research-20261027-1525.json`
has SHA-256
`5a2ff5e6ab9f371171de96d368672d52854dd47705c6abb6047de432520893c7`.
It was independently cross-certified previously. The new verifier **only**
reopens that immutable *paired summary*, not either completed execution
journal or model. It checks the certificate SHA, 164 ordered canonical
`HumanEval/0`–`HumanEval/163` records, per-item outcome booleans and program
hashes, original distinct serving sessions/model SHA, original journal SHA
references, and research-only nonapproval fields. It recomputes every cell:

| Outcome on the same canonical 164 tasks | Original count |
|---|---:|
| Both correct | 147 |
| Q8 correct only | 5 |
| Q4 correct only | 4 |
| Both incorrect | 8 |
| Total Q8 correct | **152/164** |
| Total Q4 correct | **151/164** |

The observed paired Q8–Q4 difference is **+1/164 = +0.609756 percentage
points**. This does not by itself establish a confidence-bound claim.

## Prospective exact calculation (not the frozen preregistered method)

Under *iid paired outcomes only*, let `q` be the discordance probability,
`p` the conditional probability of a Q8-only success, and
`delta = q*(2*p-1)`. Under the null `delta <= -0.02`, one must have
`q >= 0.02` and `p <= (q-0.02)/(2*q)`.

The method first constructs an exact one-sided 97.5% Clopper–Pearson
**upper** bound for `q` from 9/164 discordances, rounded upward to a
40-bit dyadic. It then tests 5/9 Q8-only outcomes with an exact conditional
binomial upper tail at the resulting maximum null `p`. Splitting 5% alpha
into 2.5% for each step and applying a union bound controls Type I error at
5% under the stated iid model. This is the same *prospective* method used in
the separately completed GSM8K cross-host research, not a retroactive
substitution for the frozen Amendment 4 procedure.

**Exact conditional p-value:**
`0.269903122650484542621066300635230103607972506407`.

**Result:** The 2% noninferiority null is **not rejected** at the prescribed
conditional 2.5% threshold. This does **not** establish Q8 inferiority,
equivalence, or an actual 2% loss. The 164-item fixed benchmark is not an iid
sample by assumption, and the paired arms were separate original research
serving sessions. The original dense comparator is absent. Therefore this
calculation cannot satisfy Gate2 or Amendment 4, regardless of its outcome.

## Independent verification and exclusions

`chowder_batch/reap_original_human164_crossquant_exact_ni_20261030.py`
source-locks the previously committed exact arithmetic implementation and
the original pair-certificate producer, then rechecks the pinned pair SHA.
`tests/test_reap_original_human164_crossquant_exact_ni_20261030.py` compares
the exact binomial tail to an independently expanded rational binomial sum
over nine discordant trials and tests forged correctness, reordered/missing canonical cases,
bad program hashes, swapped sessions, source drift, and attempted release
promotion. It explicitly prevents opening either original execution journal.

**Focused validation:** 36 CPU-only tests passed; Ruff passed. No model
inference, generated-code execution, F: evidence write, GPU/Podman action,
source/weight/certificate mutation, or original benchmark rerun occurred.

The frozen decision remains exactly:

`{"status":"BLOCKED","chosen":null,"release_approved":false}`.

Remaining original requirements include source-authenticated native HTTP
request/response lineage, dense same-host paired first200 speed and full
GSM8K/HumanEval comparisons, full BF16 chained numerics, REAP160 speed/quality,
random128 frozen control, HLE missing 29, eleven original frontier suites,
and independent prospective Amendment 4 statistical adjudication.
