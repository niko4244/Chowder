# REAP128_v2 paired noninferiority: prospective methods review (2026-10-25)

**Status: UNRESOLVED — research / governance review only; NOT a statistical amendment, selection receipt, or Gen-0 authorization.**

**Scope:** Optional replacement of a fully passing REAP128 v1 with REAP128_v2 under Amendment 4 §2. This review does **not** change the original §4 gates, the Amendment 4 §3 mandatory ordered fallback after v1 Gate 0/1 failure, the `−0.02` optional replacement margin, model or dataset identities, or the current selector's fail-closed `BLOCKED` behavior.

## 1. Frozen sources and precise conflict

1. [`docs/REAP_QWEN36_PREREG.md`](../REAP_QWEN36_PREREG.md) lines 55–59, frozen 2026-10-17, defines per-item paired differences `d_i = x_i(arm) − x_i(dense)` in `{−1, 0, 1}` and the **specific** one-sided lower bound `LB = mean(d) − 1.645 × sample_sd(d) / sqrt(n)`. Section 4 applies this bound to original Gate 1 (margin `−0.03`), Gate 2 (`−0.10`), and the negative-control comparison (strictly greater than zero).
2. [Amendment 4 §2](../REAP_QWEN36_PREREG.md) lines 171–181, written 2026-10-19 **after raw v1 summaries were visible**, adds an optional v2-v1 noninferiority comparison on **both** GSM8K (1,319 pairs) and HumanEval (164 pairs): `LB(v2 − v1) > −0.02`, “paired McNemar, one-sided 95%.” It does **not** specify a confidence-interval construction for the paired **marginal accuracy difference** at the **nonzero** margin.
3. `chowder_batch/reap_gates.py:lower_bound` computes the frozen sample-standard-deviation Wald bound. `chowder_batch/reap_select.py:_paired` uses this for a **diagnostic** v2-v1 result, but `select` returns `BLOCKED` for an otherwise eligible optional v2 replacement, citing `PAIRED_METHOD_DISCREPANCY`; it does not reinterpret Wald as a McNemar noninferiority CI.

The conventional exact McNemar test conditions on the number of discordant pairs and tests equality of paired proportions (marginal difference `Δ=0`). It is **not** by itself a one-sided lower confidence limit for the marginal difference at `Δ=−0.02`. A conditional sign-test confidence interval on the *relative direction among discordants* is likewise not automatically a CI for `(b−c)/n`. A “McNemar-based” confidence bound needs additional, frozen specification—e.g., an appropriate paired score interval, its nuisance treatment, and its behavior at zero discordant cells. Multiple published constructions can disagree. The name “paired McNemar” alone cannot disambiguate those choices.

Relevant external methodology, used only to inform this review:

- T. Tango (1998), *Equivalence test and confidence interval for the difference in proportions for the paired-sample design*, *Statistics in Medicine* **17**, 891–908, DOI [10.1002/(SICI)1097-0258(19980430)17:8<891::AID-SIM780>3.0.CO;2-B](https://pubmed.ncbi.nlm.nih.gov/9595618/). Describes a paired efficient-score noninferiority procedure, with McNemar's equality test as a special case.
- M. W. Fagerland, S. Lydersen, and P. Laake (2014), *Recommended tests and confidence intervals for paired binomial proportions*, *Statistics in Medicine*, DOI [10.1002/sim.6148](https://doi.org/10.1002/sim.6148). Compares multiple paired-proportion procedures and recommends score-type intervals among other methods.
- *Continuity corrected score confidence interval for the difference in proportions in paired data* (2022), [PMC10763857](https://pmc.ncbi.nlm.nih.gov/articles/PMC10763857/), surveys paired Wald, score and continuity-corrected variants. This is evidence **of method multiplicity**, not approval for switching to one.

## 2. Exact estimand and noninferiority hypothesis

For each **identical canonical benchmark item**, record the binary tuple `(v2_correct, v1_correct)`:

| Symbol | v2 | v1 | Meaning |
|:--|:--:|:--:|:--|
| `a` | 1 | 1 | both correct |
| `b` | 1 | 0 | v2-only correct |
| `c` | 0 | 1 | v1-only correct |
| `d` | 0 | 0 | both incorrect |

`n=a+b+c+d`, `Δ=P(v2 correct)−P(v1 correct)=P(b)−P(c)`, `Δ_hat=(b−c)/n`. **The margin is an absolute 2 percentage points, not an odds ratio, percentage of discordants, or percentage relative to v1 accuracy.** The intended optional replacement comparison is `H0: Δ ≤ −0.02` against `H1: Δ > −0.02`, separately for the frozen 1,319-item and 164-item cohorts. Both must meet the same **strict** inequality at a **one-sided 95%** level (`α=0.05`); equality at the threshold does not pass.

The §4 diagnostic Wald formula uses the **sample** standard deviation of the `n` values `{+1 for b, −1 for c, 0 otherwise}`, with `z=1.645`, not the population standard deviation or a two-sided `1.96` constant. Changing these implementation details retrospectively is not acceptable.

## 3. Deterministic adversarial examples — hypothetical, not measured quality

The optional research-only `tests/test_reap_paired_method_prospective_20261025.py` computes:

1. the **exact existing** §4 diagnostic Wald lower bound; and
2. an **illustrative, unapproved** constrained-nuisance paired score statistic at `Δ0=−0.02`, of the Tango/Nam score family. It takes `q=P(b)+P(c)` and maximizes `b·ln((q+Δ0)/2)+c·ln((q−Δ0)/2)+(n−b−c)·ln(1−q)` over `|Δ0|≤q≤1`. The illustrative score is `Z=(Δ_hat−Δ0)/sqrt((q_tilde−Δ0²)/n)`, compared to the **one-sided** standard-normal `z_(0.95)=1.64485362695`. This is a contrast at the **fixed null boundary**, **not** a fully tested or approved Tango CI implementation, a confidence-coverage certificate, or a proposed executable production gate.

The concordant counts may be distributed between `a` and `d` arbitrarily for this difference calculation. For all rows below, `b` is v2-only correct and `c` v1-only correct.

| Suite/illustrative table | `b` | `c` | `Δ_hat` | Wald lower bound | Wald above `−0.02`? | Illustrative score `Z` at `−0.02` | Score `Z > 1.64485`? |
|:--|--:|--:|--:|--:|:--:|--:|:--:|
| HumanEval, `n=164` | 0 | 1 | −0.006098 | **−0.016128** | **YES** | **1.271701** | **NO** |
| GSM8K, `n=1319` | 0 | 19 | −0.014405 | **−0.019804** | **YES** | **1.451463** | **NO** |
| HumanEval, `n=164`, no discordants | 0 | 0 | 0 | 0 | YES | 1.829464 | YES |
| HumanEval, `n=164`, high discordance | 20 | 21 | −0.006098 | −0.070516 | NO | 0.355769 | NO |
| HumanEval, `n=164`, favorable discordance | 5 | 0 | +0.030488 | +0.008336 | YES | 2.906713 | YES |

**Consequential divergence:** the first two hypothetical tables satisfy `Wald_LB > −0.02` but do not reject the noninferiority null under the illustrative score alternative. Therefore replacing the ambiguous method **after** inspecting the paired outcomes could change candidate eligibility. These are **counterexamples**, not real v2 results, not a justification to prefer the score variant, and not proof of a universal interval ordering. Neither the Wald nor the illustrative score adjudicates Amendment 4 §2 without a prior, approved method specification.

## 4. Prospective governance rule requiring independent approval

Until there is a documented, independently approved resolution, **keep optional v2 replacement BLOCKED** and label any Wald or exploratory score values *diagnostic only*. A scientifically defensible resolution must satisfy **all** of the following before implementation:

1. **Pre-disclosure control:** appoint an independent statistical reviewer and record exactly which v1, v2, dense and fallback outcomes, summaries, paired discordance tables, interval values, code revisions and internal discussion artifacts have already been observed. Amendment 4 explicitly admits visible raw v1 summaries; this disclosure cannot be erased. The decision maker must not inspect or be briefed on the paired v2-v1 outcomes or exploratory interval comparisons before freezing the choice. If those outcomes **have** been examined by decision makers, a new prospective choice cannot retroactively validate their original cohort; classify the analysis as exploratory and use a *new independent, locked, non-overlapping evaluation cohort with a predeclared estimand/margin and protocol* for any new confirmatory claim. **Do not replace the original frozen 1,319/164 gates with that cohort or pretend an amended analysis was original preregistration.**
2. **Explicit interpretation before numerical use:** document whether this is a clerical clarification that §4 Wald applies to Amendment 4's optional `−0.02` comparison, or a prospective *new* noninferiority protocol requiring a specific paired marginal-risk-difference CI (for example a precisely cited, versioned Tango/Nam score-inversion procedure). Neither path is chosen here. A McNemar equality p-value, conditional odds ratio, Wilson interval for discordant-win proportion, or arbitrary confidence method chosen because it passes **does not** satisfy the frozen risk-difference requirement.
3. **Exact statistical specification:** freeze, before seeing relevant outcomes, the target `Δ=P(v2=1)−P(v1=1)`, the two fixed full item cohorts and common-ID pairing, one-sided `α=0.05`, strict `LB > −0.02`, treatment of ties and all 2×2 cells, nuisance-parameter estimation, confidence construction/test inversion, finite-sample corrections (or none), zero-discordance limits, numerical optimizer and convergence criteria, precision and tolerance around the strict boundary, and the software library/version/source and reference numerical tables. Document whether and how familywise error applies beyond the existing conjunctive two-suite requirement.
4. **No adaptive shopping:** forbid choosing the interval, continuity correction, dataset subset, sequence, one- vs two-sided quantile, definition of `correct`, or common-ID treatment based on which candidate passes. Do not use the “most favorable of Wald and score,” an after-the-fact ensemble of multiple methods, rounded thresholds, or reinterpret any actual v1/v2 results to clear a gate.
5. **Independent implementation validation:** require adversarial synthetic tables (including `b=0`, `c=0`, both zero, near `−0.02`, highly discordant counts, `n=164` and `1319`), independent published-method reference calculations, deterministic exact version-pinned tests, and dual review of test direction and risk difference sign **before** the chosen implementation can affect a decision. Record method and code SHA and the decision authority's dated written signoff. Existing selector and frozen gates stay unchanged until a separately authorized amendment explicitly approves otherwise.
6. **State semantics and branches:** preserve current `BLOCKED` optional replacement if the method is undecided or validated evidence is unavailable. Keep original Gate D0, Gate 0, Gate 1, Gate 2, control, speed/fit thresholds and the mandatory §3 ordered fallback `reap128_v2 → reap160 → reap160_v2` after v1 Gate 0/1 failure **as written**. The optional v2-v1 CI is **not automatically an additional gate on §3 fallback**, and it cannot excuse missing full Q8 quality, authentic source/serving witnesses, or Q4 hardware fit.

**Multiplicity qualification:** requiring noninferiority on *both* fixed suites forms an intersection-union test for the union of suite-specific noninferiority nulls; using one-sided level `0.05` for each can control this specific conjunction without automatically dividing alpha by two. This narrow fact does **not** settle multiplicity created by other adaptive candidate decisions, alternative datasets or repeated post-hoc method searches. No alpha or threshold adjustment is authorized by this review.

## 5. Disposition and reproducibility

- **Frozen policy today:** `PAIRED_METHOD_DISCREPANCY` remains `UNRESOLVED`; optional passing-v1-to-v2 replacement remains `BLOCKED`. No actual model gate verdict, v2 comparison, GPU fit, or Gen-0 selection is certified by this document.
- **What the research test establishes:** the specified finite synthetic tables produce deterministically different hypothetical Wald and paired-score decisions at the exact frozen noninferiority margin; the existing selector remains blocked even when a diagnostic Wald value exceeds the margin.
- **What it does not establish:** the correct interpretation of Amendment 4, a validated production implementation of a named Tango or exact McNemar CI, prospective independence after any outcome disclosure, or coverage properties from simulations or measured model accuracy.
- **Reproduce without GPU, model, network, source dataset or calibration artifacts:**

  ```powershell
  $env:PYTHONDONTWRITEBYTECODE = '1'
  $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
  python -m pytest -q tests/test_reap_paired_method_prospective_20261025.py
  python -m ruff check tests/test_reap_paired_method_prospective_20261025.py
  ```

**Recommendation requiring human decision:** maintain `BLOCKED` until an independent statistical authority signs a dated, genuinely prospective interpretation and method specification. Where prospective conditions are no longer possible, report existing optional replacement as **exploratory, not confirmatory**, rather than choosing a method that happens to admit it.
