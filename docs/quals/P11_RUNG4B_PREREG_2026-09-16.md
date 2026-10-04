# P11 rung 4b — executable re-execution of rung 4 (measured generation budget)

Frozen **2026-09-16, before the run**. Committed to `niko4244/Chowder` and pushed
**before** `chowder train` is invoked. Rung 4b is *not* a new scientific
question: it is rung 4's frozen experiment, re-executed after correcting the two
independent defects that made rung 4 unrunnable and unsatisfiable.

## 0. Why this rung exists

Rung 4 (`docs/quals/P11_RUNG4_PREREG_2026-09-15.md`, PR #165, merged `ce8ad58`)
was executed on 2026-09-16 and **REFUSED** by its own judge
(`Chowder-Protected/runs/2026-09-16-router-healing-rung4/P11_RUNG4_RESULT_2026-09-16.md`).
The training leg completed 48/48 and produced real routing science. The
evaluation leg refused, and two defects were measured, not assumed:

**Defect 1 — the generation sub-budget was never measured.** Rung 4 declared
`generations = 0.0032` GPU-h (11.5 s) with no measured basis, because rung 3c
performed no generation at all. The frozen T10 probe measured **182.12685860006604 s
= 0.0505907940555739 GPU-h** — 15.8× the declared sub-budget. The evaluator
refused before reporting a score, which took T6–T10 down with it.

**Defect 2 — T9's baseline provenance was unsatisfiable by construction.** T9
required the baseline be "scored on the independent holdout corpus **before the
candidate training run**". The implementation cannot do that under the frozen
`paired_arms = true`: `RouterHealingEvaluator.defers_automatic_baseline()` returns
`paired_arms_for(...)` (`src/chowder/backends/router_healing.py:1532`), and when it
defers, `_run_automatic_baseline()` returns `(None, None)` **without evaluating**
(`src/chowder/project_runner.py:198-206`), creating the row for later completion by
`_paired_baseline_completer()`. `evaluate_base()`'s own docstring states:
"Under a paired project this method is NOT called." So T9-as-written could not
have passed at any budget. This is a method-level contradiction between T6
("baseline row completed from paired evidence") and T9, not a threshold.

**Finding 3 — a same-corpus re-census shows rung 4's T4a failure was
substantively correct.** Measured 2026-09-16, before this prereg was committed:
rung 3c's trained payload re-censused on rung 4's independent holdout with rung
4's instrument reads **431/512** dead, against rung 4's **436/512** and an
exactly-reproduced untrained control of **480/512**. So 48 steps is *not* better
than 12 on a like-for-like basis, and rung 4's hypothesis — that 12 steps was too
short a horizon — is **refuted**. Rung 4's T4a FAIL was therefore correct in
substance; only the bar's corpus was wrong. Full measurement, control and caveats:
`Chowder-Protected/runs/2026-09-16-router-healing-rung4b-paired-cuda/RUNG3C_RECENSUS_RESULT_2026-09-16.md`,
under the protocol frozen in `RUNG3C_RECENSUS_PROTOCOL_2026-09-16.md`. See §3.6.

Rung 4's refusal stands as written in its own document; nothing in it is
retroactively edited. Rung 4b is a forward correction.

## 1. Identity

| Field | Value |
|---|---|
| Run name / ladder rung | P11 rung 4b — executable re-execution of rung 4 |
| Evidence root | `Chowder-Protected/runs/2026-09-16-router-healing-rung4b-paired-cuda` |
| Base artifact | `F:\llm-models\Qwen3.8-9B-HotCore-CW-E16-k2-h2176` |
| Content manifest pin | SHA-256 `77520edadb9a94f4ed70636328c4bbbaafa49c75e3b47c51418851c5ad4869c4` |
| Source parent | `F:\llm-models\Qwen3.8-9B-abliterated-25-bf16` (lineage verified: conversion provenance, tokenizer identity, source manifest) |
| Device + card memory | CUDA device 0, RTX 5060 Ti, 17.1 GB |
| Instrument commits | rung-4 producers: `0177d8d` (T4a census + `grad_zero_steps`), plus this prereg and its judge |

The census producer and the generation probe did not exist when rung 4 was
preregistered; they were implemented after (#165 was docs-only) and are pinned
here by commit because 4b's thresholds read their output.

## 2. Workload (frozen, byte-identical to rung 4)

| Field | Value |
|---|---|
| Model | HotCore E16/k2/h2176 (9B-derived MoE) |
| Steps / horizon | **48 steps**, `max_steps` stop expected |
| tokens/step | 1536 (seq_len=64, batch_size=2) → 6144 tokens |
| Training corpus | `router-corpus-9b-pilot.txt`, SHA-256 `15d5f5f51a739ceee2712fe5b7b550982aba7272f633781f06d5a7ea64f47941` |
| Independent holdout | `router-holdout-independent-9b.txt`, SHA-256 `2e99668207319a1d2b702408bcc659a525fd226d027eebeaebea918e2ad21e97` |
| Census corpus | the independent holdout (same pin), `census_blocks = 2` |
| Generation probe | `router-gen-probe-prompts.txt`, SHA-256 `9db01036879af115635d2aa5452a3a43915e47c580ebff90b6e79a5db1ed1219`, 8 prompts, `max_new_tokens = 32`, `temperature = 0.7`, `top_p = 0.95` |
| Policy / dtype | `bf16-offload-transient`, `torch.bfloat16` |
| `paired_arms` | `true` |
| learning_rate / seed / probe_window | 0.05 / 1 / 2 |
| eval_batches | 2 |

**The generation probe is byte-identical to rung 4's.** That is what licenses
using rung 4's measured generation cost directly: same prompt set (same SHA),
same token cap, same sampling settings, same two arms, same resident process.

The horizon is **not** extended. Rung 4 failed T4a, and the program's own decision
rule forbids an automatic horizon increase when longer router training cannot be
shown to continue de-collapsing; a horizon change would need its own hypothesis and
its own prereg. 4b's purpose is to make the frozen experiment executable and to
finally measure T6–T10 and freeze the baseline.

## 3. Budgets (all enforced by preflight)

### 3.1 Measured basis (all measured; no estimate survives into these numbers)

From rung 4's completed training leg
(`worker-result.json` → `lifecycle.phases`) and the evaluator's own refusal
payload:

| Component | Measured | GPU-h |
|---|---|---|
| Model load (train) | 12.820008400012739 s | 0.003561113444447983 |
| 48 steps (`steady_state_steps`) | 139.08854140003677 s | 0.03863570594445466 |
| Step cost | **2.8976779458340993 s/step** (= 139.0885 / 48) | — |
| Checkpoint publication | 0.0661124000325799 s | 1.836455556460553e-05 |
| Closeout | 4.499917849898338e-06 s | ≈0 |
| Model load (eval) | 12.944961600005627 s | 0.0035958226666682296 |
| Generation, both arms (8 prompts × 32 tokens × 2) | **182.12685860006604 s** | 0.0505907940555739 |
| Generation rate | **0.355716520703254 s per generated token** (182.1269 / 512) | — |

Model loads required: **2** (training leg + the one resident paired-evaluation
leg). The baseline costs **no** additional load — it is completed from the pair.

### 3.2 Projection

| Component | Projection |
|---|---|
| Loads (2) | 12.820 + 12.945 = 25.765 s = 0.00715694 GPU-h |
| Steps (48) | 139.089 s = 0.03863571 GPU-h |
| Generations (both arms) | 182.127 s = 0.05059079 GPU-h |
| Publication + closeout | 0.066 s = 0.00001837 GPU-h |
| **Device total (unmargined)** | **346.98 s = 0.09638344 GPU-h** |
| **×1.5 margin → device ceiling** | **0.1446 GPU-h** → declared **0.1449** |

### 3.3 Enforced ceilings

`project_run_ceiling` requires the three sub-budgets to name exactly
`loads`, `steps`, `generations` **and to sum to `max_gpu_hours`** within 1e-9
(`src/chowder/backends/device_preflight.py:104-122`). The decomposition below sums
exactly, and each line covers its own measured projection:

| Budget | Ceiling | Projection | Ratio |
|---|---|---|---|
| Aggregate device | **0.1449 GPU-h** | 0.096383 | 1.503× |
| ├ `loads` | **0.0110 GPU-h** | 0.007157 | 1.537× |
| ├ `steps` | **0.0580 GPU-h** | 0.038636 | 1.501× |
| └ `generations` | **0.0759 GPU-h** | 0.050591 | **1.500×** |
| `max_load_seconds` | **20.0 s** per load | 12.945 | 1.545× |
| Step cost line | **3.5 s/step** | 2.8977 | 1.208× |
| Memory refusal | **14.5 GB** peak | 13.5486 (train, rung 4) | 1.070× |
| Wall envelope | **0.2898 GPU-h** | see §3.4 | — |

`0.0110 + 0.0580 + 0.0759 = 0.1449` exactly. The `steps` line is consistent with
the step-cost line: 48 × 3.5 s = 168.0 s = 0.046667 GPU-h ≤ 0.0580.

Experiment `estimated_gpu_hours` = **0.0964** (the unmargined measured
projection); goal `gpu_hour_budget` = **0.1449** (the ×1.5 ceiling). This mirrors
rung 4's 0.0478 / 0.072 convention.

### 3.4 Wall envelope — declared from measurement, not inherited

Rung 4 declared `M = 3.5`, producing the 0.252 envelope. That multiplier was a
rung-3c ledger **double-charge** artifact, corrected in PR #164; rung 4's own
measured training-leg multiplier is **M = 0.05632575761112902 / 0.04221518519444443
= 1.3343**. Rung 4b declares `M = 2.0`, i.e. **0.2898 GPU-h**, giving ~1.5× headroom
over the measured multiplier.

Wall charges remain **reported, never auto-failed** (§6): the device basis governs
refusal.

### 3.5 Declared generation cost line (diagnostic, not yet enforced)

Measured generation rate **0.35572 s per generated token** (CPU-resident experts
make every decoded token move expert weights — ~2.8 tokens/s, against ~10.6
tokens/s for the training forward/backward path). Declared here as evidence for a
future prereg. It is **not** a new gate in 4b; the `generations` sub-budget is what
refuses.

### 3.6 T4a's comparison basis — measured, not assumed

The bar in §4's T4a is not rung 4's `< 428` and not a self-comparison against
rung 4's own result. It is a **like-for-like predecessor**, measured on
2026-09-16 under a protocol frozen before the measurement ran:

| Router state | Steps | Dead / 512 | Basis |
|---|---|---|---|
| Untrained base | 0 | **480** | control; reproduced rung 4's value **exactly** |
| Rung 3c payload | 12 | **431** | re-censused payload, same corpus, same instrument |
| Rung 4 payload | 48 | 436 | rung 4's measurement, same corpus, same instrument |

Instrument, corpus, policy and settings were identical in all three rows: the
production `_census_blocks` / `_routing_census` / `_census_metrics` functions,
`router-holdout-independent-9b.txt` (`2e996682…21e97`), `seq_len` 64,
`census_blocks` 2 of 12, `bf16-offload-transient`, `cuda:0`. The control
reproducing 480 is what licenses treating the three numbers as one comparison.

Rung 3c's payload was verified before application: manifest kind
`router_healing_payload.v1`, `replacement`, `steps_completed` 12, 32 tensors,
tensor-file SHA `cceaedd7…`, base content `a6b20aaf…`. Cost: one model load
(13.652 s measured) plus four 64-token forward passes; no training, no optimizer,
no registry row, no run directory.

**Consequence**: T4a's bar becomes `< 431`. Rung 4 measured 436 and so failed its
own intent by 5 slots — the mis-specified corpus made a real negative result look
like a measurement artifact.

## 4. Thresholds

T1, T2, T3, T5, T6, T7, T8, T9 (threshold), T10 are **unchanged** from rung 4.
Three items are corrected, each marked ⚠ and justified in §5.

### T1 — validate before train
`chowder project-validate` passes; `chowder train` invoked only after.

### T2 — policy contract
Every worker result carries `load_policy_report.policy = "bf16-offload-transient"`,
`dtype = torch.bfloat16`, `placement_census.verified = true`, all expert params on
CPU, all gates on `cuda:0`.

### T3 — measured preflight
Memory, step-cost and model-load projections measured before step 1. Refusal if
load + steps + generation exceeds **0.1449**, any sub-budget line is exceeded,
3.5 s/step, or 14.5 GB.

### T4 — trainability ⚠ corrected
All **32 of 32** gates in scope, each observed with a finite non-zero gradient,
each recording at least one real optimizer update, `trainable = true`; frozen-tensor
digests `ok = true` with `changed == {}` and a `full`-strategy digest.

Per-step `update_steps` coverage is **reported, not gated** here; the phenomenon it
observes is gated once, in T4a(e). See §5, correction 4.

### T4a — routing-collapse and saturation ⚠ corrected

| Metric | Threshold |
|---|---|
| Census declared | `census_basis`, `census_corpus_sha256`, `census_blocks_used`, `census_blocks_available`, `expert_slots` all present; `census_corpus_sha256` = the independent-holdout pin. Missing or unlabelled census = **UNKNOWN** |
| `dead_experts_after` (a) | **< 431** — strict progress over rung 3c's trained router, re-censused on this corpus with this instrument (§3.6). This is rung 4's original "strict progress from rung 3c" intent, restored on a valid basis |
| `dead_experts_after` (b) | **< `dead_experts_before`** — strict within-run de-collapse, corpus-independent |
| `layers_with_grad_zero` (d) | **≤ 2 of 32** (unchanged from rung 4) |
| Gate update coverage (e) | **≤ 2 of 32** gates with `len(update_steps) < 48` |
| Any gate with `grad_zero_steps == 48` (f) | Must be named with topology-vs-optimizer evidence; **missing evidence = UNKNOWN**, never PASS |

**Reported, not gated**: rung 3c's cross-corpus 428; the measured triple
untrained 480 / rung 3c 431 / rung 4 436; the per-layer dead-count correlation
between the two trained routers (measured 0.888); per-layer dead-expert tables
before and after; the full per-step `update_steps` vector.

**Expected outcome, declared before the run.** The re-census refutes the horizon
hypothesis, so a 48-step re-execution cannot be expected to beat 431. **T4a is
therefore expected to FAIL**, and that is a legitimate result rather than a
lost run: T4a is not what 4b exists to deliver. 4b's deliverables are the things
that have never been produced for *any* router run — a completed
independent-holdout baseline, a measured T9, a measured T10, and evidence on
whether the census itself reproduces. Those thresholds are judged independently
by the same judge, so a T4a FAIL does not suppress them. Expected: T4a FAIL,
T9 PASS or FAIL on its own merits, T10 measured.

### T5 — horizon
48/48 steps, `stop_reason = max_steps`, `tokens_consumed = 6144`.

### T6 — paired gate contract
Exactly one evaluation worker (the resident pair); `arm = "paired"`;
`pair_error` absent; base scored on the holdout **before** payload application;
`application_control.applied_parameters` non-empty; exactly one measured
`model_load` phase; both `baseline_generation` and `candidate_generation`
measured; `holdout_loss_delta` consistent with the two arms; baseline row
`passed` with `baseline_source = "paired-candidate-evaluation"`.

### T7 — accounting
Total attributable device GPU-hours ≤ **0.1449**; the `loads` and `steps` phases
each ≤ their sub-budget; generation + publication + closeout ≤ 0.0759; zero
execution incidents; no result-carrying non-terminal experiment row; wall charges
within 0.2898 or the exceedance recorded.

### T8 — identity chain
Full-mode `local-content` base identity matches the pinned manifest in **both**
workers; train and eval re-derive the *same* base content hash; payload bound to
the re-derived base content; both corpora and the probe prompt set hash to their
pins.

### T9 — independent holdout quality ⚠ provenance corrected, threshold unchanged

**Threshold (unchanged)**: `candidate_holdout_loss < baseline_holdout_loss`.
Equality FAILS. Regression FAILS. No margin is required — the threshold is the
direction, not a delta.

**Baseline provenance (restated so it is satisfiable)**:

1. The baseline is the **resident pair's base arm**, scored on the frozen
   independent holdout inside the candidate's own evaluation, **before any payload
   is applied** to the model. The evaluator measures `base_loss` and the base
   fingerprints, and only then loads and applies the payload
   (`router_healing_eval_worker.py:503-560`).
2. The base artifact's identity must be re-derived **after training** and match the
   pin. This is the condition that makes a post-training base measurement equal to
   a pre-training one, and it is already carried: the evaluator derives
   `base_identity` itself, and the evaluator runs after training has finished.
   **T8's eval-side identity check is therefore the proof of this clause**; T9 does
   not duplicate it.
3. Training writes only to the run directory and the payload directory. It does not
   modify the base artifact; clause 2 verifies exactly that.

**Rejected alternative** (recorded so the choice is visible): spawning a separate
pre-training baseline process. It costs one additional full model load, undoing the
deliberate amortization the paired design exists to provide, and it would make T6's
single-load contract false. Verified equivalence by clause 2 is strictly cheaper and
strictly stronger evidence than a second load.

### T10 — generation sanity (unchanged)
Both arms are probed inside the same resident pair with identical settings.

| Metric | Gate |
|---|---|
| `termination_rate` | ≥ 0.90 |
| `max_token_cap_rate` | < 0.10 |
| `distinct_trigram_ratio` | > 0.70 |
| `looping_prompts` | = 0 (a prompt with 3+ consecutive identical trigrams fails that prompt) |
| `compression_ratio`, `task_score` | reported, not gated |

Perplexity/loss is **never** substituted for generation sanity: the static-prune
experiment already showed loss can look tolerable while generation collapses.

## 5. Corrections register (all dated; thresholds unchanged except as marked)

| # | Item | Rung 4 | Rung 4b | Class |
|---|---|---|---|---|
| 1 | `generations` sub-budget | 0.0032 (unmeasured) | **0.0759** (1.500× the measured 182.1269 s) | budget derived from measurement |
| 1 | Aggregate ceiling | 0.072 | **0.1449** (×1.503 measured) | budget derived from measurement |
| 2 | T9 baseline provenance | "before the candidate training run" | resident pair's base arm, before payload application, with post-training base-identity re-derivation | unsatisfiable requirement corrected; **threshold unchanged** |
| 3 | T4a dead-expert bar | `< 428` (rung-3c corpus, no declared basis) | **`< 431`** (rung 3c's payload re-censused on this corpus with this instrument, §3.6) **and** strict within-run de-collapse | rung 4's original intent restored on a valid basis; the bar is *stricter* than rung 4's `≤ 436` |
| 4 | T4 per-step update requirement | `update_steps == every step 0..47` for all 32 gates | trainability only; per-step coverage **reported**; gated once in T4a(e) | de-duplication of two overlapping thresholds |
| 5 | Wall multiplier M | 3.5 (double-charge artifact) | 2.0 (measured training-leg M = 1.3343) | envelope derived from measurement |

**What is deliberately unchanged**: the workload (48 steps, 1536 tokens/step, seed
1, lr 0.05, `bf16-offload-transient`, `paired_arms`), the artifact and every pin,
the probe prompt set and its sampling settings, the 3.5 s/step line, the 20.0 s
load line, the 14.5 GB memory refusal, T1/T2/T3/T5/T6/T7/T8, and T9's and T10's
numeric thresholds.

**On corrections 3 and 4 being "relaxations"**: both replace a bar that measurement
showed was not measuring what it claimed. Correction 3 replaces a cross-corpus
comparison with a same-corpus non-regression bar *plus* a corpus-independent
de-collapse requirement — rung 4 passed the de-collapse clause (436 < 480) and
failed the mis-specified one. Correction 4 removes a threshold (T4's per-step
coverage) that was strictly stronger than T4a's saturation clause and therefore
could only ever fail the same evidence twice; the strictest tractable form of that
claim remains gated, in T4a(d) and T4a(e). Neither changes what would count as
routing collapse.

## 6. Interpretation rules

- Judge exit 0 only when every threshold is PASS; any FAIL or UNKNOWN refuses
  certification.
- A missing or unreadable artifact is UNKNOWN, never an assumed pass.
- Wall-charge exceedance of the envelope is recorded in the result doc by a human,
  never auto-failed (the device basis governs refusal).
- Every attempt is preserved. A refused attempt is evidence. Run directories are
  never reused; a legitimate retry uses explicit attempt naming.
- No threshold is renegotiated after seeing the data.
- If the instrument itself is defective, the original refusal is preserved
  verbatim, the defect is proved to be an instrument defect and not an experiment
  failure, a regression test is added, the instrument is fixed narrowly, and
  judgment is rerun on the **same immutable evidence**.

## 7. Judge

`docs/quals/judge_rung4b_2026-09-16.py`, committed on the same change as this
prereg and dry-run against rung-4's evidence before the run. It must refuse rung 4
(no evaluation worker; `dead_experts_after` 436 is not ≤ 436-with-progress; budget
0.072 < 0.1449), proving it is not merely echoing rung 4's judge.
