# P11 rung 4c — evaluation-protocol qualification (12-step stack replay)

Frozen **2026-09-16, before the run**. Committed to `niko4244/Chowder` and pushed
**before** `chowder train` is invoked.

**This rung makes no routing-progress claim.** It replays rung 3c's exact 12-step
training workload under rung 4's evaluation protocol to answer two questions that
have never been answered for any router run:

1. **Does the evaluation protocol complete?** No router run has ever produced a
   completed independent-holdout baseline, a measured T9, or a measured T10.
2. **Does the training stack reproduce rung 3c bit-for-bit?** A seed-1 replay of
   the same 12-step workload, the same corpus and the same policy should produce
   the same router tensors. Nothing in the program has ever tested that.

## 0. Why this rung, and why not 48 steps

Rung 4 was executed on 2026-09-16 and REFUSED: the training leg completed 48/48,
the evaluation leg refused at the generation-cost gate (a 15.8× sub-budget
shortfall), which took T6–T10 with it. Rung 4b's preregistration
(`docs/quals/P11_RUNG4B_PREREG_2026-09-16.md`) fixed the budget and the baseline
provenance, but it kept rung 4's 48-step horizon.

### 0.1 Rung 4's horizon hypothesis is refuted — measured, with a control

A same-corpus re-census of rung 3c's trained payload was run on 2026-09-16 under a
protocol frozen before it ran
(`RUNG3C_RECENSUS_PROTOCOL_2026-09-16.md` / `RUNG3C_RECENSUS_RESULT_2026-09-16.md`
in the 4b evidence root). Same corpus, same instrument, same load policy:

| Router state | Steps | Dead / 512 |
|---|---|---|
| Untrained base | 0 | **480** (control — reproduced rung 4's value exactly) |
| Rung 3c payload | **12** | **431** |
| Rung 4 payload | **48** | **436** |

**48 steps is not better than 12; it is 5 slots worse**, and the per-layer
dead-count correlation between the two trained routers is 0.888 with layers 4, 22
and 31 at 15 of 16 dead in both. Rung 4's own hypothesis — *"12 steps is too short
a horizon for the router to spread load across experts"* — is refuted: the
de-collapse saturates inside the first 12 steps.

So a 48-step re-execution cannot improve routing, and would spend the full
0.1449 GPU-h ceiling for the same evaluation-protocol deliverables that a 12-step
replay produces for **0.1014**. 4c is the 12-step replay.

Rung 4b remains on file as the record of the budget and provenance corrections; it
is **superseded by 4c for execution**.

## 1. Identity

| Field | Value |
|---|---|
| Run name / ladder rung | P11 rung 4c — evaluation-protocol qualification |
| Evidence root | `Chowder-Protected/runs/2026-09-16-router-healing-rung4c-protocol-qual` |
| Base artifact | `F:\llm-models\Qwen3.8-9B-HotCore-CW-E16-k2-h2176` |
| Content manifest pin | SHA-256 `77520edadb9a94f4ed70636328c4bbbaafa49c75e3b47c51418851c5ad4869c4` |
| Device | CUDA device 0, RTX 5060 Ti, 17.1 GB |
| Instrument commits | rung-4 producers `0177d8d`; this prereg and its judge |

## 2. Workload (frozen)

**12 steps** — rung 3c's horizon. Everything else matches rung 4's evaluation
protocol.

| Field | Value |
|---|---|
| Steps / horizon | **12**, `max_steps` stop expected |
| tokens/step | 1536 (seq_len=64, batch_size=2) → **1536 tokens** |
| Training corpus | `router-corpus-9b-pilot.txt`, SHA-256 `15d5f5f51a739ceee2712fe5b7b550982aba7272f633781f06d5a7ea64f47941` |
| Independent holdout | `router-holdout-independent-9b.txt`, SHA-256 `2e99668207319a1d2b702408bcc659a525fd226d027eebeaebea918e2ad21e97` |
| Census corpus / blocks | the independent holdout (same pin) / **2** of 12 |
| Generation probe | `router-gen-probe-prompts.txt`, SHA-256 `9db01036879af115635d2aa5452a3a43915e47c580ebff90b6e79a5db1ed1219`, 8 prompts, `max_new_tokens` 32, `temperature` 0.7, `top_p` 0.95 |
| Policy / dtype | `bf16-offload-transient`, `torch.bfloat16` |
| `paired_arms` | `true` |
| learning_rate / seed / probe_window | 0.05 / 1 / 2 |
| eval_batches | 2 |

### 2.1 The replay claim rides on exact training-input equality

The reproduction prediction (T11) is valid only if the *training-relevant* spec
equals rung 3c's. Rung 3c's `run-spec.json` recorded:

```
max_steps 12  learning_rate 0.05  seq_len 64  batch_size 2  seed 1
probe_window 2  max_tokens 1536  scheduler "constant"  warmup_steps 0
load_policy "bf16-offload-transient"  device "cuda"
detailed_timing false  checkpoint_every 0
corpus_sha256 15d5f5f5…  base_content_sha256 a6b20aaf…
```

4c declares all fourteen identically. The additions below are declared **not** to
affect the trained tensors, and that is checkable rather than asserted:

- `census_corpus_path` / `census_corpus_sha256` / `census_blocks` — the census is
  read-only: it registers forward hooks, runs in eval mode under `no_grad`, and
  runs before the step loop and after it. It performs no optimizer step.
- `paired_arms`, `eval_batches`, `generation_probe` — evaluation-side only; they
  reach the evaluator's spec, not the trainer's.
- `max_gpu_hours` / `sub_budget_gpu_hours` / `max_load_seconds` — preflight
  refusal thresholds, evaluated before step 1.

If T11 fails, that reasoning is what makes the failure *interpretable*: a
difference in tensors cannot be attributed to the census or the budget, because
neither can touch them.

## 3. Budgets (all enforced by preflight)

### 3.1 Measured basis

Every number is measured — from rung 4's completed training leg
(`worker-result.json` → `lifecycle.phases`) and the evaluator's own refusal
payload pointed at rung 4's generation probe, which is byte-identical to 4c's:

| Component | Measured | GPU-h |
|---|---|---|
| Model load (train) | 12.820008400012739 s | 0.0035611 |
| Step cost | **2.8976779458340993 s/step** | — |
| **12 steps** | **34.7721 s** | 0.0096589 |
| Checkpoint publication | 0.0661124000325799 s | 0.0000184 |
| Model load (eval) | 12.944961600005627 s | 0.0035958 |
| Generation, both arms (8 prompts × 32 tokens × 2) | **182.12685860006604 s** | 0.0505908 |

Model loads required: **2** (training leg + the one resident paired evaluation
leg). The baseline costs no additional load — it is completed from the pair.

### 3.2 Projection and enforced ceilings

| Budget | Ceiling | Projection | Ratio |
|---|---|---|---|
| Aggregate device | **0.1014 GPU-h** | 0.067425 | 1.504× |
| ├ `loads` | **0.0110** | 0.007157 | 1.537× |
| ├ `steps` | **0.0145** | 0.009659 | 1.501× |
| └ `generations` | **0.0759** | 0.050591 | 1.500× |
| `max_load_seconds` | **20.0 s** per load | 12.945 | 1.545× |
| Step cost line | **3.5 s/step** | 2.8977 | 1.208× |
| Memory refusal | **14.5 GB** peak | 13.5486 (train, rung 4) | 1.070× |
| Wall envelope | **0.2028 GPU-h** (device × M=2.0) | measured train-leg M 1.3343 | 1.5× |

`0.0110 + 0.0145 + 0.0759 = 0.1014` exactly (verified to 1e-17, within the 1e-9
that `project_run_ceiling` requires). The step line is consistent with the
step-cost line: 12 × 3.5 s = 42.0 s = 0.011667 ≤ 0.0145. Experiment
`estimated_gpu_hours` = **0.0674**; goal `gpu_hour_budget` = **0.1014**.

Note the shape of this budget: **the generation probe is 75% of the whole run.**
Decoding costs 0.35572 s per generated token under `bf16-offload-transient`
(CPU-resident experts move on every token), so the evaluation protocol — not the
training — is what the budget is mostly buying.

## 4. Thresholds

### T1 — validate before train
`chowder project-validate` passes; `chowder train` invoked only after.

### T2 — policy contract
Every worker result carries `load_policy_report.policy = "bf16-offload-transient"`,
`dtype = torch.bfloat16`, `placement_census.verified = true`, expert params on
CPU, gates on `cuda:0`.

### T3 — measured preflight
Memory, step-cost and model-load projections measured before step 1. Refusal if
load + steps + generation exceeds **0.1014**, any sub-budget line is exceeded,
3.5 s/step, or 14.5 GB.

### T4 — trainability
All 32 of 32 gates observed with a finite non-zero gradient, each recording at
least one real optimizer update, `trainable = true`; frozen digests `ok = true`,
`changed == {}`, full-strategy digest present. Per-step coverage is reported.

### T4a — census reproduction and saturation

This is a **reproduction** threshold, not a progress threshold. `< 431` remains
the progress bar for any run that claims routing improvement (it is gated in
rung 4b); it is **not** gated here, because gating a 12-step replay on beating the
12-step predecessor would gate a bar that existing measurement says is
unreachable at this horizon — and would tell us nothing about the evaluation
protocol, which is what 4c is for.

| Metric | Threshold |
|---|---|
| Census declared | `census_basis`, `census_corpus_sha256`, `census_blocks_used`, `census_blocks_available`, `expert_slots` present; corpus sha = the independent-holdout pin |
| **Control** (a) | `dead_experts_before` == **480** exactly — otherwise the instrument or corpus drifted and nothing here is comparable |
| **Reproduction** (b) | `dead_experts_after` == **431** exactly — the value rung 3c's payload produced on this corpus with this instrument |
| Strict de-collapse (c) | `dead_experts_after` < `dead_experts_before` |
| Full-horizon saturation (d) | Any gate with `grad_zero_steps == 12` must be named with topology-vs-optimizer evidence; **missing evidence = UNKNOWN**, never PASS |

Reported, not gated at 12 steps: `layers_with_grad_zero` and
incomplete-update-coverage counts, against rung 3c's recorded 3/32 and rung 4's
measured 4/32. The `≤ 2 of 32` bars in rungs 4 and 4b are 48-step improvement
bars; a 12-step replay does not claim them.

### T5 — horizon
**12/12** steps, `stop_reason = max_steps`, `tokens_consumed = 1536`.

### T6 — paired gate contract
Exactly one evaluation worker (the resident pair); `arm = "paired"`;
`pair_error` absent; base scored on the holdout before payload application;
`application_control.applied_parameters` non-empty; one measured `model_load`;
both `baseline_generation` and `candidate_generation` measured;
`holdout_loss_delta` consistent; baseline row `passed` with
`baseline_source = "paired-candidate-evaluation"`.

### T7 — accounting
Total attributable device GPU-hours ≤ **0.1014**; `loads` and `steps` phases each
within their sub-budget; generation + publication + closeout ≤ 0.0759; zero
execution incidents; no result-carrying non-terminal experiment row; wall charges
within 0.2028 or the exceedance recorded.

### T8 — identity chain
Full-mode `local-content` base identity matches the pinned manifest in both
workers; both re-derive the same base content hash; payload bound to the
re-derived base content; both corpora and the probe prompt set hash to their pins.

### T9 — independent holdout quality
**First measurement for any router run.** Threshold:
`candidate_holdout_loss < baseline_holdout_loss`. Equality FAILS; regression
FAILS; no margin is required. The baseline is the resident pair's base arm, scored
on the frozen independent holdout before payload application, with the evaluator's
own post-training base identity as the proof that the base artifact was untouched.

### T10 — generation sanity
Both arms probed in the same resident pair, identical settings.

| Metric | Gate |
|---|---|
| `termination_rate` | ≥ 0.90 |
| `max_token_cap_rate` | < 0.10 |
| `distinct_trigram_ratio` | > 0.70 |
| `looping_prompts` | = 0 |

`compression_ratio` and `task_score` are reported, not gated. Loss is never
substituted for generation sanity.

### T11 — stack reproduction (NEW)
`payload.tensor_file_sha256` == rung 3c's published value
`cceaedd792ee8f0eae7f2f6bc46a2208f98ff5654b5b0cd665979550a4a83792`, and
`payload.steps_completed` == 12.

**The tensor file is the test; the manifest hash is not.** The manifest embeds
`spec_digest` and the declared budget, so 4c's manifest hash *must* differ from
rung 3c's (`a43b4d96…`). Only the trained tensors can be identical, so only they
are gated. The manifest hash is reported.

A T11 FAIL is a real finding, not a failed run: it would establish that the
training stack is **not** bit-reproducible at this horizon, which would mean every
cross-run comparison in this program has an unquantified reproducibility margin.
The result doc must then report both T11 and T4a(b): identical tensors *and* a
reproduced census is the strong outcome; a census that reproduces while tensors
differ is the weaker, still-informative outcome.

## 5. Declared predictions (recorded before the run)

| Prediction | Value |
|---|---|
| `dead_experts_before` (control) | **480**, exactly |
| `dead_experts_after` | **431**, exactly |
| Payload tensor file SHA | **`cceaedd7…a83792`**, bit-identical to rung 3c's |
| T9 direction | **unknown** — 12 steps of router-only training is not expected to move holdout loss much; either direction is reported honestly |
| T10 | **unknown** — the first generation-sanity measurement ever taken on this artifact. The static-prune finding warns that a tolerable loss can coexist with collapsed generation, so T10 is the measurement most likely to surprise |

## 6. What this rung cannot establish

- It makes **no routing-progress claim**. A reproduced 431 is not an improvement
  and must not be cited as one.
- It does not test the 48-step horizon; that question is already answered
  negatively by §0.1.
- T11 proves reproduction only at 12 steps on this hardware with this policy. It
  says nothing about 48 steps, and nothing about cross-hardware reproducibility.
- A passing T9 does not make the HotCore artifact a useful model; it means the
  candidate arm scored lower than the base arm on this holdout.

## 7. Interpretation rules

- Judge exit 0 only when every threshold is PASS; any FAIL or UNKNOWN refuses
  certification.
- A missing or unreadable artifact is UNKNOWN, never an assumed pass.
- Wall-charge exceedance is recorded by a human, never auto-failed.
- Every attempt is preserved; run directories are never reused.
- No threshold is renegotiated after seeing the data.
- If the instrument is defective: preserve the original refusal verbatim, prove it
  is an instrument defect rather than an experiment failure, add a regression test,
  fix narrowly, and rerun judgment on the same immutable evidence.

## 8. Judge

`docs/quals/judge_rung4c_2026-09-16.py`, committed with this prereg and dry-run
against rung-4's evidence before the run. It must refuse rung 4 (no evaluation
worker; 48 steps is not 12; the payload tensor hash is rung 4's, not rung 3c's),
proving it is not echoing the rung-4b judge.
