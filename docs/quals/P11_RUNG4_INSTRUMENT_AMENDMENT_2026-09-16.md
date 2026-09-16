# P11 rung 4 — dated instrument amendment (2026-09-16)

**Dated 2026-09-16, before any rung-4 compute.**

## Why this amendment exists

The rung-4 preregistration
([`P11_RUNG4_PREREG_2026-09-15.md`](P11_RUNG4_PREREG_2026-09-15.md), PR #165)
froze three new thresholds — T4a, T9 and T10 — but PR #165 was **docs-only**:
it added the prereg, the judge, the harness and the holdout corpus, and **no
implementation**. Two of those thresholds read worker artifacts that no code
produces, so rung 4 could not be certified if it ran.

This amendment changes **no threshold, no budget, no workload and no
hypothesis**. It adds the missing *producers*, with tests, so the frozen
thresholds can actually be measured.

| T | Frozen requirement | Producer before | Producer after |
|---|---|---|---|
| **T4a** | `dead_experts_after < 428`; per-layer saturation evidence | **none** — the training worker emitted no `metrics` key and its `utilization` was hardcoded `{"status": "not_reported"}` | training-leg routing census |
| **T10** | `generation_sanity` from real sampling | **none** — the evaluator performed no generation at all | generation-sanity probe |
| T9 | `candidate_holdout_loss < baseline_holdout_loss` | field fallback existed | unchanged |
| T1–T8 | unchanged | present | unchanged |

The evidence for the gap, including the judge's verbatim refusal of rung 3c, is
preserved at
`Chowder-Protected/runs/2026-09-16-router-healing-rung4/` and must not be
deleted.

## What was added

### 1. Training-leg routing census (T4a)

`src/chowder/backends/router_healing_worker.py` now runs a top-1 routing census
with forward hooks on every `...mlp.gate` module — the same measurement the
evaluator already used — **before** the first step and **after** the last, and
emits a `metrics` block on the training result:

```
metrics.census_basis              "declared-census-corpus" | "training-corpus"
metrics.census_corpus_sha256      the hashed basis
metrics.census_blocks_used / _available
metrics.expert_slots              512 for this artifact (32 x 16)
metrics.dead_experts_after        <- T4a's number
metrics.dead_experts_before       the same basis, before training
metrics.dead_experts_per_layer / _per_layer_before
```

Three properties make the claim honest rather than merely present:

- **The measurement basis is named, always.** A census over the *training*
  corpus flatters the router, because those are the exact tokens it was
  optimised on. The basis is therefore declared, hash-checked, and recorded; the
  fallback to the training corpus is labelled `"training-corpus"` so it can
  never be read as held out.
- **The census corpus is hash-pinned.** A declared census corpus is verified
  against its declared SHA-256 before any measurement, exactly like the
  training corpus and the holdout.
- **The census is a measurement, not a roll of the dice.** It runs in eval mode
  (dropout in train mode would make the same trained router report different
  expert usage run to run) and restores the model's prior mode afterwards.

### 2. Per-component saturation evidence (T4a)

`src/chowder/trainability.py` adds `grad_zero_steps` and `grad_nonzero_steps`
to each component record: the number of observed steps whose gradient was
exactly zero. The pre-existing `gradient_states` *set* cannot express the
difference between "zero on one step" and "zero on every step", and the judge
names a saturated layer precisely by `grad_zero_steps == <horizon>`.

### 3. Generation-sanity probe (T10)

`src/chowder/backends/router_healing_eval_worker.py` adds a real sampling probe
that measures the generated text — never perplexity:

```
generation_sanity.termination_rate        gated (>= 0.90)
generation_sanity.max_token_cap_rate      gated (< 0.10)
generation_sanity.distinct_trigram_ratio  gated (> 0.70)
generation_sanity.looping_prompts         gated (== 0)
generation_sanity.compression_ratio       reported, not gated
generation_sanity.task_score              honestly UNMEASURED (no answer keys)
generation_sanity.arm                     "candidate" (gated) | "base"
generation_sanity_base                    the base arm, for comparison
```

Design commitments:

- **Opt-in.** The probe runs only when a prompt set is declared. A run that
  declares none records `UNMEASURED` with a reason — never a silent pass, and no
  existing evaluation pays a generation cost it never preregistered.
- **The prompt set is pinned.** Prompts are read and SHA-256-checked before the
  first measurement, so a drifted prompt set fails before any score is taken. An
  *empty* prompt set is refused rather than reporting healthy ratios over no
  evidence.
- **The decoding settings are validated and recorded.** `max_new_tokens`,
  `temperature` and `top_p` are required, range-checked, and echoed into the
  result so a reader can see exactly what produced the numbers.
- **Both arms, identical settings.** The base arm's probe is recorded beside the
  candidate's, so a reader can see whether the payload changed generation
  behaviour or only the loss. T10 gates on the candidate.

## What this amendment deliberately does NOT change

- **No threshold.** T4a's `< 428`, `<= 2` grad-zero layers, and T10's 0.90 /
  0.10 / 0.70 / 0 targets are untouched, as are T1–T9, every sub-budget, the
  0.072 aggregate ceiling, the 14.5 GB memory line and the 3.5 s/step line.
- **No workload.** 48 steps, 1536 tokens/step, seed 1, lr 0.05,
  `bf16-offload-transient`, the same artifact and corpora.
- **No dependency change.**
- **No judge change.** The judge is used exactly as frozen. (Its mapping of a
  *missing* measurement to `FAIL` rather than `UNKNOWN` is a separate, narrow
  observation, recorded in the rung-4 result document and deliberately left
  alone here.)

## Why the census basis matters for T4a's arithmetic

T4a's threshold is "strictly fewer than rung 3c's 428", and rung-3c's 428 was
measured on a **held-out** corpus. Comparing a training-corpus census against a
held-out one would bias toward a false PASS, because routing is less collapsed
on data the router was trained on. The rung-4 project therefore declares the
**independent holdout** (`router-holdout-independent-9b.txt`,
`2e996682…21e97`) as its census corpus, with `census_blocks: 2` to match the
evaluator's `eval_batches: 2`, so both legs and both rungs are held-out basis.

One residual caveat is recorded rather than hidden: rung 3c's 428 was measured
on `router-holdout-9b-pilot.txt` while rung 4's census uses
`router-holdout-independent-9b.txt`, so the two numbers are the same *kind* of
measurement on different held-out corpora. That is stated in the rung-4 result
document; it is not a reason to move the threshold.

## Tests and mutation evidence

New: `tests/test_router_census.py` (12) and
`tests/test_router_generation_sanity.py` (15), plus a saturation case in
`tests/test_trainability.py`.

Every new guard was mutation-checked, and each mutation failed exactly its own
test:

| Mutation | Caught by |
|---|---|
| `grad_zero_steps` always 0 | the two trainability count tests |
| `dead_experts_after` becomes "all slots" | both census-metric tests |
| census basis never labelled | the declared-corpus test |
| looping detector never fires | the three looping/combined tests |
| cap-hit detector never fires | the combined probe test |

Full suite: **2087 passed, 77 skipped** — no regressions. Ruff clean on `src`
and `tests`.

## Order of operations

1. This amendment and the producer change are **reviewed and landed** on `main`
   first. The producers are a change to a qualified measurement path; they do
   not get to ride along inside a frozen run.
2. The rung-4 project is built from the frozen numbers (48 steps, 0.072 GPU-h,
   census corpus = independent holdout, generation probe declared).
3. `chowder project-validate`, then the run, then the frozen
   `judge_rung4_2026-09-15.py`.

Nothing in step 2 or 3 may alter a threshold. If the instrument still cannot
produce a number for some threshold, the run is refused again rather than the
threshold reinterpreted.
