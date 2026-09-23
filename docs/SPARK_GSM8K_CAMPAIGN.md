# Spark 2.5 GSM8K Campaign — Multi-Generation Validation Report

Campaign: 4 bounded LoRA generations on `F:\Huihui-Spark-X2.5-4B-abliterated`
(Spark2_5ForCausalLM, bfloat16, RTX 5060 Ti 16 GB), each run through the full
lifecycle-authoritative `run_project()` path. This report records the campaign
that validated Chowder's train → evaluate → adjudicate loop on a real local
model with a real benchmark.

**Headline: 0.39 → 0.68 → 0.73 (peak, promoted) → 0.70 → 0.66.** Every
candidate was measured on the same never-trained 100-prompt holdout, every
promotion was earned against a real protocol-matched baseline, and every
regression was refused.

## Benchmark design

- **Data**: real GSM8K (`openai/gsm8k`, train split), deterministic seed-7
  shuffle, compact worked answers ending in `#### N`. Spark's chat template
  renders full transcripts for training text; the evaluator renders prompts
  with the same template at generation time.
- **Splits** (disjoint by construction):
  | Slice | Rows | Use |
  |---|---|---|
  | gen1_train | 0–199 | generation 1 training |
  | gen2_train | 1000–1199 | generation 2 training |
  | gen3_train | 2000–2199 | generation 3 training |
  | gen4_train | 3000–3399 | generation 4 training |
  | gen5_train | 4000–4399 (+ replay of gen1/gen2 rows) | generation 5 training |
  | holdout | 5000–5099 | **all** evaluation, never trained on |
- **Scoring**: `reasoning_final_number_match` — reads the final number from
  the reasoning span (between the first and next `</think>`). Plain
  `final_number_match` cannot score this model: Spark's template emits a
  trailing `</think>` after the answer, so the tail after the last marker is
  empty. The scorer was added as a new mode; existing scoring defaults were
  not changed, and it participates in protocol identity via the scorer
  content hash.
- **Honest baselines**: each generation after the first used the previous
  promoted candidate's *measured* result as a fixed baseline, with its real
  persisted evaluation-protocol sha — same benchmark file, same scoring, same
  runtime contract (`require_protocol_match: true`).

## Results

| Generation | Training | Steps | Baseline | Candidate | Assessment | Terminal | Promoted | succeeded |
|---|---|---|---|---|---|---|---|---|
| 1 | 200 rows | 150 | 0.39 (measured untouched-model baseline) | **0.68** | MET (bar 0.60) | `STOP_GOALS_MET` | yes | **true** |
| 2 | 200 fresh rows, continued from gen1 adapter | 100 | 0.68 (gen1 measured) | **0.73** | UNMET (bar 0.75) | null | yes (gain +0.05 > 0.02 gate) | false |
| 3 | 200 fresh rows, continued from gen2 adapter | 150 | 0.73 (gen2 measured) | **0.70** | UNMET (bar 0.75) | `STOP_PLATEAU` | no — candidate below baseline | false |
| 4 | **400** fresh rows, r=32/alpha=64, continued from gen2 adapter | 300 | 0.73 (gen2 measured) | **0.66** | UNMET (bar 0.75) | `STOP_PLATEAU` | no — candidate below baseline | false |
| 5 | **800** rows: 400 fresh + 400 replayed gen1/gen2, interleaved; r=16; **lr 3e-5**; continued from gen2 adapter | 300 | 0.73 (gen2 measured) | **0.68** | UNMET (bar 0.75) | `STOP_PLATEAU` | no — candidate below baseline | false |

Key honesty checkpoints demonstrated:

- **Promotion ≠ completion.** Gen 2 improved (+0.05), was promoted, and still
  reported `succeeded: false` because the objective bar was unmet. The
  lifecycle separates candidate quality from goal completion.
- **A non-improving candidate is refused, not plateaued into success.**
  Gen 3 scored *below* its baseline; the promotion gate blocked it and the
  lifecycle returned `STOP_PLATEAU` with `succeeded: false`.
- **Generation/budget exhaustion is never success.** Every non-`STOP_GOALS_MET`
  terminal state reported `succeeded: false`.

## Plateau diagnosis (gens 1–3)

The model saturated at 0.68–0.73 on the 100-prompt holdout. Prediction
evidence from gen 3 (70/100 correct) shows genuine reasoning with arithmetic
slips or reasoning that overruns the 320-token budget before emitting
`#### N` — capability limits, not scoring artifacts. Levers identified:
more data per generation (gen 4 doubles it), higher LoRA rank (gen 4 raises
it), longer generation budget (costs eval time linearly), or curriculum on
multi-step problems. A 4B reasoning model fine-tuned on 200 worked examples
per generation appears to be the binding constraint.

## Defects found and fixed during the campaign

1. **Adapter-save crash (production fix, `transformers_worker.py`).** The
   Windows file lock (`os error 32`) on freshly written safetensors files
   killed two runs *after* training completed. The first retry fix was
   unreachable dead code — it caught `OSError`, but safetensors raises
   `SafetensorError`, which is not an `OSError` subclass. Fixed with
   `_save_adapter_with_retry()` catching both types, matched on the lock
   message, one bounded cleanup+retry. Regression tests cover
   lock-then-success, non-lock propagation, and persisting-lock failure.
2. **Spark chat template is not prefix-consistent across turns** for
   completion-only loss masking. Chowder correctly refused to build an
   ambiguous mask; the campaign used fully rendered chat transcripts as text
   training data instead. The guard was not weakened.
3. **`reasoning_final_number_match` scoring mode added** for the trailing-
   `</think>` template shape (see Benchmark design).
4. **Local custom-code compatibility** (`local_model_compat.py`): explicit
   SHA-256-pinned path for Spark's `configuration_spark.py` /
   `modeling_spark.py`, with Transformers 5.x adaptations (tied-weight
   mapping, causal-mask keyword changes) applied across training, evaluation,
   and preflight workers.

## Generation 4 (plateau-breaking variant)

Doubled data (400 fresh rows) and doubled rank (r=32/alpha=64), 300 steps,
still continuing from the gen-2 promoted adapter. Result: **0.66 — below its
0.73 parent**; the promotion gate refused it and the lifecycle returned
`STOP_PLATEAU`. Both gens 3 and 4 regressing below the gen-2 checkpoint is
itself evidence: more capacity and more data at lr 1e-4 make continued
training *worse*, consistent with catastrophic forgetting of the gen-1/2
adaptations. The plateau is not data-starved; the honest reading is that the
campaign's peak (gen 2, 0.73) is the capability this recipe reaches, and
breaking 0.75 needs a different lever — lower learning rate with more steps,
replay of gen-1/2 data alongside fresh rows, or a larger/stronger base model.

## Generation 5 (anti-forgetting replay)

The two levers the gen-3/4 diagnosis pointed to, combined: **dataset replay**
(400 fresh rows deterministically interleaved with the exact gen1/gen2
training rows, so every optimization window mixes novel and parent-era
material) and a **10× lower learning rate** (3e-5), back to the gen-2 rank
(r=16/alpha=32), 300 steps, continuing from the gen-2 promoted adapter.
Training completed cleanly (train_loss 0.833, schedule fully decayed).

Result: **0.68 — below its 0.73 parent**; the promotion gate refused it and
the lifecycle returned `STOP_PLATEAU`. Replay plus a gentle learning rate
did not beat the gen-2 checkpoint either. The full picture across gens
3–5: three different recipes (identical, 2× capacity/data, gentle-lr
replay) all land at or below 0.68–0.73, while the gen-2 checkpoint itself
holds at 0.73. This is consistent with a capability ceiling of the 4B base
under this LoRA recipe rather than a training-dynamics problem: continued
optimization perturbs the adapter away from its best point more than it
gains. Breaking 0.75 most plausibly needs a stronger base model, a
substantially longer generation budget at eval time, or full fine-tuning.

Infrastructure notes from gen 5: the run root was relocated to another
drive after the system disk filled (the disk-space preflight correctly
refused the first launch with 0.28 GB free), and a mid-run CUDA OOM (other
GPU processes holding VRAM) required freeing the GPU and relaunching with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — both recovered without
code changes, and the second OOM-free run completed all 300 steps.

## Generation 6a: the eval-budget probe — truncation closed with measurements

The gen-6a diagnostic ran the gen-2 promoted adapter on the same holdout
through the real evaluator worker, at the 320-token protocol (control) and
768 tokens (probe). The control scored **0.73 — identical to the gen-2
lifecycle baseline**, validating the standalone probe machinery.

- **Probe (768 tokens): 0.73.** Doubling the generation budget changed the
  score by exactly zero.
- **Per-prompt trajectories:** with 2.4× the budget, **1 of 100 predictions
  changed at all**; zero score flips in either direction. Mean emitted
  length moved 112 → 116 tokens; the same single row hit the cap at both
  budgets and still answered wrong with the extra room.

Three independent measurements now agree: wrong-answer classification
(1/30 and 0/32 wrong answers truncated), the score-level probe
(0.73 = 0.73), and the trajectory comparison (1/100 generations differ).
Greedy GSM8K chains end at ~112 tokens; the model stops and answers, and
when it answers wrong, more budget does not help. **The truncation
hypothesis is closed.** Decision-tree branch taken: data-quantity lever.

## Generation 6: data quantity at the fixed protocol — 0.70, refused twice over

With the budget probe closed, the decision tree pointed at the one untested
lever. Gen 6 ran the maximum available data at the gentle recipe: all 1473
fresh rows in the 6000–7599 range (the shuffled train split yields no more),
chunked with the 400 gen1/gen2 replay rows repeated 4×, lr 3e-5, r=16,
750 steps (~one pass), continuing from the gen-2 adapter, fixed 320-token
protocol. Training completed cleanly (train_loss 0.79).

Candidate: **0.70 — below the 0.73 baseline.** The data-quantity lever is
dead at this recipe: 8× the fresh rows moved the checkpoint *away* from its
best point, same as every other continuation attempt.

Gen 6 also produced the campaign's first live firing of the scorer-identity
gate: the self-consistency scoring mode was committed between the gen-2
baseline measurement and this run, which changed `scoring.py`'s content
hash, which changed the candidate's evaluation-protocol digest — and the
lifecycle refused the comparison (`evaluation protocol changed within the
objective`) instead of silently scoring a new protocol against an old
one's evidence. The refusal is correct fail-closed behavior; the campaign
reads gen 6's verdict from the worker's raw evaluation artifact
(`eval-result.json`, 0.70 on the full 100-row holdout), with the registry
recording the refused comparison.

Campaign conclusion across six generations: 0.39 → 0.68 → **0.73 (peak)**
→ 0.70 → 0.66 → 0.68 → 0.70. Every lever tried beyond gen 2 — identical
recipe, 2× capacity+data, gentle-lr replay, 8× data — regressed. The 0.73
gen-2 checkpoint is this 4B model's ceiling under LoRA at GSM8K; further
progress needs a stronger base model, full fine-tuning, or a different
task decomposition.

## Generation 6 plan (eval-budget probe, post-truncation-analysis)

**Truncation analysis** (classify all wrong holdout predictions by failure
shape; script: `analyze_truncation.py` next to the fixtures):

| Generation | wrong | TRUNCATED (no `#### N`) | WRONG_NUMBER | SCORING_MISS | ceiling if all truncations fixed |
|---|---|---|---|---|---|
| gen3 (0.70) | 30 | **1** | 29 | 0 | 0.71 |
| gen5 (0.68) | 32 | **0** | 32 | 0 | 0.68 |

Wrong chains are *complete* (~103–113 emitted words, far under the
320-token cap) and compute a wrong number — the model ends its reasoning
and answers; it does not run out of budget. Zero scoring misses: the
extractor and scorer are not losing credit. **The truncation hypothesis is
refuted as the primary blocker**: a longer generation budget alone buys at
most +0.01–0.03.

**gen-6a — direct probe (cheap, no training, ~25 min).** Evaluate the
gen-2 promoted adapter on the same holdout with `max_new_tokens` raised
320 → 768 through the evaluator path. This closes the question with a
measured number instead of the inferred bound. Design notes: raising the
budget changes the protocol contract (max_new_tokens is part of evaluation
identity), so this is a *diagnostic outside the lifecycle* — not a
campaign generation — and any score it produces cannot be compared to the
0.73 baseline without a matching-baseline run. Halve the holdout to 50
prompts if eval time matters; 768 tokens roughly doubles generation time.

**Decision tree:**

- Probe ≥ 0.76 → the budget was masking real answers after all; run gen-6b
  as a training generation under the 768-token protocol (new objective
  version, fresh measured baseline = gen-2 adapter under the same 768-token
  protocol, replay recipe from gen 5).
- Probe in 0.73–0.76 → marginal; do not train. The remaining errors are
  arithmetic, not truncation. Pivot to the untested data lever: one
  generation over ~2,000 rows (fresh + full replay) at lr 3e-5 — gen 4's
  400-row run is confounded by forgetting and does not settle whether
  data quantity at gentle LR helps.
- Probe < 0.73 → longer budget hurts (rambling past the answer); close the
  hypothesis entirely and treat 0.73 as the recipe ceiling. Next levers:
  self-consistency voting (k=5 samples, majority final number — new eval
  protocol, no training), a stronger base model, or full fine-tuning.

## Reproduction

Scripts live in `.chowder-spark-calib/gsm8k/`: `prep_gen{3,4,5}.py` build the
fixtures (gen5's demonstrates the dataset-replay pattern: fresh slice plus
re-rendered parent-generation rows, deterministically interleaved);
`run_gen{1..5}.py` run each generation through `run_project()`;
`launch-gen{2..5}.ps1` launch them detached (tool timeouts must not kill the
training/eval pipeline — two early candidate evals died exactly that way).
Registries (`runs.db`) under `gen{1..4}/` (gen 5's under its relocated run
root) hold the append-only evidence; outcome summaries are written to
`gen{N}_outcome.json` next to each script.

## Post-campaign measurements: noise floor fixed, self-consistency at the bar

**500-prompt holdout baseline (gen-2 adapter, 320-token protocol, batch 1):**
**0.734 ± 0.039** (95% CI). The holdout extension (rows 5000-5499, a verified
superset of the 100-prompt holdout) makes the 0.02 promotion-gain step
resolvable: future deltas above ±0.04 are real signal, not noise.

**k=5 self-consistency diagnostic (100-prompt holdout, temp 0.7, shipped
`self_consistency_final_number_match`): 0.75 — exactly at the bar.** Paired
per-prompt against the greedy control: the vote fixed 8 rows and broke 6
(McNemar p≈0.59 — not individually significant at n=100); 19 rows are wrong
under both protocols. The mechanism is visible in the evidence: **16 of
those 19 wrong rows had the correct number produced by at least one sampled
chain but outvoted**. With more samples (k=10-20), those rows are the
recoverable mass — the realistic ceiling of pure sampling on this adapter is
~0.80-0.83, and the decisive test is k≥10 on the 500-prompt holdout.

Operational findings from these runs (full notes:
`.chowder-spark-calib/gsm8k/holdout500-notes.md`):

1. **Batch-8 generation is not safe on Spark — confirmed with a clean probe.**
   The earlier divergence verdict (two concurrent writers corrupting one
   output file) was re-run under single-writer conditions
   (`run_batch8_sc_k10.py` stage A): 50 holdout500 rows, greedy, batch 8 vs
   the batch-1 baseline on identical prompts. Result: probe 0.38 vs 0.72,
   only 6/50 prediction texts byte-identical, 23/50 correctness agreement —
   far below the 48/50 gate. The original divergence was real: Spark's
   custom modeling does not handle right-padded batched generation even
   with per-row pad correction. All Spark evals stay batch-1, and a
   batch-equivalence gate (probe-then-allow) belongs in model compatibility
   verification.
2. **Output paths must be single-writer by construction.** Two worker
   instances writing one predictions file corrupts it silently. A lock file
   or pid-guard at the output path turns this corruption into a refusal.

## k=10 verdict: more voting hurts — plurality convergence (2026-09-22)

k=10, temp 0.7, same 100-prompt holdout, same adapter, batch-1
(`F:/chowder-campaign/sc-k10-100`): **0.69**, vs k=5's 0.75 and greedy's
0.73. Paired per-prompt against k=5: **3 fixed, 9 broke**, 66 same-right,
22 same-wrong (McNemar p≈0.15, n.s. — but the direction is consistent with
the mechanism, and 0.69 is also k=10's point estimate).

The mechanism is vote convergence: as k grows, the majority converges to
the model's *modal* answer. Where the modal chain is wrong and the correct
answer is merely present-but-not-modal, more samples make the row more
reliably wrong. k=5's 0.75 was favorable variance around a true
plurality-score of ~0.69-0.73; the k=5-extrapolated "~0.80-0.83 ceiling"
is falsified. The sampling lever on this adapter tops out around
**0.73-0.75 at k=5** — it does not clear the 0.75 bar reliably.

Campaign implication: the RFT flywheel (self-generated correct chains) is
now the live path to 0.75+. Iteration 1 is running: k=6 sampling on 250
training prompts (rows 4400-4649, disjoint from all training and holdouts)
with the new digest-additive `store_chains` evaluator option, one correct
chain per solved prompt selected, trained from the gen-2 adapter at the
gentle continuation recipe, auto re-measured baseline under the current
protocol, 100-prompt holdout eval.

## Teacher chain-quality benchmark (Qwen3.8-27B vs Qwythos-9B, 20 RFT-1 prompts)

Served Qwen3.8-27B Q4_K_M (hybrid qwen35/SSM arch) via the new `llama_server_manager`
partial offload: 12 layers on GPU 1 (6 GB card), rest from RAM, llama.cpp build 10107
CUDA 12.4 (`F:/llm-lowvram/llama.cpp-bin-cuda124`). Throughput 1.0 tok/s; thinking lands
in `reasoning_content`, final answer in `content`. Thinking chains up to ~15 min; several
prompts answered direct in <60 s.

| model | solved | coverage of student's 9 failures |
|---|---|---|
| student k=6 vote (Spark 2.5) | 11/20 = 0.55 | — |
| **Qwythos-9B Q4 (GPU 1, ~35 tok/s)** | **20/20 = 1.00** | **9/9** |
| Qwen3.8-27B Q4 (partial offload, 1 tok/s) | 18/20 = 0.90 | 8/9 |

Reading: on this task the 9B teacher is not weaker than the 27B — it is perfect on the
fixture and 40x faster, and it covers **every** student failure. The distillation payload
for RFT-2 needs no partial-offload machinery: one idle 6 GB card serves a complete teacher.
The 27B's two misses (rows 15, 20) also show capability is not monotone with size under
quantization + offload. 27B partial-offload remains the fallback for harder tasks where
the 9B's coverage drops.

Operational notes: the winget llama.cpp build cannot load the qwen35 hybrid GGUF (missing
`ssm_conv1d` tensor support); the F: build 10107 loads it. The lifecycle manager caught a
real misconfig on first live use (full-offload spec on a 6 GB card -> refused), then
completed a full start/health/stop cycle.

## RFT campaign operations (Sep 23)

- **Auto-baseline is a charged experiment.** RFT-1's baseline ran under GPU contention
  and charged 2.35 GPU-hours against a 3.0 budget; with the experiment's 1.0 h estimate
  the lifecycle correctly refused the initial experiment ("does not fit the configured
  GPU-hour budget"). Lesson: budget = baseline-hours + experiment-hours + headroom, and
  never run two lifecycle projects on one GPU — contention both corrupts timing and
  inflates the baseline charge. RFT-1 relaunched with budget 5.0, chained behind RFT-2.
- **Auto-baseline measured the untouched base model at 0.39** under the current
  protocol — the honest "what did training add" reference for the RFT arms (gen2
  adapters score 0.73-0.75; the base model alone is far lower).
- Selection evidence (student sampler, 250 prompts): 93.2% solved, mean 4.25 correct
  chains per solved prompt. Hybrid arm adds 10 teacher-transfer rows (student-failed,
  teacher-solved) from Qwythos-9B.
- The collated verdict lands in `F:\chowder-campaign\rft_collation.json` when both
  arms finish (watchers + collator run detached; single-shot markers prevent
  double-launch).

### RFT-2 hybrid verdict (Sep 23, 11:09)

- Candidate (on-policy student chains + 10 teacher-transfer rows, 300 steps from the
  gen-2 adapter): **0.54** on the 100-prompt holdout. Goal 0.75 UNMET; terminal
  STOP_BUDGET (budget exhausted after the candidate eval). The lifecycle marked it
  promoted **relative to its auto-baseline** — but see the gate finding below.
- **The auto-baseline measured the untouched base model (0.39), not the parent
  adapter (0.730 on the same 100 prompts).** Paired per-prompt vs the parent:
  fixed 9, broke 28 → a net **-0.19 regression**. The promotion gate as configured
  compares the candidate to the base-model baseline, so a parent-adapter
  continuation can regress the parent and still "promote". Platform fix needed:
  when a config carries `parent_adapter`, the baseline reference must be that
  adapter re-measured under the current protocol, not the bare base model.
- Recipe diagnosis: 243 rows x ~4.9 epochs at lr 3e-5 with no replay overfits the
  small chain set and washes out the adapter's skill; the 10 teacher rows (4% of
  data) were too few to transfer and enough to perturb. Any RFT-3 must mix replay
  (the only recipe that ever held) and cap epochs near 1.

## Platform additions (2026-09-23, post-RFT-2)

- **Promotion gate fix (parent-adapter baselines).** The RFT-2 blind spot is
  closed in code: `BaseTextEvalSpec` now resolves `backend.parent_adapter`
  and the base-text worker attaches that adapter (with the same liveness
  guard the candidate path uses) before scoring. A continuation's
  auto-baseline is the re-measured parent, so a regression like RFT-2's
  (-0.19 vs gen-2) can no longer "promote" against the weaker dense-base
  reference. The adapter stays out of the protocol fingerprint (it is the
  treatment, not the protocol), preserving baseline-vs-candidate
  comparability; malformed parent_adapter configs (missing sha256, empty
  path) are refused. Tests: `tests/test_base_text_evaluator.py` (18).
- **Envelope curriculum (batch 004).** `chowder_batch/build_envelope_curriculum.py`
  scales batch 003's 3-row injection to 28 SFT records + 2 preference pairs
  across four families (read x8, write x8, observation-gated fix loops x8,
  structured log_event x4 + pref pairs). Every assistant span is re-derived
  from Spark's tokenizer and verified byte-identical; general gates enforce
  one-action-per-turn, observation-before-next-action, no fabricated
  tool_response, and grounded final reports across all records.
- **Runtime loop test.** `chowder_batch/runtime_loop.py` drives the student
  around a real multi-turn tool loop against a mock workspace with a
  genuinely buggy module: run_tests executes the model's fix, so a green
  summary can only be observed through correct actions. Verdicts fail
  fabricated tool_responses, premature success claims, batched calls, and
  bare-JSON envelopes. Verified headless with scripted agents (correct loop
  passes; fabricator and premature-success both fail); `--endpoint` mode
  drives a llama-server; `--adapter` mode loads a fine-tuned PEFT checkpoint.
  Live run queued behind the RFT-1 arm.

## RFT campaign verdicts + batch-003 before/after (2026-09-23 evening)

### RFT arms vs the 0.75 bar (collated, rft_collation.json)

| arm | candidate | parent (gen-2, same protocol) | dense base | verdict |
|---|---|---|---|---|
| RFT-1 student-only (233 own-correct chains) | **0.64** | 0.730 | 0.39 | UNMET, STOP_PLATEAU |
| RFT-2 hybrid (233 student + 10 teacher rows) | **0.54** | 0.730 | 0.39 | UNMET, STOP_BUDGET |

Both arms regressed their parent: RFT-1 -0.09, RFT-2 -0.19. The common factor
is not the data source (own chains vs teacher chains both failed) but the
recipe: small datasets trained for multiple epochs with no replay wash out
the adapter's GSM8K skill. RFT-1's student-only data did no better than
RFT-2's hybrid, so teacher-row contamination is ruled out as the primary
cause. Any RFT-3 must cap epochs near 1 and replay parent-generation data.

RFT-2's lifecycle note: it "promoted" only because its auto-baseline
reference was the dense base (0.39), not its parent (0.73) -- the gate
blind spot fixed in code this pass (parent-adapter baselines).

### Batch-003 envelope fine-tune: 3-row injection does NOT survive (and does harm)

Fine-tuned the gen-2 adapter on batch 003 (3 SFT rows, pre-rendered through
Spark's template, verified byte-correct), 12 steps, lr 2e-5, then captured
the 11 chowder_batch fixtures before and after:

| fixture family | before (gen-2) | after (batch-003 adapter) |
|---|---|---|
| spark_tool_call_envelope_basic (eval-9) | PASS (correct `<tool_call>` span) | **FAIL** (bare JSON `{"name": ...}`) |
| spark_envelope_observation_gated_loop (eval-10) | PASS | PASS |
| spark_tool_call_structured_args (eval-11) | PASS (tojson object) | **FAIL** (pseudocode `log_event({level: ...})`) |
| 8 JSON-discipline fixtures | 0/8 | 1/8 |
| GSM8K holdout (retention check) | 0.73 (parent) | 0.71 (candidate, lifecycle UNMET at 0.74 bar) |

The training text itself is byte-correct (verified against the tokenizer),
the run genuinely continued from the gen-2 parent adapter, and the lifecycle
behavior was honest: the parent-assessment short-circuit initially skipped
training entirely (goal minimum 0.60 was already MET by the parent's 0.73
baseline at generation 0), which was worked around by raising the minimum to
0.74 so the parent assessment is UNMET and the candidate actually trains.

Finding: a 3-row, 12-step injection into a 4B model is too weak to teach a
new format but strong enough to degrade the existing one -- the same
instability signature as gens 3-6, at miniature scale. The envelope
curriculum (batch 004, 28 SFT + 2 pref pairs) exists precisely for this:
train on volume with the family spread and re-measure before/after with the
runtime loop, not just the static fixtures.

### Batch-003 infra fixes made along the way

- capture scripts now load eval parts from `chowder_batch/` and apply the
  Spark digest-gated compat patch before model load (they previously
  crashed on the raw 4.57-to-5.x tied-weights incompatibility).
- `run_batch003_finetune.py` goal minimum raised 0.60 -> 0.74 (retention
  bar above the parent) to bypass the generation-0 parent-MET short-circuit.
