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

### Live runtime-loop verdict (mock workspace, real adapters)

`chowder_batch/runtime_loop.py --adapter` run against both adapters on the
buggy `version.py` workspace (12-turn budget, greedy):

- **gen-2 adapter (0.73)**: emits the correct `<tool_call>` envelope every
  turn, calls run_tests first, then loops 11x on `run_tests` with a
  hallucinated `args` key — it never attempts a `write_file` fix and burns
  the budget without a final report. Envelope discipline is real; the
  multi-turn diagnose-fix-verify loop is NOT (it was never trained).
- **batch-003 adapter**: envelope degraded to mis-formatted calls
  (read_file with `test_1.2`-style paths, repeated ERROR observations) —
  consistent with the static before/after regression.

The loop test works as designed: it produces an honest FAIL for both
adapters today and gives batch-004 training a concrete end-to-end target —
turn-by-turn envelope discipline PLUS stopping only on an observed green
summary. Also fixed in this pass: the runtime loop now applies the Spark
digest-gated compat patch (it crashed on the raw custom-code path before),
and the mock-workspace machinery test still passes offline.

### Batch-004 envelope curriculum: static format gains, but no usable repair loop

Batch-004 trained the gen-2 parent adapter on the 28-row envelope curriculum
for one pass (28 steps, lr 2e-5, LoRA r=16/alpha=32), followed by the fixed
100-prompt GSM8K holdout and the 11-fixture capture. The lifecycle measured
parent 0.730 and candidate **0.700**, ending `STOP_PLATEAU` / `UNMET` at the
0.74 retention bar; it did not promote the candidate.

| fixture family | before (gen-2) | after (batch-004 adapter) |
|---|---:|---:|
| spark_tool_call_envelope_basic (eval-9) | PASS | PASS |
| spark_envelope_observation_gated_loop (eval-10) | PASS | PASS |
| spark_tool_call_structured_args (eval-11) | PASS | FAIL |
| 8 JSON-discipline fixtures | 0/8 | 1/8 |
| **all 11 fixtures** | **3/11** | **3/11** |

Batch-004 therefore retained the two core envelope fixtures, gained one
non-envelope JSON-discipline fixture, but still failed structured arguments.
The score is not a promotion signal: it is a small, mixed-format regression
relative to gen-2 on the full fixture set and -0.03 on GSM8K.

The live runtime loop was then run with the batch-004 adapter for 12 turns:

- calls: `run_tests` 4, `read_file` 6, `write_file` 1;
- `green_seen: false`; no passing `run_tests` summary was observed;
- the final report claimed/started to claim success despite the red suite;
- verdict: **FAIL**, with the same premature-success violation as the prior
  adapters.

Finding: the larger curriculum is enough to preserve the basic envelope on
this fixture and produce a write attempt, but it does not teach grounded
observation use, correct file targeting, or green-gated stopping. Do not
promote batch-004. The next experiment should target those specific behaviors
with replay of gen-2 outputs and explicit negative examples for fabricated
observations and premature success, rather than adding more single-turn
envelope rows.

### Batch-005 replay-heavy repair curriculum: replay did not rescue the adapter

Batch-005 continued gen-2 on 60 rows: 48 deterministic GSM8K replay examples
and 12 template-rendered repair loops. Each repair loop observes a red test
summary, reads the responsible file, writes a fix, observes a green summary,
and only then emits its final report. Training was capped to one pass (60
steps) at lr 1e-5, with the parent-adapter baseline and the same 100-prompt
GSM8K retention protocol.

| metric | gen-2 parent | batch-005 candidate |
|---|---:|---:|
| GSM8K holdout | 0.730 | **0.680** |
| all 11 static fixtures | 3/11 | **2/11** |
| envelope fixtures (eval-9/10/11) | 3/3 | 2/3 |
| JSON-discipline fixtures | 0/8 | 0/8 |

The lifecycle ended `STOP_PLATEAU` / `UNMET` at the 0.74 bar and did not
promote the candidate. The live 12-turn runtime loop also failed: it made one
`run_tests` call followed by eleven increasingly malformed `read_file`
paths, never wrote a fix, never observed green, and exhausted the budget
without a final report.

Finding: replay plus green-gated positive transcripts did not preserve GSM8K
or repair behavior. The likely issue is not the presence of replay rows, but
the single-pass LoRA update on a small, highly templated repair set: the
candidate became more deterministic about the wrong read path and still
could not connect the observed failure to the correct workspace file. Do not
promote batch-005. A subsequent attempt should preserve the parent through
an even smaller adapter update or train only a repair-specific module while
keeping the gen-2 adapter frozen, and should include negative examples that
penalize repeated nonexistent reads.

### Batch-006 corrected teacher data: retention improved, repair behavior did not

Batch-006 applied the two requested teacher-data corrections: rejected
batched calls are represented explicitly, and the repeated-test preference
pair includes the preceding red observation. The dataset contains 32 positive
repair trajectories, including four still-red recovery loops, plus eight
preference pairs. It was rendered through Spark's chat template and mixed
with 48 gen-2 replay rows for one 60-step pass at lr 5e-6.

| metric | gen-2 parent | batch-006 candidate |
|---|---:|---:|
| GSM8K holdout | 0.730 | **0.720** |
| all 11 static fixtures | 3/11 | **2/11** |
| envelope fixtures (eval-9/10/11) | 3/3 | 2/3 |
| JSON-discipline fixtures | 0/8 | 0/8 |

The lifecycle ended `STOP_PLATEAU` / `UNMET` at the 0.74 bar and did not
promote the candidate. GSM8K retention improved from batch-005's 0.68 to
0.72, but remained below the parent. The live 12-turn loop still failed:
one `run_tests` call was followed by eleven nonexistent `read_file` paths,
with no `write_file`, no green observation, and no final report.

Finding: the corrected teacher data and lower learning rate reduced the
GSM8K regression but did not change the model's runtime failure mode. The
static envelope fixtures remain insufficient as a promotion signal for the
multi-turn repair policy. Do not promote batch-006. The next experiment
should change the training mechanism rather than add more near-duplicate
positive repair rows: freeze gen-2, train a repair-only module or use a
preference-aware objective, and include runtime traces where nonexistent
reads receive an explicit negative reward.

### Batch-007: frozen gen-2 repair-only reward-weighted module (prepared)

Batch-007 changes the mechanism rather than adding another continuation pass.
The new `chowder_batch/runtime_trace_reward.py` scores saved transcripts with
signed additive reward. It awards reward for a real `write_file`, an observed
`N passed` summary, a post-write final report, and bounded completion. It
applies explicit negative reward to every nonexistent read, repeated read
paths, recurrent malformed path families, tests before a write, repeated
post-write tests without green, budget exhaustion, fabricated observations,
and premature success claims. `runtime_loop.run_loop()` now embeds its
transcript and the reward report in each verdict, so the score is auditable
from the same run.

The repair-only backend path is guarded by `backend.repair_only: true` and a
hash-bound gen-2 `parent_adapter`. The worker loads gen-2 with
`is_trainable=False`, verifies that every gen-2 LoRA parameter is frozen,
adds a separately named `repair` LoRA, verifies that it has trainable
parameters, and records both invariants in worker provenance. The adapter
artifact retains `default` plus `repair`; evaluators activate both and check
liveness independently.

`chowder_batch/build_repair_reward_data.py` produced 24 rows: 12 positive
observed-green repair trajectories and 12 negative-reward rows targeting
nonexistent reads, repeated tests, and ungrounded success. The trainer recipe
is `.chowder-spark-calib/gsm8k/run_batch007_repair_only.py`, capped at 24
steps at lr 2e-6 with no replay. The dataset builder and offline tests pass.

#### Batch-007 infrastructure verification

Before any rerun, the reward-column failure was reproduced against a CPU
`Trainer` step for both text and chat rows. The worker now removes all source
columns during `datasets.map()` and returns exactly one encoded reward field;
it also disables Trainer's unused-column pruning for reward-aware runs. The
negative objective is capped token unlikelihood, not unbounded negative NLL.
Focused CPU coverage is in `tests/test_reward_training_cpu.py` and includes
finite extreme-logit behavior, one-step text/chat training, event rows, bundle
hashing, and runtime-gate vetoes.

`chowder_batch/build_event_reward_data.py` emits one row per tool action with
context, action, observation, event label, and signed reward.
`chowder_batch/runtime_benchmark.py` exercises three deterministic repair
workspaces and reports `runtime_reward`, `runtime_green_rate`, and
`runtime_nonexistent_read_rate`. `Goal` and the existing promotion gate now
support hard runtime-reward and nonexistent-read thresholds; missing runtime
evidence is a veto.

The adapter bundle is now `chowder-adapter-bundle-v1` with frozen `default`
plus `repair`, linear activation, repair-content hash, and a root parent
adapter hash. Both text evaluators validate the manifest before loading the
combined adapter.

The corrected Spark batch-007 rerun completed training and evaluation. The
candidate scored **0.720** on the GSM8K holdout versus the frozen gen-2 parent
**0.730**, with no candidate error. The lifecycle ended `STOP_PLATEAU` /
`UNMET` at the 0.74 bar and did not promote the candidate. The reward-column
failure and the earlier manifest-root validation failure are therefore both
resolved in the real run; batch-007 remains rejected on measured GSM8K
performance, not on a training or adapter-load error.

#### Batch-007 live runtime comparison

The runtime benchmark is now part of the Transformers evaluation protocol when
`evaluation.runtime_benchmark.enabled` is true. Both arms load the same Spark
model and use the same three workspaces, turn budget, and tool protocol. The
completed comparison was:

| metric | gen-2 parent | batch-007 candidate |
|---|---:|---:|
| GSM8K holdout | 0.730 | 0.720 |
| runtime green rate | 0.333 | 0.333 |
| runtime nonexistent-read rate | 0.333 | 0.333 |
| runtime reward | -4.333 | -4.333 |

The candidate and parent traces are behaviorally identical on this benchmark.
Both solve `sum_text`; both fail `version_parser` after writing an incorrect
fix, then read a nonexistent test path and repeat red tests; both stop after
only reading `slugify.py`. The runtime safety gate therefore vetoes promotion
even before the GSM8K target is considered.

#### Proposed batch-008: event-grounded repair-only module

Batch-008 should remain a frozen-gen-2 repair-only run, but change the data
unit and the acceptance rule:

1. Build action-level rows from the three live traces with the exact context,
   tool call, observation, event label, and signed action reward. Include the
   failed version-parser write as a negative row, the nonexistent
   `test_version.py` read as `-4`, repeated red tests as `-1` each, and the
   slugify early stop as a negative completion example.
2. Add corrected positive demonstrations for all three tasks. The version
   demonstration must contain the real padding loop, and the slugify
   demonstration must write the file, run tests, observe green, and only then
   report success.
3. Preserve completion-only assistant masking for chat rows and use the bounded
   unlikelihood objective. Keep the adapter small (`r=8`) and lower the update
   rate to `1e-6` for 12--16 steps to reduce the observed 1-point GSM8K
   regression.
4. Declare runtime gates before training: `runtime_green_rate == 1.0`,
   `runtime_nonexistent_read_rate == 0.0`, and candidate runtime reward above
   the parent. Keep GSM8K non-regression as a separate protected metric.
5. Abort promotion if the runtime evaluator is missing, if either arm has
   incomplete runtime evidence, or if the candidate merely matches the parent
   runtime score.

The next experiment should be accepted only if it improves runtime behavior
without sacrificing the frozen parent’s GSM8K baseline; batch-007 currently
fails both runtime safety gates and the GSM8K target.

#### Batch-008 result and RRSI pivot

The first batch-008 event-only run exposed a real training-system issue rather
than a model-quality signal: Spark's chat template is not prefix-consistent for
completion-only masking, and an initial masked-text collator also tried to pad
unmasked labels incorrectly. The worker now supports `completion_field`, searches
for the completion token subsequence in the rendered text, pads labels to the
input length, and uses a seq2seq padding collator for those rows. A corrected
16-step run then completed training and evaluation. It reached GSM8K **0.700**
with runtime green rate `0.333`, nonexistent-read rate `0.333`, and runtime
reward `-4.333`; it was rejected. The parent remains `0.730` with the same
runtime metrics.

A guarded harness variant was also tested directly against gen-2 and batch-008.
It made behavior worse: parent runtime reward fell to `-8.333`, candidate to
`-12.333`, while green rate stayed `0.333` and nonexistent-read rate stayed
`0.333`. It is therefore rejected rather than enabled.

The next direction follows RRSI (arXiv:2609.24972): evolve the frozen-model
harness, not more adapter weights. The new `chowder.harness_evolution` module
implements the paper's transferable pieces: annealed edit budgets, leakage
screening, noise-aware conservative acceptance, cost-aware acceptance, and
component pruning. The current plain harness remains the incumbent; guarded
prompt/control-flow edits must earn their place on held-out tasks before they
can become the default.

### Batch-009: controlled harness experiment

Batch-009 isolates harness changes from any training update. The gen-2 parent
(`F:\Huihui-Spark-X2.5-4B-abliterated` + gen-2 adapter) and the plain harness
stay frozen as the reference; every candidate differs only in harness code.

#### Expanded benchmark

`chowder.runtime_eval` was rebuilt around a task-family model. The evolve split
now has **24 development tasks** and the held-out split **13 tasks**, spanning
five families in both splits: imports, stateful bugs, multi-file repairs,
failed-first fixes, and misleading test output. Each task carries a `check`
function instead of a single-substring oracle; the old substring oracle had
latent false-greens (`sum_text`'s marker was a substring of its own bug, and
`shared_memo`'s matched the bug itself). `_is_green` now rejects "0 passed",
partial-count lines, and observations containing failure/traceback markers, so
misleading output can be scored correctly. Tests assert that every task starts
red and that the two splits stay disjoint.

New named metrics are published by `run_live_benchmark` and whitelisted by both
text evaluators through `RUNTIME_METRIC_KEYS`: green completion, invalid
(nonexistent) reads, premature completion, repeated actions, execution cost
(read 1 / write 2 / run_tests 5), and policy tokens, plus per-family green
rates.

#### Two generic mechanisms

* `state_aware` — file discovery from actual workspace state: a system message
  listing the real files, refreshed every turn, and a miss that reports the real
  available paths. Nothing is hard-coded; the listing comes from the live
  workspace dict.
* `recovery` — an explicit post-red state: after a failed `run_tests`, the
  harness requires a corrected `write_file` and refuses to execute a re-run
  until the workspace changes (the blocked call is recorded as a zero-cost
  synthetic event and is not counted as a repeated real action).

The old `guarded` prompt is retained only as a legacy variant; it is not one of
the mechanisms.

#### Paired results (identical tasks, decoding, seeds; full traces in
`F:/chowder-campaign/batch009-harness/harness_compare.json`)

| metric | plain evo | state_aware evo | recovery evo | plain held | state_aware held | recovery held |
|---|---:|---:|---:|---:|---:|---:|
| reward | -10.42 | -9.75 | -6.63 | -13.08 | -4.85 | -8.15 |
| green completion | 0.292 | 0.292 | 0.375 | 0.231 | **0.538** | 0.385 |
| invalid reads | 0.250 | 0.208 | 0.250 | 0.615 | 0.462 | 0.538 |
| premature completion | 0.708 | 0.750 | 0.750 | 0.769 | 0.615 | 0.692 |
| repeated actions | 0.417 | 0.792 | **0.250** | 0.692 | 0.923 | 0.462 |
| execution cost | 7.54 | 9.25 | 6.63 | 11.31 | 10.46 | 8.31 |
| policy tokens | 5339 | 13885 | 14015 | 3924 | 6203 | 5899 |

Both mechanisms passed the regularized selector (`accepted: evolve gain
transferred without held-out regression`). Trace inspection explains why, and
where each one breaks:

* `state_aware` converts the dominant early-report failure (report after only a
  read) into completed repairs. Plain traces like `read -> (stop)` became
  `read -> write -> run_tests[green] -> report` on held-out tasks
  (`mutable_kwargs`, `shared_buffer`, `size_parser`, `import_typo`,
  `import_pkg_reexport`). The listing steers the model toward missing imports
  (`lib/__init__.py` instead of hallucinated `lib/shout.py`).
* `recovery` fixes the repetition failure: repeated actions fall 0.417 -> 0.250
  on evolve, and execution cost falls with them, because a blocked re-run is
  never executed. Its failures shift to acting without re-reading the target.
* Costs are real: both mechanisms roughly triple evolve policy tokens, and
  `state_aware` regressed two evolve tasks (`single_file` 0.33 -> 0.00,
  `multi_file` 0.25 -> 0.00) plus one held-out multi-file task while it fixed
  four held-out families. `state_aware` also raised evolve repeated actions to
  0.792 — the file listing invites revisits.

The failure modes that remain in every arm are semantic: `failed_first_fix`
tasks need a genuinely different second-attempt fix, and misleading-output
tasks need the model to distrust a stale report line. No prompt mechanism in
this batch addressed those.

#### Batch-009 outcome and the training gate

This is the first transferable harness improvement since batch-008: `state_aware`
lifts held-out green completion from 0.231 to 0.538 (including 1.000 on the
held-out import and stateful families), and `recovery` is a cheaper, safer
loop. The next training batch, if any, should train only on
genuinely new trajectories generated under the winning harness on the expand
split, keeping the held-out tasks here untouched as the final test set.The alternatives — accept the token cost on the harness alone, or generate the new
trajectories under a combined `state_aware+recovery` harness — are exactly the
next paired measurements to take.

#### 2026-09-25 — batch-009 v2: sixth family, compact prompt, green revocation (code, not measurements)

The batch-009 harness code was substantially revised on 2026-09-25. **No live
rerun has happened since**: everything in the paired-results table above is
historical, computed under the *v1* semantics and the *five-family* benchmark,
and must not be quoted as current. The stale artifact
`F:/chowder-campaign/batch009-harness/harness_compare.json` (24 evo / 13
held-out tasks, old scoring) is refused by the new code via the `run_config`
fingerprint.

What changed in `chowder.runtime_eval` / `batch009_harness_experiment.py`:

* **Sixth task family `wrong_second_fix`.** Targets the dominant semantic
  failure left after v1: the second attempt after a red test is a cosmetic
  variant of the first fix rather than a genuinely different repair. The
  synthetic forced-red `wrong_second_fix_partial_red_seen` row was removed;
  the family is scored purely on real partial-then-corrected trajectories.
  Splits were therefore grown past the 24/13 recorded above; the v2 numbers do
  not exist yet.
* **Scoring semantics change — green is revocable.** A content-changing
  `write_file` after a green test, or a subsequent red test, resets
  `green_seen_so_far`; `green_seen` now means the *last* state is green, and a
  final report counts only if green still holds at report time. Any historical
  green-rate comparison across this change is invalid. `budget_exhausted` is a
  new trace role (distinguished from premature reports).
* **Compact `state_aware` prompt.** The default system message is now
  `"Files: <paths>. Read listed paths; test before success."` — the verbose
  form is retained as `state_aware_legacy` for a like-for-like cost
  comparison. `RUNTIME_METRIC_KEYS` additionally records prompt/policy char
  and token counts, so the compact-prompt token saving is directly measurable
  in the next run.
* **`state_aware+recovery` combined harness** is first-class (harness list
  validated: unique, known, plain required; stale/unreadable checkpoints abort
  with `SystemExit` instead of silently resuming; the model loads only when
  arms are missing; resume requires metrics present). The `run_config`
  fingerprint covers both splits' task names, max_turns, max_new_tokens,
  harnesses, and resolved base/parent paths, tagged
  `harness_version="batch009-v2-wrong-second-fix-compact-state-aware"`.
* **Trajectory/export guards** (`exp_e_pipeline.py`): repair trajectories are
  evolve-only, re-verified for green-at-end plus final-report-after-green, and
  batch-010 export rejects held-out task names, held-out content hashes,
  held-out metadata fields, duplicate digests, and stale traces.

Blocked (unchanged): the paired v2 rerun — including the combined
`state_aware+recovery` arm, the compact-prompt token measurement, and any
batch-010 trajectory generation — requires the GPU backend, which remains held
by an unrelated llama-server process (PID 3764) with probes to the historical
ports timing out. No v2 artifact has been fabricated to fill the gap.
