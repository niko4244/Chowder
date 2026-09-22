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
