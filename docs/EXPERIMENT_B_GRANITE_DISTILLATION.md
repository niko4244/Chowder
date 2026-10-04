# Experiment B: Granite 4.0 H Tiny Distillation

## Status

**Research prototype; no adapter training or distillation result yet.** Granite is the initial student candidate. The unmodified baseline, tiny-model compatibility probe and v4 curated teacher dataset are recorded, but no Granite HF adapter or checkpoint has been trained. Local storage and GPU headroom do not support a safe full-checkpoint attempt at the last check.

The experiment preserves Chowder's existing model and campaign. Repair examples use only the development tasks `two_file_fix` and `class_counter`. Never generate from `HELDOUT_TASKS` or Experiment E eval-repair tasks.

## Baseline and tokenizer

`.chowder-spark-calib/exp-b/granite_baseline.json` records Granite 4.0 H Tiny as a **Q3_K_M llama.cpp inference model**, on preliminary small samples:

| Slice | Correct | Mean latency | Examples |
|---|---:|---:|---:|
| GSM8K | 0.60 | 9.46 s | 10 |
| Factual | 0.25 | 3.94 s | 8 |

These are not full-benchmark or teacher-equivalence results, and do not establish a matched Qwen/Granite protocol. The tokenizer report gives Qwen vocabulary size 248,077 and Granite 100,352 with different tokenizations. Token-level KL over shared vocabulary indices is invalid; condition C is rejected without a principled alignment.

## Tiny compatibility probe

The compatibility verification is recorded in `.chowder-spark-calib/exp-b/toolchain_check_v3_verify.json`. It uses a **random 4-layer, hidden-size-64 GraniteMoeHybrid model**, not the pretrained Granite checkpoint:

- All 55 trainable tiny-model tensors had gradients and finite loss.
- Conventional module LoRA covered the present attention, Mamba and shared-MLP projection modules. It skipped all eight fused MoE expert parameters.
- PEFT `target_parameters` adapted all eight fused expert targets in the probe and produced adapter gradients. Chowder's production worker currently exposes `target_modules` only; actual expert-target training is unsupported/unverified.
- The tiny-model probe observed gradients/routed tokens across routers, shared MLP, experts and Mamba blocks.
- A bitsandbytes 4-bit CUDA linear plus low-rank branch backpropagated on the RTX 5060 Ti. This is a primitive check only: `full_granite_qlora_verified` remains false.

## Teacher-data generations

`.chowder-spark-calib/exp-b/teacher_demos_v1.json` is preserved unchanged. It has 19 rows; four repair `write_file` calls lack content, so it cannot substantiate verified repair demonstrations and is not used for training.

`teacher_demos_v2.json` is also preserved. Its first end-to-end generation exposed a provenance defect: the harness displayed source-predicate “passed” observations, while trace replay executed the Python candidate and found a mismatch. That repair trace was quarantined, as it should have been. The v2 derived JSONL files are not overwritten or silently promoted.

The builder was changed so the runtime harness itself uses the same bounded AST interpreter as trace replay; code candidates are interpreted rather than executed, unsupported constructs are quarantined, and tool actions are checked against the parsed call and observed result. The historical v3 artifacts remain preserved: 15 approved SFT rows, 2 verified repair traces, 9 quarantines, A=9 rows and B=15 rows. They are not overwritten or silently promoted.

The current `.chowder-spark-calib/exp-b/teacher_demos_v4.json` records 26 candidates:

- **18 approved SFT rows** and **2 replay-verified execution traces**, one for each allowed development task (`two_file_fix`, `class_counter`).
- **6 quarantined candidates**, excluded from training.
- Condition A has 12 SFT rows. Condition B has 18 rows, adding six action-selection rows. Condition C remains rejected because the Qwen and Granite tokenizer vocabularies do not align.

Both v4 traces were replayed with the bounded interpreter that supplied the model-visible test observations. `two_file_fix` passed after one test run; `class_counter` included an observed red test followed by recovery and a green test. Trace rows remain separate from SFT rows. Eight reward events were derived from the verified traces for analysis; they are not mixed into either SFT dataset.

The derived v4 chat datasets are `teacher_demos_v4_condition_a.jsonl` and `teacher_demos_v4_condition_b.jsonl`. Their recorded SHA-256 digests and row counts are:

| Condition | Rows | SHA-256 |
|---|---:|---|
| A | 12 | `1b09b74ea82f06d199bcef0b174862ffa60ef2e98cd6bc7eb6acbfa1932b00b8` |
| B | 18 | `1073ea860b7e6f938d6cf799dcb29268031a6b16614fd10e39961d66537794c4` |

The matching metadata is in `teacher_demos_v4_training_conditions.json`. Its rank-4 recipe is a pilot proposal, not a validated full-model setup. In particular, Chowder's production worker does not support PEFT fused-expert `target_parameters`; do not claim MoE-expert coverage in any production adapter.

## Resources and blockers

At the latest read-only preflight, existing llama.cpp endpoints on ports 18081 (teacher), 18082 (Spark) and 18083 (Granite) all answered `/v1/models`; none was started or stopped. The single visible GPU was an RTX 5060 Ti (16 GiB) with about 3.1 GiB free at that moment. C: had about 0.27 GiB free and F: about 9.92 GiB. The full Granite Hugging Face safetensors checkpoint is not locally available; the Granite GGUF is inference-only.

Do not download/load full weights or train until the user provides/authorizes adequate storage and a fresh preflight shows sufficient disk, GPU and endpoint headroom. Do not delete files or stop existing processes to create space. A usable dataset is **not** proof that the full-model recipe fits or that a training run is safe.

## Validation and remaining work

Before v4 artifact generation, `py_compile` and the 20 focused Experiment B teacher-data tests passed; the combined Experiment B/reward/training-data suite had 60 passing tests, and Ruff passed at the prior checkpoint. After generation, both JSONL files matched their recorded row counts and SHA-256 values; all 12/18 rows passed Chowder's chat-message validator and ended with nonempty assistant completions. Reward-event conversion from the two verified traces also completed. These post-generation checks validated the artifacts; they were not a new full-model training run.

Not yet verified: full Granite Transformers load or QLoRA; preprocessing through the exact Granite tokenizer/chat template; worker compatibility for the proposed config; any adapter/checkpoint or training; matched post-training evaluation on untouched held-out tasks. The experiment's success condition is unmet. Do not call Granite equivalent to Qwen without such held-out evidence.
