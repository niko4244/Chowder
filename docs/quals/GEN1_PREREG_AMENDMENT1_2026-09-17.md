# Generation 1 Preregistration — Amendment 1 (2026-09-17)

Written after re-reading the merged production training path
(`transformers_worker.py` / `transformers_peft.py` /
`activation_offload_worker.py`) and before any Gen-1 training compute.
It pins what the original prereg left unspecified and corrects one
configuration that real arithmetic proves infeasible. Nothing already
frozen (target, thresholds, battery, budgets, seeds, promotion rule)
changes.

## A1 — training precision/quantization pinned (feasibility, measured)

The original recipes table omitted precision. Arithmetic against this
card (RTX 5060 Ti, 15.93 GiB usable):

- bf16 full-precision 9B weights ≈ 18.6 GB > card — **impossible**.
- The production `activation_offload` mechanism offloads **saved
  activations** (`saved_tensors_hooks`), never weights; no production
  mechanism offloads weights. Full-precision training cannot be made to
  fit by any merged mechanism.
- 4-bit QLoRA (nf4, `quantization: "4bit"`, `device_map=0`) is a merged,
  production, real-hardware-commissioned configuration on this exact
  card: the documented Unsloth commissioning trained an 8B Qwen3
  abliterated model at **6.5 GB peak VRAM**; the transformers backend's
  nf4 path (`BitsAndBytesConfig`, double quant, bf16 compute) is the
  same mechanism in the executor Gen-1 uses.

Pinned for both recipes: `precision: "bf16"` (compute), `quantization:
"4bit"` (nf4 weights), `gradient_checkpointing: true`.

## A2 — sequence/batch reshaped, effective batch preserved

seq_len 2048 × batch 8 risks activation blow-up under checkpointing on
16 GB. Pinned: **seq_len 1024, batch 4, gradient_accumulation 4** — identical
effective batch (16 sequences per optimizer step) and identical
optimizer-step count (200).

## A3 — project hard-gate mapping (no double rule)

The project-level gate stays real but non-conflicting: goal metric
`quality` (the evaluator's `normalized_exact_match` on the termination
suite), `baseline.mode: "auto"` (the parent is measured by the same
production evaluator), `minimum_promotion_gain: 0.10`,
`require_protocol_match: false`. The **growth promotion rule owns the
frozen scientific thresholds** (target ≥ 0.90 etc.) via
`MetricBinder`/`evaluate_promotion`; the project gate is a
production-plausibility floor, not the scientific verdict.

## A4 — parent measurement reuse (declared, not hidden)

The parent's target-instrument measurement is the **Gen-0 freeze
diagnostics** (protocol `gen0-freeze-protocol-v1`, digest-pinned
`5c8b18ab…`, EOS rate 0.000 / cap-hit 1.000 / trigram 0.988, measured
2026-09-17 08:06 UTC). The instrument's frozen sampling contract is
identical to the candidate's; re-measuring the unchanged parent weights
(content digest `59e767aa…`) would reproduce those numbers at ~0.11
GPU-h cost. The parent run is bound from the freeze evidence with its
generation version and digest; the candidate is measured fresh. This
satisfies the independence requirement (parent measured before any
candidate existed) and is recorded here before the candidate's number
exists.

## Unchanged

Target weakness and thresholds; battery and budgets; stopping rules; the
mechanical promotion rule; seeds; recipe-a/recipe-b lr values (1e-4 /
2e-4); LoRA rank 16 / alpha 32; target modules; max_steps 200.
