# Generation 1 Preregistration — Amendment 2 (2026-09-17)

Written after real hardware execution refuted part of Amendment 1's
feasibility basis. Every claim below is a measurement from this machine,
taken before Gen-1 training is (re)launched. Nothing frozen (target,
thresholds, promotion rule, seeds, batteries) changes; the execution
vehicle does, because the execution vehicle must run on real physics.

## B1 — bitsandbytes quantized paths are NOT executable on this card (measured)

Amendment A1 pinned 4-bit QLoRA (nf4) citing the Unsloth commissioning
(8B QLoRA, 6.5 GB peak). Direct probes refute the stock-bnb path:

- `BitsAndBytesConfig(load_in_4bit=True, nf4, double quant, bf16 compute)`
  on the exact parent: **native segfault** (process dies mid-load, ~40%
  through the weight iterator, exit 3221225477 from the evaluator
  subprocess and `Segmentation fault` under direct probe).
- `load_in_8bit=True`: **native segfault** at the same stage.
- Small-tensor `bitsandbytes.functional.quantize_4bit/dequantize_4bit`
  on the same device: **succeeds** (dequant err 0.097) — the crash is
  load-path-specific (large blockwise quantization during model load),
  not a trivially dead library.
- Suspected root cause: bnb 0.50.2 native kernels vs this GPU's new
  architecture (RTX 5060 Ti, Blackwell, sm_120). Not fixed here; the
  honest conclusion is "refuse this path on this machine."

Conclusion: the A1 QLoRA configuration cannot execute here. Attempt
directories attempt-01..06 (project-validate refusals, then the real
trainer's baseline-evaluator crash) are preserved as the refusal record.

## B2 — the training vehicle: production bf16 with Memory-Fabric streaming (measured)

The production executor already carries a merged, tested mechanism that
moves frozen base weights off the card during LoRA training:
`backend.training.frozen_layer_streaming = "always"` (Memory Fabric,
PRs #81-#82). Base layers are frozen under LoRA, so streaming them from
pinned RAM trains exactly the same parameters with ~18.6 GB of bf16
weights no longer resident. Pinned for both recipes:

- `precision: "bf16"` (compute), `quantization: "none"`
- `training.frozen_layer_streaming: "always"`
- `training.activation_offload: "off"`, `training.optimizer_tiering: "off"`
  (activation_offload's WDDM `resource already mapped` flakiness is
  documented under memory-fraction pressure in
  `docs/MEMORY_FABRIC_ACCEPTANCE.md`; it is not needed here and is not
  invited)
- All other knobs unchanged from A2/A1 (seq 1024, batch 4, grad-accum 4,
  200 steps, LoRA r16/alpha32, lrs 1e-4 / 2e-4).

Cost note, recorded honestly: streaming trades VRAM for PCIe traffic.
The preregistered device GPU-h ceilings (0.25/recipe) are unchanged; if
the measured step cost exceeds them, the binding refuses — that is the
honest outcome, not a threshold to move.

## B3 — evaluation vehicle: production evaluator with `placement: "offload"` (measured)

The production evaluators previously loaded the base fully onto the card
(`quantization: "none"` → OOM for 18.6 GB; `quantization: "4bit"` → B1
segfault). Merged in this cycle: `evaluation.placement = "offload"` —
the exact Generation-0 freeze policy (bf16 weights CPU-summoned, decoder
layers CPU-pinned, transient per-token copies streamed). Placement is
carried by the evaluators' protocol fingerprint.

Real-hardware probe of the production `BaseModelTextEvaluator.evaluate`
path with `placement: "offload"` on the exact parent: **completed** —
2 prompts × 32 new tokens, wall 194.5 s (≈ 3.0 s/token, consistent with
the Gen-0 measured 2.87 s/token PCIe physics). The candidate arm's
adapters are applied over the same dispatched base (`PeftModel.from_
pretrained` after dispatch), so baseline and candidate are scored under
one identical protocol. Pinned: `evaluation.placement: "offload"`.

## B4 — budget consequence of B2/B3 (arithmetic, frozen)

With the measured ~3.0 s/token evaluation physics and 16-diagnostic-prompt
× 128-token generation, the target-instrument measurement costs ≈ 0.10
GPU-h; model load ≈ 0.0148 GPU-h (measured at Gen-0 attempt 2). The
evaluation ceilings in the prereg (load 0.02 / target 0.12 / aggregate
1.00) remain unchanged and sufficient. The protected battery is carried
from the frozen Gen-0 measurement (as A4 already declared), because
re-measuring math500/mgsm under the same protocol would exceed the
aggregate ceiling for a quantity that cannot regress below 0.0.

## Unchanged

Target weakness, instrument, thresholds (EOS ≥ 0.90; unclosed-think ≤
0.250; loops = 0; trigram ≥ 0.900), promotion rule, secondary gates,
seeds, curriculum, contamination firewall (real fingerprints), stopping
rules, and the aggregate budget envelope all stand as frozen in the
prereg and Amendment 1. A REJECTED verdict remains an acceptable,
durable outcome.
