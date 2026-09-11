# Pre-registration — hot-core router-healing pilot

Written **before** the run. The question is narrow and falsifiable: *does training
the router move perplexity at all?* Everything upstream of this bounds what
routing could buy; nothing yet shows a router buying any of it.

## Why pre-register

The thesis this pilot tests is one I have argued for across several steps, which
is exactly when a success criterion invented after seeing the number becomes
worthless. The thresholds below are fixed now, including the one that would make
me call the result a failure.

## Model under test

`F:\llm-models\Qwen3.8-9B-HotCore-E16-k2-h2176` — hot-core upcycled, E=16,
`top_k`=2, core 2176, cold 632/expert, active 3,440 of 12,288 channels (f=0.2799).
Converted checkpoint digest recorded in `evidence/hot-core-upcycling/`.

## Trainable scope

`freeze_for_router_healing` defaults: `mlp.gate.weight` (router) and
`mlp.shared_expert_gate.weight`, 64 tensors / 2.228M params, 6.974B frozen. No
`prepare_model_for_kbit_training` (it upcasts the frozen expert bank to fp32 and
blows the card — measured).

**Attribution caveat, stated up front.** The shared-expert gate is in scope, and
it controls the hot core's contribution (it sits at 0, i.e. sigmoid 0.5, which the
×2 `down_proj` pre-scale calibrates to unit scale). An improvement could therefore
come from the gate re-weighting the core rather than from the router learning to
route. The run records per-group weight movement so the two can be told apart, and
a **router-only ablation is the designated follow-up** if the pilot passes. A pass
driven entirely by the gate does NOT support the routing thesis, and will be
reported as such.

## Data splits — three-way disjoint

| split | indices in `grpo_prompts_borderline_696.jsonl` | n | use |
|---|---|---|---|
| rank | 0, 2, …, 62 | 32 | already consumed: built the channel ranking |
| eval A | 1, 3, …, 63 | 32 | scoring; the 9.8444 baseline came from exactly this |
| train | 64 … 631 | 568 | training only |
| eval B | 632 … 695 | 64 | never seen by ranking or training — generalisation |

Eval A is reused deliberately so the result is directly comparable to the
converted model's recorded init. Eval B exists because eval A's prompts are from
the same contiguous region as the ranking split, and a gain that appears only on
eval A would be weak evidence.

## Recipe

seq 768 (the measured fit ceiling; 1024 oversubscribes), batch 1, accum 1, AdamW
(β 0.9/0.999, wd 0.01), lr 1e-3 with 10-step linear warmup then cosine, grad clip
1.0, gradient checkpointing, nf4 with `shared_expert_gate` skipped. 150 optimiser
steps, evaluated at step 0 and every 25 steps. ~7.1 s/step → ≈25 min total.

lr 1e-3 is high for a fine-tune and deliberate: the router is zero-init and has
150 steps to escape. Divergence is a plausible outcome and the step-0/every-25
trajectory is there to catch it.

## Reference points (all measured, held-out)

| | eval-A ppl | ratio to dense |
|---|---:|---:|
| dense parent | 5.3137 | 1.000× |
| per-token oracle at f=0.28 (unreachable ceiling) | ~5.47 | 1.03× |
| **static hot prune at f=0.28** (no router) | **8.9096** | 1.69× |
| **converted init (this model, step 0)** | **9.8444** | 1.853× |

## Pre-registered outcomes

Evaluation is deterministic (no sampling), so any change on a given split is real
for that split; the open question is generalisation, which is what eval B is for.

- **PASS** — best eval-A ppl ≤ **9.648** (≥2% below the 9.8444 init) **and**
  eval B moves in the same direction.
- **THESIS MILESTONE** — best eval-A ppl ≤ **8.9096**. This is the number that
  matters: it would mean token-conditional routing beats static hot pruning at
  *equal active compute*, which is the entire justification for building an MoE
  here rather than simply pruning. Below this, pruning remains the better answer.
- **WEAK PASS** — eval A improves ≥2% but eval B does not. Recorded as
  overfitting to 2.2M parameters on a small corpus, not as support.
- **FAIL** — eval-A ppl does not improve by 2%, or the run diverges.
- **ATTRIBUTION FAIL** — improvement attributable to `shared_expert_gate`
  movement with the router essentially static. Reported as not supporting the
  thesis regardless of the perplexity number.

## Known limitations, acknowledged now

- The training corpus is ~568 short MMLU-style prompts (~25k tokens). This is a
  pilot for *signal*, not a trained model, and 150 steps over ~33 768-token
  blocks is several epochs — overfitting is expected and is why eval B exists.
- Perplexity is not the program's target metric; GSM8K is. A perplexity move is
  necessary-but-not-sufficient evidence.
- The router is zero-init, so at step 0 logits are tied and `torch.topk` selects
  the same two experts for every token (an implementation detail, not a
  guarantee). Gradient does reach all 16 router rows, because `softmax` runs over
  all experts before `topk` — verified in the installed transformers source — so
  symmetry can break, but the 14 unselected experts receive no signal about what
  they *would* have contributed. Classic MoE cold start; a pilot that fails may
  fail for this reason rather than because the thesis is wrong, and that
  distinction will be stated rather than smoothed over.
