# Pre-registration: REAP expert pruning of Qwen3.6-35B-A3B for a 16 GB card

Frozen 2026-10-17, **before any pruned checkpoint exists and before any arm is
scored**. Any later change is an amendment that names what it changes and why,
and no gate or threshold moves after a result is visible.

## 1. Question

Can REAP (arXiv 2510.13999), calibrated on our own workload, remove half of
Qwen3.6-35B-A3B's routed experts so the model runs fully GPU-resident on an
RTX 5060 Ti 16 GB, without measurable loss against the dense parent on the
gates below?

## 2. Arms

All arms are converted locally from BF16 safetensors with the same llama.cpp
build (10985, `760984655`). Each uses `convert_hf_to_gguf.py --no-mtp`, then
`llama-quantize Q8_0` with no imatrix, and every arm uses the same chat
template.

| arm | what | experts/layer |
|---|---|---:|
| `dense` | `Qwen/Qwen3.6-35B-A3B` at commit `995ad96eacd98c81ed38be0c5b274b04031597b0` | 256 |
| `reap128` | REAP saliency over calibration v1, top 128 per layer | 128 |
| `random128` | seeded random 128 per layer (seed 20261017), **negative control** | 128 |
| `reap160` | REAP top 160 per layer, **pre-declared fallback**, built only if `reap128` fails Gate 0 or Gate 1 | 160 |

Calibration v1: `calibration.jsonl`, sha256
`d43c1f8d49d8d7c70872367ba8a56e7052a3dd49351463eed292fd122a631725`. It has
384 rows and 651,861 tokens:
- 192 SWE-smith agentic trajectories;
- 96 OT3 code reasoning;
- 48 OpenR1 math;
- 48 OpenStax prose.

No Fable-derived text is included. Rows sharing a 13-gram with GSM8K test,
MATH-500, MGSM or HumanEval were dropped (1 row).

## 3. Serving and decoding (identical for every arm)

- `C:\llama.cpp\bin\llama-server.exe -m <arm.gguf> --fit on -np 4 -c 16384 --jinja`
  (4 slots × 4,096 tokens), `--n-cpu-moe` as `--fit` chooses.
- Greedy decoding: `temperature 0`, `top_k 1`, `seed 7`, with
  `chat_template_kwargs.enable_thinking=false`. **Conclusions do not cover
  thinking mode.**
- `max_tokens`: GSM8K 1,024, HumanEval 768, MATH-500 2,048.
- Harness: `chowder_batch/reap_eval.py` at the commit that adds this
  document. It scores with Chowder production `scoring.score`
  (`final_number_match`, i.e. the answer after the last `</think>` and the
  last comma-aware number). An item whose generation did not stop on its own
  (`finish_reason != "stop"`) scores 0. HumanEval runs the model's code
  against the canonical tests in an isolated `python -I` subprocess with a
  10 s timeout.

## 4. Gates (in order; each is evaluated only if the previous passed)

Per-item paired differences `d_i = x_i(arm) − x_i(dense)` ∈ {−1, 0, 1}.
`LB = mean(d) − 1.645 · sd(d) / √n` is the one-sided 95% lower bound.

- **Gate D0, observation sanity.** The full 40-layer REAP observation must
  report calibration perplexity < 20. Otherwise the observer has a defect and
  nothing downstream is valid.
- **Gate 0, generation health on GSM8K (1,319 items).** `reap128` passes if
  its terminated rate is ≥ dense − 0.02 **and** its degenerate rate (distinct
  trigram ratio < 0.5) is ≤ dense + 0.02.
- **Gate 1, GSM8K accuracy (1,319 items).** Pass if `LB(reap128 − dense) > −0.03`.
- **Gate 2, HumanEval pass@1 (164 items).** Pass if `LB(reap128 − dense) > −0.10`.
- **Control gate, saliency matters.** `random128` is scored on GSM8K. Pass if
  `LB(reap128 − random128) > 0`. If it fails, the REAP ranking is not shown
  to beat chance, and any `reap128` result is reported as "50% expert drop",
  not as a REAP result.
- **Speed and fit.**
  - `reap128` must run with **no** experts offloaded (`--fit` places every
    layer on the GPU) and decode at ≥ 1.3× the dense arm's mean tok/s under
    the same flags. Decode speed is the mean of llama-server's per-request
    `predicted_per_second` on GSM8K.
  - VRAM is recorded with the user's other GPU residents present, and the
    record says so.

Reported only, not gated:
- MATH-500 (500 items);
- a dense-vs-dense rerun on the first 200 GSM8K items (the nondeterminism
  noise floor);
- the GGUF sizes.

## 5. Decision

- `reap128` passes D0, 0, 1, 2, the control gate and the speed/fit gate:
  `reap128` **is** the custom 16 GB Qwen MoE, and it enters Chowder as a new
  Gen-0 candidate under its own freeze.
- `reap128` fails Gate 0 or Gate 1: build `reap160` and apply the same gates.
  Its fit gate is relaxed to "runs at ≥ 1.3× dense"; partial offload is
  allowed and reported.
- Both fail: REAP at these depths does not preserve quality on this workload.
  The answer is the dense parent with expert offload, and the REAP track ends.
- Gate 2 alone fails while 0 and 1 pass: report it as a coding regression,
  and do not promote.

## 6. Known limitations, stated up front

- Single seed, greedy decoding, thinking disabled.
- Calibration v1 contains Claude-3.7-generated SWE-smith trajectories; their
  text only ranks experts and is never trained on.
- Hidden states in the observer are bf16, so saliency carries bf16 rounding.
- The dense arm needs `--n-cpu-moe` offload on 16 GB, so its speed number
  includes PCIe/CPU expert traffic. That is the configuration a user would
  actually run, which is why the gate compares against it.
