# A4 plan — scale complete-trace distillation and test it on harder math

Status: proposal, awaiting operator approval (2026-09-28). Nothing here has run.

## Where A3 left us (measured)

Pinned GSM8K set (openai/gsm8k test @ 740312ad, 120 problems), greedy, 2048 tokens,
EOS-gated boxed-aware scorer (`feat/kaggle-a3-lane` @ 9f804e4), re-scored from saved predictions:

| arm | correct | EOS-finished | avg tokens | vs base (paired, 95% CI) |
|---|---|---|---|---|
| base Qwen3-1.7B | 68/120 (56.7%) | 73 | 1,621 | — |
| condition_a (chunked OT3) | 23/106 | 33 | — | −36.8 pt [−47.2, −26.4] |
| **A3 local bf16** (2,000 complete R1 traces, 2 ep) | **95/120 (79.2%)** | **106** | 1,055 | **+22.5 pt [+14.2, +30.8]**, 30W/3L |
| A3 Kaggle fp16 (same data/recipe, 2xT4) | eval running | | | reported separately |

Open questions A3 cannot answer: (1) does more complete-trace data keep helping, and
(2) does the gain hold on harder problems than GSM8K, where A3 is already at 79%?

## A4 design (one variable: data scale)

* **Data.** Same builder (`build_openr1_pilot.py`), same rules (complete, math_verify-correct,
  single closed non-empty think, boxed answer, rendered ≤ 4096 by the trainer's renderer,
  decontaminated vs GSM8K test + MATH-500), applied to all 10 shards of OpenR1-Math-220k
  `default` @ e4e141ec. Measured yield on shard 0: 2,994 / 9,374 (32%) → est. ~30k usable.
  **A4 draws 6,000 train + 200 dev**, problem-disjoint from each other; A3's 2,000 are a
  subset of A4's train set so the comparison is purely "more of the same data".
* **Recipe.** A3's exactly (LoRA r16/α32, q/k/v/o, lr 2e-4 cosine, 2 epochs, effective
  batch 32, max_length 4096, seed 2026), Kaggle fp16 2xT4 variant. 2·⌈6000/32⌉ = 376 steps.
* **Cost.** A3 took 3.5 h for 126 steps on 2xT4 → A4 ≈ 10.5 h: one 12 h session, with
  `--save-steps 10` so an overrun resumes (resume proven in proof-2/3). 26.2 h quota left
  this week (refresh 2026-10-03).

## Evaluation (both benchmarks, one environment for every arm)

* **GSM8K**: the same pinned 120, unchanged protocol — continuity with A3.
* **MATH-500** (cached, 500 problems, levels 1–5; 180/500 answers are non-numeric LaTeX
  like `\frac{14}{3}`, `3\sqrt{13}`, `\text{Evelyn}`): final-number matching cannot score
  these. Add a `math_verify_match` scoring mode (math-verify 0.9.0 is installed; it is
  the checker OpenR1's own correctness flags use) behind the same EOS gate and boxed
  extraction. Budget 4096 tokens (MATH reasoning runs longer than GSM8K).
* **Size.** Full MATH-500 at 4096 tokens is ~a day per arm locally. Use a pinned,
  seed-2026 stratified subset of **150 problems (30 per level)**; arms: base, A3, A4.
* **Where.** All arms in one environment so precision is not a confound: either all
  local (bf16, ~8 h/arm on MATH-150) or all on Kaggle (fp16; needs a small eval kernel,
  runs arms in parallel sessions). Decide by quota after A4 training.

## Promotion gate (decided before any number exists)

A4 is preferred over A3 only if, on paired rows:
1. MATH-150: A4 − A3 95% CI lower bound > 0; and
2. GSM8K: A4 − A3 CI lower bound > −5 pt (no meaningful regression); and
3. EOS rate on both benchmarks not lower than A3's.
Otherwise A3 stays the reference, and the result is recorded either way. Decision is
always `requires_operator_review`; training loss is never evidence.

## Needs operator approval

1. Download the other 9 shards (~1.93 GB, Apache-2.0, pinned @ e4e141ec).
2. Upload `chowder-openr1-a4` as a private Kaggle dataset (~45 MB).
3. ~11 h Kaggle GPU quota for A4 training (+ eval quota if evaluated on Kaggle).

## Work before any GPU time (CPU only)

* `math_verify_match` scoring mode + tests (EOS gate, boxed extraction, LaTeX equivalence).
* MATH-150 pinned subset builder (stratified, sha256-pinned, disjoint from training).
* Builder run over 10 shards; manifest + independent re-render audit (as for A3).
