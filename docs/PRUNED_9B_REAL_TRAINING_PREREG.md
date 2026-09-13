# Pre-registration — first real training run on the pruned 9B

Written **before** the run. Every threshold below is fixed now, including the one
that makes me call it a failure.

## What is being trained, on what

* **Model:** `F:\llm-models\Qwen3.8-9B-Pruned-CW-3456` — `qwen3_5` hybrid, FFN
  pruned to 3,456 of 12,288 channels by a corpus-wide activation ranking. Total =
  active = 5.937B. It starts at **1.863× / 3.014×** dense perplexity on two held-out
  splits, i.e. pruning cost real capability.
* **Data:** the **GSM8K train split** (7,473 human-written worked solutions), local
  cache, rendered as plain text (`Question: … Answer: <solution>`). Train and test
  are the official disjoint splits, and no test problem is trained on.

  *Why not the agentic SFT corpus, which was the first choice:* 3,822 of its 3,954
  rows (97%) contain a `tool` role, and Chowder's chat contract allows only
  system/user/assistant. Widening a validated assistant-masking contract to fit one
  run is exactly where silent mislabeling would come from, so the corpus was dropped
  rather than the contract. Using GSM8K train instead also makes this a direct
  capability test on the program's primary metric rather than an off-distribution
  transfer test, which is a better experiment.
* **Engine:** Unsloth (5.96 GB / 65 s for the level-2 probe versus 11.66 GB / 137 s
  for Transformers, at identical 200-module coverage).
* **Eval:** GSM8K, the program's stated primary metric, 50 problems from the locally
  cached `openai/gsm8k` test split, scored with the new `final_number_match` mode at
  `max_new_tokens: 768`.

## Why GSM8K is scored this way, and what split it measures

`normalized_exact_match` cannot score GSM8K: the model shows its work, so a correct
answer never equals a bare number. `final_number_match` takes the last number on
each side, comma-aware — both properties taken from the frontier repo's
`FINDINGS-GSM8K-EVAL-BUG.md`, which records a harness whose extractor split
"$70,000" into `["70","000"]` and scored a correct answer wrong, after which a
self-improvement loop "trained on a failure" that was not one. The 768-token budget
is from the same finding (256 truncated reasoning before the final number).

**Train and test are the official disjoint GSM8K splits.** This is in-distribution
capability training, not transfer: the question is whether the pruned model can
recover arithmetic ability it lost to pruning. No test problem appears in training,
and the eval set's 50 prompts are fingerprinted by Chowder's holdout index.

## Stated expectation, so the result cannot be reinterpreted afterwards

Now that training is in-distribution, I expect GSM8K to **rise**. That is a
stronger and more falsifiable claim than the transfer version this document
originally made, and it is stated before the run precisely so it can be wrong. 500
steps over ~2,000 of 7,473 problems is still a small budget, so a large gain would
be surprising; a fall would be a genuine negative result about LoRA on a pruned
backbone and will be reported as such rather than explained away.

What I am actually testing is the engineering claim, which has never been
demonstrated at this scale: that Chowder trains this checkpoint on a real corpus, at
full module coverage, inside 16 GiB, with a real capability number measured on both
sides by the same protocol and judged by the gate.

## Recipe

Unsloth engine, `text_only` (text decoder), 4-bit; LoRA r=16 α=32 over the verified
ten-name hybrid list (`q/k/v/o_proj`, `in_proj_qkv`, `in_proj_z`, `out_proj`,
`gate/up/down_proj`) passed as a suffix-match regex so all 200 modules are covered;
`max_length` 1024; batch 1 × grad-accum 4; lr 2e-4 cosine; **500 optimiser steps**
(~2,000 problems, about a quarter epoch); seed 123.

Gate: single metric `gsm8k`, maximise, `minimum_promotion_gain: 0.02` (one problem
in fifty), `require_protocol_match: true`, automatic baseline.

## Pre-registered outcomes

Engineering (the primary claim):

* **PASS** — the run completes with coverage 200/200, a live adapter, a finite
  decreasing loss, peak VRAM under 15.93 GiB, and GSM8K measured on both base and
  trained through the identical protocol.
* **FAIL** — any of: coverage refused, adapter refused as inert, non-finite loss,
  OOM/oversubscription (judged by headroom and step-time blowup, never by an OOM
  exception — this platform pages instead of raising), or an evaluation that cannot
  produce a number.

Capability (secondary; a rise is the stated expectation):

* **RECOVERS** — GSM8K rises by ≥ 0.02 over the measured baseline. This is the
  predicted outcome, so it is the *weakest* evidence here: a confirmed prediction at
  n=50 still needs replication before it means much.
* **FLAT** — within ±0.02. Would mean 500 steps of in-distribution data bought
  nothing measurable at n=50.
* **DEGRADES** — falls by > 0.02. A genuine negative result: LoRA on a pruned
  backbone made arithmetic worse. It is **not** evidence the training path is broken,
  and I will not relabel it as such.

## Limitations acknowledged in advance

* n=50 GSM8K problems. One problem is 0.02, so the gate threshold is exactly one
  problem — differences of a few points are noise at this sample size.
* 500 steps over ~2,000 of 7,473 problems is about a quarter epoch, chosen to fit a
  bounded session, not sized to recover what pruning cost.
* A single seed, one run per arm. No variance estimate.
* `final_number_match` is new code. Its extraction rules are tested, including the
  documented comma case, but this is its first use on a real model.
