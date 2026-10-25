# Frozen REAP128_v2: actual construction / scientific qualification

## Frozen source and selection identity (unchanged)

- Branch: `feature/reap128v2-selection-20261025`, draft PR #282.
- Source original Qwen/Qwen3.6-35B-A3B upstream revision:
  `995ad96eacd98c81ed38be0c5b274b04031597b0`.
- BF16 observation: `C:\llm-work\reap\observe-v2-bf16-atomic-20261024-222950-223f34d3`,
  observation SHA-256
  `799cc170767f6b17b40c3b01eb3f95ec71310a1154a1ce0570fb07468dde76ed`;
  704 sequences × 40 layers, D0 perplexity 15.451270210051794 = PASS.
- Frozen 40×128 selection:
  `C:\llm-work\reap\reap128-v2-selection-20261025-002722\selection.json`,
  SHA-256
  `cff1138104455b51d09465e3eba4b98e3453203e09c24f6dafbe26f46d502082`.

## Fresh host admission and CPU build

At admission, approximately 15.62 GiB physical RAM, 20.84 GiB free Windows
commit, and 69.43 GiB free on the target G: volume were available.
All 26 safetensors headers were independently checked against the original
40×256 BF16 tensor geometry and the preregistered public-source pin.
The projected pruned checkpoint has **37,981,137,632 tensor bytes**
(**35.3727 GiB**) across 26 shards. Largest output shard is 2.4717 GiB.
An 8 GiB operational reserve is in addition to the staged-output estimate.

The separate *CPU-only* admission passed using a conservative 6.1935 GiB
single-shard source/output memory estimate plus 8 GiB RAM and commit buffers.
No CUDA device was exposed to the owned child. The host had two active
independently owned llama.cpp/Ollama model residents occupying most of the
RTX 5060 Ti VRAM; they remain untouched. No GPU qualification was admitted.

The actual source-bound supervised run was started on October 25:

```powershell
python -B -m chowder_batch.reap_prune_supervised --execute `
  --log 'C:\llm-work\reap\reap128-v2-cpu-prune-supervised-20261025.log' `
  --model-dir 'G:\llm-cold\bf16\Qwen3.6-35B-A3B' `
  --weights-dir 'G:\llm-cold\bf16\Qwen3.6-35B-A3B' `
  --calibration 'C:\llm-work\reap\calib-v2-ornith-bare-20261023' `
  --observations 'C:\llm-work\reap\observe-v2-bf16-atomic-20261024-222950-223f34d3' `
  --selection 'C:\llm-work\reap\reap128-v2-selection-20261025-002722\selection.json' `
  --out 'G:\llm-cold\bf16\Qwen3.6-REAP128-V2'
```

Owned child initially identified as PID 81264. The source-payload verification
stage read all 26 original BF16 shard hashes before it created any temporary
model output directory. As of approximately 02:57 local, the model remained
under a private G: staging directory; two output shards had been written.
**That interim state is not a completed model or a promotion decision.**
Consult the complete run log, the published output path, and an independently
generated postbuild verification receipt to determine the final outcome.

## Required model and scientific gates (no automatic promotion)

1. The final G: BF16 directory must exist and contain 26 checked shards,
   original frozen tokenizer metadata and 40×128 output model configuration.
   Use `chowder_batch.reap_verify_pruned` first in limited mode, then
   `--full-compare` for every retained expert row, every unchanged tensor,
   SHA-256 of all output shards, and the registered source-pruning receipt.
   Save each receipt outside the model tree. **This must complete before GGUF.**
2. A CPU-only per-layer Qwen reference streaming smoke is implemented at
   `chowder_batch.reap_streaming_bf16_smoke`. It checks a single bounded
   local-tokenizer forward pass through all 40 BF16 pruned layers and final
   finite logits without loading all model weights into host RAM.
3. After complete BF16 verification, `chowder_batch.reap_gguf_build_plan`
   checks the pinned llama.cpp converter, available storage and distinct
   target filesystems for BF16 GGUF intermediate, Q8_0 quality arm, and
   BF16-derived Q4_K_M fit arm. Planned storage: BF16 intermediate F:,
   Q8_0 output I:, Q4_K_M output G:. No imatrix, no MTP, no Q8→Q4
   requantization. Conversion must not race source construction.
4. Original-model CUDA numerical parity, the actual Q8_0 quality suite
   (GSM8K 1,319 / HumanEval 164), Q4_K_M GPU placement and
   200-item throughput, and the random128 negative-control comparison
   remain distinct required qualifications. No inference job may evict or
   terminate the existing CUDA/Ollama resident workloads to make room.
5. The v1 REAP128 GSM8K Gate 1 lower bound is −0.037807 (strict threshold
   greater than −0.03), so it **failed**. Frozen Amendment 4 requires that
   both REAP128_v2 and REAP160 be fully scored before the deterministic
   ordered selection can choose an arm.

## Separate historical REAP160 speed evidence

The source read-only reconciliation is in
`docs/quals/reap160-historical-speed-reconciliation-20261025.json`,
SHA-256
`b670946fe7178bde445f1c17f3f220889b315a9b7d1007ec0a2ef234a086fa0c`.
The independent load-only probe reports all 41/41 GGUF layers on GPU, a
successful fit, and a 3.8522875643544654× speed ratio from the frozen
200-item historical comparisons. Its original `serve.json` had a null
expert-offload count. Because the probe was separately executed and is not
fully bound to the earlier session, it **corroborates** a fit but cannot
silently convert the canonical speed gate from UNKNOWN into PASS.

## Deterministic frozen-gate snapshot before REAP128_v2 scoring

The canonical frozen `reap_select` command was run against the unchanged
historical full evaluation corpus and the newly D0-qualified original v2
observation, saving an immutable evidence snapshot:

`docs/quals/reap128-v2-frozen-gate-snapshot-prequality-20261025.json`

SHA-256
`7043980b1a5551131c348d720704382eeee42b63001afcccd9f031b7528bb42e`.
The selector deliberately exits nonzero with
`{"status":"BLOCKED","chosen":null,"reason":"amendment 4 requires both REAP128_v2 and REAP160 full gates"}`.
Current historical arm computations: REAP128 v1 D0 PASS, Gate0 PASS,
Gate1 **FAIL**, Gate2 PASS, control PASS, speed UNKNOWN; REAP160 D0,
Gate0, Gate1, Gate2 and control PASS, speed UNKNOWN. The new REAP128_v2
has not been scored on GSM8K/HumanEval/Q4 hardware, so no candidate is selected.

**Release decision until all required evidence exists: BLOCKED, no Gen-0
promotion or owner approval implied.** Neither frozen thresholds nor the
predeclared arm-selection order may be weakened.
