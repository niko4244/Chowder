# REAP128_v2 expert selection from original BF16 D0 — 2026-10-25

## Status and provenance

**VERIFIED: the frozen REAP128_v2 expert selection is deterministic and complete.**
**NOT YET TESTED: reference CUDA numerical parity, BF16 model pruning, GGUF conversion,
and downstream quality/speed gates.** This selection is not Gen-0 promotion.

- Repository base: `fix/reap-bf16-atomic-resume-20261024` @
  `e6903951c55cf109a53e0b8f7ada139809f72378`.
- Frozen observer:
  `C:\llm-work\reap\observe-v2-bf16-atomic-20261024-222950-223f34d3`.
- Frozen calibration:
  `C:\llm-work\reap\calib-v2-ornith-bare-20261023`.
- Pinned original BF16 source:
  `G:\llm-cold\bf16\Qwen3.6-35B-A3B`.
- Original observer SHA-256:
  `799cc170767f6b17b40c3b01eb3f95ec71310a1154a1ce0570fb07468dde76ed`.
- Frozen calibration SHA-256:
  `f6a09b694ae43a6d8b44b5180a861856501295eabf3d5297eb9749a008cdd929`.
- Recorded 704-sequence × 40-layer D0: PPL **15.451270210051794**, threshold
  below 20, **PASS**. The 41-generation source-bound receipt was independently
  checked against small per-layer hashes and the registered upstream identity.
  This was not a new full source-weight hash or proof of CUDA execution.

## Published source-bound selection

The canonical `reap_prune.kept_experts()` function ranks 256 source experts
per layer using `reap_sum / count`, puts never-routed experts last, applies
stable descending ordering (lower source ID wins ties), and keeps 128. The
destination layout preserves ascending source IDs. No saliency formula was
modified.

- Versioned artifact:
  `docs/quals/reap128-v2-bf16-selection-20261025.json`.
- Working artifact:
  `C:\llm-work\reap\reap128-v2-selection-20261025-002722\selection.json`.
- **Exact artifact SHA-256**:
  `cff1138104455b51d09465e3eba4b98e3453203e09c24f6dafbe26f46d502082`.
- Frozen selection source-input digest:
  `193bc45dfb5b84cf395499864226eda848159bda3ecb7a5e0b1d4f4c30a8c12c`.
- All 40 layers selected exactly 128 unique original expert IDs; the manifest
  includes full ranked order, sorted retained source IDs, source-to-destination
  indices, destination-to-source IDs, layer-stat hashes, and diagnostics.
- Every layer has exactly 9,541,056 routed assignments and all 256 experts used.
- REAP sum contribution retained per layer: minimum **68.29%**, mean **78.75%**,
  maximum **87.20%**. This statistic is not a model quality prediction.
- Score boundary margin across layers: minimum **0.0000187761**,
  mean **0.0010090673**, maximum **0.0062830967**. Small margins merit
  sensitivity analysis but are not additional acceptance gates.
- Diagnostic overlap with historical REAP128 v1: 4,730 of 5,120 selected
  slots (mean 118.25 shared experts/layer), with historical random128:
  2,599 (mean 64.975/layer), and with historical REAP160: 4,988
  (mean 124.7/layer). Different calibrations limit interpretability.

## Verified consumer preflight

The canonical `reap_build_preflight.py` was executed with the **selection
file supplied** and returned `PREREQUISITES_VALIDATED_NOT_BUILT`, including
`selection_sha256` and a pruning command that passes the manifest to the
pruner. This does not open model weight shards or create the target directory.

```powershell
python -B chowder_batch/reap_build_preflight.py --arm reap128_v2 `
  --model-dir 'G:\llm-cold\bf16\Qwen3.6-35B-A3B' `
  --weights-dir 'G:\llm-cold\bf16\Qwen3.6-35B-A3B' `
  --calibration 'C:\llm-work\reap\calib-v2-ornith-bare-20261023' `
  --observations 'C:\llm-work\reap\observe-v2-bf16-atomic-20261024-222950-223f34d3' `
  --selection 'C:\llm-work\reap\reap128-v2-selection-20261025-002722\selection.json' `
  --out 'G:\llm-cold\bf16\Qwen3.6-REAP128-V2'
```

## Next authorized model-build phase (NOT EXECUTED)

**BLOCKED until operational authorization and fresh resource admission.**
The preflight-printed command is:

```powershell
python -B chowder_batch/reap_prune.py `
  --model-dir 'G:\llm-cold\bf16\Qwen3.6-35B-A3B' `
  --weights-dir 'G:\llm-cold\bf16\Qwen3.6-35B-A3B' `
  --observations 'C:\llm-work\reap\observe-v2-bf16-atomic-20261024-222950-223f34d3' `
  --keep 128 --method reap `
  --selection 'C:\llm-work\reap\reap128-v2-selection-20261025-002722\selection.json' `
  --out 'G:\llm-cold\bf16\Qwen3.6-REAP128-V2'
```

Before launching: recheck source and selection digests, output nonexistence,
host RAM/commit availability, available disk, estimated ~35.4 GiB of BF16
pruning output plus temporary staging headroom, and ownership of the intended
destination. A separate ~35 GiB BF16 GGUF intermediate cannot safely share
the currently available G: headroom with the pruned model. Plan isolated
conversion on F: and quantized outputs on volumes with independently checked
free space. Do not start model/GPU loads or disrupt shared model services.

Once the BF16 candidate is verified, follow the frozen Q8_0 GSM8K/HumanEval
quality and random-control protocol; independently produce Q4_K_M for the
RTX 5060 Ti 16 GiB placement and throughput gate. The old `reap128`
failed GSM8K Gate 1, so the Amendment 4 REAP128_v2 and REAP160 comparisons
remain mandatory. The historical REAP160 Q4 fit/speed resolution must be
reconciled with its canonical `serve.json` witness without altering archived
evidence.

## Engineering hardening and remaining limitations

- Pruning now rejects a 40×256 source without `observation.json`, a 704-row
  v2 observation without its registered BF16 receipt, and an unregistered
  384-row v1 calibration hash. The direct-script package import was repaired.
- The optional `--selection` is revalidated against its exact source
  observation, layer statistics, expert layout, source config, and index before
  any model weights are opened. Preflight also consumes and hashes it.
- **Continuation, 2026-10-25:** For the registered 704-row/40-layer
  REAP128_v2 128-expert build, `--selection` is now mandatory, and the
  intended prune destination must match the manifest exactly. Direct pruning
  also refuses output paths that overlap original model, weights or observation
  trees, and binds the sealed observer's source paths to the actual model
  arguments. The production preflight independently checks the live upstream
  **metadata and 26 shard sizes** against the registered public pin; it does
  not read all weight payloads.
- At the **actual prune build** (not this metadata-only preflight), the
  pruner streams each of the original 26 shard payloads and checks the live
  bytes against the pinned upstream LFS SHA-256 hashes **before creating the
  staged output**. That full ~71.9 GB verification is an intentional
  scientific-lineage expense of the model build, not a completed rehash in
  the present qualification.
- Read-only production safetensors-header inspection, no payload reads:
  **26** shards, **1,045** indexed tensors, **120** expert/router tensors,
  original tensor payload **66.9655 GiB**, projected pruned tensor payload
  **35.3727 GiB**, difference **31.5928 GiB**, largest projected output
  shard **2.4717 GiB**. These are tensor bytes, excluding safetensors
  headers and small model metadata. Disk availability must be reconfirmed
  before the actual staging job.
- The production `reap_build_preflight.py` now repeats that 26-shard
  **header-only** census at the real pinned source, validates all 120
  Qwen `gate_up` / `down` / router shapes and BF16 dtypes, and refuses
  insufficient disk space before printing an executable pruning command.
  The original `--selection` and live-source metadata pins are also checked.
  On the October 25 live admission snapshot, expected pruned tensor bytes
  **37,981,137,632**, intended-stage minimum including an operational
  8 GiB reserve **46,571,072,224** bytes, available G: free space
  **74,565,382,144** bytes; result
  `HEADER_SIZED_AND_DISK_ADMITTED_NOT_BUILT`.
  The 8 GiB reserve is an operational safety margin, **not** a change to any
  registered model-quality criterion. No source tensor payloads were loaded.
- A separate read-only host snapshot showed approximately **18.39 GiB**
  available physical RAM, **28.70 GiB** available Windows commit, **833**
  processes, and **11,287 MiB** free on the RTX 5060 Ti. These values are
  transient and do **not** authorize a substantial job or establish safe
  memory headroom for an unrestricted run; re-admit immediately before
  starting the CPU pruning/conversion pipeline.
- Bounded CPU-only synthetic reference parity now compares the installed
  Qwen3.5 MoE experts with the observer's hook: exact BF16 outputs,
  independently accumulated router statistics (rel `2e-5`, abs `1e-4`),
  and exact BF16 decoder activations across both full and linear attention.
  The final next-token NLL uses rel `1e-6`, abs `2e-3`; **3 tests passed**.
  This materially strengthens code-level equivalence but is **not a CUDA
  numerical parity test of original BF16 weights**.
- Git's default Windows `core.autocrlf=true` corrupted the exact bytes of the
  versioned upstream manifest in a new worktree; `.gitattributes` now excludes
  both byte-addressed JSON manifests from text transformations. No hash pin or
  acceptance threshold was changed to work around the failure.
- **NOT YET TESTED:** independently recomputing source NLL from BF16 logits;
  actual reference-vs-observer CUDA forward parity; absolute external execution
  attestation; full bytes of active scratch slab in this audit; power-loss
  durability; any v2 pruned model; conversion and serving/evaluation.
