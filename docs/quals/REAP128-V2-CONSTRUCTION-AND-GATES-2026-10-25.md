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
model output directory. **The owned pruner completed with return code 0 and
atomically published the final G: checkpoint at 03:27 local on October 25.**
The build log ends in `OWNED_PRUNER_EXIT: 0, output_published: true`.
The published directory contains 26 BF16 safetensors shards, the untouched
tokenizer/model support files, `config.json`,
`model.safetensors.index.json`, and `reap-prune.json`. Its 40 decoder
layers retain 128 selected experts each (down from 256) and the recorded
tensor payload is **37,981,137,632 bytes**.

An independent production `reap_verify_pruned` sample verifier **passed**
all 26 safetensors headers, all 120 routed expert tensor shapes, 720
bitwise source/destination expert-row samples, the exact selection SHA-256,
original BF16 observation identity, and the published pruning receipt.
Receipt: `C:\llm-work\reap\reap128-v2-bf16-header-sample-verification-20261025.json`,
status `HEADER_AND_EXPERT_SAMPLE_PASS`.
An unmodified, byte-pinned duplicate is checked into
`docs/quals/reap128-v2-bf16-header-sample-verification-20261025.json`
with SHA-256
`83c953825ada3787b22675a5b3ca5790a1971f92e133a24e2325306c67dd441a`.
The stronger `--full-compare` result, an actual final-logits inference
witness, GGUF results, and scientific scores must each be reported separately.

### Runtime qualification interruption (approximately 03:56–04:02 local)

A separate immutable full bitwise `--full-compare` verifier was launched,
supervised under owned-child resource guards. During its run an **unowned
Ollama llama-server** started at 03:56:15 (PID 59020, parent Ollama PID 15608,
262144-token context), occupying nearly all GPU memory and substantial
Windows commit. The supervisor ended the full compare with return code **3**
because host-admission resources were depleted; **no full-equivalence
receipt was published**. Do not label the full check PASS or FAIL on the
science; it was **ABORTED BY RESOURCE GUARD**.

A distinct real-model CPU-only streamed forward had already passed decoder
layers 0, 9, 19, and 29 through both full and linear attention. Before
final logits, the live host briefly reached effectively zero available
physical RAM/commit. Its specific owned Python PID 77212 was then
terminated, returning code 15, preserving all independent services.
**No complete BF16 CPU inference receipt exists.** These intermediate
layers prove a bounded load/execution prefix, not completed inference
correctness or a quality score.

The newly added `reap_streaming_smoke_supervised` prevents a repeat by
requiring a full-equivalence receipt, 16 GiB free physical RAM, 24 GiB
available Windows commit, model/report disk headroom, a new audit log,
and an owned-child runtime guard. A subsequent host snapshot showed
approximately 12.14 GiB RAM and 12.70 GiB commit available, confirming
that a new substantial CPU or CUDA job is **not admitted**. GPU free memory
was only about 611 MiB, with existing services retained. Recheck host
state rather than assuming these transient values are permanent.

**Frozen model remains completely published and untouched.** This
interruption affects only scientific verification and downstream
conversions, never the proven 26-shard source-bound construction or
720 sampled bitwise expert-row checks.

## Resumable full-source equivalence, October 25 continuation

The previous monolithic `--full-compare` was host-aborted before producing
any receipt. To avoid losing another expensive full pass on unpredictable
shared-memory fluctuations, `chowder_batch.reap_verify_shards` now compares
all original BF16 source tensors against the reconstructed candidate
**one original safetensors shard per owned CPU subprocess**. Each completed
shard receives an exclusive, source-bound certificate containing exact
retained-expert row comparisons, all unchanged BF16 tensor comparisons,
source SHA-256, pruned shard SHA-256, and selection/model receipt hashes.
The separate process exits after each shard to release Python memory.

There are 26 indexed shards. Completed receipts live outside the models at
`F:\reap-v2-equivalence-shards-20261025`, so a resource abort never
requires expensive already-attested shards to be repeated. The supervisor
performs fresh RAM/commit and G:/F: disk admission before each owned
subprocess, exposes no CUDA device, and stops **only** its own subprocess if
resource floors are breached. Dry-run is default; `--execute`, an
exclusive log path, and `--max-new-shards` are required for bounded
actual work. No process belonging to Ollama, llama.cpp, or other users
is affected.

```powershell
python -B -m chowder_batch.reap_verify_shards `
  --source 'G:\llm-cold\bf16\Qwen3.6-35B-A3B' `
  --observed 'C:\llm-work\reap\observe-v2-bf16-atomic-20261024-222950-223f34d3' `
  --selection 'C:\llm-work\reap\reap128-v2-selection-20261025-002722\selection.json' `
  --candidate 'G:\llm-cold\bf16\Qwen3.6-REAP128-V2' `
  --receipts-dir 'F:\reap-v2-equivalence-shards-20261025'
```

After all **26** immutable shard certificates exist, invoke the same
command with
`--finalize 'F:\reap128-v2-full-source-equivalence-20261025.json'`.
Finalization rehashes **all** live original source and all candidate
weight shards to reject stale/mutated certificates and checks the global
37,981,137,632-byte tensor geometry and 120 routed tensor census.
Only successful finalization may produce the standard
`FULL_SOURCE_TENSOR_EQUIVALENCE_PASS` receipt accepted by the strict
CPU inference and GGUF production preflights.

**Partial shard certificates are not a full model qualification.** Do
not mark inference, Q8/Q4, CUDA parity, speed, or Gen-0 as passed
until their separate preregistered measurements and mandatory gates exist.

### October 25 execution and resource admission results

- The legacy PyTorch/safetensors full-verification path still allocated
  approximately **7.1 GiB private memory** when opening the first large
  original/source shard. The owned runtime guard stopped the initial
  worker when Windows commit crossed its **8 GiB free safety floor**,
  exit code 3; no proof was written.
- The exact-comparison implementation now validates BF16 safetensors
  metadata (bounded header size, dtype/shape, contiguous unambiguous
  offsets), uses file-backed read-only mappings, and compares each
  selected expert row and unchanged tensor in chunks of at most
  **8 MiB**. It detects bitwise changes including signed-zero differences.
  Full verification no longer calls the PyTorch-backed `safe_open`
  tensor accessor at all.
- The bounded-memory design retains the **8 GiB free Windows commit
  runtime floor** plus 8 GiB of prelaunch margin, giving a 16 GiB
  minimum commit admission. It also requires 12 GiB physical RAM and
  independent target/receipt-volume disk reserves. These are
  operational admission constraints, **not scientific-gate thresholds**.
- Subsequent headerless-worker launches were twice refused **before
  spawning any child**: fresh snapshots were respectively about
  11.74 and 11.78 GiB available RAM, below the 12 GiB requirement.
  No new shard certificates were produced and no model or service
  was changed. The receipts directory remains safely reusable.

These were historical initial refusals, superseded by the October 25
**completed 26-shard full-source verification** below. The refusals were
operational host admission events, never REAP scientific failures.

## October 25 full production BF16 source-equivalence result

The source-bound `reap_verify_shards` comparator subsequently verified
**all 26/26 real BF16 safetensors shards** with exact selected-expert row
and unchanged-tensor bitwise comparisons. It recorded **16,266 exact
tensor/row checks**, all **120 routed tensor geometries**, the canonical
40×128 selection and the **37,981,137,632-byte** candidate tensor payload.
Individual per-shard immutable proof files reside in
`F:\reap-v2-equivalence-shards-20261025`.

An independent full-model finalization then rehashed **every live
original and candidate shard** and confirmed each byte digest against
the frozen source identities and shard certificates. The result was
`FULL_SOURCE_TENSOR_EQUIVALENCE_PASS`; the source weights and all
original observation/checkpoints were left untouched.

Complete receipt:
`F:\reap128-v2-full-source-equivalence-20261025.json`

Receipt SHA-256:
`0746c8c06764f784fa7134ecdc4a18954a761372f865256c9e0120588cfbbc7d`

This closes the source-identity, expert-selection, tensor-geometry
and exact BF16 construction-integrity requirements. It is **not**
an end-to-end inference, CUDA numerical-parity, quantized-quality,
GPU-fit, speed, or Gen-0 selection gate.

## October 25 actual 40-layer BF16 CPU inference result

After the complete source equivalence gate passed and an independently
admitted CPU-only host check (about 16.4 GiB RAM and 28.7 GiB Windows
commit available), the **real pruned checkpoint** executed all 40
decoder layers: **30 linear-attention** and **10 full-attention**. All
four immutable activation checkpoints (layers 0, 9, 19, and 29) were
saved on F:, and the model computed finite final logits with
**predicted next-token ID 9542**. Its final-logits SHA-256 is
`493e11204820af204b64947ea753efb712c177d51963569f8a55a5fbc51daae0`;
actual end-to-end elapsed time was **312.75 seconds**.

Source-bound inference receipt:
`docs/quals/reap128-v2-bf16-real-inference-20261025.json`
(identical to
`F:\reap128-v2-bf16-real-inference-20261025.json`).

Receipt SHA-256:
`a2ffe0e146e4bc75853e5c2d62b8ce4620a9ccd4e851dd21c69622f941e970b4`

Result:
`CPU_BF16_END_TO_END_SMOKE_PASS_NOT_QUALITY_GATE`.
This proves a bounded, genuine 40-layer BF16 model forward, but is
**not** calibrated perplexity, GSM8K/HumanEval, CUDA numerical parity,
quantized-model quality, Q4 fit/speed, or scientific selection.

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
   The `reap_streaming_smoke_supervised` guarded wrapper now optionally
   accepts `--progress-dir` (an existing directory outside both the
   original and candidate BF16 trees). At layers 0, 9, 19, and 29, the
   inference worker records an immutable BF16 activation snapshot plus a
   SHA-256 receipt bound to the frozen checkpoint, exact input tokens,
   model config and the completed full-source verification digest.
   On a later admitted run, the same prompt/config/proof can safely
   resume from the highest complete checkpoint without revisiting earlier
   layers. Any stale, corrupted or missing predecessor record fails
   closed. A positive **40-layer CPU synthetic test** confirmed resumed
   final-logits SHA-256 matches the uninterrupted model forward exactly.
   These intermediate snapshots are operational progress only; **no
   actual checkpoint inference PASS** is claimed until all 40 layers and
   final finite logits are independently witnessed. **The real 40-layer
   candidate has now satisfied that bounded inference smoke**; see
   source-bound actual receipt and metrics above.
3. After complete BF16 verification, `chowder_batch.reap_gguf_build_plan`
   checks the pinned llama.cpp converter, available storage and distinct
   target filesystems for BF16 GGUF intermediate, Q8_0 quality arm, and
   BF16-derived Q4_K_M fit arm. Planned storage: BF16 intermediate F:,
   Q8_0 output I:, Q4_K_M output G:. No imatrix, no MTP, no Q8→Q4
   requantization. Conversion must not race source construction.
   The guarded `reap_gguf_supervised` now supports explicit `--resume`
   for an interrupted later stage. Every completed BF16/Q8_0/Q4_K_M GGUF
   file has a new exclusive `.reap-stage.json` receipt, which binds the
   candidate full-equivalence proof SHA-256, pinned converter/quantizer,
   exact frozen commands, input GGUF digest, output size and output SHA-256.
   Resumption rereads the entire previously published GGUF to verify the
   sealed hash before reusing it. An existing output without the exact
   receipt, a changed recipe or a same-size changed payload fails closed.
   No stage may be reused based on magic bytes, filename or size alone.
   The newly pinned GGUF topology validator also checks the exact
   40×128 MoE tensor inventory and BF16/Q8_0/Q4_K_M stage file types
   (32/7/15) before publication; real gguf-py writer/reader fixtures
   passed these tests. The candidate's GGUF prebuild was independently
   admitted on F:/I:/G: with disk reserves, but that is not conversion.
   **Actual subsequent BF16 GGUF conversion succeeded:** the source-bound
   frozen converter published
   `F:\reap128-v2-bf16-intermediate-20261025.gguf`, **37,122,439,488
   bytes**, full-content SHA-256
   `80341aa776ad9b5f937c21b081b6d76de4a2ea277e5fb88c9ae2634a9da8f4c6`.
   Its production header check saw 733 tensors, 40 blocks, 128 experts
   per block, top-k 8, and all 162 mandatory MoE tensor shapes with BF16
   GGUF file type 32. Source/full-BF16 verification parent SHA is
   `0746c8c06764f784fa7134ecdc4a18954a761372f865256c9e0120588cfbbc7d`.
   Exact source-bound stage witness is versioned at
   `docs/quals/reap128-v2-bf16-gguf-stage-20261025.json`, SHA-256
   `8754ba965efd50ec7d07b1b2e63940c096540b29c5446fd07efdc1b187ebb361`.
   The inner topology receipt's `gguf_path` names the private staging
   file used for header verification, whereas the authoritative witness
   `output` is the published final path and its full SHA-256.
   No model quality or loading claim follows from the GGUF file.

   The first Q8_0 quantization launched from this BF16 file but its
   owned-child guard aborted near tensor 120/733 due to transient
   **physical RAM below the 7 GiB runtime floor**; this was neither
   a scientific failure nor a source-provenance mismatch.
   Its partial, unpublished Q8 private file was retained. Re-entry uses
   the verified source stage receipt and explicit
   `reap_gguf_supervised --resume`; original BF16 and the BF16 GGUF
   are not rebuilt.
   If Q8 repeatedly fails host admission, the explicit
   `--resume --only-q4` option can run frozen BF16→Q4_K_M
   independently from the same exact-BF16 source while marking Q8
   as skipped/unknown. It neither requantizes from Q8 nor promotes
   a partial GGUF build to a complete scientific qualification.
   This option must never run concurrently with another quantizer.
   After **two** Q8_0 attempts were independently aborted by the same
   unowned-physical-RAM safety floor (and neither published a file), the
   pinned llama.cpp's documented `--max-buffer-size` *row-slab memory*
   option was examined as a non-algorithmic resource workaround.
   The pinned source in `src/llama-quant.cpp` applies that value only
   to temporary per-tensor row-slab sizing; the original BF16 tensor
   source, Q8_0/Q4_K_M types, quant kernel, thread count, and absence of
   imatrix remain unchanged. A separate deterministic synthetic
   **BF16 Qwen35 MoE** with 2 layers, 16 experts and large 256×256
   expert rows verified that default buffering versus **1 MiB**
   memory-bounded slabs produced **bitwise identical complete GGUF
   outputs** for both Q8_0 and genuine Q4_K_M K-quant:
   Q8_0 SHA-256
   `4fad228106a6650fd32ddc97da895653ccb315bd6999ef682a10ae6819bd3887`;
   Q4_K_M SHA-256
   `605129350c854b104b64677d0fedb30fafb6d6539cd1327558fbfbea0782752b`.
   Versioned oracle receipt
   `docs/quals/reap128-v2-quant-buffer-equivalence-v3-20261025.json`,
   SHA-256
   `f9276dfed80571f949ed74925d7529c138a3b172238f53e0e3dc571261c2f333`.
   This validates *that synthetic input*, not an unexecuted production
   quantizer or a new scientific threshold. An explicit optional
   `--quant-buffer-mib` is now bound to each individual quantized-stage
   receipt as an operational parameter; resumption refuses a different
   value. It does not alter the full-source BF16 stage's frozen recipe.
   The second Q8 private partial (2,811,457,536 bytes) was likewise
   deleted only after the owned supervisor recorded the same physical
   RAM abort and no quantizer process remained. The independent audit
   `docs/quals/reap128-v2-incomplete-q8-cleanup-second-20261025.json`
   is byte pinned (SHA-256
   `d479f722881f61f339b815179b2286c80ec8430dece150514400cbc0e5db8e0c`).
   This restored I: disk headroom without touching any completed or
   published artifact.

   **Experimental split-based Q8_0 recovery (not yet production
   executed):** the pinned `llama-gguf-split` and `llama-quantize
   --keep-split` path on a deterministic BF16 Qwen35-MoE fixture
   produced a six-part Q8_0 and Q4_K_M merged GGUF whose entire tensor
   inventories and every tensor payload matched separate monolithic
   reference quantizations **bit-for-bit**, though the whole-file
   metadata and file SHA differ due to split headers. A second
   standalone-metadata experiment independently quantized three
   disjoint mini-GGUF tensor groups and merged them in source order;
   **all 10 Q8_0 tensors** again matched monolithic bytes. These two
   fixtures demonstrate a promising bounded-memory alternative but do
   **not** establish production equivalence, science, or deployability.
   `chowder_batch.reap_gguf_chunked` now implements a CPU-only source
   chunk-layout planner, exact raw BF16 mini-GGUF preservation, the
   pinned executable's per-mini quantization, per-chunk SHA receipts,
   guarded merging, cross-volume staged publication, and bitwise tensor
   validation of a merged Q8_0 GGUF. The default CLI is read-only
   (`READ_ONLY_CHUNK_PLAN_NOT_EXECUTED`); `--execute` additionally
   requires separately mounted BF16 scratch and Q8 output/chunk drives,
   a registered full 26-shard verification receipt, original BF16
   GGUF full SHA-256, exact pinned quantizer binary, an exclusive log,
   and fresh host RAM/commit/disk checks before launching each tiny
   quantization child. A **synthetic end-to-end positive test** confirms
   complete Q8_0 publication with all independent tensor bytes
   identical to monolithic quantization. None of this is a production
   Q8_0 evaluation or Gen-0 selection claim.

   The third monolithic Q8_0 attempt, including a 64 MiB buffer, still
   failed the same physical RAM floor around tensor 194/733. Its
   6,072,422,400-byte unpublished owned private output was audited and
   discarded; audit receipt:
   `docs/quals/reap128-v2-incomplete-q8-cleanup-third-20261025.json`,
   SHA-256
   `612a119e6c303ec1b10be94cbec70d62a4125a209b727c659aed92cfa58ac57a`.
   A separate BF16→Q4_K_M run with a 1 MiB row-slab cap similarly
   reached approximately tensor 150/733 before the same RAM guard
   refused further execution. Its incomplete 3,303,043,072-byte
   temporary was isolated and then audited for exact owned-process
   cleanup; audit:
   `docs/quals/reap128-v2-incomplete-q4-cleanup-20261025.json`,
   SHA-256
   `bf7c9021a3919543bce72ba3b4b7a538173316652557db829fd2d40365d3e0c9`.
   Neither abort is an indication of scientific quantized-model
   quality; no valid Q8_0/Q4_K_M GGUF or frozen-Gate PASS resulted.
   Before the resumed quantizer launched, the exact abandoned private
   file (4,206,747,648 bytes) was removed after checking the original
   owned-child abort log and absence of an active quantizer, restoring
   roughly 3.92 GiB of I: space. The full deletion/ownership audit is
   frozen at
   `docs/quals/reap128-v2-incomplete-q8-cleanup-20261025.json`
   (SHA-256
   `633a669c35d9ded407b8b326538df8bcb5f29eb3ddb98b2777320209e8f5ec16`).
   No valid GGUF, pinned original weight, checkpoint, or receipt was
   removed. The previously quoted incomplete-partial retention describes
   the initial guard outcome, not its later audited cleanup.

   A bounded CPU-only `reap_gguf_quant_sample` probe is available
   after successful Q8_0 publication. It measures BF16 versus Q8_0
   dequantized weight differences at four deterministic 256-value
   positions in each of 120 routed expert tensors, binding both stage
   receipts without allocating whole tensors. This is descriptive
   quantization evidence, **not** preregistered quality qualification.
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

## Preserved historical evaluation and clean v2 workspace

An **independent** copy of the SHA-attested historical baselines was
atomically created at
`C:\llm-work\reap\eval-v2-isolated-20261025` using
`chowder_batch.reap_eval_workspace`. The new workspace contains 26 small
frozen reference/serve/evaluation files for `dense`, `dense_q4`,
`random128`, `reap128`, `reap128_q4`, `reap160`, and
`reap160_q4`, each checked byte-for-byte against its original. The
workspace includes `frozen-historical-baselines.json` with SHA-256
`06c1b562fd9e655b464b82ef52fe60d0d971f9314cd40cac38485069fcdf5463`.
The original `C:\llm-work\reap\eval` is unchanged.
The canonical selector was executed on the copied root and returned the
same intentional `BLOCKED` decision with only historical REAP128/REAP160
present. Future v2 scoring can use this isolated root with independently
witnessed `reap128_v2` / `reap128_v2_q4` directories; historical
artifacts are never resumed or appended in place.

**Release decision until all required evidence exists: BLOCKED, no Gen-0
promotion or owner approval implied.** Neither frozen thresholds nor the
predeclared arm-selection order may be weakened.
