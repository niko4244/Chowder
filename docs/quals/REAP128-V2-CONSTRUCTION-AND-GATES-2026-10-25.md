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

## October 25 serial chunked Q8 recovery and independent Q4 readiness

The first actual serial 42-chunk Q8 build sealed source-bound chunks 001 and
002, including their full SHA receipts on I:, before a *validation error* in
chunk 003. This was not a quantizer numerical failure: the pinned llama.cpp
quantizer writes two FP32 `blk.{0,1}.ffn_gate_inp_shexp.weight` tensors as
shape `[2048]` instead of the BF16 GGUF source metadata `[2048, 1]` while
preserving the original FP32 payload bit-for-bit. Independent live GGUF
reader checks confirmed precisely two shape changes among chunk 003's 39
tensors, with identical 8,192-byte payloads for each affected tensor.

The converter's strict identity check now permits *only* the removable
trailing singleton dimension of copied F32 tensors, and requires that their
complete raw data SHA match. Other shape/type/data changes remain refused.
An OS-held, nonblocking, persistent-inode ownership lock guards future Q8
resumes, so two same-target jobs cannot overlap and an interrupted owner's
lock is released by the OS without removing the common lock file.

The sole unsealed chunk 003 output and its private BF16 mini-source were
independently audited against the exact original tensor group and SHA-256
before *only those two owner-created files* were discarded, with no owning
quantizer still running. The two sealed chunk receipts remain unchanged.
Audited orphan Q8 SHA-256:
`1f21282c20e0e93f9a2cab7318e5b4a2f4020c1f581c33f448555682a0e0c774`.
Mini-source SHA-256:
`076d42cd9e5ac1406ab74d80c9af8a98005664bf3afa220cec40b70efe7f12f3`.
Audited cleanup JSON:
`docs/quals/reap128-v2-q8-orphan-chunk003-audit-20261025.json`, SHA-256
`051ccbd347609e20b493464a1cd9900ed8834b0d660d45f621ab1c536f81bf34`.

A fresh serial Q8 recovery was launched against the unchanged published
37.12-GB BF16 source with the identical chunk identity and a new audit log
`F:\reap128-v2-q8-streaming-recovery1-20261025.log`. Its observed startup
is **not** completion of the full Q8, quantized model loading, registered
GSM8K/HumanEval, or GPU gates. Chunks are certified only after their own
receipt is sealed; no incomplete source is promoted.

For Q4_K_M, `chowder_batch.reap_gguf_q4_chunked` now provides a separate
CPU-only, BF16-source-bound, guarded sequential mini-GGUF workflow, including
per-mini and final bitwise tensor hashes, typed K-quant metadata, parent
source equivalence, protected cross-volume publication and restart refusal
for unsealed artifacts. The first pinned real-quantizer tiny MoE fixture
verified that all Q4_K_M per-tensor types, shapes, and data match the
monolithic tiny BF16 reference. **The real 40×128 production Q4 is not yet
built or measured** and must not run concurrently with the Q8 constructor.
The registered Q4 speed, fit and full scientific gates remain pending.

## October 25 evaluation evidence-integrity regression fixes

- The actual future **chunked Q8 final** has a distinct
  `.reap-chunked.json` construction schema, not a monolithic
  `.reap-stage.json`. `reap_gguf_quant_sample` now accepts that exact
  construction witness only after validating registered BF16 ancestry,
  quantizer/spec hashes, entire Q8 file checksum, 733 unique tensor
  headers, 42 ordered source groups, and every quantified raw tensor
  hash/shape/type. It refuses conflicting or contradictory receipts.
  It does not turn weight-error windows into benchmark quality.
- `reap_gates` now rejects 200-speed-sample cohorts that are not the
  **first 200 frozen GSM8K IDs on both sides**. It also makes Q4
  speed/fit `UNKNOWN` if a resumed serving record/log or an active
  serving lock exists, rather than using a stale original offload count
  for measurements from an unbound later session. REAP160's historic
  independent load-only GPU probe continues to be corroborating, not
  eligible as a same-session speed PASS.
- `reap_eval` now preserves and revalidates earlier suite summaries,
  item results, input identity, scores and SHA when another frozen
  suite is run later. A cooperative exclusive summary lock and atomic
  replacement stop legitimate resumed evaluators from losing prior
  GSM8K evidence upon a HumanEval-only re-entry. Corrupt, orphaned,
  mismatched or stale previous records fail closed.
- `reap_select` requires newly produced v2 quality/200-item GGUF arms
  to carry matching model/serving SHA, suite-specific `.inputs.json`
  and frozen prompt/decode/scoring/source-population fingerprints.
  V2 Q8 and Q4 GSM8K source SHA/count must agree. Historical archived
  reference arms remain separately grandfathered; their missing new
  witnesses never substitute for v2 data. Receipt-level witness
  integrity cannot prove an independently rebuilt upstream dataset
  or live resident model process against *collusive* forged receipts;
  those stronger external attestations remain separate requirements.

The focused post-change CPU regression matrix passed **198 tests** with
**3 intentional skips** of tiny real quantizer tests while the production
Q8 owner was running. The skipped pinned-quantizer miniature proofs had
passed before the additional concurrent-conversion protection was
installed. Ruff and Git whitespace checks passed. This is regression
evidence, not model-quality validation.

A separate real archived-evaluation selector replay (not a new quality
run) wrote
`F:\reap128-v2-frozen-post-evidence-integrity-20261025.json`
(byte-identical repository evidence at
`docs/quals/reap128-v2-frozen-post-evidence-integrity-20261025.json`)
with SHA-256
`82f3583eebd0436d07cc13739068ececfd2d708f2e39bde3c025253f03a0af46`.
The unchanged frozen decision was exactly
`BLOCKED`, `chosen=null`, reason **both REAP128_v2 and REAP160 full
gates are required**. The PR remains a draft; release approval is absent.

## Frozen scoring-source fingerprint and interrupted-finalization recovery

The registered original offline Hugging Face benchmark loaders were rerun
*locally, CPU only*, without model inference, download, GPU use, or
change to scoring/decoding. The exact rendered full source populations are:

- `openai/gsm8k` `main/test`: **1,319** items,
  `_digest(_load('gsm8k'))` SHA-256
  `913ca81f724cba16a07099608047e7522b00910c1c7fe13499c501ac9e4bad89`;
  item IDs `gsm8k-0` through `gsm8k-1318`.
- `openai/openai_humaneval` `test`: **164** items,
  `_digest(_load('humaneval'))` SHA-256
  `49fb688c6788b0531803dfd1cb85a69926ec5a7179e74cdfffc53ba52a92b420`;
  item IDs `HumanEval/0` through `HumanEval/163`.

Exact file/renderer and scope:
`docs/quals/reap-frozen-scoring-source-fingerprints-20261025.json`,
SHA-256 `62d32cf168712b298159e3f33803acd80b4f86cb244030a9adfee48ab951da00`.
This locally cached population proof is stronger than accepting two matching
claimed v2 hashes but is **not** independent external Hub version verification,
model evaluation or authority to modify preregistration.

An interruption after a **finished and independently sealed F: merged Q8
GGUF** can strand the original Q8 constructor because verified I: chunk
payloads are intentionally removed before final cross-volume publication.
The new `chowder_batch.reap_gguf_chunked_recover` path does **not**
requantize, reconstruct missing tensors, override an existing complete
receipt, delete another process's files, or run during its active owner.
It requires the exclusive Windows file lock, exact registered BF16 GGUF
and 26-shard proof SHA, frozen quantizer identity, unique merged F: assembly
and its SHA, all original per-chunk input/output tensor certificates,
explicit deletion-log events for absent I: parts, full 733-tensor model
geometry and raw payload hashes, and exact rehashed cross-volume publication.
Its default is read-only `RECOVERY_PREFLIGHT_PASS_NOT_PUBLISHED`; explicit
`--execute` publishes only an already proved F: assembly under fresh
resource admission, leaving scientific gates `NOT_EVALUATED`.
No emergency recovery was invoked while the legitimate Q8 supervisor ran.

A synthetic actual-GGUF fixture exercised 733 headers (including the
pinned quantizer's byte-identical FP32 `[2048,1]` → `[2048]` normalization),
seven adversarial missing/forged receipt variants, and a checksum-verified
cross-volume publication simulation without a quantizer or GPU child.
**34 CPU-only Q8 recovery/numerical tests passed**, including the numerical
sampler's corrected treatment of copied FP32 singleton dimensions. Production
GGUF and quality evidence remain distinct from these tests.

### Live GPU/CUDA qualification admission refusal

The original frozen CUDA host guard was read *passively* with no model
load, model-server connection, GPU context or service changes. At the
October 25 read it reported **12.673 GiB physical RAM free** (admission
requires at least 16 GiB), **25.731 GiB Windows commit free** (at least
24 GiB), **10.890 GiB GPU memory free** (at least 12 GiB), and live
protected Ollama/llama.cpp serving ports. Each of the RAM, GPU and
protected-service conditions independently prohibits launching the
registered original-BF16 CUDA numerical parity run. This was one
snapshot only and was never represented as a full 60-second admission.

Evidence:
`docs/quals/reap128-v2-cuda-host-static-refusal-20261025.json`,
SHA-256
`27a3b6cded0404977f4728296c88956767c89ddbf1004ff0abb09f3ce95bf10f`.
Therefore **CUDA numerical parity remains NOT_MEASURED**, and GPU fit,
Q4 200-item same-session performance and REAP160 repeat placement
cannot be certified by this snapshot. Original service ownership and
scientific thresholds are not relaxed to bypass these limits.

### Integrated Q4 recovery and serving evidence integrity

The Q4_K_M workflow now seals an atomic, immutable finalization ledger
before deleting any certified chunk. It independently rechecks Q4_K, Q5_K,
Q6_K, retained FP32 and other allowed tensor types, source/input proofs,
quantizer command, finished merge SHA and exact raw payloads. Recovery
from a partial cross-volume transfer checks its byte prefix before resuming;
crashes after final rename but before receipt are also recoverable. Original
BF16 weights and approved model services remain untouched. Production Q4
quantization must still be launched serially after Q8 termination and fresh
resource admission; synthetic construction tests are not a quality gate.

The Q4 serving source receipt now withholds claimed offload PASS until a
fresh same-session, witnessed first-200 GSM8K cohort has finite per-row
throughput, the exact startup server-log prefix SHA, serving-model digest,
frozen input witness, per-item output and summary hashes. The speed-gate
consumer revalidates the source/flags/offload/log/rows and refuses stale,
resumed, forged or absent certificates for newly built v2 Q4 arms.
The selector pins the official locally cached GSM8K/HumanEval population
fingerprints above for new v2 arms. Historical v1 and REAP160 evidence
remains separately assessed; all frozen thresholds and Amendment4 ordering
are unchanged. GSM8K/HumanEval multi-suite re-entry retains, rechecks and
atomically republishes each independently certified suite without silently
discarding the other.

The integrated CPU-only construction/eval/gate/selection/serving/parity
preflight regression matrix completed **295 passed, 3 intentionally
skipped** after streaming merge and retry-manifest regressions, with
Ruff clean and no Git whitespace errors. No model-quality
PASS or GPU placement is inferred from these regression tests.

### October 25 11:23 — guarded Q8 merger memory-floor interruption

The exclusive original Q8 constructor finished and independently SHA-sealed
all **42/42** source-bound mini-GGUF chunks on I:, covering exactly **733
unique tensor names**. Its owned Python merge worker began full assembly
but the unchanged runtime physical-memory floor triggered the monitored
abort: `OWNED_CONVERSION_ABORT: insufficient unowned physical RAM`.
Supervisor and worker exited. The 42 previously sealed GGUF chunks and
receipts remain intact, with their saved file sizes and source identities
rechecked after the abort. **No F: merged assembly or I: final Q8 model
exists; there is no Q8 numerical or quality PASS.** This was an
admission/supervision abort, not an established GGUF payload mismatch.

The new immutable post-abort witness is
`docs/quals/reap128-v2-q8-merge-memory-guard-abort-20261025.json`, SHA-256
`72c95cce6eaef7a3d7dff37e7ad72b2fa54e2ac691a6f9e00b43a2435e001e29`.
Its 42 file/receipt size and identity checks **do not claim a fresh full
20-GB chunk payload rehash**; the original verified per-part digests are
stored, and any admitted exclusive restart must rehash them before reuse.
No identical merge was relaunched; the corrective work is a one-reader,
bounded-memory streaming GGUF assembly with synthetic byte-for-byte and
adversarial regressions, subject to the same unchanged safety floors.

### Bounded-memory merger fix and exclusive resumed verification

The Q8 merger no longer retains all 42 tokenization-heavy GGUFReader
objects, tensor ndarray views and mmap bases. It reads each source part
individually and saves only primitive file offsets, raw lengths, data
shapes/types and tensor identities. In the frozen BF16 original tensor
order it writes the canonical GGUF header/KV/tensor-info, then copies
the registered Q8/F32 raw payloads in bounded chunks, rechecking their
SHA-256 during copy and reproducing the original GGUF alignment. The
finished output is reread independently to validate all tensor identities,
types, shapes and raw SHA, before the old serial parent can seal/publish.
An unexpected orphan F: merged output or merge receipt now fails closed
before duplicate final-assembly work, preserving abandoned bytes for audit.

Synthetic RED proved the prior implementation still held one reader when
opening the next part. GREEN tests demonstrate no prior reader alive,
bounded reads, mixed Q8/F32 exact *whole-file byte* identity with the
previous GGUFWriter route, tamper refusal and orphan refusal. After the
first merge supervisor stopped, the original pinned tiny **real**
Q8_0 and Q4_K_M llama-quantize tests were admitted and rerun without any
GPU/model service changes: **25 passed**, Ruff clean. Those tiny oracle
proofs do not constitute full production model numerical/quality results.

The new serial production resume was launched with the **same 42 sealed
chunks and pinned source** after an independently observed resource
admission of 18.92 GiB physical RAM and 28.98 GiB Windows commit free,
with no competing quantization owner. Its new exclusive audit log is
`F:\reap128-v2-q8-streaming-recovery2-bounded-merge-20261025.log`.
The original >=7 GiB physical-runtime safety floor is unchanged; a
freshly launched/ongoing assembly is **NOT** equivalent to final Q8
publication, full GGUF SHA, verified 733-tensor output or scientific PASS.

The separate **post-merge** emergency recovery reader also now accepts
multiple old immutable F: `merge-input-*.json` rosters after guarded
merge attempts, **only if every roster exactly equals the original 733
tensor SHA manifest reconstructed from all 42 source-bound receipts**.
It refuses a single mismatching/forged historical roster. This closes a
separate crash-recovery hole in which an aborted earlier merge left one
small manifest, a later complete merge left a second identical manifest,
and the emergency recovery path incorrectly required exactly one.
Adversarial synthetic tests passed without touching the live assembly.

### Bounded-memory Q4_K_M merger before production dispatch

Independent code review found the same all-part `GGUFReader`/mmap lifetime
amplification in Q4 assembly. The Q4 streaming merger now keeps only
primitive source-bound part offsets, sizes and GGML types while inspecting
one source file at a time; it emits the canonical type-15 GGUF tensor
table in original BF16 order, streams Q4_K/Q5_K/Q6_K and retained types
in bounded byte blocks, SHA-verifies each raw payload during copy, and
independently rereads the entire output for tensor/type/shape/hash
integrity. It requires lossless original FP32 bytes through the pinned
single-unit shape normalization. The existing crash-durable Q4 ledger,
partial transfer recovery and safety guards remain unchanged.

Synthetic tests reproduced prior GGUFReader retention RED and proved
exact *whole-file* compatibility with the historical writer for mixed
Q4_K/Q5_K/Q6_K/F32 tensors after streaming GREEN. Adversarial payload
tamper and no-retained-reader tests pass. The scoped Q4/recovery CPU matrix
passed **29 tests, with one deliberate pinned-quantizer skip** while
the exclusive production Q8 merge was active. Full original 40x128 Q4
output is **NOT BUILT** and remains serially gated behind completed Q8.

### Second guarded Q8 merge and independent orphan reattestation

The bounded streaming merger verified the original 42 Q8 chunks and
finished writing a **19,748,435,968-byte** GGUF F: assembly at
`F:\reap-v2-q8-streaming-scratch-20261025\merged-q8-720fdec6b18c4ce588b530745113645b.gguf`.
Its last verification step, however, was still hashing the entire
18.392-GiB model via memory-mapped ndarray views. Its resident memory
increased until the unchanged **7 GiB host free-memory guard** correctly
aborted the exclusive worker before any final F: receipt was sealed.
All **42/42 original SHA-sealed I: chunks** and the complete F: assembly
were preserved; no I: Q8 final output exists.

The merger's full-model validator now uses GGUFReader **metadata only**
and rehashes each tensor by bounded file reads at its registered offset,
without retaining mapped data pages in the worker's resident memory.
Separately, `reap_gguf_chunked_recover` now supports an explicitly
distinguished **independent complete-orphan certification**: source-BF16
and 26-shard ancestry, all 42 surviving chunk file and tensor hashes,
the failed owner's RAM-abort log, all 733 merged tensor types/shapes/raw
hashes, frozen 40x128 metadata and full F: SHA must match. Its
`--seal-unsealed-orphan-only --execute` mode issues an exclusively
created **F: independent recovery receipt**, clearly recording that
the original worker never sealed the file; it never deletes original
chunks or publishes I:. The ordinary Q8 supervisor can later reuse
only that fully verified source-bound assembly, avoiding another giant
quantizer/merge run. Unknown or forged F: witnesses fail closed.

Synthetic complete 733-tensor orphan reattestation, lossless FP32 fold,
direct buffered span hashes, missing original chunk and tampered data
refusal, duplicate receipt refusal and F: seal-only semantics were
tested **15 passing CPU regressions** (Ruff clean). The admitted *real*
18.39-GiB F: certification was launched independently against its
published BF16 parent and original sealed chunk receipts. Its actual
full-byte/733-tensor result is **not yet verified** while that process
continues, and no scientific gates may be promoted from it.

An independently invoked metadata-only
`reap_gguf_header_preflight --expected-file-type 7` on the complete-size
F: assembly returned `GGUF_METADATA_TOPOLOGY_MATCH_ONLY`: **19,748,435,968
file bytes; 733 GGUF tensor headers; 40 decoder blocks; 128 experts per
block; routed top-k 8; 162 required shape checks; GGUF file type 7**.
No GGUF full-file SHA was computed or claimed by this *header-only*
instrument, and it does **not** attest source raw tensor payloads,
actual llama.cpp load, GSM8K/HumanEval or deployment fit. Immutable
header receipt: `docs/quals/reap128-v2-q8-F-orphan-header-only-20261025.json`
(SHA-256 `80b8122673c4604f5058961e86df53577b706a73833b648d40cebaffc4ceb1e1`).

### October 25 12:44 — independently certified full F: assembly

The guarded F: complete-orphan certifier **completed successfully** after
independently rechecking the original BF16 and 26-shard provenance,
all **42/42 certified Q8 chunks**, 733 BF16/quantized raw tensor
signatures, 40x128/top-k 8 topology and the complete F: model SHA.
The F: merged GGUF is **19,748,435,968 bytes**, SHA-256
`89dcaf33b61f59c83a31b718ba98b73c33437d0320ae0cc56737a5d7e43519c6`.
The immutable separately labeled *independent* F: certificate is
`docs/quals/reap128-v2-q8-F-full-independent-tensor-cert-20261025.json`,
receipt SHA-256
`22145b94628396a288e7f301df1d7217436c0605b38cb60943b24fc84e3545d3`.
Its supervision-log SHA independently matches the real guarded
second-attempt RAM-abort log; it does not forge an original-worker seal.

An exclusive original publisher was subsequently admitted and started
to reuse and rehash this independently certified F: assembly. The **final
I: GGUF and sidecar are NOT PUBLISHED** until its source-bound cleanup,
cross-volume copy and output SHA fully complete. All frozen quality,
speed, GPU and Gen-0 gates remain unmeasured/blocked separately from
construction. The Q8 numerical sampler now verifies final Q8 GGUF raw
tensor SHA through bounded file range reads instead of mmap ndarray
payload hashing, to avoid repeating the physical-memory guard abort;
**24 sampler CPU tests passed**, including adversarial mmap-page refusal.

### Q4 complete-file verifier hardening while Q8 publisher owns conversion

The original Q8 final-memory abort also exposed two Q4 full-assembly
verification paths that still faulted the *entire* output GGUF tensor
arrays into the worker's resident memory. Both `merge_q4` final reread
and `_validate_assembled_q4` (crash-durable ledger/reattest consumer)
now verify every raw tensor SHA using the existing bounded
`core.hash_tensor_span()` file-offset reader. GGUFReader is used only
for model and tensor metadata; no threshold, quantization type,
source lineage or frozen gate changes. Adversarial synthetic tests prove
the independent assembled-model proof refuses mapped quantized payload
hashing and uses exactly one buffered span per witnessed output tensor.
With the live Q8 owner still present, **34 Q4/recovery CPU tests passed,
one pinned real quantizer test was deliberately skipped**, Ruff clean.
Full Q4_K_M remains **NOT EXECUTED** pending final Q8 publication and
fresh exclusive host admission.

### Guarded Q8 publication, audited I: cleanup and interrupted-copy recovery

The original guarded Q8 publisher independently reverified and admitted
the sealed F: assembly with event
`EXACT_ORPHAN_ASSEMBLY_REATTESTED_AND_REUSED`, retaining the registered
full SHA `89dcaf33b61f59c83a31b718ba98b73c33437d0320ae0cc56737a5d7e43519c6`.
It then individually rehashed and deleted all **42/42** original Q8 mini
GGUF payload files on I:, emitting SHA-bound
`EXACT_SOURCE_BOUND_PART_CLEANED` for each. **All 42 source-bound chunk
receipt JSON files remain intact**. The original full-size F: certified
GGUF has not been modified. Its I: private cross-volume copy reached the
expected **19,748,435,968** bytes, but final I: public naming/receipt
must not be claimed until an independently verified full SHA and atomic
publication finish.

An additional **restart-only, no-quantizer emergency Q8 recovery path**
now protects this vulnerable transfer window: after acquiring the original
exclusive chunk lock, independently rechecking BF16 ancestry, all 42
individual deletion-log entries, source-tensor witnesses and the sealed F:
GGUF, the recovery method recognizes at most one same-volume source-owner
`.chunked-owned-<uuid>.gguf` or prior `.recovery-owned-<uuid>.gguf` copy.
Every existing transfer-prefix byte is compared against the certified F:
model in bounded blocks before appending only the missing suffix. Mismatched,
untrusted, duplicate, oversized or symlink candidates fail closed, as does
insufficient remaining destination reserve or the original RAM/commit guard.
Only full SHA equality permits final rename and exclusive receipt.
Tests: **18 recovery CPU cases passed**, including exact-prefix recovery
for both registered owned names, corrupt-prefix refusal, duplicate candidate
refusal and guard-abort preservation. No new model-quality gate is asserted.

### October 25 13:19 — full Q8_0 published on I: with final 733-tensor receipt

**Production Q8 GGUF publication PASS (construction only).** The original
exclusive guarded publisher rechecked the sealed F: assembly, emitted all
**42/42 SHA-bound chunk-cleanup events**, completed the guarded I:
cross-volume transfer, verified the final copied SHA and issued
`CHUNKED_Q8_FINAL_PUBLISHED` for model
`I:\reap128-v2-q8_0-20261025.gguf`, **19,748,435,968 bytes**, SHA-256
`89dcaf33b61f59c83a31b718ba98b73c33437d0320ae0cc56737a5d7e43519c6`.
The published I: `.reap-chunked.json` sidecar records the frozen BF16
parent SHA, fixed pinned quantizer, **42 distinct original groups and 733
unique raw tensor receipt names**, and explicitly states
`quality_gate=NOT_EVALUATED`, `gpu_fit_gate=NOT_EVALUATED`.
An independently exact-byte-copied immutable construction sidecar is
`docs/quals/reap128-v2-q8-I-final-construction-receipt-20261025.json`,
receipt SHA-256
`a1cfe1464df4579ea9d5ae1320efc01c0de5dae1070235caca18c821ce67479d`.
The final model's SHA was certified by the guarded publisher; independent
Q8-vs-BF16 numerical window testing and external scientific evaluation
still require their **own** measurements and cannot be inferred from this
construction receipt. The publisher subsequently **exited code 0**, after
logging `EXACT_ASSEMBLED_TEMP_CLEANED` against the same full-model SHA,
releasing the verified F: temporary GGUF and recovering **37.52 GiB free**
on F:. Original Q8 constructor ownership is fully cleared.

### Independently guarded BF16-to-Q8 numerical window measurement

The CPU-only `chowder_batch.reap_gguf_quant_sample --execute` was launched
only after the full I: Q8 GGUF and sidecar were published and the original
Q8 supervisor exited. It uses the existing `_owned_stage` resource guard
and registered published BF16/Q8 SHA witnesses, before comparing four
deterministic 256-value windows in each of three routed expert tensors
through all 40 decoder layers. Admission recorded **12.23 GiB available
physical RAM and 20.13 GiB free Windows commit**, CUDA disabled and no
competing quantizer. Child PID **32748**, audit log
`F:\reap128-v2-q8-numeric-supervised-20261025.log`, intended report
`F:\reap128-v2-q8-bf16-numeric-sample-20261025.json`.
This real numerical test **completed with guarded exit 0**. The independent
source-bound result is **120/120 expert tensors**, four 256-value windows
per tensor, **122,880 actual BF16-vs-Q8 dequantized weights**, maximum
absolute error **0.00028896331787109375**, mean absolute error
**0.0000478359017317113**, root mean squared error
**0.000058366718970777665**. Full result source SHA and final quantized
GGUF SHA match immutable construction ancestry. Receipt copied byte-exact
to `docs/quals/reap128-v2-q8-bf16-numeric-sample-20261025.json` with
SHA-256 `a7d236af0fdad53b65f4b2271f895a05b14f84653bf40efef52c00aa18e0920c`.
This is a **DESCRIPTIVE_QUANTIZATION_SAMPLE_NOT_QUALITY_GATE**; neither
GSM8K nor HumanEval, original CUDA, model-fit or speed is measured here.

Full Q4_K_M had been initially refused at 10.99 GiB available RAM, below
the original **12 GiB minimum start threshold**. A later transient 12+
GiB snapshot coincided with the guarded Q8 numeric child; Q4 cannot run
concurrently, and no unrelated live Ollama/llama.cpp service was stopped.
At the next post-audit snapshot (13:38:50 CDT), physical RAM was **14.07
GiB**, free Windows commit **16.92 GiB**, and F/G/I free space
**37.52/33.90/15.95 GiB**. A PowerShell command attempt failed argument
parsing before any Q4 side effects, and subsequent admission fell to
**15.25 GiB** free commit, below the unchanged 16-GiB start floor.
Full production Q4 therefore remained **NOT STARTED**, and unrelated
services were preserved.

### Source-bound Q4 production admission and independent fidelity verifier

After Q8 numerical sampling exited and its results were SHA-sealed,
original Q4 construction resource admission recovered to **15.95 GiB
physical RAM and 18.36 GiB available Windows commit**. The code's own
`resource_problems(..., initial=True)` and exclusive process census
passed; separate F: scratch and G: chunk directories were both empty.
Original `chowder_batch.reap_gguf_q4_chunked --execute --max-mib 1024
--slab-mib 1` CPU-only production started from the registered
`F:\reap128-v2-bf16-intermediate-20261025.gguf` parent with the pinned
quantizer. **This is a guarded start, not a Q4 publication claim**;
first stage rereads complete parent BF16 SHA before constructing chunks.
At 13:49 CDT owner PID **81772** had read **24.85 GiB** of the frozen
source, and no Q4 chunk output or receipt existed; host RAM and free
commit were **16.12/22.95 GiB**. All unrelated GPU services remained.

Independent companion `chowder_batch/reap_gguf_q4_numeric_sample.py`
now validates only a published `Q4_K_M_TENSOR_EXACT_CONSTRUCTED_...`
GGUF with original full BF16 stage SHA and frozen stage/quantizer source
identity. It rehashes the *entire* Q4 output and **all 733 tensor raw
file spans** against the final receipt, checks full 40x128 top-k 8
geometry and canonical source order, and checks copied BF16/F32 tensors
still match original bytes. Separate deterministic 256-value windows
dequantize actual Q4_K, Q5_K, Q6_K and Q8_0 expert block types to
measure absolute/RMS deviations from original BF16 values. A produced
receipt always remains descriptive and **NOT A QUALITY GATE**. Synthetic
independent CPU suite **11 passed**, including all 733 mixed-type
headers, forged source/output/quantizer/tensor payload/selected-gate
refusal and resealed copied F32 mutation refusal; Ruff clean. **No real
Q4 numeric results can be reported until the full Q4 construction exits
and is independently verified.**

### Frozen Q4 guarded two-chunk recovery and unchanged floor admission wait

The first production Q4 run SHA-verified the full BF16 parent, then
successfully produced and SHA-sealed the first **2 of 42** chunks before
the third new chunk was started. Chunk 001: Q4 GGUF SHA-256
`6a457d91a36b07e883dbb1dda241a9f546eeb964741c023bae87f6e45089308c`
(3 tensors including actual Q4_K attention-gate and Q6_K embedding/QKV);
chunk 002: Q4 GGUF SHA-256
`6213637edc0b0ffb8b4b3c915b28a1dfb2398cfc1a65a6739a5f3b89ca42628d`
(7 mixed-K tensors including routed FFN 128 experts).

The parent refused its next per-chunk start admission with
`Q4 prelaunch guard or unsealed mini refused`, **exit 2**. Independent
post-exit inspection found **no active Q4/quantizer owner**, **no orphan
F: source mini**, and exactly two matching immutable G: chunk GGUFs and
receipts; the OS-level `.reap-gguf-chunked.lock` inode remains durable.
The host's RAM/commit briefly dropped below the unchanged **12/16 GiB**
new-chunk launch floor, while neither the **7/8 GiB runtime floor** nor
any original source/model service was reduced or modified.

To avoid another full 37-GiB source-hash restart on every transient
RAM/commit fluctuation, the exclusive Q4 supervisor now waits for a
bounded interval on **the exact same `resource_problems(initial=True)`
thresholds** before starting an additional mini. It polls the competing
quantizer/process census each time, writes sparse guarded-wait/recovered
events, immediately rejects permanent disk-floor problems, and fails
closed after the finite admission deadline. Abandoned source minis still
fail immediately, rather than being silently overwritten. CPU regression
**51 passed, one real quantizer regression deliberately deselected**,
Ruff and whitespace checks clean. The supervisor restarted with the
same frozen parent SHA, recipe, quantizer, chunk topology and slab,
under fresh recovered 12/16 GiB admission, using a new exclusive log at
`F:\reap128-v2-q4-streaming-resume2-20261025.log`. It must revalidate
the existing 2 original chunk receipts before any further conversion.
**No full Q4 GGUF/quality-gate completion is claimed by this recovery.**

### October 25 14:46 CDT — seven certified Q4 chunks, prolonged admission refusal

The first resumed exclusive Q4 owner advanced production to **7/42**
SHA-sealed Q4 chunk GGUFs, each with its intact per-chunk immutable
receipt on G:. The next per-chunk start was correctly refused after the
bounded ten-minute resource wait reached its deadline while host RAM and
Windows free commit remained below **unchanged 12/16 GiB initial floors**.
The original owner exited code 2; independent live process census found
no competing quantizer or Q4 owner. All seven already-certified chunk
files and receipts remain intact; the final full Q4 GGUF is **not built**.
At recovery inspection the host independently had **16.30 GiB free RAM,
23.44 GiB free Windows commit**, and F/G/I free space
**37.52/31.83/15.95 GiB**. The prior timeout lost no certified tensor
work, but required another expensive original BF16 identity pass.

The next finite per-chunk admission wait is now **3600 seconds instead
of 600 seconds** under the same original process owner, unchanged
`resource_problems(initial=True)` and untouched 12/16 GiB entry,
7/8 GiB runtime and 6 GiB disk floors. It holds the exclusive lock,
launches no quantizer until admission passes, checks competing-process
ownership throughout, and continues to refuse disk breaches, modified
minis or uninspectable owners immediately. New CPU-only regression
matrix: **33 passed, one real quantizer case intentionally deselected**,
Ruff clean. No GSM8K/HumanEval, CUDA parity, fit/speed, REAP160 or Gen-0
status is changed by this production recovery.

### October 25 15:05 CDT — safe owner-only mmap cache reclamation

Third exclusive Q4 supervisor resumed from seven immutable GGUF and
receipt pairs after a fresh full BF16 parent SHA recheck. Audit log:
`F:\reap128-v2-q4-streaming-resume3-20261025.log`. It independently
emitted `EXACT_Q4_CHUNK_REUSED` for original indices 0 through 6 but
again entered `Q4_CHUNK_RESOURCE_WAIT` before part eight. Independent
host inspection found that the **owned** Python parent retained about
**6.766 GiB resident working set** from read-only source GGUF mmap
pages, leaving only **9.578 GiB available physical RAM**, below the
untouched 12-GiB start threshold. No external service was touched.

One independently identified Windows-owned PID 57764 process (the exact
Q4 restart command, with no active quantizer child) was trimmed using
Windows `EmptyWorkingSet` on that **single owned process**. Its resident
working set dropped from **6.766 GiB to 0.094 GiB**, restoring available
physical RAM from **9.578 GiB to 15.475 GiB** without stopping Python,
invalidating a mapped file, lowering a guard or touching unrelated
Ollama/llama.cpp GPU servers. The live original supervisor then logged
`Q4_CHUNK_ADMISSION_RECOVERED` after 31 polling intervals and proceeded
to genuine next-chunk quantization. No model-quality claim follows.

For future restarts, the source-side Q4 code now permits **one** automatic
`EmptyWorkingSet(GetCurrentProcess())` per failed per-chunk RAM/commit
admission. It exclusively trims the executing Python owner, never opens
or modifies external process handles, and *immediately re-evaluates the
same* 12-GiB RAM/16-GiB commit/6-GiB disk start thresholds. If self-trim
is unavailable or cannot restore admission, the finite guarded wait
continues unchanged; disk failures still fail closed immediately.
New adversarial regression covers one-shot trim/recheck, original
resource-floor enforcement, bounded timeout, external process refusal,
and non-Windows no-op. The currently running Python process imported
the prepatch code at launch; its successful owned trim was performed
separately against the exact verified PID, not hot-reloaded into it.

The recovered original owner subsequently SHA-sealed **chunk 008/042**,
with 10 mixed Q4_K/F32 tensors, status
`CHUNK_Q4_K_M_BITWISE_WITNESSED_NOT_QUALITY_GATE` and original production
chunk GGUF SHA-256
`5ba6202184804c0b5dec921b1ccfaed257ea36a57dfc6745a57b8dd2074fec1d`.
Its immutable chunk receipt SHA-256 is
`93ad1d8c0b618f6048fdde53d57554b4511d4508addb7e9083f619e62c2d1506`.
At 15:09 CDT, **8/42** production chunks were certified. Full GGUF merge,
published Q4 receipt, model load, numeric/full quality and Gen-0 selection
remain unfinished.

### October 25 15:16 CDT — ten certified Q4 chunks and automatic self-trim restart

The original third exclusive Q4 run successfully certified source-bound
chunks **009** and **010**, reaching **10/42** production chunks. Chunk
010 contains six mixed-K tensors and has GGUF SHA-256
`9cfa5cbdac6e1a7672288fbd122ab120b4f0f66fa11fb0b0124e887b70f5aa50`;
its receipt SHA-256 is
`96c4c90d7bd78ddf4d0ad83fb3030b0d11692d1f84df5256f3514a2b6d37063b`.
At the next **Q4_CHUNK_RESOURCE_WAIT** boundary, with no active
quantizer child and no unsealed F: scratch mini, the exact *owned*
original supervisor PID 57764 was verified, suspended and deliberately
terminated to activate the now-tested **automatic current-process-only
mmap-cache trim** in a fresh Python interpreter. Unrelated processes and
all ten SHA-sealed GGUFs/receipts were untouched. The old supervised
session exited nonzero due this explicitly operator-initiated stop,
**not** a data-integrity or quantization failure.

Following fresh >=12 GiB physical RAM and >=16 GiB available commit
admission, the same frozen source SHA, pinned quantizer, 42-piece map,
1-MiB slab and protected F/G paths were restored under exclusive Q4
recovery log `F:\reap128-v2-q4-streaming-resume4-auto-trim-20261025.log`.
New supervisor PID 72824 was launched, preserving all original chunk
receipts for independent full-SHA continuation. The new Windows API
self-trim is used **only** when actual per-chunk host RAM/commit admission
fails, no more than once per wait interval, and immediately rechecks the
unchanged safety floors. At documentation freeze, **10/42** chunks are
certified and the fourth source-bound resume remains in progress. Full
Q4 model publication, science and deterministic Gen-0 remain blocked.

### October 25 15:34 CDT — fourth exclusive Q4 owner verified live; automatic trim observed

Independent Windows process and immutable-evidence inspection verified
the **same** supervised original PID 72824 (not a competing restart), exact
registered BF16 source SHA, 42-part plan, 1-MiB quantizer slab, isolated
F:/G: workspaces and exclusive recovery log
`F:\reap128-v2-q4-streaming-resume4-auto-trim-20261025.log`.
The fourth run emitted `EXACT_Q4_CHUNK_REUSED` for **all ten** original
indices 0–9. It subsequently logged
`Q4_OWNED_MMAP_CACHE_TRIM` with `succeeded: true` and
`unchanged_start_floor_required: true`. This is now **live production
evidence for the integrated current-process-only trim**, rather than
merely the earlier independently performed owned-PID experiment or a
CPU-mocked test. The source and external GPU servers were not modified.

The verified original owner then **published new chunk 011/042** (zero-based
event index 10) with an exact immutable receipt, Q4 GGUF SHA-256
`77525e92246a1285d36ed5251f7df2fe3fdc3c179e2a172fba42283afe85521d`.
At 15:34 CDT, **11 original Q4 chunks and 11 receipts** were present and
the parent was preparing or processing subsequent work; no final public
Q4 GGUF or `.reap-chunked.json` receipt had been observed. Available
physical RAM was ~18.12 GiB, Windows free commit ~24.90 GiB, and
previous measured F/G/I free space ~36.70/31.02/7.99 GiB. The
unchanged 12/16 GiB start, 7/8 GiB runtime and disk safety floors remain
in effect. No registered GSM8K/HumanEval, BF16 CUDA parity, Q4/GPU
fit, same-session 200-row speed or REAP160 requalification is implied.
**Gen-0 remains BLOCKED; PR #282 remains draft.**

### October 25 15:39 CDT — bounded Q4 merge disk preflight and continued real chunks

A fresh read-only audit of final Q4 publication identified a consequential
scratch-volume failure interval: the parent Q4 startup checks for at least
8 GiB free on F:, but the future `--merge-worker` must write an additional
complete Q4 GGUF to F: before the durable finalization ledger can be sealed.
If free space disappeared between serial chunks and full assembly, the
runtime 6-GiB disk guard could stop the worker with a partial orphaned
assembly. The existing fail-closed recovery deliberately refuses that
unsealed partial artifact, so this required a preventive admission check.

The independent `merge_q4` process now conservatively requires *at least*
the sum of the exact 42 input mini-GGUF file sizes **plus the original 6-GiB
scratch-volume reserve** before opening the output GGUF writer. Each mini
GGUF includes its own metadata; their combined sizes upper-bound the full
merged single-metadata model. Missing/symlinked chunk inputs also fail
closed before output creation. Crucially the fourth **currently running**
parent has already imported its Python `_build`; the subsequent newly
spawned `--merge-worker` will import this amended source at launch. The
check therefore protects the existing original owner without forcing a
restart, modifying its live process, lowering any floor or touching
unowned workloads. Exact mixed-K payload verification and immutable
source/receipt checks remain unchanged. The guarded G: final-copy phase
already checks full remaining transfer bytes plus the same 6-GiB reserve
after retiring individually certified chunk payloads against its sealed
F: assembly.

At 15:39 CDT, the original owner additionally sealed **chunk 012/042**
(GGUF SHA-256
`7727f054a5c74e06b803280725c88c878eb07fcbf020c7e840f42bae11264445`)
and **chunk 013/042** (SHA-256
`6e5fd38c669aa831e7f482dac5b06a1e125e08068471c99ea980e6aac5be221e`),
each with its own durable proof; **13/42 chunks** are now complete. These
are exact partial Q4 construction milestones, not a final 733-tensor
model, descriptive BF16-vs-Q4 numerical result, registered quality PASS,
or scientific approval. Gen-0 remains `BLOCKED` and PR #282 remains draft.

The new low-space refusal and exact-capacity success fixtures prove the
check runs **before creating an orphan assembly**. The prime's offline
Q4 construction, Q4 numerical sampling and chunk-recovery focused matrix
completed **56 passed, one live pinned-quantizer test deliberately
deselected, two upstream SWIG deprecation warnings** in 15.41 seconds.
Ruff passed. Neither test invokes the production quantizer, uses a GPU,
or substitutes fixtures for full real-model evidence.

### October 25 15:48 CDT — Q4 numerical evidence publication hardening

The independent future full-model Q4-versus-BF16 sampler previously
wrote its source-bound descriptive JSON directly to the authoritative
output path. A crash after a partial write could strand invalid JSON in
that path and block a legitimate source-bound retry. It now reuses the
constructor's exclusive `_publish_exclusive` mechanism: write the
complete bounded evidence to an unpredictable private dot-pending file,
flush and `fsync`, then atomically publish **without replacing an existing
receipt**. Python exceptions remove only that owned pending file. A hard
kill may leave an uncommitted private pending sibling, but cannot expose
its partial bytes as the authoritative output. No quantizer, frozen SHA,
tensor-inventory, sample mathematics, score threshold or gate status was
changed. The sampler remains
`DESCRIPTIVE_Q4_QUANTIZATION_SAMPLE_NOT_QUALITY_GATE` and will only run
on a complete, independently published 733-tensor Q4 model after fresh
host admission.

Six new CPU adversarial tests exercise source-bound tiny mixed-K output,
flush/fsync ordering, an interrupted partial write, failed atomic rename,
a competing authoritative-destination publication race and a stale
private pending orphan. The prime's combined offline Q4 construction,
Q4 numerical, and Q8 recovery suite passed **62 tests**, with **one
live pinned-quantizer test deliberately excluded** and two SWIG
deprecation warnings; Ruff passed. These are engineering regressions,
not replacement benchmark quality evidence.

At 15:48 CDT the original exclusive Q4 supervisor PID 72824 remained
live, verified chunk **016/042** had GGUF SHA-256
`a3f0735d2c7b19368330fd5b000d39762b63f4e9a009ff2e7f2de12a0e07ef07`,
and **16 immutable receipts** were present. The next quantizer child was
observed as PID 28896 under that owner; its still-unsealed output is not
counted as progress. No full Q4 GGUF/sidecar or measured Q4 BF16 numeric
receipt had been published. The 704×40 original observation, 26-shard
BF16 candidate, finished Q8 and independent Q8 numerical measurement
are preserved. Frozen Amendment4 Gen-0 remains `BLOCKED` and PR #282
remains draft pending genuine full quality, control and hardware evidence.

### October 25 16:00 CDT — real Q4 continuation and paired source-witness validation

Independent inspection found the **original** supervisor PID 72824 still
running the exact registered Q4 command, with **20 of 42** mixed-K GGUF
chunks and matching source-bound immutable receipts successfully published.
Original indices 18 and 19 were newly sealed with GGUF SHA-256
`e8d5a5714185494430c0b87543be5685eb4fcb77505b38f8427d585135590376`
and `35dede7978466c158e8445c03c8ec86a05fb94474a6531295904f7e0b58ee6d1`,
respectively. The final public Q4 model/sidecar still does not exist. At
that read the host had 12.00 GiB free physical RAM and 23.76 GiB available
Windows commit; the supervisor's unchanged per-chunk admission guard
controls new child starts, and no unrelated process was terminated.

The separate `reap_compare` diagnostic was found to permit a potentially
misleading apparent quality **win** for `reap128_v2` from a GSM8K-only
or shortened paired corpus, without validating genuine model/serving
or official dataset source witnesses. This does not meet frozen scientific
qualification. Comparison of a new v2 *quality arm* now refuses unless
**every participating arm** has exactly both full registered GSM8K 1,319
and HumanEval 164 IDs. New v2 quality arms additionally require the
selector's frozen official source-population fingerprint, decoding,
scoring, serve.json and per-suite input witness. All per-item correctness
fields must be real booleans rather than forged numeric values, and the
diagnostic margin must be finite and nonnegative. Model/serving and
dataset input witness SHA values are included in the descriptive report,
and any corresponding input changing while pairing runs fails closed.
Historical generic `reap128` diagnostic subsets retain their original
scope, with no alteration to the registered selection thresholds.

Adversarial CPU tests prove refusal of missing/short suites, resealed
nonofficial GSM8K source digests, noncanonical IDs, numeric nonboolean
correctness, invalid statistical margins, and post-verification
serve/inputs/rows mutation, alongside a positive full-population
synthetic source-witness fixture. The combined scoped comparison,
selection and frozen gate regression suite passed **143 tests** (one
preexisting invalid-regex-escape warning), Ruff and diff checks clean.
This is readiness/evidence-integrity engineering only: the positive
test fixture is not registered quality, no new model inference was run,
and an internally matching serve.json is not independent proof that a
specific GGUF was resident on a GPU. Selector Amendment4 `BLOCKED`,
Q4 speed `UNKNOWN`, protected GPU services and all original thresholds
remain unchanged.

### October 25 16:06 CDT — official per-item ID binding in deterministic selector

The existing selector already checked exact frozen rendered GSM8K/HumanEval
population fingerprints and input/serve claims for new v2 arms, but a
separately resealed `.jsonl` containing an equal *count* of wrong IDs
could pass those checks if the paired legacy rows were also manipulated.
New full v2 quality evidence must now carry exactly `gsm8k-0` through
`gsm8k-1318` and `HumanEval/0` through `HumanEval/163`; new Q4 speed
evidence requires exactly the first `gsm8k-0` through `gsm8k-199`.
Unexpected, missing, duplicated or injected identifiers fail closed,
even with internally matching SHA/summary and original dataset claims.
No change to frozen first-200 ordering, score thresholds, historic v1
failure or Amendment4 fallback order. New RED/GREEN regressions cover
canonical full and 200-item cohorts, each of three resealed spurious-ID
variants and preserved official-source pinning. The combined CPU
selection/comparison/gates/speed-reconciliation regression passed
**152 tests**, one existing non-scientific Python invalid-regex warning,
Ruff clean. These tests do not constitute model inference or new gates.

A read-only actual **historical** selector replay against the existing
isolated 26-file baseline workspace and original registered v1/v2 D0
records returned exact decision
`{"chosen":null,"reason":"amendment 4 requires both REAP128_v2 and REAP160 full gates","status":"BLOCKED"}`;
`release_approved=false`, 28 baseline/calibration input digests. The
REAP128 v1 computed Gate1 remains FAIL. Later v1 gates are presented as
`NOT_EVALUATED` after this failure (with their separate computed statuses
retained); REAP160 original quality gates remain PASS but canonical
same-session speed is still UNKNOWN. The replay is **not** a new model
quality measurement and never overrides archived historical receipts.

At 16:06 CDT the untouched original Q4 PID 72824 was independently
observed through **23 of 42** source-bound sealed GGUF parts, newest
index 22 SHA-256
`311110a9248e6a5afb4e1f21f7e57aa45d1b1498129e1eb171167efdf48dc9d6`.
The complete Q4 publication, real Q4 numerical fidelity and registered
science remain pending. No quantizer duplication, GPU ownership change
or resource-floor reduction occurred. PR remains draft; Gen-0 BLOCKED.

### October 25 16:11 CDT — crash-safe official evaluator input-witness publication

The registered evaluation runner's authoritative `<suite>.inputs.json`
identity previously used a direct exclusive `open('x')` followed by
`json.dump`. An abrupt stop mid-write could expose corrupt, partial JSON
under the authoritative evidence name and prevent an otherwise sound
future experiment re-entry. The runner now builds a unique same-directory
private pending witness, flushes and `fsync`s all bytes, then exclusively
publishes without overwriting an existing source/model identity (Windows
rename or POSIX hard link). On Python exceptions it removes only its
owned pending file; a hard-killed process may leave an untrusted private
orphan, never a half-written authoritative witness. Symlink/nonregular
prior witnesses fail closed. The cooperative per-arm summary publication
now also `fsync`s its completely serialized pending bytes before its
already-guarded atomic summary replacement. No source prompts, scoring,
decode flags, resumption rules, model SHA, thresholds or real results
were altered.

Adversarial CPU tests prove full JSON and `fsync` precede first official
visibility, interrupted partial writes leave no false authority,
concurrent publish cannot replace another serving source identity,
and a previous orphan is never promoted. The combined offline
evaluator/isolated-workspace/compare/selector suite passed **109 tests**,
Ruff and whitespace checks clean. This is evidence-integrity hardening,
not a registered quality PASS or new serving session.

At 16:11 CDT, real original exclusive Q4 PID 72824 was still alive with
**24/42** SHA-sealed chunks; newly published chunk 024 GGUF SHA-256
`8148af7454077fb075b9261676ee1cd591082f19f9a7b3bf630da97e11f29f27`.
No complete Q4 GGUF or sidecar existed. Available physical RAM was
~16.02 GiB, free Windows commit ~23.88 GiB, and F/G/I free disk space
~36.65/27.17/12.75 GiB. The model owner, original quantizer and
unrelated GPU workloads were untouched. All mandatory scientific
qualification and deterministic Gen-0 gates remain pending/BLOCKED.
