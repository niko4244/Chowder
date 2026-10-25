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

These outcomes demonstrate a fixed memory-intensive code path but
**not yet successful full-source verification of a production shard**.
Do not reinterpret these resource refusals as REAP scientific failures.

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
   The guarded `reap_gguf_supervised` now supports explicit `--resume`
   for an interrupted later stage. Every completed BF16/Q8_0/Q4_K_M GGUF
   file has a new exclusive `.reap-stage.json` receipt, which binds the
   candidate full-equivalence proof SHA-256, pinned converter/quantizer,
   exact frozen commands, input GGUF digest, output size and output SHA-256.
   Resumption rereads the entire previously published GGUF to verify the
   sealed hash before reusing it. An existing output without the exact
   receipt, a changed recipe or a same-size changed payload fails closed.
   No stage may be reused based on magic bytes, filename or size alone.
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
