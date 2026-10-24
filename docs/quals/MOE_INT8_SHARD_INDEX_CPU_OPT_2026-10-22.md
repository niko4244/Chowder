# INT8 checkpoint shard resolution: safe CPU optimization

Scope: the safetensors index-to-file resolver in
`src/chowder/backends/moe_int8_experts.py::_shard_files`; **not** REAP
candidate selection, GPU expert dequantization, training, or LoRA export.

The starting code at integration commit `f099d3a` compared each tensor's
index-listed shard against a Python list of previously seen paths. Its work
grew with both index tensor count and shard count. It also accepted
index-specified `../` and absolute paths, and skipped unindexed `.safetensors`
files. The new resolver preserves first-seen index ordering and checks names
in a set. For indexed checkpoints it rejects malformed, empty and off-root
shard names, missing/unindexed shards, and linked shards. An index-free
single-file checkpoint retains the alphabetical fallback.

## Reproducible CPU microbenchmark

```powershell
python chowder_batch/bench_moe_shard_index.py --keys 1026 --shards 26 --loops 100
python chowder_batch/bench_moe_shard_index.py --keys 16000 --shards 26 --loops 20
```

Measured on the Windows host, Oct 22 2026, median wall-clock of metadata-only
resolutions, identical one-byte synthetic shard files in a temporary directory:

| Synthetic index | f099d3a reference | Optimized | Relative |
|---|---:|---:|---:|
| 1,026 tensor keys, 26 shard files (matches REAP160 index *counts*) | 8.1153 ms | 3.6016 ms | 2.25× |
| 16,000 tensor keys, 26 shard files (scaling stress) | 112.7519 ms | 11.5392 ms | 9.77× |

These are **index-resolution timings only**, not checkpoint loading, full
int8 quantization, training-step time or GPU throughput. The real REAP160
index contains 1,026 keys mapped to 26 shard names. Thus this optimization
saves only milliseconds per load on the current real index; it cannot make
27.48 s/step materially faster. That reported warm-step figure is dominated
by CPU-to-GPU transfer and int8 dequantization over a PCIe Gen3 x8 link,
and the code's earlier measured retention gain is already default-on.

## Compatibility and outstanding acceptance

All new tests run on CPU with tiny fake files and cover first-seen order,
duplicates, missing and extra shards, path traversal and absolute paths,
incorrect index types and symlinks. Existing synthetic MoE loader tests run
separately. No production model tensor or Q4 weights were opened or changed;
reading a real index is a metadata-only check.

The frozen REAP v2 plan still requires `reap128_v2` and `reap160` to be scored
following v1 Gate 1 failure. No new model was built here, no evaluation gate
has been advanced, and LoRA inference compatibility has not been certified by
this loader-index microbenchmark. A PEFT adapter is not directly equivalent
to an already-merged GGUF checkpoint: inference attachment needs separate
base-model/adapter compatibility evidence and a verified loader path.
