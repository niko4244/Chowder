# Frozen REAP128_v2 D0 observation: CPU input admission

`chowder_batch.reap_v2_observer_preflight` inspects the frozen Amendment-4
calibration and the original BF16 checkpoint *before* scheduling the expensive
40-layer observation. It uses only the Python standard library and existing
calibration identity code. It reads small metadata files and safetensors headers;
it does not load weight payloads, initialize CUDA, run the observer or write a
result. A successful exit authorizes no GPU job or Gen-0 promotion.

From the Chowder repository root, the command validated on the local Windows
machine on 2026-10-23 was:

```powershell
python -B -m chowder_batch.reap_v2_observer_preflight `
  --calibration C:\llm-work\reap\calib-v2-ornith-bare-20261023 `
  --model-dir G:\llm-cold\bf16\Qwen3.6-35B-A3B `
  --out C:\llm-work\reap\observe-v2-full-registered-20261023
```

It returned `INPUTS_ADMITTED_OBSERVATION_NOT_RUN` with the registered 704-row,
1,192,636-token v2 calibration, SHA-256
`f6a09b694ae43a6d8b44b5180a861856501295eabf3d5297eb9749a008cdd929`.
The optional admitted Ornith file has 39 candidates but zero eligible bare
renders, so the calibration population remains 704 rows. Source model evidence:
26 existing BF16 safetensors shards, 1,045 indexed tensors, 40 decoder layers,
256 experts, top-8, embedding/norm/lm_head and 120 required routed tensors.
The input config SHA-256 was
`93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99`
and index SHA-256 was
`41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83`.
The large shard payloads (71,903,776,776 bytes) were **not** hashed, so this
inspection does not attest an exact model weight revision. It also does not
verify that the installed Torch/Transformers versions or GPU state can complete
the forward pass.

The receipt's `observation_command_not_executed` is the full 704-sequence,
40-layer observer invocation, with no `--limit-seqs` and no `--layers` shortcut.
Gate D0 requires a separate completed observer measurement with calibration
perplexity below 20; all subsequent frozen admission/quality/fit gates remain
unchanged. The intended `--out` directory must be absent; the preflight refuses
overlap with input directories and refuses overwrite.

The observer keeps all per-sequence hidden states in host RAM. At a declared
1,192,636 tokens × 2,048 hidden dimensions × 2 BF16 bytes, that is at least
4.55 GiB of hidden data alone, excluding Python, attention activations, model
weights and other processing buffers. The final vocabulary projection is
bounded to 64 positions at a time in `reap_observe.py`.

### Local Windows commit-pressure mitigation (October 23, 2026)

The first real full 704-row observation started on the RTX 5060 Ti after
pausing the Sharp-Spark server. It completed only layer 0 in 220.3 seconds,
with 9,541,056 expert-token assignments (8 x 1,192,632 actually retokenized
positions, 256 experts). A local watchdog aborted the unfinished run during
the next layer after free system commit fell to 1.41 GiB for two consecutive
samples (VRAM free 0.72 GiB). The watchdog restored Sharp-Spark. The only
result is layer-00.pt; it contains no hidden-state continuation checkpoint,
no D0/perplexity and cannot resume at layer 1.

For a NEW registered full run, an optional exact-BF16 file-backed storage
path is now supported:

    python -u -B chowder_batch/reap_observe.py --model-dir G:\llm-cold\bf16\Qwen3.6-35B-A3B --calibration C:\llm-work\reap\calib-v2-ornith-bare-20261023\calibration.jsonl --out C:\llm-work\reap\observe-v2-full-registered-20261023-r2 --hidden-disk-dir F:\reap-v2-hidden-full-20261023-r2

Both output and scratch directories must be new; the file-backed scratch
directory is automatically created only after a disk-space check. Its
contiguous uint16 payload reinterprets each 16-bit BF16 hidden element
without conversion, preserving the original sequence offsets and arithmetic
order. The original default remains the RAM tensor list. The disk slab needs
at least 4.55 GiB plus a 2 GiB reserve on the scratch volume and reads/
writes all 704 sequences at each decoder layer. This reduces Windows private
commit use but increases disk I/O. Interrupted scratch directories are
never re-used or silently qualified; only complete new observation.json
files support D0.

Each layer now releases last-sequence CUDA temporaries before the next
layer is loaded, and records PyTorch allocator peak/remaining and CUDA
global free VRAM after cleanup. A native RTX 5060 Ti smoke on the pinned
source with two sequences, layers 0 and 1 completed with identical
32,768 routed assignments and 256 hit experts per layer; CUDA peak
was 2,020-2,021 MiB; allocated CUDA after cleanup returned to 8.1 MiB,
global free was 14,162 MiB. This proves the low-memory forward path works
for those two rows only; no full registered observation gate was measured.

CPU bit-for-bit BF16 special-pattern/ragged-two-layer tests and existing
observer preflight, checkpoint assignment and NLL tests passed (62 tests,
Ruff clean). Stop or pause only owned model workloads; do not shut down
the communication bridge or alter the frozen calibration/model.

For context only, the existing **v1** observer receipt has 384 sequences,
651,857 measured tokens, 6,960.5 cumulative layer seconds (1,001.2 seconds
of layer loading) and perplexity 10.646. Scaling just its non-load layer time
by the declared v2/v1 token ratio gives approximately 11,904 cumulative layer
seconds before final NLL, runtime variation or other overhead. This is a
resource-planning extrapolation across different datasets, **not** an executed
v2 measurement, a run-time guarantee or evidence that v2 passes D0.

For CPU regression verification:

```powershell
python -B -m pytest -q tests/test_reap_v2_observer_preflight.py
python -B -m ruff check chowder_batch/reap_v2_observer_preflight.py tests/test_reap_v2_observer_preflight.py
```
