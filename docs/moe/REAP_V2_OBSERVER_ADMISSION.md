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

The receipt's `observation_command_not_executed` is now a TEMPLATE for the
version-controlled **supervised, two-buffer atomic BF16 observer**, with
no `--limit-seqs` and no `--layers` shortcut. It cannot run until the operator
replaces the unique unused scratch/log placeholders and provides a dedicated
safe host resource window. NEVER run the old direct in-RAM command as registered
D0: it has no sealed atomic checkpoint.
Gate D0 requires a separate completed observer measurement with calibration
perplexity below 20; all subsequent frozen admission/quality/fit gates remain
unchanged. The intended `--out` directory must be absent; the preflight refuses
overlap with input directories and refuses overwrite.

The registered full observer now commits exact BF16 hidden states between
two 4.55-GiB file-backed slabs with durable per-layer `CURRENT` and hashes.
The default *uncheckpointed smoke-only* observer still uses host RAM tensors;
do not use that code path for the 704×40 registered D0. The 1,192,632 actually
retokenized tokens require two slabs totalling about 9.10 GiB, plus 4 GiB
scratch reserve. The final vocabulary projection is bounded to 64 positions
at a time in `reap_observe.py`.

### Local Windows commit-pressure mitigation (October 23, 2026)

The first real full 704-row observation started on the RTX 5060 Ti after
pausing the Sharp-Spark server. It completed only layer 0 in 220.3 seconds,
with 9,541,056 expert-token assignments (8 x 1,192,632 actually retokenized
positions, 256 experts). A local watchdog aborted the unfinished run during
the next layer after free system commit fell to 1.41 GiB for two consecutive
samples (VRAM free 0.72 GiB). The watchdog restored Sharp-Spark. The only
result is layer-00.pt; it contains no hidden-state continuation checkpoint,
no D0/perplexity and cannot resume at layer 1.

**Do not launch or resume the historical r2 output or F: scratch.** They are
incomplete single-buffer experiments without atomic `CURRENT` and cannot
resume. Use `docs/REAP-V2-FULL-RUN-OPERATOR-2026-10-24.md` for the exact
**fresh, unique path**, read-only `--doctor`, explicit `--launch` and
interrupted-run `--resume` workflow. The launch wrapper does not pause or
restore any model services: the resource maintenance window is arranged
separately, by the owner, before the launch. Two BF16 slabs preserve exact
uint16 bits and per-layer statistics; completed generations can resume after
an owned-worker safety stop. A run without completed D0 is never promoted.

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
