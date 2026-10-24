# Frozen REAP128_v2: BF16 layer checkpoint and recovery

This patch does **not** recover the failed r2 run. r2 has only a single,
partly overwritten 4,885,020,672-byte slab (mixed layer 10/11 states).
Its eleven completed saliency-stat files are diagnostic evidence, not a
checkpoint. The full 704-sequence, 40-layer Gate D0 has **NOT** been measured.

## Running and resuming

Use the exact frozen model, weights, calibration, and numerical environment
from the approved preflight. Do not launch until the external host-resource
supervisor and model provenance gates admit a new experiment.

Fresh observation: choose unique, nonexistent locations for both paths:

```powershell
python -B chowder_batch/reap_observe.py --model-dir <MODEL_DIR> --weights-dir <WEIGHTS_DIR> --calibration <FROZEN_CALIBRATION_JSONL> --out <NEW_OUTPUT_DIR> --hidden-disk-dir <NEW_F_VOLUME_SCRATCH_DIR>
```

After an interruption **with at least generation 0 committed**, confirm
that the prior observer process has exited and use the *same* paths,
changing only the checkpoint flag (no `--hidden-disk-dir`):

```powershell
python -B chowder_batch/reap_observe.py --model-dir <SAME_MODEL_DIR> --weights-dir <SAME_WEIGHTS_DIR> --calibration <SAME_CALIBRATION_JSONL> --out <SAME_OUTPUT_DIR> --resume-hidden-disk-dir <SAME_F_VOLUME_SCRATCH_DIR>
```

Never pass the old r2 scratch directory: it lacks the v1 checkpoint layout,
generation-zero embeddings marker, dual slabs, and committed checksum chain.
An interruption before all embeddings and generation 0 were committed is
**not** resumable. Preserve those incomplete diagnostics and start a new run
in new unique output/scratch directories. An interrupted NLL after generation
40 may be rerun from the last verified committed hidden states.
Do not combine the two disk flags. Never edit `CURRENT`, checkpoint manifests
or source files to force a resume.

## Crash contract

- One immutable active BF16 slab and one alternate output slab; raw uint16
  payload, no numeric casting, 704 ordered row writes.
- Generation 0 commits **all embeddings**. Generation N commits decoder layer
  N-1 only after all rows, BF16 flush + file fsync, SHA-256 of the entire slab,
  and fsynced statistics/manifest files.
- `CURRENT` is the **last** same-directory atomic publication step; a crash
  earlier leaves the prior generation authoritative. Resume checks all
  manifests, contiguous layer IDs, source identity and statistics hashes, then
  validates the active full-slab hash. The standby buffer may be incomplete.
- Statistics are canonical in the scratch directory on one volume; copies in
  `--out` are regenerated mirrors and cannot authorize resumption. A live OS
  file lock prevents concurrent writers. Orphan temp files are not checkpoints.
- A full completed observation result is written atomically only after all
  40 layers, next-token NLL and a repeat input hash check. A perplexity
  failing D0 is still a completed **failure**; no PPL is presumed or fabricated.

The identity records the full **payload** SHA-256 of every indexed weight
shard, tokenizer/config/index hashes, exact retokenized ordered token IDs and
row lengths, calibration SHA, current implementation and reference library
code hashes, model/weights/output paths, NVIDIA driver and GPU identity,
and platform/torch/transformers/CUDA/SDPA numeric settings. Every resume
rechecks the entire identity before loading weights.
This is intentionally a substantial read-only I/O operation on the original
BF16 shards.

For the original observed 1,192,632 token IDs and 2048 hidden dimensions,
each slab is 4,885,020,672 bytes; two slabs are **9,770,041,344 bytes**,
before a reserved 4 GiB of free space and the original BF16 model storage.
Use a spacious scratch volume (historically F:), not space-constrained C:.

## Remaining scientific and operational gates

- CPU-only fixture parity/corruption tests verify storage and crash semantics,
  **not** arithmetic equivalence of the entire 40-layer CUDA model.
- The final `observation.json` carries a complete sealed generation-chain
  receipt including source identity, final state hash, and per-layer statistics
  hashes. The v2 selection/build gate and pruning reader verify the receipt
  against their copied statistics; the scratch source remains authoritative.
- Windows memory-map flush and file `fsync` are required. `os.replace`
  provides same-volume atomic publication at the filesystem namespace level;
  NTFS/power-loss directory-metadata durability is not independently verified
  and cannot be asserted from these tests. Power loss must fail closed if any
  marker or checksum is torn.
- r2 host-RAM exhaustion coincided with a large unrelated process-creation
  burst. This suggests host contention, not a proved CUDA leak. Future runs
  still need multi-sample admission and per-PID host telemetry. Never terminate
  unrelated agents to make room.
- The original frozen 704x40 D0 remains **BLOCKED** until a new complete,
  independently verified observation and measured perplexity result exist.
