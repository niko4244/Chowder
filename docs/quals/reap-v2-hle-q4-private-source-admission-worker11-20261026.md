# Original REAP128_v2 Q4 — private HLE source admission (worker-11)

Date: 2026-10-26. Independent experimental worktree
`C:\Users\nikma\wt\chowder-reap128v2-hle-q4-worker11-20261026`,
branch `feature/reap128v2-hle-q4-private-worker11-20261026`,
based on clean `e62cf4e2952bbd97b5b81ae9a5c81fbc2e90f6fb`.
The original scientific selector, protected private GGUF, Kaggle jobs and
draft PR #282 are untouched.

## Measured remote admission

* Hugging Face connected account and local `hf auth whoami` both identify
  account `NIKO42`. The official `cais/hle` repository is gated and its
  revision observed through the authenticated Hub metadata is
  `5a81a4c7271a2a2a312b9a690f0c2fde837e4c29`.
* The exact file `data/test-00000-of-00001.parquet` is visible in the
  source listing; authenticated, redirect-disabled **HEAD** returned
  HTTP **403**. It did **not** download benchmark questions, answers or
  images. No local `datasets--cais--hle` Hugging Face cache exists.
  Dataset file full-blob SHA-256 and accessible file length therefore
  remain **unknown** and cannot be invented from repository metadata.
* Kaggle's original
  `nikmarco/reap128v2-random128-full-postload` was
  `KernelWorkerStatus.RUNNING` at last preflight; original Q4 full and
  original Q8 postload full kernels returned `COMPLETE`. Remaining
  GPU quota at the latest source preflight was **16.34 GPU-hours**.
  No complete inventory of competing account GPU jobs was independently
  established.
* The source-only CLI returned `status: REFUSED`, with exact independent
  blockers `HLE_GATED_ACCESS_DENIED`,
  `PROTECTED_KAGGLE_JOB_ACTIVE_OR_UNVERIFIED`,
  `FULL_GPU_JOB_INVENTORY_NOT_ATTESTED`,
  `RUNTIME_GPU_MOUNT_AND_MEMORY_NOT_ATTESTED` and
  `JUDGE_RUNTIME_NOT_ATTESTED`.
  It reports `gpu_job_launched:false`, `dataset_rows_read:0`,
  `generated_programs_executed:0`,
  `runtime_one_item_approved:false`,
  `scientific_selection:BLOCKED`, `chosen:null` and
  `release_approved:false`. This is a refusal receipt, not benchmark
  accuracy or speed.

## Frozen source-only experimental contract

The implementation is `chowder_batch/reap_hle_q4_source_preflight.py`.
It fixes original Q4 GGUF SHA
`2d64b786461b99a335fdbbdf7c4d3a6758b21409bce755751714ab7cd8f63bf7`
(11,563,915,264 bytes), original Q4 construction sidecar SHA
`cf16fbf9e6926719bab873bd74b7062fa22365a280451ce3db9a9f8dc3031f1c`,
upstream llama.cpp source
`7609846557c50f9d984719a9e1e8c5f3d02f807b`,
and pinned Linux binary SHA
`f10032228f1d80b0f6639cd3bcdcc7194bbc6638368641cac7f1396775b5186d`.
The distinct Linux binary has `GGML_CUDA_NO_VMM=ON` and is not
numerically established equivalent to the original Windows binary.

Official prediction and judge code is pinned to
`centerforaisafety/hle` Git commit
`22ed3074b1e7b134bcbc09028d0ba320839b0655`.
The actual pinned `run_model_predictions.py` SHA is
`d2062d8afcfb8968f350bbae953e897d4303a6f72a4ae7d97b7db0137a333b17`;
`run_judge_results.py` SHA is
`a796e74d6e58beab3bd88f46182f599443a102308b2d1c7e0b4b22f578468446`.
Both source files were fetched as **small, public source code only** and
verified with the same SHA-256 values. The frozen official judge model
identifier is `o3-mini-2025-01-31`. No paid judge requests were made.

The proposed **one-item** no-tools study selects the first source-ordered
question with absent/empty image, with no questions or answers copied to
this repository. Prediction formatting uses the exact official pinned
system prompt, `temperature=0`, declared seed 0, max 8,192 completion
tokens, single session and `--fit on -np 1 -c 16384 --jinja`. This is a
separately frozen research geometry: the historical Q4 original four-way
16,384-context serving geometry cannot provide 8,192 tokens per sequence.
Any future runtime must independently prove its exact model, native binary,
dataset source, CUDA placement, serving PID/port, bounded cgroup/VRAM,
one-item completion and separate official judge. Text-only sample accuracy
must be clearly labeled; no multimodal or official full-HLE leaderboard
result can be inferred from it.

## Reproduction and next admission

Read-only (the command exits **2** while blocked):

```powershell
python chowder_batch/reap_hle_q4_source_preflight.py
```

When the authenticated HLE dataset becomes accessible, independently
establish the file's exact SHA-256 before passing it as
`--dataset-blob-sha256 <verified_sha256>`. Even a successful source HEAD
does not grant GPU launch: a separate complete account job inventory,
source-bound private runtime/mount validation, independent CUDA resource
admission and judge runtime attestation are required. Do not run a GPU
one-item preflight while the original random128 job is active.

`python -m pytest tests/test_reap_hle_q4_source_preflight.py -q`
passed **23 tests**; Ruff passed. These tests exercise synthetic malformed
metadata, wrong SHAs, unknown states, active jobs, inaccessible source,
malformed examples and the permanent separation of read-only source
preflight from runtime approval.
