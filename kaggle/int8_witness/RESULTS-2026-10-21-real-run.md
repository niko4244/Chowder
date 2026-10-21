# The first real Kaggle run — 2026-10-20/21

Run: `nikmarco/chowder-int8-witness-step` (kernel version 1, private, GPU),
`max_steps=1` at `max_length=512`, checkpoint dataset
`nikmarco/qwen36-reap160-bf16` (38 files, 46.06 GB).

**It never reached a step.** The session was cancelled at 03:24 by the 12-hour
cap with the load at **16 of 26 shards**, 24 layers quantized, host private
memory 11.77 GB and climbing.

## The measurement that matters

| | |
|---|---|
| shards read | 16 of 26 (**29.25 GB** of 46.04) |
| time in the load | **707.7 min** (11 h 47 m) |
| effective read rate | **0.69 MB/s** |
| extrapolated whole-checkpoint load | **18.5 h** |
| the same load from local disk (Windows box) | 26 min ≈ **30 MB/s** |
| GPU quota before / after | 10.75 h / **0.00 h** (refreshes 2026-10-24) |

The GPU was idle for all of it. This is not the memory wall that ended the
Windows run — it is the read path: `/kaggle/input/...` is a read-only dataset
mount, and the loader reads one tensor at a time, so every tensor pays a round
trip. A 43x gap against local disk is the whole failure.

## What it does not change

The capacity probe's result stands: the 18.82 GB grid holds at 19.89 GB RSS of
31.35 GB, so host memory is not the obstacle on a Kaggle session. Private memory
here tracked the same path — 11.77 GB at 24 of 40 layers, on curve for ~19 GB.

## The fix this run bought

`run_step.py` now stages the checkpoint onto local scratch (`/tmp`, 1,067 GB
free) before the worker starts, and reports the copy's rate. If one sequential
copy also crawls, the mount is simply slow and Kaggle cannot host this at all;
if it is tens of MB/s, the next session loads in minutes. That copy is CPU-only
work, so it can be measured without spending GPU quota — which is the only way
to answer it before 2026-10-24.

Nothing was written to the pre-registration: no witness, no gate row, and the
table still reads 8 gates with 2 measured by the CPU census and 6 `not measured`.
