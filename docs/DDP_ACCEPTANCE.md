# DDP (2-GPU) acceptance

Chowder can drive multi-GPU LoRA training via `accelerate launch --multi_gpu`
(`TransformersPeftExecutor._worker_command`, `backend.runtime.active_accelerator_count`).
`HardwareProfile`/`HardwareTopology` deliberately never pool separate accelerators
into one virtual VRAM figure -- two 16 GB T4s are two 16 GB pools, not one 32 GB
device, because under DDP every rank holds a full model replica and the correct
memory comparison is always the smallest single device, never the sum
(`_min_device_vram_gb`).

That command-construction and accounting logic has unit-test coverage
(`tests/test_transformers_backend.py`,
`test_worker_command_uses_accelerate_launch_for_multiple_accelerators`,
`test_executor_launches_accelerate_when_multiple_accelerators_requested`, etc.),
but those tests mock the subprocess (`Popen`) entirely -- they prove the command
Chowder *would* run is correct, not that a real `accelerate launch` genuinely
engages two devices, that checkpoints don't corrupt under two writers, or that
cancellation tears down every rank. Proving that requires two real GPUs, which
CI does not have.

## What `tests/test_ddp_acceptance.py` proves for real

Every test in that file requires `CHOWDER_REAL_ML_SMOKE=1` **and** two real,
visible CUDA devices (`torch.cuda.device_count() >= 2`), checked at collection
time -- it is not possible for these tests to pass by accident on a single-GPU
or CPU-only machine.

- **`test_real_ddp_two_gpus_trains_evaluates_and_reports_correct_gpu_hours`** —
  the core Phase 5 flow: `active_accelerator_count=2` → `accelerate launch` →
  both T4s genuinely engaged (`resource_usage.active_accelerator_count == 2`,
  which only a real `world_size == 2` produces) → LoRA training completes →
  exactly one valid adapter artifact (not one per rank, not corrupted by two
  writers) → the independent `transformers-text` evaluator reloads and scores
  it → `gpu_hours == wall_hours * 2` (this falls out of
  `ResourceUsage.from_wall_time` once `active_accelerator_count` is proven
  correct, so the real proof point is the device count, not the arithmetic).
- **`test_real_ddp_cancellation_terminates_every_rank_and_frees_gpu_memory`** —
  cancels a real in-flight DDP run and, after a bounded wait, requires
  `nvidia-smi --query-compute-apps` to show no new PID holding a device. An
  orphaned rank after cancellation would otherwise silently corrupt the next
  run's GPU-hour accounting.
- **`test_real_ddp_one_rank_crashing_is_reported_as_a_failure_not_a_silent_success`** —
  deliberately crashes rank 1 (via a test-support-only hook in
  `transformers_worker._crash_rank_for_ddp_acceptance_test`, inert unless the
  `_CHOWDER_DDP_ACCEPTANCE_CRASH_RANK` env var is explicitly set -- normal runs
  never set it) and requires the executor to raise a real failure, never build
  an artifact from only the surviving rank's work.
- **`test_real_ddp_checkpoint_save_produces_one_checkpoint_no_rank_duplication`** —
  with `save_strategy=steps`, requires every saved checkpoint directory to be
  well-formed (a duplicate-writer race would corrupt file contents, not just
  add extra directories).

## How to run it

On a machine with two real GPUs (e.g. a Kaggle notebook with the T4 x2
accelerator, which requires a phone-verified account):

```bash
pip install -e ".[train,dev]"
CHOWDER_REAL_ML_SMOKE=1 python -m pytest -q tests/test_ddp_acceptance.py -v
```

## Result

**Not yet run for real.** This suite has been written and verified to skip
cleanly (for the right reason -- device count, not the env var) on
single-GPU/CPU development machines and in CI, and its non-DDP code paths are
exercised by the existing real-ML smoke suite. It has not yet executed to
completion on real 2-GPU hardware. Update this section with the real run's
result (pass/fail, GPU model, driver/torch/accelerate versions, date) once
that happens.
