# KaggleProvider acceptance — first real end-to-end screening run

Date: 2026-10-03 · Operator machine: Windows (the local controller) ·
Provider: `chowder.scientist.compute.KaggleProvider` · Runner:
`kaggle/provider_first_push.py` · Total real cost: **0.0155 device GPU-hours**
(≈ 56 s of 2×T4 wall) of the operator's 30 h weekly Kaggle GPU quota.

This is the provider-path acceptance record the design demanded before the
screening lane is trusted (docs/COMPUTE_PROVIDERS.md §6). Like
docs/DDP_ACCEPTANCE.md, it records a real run with real artifacts — no
simulated results anywhere.

## What ran

1. **Quota reconciliation against the live API** (`quota_view`): the operator's
   real weekly budget was read before anything shipped — 30.0 h allowed,
   0.0 used at start. The declared provider model was replaced by the real
   numbers (`sync_quota_from_api`), and the run was gated on them.
2. **A real kernel push** (`KaggleApi.kernels_push`, kaggle package 2.2.3):
   - pinned commit `1f02af0522ec2c2771259422f91a8c46b3f29a44` (the same commit
     that performed the push; verified present on the GitHub remote);
   - `machine_shape: NvidiaTeslaT4` → the T4×2 shape;
   - private script kernel, internet enabled (pinned pip install);
   - generated `kernel.py` from the provider template: pinned-commit install →
     `direct_url.json` cross-check → `capture_environment_fingerprint` →
     campaign spec persisted → operator command.
3. **The operator command on real hardware**: an fp16 matmul probe
   (2048×2048 × 10 iterations) — small, real, measured; not a training
   campaign.
4. **Session polling** (`kernels_status` → `KernelWorkerStatus`) and **output
   fetch** (`kernels_output`): 4 artifacts landed.
5. **Evidence recording**: the fetched `chowder_result.json` +
   `environment.json` became a run-grounded `ExperimentObservation`
   (`obs-provider-acceptance-20261003-144306`) written to durable research
   memory, `run_id` = the output directory (path-based grounding, the CLI
   resolver convention), `hardware_class = kaggle_2x_t4_16gb`.

## Results (verbatim from the fetched artifacts)

`chowder_result.json` (the kernel's own claim, template fields merged):

```json
{
  "campaign_spec_received": true,
  "dtype": "torch.float16",
  "exit_code": 0,
  "experiment_id": "provider-acceptance-20261003-143049",
  "gpu_count": 2,
  "gpu_name": "Tesla T4",
  "kernel": "Batch",
  "matmul_fp16_tflops": 0.34,
  "status": "complete",
  "torch": "2.11.0+cu128"
}
```

`environment.json` (the `BackendFingerprint` captured inside the kernel —
note the pinned-commit cross-check passed: `chowder_commit_sha` equals the
requested sha, so pip really installed that commit):

```json
{
  "accelerate_version": "1.14.0",
  "bitsandbytes_version": "0.50.2",
  "chowder_commit_sha": "1f02af0522ec2c2771259422f91a8c46b3f29a44",
  "cuda_runtime_version": "12.8",
  "device_map_summary": "{\"\": 0}",
  "dtype": "float16",
  "gpu_count": 2,
  "gpu_models": ["Tesla T4", "Tesla T4"],
  "python_version": "3.13.15",
  "quantization": "4bit",
  "torch_version": "2.11.0+cu128",
  "transformers_version": "5.16.1"
}
```

The efficiency number is deliberately recorded under the
`efficiency:matmul_fp16_tflops` surface with
`hardware_class = kaggle_2x_t4_16gb`: per the hardware-context rule
(docs/COMPUTE_PROVIDERS.md §3) it stays scoped to this hardware class and can
never be averaged with a local-GPU number into a "capability" claim.

## Live-API findings (found by this run, fixed and regression-tested)

1. **`kernels_push` returns a URL-path ref** (`/code/{owner}/{slug}`), not the
   `{owner}/{slug}` form `kernels_status`/`kernels_output` require. The
   provider now normalizes the ref at the submit boundary
   (`KaggleProvider._normalize_kernel_ref`); regression test
   `test_push_response_ref_is_normalized_to_owner_slug`.
2. **The kaggle client's console chatter can crash output fetch on Windows**
   (cp1252 `UnicodeEncodeError` while printing its post-download summary, even
   with `quiet=True`). The provider now redirects the client's console during
   fetch and, if an encode error still escapes, falls back to the artifacts
   that landed on disk — the files, not the console, are the evidence
   (regression test `test_output_fetch_survives_client_console_encode_errors`).
   The runner also reconfigures its own streams to UTF-8.

## Honest scope

- Proven by this run: push → queue/run → status → output → evidence, quota
  reconciliation, fingerprint capture, pinned-commit verification, the
  hardware-context discipline on a real artifact.
- Still true: the operator command is a probe, not a training campaign. A
  screening run of a real candidate (the compiled `campaign_spec` executed
  end-to-end by an operator-supplied trainer) is the next acceptance step and
  requires choosing that trainer; the plumbing exercised here is exactly what
  will carry it.
- The observation's `wall_gpu_hours` records the request's estimate (0.1);
  per-run metering inside the kernel is future work.

## Run 2 (2026-10-03, later same day) — per-run metering live-proven

Second real run through the identical path, pinning commit
`0ae358faecf710e917b3b56115dc1a07b621001d` (the metered template).
`chowder_result.json` (verbatim):

```json
{
  "accelerator_count": 2,
  "campaign_spec_received": true,
  "device_gpu_hours": 0.004621,
  "dtype": "torch.float16",
  "exit_code": 0,
  "gpu_count": 2,
  "gpu_name": "Tesla T4",
  "kernel": "Batch",
  "matmul_fp16_tflops": 0.38,
  "metering": "measured_wall_clock_x_attached_accelerators",
  "status": "complete",
  "torch": "2.11.0+cu128",
  "wall_gpu_hours": 0.00231,
  "wall_seconds": 8.318
}
```

What this proves:

- **The kernel measures its own cost**: wall clock around the operator
  command (8.318 s) × the accelerators actually attached (2) = 0.004621
  device-hours, replacing the 0.2 request estimate (which was ~43× the
  reality — exactly why estimates must not settle budgets when measurement
  is available).
- **Quota settlement uses the measured cost**: the provider reconciled its
  weekly model with 0.004621 on completion, not 0.2.
- **A real metering distinction discovered by this run**: after both runs,
  Kaggle's own `quota_view` reports 0.0326 device-hours consumed of the
  weekly 30 h — more than our two measured commands (0.0046 + ~0.005)
  because the SESSION (container start, pinned pip install, torch import,
  teardown) also bills. The two accountings answer different questions and
  both are recorded: in-kernel metering attributes cost **per experiment**;
  `quota_view` is the authoritative **weekly** budget. The provider's
  `sync_quota_from_api()` remains the gate for availability; the kernel's
  metering feeds evidence and the mission ledger.
- Total free-tier spend across both runs: ~0.033 of 30 weekly device-hours
  (≈0.1%). The lane stays free by design; the paid RunPod probe was
  deliberately deferred (operator decision: keep it free — Kaggle is the
  lane), so `docs/RUNPOD_PROVIDER_ACCEPTANCE.md` remains pending until the
  operator chooses to spend.
