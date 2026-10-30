# REAP128_v2 external GPU qualification: measured Kaggle admission (2026-10-25)

**Status: engineering preflight only; no official Gen-0 PASS, model selection, release or scientific quality/speed qualification.**

## Immutable local baseline and guard

- Windows feature branch `feature/reap128v2-selection-20261025`, initial published HEAD `3ee98362826cb7ecb8043a1fd06ed514244479d8`.
- Exact original Q4 candidate `G:\reap128-v2-q4_k_m-20261025.gguf`, size `11563915264`, SHA-256 `2d64b786461b99a335fdbbdf7c4d3a6758b21409bce755751714ab7cd8f63bf7`. Q8/BF16 originals, GGUF chunks, tensor receipts and frozen selector remain unchanged.
- The Windows CUDA smoke `F:\reap128-v2-q4-real-gpu-smoke-20261025-2300` completed with authentic Q4 model serving under `--fit on -np 4 -c 16384 --jinja`: first official GSM8K sample scored 0/1 and first official HumanEval sample 1/1, both terminated. These **single-item diagnostic scores** do not establish either entire 1319/164 gate.
- Original first-200 four-parallel candidate speed attempt `F:\reap128-v2-q4-registered-speed200-20261025-supervisor.json` has SHA-256 `836e4e886f11819075d75a461eec028f0453b9fd3d00e2c9553e0d3177dd81ed`, original startup/server log SHA-256 `94057821a77c7eaf15384850d1b155cb540904ef10e206a5f394289faa6012f1`. This attempt aborted safely with **12 rows**, dropping from 21.41 to 6.98 GiB available physical memory and 28.84 to a minimum 10.41 GiB available commit; its one-sided resource drop is an observed abort lower bound, not a 200-item model RAM peak.
- The new read-only `chowder_batch/reap_q4_resource_admission.py` pins both above original evidence SHA values, checks 7/8 GiB host floors, original command/cohort identity and original native 12,288 MiB VRAM admission, and includes a conservative separate prompt-cache growth allowance. The fresh unique local receipt at `F:\reap128-v2-q4-host-admission-20261025-2347.json` returned `REFUSED`, not an estimator claiming test PASS. It required **29.3146 GiB free physical RAM**, **34.3146 GiB Windows available commit**, **12,288 MiB GPU free**, and 8 GiB runtime disk reserve; actual at that check was 10.4451 GiB RAM, 16.1212 GiB commit and 8,172 MiB VRAM. No model was launched by preflight.
- Spark on Windows port 8767 and supervised Ollama on port 11434 were verified restored and alive, original router on 8766 untouched. Registered temporary port 8790 was not listening when these checks were made.

## Kaggle user-authorized feasibility probes — actual, not requested hardware

The user's authenticated `nikmarco` Kaggle account was verified through its installed CLI, with 30.00 GPU hours and 20.00 TPU hours remaining before these probes. Both **private** probes ran without any original model data, and outputs were downloaded to unique F: directories.

| Probe | Actual GPUs and free VRAM | Host free RAM | Swap | Scratch available | Classification |
|---|---|---:|---:|---:|---|
| [T4x2 fit](https://www.kaggle.com/code/nikmarco/chowder-reap128v2-gpu-fit-20261025) | 2 × Tesla T4, **14,912 MiB free each**, 15,360 MiB capacity per GPU | **30.3416 GiB** | 0 GiB | **19.5024 GiB** | Actual hardware inspection PASS; original scientific gate UNKNOWN |
| [A100 request fit](https://www.kaggle.com/code/nikmarco/chowder-reap128v2-a100-fit-20261025) | Requested `NvidiaTeslaA100`, **actually allocated 2 × Tesla T4**, each 14,912 MiB free | **30.3153 GiB** | 0 GiB | **19.5024 GiB** | A100 allocation **NOT demonstrated**; do not advertise an A100 |

T4 device VRAM admissions individually exceed the 12,288 MiB original Windows launch check; model serving, GPU tensor placement and sustained 7-GiB RAM margin **have not** been demonstrated by these probes. The original full candidate uses 11,563,915,264 bytes Q4 and the original dense Q4 comparator uses 21,166,757,984 bytes. Kaggle `/kaggle/working` offers less than 20 GiB persisted scratch; model data must be mounted read-only via private Kaggle datasets rather than copied there. Separate cards do not combine into a 30 GiB contiguous device; placement must be attested by actual llama.cpp loading logs.

**Cross-host methods constraint:** the frozen original explicitly requires live Windows process/listener ownership, host Windows commit floor and native server SHA/session proof. Kaggle is Linux and has no equivalent Windows `available commit` field or original Windows executable bytes. Any Linux score must have its **own** registered source/runner build SHA, row timings, captured source digest, correct official benchmark code, complete 1319/164 cohorts and independently justified protocol equivalence. One cannot silently promote Kaggle results to frozen original Windows scientific PASS. Actual toolchain version from the prior Windows binary is `0.4.1-dev` build 10985, commit prefix `760984655`; full upstream Git commit resolved as `7609846557c50f9d984719a9e1e8c5f3d02f807b`.

## Independent statistics caveat

See `docs/quals/reap-v2-paired-ci-prospective-method-review-20261025.md`: Amendment 4 optional v2-v1 replacement says paired McNemar-style one-sided 95% margin `−0.02` but does not specify a unique matched marginal-risk-difference CI, unlike §4's paired Wald formula. Synthetic comparisons demonstrably diverge, so no favorable post-hoc method selection, threshold change or backfilled PASS is permitted; an independent prospective method decision remains required for that optional branch.

## Unmet original selection evidence

The original mandatory official data populations remain GSM8K **1,319** and HumanEval **164**, with separately registered original dense and random128_v2 controls. True Q4 same-serving-session first 200 official GSM8K items for each candidate and original dense_v2_q4 comparator, GPU placement and throughput ratio ≥ **1.3×**, BF16/Q8 parity where originally required, REAP160 fallback evidence and unchanged gate/fallback order remain outstanding. A partial Kaggle probe or Windows 12-row abort is **not** a qualifying first-200 speed score. The latest source-bound frozen selector replay `F:\reap128-v2-preflight-refused-gen0-replay-20261025.json` had `decision.status=BLOCKED`, `chosen=null`, `release_approved=false`. Draft PR #282 must not be merged or released.

## Reproducibility and no-risk rules

- The private Kaggle model ingestion stage is a **same-volume NTFS hardlink**, not a copy or rewrite of the original GGUF; the Kaggle dataset upload is private by default and never uses `--public`. The exact remote dataset existence must be verified before referencing any uploaded weight as available.
- The private [pinned CUDA Linux toolchain notebook](https://www.kaggle.com/code/nikmarco/chowder-reap128v2-cuda-toolchain-20261025) attempts to reproduce the **full original upstream llama.cpp Git source SHA** on Kaggle. Initial version 1 could clone but could not fetch by truncated 9-digit commit ID; version 2 pins the full 40-character SHA. A completed native binary must have independent build success and SHA witness before use. No model GPU inference or release is implied by a build-only notebook.
- CPU-only adjacent regression: `83 passed` on `tests/test_reap_q4_resource_admission.py`, `tests/test_reap_paired_method_prospective_20261025.py`, `tests/test_reap_serve.py`; Ruff PASS and `git diff --check` PASS before this document. Keep original model artifacts, owner services, source/benchmark witnesses, and immutable results unchanged.
## Additional frozen corpus transfer and Linux build diagnosis

A new **private** source-only Kaggle dataset, [`nikmarco/reap128v2-original-official-inputs-20261025`](https://www.kaggle.com/datasets/nikmarco/reap128v2-original-official-inputs-20261025), was created and its file listing was verified independently. Original full rendered input SHA values, rederived offline from the frozen repository evaluator:

| Source | Official count | Original rendered digest | Exported immutable JSON file SHA-256 |
|---|---:|---|---|
| GSM8K | 1319 | `913ca81f724cba16a07099608047e7522b00910c1c7fe13499c501ac9e4bad89` | `4dae0860ba5b9785343070a4919504f0aa4eccb01a96108b5b851d23e059a522` |
| HumanEval | 164 | `49fb688c6788b0531803dfd1cb85a69926ec5a7179e74cdfffc53ba52a92b420` | `5144d31cf337b25ce262b780ed103b128b1e9d289d3c0d15f4fe34933bb705d5` |

This private dataset was exported to `F:\reap128-v2-kaggle-official-source-20261025`. It contains only canonical benchmark inputs/expected data and a digest manifest, **no scored model responses and no proof of a model run**.

The private pinned Linux CUDA build initially failed fetching **truncated** commit `760984655` (Git refuses ambiguous short fetch). Fetching the complete `7609846557c50f9d984719a9e1e8c5f3d02f807b` succeeded and CMake identified `llama.cpp version 0.4.1-dev`, CUDA 12.8 and two SM75 GPUs. Configuration then failed because Kaggle's toolkit did not expose CMake's `CUDA::cuda_driver` imported link target. The later **explicit experimental build variant** used `-DGGML_CUDA_NO_VMM=ON`, a CMake-supported option that avoids the missing CUDA driver link target. This changes the native allocator configuration relative to the Windows binary and therefore must be disclosed as **nonidentical cross-platform infrastructure**, not hidden beneath the phrase “same binary.” Native compilation success and resulting `llama-server` SHA must still be verified from the private kernel receipt; never infer either from Kaggle's generic `COMPLETE` status alone. The pinned source identity is unchanged.