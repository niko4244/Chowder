# REAP128_v2: real Kaggle CUDA runtime, memory attribution and model inventory (2026-10-26)

**Status:** source-bound engineering qualification and one real native Q4 inference PASS. Complete 1,319/164 quality and 200-item speed are **not yet certified**. The original frozen selector remains `BLOCKED`, `chosen:null`, `release_approved:false`. This document does not authorize promotion, release, training or a revised statistical method.

## 1. Model and original frozen source integrity

- Original constructed BF16: `F:\reap128-v2-bf16-intermediate-20261025.gguf`, **37,122,439,488 bytes**, original SHA `80341aa776ad9b5f937c21b081b6d76de4a2ea277e5fb88c9ae2634a9da8f4c6`.
- Original Q8: `I:\reap128-v2-q8_0-20261025.gguf`, **19,748,435,968 bytes**, original SHA `89dcaf33b61f59c83a31b718ba98b73c33437d0320ae0cc56737a5d7e43519c6`.
- Original Q4_K_M: `G:\reap128-v2-q4_k_m-20261025.gguf`, **11,563,915,264 bytes**, original SHA `2d64b786461b99a335fdbbdf7c4d3a6758b21409bce755751714ab7cd8f63bf7`.
- Original 42-chunk Q4 sidecar SHA `cf16fbf9e6926719bab873bd74b7062fa22365a280451ce3db9a9f8dc3031f1c`; its 733 tensor/42 group structure and raw tensor certificate are preserved. The sidecar was freshly hashed and uploaded to a **separate private** Kaggle source dataset. No original model file was modified, uploaded publicly, or quantized again.
- Kaggle itself rehashed the **full mounted Q4 GGUF**, and the complete original 1,319 GSM8K / 164 HumanEval source files. Independent private SHA attestation receipt: `F:\reap-v2-kaggle-private-witness-output-20261026\reap-v2-original-artifact-integrity.json` (`all_pass:true`).
- Private actual hardware: two Tesla T4 GPUs, each 15,360 MiB total, each initially 14,912 MiB free; the separate request labeled A100 allocated T4x2, **not A100**. Guest initial system physical free ~30.3 GiB, scratch ~19.5 GiB, no swap. Only the *actual* device census may be used.

## 2. Source-backed active and total parameters

An independent enumeration of all 733 names/shapes in `G:\reap128-v2-q4_k_m-20261025.gguf.reap-chunked.json` yielded:

| Parameter-weight inventory | Weight elements |
|---|---:|
| Total stored across GGUF tensors | **18,543,997,568** |
| Always-present / non-routed-expert tensor elements | **2,437,870,208** |
| All 40 × 128 routed-expert tensor sets | **16,106,127,360** |
| Expert tensors selected by the top-eight router per token | **1,006,632,960** |
| Approximate active parameter-weight elements per routed token | **3,444,503,168** (18.575% stored) |

There are 120 routed expert packed tensors (up, gate, down × 40 layers), each with shape permutations of `128 × 512 × 2048`. At each routed layer the eight selected experts contribute `8 × 3 × 512 × 2048 = 25,165,824` selected weight elements, across 40 layers `1,006,632,960`. This is an architectural parameter-accounting estimate, **not a calibrated per-token FLOP count**: embeddings, gates, routing, and non-matrix operations do not all execute as a full multiplication of every stored constant tensor.

Original numerical BF16↔Q4 sample certificate, preserved: 122,880 paired values over 120 routed expert tensor windows; mean absolute error `0.000555533053235043`, RMSE `0.0007138570046274973`, max absolute error `0.003823518753051758`. These are quantization-fidelity descriptions, not answer accuracy or forward parity PASS.

## 3. Actual pinned Linux native CUDA serving

Private source build: [`nikmarco/chowder-reap128v2-cuda-toolchain-20261025`](https://www.kaggle.com/code/nikmarco/chowder-reap128v2-cuda-toolchain-20261025). The exact Windows executable's upstream **source Git commit** `7609846557c50f9d984719a9e1e8c5f3d02f807b` was checked out and successfully built on Linux CUDA 12.8. Kaggle's CMake lacked imported `CUDA::cuda_driver`; only this external experimental build disabled CUDA VMM with `-DGGML_CUDA_NO_VMM=ON`. Its distinct native `llama-server` SHA-256 is `f10032228f1d80b0f6639cd3bcdcc7194bbc6638368641cac7f1396775b5186d`, and it executed `--version` reporting `0.4.1-dev`, `7609846` before model loading. **Source commit identity is established; binary and allocator parity with Windows are not established.**

The first external guarded model attempt refused admission without loading when reading/hashing the 11.56 GB private model produced only **18.654 GiB unused cgroup memory** despite **30.185 GiB physical RAM reported free**. This is the observed distinction between guest MemAvailable and the cgroup charge; no RAM floor was reduced.

A modified, separate second run used Linux `posix_fadvise(..., POSIX_FADV_DONTNEED)` on clean read-only data after hashing the same exact bytes, before admission. That restored observed **29.448 GiB unused cgroup memory**, versus the 24 GiB *additional conservative* model-load admission. The real supervised native Q4 model subsequently loaded under the exact original serving flags:

`--fit on -np 4 -c 16384 --jinja`

Evidence: private [`nikmarco/reap128v2-q4-native-smoke-20261026`](https://www.kaggle.com/code/nikmarco/reap128v2-q4-native-smoke-20261026) version 2; downloaded read-only result `F:\reap-v2-kaggle-q4-native-smoke-v2-output-20261026\reap-v2-kaggle-q4-native-smoke.json`, SHA `429de0f7e22b5b16ff8b7697cbcb395ed8dee5732c5d6608d4e881c11148e18f`; original native startup/inference log SHA `081f023b9c6c8c905767d31787e2091427337858c74f76e238ee50e11885f5ee`.

- Original model SHA-256 and construction sidecar SHA verified **before** serving.
- `llama-server` reported **`offloaded 41/41 layers to GPU`** and native model loaded successfully after **66.9 seconds**. This is a model-level placement observation, **not** certification of a complete original `first-200` speed cohort or per-expert CPU placement proof.
- The first frozen GSM8K prompt returned a natural EOS `finish_reason: stop`, **259 completion tokens** in **4.492 seconds**. The simple single-request quotient is **57.66 output tokens/second**. It is not 200-item sustained throughput and does not prove candidate/dense ratio ≥ 1.3×.
- Initial free VRAM **14,912/14,912 MiB**, observed immediately after test **8,983/9,095 MiB** across T4 devices. Measured differences are approximately **5,929/5,817 MiB** GPU memory consumed relative to prelaunch, not sampled exact instantaneous peak.
- Sampled boot minimum free GPU was **9,001/9,113 MiB**. At the end of the response, physical RAM free **29.576 GiB** and cgroup unused **17.938 GiB**. This margin is above the original 7 GiB physical runtime floor and the conservative 8 GiB container surrogate. Windows available commit cannot be attested on Linux.
- The original Windows Spark server (`8767`), Ollama daemon (`11434`) and model router (`8766`) were separately verified HTTP 200 after the Kaggle experiments. They were **not paused** during these remote probes.

## 4. Real full two-suite experiment and exact evidence semantics

A new **private** external full Q4 experiment was launched as [`nikmarco/reap128v2-q4-full-evaluation-20261026`](https://www.kaggle.com/code/nikmarco/reap128v2-q4-full-evaluation-20261026). It is intended to run every frozen GSM8K **1,319** and HumanEval **164** item with the exact original original scorer source hash `d2e59aca5c0a621bac8d4979d522c9b6107f811610567d21c92a2b53960860eb`, evaluator source hash `9736714f21834a45c23599a50fa92ef3bebc6cf70967aca7e2329d26b4d73ee1`, identical original JSON prompts/expected test code and correct full-source digests. A separate CPU-only private Kaggle original-scorer check passed GSM8K correct, nonterminating rejection, and synthetic HumanEval isolated Python execution.

The current provisional experiment must require the live model PID to own `127.0.0.1:8790`, exact pinned model/build SHA, four-way scoring, frozen decode settings, resource monitoring and a *single serving session* per complete suite. Only all complete, same-session row sets permit an output summary. Any aborted partial or mixed-session rows remain ineligible. This Linux study is still **provisional** under the original Windows-specific attestation and no substitute for a full independent platform-parity review. The Windows resource floor and the original preregistered statistical method cannot be presumed satisfied on a different operating system. **Do not treat an asynchronous Kaggle run as already completed or claim any full accuracy value until its original actual rows are inspected.**

The original dense Q4 comparator is a separate pinned artifact, **21,166,757,984 bytes**, SHA `2c6d9466129e8802839566b35728e2506aaf1e8705cd137009e2f08feb9dd049`. Upload, allocation and independent full 200-item comparator are **not** established by candidate-only evaluation. All original Q8/BF16 parity, full dense_v2/random128_v2 fresh controls, REAP160 speed and optional paired-CI prospective governance remain unqualified. No scientific gate or Gen-0 selection can change on this record alone.

## 5. New CPU-only protection

`chowder_batch/reap_kaggle_cgroup_admission.py` is a separate **read-only, fail-closed** cgroup-v2 attribution and resource preflight. It never writes a cgroup, frees cache, starts a model or certifies Windows commit. It requires consistent cap/current/peak, file/anonymous cache telemetry and OOM events; clean file-cache bytes are **descriptive only, not optimistically counted as free memory**. Its decision may only admit an experimental smoke attempt, never a frozen scientific gate. `tests/test_reap_kaggle_cgroup_admission.py` covers OOM, missing/duplicate/nonfinite values, insufficient resources, and the actual file-cache/cgroup misinterpretation. These tests and the earlier source-bound Windows admission regressions passed **36 tests** CPU-only, Ruff clean; they do not replace full model qualification.

The frozen preregistration's paired-Wald-versus-paired-McNemar risk-difference CI ambiguity is still subject to the independent prospective review documented in `reap-v2-paired-ci-prospective-method-review-20261025.md`. Never choose a favored method after observing real v2-v1 results. Maintain the original fallback order and `BLOCKED` until complete authentic scientific gates are demonstrated.