# REAP128_v2: independently measured dense-Q4 cgroup abort and cross-platform single-GPU parity (2026-10-26)

**Frozen Gen-0 determination:** `BLOCKED`, `chosen:null`, `release_approved:false`. This document contains measured research evidence, **not** a speed or official quality PASS. Original model files and serving flags remain immutable. The separate private full 1,319+164 Q4 candidate and first-200 candidate GPU evaluations are registered outside the Windows selector and must not be credited before their complete, source/session-bound receipts are inspected. Original PR #282 stays draft/unmerged.

## Published source and frozen identities

- Worktree branch `feature/reap128v2-selection-20261025`, prior published HEAD `dc0b1246dc4cd9d454d8c5ac25b1637dfd62592a`.
- Original candidate Q4_K_M: **11,563,915,264 bytes**, SHA-256 `2d64b786461b99a335fdbbdf7c4d3a6758b21409bce755751714ab7cd8f63bf7`; 42-group original receipt SHA `cf16fbf9e6926719bab873bd74b7062fa22365a280451ce3db9a9f8dc3031f1c`.
- Original `dense_v2_q4`: **21,166,757,984 bytes**, SHA-256 `2c6d9466129e8802839566b35728e2506aaf1e8705cd137009e2f08feb9dd049`. Private dataset [`nikmarco/reap128v2-frozen-dense-q4-20261026`](https://www.kaggle.com/datasets/nikmarco/reap128v2-frozen-dense-q4-20261026) original length was verified through Kaggle's file listing. The full SHA was checked locally prior to the private upload.
- The original 1,319-item GSM8K rendered-source digest `913ca81f724cba16a07099608047e7522b00910c1c7fe13499c501ac9e4bad89`; original 164-item HumanEval digest `49fb688c6788b0531803dfd1cb85a69926ec5a7179e74cdfffc53ba52a92b420`. Kaggle privately rehashed both exact files before the candidate native test.
- The same source-pinned native Linux llama.cpp Git commit `7609846557c50f9d984719a9e1e8c5f3d02f807b` was used, with a **disclosed** Linux-specific `GGML_CUDA_NO_VMM=ON` build (not Windows byte-identical); binary SHA `f10032228f1d80b0f6639cd3bcdcc7194bbc6638368641cac7f1396775b5186d`. All registered model launches retained `--fit on -np 4 -c 16384 --jinja`.

## Native dense first-200 benchmark attempted; safety-aborted

A new **private** external original dense-Q4 first-200 experiment [`nikmarco/reap128v2-pinned-dense-q4-first200`](https://www.kaggle.com/code/nikmarco/reap128v2-pinned-dense-q4-first200) used the full original 1,319-source witness and attempted only the first 200 GSM8K prompts in a separately registered native 4-way serving session. It **did not** complete the 200 rows, nor produce an admissible throughput comparator.

Its original machine receipt, downloaded to `F:\reap-v2-kaggle-dense200-aborted-evidence-20261026\reap-v2-kaggle-dense200-native.json`, has SHA-256 `b2ef325bb9c3165439e669326746fdb9165a5a66975890c99cbfd56dda8c6c0a`. The native log is in that same original scratch evidence folder, with receipt-pinned log SHA `daed166ac2c903b6de379fa349bb16d5627e13986b987f5ddc6715ae4370a885`. The monitored supervisor collected **73** read-only memory/GPU samples.

| Live guarded dense-Q4 fact | Measurement |
|---|---:|
| Initial unused Linux container/cgroup memory after source checksum and read-only page-cache eviction | **29.444 GiB** |
| Lowest sampled unused cgroup memory before stop | **9.992 GiB** |
| Observed cgroup headroom reduction | **19.452 GiB** |
| Lowest sampled *system* available physical RAM | **29.717 GiB** |
| Lowest sampled free GPU0 VRAM | **14,807 MiB**, versus **14,912 MiB** initial |
| Stronger supervised internal cgroup-pressure stop threshold | **10 GiB** (stricter than original 8 GiB runtime floor) |
| Native server model load and independent complete 200-score certification | **NO; supervised resource abort** |

The dense 21.17 GB GGUF charged the **container memory** during load/mapping before a meaningful GPU tensor offload occurred. Host-level `MemAvailable` near 29.7 GiB was misleading: the container cgroup had only 9.992 GiB unused. The supervisor stopped only its owned server subtree before the frozen 8 GiB reserve was breached. No row should be recycled into another speed session or interpreted as a zero-throughput quality result. This is a **measured resource-capacity failure**, not a poor model result. Do not retry the same 30-GiB-cgroup configuration with unchanged model, context and four-way serving in the hope of success.

The host-size conclusion is bounded: at least 19.452 GiB of cgroup headroom was consumed during incomplete native dense loading, before all runtime/KV/cache demands. Sustaining an 8 GiB original minimum plus a 2 GiB guard and additional post-load room requires **more than 29.444 GiB unused cgroup** in this measured configuration; a reproducible isolated Windows or equivalently instrumented test runner should provision at least **48 GiB truly unused container/host memory**, ideally **64 GiB+ free physical RAM and available commit**, and enough freely allocatable VRAM for a 21.17 GB dense model plus original four-way context overhead (prefer **48 GiB VRAM**). These are conservative **run-host specifications**, not verified sufficiency claims; fresh read-only admission and loaded-model memory peaks are mandatory.

## Triple comparison of precisely the same `gsm8k-0` on the same original Q4 SHA

A third private model session [`nikmarco/reap128v2-single-t4-parity-20261026`](https://www.kaggle.com/code/nikmarco/reap128v2-single-t4-parity-20261026) was executed using the original exact flags and same frozen system/user prompts and greedy decode settings, but with a **single visible Tesla T4 GPU** (`CUDA_VISIBLE_DEVICES=0`) rather than the earlier two T4s. It safely loaded the original GGUF with sampled GPU usage **11,447 MiB on T4 0, zero on T4 1** and post-evaluation available cgroup memory **17.968 GiB**. Its full original native receipt `F:\reap-v2-single-t4-parity-evidence-20261026\reap-v2-kaggle-q4-single-t4.json` has SHA `980a40cc2abd490ccefe9bc2ff00ef36505fbe8e2e03ca8046155dba9d4a7a1b`. Replayed the **original frozen `reap_eval._score`** on all three one-item records; each had an actual natural EOS stop.

| Canonical official `gsm8k-0` prompt | Windows RTX 5060 Ti | Kaggle two T4s | Kaggle *one* T4 |
|---|---|---|---|
| Original scorer correctness | **false** | **true** | **true** |
| `finish_reason` | stop | stop | stop |
| Completion tokens | 242 | 259 | 259 |
| Decoded text SHA-256 | `9d504106e27bd7c173b0ffe31b6bb9e974d0ceafc8b04718e4610d087eb7c03b` | `26d82eaf478d9ccdff0030f797d546b9008aa4a1f4dd64a8683a3ff3bae6ae0a` | **the same** `26d82eaf478d9ccdff0030f797d546b9008aa4a1f4dd64a8683a3ff3bae6ae0a` |

The one-T4 and two-T4 responses and correctness matched *exactly* on the observed input, so two-way splitting is **not necessary** to explain the observed Windows-vs-Linux result difference on `gsm8k-0`. Hardware architecture, native CUDA implementation, allocator, compiler and/or numerical kernels are still unisolated causal candidates. A single test cannot establish broad parity, nor can an externally *more favorable* score justify scientific promotion. Do not choose a device topology because it makes one score favorable, or silently substitute an architecture/runner in Amendment 4.

## Original frozen selection replay

The exact original `chowder_batch.reap_select` was independently rerun on the preexisting admissible historical evaluation root `C:\llm-work\reap\eval-v2-isolated-20261025`, original v1 and BF16-v2 observation files and unchanged calibration root. New exclusive output `F:\reap-v2-gen0-independent-selector-replay-20261026-0145.json` has SHA-256 `e36aa022c416a40308b1d02da4688fb73e8896cbd60ca587e6cb015e83205791`. It returned **`decision.status=BLOCKED`, `chosen=null`, `release_approved=false`**, explaining both original REAP128_v2 and REAP160 full gates remain mandatory. The selector returned nonzero because the selection was blocked; that does **not** mean it passed or failed the original quality benchmark.

The prospective review `docs/quals/reap-v2-paired-ci-prospective-method-review-20261025.md` still documents an unresolved McNemar-style versus paired-Wald one-sided confidence-interval method conflict. No method was chosen from observed results, and the optional v2-v1 replacement route remains unqualified. Historical original REAP128-v1 Gate1 lower bound **−0.03780741197234255** fails frozen **−0.03**. REAP160 speed remains **UNKNOWN** without newly certified original-session evidence.

## Running/queued external evidence (not yet recognized by frozen selector)

- The privately launched source-attested Q4 full two-suite experiment [`nikmarco/reap128v2-q4-full-evaluation-20261026`](https://www.kaggle.com/code/nikmarco/reap128v2-q4-full-evaluation-20261026) runs all 1,319+164 original inputs with the exact original scorer, four-way concurrency, source attestation, actual Linux loopback listener ownership and read-only memory floors. **Do not write its prospective accuracy/throughput until the completed item-level output and per-suite hashes are independently inspected**. Any abort is not complete official evidence, and even complete Linux results remain provisional due to the documented Windows numerical divergence.
- A distinct original candidate first-200 throughput experiment [`nikmarco/reap128v2-q4-first200-20261026`](https://www.kaggle.com/code/nikmarco/reap128v2-q4-first200-20261026) was launched separately under the original same-serving-session policy. It was submitted only after the dense run finished and released the second Kaggle GPU slot. It cannot provide the **candidate/dense ≥1.3×** PASS without a fully admitted, complete contemporaneous original dense comparator.
- Independent original random128_v2 negative control and BF16-derived Q8 candidates have been locally SHA-verified and submitted as separate **private** Kaggle dataset uploads. Do not claim remote availability until the API actually reports a ready dataset, and do not attempt their 19.75-GB-weight CUDA load on the already-measured insufficient 30-GiB cgroup without a new resource-admitted configuration.

Original Windows Spark server port `8767`, supervised Ollama daemon port `11434` and original model router port `8766` were verified alive and listening after the separate cloud experiments. Keep all source GGUFs, base model blobs, private weights, original JSON certificates and histories unchanged. Keep PR #282 open/draft/unmerged and no release/Gen-1 authorization.