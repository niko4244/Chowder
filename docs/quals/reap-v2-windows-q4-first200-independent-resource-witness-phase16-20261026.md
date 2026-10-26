# REAP128_v2 original Windows Q4 first-200: independent source and resource witness (2026-10-26)

**Scope:** independently inspect an already completed, registered Windows Q4 candidate-only first-200 GSM8K run. This is **not** a full-population quality gate, dense-model comparison, or Gen-0 speed PASS. The original Amendment 4 decision remains `BLOCKED`, `chosen:null`, `release_approved:false`.

The original immutable private source files are under `F:\reap-v2-q4-windows-registered-first200-phase15c-20261026\`. A separate exclusive independent read-only verification receipt was written to `F:\reap-v2-q4-windows-first200-independent-source-and-resource-audit-20261026-phase16.json`, SHA-256 **`5639f303c761d13c8b1375e1be160c5f70f91c85069bb691947649452936006e`**. Preserve these private files without replacement or publishing generated text.

## Measured and verified

| Property | Original measured result |
| --- | ---: |
| Frozen original first-200 canonical GSM8K IDs | 200/200 distinct, no missing or foreign IDs |
| Correct | **187/200 = 93.5%** |
| Naturally terminated | **192/200 = 96.0%** |
| Generated completion tokens | **73,224** |
| Complete wrapper elapsed wall time, including load | **452.171 s** |
| Completion tokens / complete wrapper wall time (not the registered speed ratio) | **161.94 tokens/s** |
| Resource samples | **148**, maximum interval **3.086361 s** |
| Minimum observed available physical RAM | **10.0643 GiB** (original floor 7 GiB) |
| Minimum observed available Windows commit | **14.7239 GiB** (original floor 8 GiB) |
| Minimum observed disk free | **35.7820 GiB** (original watchdog floor 8 GiB) |
| Minimum sampled GPU free during model residency | **2,859 MiB** |
| Initial GPU free, before model load | **14,346 MiB** |
| Original native CUDA offload | **41/41 layers** |
| Native context | **4 sequences × 4,096 = 16,384** |
| Native CUDA KV | **320 MiB** |

Original Q4 GGUF SHA-256 attested in the original frozen inputs, server and terminal: `2d64b786461b99a335fdbbdf7c4d3a6758b21409bce755751714ab7cd8f63bf7`. Original source GSM8K 1,319-item population SHA-256: `913ca81f724cba16a07099608047e7522b00910c1c7fe13499c501ac9e4bad89`. Original serving flags were exactly `--fit on -np 4 -c 16384 --jinja`; deterministic decoding source declared temperature 0, top-k 1, seed 7, 1,024 max completion tokens. The source-bound original session is `e3f9c268b2bb4d958cc55648c0c4c8e0`, server PID `77916`, loopback port `8790`.

The verifier independently recomputed the SHA-256 of original row, input, server-log, serve, summary, resource-sample and watchdog files; compared original cross-file row/serve/summary/session hashes; checked exact first-200 task-ID set, aggregate correctness/termination/token counts, original native CUDA placement and KV log evidence; checked no `correct:true` with `terminated:false`; and verified the complete 148-sample resource series stayed above the original physical/commit floors without a monitoring gap over five seconds. Original rows SHA-256: `f3bae8ada933f0b85dae4d8946542f1901b02ef9e5ffead8411c4d449ea2472e`.

**Limitations:** This witness is an independent verification of source-bound recorded outputs and resource samples, not an independent second model inference run or a fresh full 1,319-item Windows evaluation. The end-to-end token rate includes model startup and therefore must **not** be mixed with another model's evaluation-only token rate to establish the registered ≥1.3× candidate/dense threshold. No original dense-Q4 first-200 comparator is admitted or present, so the speed ratio remains **UNKNOWN**. The full Q8 Linux HumanEval 164-item independent executable replay also remains pending due to previously demonstrated Kaggle sandbox-isolation refusal. Preserve the historical REAP128-v1 Gate1 failure and unresolved optional paired-CI governance without reinterpretation.
