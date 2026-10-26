# REAP128_v2 Q4 complete original-population Kaggle evaluation: independent verification

**Decision:** Original frozen Amendment 4 remains **`BLOCKED`, `chosen:null`, `release_approved:false`**. This report records *complete* private Linux-CUDA research results, **not** an admissible Windows-backed Gen-0 PASS. Original weights, scoring/serving flags, thresholds, historical evaluation roots, and private credentials were not changed. PR #282 remains draft/unmerged.

## 1. Immutable identity and complete evidence

The exact published 40-layer / 128-expert / top-8 original REAP128_v2 GGUF, SHA-256 `2d64b786461b99a335fdbbdf7c4d3a6758b21409bce755751714ab7cd8f63bf7`, was hashed **inside Kaggle** against the mounted **private** original Q4 dataset. Its Q4 source-construction receipt SHA is `cf16fbf9e6926719bab873bd74b7062fa22365a280451ce3db9a9f8dc3031f1c`. Original full canonical input digests:

| Frozen suite | Population | Rendered source SHA-256 |
|---|---:|---|
| GSM8K | 1,319 | `913ca81f724cba16a07099608047e7522b00910c1c7fe13499c501ac9e4bad89` |
| HumanEval | 164 | `49fb688c6788b0531803dfd1cb85a69926ec5a7179e74cdfffc53ba52a92b420` |

Native source was pinned to the same original upstream llama.cpp Git commit as the Windows binary, `7609846557c50f9d984719a9e1e8c5f3d02f807b`; the **different Linux binary** SHA is `f10032228f1d80b0f6639cdcc7194bbc6638368641cac7f1396775b5186d`. Due to a missing Kaggle `CUDA::cuda_driver` imported CMake target, Linux was explicitly built with `GGML_CUDA_NO_VMM=ON`. It is **not** byte-identical or numerically validated as equivalent to the original Windows build. Frozen serving flags were kept exactly: `--fit on -np 4 -c 16384 --jinja`. Same scorer SHA `d2e59aca5c0a621bac8d4979d522c9b6107f811610567d21c92a2b53960860eb`; evaluator SHA `9736714f21834a45c23599a50fa92ef3bebc6cf70967aca7e2329d26b4d73ee1`.

The completed original cohort and rows are in **private** Kaggle notebook [`nikmarco/reap128v2-q4-full-evaluation-20261026`](https://www.kaggle.com/code/nikmarco/reap128v2-q4-full-evaluation-20261026). **Single original serving-session ID:** `a2a6b83e345145839cc333561b2f3baf`. The original native server log was separately downloaded and rehashed in full, SHA `5dfd24b1beac5445cd194b684bdadae0cd37cbe6d34a6337f9c6b1e55e8b9b48`, and confirms `offloaded 41/41 layers to GPU`, `n_ctx=16384`, `n_seq_max=4`, `n_ctx_seq=4096`, `160 MiB CUDA0 + 160 MiB CUDA1` KV buffers, model-ready and loopback listening. Different GPU placement from original Windows is disclosed.

### Fully completed results, with independent per-item audits

| Cohort | Original recorded `correct` | Independent replay | EOS terminated | Score | Wall seconds |
|---|---:|---|---:|---:|---:|
| GSM8K, all 1,319 | **1,229** | All 1,319 independently re-scored with exact original last-number match | 1,281 | **93.17665%** | 4,639.150 |
| HumanEval, all 164 | **151** | **All 164** generated programs independently re-executed in a separately submitted private CPU-only Kaggle notebook under restricted subprocess privileges; **164/164 agreed** | 162 | **92.07317%** | 457.068 |

Original complete GSM8K scored-rows SHA `3b09d337f567729192e9c302ec698cf1e34d7451538847939e3bf9a36a9557db`; HumanEval SHA `92ec6439be77a7dbeb1a80448a0bb664e3cb6113a9cc0d356f7dbe300db69c43`. Both files have the full unique canonical task IDs and same session. Original full machine receipt SHA `8f9ed207f0fa314281b6ddb9ed62e7ccd7f267a602ac5d1336f393202810e392`. Independent Windows-side complete-cohort source/session/scoring audit: `F:\reap-v2-q4-complete-row-provenance-independent-20261026.json` SHA `b3ae5b8d76377c66ff90b2ee73dc6266d4568b425737c3adb04c21a5e6b5249d`.

**Independent HumanEval security and exact scoring:** A different **CPU-only**, **internet-disabled**, **no-model-weights** private Kaggle notebook [`nikmarco/reap128v2-humaneval-independent-20261026`](https://www.kaggle.com/code/nikmarco/reap128v2-humaneval-independent-20261026) re-executed all 164 generated programs with the original frozen input source/code-extraction semantics, but additional separate low-privilege subprocess limits (CPU, address space, file size, no root privilege where available, Python isolated mode). Its result had **151 passes, 13 nonzero exits, 0 timeouts, 164 matching original recorded booleans**, output receipt SHA `a413c2a5f700dd44cdac5ecfc419d5286c1a4110a3ac52f1fb8ebf0cdcee13ea`. This establishes robust **Linux row-score reproducibility**, **not** numerical output parity with Windows or an approved original Gen-0 gate.

### Separate same-session first-200 Q4 candidate speed cohort

Private original candidate [`nikmarco/reap128v2-q4-first200-20261026`](https://www.kaggle.com/code/nikmarco/reap128v2-q4-first200-20261026), serving-session `1464485da31e4616963278e614523333`, produced all **200** frozen first GSM8K results. Original row SHA `95d54cfb1b204eff5855e2811142f9d721549c6212443e934eee6fc9107b0a0c`; receipt SHA `a2ad78bffca1c305fecad830c44f283f1f0b3ba044670264a4e4cecc0234a615`. Independent strict row/source/session rehash and scoring reproduced **189/200 correct**, **193/200 naturally terminated**, **72,872 output tokens**, and **718.474 seconds** in the registered **four-way** session. Descriptive whole-cohort throughputs: **101.426 generated tokens/s** and **0.278368 completed questions/s**. These are genuine measured candidate statistics, not an official speed **ratio**.

The original `dense_v2_q4` same-first-200 comparator source SHA `2c6d9466129e8802839566b35728e2506aaf1e8705cd137009e2f08feb9dd049` was transferred privately, launched under the same nominal flags, and **resource-aborted** without complete serving/inference, at 9.992 GiB remaining container memory. Its exact independent 73-sample receipt SHA `b2ef325bb9c3165439e669326746fdb9165a5a66975890c99cbfd56dda8c6c0a`. **No certified dense throughput, no speed ratio and no ≥1.3× PASS.** Never combine its aborted rows or sessions with completed candidate rows.

## 2. Model footprint, numerical evidence and measured native resource pressure

| Model | Original GGUF bytes | SHA-256 prefix | Scientific role |
|---|---:|---|---|
| BF16 intermediate | 37,122,439,488 | `80341aa7` | Frozen source construction; full CUDA forward parity unmeasured |
| Q8_0 | 19,748,435,968 | `89dcaf33` | BF16-derived source; full quality/CUDA parity unmeasured |
| Q4_K_M | 11,563,915,264 | `2d64b786` | Completed **provisional** Linux full quality and first-200 candidate |
| Original dense Q4 reference | 21,166,757,984 | `2c6d9466` | Complete first-200 speed **not available**: safety abort |

The original 733-tensor Q4 construction witness accounts for **18,543,997,568 total parameter-weight elements**; **2,437,870,208 always-present** and **16,106,127,360 stored routed experts**. In the 40-layer top-eight-of-128 architecture, **1,006,632,960** routed-expert elements are selected per token plus always-present elements, giving **3,444,503,168 approximately active weight elements/token** (18.575% of stored). These are not a measured FLOP count; actual attention/control/gating execution must not be equated with stored parameter count.

Previously complete, immutable **BF16↔Q8** dequantized expert sampling (120 tensors × 4 × 256 = 122,880 values) had **MAE 0.0000478359017317113**, **RMSE 0.000058366718970777665**, max error **0.00028896331787109375**, SHA of numeric receipt `a7d236af0fdad53b65f4b2271f895a05b14f84653bf40efef52c00aa18e0920c`. Previously complete **BF16↔Q4** same-size deterministic sampling had **MAE 0.000555533053235043**, **RMSE 0.0007138570046274973**, max **0.003823518753051758**, source numeric receipt SHA `925854285a579e7cd49232eb5a4a3d73f14e9a2ae12ee5b5a599f7483157a5d0`. These are separate, descriptive weight fidelity windows and **not** matched-token Q8/Q4 model-quality or CUDA parity claims.

Native completed full-candidate session had **1,054 observed resource samples**. Minimum **21.055 GiB physical RAM available**, **9.451 GiB unused cgroup memory** (runtime floor preserved at 8 GiB), **8,955/9,067 MiB observed GPU free** versus **14,912/14,912 MiB prelaunch**. These support *observed* GPU allocated/free differences **5,957/5,845 MiB**, not hard instrumented instantaneous high-water allocations. Its native 4-way GPU split showed 41/41 layers offloaded and 160 MiB KV on each GPU. Initial unused cgroup memory was **29.443 GiB**, so the maximum sampled reduction to 9.451 GiB is **19.992 GiB** of *container headroom*, not an interchangeable physical-RAM/commit figure. There is **no Windows commit counter on Linux**.

The dense 21.17-GB original reference failed safe before GPU tensor offload, while guest `MemAvailable` still read 29.717 GiB: the primary measured failure is **host/container-accounted model-load pressure**, distinct from device VRAM. See `docs/quals/reap-v2-dense-fail-safe-and-single-t4-parity-20261026.md`. A future original topology test must genuinely provision separate, bounded **≥48 GiB unused host/container headroom** (prefer ≥64 GiB free physical/commit), appropriate device VRAM (prefer a genuine ≥48 GiB CUDA accelerator for dense fully-GPU fit), the **original 7 GiB RAM, 8 GiB commit runtime floors**, and **12,288 MiB free CUDA admission**, with original `--fit on -np 4 -c 16384 --jinja` unchanged. These hardware figures are **candidate test configurations, not proof of successful model fit**.

## 3. Numerical platform non-equivalence and exact frozen gate status

For the same original `gsm8k-0`, fixed prompt/decode/model GGUF SHA, original Windows RTX 5060 Ti returned a terminated **incorrect** answer, while Kaggle dual- and single-T4 returned an identically hashed terminated **correct** answer. Original complete evidence documents the different CUDA-VMM build, GPU architectures and KV-placement. A single-T4 test rules out *requiring* dual-GPU splitting for the observed discrepancy, **not** all causes. The Linux full-population scores cannot be promoted as automatically Windows-equivalent, especially after observing one favorable score difference. Do not choose a numerical method or execution platform based on outcomes.

| Frozen assessment | Admissible status | Grounded basis |
|---|---|---|
| BF16 40×128 construction / Q8 and Q4 tensor/source checks | **PASS construction, not performance** | Full source and GGUF construction receipts, numerical samples |
| REAP128-v1 D0 / Gate0 / Gate1 | **PASS / PASS / FAIL** | Original paired Gate1 LB **−0.03780741197234255**, frozen threshold **−0.03** |
| REAP128_v2 candidate full *Linux-population scoring* | **COMPLETE provisional** | Source-bound 1,319+164 with independent Linux scoring, not original Windows gate |
| REAP128_v2 original Windows Gate0/1/2 & fresh dense/random128 control | **UNKNOWN / NOT EVALUATED** | No complete original compatible candidate/control reference evidence |
| REAP128_v2 original dense Q4 GPU-fit, same-session first-200 speed and ≥1.3× ratio | **UNKNOWN / NOT CERTIFIED** | Dense native resource abort; no 200-row reference measurement |
| BF16↔Q8↔Q4 forward CUDA parity and Q8 quality | **UNKNOWN** | Weight dequantization sample is not forward inference equivalence |
| REAP160 historical D0, Gate0, Gate1, Gate2, control | **PASS** as recorded by original selector | Existing historical original evidence, unchanged |
| REAP160 certified original speed | **UNKNOWN** | Old throughputs lack required current `serve.json` and complete original speed-session proof |
| Optional v2-v1 paired McNemar-style −0.02 noninferiority | **NOT CERTIFIED** | Original §4 paired Wald versus Amendment 4 McNemar-style CI definition still ambiguous; prospective independent human preregistration required |
| **Deterministic Gen-0 selection / release** | **`BLOCKED` / `chosen:null` / `release_approved:false`** | Original frozen selector replay, no threshold relaxation or row/session substitutions |

Independent original selector was replayed against **only the still-admissible historical source root** and produced `BLOCKED`, exclusive receipt SHA `e36aa022c416a40308b1d02da4688fb73e8896cbd60ca587e6cb015e83205791`. Original REAP160 speed remains `UNKNOWN` even though historical unbound throughput numbers exist. Original unresolved McNemar-vs-Wald governance remains as detailed in `docs/quals/reap-v2-paired-ci-prospective-method-review-20261025.md`. Post hoc choice is prohibited; optional noninferiority must remain uncertified without independent prospective signoff / genuinely held-out resolution.

**Remaining execution blocker:** the user's original Windows RTX 5060 Ti workstation cannot presently admit the full frozen dense/reference/control workload at its measured RAM/commit/VRAM levels; Kaggle's actual 30-GiB-cgroup two-T4 CUDA environment safety-aborted the frozen dense reference, and its outputs have demonstrated numerical non-equivalence with original Windows. The original random128_v2 negative control and Q8 have separately been submitted as **private** datasets, but no original all-suite same-platform control or Q8 full inference should be claimed without genuine capacity and source-output equivalence. These specific unmet original gates, rather than any arbitrary score threshold change, determine the `BLOCKED` result.

**Preservation:** original private weights, original JSON full results and all immutable construction certificates remain in their protected locations. Original Windows Spark, supervised Ollama and model router were verified HTTP 200 after all cloud experiments. No PR merge, deployment, release approval or Gen-1 training was performed.
