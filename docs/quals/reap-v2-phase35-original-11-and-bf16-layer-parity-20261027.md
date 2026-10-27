# REAP128_v2 phase35 — actual BF16 CPU/CUDA layer evidence and frozen 11-suite ledger

**Date:** 2026-10-27 (America/Chicago). **Branch:** `experiment/reap128v2-test-unblocked-20261026`, based on source-pinned HEAD `3fb79e00145f1d8c12bacdd51a1f5b1dfd40ffb9`. **Selection:** `BLOCKED`, `chosen:null`, `release_approved:false`. Existing PR #282 must remain OPEN, DRAFT, UNMERGED. No Gen-1, merge, release or official benchmark promotion.

## Measured BF16 CPU/CUDA numerical investigation — not D0/Gate0/1/2

Implementation `chowder_batch/reap_bf16_cuda_layer_parity_20261027.py` executes a **real** original pruned REAP128_v2 layer, separately on local CPU and CUDA RTX 5060 Ti, with identical deterministic two-token `torch.bfloat16` input, source model config, attention masks, rotary embeddings and actual original PyTorch/Hugging Face decoder layer. It does not generate model completions, execute untrusted code, change weights, load an entire 37.98GB checkpoint, stop services or bypass the registered native `--fit on -np 4 -c 16384 --jinja` quality/speed protocol. Neither the synthetic activations nor the CPU layer is a substitute for frozen original calibrated D0.

Local checkpoint `G:\llm-cold\bf16\Qwen3.6-REAP128-V2`, source-confirmed 40 layers, 128 experts/layer, top8 routing. **Original independent full-source equivalence proof**: `docs/quals/reap128-v2-full-source-equivalence-20261025.json`, SHA-256 `0746c8c06764f784fa7134ecdc4a18954a761372f865256c9e0120588cfbbc7d` (26 original shard SHA pins). Every selected shard in each phase35 source-pinned layer probe was **rehash-checked against this already published proof** before inference; the untouched unselected 24 or 25 shards were *not* rehashed by these new runs. Actual selected layer tensors were separately hashed. This is genuine selected-layer source ancestry, not a new full 26-shard or all-40-layer audit.

| Real paired BF16 output evidence | Layer 0 (linear attention) | Layer 3 (full attention) |
|---|---:|---:|
| Original layer outputs compared | 4,096 | 4,096 |
| Exactly equal BF16 elements | 3,816 | 2,390 |
| Max absolute difference | 0.00390625 | 0.0078125 |
| Mean absolute difference | 0.0000575576 | 0.000755504 |
| RMS absolute difference | 0.000323262 | 0.001400875 |
| Original shard IDs checked | 00001 and 00002 of 00026 | 00003 of 00026 |
| Original source-selected SHA check | matched 2/2 | matched 1/1 |

The first real layer3 research receipt was created at `F:\reap-v2-bf16-cuda-layer-parity-20261027-0301\original-layer3-cpu-cuda.json`, SHA-256 `32173fe78f231f4bd5a096deec09a1c5844712798f1a1bdd68843ec6b49e39fc`. The **later source-pinned + physical NVIDIA GPU census** layer3 replay had the *same exact numerical values* and a separate receipt `...\original-layer3-source-pinned-physical-gpu-cpu-cuda.json`, SHA-256 `9228f76bb505f95680b6b49163644cec57a1eecbe7bd1d83a35aea6369a95f6e`. Native GPU0 (RTX 5060 Ti; 16,311 MiB total) was sampled at **14,526 MiB free before, 14,240 MiB free afterward**. The equivalent final source-pinned/physical-GPU layer0 run is `...\original-layer0-source-pinned-physical-gpu-cpu-cuda.json`, SHA-256 `0a632ba0f87db8604159728e79bc5edb3a299cdc195e3fa6150d6599d053c22b`, and **14,458 MiB free before, 14,259 MiB free after**. The earlier source-pinned layer0 receipt without native physical census is `...\original-layer0-source-pinned-cpu-cuda.json`, SHA-256 `64ee11f09f3d3dacca4eb6bea3ba9dcb42bcfbe2b866469edb5418fbea6528f9`. Host-wide free values are **not** isolated model peak allocations.

The experimental code refuses altered Git-frozen proof bytes, divergent selected shard SHA, wrong 40/128/top8 geometry, symlinked missing weights, missing actual GPU BF16 capability, unsafe Windows physical/commit floor, wrong GPU identity, or native physical GPU free VRAM less than 8 GiB. Its receipt reports `entire_40_layer_logit_parity_proven:false`, `original_windows_native_gguf_gate_eligible:false`, `quality_gate:"UNKNOWN"`, `gen0_status:"BLOCKED"`. **No inference is made about end-to-end logits, 40-layer CPU/GPU numerical equality, BF16 vs Q8/Q4 GGUF quantization accuracy or official generated-code HumanEval outcomes.**

## Established original model and task research (never cross-pool)

| Frozen evidence | Verified observation | Qualification boundary |
|---|---|---|
| Original Windows Q4 GSM8K 1,319 | **1,224/1,319 = 92.797574%**, all item texts independently rescored, native 41/41 CUDA, zero CPU expert offload, immutable receipts | Candidate accuracy only; **not** one of the 11 registered frontier suites; no dense comparison |
| Original Windows Q4 GSM8K first200 | **189/200**, frozen per-item mean predicted/s **40.7702977** | Matching original dense first200 **missing**; ≥1.3x speed UNKNOWN |
| Research Linux Q4/Q8/random128 GSM8K 1,319 | Q4 **1,229**, Q8 **1,220**, random128 **964** correct | Original source-backed research on *another host*; not pooled with Windows official gate |
| Research Linux Q4/Q8/random128 HumanEval 164 | Q4 **151**, Q8 **152**, random128 **45** reported correct | **Independent generated-code execution not attested**, research only, not an admissible original Gate2 |
| Original Q8 HLE first session | **99/128** real answered/scored items; normalized heuristic matches **0/99** | Exact missing29 private GPU continuation last confirmed RUNNING; no official HLE judge and no false single-session merge |
| Original Q4 HLE text subset | **128/128** answered; **3/128** *nonofficial normalized heuristic* matches | No official HLE judge score |
| AutomationBench original Q4 public pilots | **Six tasks strict0/6** on original 64K profile; **two tasks strict0/2** on distinct 73,728-token profile | Two different protocols, **do not pool**; Q8 original600 full score absent |
| AutomationBench original CPU-only sealed source/wheels | Original locked 108-wheel and public600 input staged; private kernel COMPLETE | Original JSON output API HTTP429, so wheel install / 600 source loader **UNKNOWN**; new CPU gate explicitly BLOCKS Q8 GPU runners |

Native original model storage **18,543,997,568** weight elements, estimated **3,444,503,168** active weight elements/token (about 18.575%, **not** measured FLOPs); 40 decoder layers, 128 routed experts/layer, top8 per token. Original GGUF weights: BF16 intermediate **37,122,439,488** bytes, Q8_0 **19,748,435,968** bytes, Q4_K_M **11,563,915,264** bytes. The pruned HF BF16 shard tree is separate (**37,981,266,120** file bytes). Original physical per-model GPU/RAM peak residency and BF16 all-layer CUDA/logit parity remain **UNKNOWN**; do not confuse GGUF file bytes, host-wide GPU-free minima, active element fraction and actual per-token FLOPs.

## Frozen original eleven frontier suites: evidence status, not synthetic scores

Source registry: `benchmark_chart/references.json`, schema `reap128-v2-eleven-benchmarks.v1`. A named task is not credited as complete merely because a partial pilot or source-only tool exists.

| # | Original preregistered suite | Actual eligible original score now | Exact blocker / next admissible step |
|---|---|---|---|
| 1 | HLE text/no-tools | **UNKNOWN** | Real original Q4 128/128 research and Q8 99/128 first session; missing29 distinct private native session and official HLE judge still absent |
| 2 | Terminal-Bench 4.0 | **UNKNOWN** | Need exact registered Harbor agent/Claude Code tool harness, original tasks, disposable terminal containers and official grader |
| 3 | DeepSWE v1.1 | **UNKNOWN** | Need original 113-source repo-task cohort, mini-SWE agent execution, source-locked repo sandbox/verifier |
| 4 | ProgramBench | **UNKNOWN** | Need original 200 reverse-engineering tasks, executing agent, original binary/docs and hidden behavioral tests |
| 5 | NL2Repo-Bench | **UNKNOWN** | Need original 104 repository generation task cohort, OpenHands-compatible isolated tool host, builds/tests and official grader |
| 6 | AutomationBench v1.0.6 | **UNKNOWN** | Original Q4 separate 6+2 attempts only, CPU600 proof output unrecovered (429), Q8 prepared notebook missing offline wheel mount/isolated dependencies; no original600 worldstate grader completed |
| 7 | Agents' Last Exam v1.1 | **UNKNOWN** | Need source-locked 122 tasks, agent/tools, independent environments and official scoring |
| 8 | ApexBench v1.1 | **UNKNOWN** | Vendor-labeled original exact harness/source still unverified; no similarly named substitute permitted |
| 9 | Chartography (with tools) | **INELIGIBLE / UNKNOWN** | Original Q4/Q8 text-only GGUF cannot ingest required original chart image inputs; no OCR or multimodal proxy substitution |
| 10 | BabyVision (with tools) | **INELIGIBLE / UNKNOWN** | Original text-only model cannot consume registered visual inputs or tools, no fabricated zero or replacement-model result |
| 11 | ZeroBench-main (with tools) | **INELIGIBLE / UNKNOWN** | Original image input, tools and pass@5 protocol not supported by frozen text-only candidate |

The **11-suite scorecard is intentionally not 11 scored suites**. Missing original graders, image modality or independent execution evidence are UNKNOWN/ineligible, **not 0%, PASS or complete N/T placeholders**. Likewise a nonofficial HLE normalized-match heuristic is not the official HLE LLM judge.

## Original gates and selection

- **D0:** full frozen BF16 704-sequence calibration and 40-layer perplexity identity; new layer parity does not replace D0.
- **Gate0:** original dense/Q4 1,319-item termination and trigram comparisons; **UNKNOWN** without comparable full dense original evidence.
- **Gate1:** strict paired sample-Wald lower bound > **−0.03** on original full 1,319 GSM8K; complete admissible dense/v1-matched source outcomes not present. Historical REAP128-v1 bound **−0.03780741197234255**, **FAIL**.
- **Gate2:** strict paired sample-Wald lower bound > **−0.10** original complete independently executed HumanEval164; **UNKNOWN** while generated-code sandbox proof is not approved.
- **Random128** original paired negative control LB>0; original Linux research point comparisons are descriptive, not host-identical frozen release evidence.
- **Speed:** ≥**1.3×** original first200 on same hardware/protocol; original Windows dense first200 not completed, **UNKNOWN**. Earlier Linux dense 21.17GB GGUF cgroup load aborted at 9.992GiB unused; never lower resource floor to get a score.
- **Amendment4 optional v2 replacement:** needs original complete independently adjudicated paired CI for Δ>−0.02 on both sources. Wording ambiguously invokes one-sided paired McNemar versus registered paired-Wald; **NO independently signed prospective method**. No retroactive method selection.
- **Fallback:** original REAP160 matched original quality/speed UNKNOWN. The original deterministic selector cannot assert release or promote an unverified fallback.

**Next source-eligible work:** finish exact two-session Q8 HLE missing29 authentic receipt only after existing job COMPLETE; recover actual private CPU600 receipt without repeated API429; qualify disposable independently observed Python3.13 HumanEval subprocess isolation before executing *any* generated code; measure full numerical/logit agreement using an admitted source-bound runner; finish original dense first200 on a genuinely larger eligible host; obtain independently approved *prospective* Amendment4 interval method. Preserve all 11 task identities and full frozen scorer, serving flags and safety floors. Selection remains `BLOCKED`, `chosen:null`, `release_approved:false`.

The 2026-10-27 live host-side independent-witness preflight confirmed the pinned local `python:3.13-slim` container image (`72036a705a7cff38924efd674e60654666cb283faf7b3fad160b34007f7395bd`) and existing LiteLLM/PostgreSQL containers remained up. However, `podman machine ssh` was **REFUSED** by its SSH client: the Windows private key's permissions were too open and the key was ignored. Consequently the required outside-workload raw `/proc` namespace, cgroup, memory, process-tree and no-egress witness **could not be captured**. We did **not** change Podman machine SSH keys/ACLs, restart services, start generated code or set the frozen attestation's missing two approval SHA pins. This is a measured infrastructure blocker, not a HumanEval accuracy or sandbox PASS.

**Focused and original-gate regression at this wave:** `147 passed` across exact HLE continuation, locked AutomationBench wheelhouse/CPU receipt, original Windows Q4 full1319, old HLE audit, synthetic HumanEval-v2 attestation and new BF16 layer parity; new/changed Python files also passed Ruff and compileall. The test pass certifies guarded code paths, *not* absent external source receipts or official benchmark scores.
