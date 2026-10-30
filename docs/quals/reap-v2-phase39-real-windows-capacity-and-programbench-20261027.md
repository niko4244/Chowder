# REAP128_v2 Phase39 — live original-model frontier score, Windows capacity and ProgramBench200 source

**As-of independent private score checkpoint: 2026-10-27 12:06 CDT. Objective still OPEN. No original Gen-0 selector candidate is approved.**

Status: **BLOCKED / chosen:null / release_approved:false**. Frozen original scientific protocol, original weights, registered four-slot --fit on -np 4 -c 16384 --jinja and the eleven original source identities are unchanged. PR niko4244/Chowder#282 must remain OPEN DRAFT UNMERGED. Do not initiate model training, Gen-1, deployment or a release.

## 1. Genuine original-model Q4 AutomationBench v1.0.6 execution

The independent scorer reopened all authentic official WorldState per-task records under F:\reap-v2-q4-original-automationbench600-checkpointed-20261027-0953. Registered source population: 600 original public items from official commit 4a8e1061254004d9dac807054eed33fad7d1ff14, prompt SHA 1c8feeec1f680b05b392a5d00fc634edb143c3e7bdd152c476af50cebcec57b5. Exact original Q4 GGUF SHA 2d64b786461b99a335fdbbdf7c4d3a6758b21409bce755751714ab7cd8f63bf7, 11,563,915,264 bytes. Original per-item evaluator --num-examples 1 --skip N --max-steps 44 --toolset api is source-identical and its pass/fail rubric is unchanged.

| Source-verified observed metric | Value |
|---|---:|
| Complete original items with immutable official evaluator JSON and independently rescored strict checkpoint | **27/600 (4.5%)** |
| Original strict full-task passes for this **non-random prefix** | **0/27**, NOT 0/600 |
| Required positive original assertions passed | **7/108** |
| Exclusion/negative assertions passed | **160/160** |
| Actual native REAP Q4 model calls | **897** |
| Actual original-world tool calls | **1,218** |
| Evaluator input tokens, including repeated conversation | **15,483,214** |
| Evaluator output tokens | **302,589** |
| Aggregated official model time | **6,458.708 s** |
| Next original case | index27 (28th task), unscored at checkpoint |
| Official full600 score | **UNKNOWN, run ongoing** |

Private immutable 12:06 research receipt: F:\reap-v2-q4-original600-source-bound-live-score-20261027-1206.json; SHA256 **B6A808E447DECACE7A7C60E3788435F842B74ABC8323E79D61CCD83A89E33D65**. This is actual original REAP inference, not synthetic scoring or a smoke test. The model/guard serving profile is a separately named experimental run --fit on -np 1 -c 73728 --repeat-penalty 1.1 --dry-multiplier 0.8 --temperature 0.7 --jinja, so its results cannot fill the frozen four-slot Gate0/1/2 cells. The full600 task-order prefix is not a random sample and descriptive 0/27 is NOT a claim about the complete population.

The earlier initial original monolithic run (authentic model and harness, 3/570 groups) never flushed a valid full official result, was stopped, and is NEVER pooled with the present independently checkpointed native serving session. A new local model/official evaluator must not be launched while the existing native runner is active. The existing safe hourly REAP qualification monitor was previously updated to look up existing scores and resume only after the source-owned server fully exits and resource admission passes. The present transcript has not paused, restarted, reconfigured, terminated or duplicated this genuine run.

## 2. Actual Windows memory, commit, KV cache, GPU residency and active computation

Added native Windows sidecar chowder_batch/reap_original600_live_windows_resources_20261027.py: read-only GetPerformanceInfo source for committed-memory total/limit, physical RAM, running original process Windows working set/private bytes, cumulative page faults, one physical NVIDIA GPU, driver-reported host GPU memory and guarded source-owner manifests. The sidecar rejects PID reuse, server/model/flags change, fake/unowned output, OS mismatch, nonmonotonic time, counter resets, missing historical owner SHA, inconsistent WDDM usage, or absent startup allocator evidence. NO model weights, config, ports, processes, GPU workload or scorer are modified.

Source-bound **8-sample real active Windows Q4** host-only resource receipt: F:\reap-v2-q4-original600-actual-native-windows-ram-vram-20261027-1155.json SHA256 **9991F92EB45607776A4C460C3BEBDC84D57AABA131308E3B07332C9EC69F618B**.

| Native source-matched measured window (15.72 s) | Value |
|---|---:|
| Maximum original native-server working set | 25,937,989,632 bytes (~24.16 GiB) |
| Maximum native private committed bytes | 24,651,526,144 bytes (~22.96 GiB) |
| Minimum available actual Windows physical RAM | 13,424,844,800 bytes (~12.50 GiB) |
| Minimum available Windows commit headroom | 33,556,549,632 bytes (~31.25 GiB) |
| Maximum total Windows committed bytes | 78,550,978,560 |
| Maximum physical host GPU used | 15,817 MiB |
| Minimum physical host GPU free | 234 MiB |
| WDDM VRAM reserved outside driver used/free counters | 260 MiB |
| Incremental native process page faults | 98,870 |
| Original completed scored tasks at beginning and end | 22 and 22 |

This is measured native host and process observation, **NOT an exclusive GPU allocation or whole-run peak**. Another supervised Ollama instance may have been active. The running Q4 server and other user services are left undisturbed. The 234MiB host GPU headroom makes a concurrent 16GB-local Q8 or dense evaluation UNSAFE. The original frozen 7/8GiB runtime RAM/commit safety reserve and independently required start GPU floor have NOT been lowered.

**Separate independent native startup allocator receipt**, sampled against the very same live original process and its immutable GGUF: F:\reap-v2-q4-original600-allocator-kv-resource-20261027-1158.json SHA256 **A88877A031A02DFB8EA98A3A84F8A30BD2AC0EB2BE3D9C4101DF040C195CA647**.

| Native llama.cpp original allocator proof | Measured or configured |
|---|---:|
| CUDA model tensor buffer | **10,619.88 MiB** |
| CPU-mapped model metadata/tensors buffer | 397.85 MiB (NOT CPU expert compute) |
| CUDA f16 KV buffer | **1,440.00 MiB**, 73,728 cells, 10 full-attention layers |
| CUDA recurrent-state buffer | 62.81 MiB |
| CUDA compute buffer | 172.05 MiB |
| Host-pinned compute buffer | 80.02 MiB |
| Native prompt cache RAM ceiling | **8,192 MiB CONFIGURED LIMIT**, NOT measured actual/peak allocation |
| GPU offload | **41/41** layers (39 repeating, output, other) |
| CPU expert offload | **ZERO** |
| Architecture | 40 decoder blocks, **128 experts** per MoE block, **8 active** per token per MoE block |
| Native concurrency and context | 1 slot, 73,728 context (experimental only) |

The recorded BF16 source-equivalence geometry is **18,543,997,568 stored parameters**, approximately **3,444,503,168 active parameters per token** (sparse expert routing, not a standalone dense-weight count). Original BF16 pruned source safetensors 37,981,266,120 bytes; intermediate BF16 GGUF 37,122,439,488 bytes; published Q8_0 GGUF 19,748,435,968 bytes (SHA 89dcaf33b61f59c83a31b718ba98b73c33437d0320ae0cc56737a5d7e43519c6); Q4_K_M GGUF 11,563,915,264 bytes as above. Original BF16 40/40 single-layer CPU/CUDA diagnostics (all26 weight shards independently rehashed) exist but DO NOT prove chained full-model logits parity or satisfy original Gate0.

## 3. Authentic ProgramBench200 registered source and full original scorer denominator

Downloaded **OFFICIAL RELEASE v1.0.1** (git commit **1fe64c87a1318998636850cead00b0db80e2d928**) from https://github.com/facebookresearch/ProgramBench into a separate **F:** read-only development checkout, no edits to the Chowder weights or source evaluator. The release contains **201** source task directories. One is upstream's **testorg__calculator.abc1234** synthetic unit-test fixture, explicitly referenced in the upstream official tests/test_eval.py (SHA 0445fd47569492247776f10278639cf281dda0724d39c2113a80a8fb84002fdb). Excluding exactly that verified fixture leaves the registered **200 real tasks**.

All 200 task.yaml files, source hashes, original repository/commit identities, tests.json files, original ignored-branch and ignored-test lists were loaded and validated by chowder_batch/reap_original_programbench200_population_20261027.py. There are **248,089 nonignored genuinely scorable behavioral tests** after using original source's explicit exclusions, matching official ProgramBench's approximately 248k published scope. Source-bound ordered original-task census SHA256 **27a647cf7c6aa45e510b94c9dbc3ebdd57dc3d8c6836e0856c7b59b9986c447a**. Private immutable full200 metadata receipt: F:\reap-v2-original-programbench200-248089-source-census-20261027-1203.json SHA256 **C8B0CD2367B578D8ABB1EC1570933C69D1F73187283AAA9027EE2758A936DB28**. This is a verified ORIGINAL dataset and grader denominator, **NOT one REAP-generated ProgramBench submission or score**.

Original ProgramBench requires a Linux x86_64 cleanroom task image, **agent has no internet, no program source, no decompilation and no hidden tests**, a complete agent-produced source tree in submission.tar.gz, and isolated behavioral-verifier execution (official programbench eval); full200 actual execution cannot be safely admitted on the shared 16GB GPU while the original Q4 AutomationBench server is still live. Never report 0/200 merely because no full200 REAP model submissions exist.

## 4. All eleven original frontier suites — actual graded model state

| # | Registered authentic suite | Q4 current source-verified result | Q8 current source-verified result | Completion status |
|---:|---|---|---|---|
| 1 | Humanity's Last Exam, no tools, text only | Original128 model answers; **3/128 nonofficial heuristic** only, NOT official judge | Original99/128, **0/99 heuristic**, new missing29 second Kaggle session aborted on genuine cgroup mmap load with zero valid rows | Official full HLE score UNKNOWN; Q8 missing29 needs larger admitted cgroup and exact authorized original judge |
| 2 | Terminal-Bench 4.0 | Source-matched original Harbor v4 task attempted, three actual Q4 sandbox action rounds; 8192 context overflow, **0 legitimately graded tasks** | Not run | Full66 Harbor/Terminus-2 score UNKNOWN; need admitted >=32k isolated context and original verifier |
| 3 | DeepSWE v1.1 | No official 113-task agent verifier output | Not run | UNKNOWN |
| 4 | ProgramBench original | Authentic official **200 tasks/248089 scorable tests source certified**, ZERO REAP code submissions | No REAP code submissions | Actual model grade UNKNOWN, no synthetic fixture counted |
| 5 | NL2Repo-Bench | No original complete104 agent verifier score | Not run | UNKNOWN |
| 6 | AutomationBench v1.0.6 | **27/600 independently scored source-bound, 0/27 strict, live** | No original qualified full600 result; Kaggle Python3.13 CPU result endpoint HTTP429, one rate-aware pull attempted, NO duplicated job | Full600 UNKNOWN, Q4 active and protected |
| 7 | Agents' Last Exam v1.1 | No official full122 grader | Not run | UNKNOWN |
| 8 | ApexBench v1.1 | Exact source/registered identity still unverified | Not run | UNKNOWN, no similarly named benchmark substitutions |
| 9 | Chartography with tools | Original text-only checkpoint lacks native chart image encoder | Same original architecture | Native multimodal INELIGIBLE; no OCR/VLM substitution or official score |
| 10 | BabyVision with tools | Original text-only checkpoint lacks native image input | Same architecture | Native multimodal INELIGIBLE |
| 11 | ZeroBench-main with tools | Original text-only checkpoint lacks native image input | Same architecture | Native multimodal INELIGIBLE; official pass@5 UNKNOWN |

No suites are reported as FULLY GRADED merely because a dataset/harness, benign sandbox probe, metadata signature, task pilot, tool trace or 27 authentic task receipts exist. A result that is missing remains **UNKNOWN**, not zero.

## 5. Complete model/control evidence and frozen scientific gates

- Original Windows Q4_K_M 1,319/1,319 GSM8K source-qualified: **1,224 correct (92.797574%)**, true 41/41 CUDA layers, registered --fit on -np 4 -c 16384 --jinja and independent scorer, no need for duplicate run. It is OUTSIDE the eleven frontier suite.
- Original Linux Q4 and Q8 full1319 research cohorts Q4 1,229/1,319 and Q8 1,220/1,319 are from distinct environment/serving profiles. They cannot be pooled with frozen four-slot Windows rows. Historical separate HumanEval164 model-generated research responses exist for Q4 and Q8 (151/164 and 152/164 recorded), but the independent frozen v2 generated-code execution capability and outside-host attestation approvals remain explicitly unset. Its actual original Gate2 score is UNKNOWN. A separate V38 research-only executor encountered a tool-side safety-admission block for its benign holdout; do not reroute/bypass it.
- Original REAP128-v1 paired Gate1 noninferiority lower bound **−0.03780741197234255 FAILED** versus frozen strict >−0.03. Original REAP160 candidate quality and speed remain unknown, random128 complete native same-protocol control unknown, dense original 1319 and first200 same-serving-session native speed unavailable/previous inadequate-host load failed, frozen true 1.3x speed ratio UNKNOWN.
- Gate0 (original dense comparison, EOS/trigram), Gate1 (registered paired Wald LB >−0.03), Gate2 (frozen independently witnessed original HumanEval), Q8/GGUF vs BF16 end-to-end numerical equivalence, Amendment4 optional prospective paired CI interpretation, random128 control, REAP160 fallback and deterministic selection remain UNKNOWN/BLOCKED. Observed historical data must NOT be used to choose Amendment4 interval post hoc.
- NO frozen scientific gate was changed, weakened, retconned or marked PASS from a partial score.

## 6. Actual additions, reproducibility and next executable action

New exact source-bound resource sidecar + original ProgramBench200 census, **33 adversarial tests PASS** across those two files and their unit-test companions, Ruff clean. Source guard/checkpoint integration already passed 87 scoped tests and Ruff at prior commit 8d6cf2a. Running Q4 remains untouched. To refresh real on-disk score WITHOUT new model or scorer:

    python -m chowder_batch.reap_original600_live_independent_audit_20261027 --original-run F:\reap-v2-q4-original-automationbench600-checkpointed-20261027-0953

To reverify exact original ProgramBench source (NO scoring, ONLY population):

    python -m chowder_batch.reap_original_programbench200_population_20261027 --official-checkout F:\reap-v2-programbench-official-v1_0_1-source-20261027 --private-out NEW_EXCLUSIVE_F_DRIVE_JSON

The next executable priority is finishing the **single ACTIVE source-pinned original Q4 AutomationBench600 run**, with immutable per-item WorldState receipts. After that job exits naturally and existing original-resource admission is reverified, resume the same F: checkpoint root rather than restart tasks. Do not load another Q4/Q8/dense GPU server while original Q4 remains live. Recover Q8 CPU proof only after Kaggle rate-limit relief; one targeted retrieval this round returned HTTP429, no duplicate Kaggle job submitted. An independent larger-memory runner will be required for the original Q8 HLE missing29, native dense speed and full original HumanEval sandbox.

## 6. Human-authorized explicit conservative resource admission bypass (12:09 CDT amendment)

Following the operator's direct request to bypass the previously conservative **prestart** RAM/commit/VRAM thresholds, the experimental-only checkpointed original AutomationBench600 runner adds the explicit option `--bypass-conservative-resource-admission`. This does **not** rewrite this report's preceding genuine measurements: resource measurements are observed facts, not configurable limits. It does not change the serving flags, task population, original grader or source/model hashes and does not affect the frozen four-slot Gen0 gates.

The optional admission removes precisely three *prestart* conservative rejection reasons: RAM <29.3146 GiB, commit <34.3146 GiB, GPU <12,800 MiB. Actual continued Windows process reserves are still enforced at **7 GiB physical and 8 GiB commit**; minimum available GPU is now calculated from the actual *source-bound loaded Q4* native allocator: **10,619.88 + 1,440 + 62.81 + 172.05 MiB**, plus the **260 MiB measured WDDM unaccounted reservation** (implementation minimum **12,555 MiB**). Physical model allocation cannot be bypassed by software flags. Original ports in use, an already active evaluator, wrong source/weights, disk floor and context incompatibility still refuse.

This is an explicitly distinct experimental resource admission mode with a new private sidecar receipt written on a future admitted owned native load. **It has NOT been used to start a new model**. The live read-only negative control at 12:09 measured **205 MiB free GPU** with original native PID still running, detected occupied port8892 and correctly refused a duplicate native server; no output directory was created. Thus no independent process was disturbed. This does not retroactively change the resource admission of the running task or create an official benchmark score. The executable CLI on an independently verified *inactive* owning process is:

    python -m chowder_batch.reap_q4_automationbench_full600_resumable_20261027 --out F:\reap-v2-q4-original-automationbench600-checkpointed-20261027-0953 --execute --max-new 600 --bypass-conservative-resource-admission

**Never invoke while the original Q4 model server or benchmark evaluator is live.** This option is explicitly a source-bound exploratory host capacity override, **not** a generated-code sandbox bypass, official scientific gate bypass or authorization to load a second Q8/dense model into 205 MiB available VRAM. The combined native resource-check, source-bound official scorer and Q4 checkpointed-run suite passes **71 adversarial unit tests**, Ruff clean, under the new explicit bypass source. The separate original ProgramBench200 source census passed 12 tests and reports no REAP model submissions.
