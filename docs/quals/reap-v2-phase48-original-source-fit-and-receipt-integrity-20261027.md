# Phase48 — original source-path integrity, real fit witness and private-Job receipts

**Campaign remains OPEN / Gen-0 BLOCKED; chosen:null; release_approved:false.**
Authorized worktree `C:\Users\nikma\wt\chowder-reap128v2-selection-20261025`,
branch `feature/reap128v2-selection-20261025`, OPEN DRAFT
[Chowder PR #282](https://github.com/niko4244/Chowder/pull/282).
This is a source/instrumentation qualification milestone, **not** a newly
completed native benchmark, official gate, cloud-host qualification or permission
to train/deploy Gen-1. Read with the immutable Phase34, Phase37, Phase45–47
reports and original preregistration. No frozen scorer, gate, selector, model,
quantization, native serving flags or evidence was changed.

## Independently observed source and hardware facts

* Frozen ordered original datasets: GSM8K **1319/1319** rendered SHA256
  `913ca81f724cba16a07099608047e7522b00910c1c7fe13499c501ac9e4bad89`;
  HumanEval **164/164** rendered SHA256
  `49fb688c6788b0531803dfd1cb85a69926ec5a7179e74cdfffc53ba52a92b420`.
  Original Windows `reap_eval.py` byte SHA256
  `9736714f21834a45c23599a50fa92ef3bebc6cf70967aca7e2329d26b4d73ee1`.
* Original candidate Q4_K_M `G:\reap128-v2-q4_k_m-20261025.gguf`:
  **11,563,915,264 bytes**, registered GGUF SHA256
  `2d64b786461b99a335fdbbdf7c4d3a6758b21409bce755751714ab7cd8f63bf7`,
  original sidecar SHA256
  `cf16fbf9e6926719bab873bd74b7062fa22365a280451ce3db9a9f8dc3031f1c`.
* Original candidate Q8_0 `I:\reap128-v2-q8_0-20261025.gguf`:
  **19,748,435,968 bytes**, registered GGUF SHA256
  `89dcaf33b61f59c83a31b718ba98b73c33437d0320ae0cc56737a5d7e43519c6`,
  original sidecar SHA256
  `a1cfe1464df4579ea9d5ae1320efc01c0de5dae1070235caca18c821ce67479d`.
  This phase reopened both original sidecars, validated the *unchanged*
  registered 733-tensor / 42-construction-chunk provenance and 40-layer expert
  inventory, including frozen BF16/quantizer identity, **without rehashing
  multi-GB GGUF contents or loading models**. The SHA values above are the
  registered originals, **not newly remeasured full-file digests**.
* Full stored BF16 checkpoint **26 shards / 37,981,266,120 bytes**; original
  BF16 GGUF **37,122,439,488 bytes**; Q8 and Q4 sizes above. Original dense
  Q4 **21,166,757,984 bytes**, dense Q8 **36,903,139,424 bytes**, random128
  Q8 **19,748,435,872 bytes**. Stored model **18,543,997,568 parameters**;
  roughly **3,444,503,168 active/token**, with 128 stored routed experts and
  8 active experts per routed layer/token.
* Early live shared-host witness: about **11.35 GiB genuinely free RAM**,
  about **25.12 GiB OS-reported free virtual/commit capacity**, NVIDIA
  RTX 5060 Ti **16,311 MiB physical**, about **400 MiB free**, GPU 93%;
  contemporary protected Q4 server working set **22,783 MB** and independent
  Ollama native model working set **3,078 MB**. These are point samples.
* Later authentic `evidence/reap-v2-phase48-live-dedicated-admission-refused-20261027.json`
  with Windows commit counters: **25,470,189,568 B free physical RAM**
  (23.72 GiB), **40,344,719,360 B commit headroom** (37.57 GiB),
  **16,311 MiB total / 12,825 used / 3,226 free GPU MiB**, 10% sampled
  utilization; foreign protected `llama-server`, Ollama and Podman services,
  plus port 8890 listener. Dynamic host samples are **not isolated original
  four-slot model-lifetime peak RAM/VRAM/commit, sustained throughput or
  same-session comparisons**. Those values are UNKNOWN.
* Dedicated native preflight **REFUSED** (exit 2); shortages in 48 GiB genuinely
  free physical memory, 48 GiB commit, 48 GiB CUDA GPU class/40 GiB genuinely
  free VRAM, original minimum native free VRAM, and foreign-process/port
  exclusivity. GGUF and native executable full-file SHA measurement was
  intentionally **skipped on failed admission**, not disproven.

## Actual defects fixed in source, with CPU-only adversarial tests

1. **Relocating unchanged GGUF sidecars invalidated original model identity.**
   Earlier code-only package instructions suggested `C:\reap\models`, but
   registered original Q4 and Q8 immutable construction sidecars attest their
   exact **G:/I: absolute GGUF output paths**. Copying unchanged sidecars to
   another model location fails the unmodified `reap_serve` source verifier.
   Added `reap_v2_original_artifact_portability_20261027.py` and integrated
   its exact original path, byte-size, duplicate-key-free sidecar, source-SHA
   and registered construction preflight into the frozen wave *before* costly
   model hashing or native Job creation. The wave's five-arm nonexecuted plan
   now identifies actual model paths, not hypothetical `C:\dedicated-reap`
   ones. Original G:/I: Q4/Q8 construction sidecars both passed read-only.
2. **False successful-fit runtime witness.** Layer log fragments showing
   `offloaded 41/41 layers to GPU`, context 16,384 and a CUDA KV size were
   incorrectly sufficient to mark `watched_fit` true **without** evidence of
   the actual native fit decision. A source-bound guard now requires the
   precise, original `common_fit_params: successfully fit params to free device
   memory` marker, correct registered GGUF load, 41/41, 4×4096 effective
   slots, 16,384 context, bounded measured KV and zero MoE CPU expert offload.
   The genuine original full1319 native server log was replayed; its authentic
   CUDA KV allocation **320 MiB** and `n_expert=128`, `n_expert_used=8`
   pass. Removing just the fit marker while retaining 41/41 **fails closed**;
   forged/foreign paths, CPU spill, changed context or listener also refuse.
   Historical server-log replay is a runtime-parser test, **not** new inference.
3. **Private Job receipt and termination ambiguity.** The owner now retains
   PID/create-time census evidence, verifies original startup-log source
   observations, rehashes the offline source receipt after termination, joins
   actual owner-start and terminal receipt identities, checks that original
   Job-member PID/birth identities are no longer alive, and refuses clean
   runtime exit if the actual owner-start bytes changed. Closing the private
   Job is attempted despite individual handle-close failures; a close error
   is recorded as a breach, cannot claim confirmed Job closure and cannot
   yield a clean exit. Adversarial synthetic tests cover source/start receipt
   tamper and Job close faults with no subprocess or GPU launched. The
   stand-alone `assess_bound_owner` validates exact root/native/evaluator
   command-graph data, but a **fresh independent complete live Job/argv/parent
   graph has NOT yet been wired and qualified in the production watcher**.
   Thus full end-to-end original owner boundary qualification remains open.
4. **Generated HumanEval grade injection/trace-source gaps.** Existing
   `build_selector_rows(cases, grade)` can accept a caller-forged memory
   grade with a copied external witness SHA; the historical scorer also
   checks per-item trace strings more weakly than real outside-host trace
   bytes. New `reap_v2_humaneval_certificate_verifier_20261027.py` reopens
   the actual immutable 164 canonical tasks, original Q8/dense captures,
   registered model/serving source, external signed witnesses and *every*
   per-case independent process/cgroup trace file; recomputes original scores
   and source-paired Gate2 diagnostic without trusting a caller-provided
   grade. The independent approval SHA maps remain intentionally empty.
   **This verifier is not yet the exclusive ingestion implementation for
   the existing historical helper or frozen gate bridge.** No Q8/dense
   officially approved independent full164 sandbox witnesses exist, and no
   generated candidate code was executed in this phase; Gate2 UNKNOWN.

Source-controlled five-arm plan:
`evidence/reap-v2-phase48-frozen-fivearm-source-plan-20261027.json`.
Original models: dense Q4 `G:\llm-cold\gguf\dense-q4_k_m.gguf`,
REAP Q4 `G:\reap128-v2-q4_k_m-20261025.gguf`,
dense Q8 `G:\llm-cold\gguf\dense-q8_0.gguf`,
REAP Q8 `I:\reap128-v2-q8_0-20261025.gguf`,
random128 Q8 `G:\llm-cold\gguf\random128-q8_0.gguf`.
All five **PREPARED_NOT_EXECUTED_REQUIRES_FRESH_ADMISSION**; plan mode did
not load a single weight tensor.

## Established real research is preserved, never pooled across sessions

| Original evidence | Authenticated separate research observation | Frozen inference |
|---|---|---|
| Windows original REAP Q4 GSM8K | **1,224/1,319 (92.797574%)**, 1,281 native item terminations, 459,264 generated tokens; original 4-slot native 41/41 CUDA layers, 0 CPU experts | Not a Q8-vs-dense Gate1 comparison |
| Original REAP Q4 first200 | **189/200**, mean **40.770297720291175 predicted tok/s** in original four-slot serving; distinct historical Q4 187/200 not combined | Original same-host dense Q4 first200 missing; speed ratio UNKNOWN |
| Original Q4 vs Q8 HumanEval | Q4 **151/164**, Q8 **152/164**; both correct 147, Q8-only 5, Q4-only 4, neither 8 | Q8-versus-original-dense 164 external-execution Gate2 missing |
| Linux REAP Q4/Q8/random128 GSM8K | Q4 **1229/1319**, Q8 **1220/1319**, random128 Q8 **964/1319** | Distinct Linux sessions/platform, not frozen Windows paired control |
| BF16 isolated layer CPU/CUDA parity | **127,806/163,840 bit-exact**, maximum absolute difference **0.052734375** | Does not establish whole-model 40-layer CUDA parity/D0 |
| Historical REAP128-v1 | BF16 D0 PASS, Gate0 PASS, **Gate1 FAIL**, one-sided lower bound **-0.03780741197234255** against strict **>-0.03** | Failure retained, not upgraded by v2 research |
| Original REAP160 | Separate historical **1244/1319** GSM8K research | New full frozen REAP160 fit, parity and speed UNKNOWN |
| AutomationBench600 | guard.v3 genuine **87/600** prefix, 3 strict pass; guard.v1 separate **43/600**, 0 strict | Neither prefix is complete official original suite |
| Separate original Q4 LiveBench subsets | **139/200** across four distinct 50-item subsets | Not official aggregate or one of the frozen eleven |

## Frozen registered qualification ledger

| Requirement / original decision criterion | Actual status |
|---|---|
| REAP128_v2 BF16 original whole-model D0, numerical EOS/trigram behavior | **UNKNOWN** |
| REAP128_v2 40-layer original full CPU/CUDA numerical parity | **UNKNOWN** |
| REAP128_v2 Gate0 quality/safety | **UNKNOWN** |
| REAP128_v2 Gate1 Q8 versus dense Q8 original GSM8K full1319; one-sided sample-SD Wald lower bound > **-0.03** | **UNKNOWN** |
| REAP128_v2 Gate2 Q8 versus dense Q8 original sandboxed HumanEval full164; lower bound > **-0.10** | **UNKNOWN** |
| Frozen random128 Q8 control against original candidate/dense, matched source/host | **UNKNOWN** |
| Candidate Q4 versus original dense Q4 original first200 native ≥ **1.3×** throughput | **UNKNOWN** |
| REAP160 full qualified source/fit/gates/speed | **UNKNOWN** |
| Amendment4 nonzero-margin **-0.02**, one-sided McNemar-compatible paired 95% noninferiority CI vs original §4 **Z=1.645/sample-SD Wald** | **BLOCKED** pending prospective independent, non-data-selected method decision |
| Deterministic original selector / deployment / Gen-1 | **BLOCKED; chosen:null; release_approved:false** |

No rule may treat zero unscored rows as wrong, join two separate native
serving sessions as one, replace a missing dense/control row with candidate
research, or promote constructor/source-path proof into scientific PASS.

## Original eleven registered frontier suites

| Registered suite | Authentic observed progress | Official full score |
|---|---|---|
| 1. HLE text-only (128) | Original Q4 128 answers, heuristic 3/128; Q8 99 answers, heuristic 0/99, remaining 29 and official judging missing | **UNKNOWN** |
| 2. TerminalBench4.0 (66) | Original Q4 one agent three tool actions, context failure; no complete verifier | **UNKNOWN** |
| 3. DeepSWE1.1 (113) | No original complete officially graded cohort | **UNKNOWN** |
| 4. ProgramBench1.0.1 (200) | Official source and 248,089 real nonignored tests verified; original candidate solutions not fully run | **UNKNOWN** |
| 5. NL2Repo-Bench (104) | No complete original agent/repository grading | **UNKNOWN** |
| 6. AutomationBench1.0.6 (600) | Source-bound partial original 87/600 guard.v3 and separate 43/600 guard.v1; no pooling | **UNKNOWN** |
| 7. Agents' Last Exam1.1 (122) | No complete original agent grading | **UNKNOWN** |
| 8. ApexBench1.1 | Official v1.1 complete source/protocol not independently verified | **UNKNOWN** |
| 9. Chartography/tools (100 images) | Original checkpoint lacks native image input | **NOT NATIVE-ELIGIBLE** |
| 10. BabyVision/tools (388 images) | Original checkpoint lacks native image input | **NOT NATIVE-ELIGIBLE** |
| 11. ZeroBench-main/tools (100 images, pass@5) | Original checkpoint lacks native image input; no unapproved OCR/VLM substitute | **NOT NATIVE-ELIGIBLE** |

## Reproducible next experiment on genuinely dedicated admitted resources

1. Provision a **separately authorized Windows host** with ≥48 GiB *actually
   unused physical RAM*, ≥48 GiB real Windows commit headroom, a ≥48 GiB
   CUDA-class GPU with ≥40 GiB genuinely free VRAM, ≥8 GiB free disk,
   no foreign conflicting services and an unused private port. These are
   conservative *provisioning*, not gate PASS. Original four-slot minimum
   free start thresholds (~29.315 GiB physical, ~34.315 GiB commit, 12,288 MiB
   CUDA VRAM) remain separate original floors.
2. Restore identical original GGUF bytes, sidecars and exact **G:/I:/F:**
   drive layout. Do not edit signed sidecars, frozen model SHA metadata,
   model source/weights, or original evaluator/selector. Verify full GGUF
   and native executable SHAs *after physical admission*, canonical HF cache,
   exact source-only portable package/CPython ABI and documented dependency
   versions. Native `F:\llama.cpp\bin\llama-server.exe` must be original
   qualified binary; always use **`--fit on -np 4 -c 16384 --jinja`**.
3. Noninvasively repeat admission before each new *private Job* process.
   Use fresh exclusive output+receipt paths, original `frozen_wave`
   `--execute --arm ... --authorize-owned-run --receipt-dir` interface;
   never adopt existing rows, stop unrelated servers, or launch on this
   currently shared RTX 5060 Ti. Require verified native fit, 41/41, zero
   CPU expert spill, 4×4096 slots, stable listener/Job ownership, real
   process-tree extinction, original full source/manifest hashes and
   original per-row termination/source receipts.
4. Complete full1319 original candidate Q8, original dense Q8 and random
   control; original same-host dense Q4 first200 to match existing Q4
   research only if its serving contract is scientifically compatible
   (otherwise redo both fresh, with independent original sessions). For
   HumanEval full164, first acquire independent **outside-host sandbox,
   reviewer and per-case process/cgroup/trace** approvals before executing
   any generated program; compare original Q8 versus dense Q8 only.
   Run REAP160 and admissible frontier suites via their authentic registered
   harnesses. Prospective independent statistical-method adjudication
   must precede any Amendment4 test.
5. Recompute the original unmodified frozen gate engine/selector from
   independently approved source-pinned complete evidence; no alternate
   pass rule, regression stand-in or synthetic approval. Until then
   **`{"status":"BLOCKED","chosen":null,"release_approved":false}`**.

Private owned-process close/exit and source-audit unit tests do not establish
full production Win32 child graph attestation, BF16 parity, real native model
fit or completed benchmark outcomes. No prior immutable package was overwritten.
