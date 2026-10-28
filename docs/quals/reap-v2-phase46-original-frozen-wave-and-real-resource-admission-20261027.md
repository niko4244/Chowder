# REAP128_v2 Phase46 — integrated original frozen wave, native admission and deterministic Gen-0

**As-of:** 2026-10-27 CDT. **Branch:** `feature/reap128v2-selection-20261025`;
draft PR [niko4244/Chowder#282](https://github.com/niko4244/Chowder/pull/282).
This is a source-backed **engineering and bounded native observation** update;
the broader frozen scientific qualification remains OPEN. No newly provisioned
dedicated runner, GPU inference session, generated-code sandbox, deployment,
weights mutation, upstream merge or Gen-1 training is claimed.

## 1. Implemented, testable source-controlled acceptance and execution interfaces

| New additive module | Genuine function and fail-closed boundary |
|---|---|
| `chowder_batch/reap_v2_frozen_wave_20261027.py` | Deterministic exact original Q4 dense first200 / Q4 candidate first200 / dense Q8 full1319 / candidate Q8 full1319 / random128 Q8 full1319 **command graph**. Uses the unchanged `reap_serve.py` and `reap_eval.py` only for GSM8K. Plan mode reads existence/size, not big weights or model results. Hard-blocks invented model identities, old output sessions, native HumanEval `python -I`, frozen flag drift, auto-launch without independent native owner admission. Outputs are instructions, **not completed tasks**. |
| `chowder_batch/reap_v2_dedicated_runner_admission_20261027.py` | Real Windows `GetPerformanceInfo` physical free/commit charge and headroom, native NVIDIA GPU counters with WDDM reserved-memory separation, full protected-process/listener census, disk floors; once eligible, SHA256/stat of exact original model and native server, then a **second** live resource/process measurement. Supports Q4, Q8, dense Q4, dense Q8, random128 Q8 and BF16 GGUF. Refuses if any native floor fails. `READY_FOR_CAPACITY_PROBE_ONLY` **never** authorizes a run or Gen-0 approval. |
| `chowder_batch/reap_v2_live_resource_witness_20261027.py` | Bounded read-only repeated physical RAM/Windows commit/GPU and named process working sets; fsynced exclusive snapshot. Separates host-wide sampling from isolated model lifetime peak, KV or sustained fit. |
| `chowder_batch/reap_v2_frozen_gate_evidence_bridge_20261027.py` | Accepts only independently source-owned SHA-pinned original model/scorer/dataset/serving reviews, all canonical Q8/dense GSM8K1319 and HumanEval164, random1319, matching Q4 first200, exact original serving geometry, same-hardware speed, no session pooling, independent **both-arm** generated-code sandbox receipts. Delegates accepted evidence to existing `reap_select.evaluate` without changing original prereg thresholds or frozen fallback order. Reviewer SHA/security approval constants intentionally **unset** until authentic approvals exist; current output BLOCKED and every unproved gate UNKNOWN. |
| `chowder_batch/reap_v2_authentic_amendment4_replay_20261027.py` | Independently replays the existing immutable **original historical v1** frozen selector snapshot (SHA256 `7043980b1a5551131c348d720704382eeee42b63001afcccd9f031b7528bb42e`). Confirms historical Gate1 `FAIL`, lower bound `−0.03780741197234255`, strict threshold `−0.03`, and unchanged order requiring **both** REAP128_v2 and REAP160 complete gates before selection. Does **not** reconstruct missing new raw model rows. |

New tests exercise source SHA/flags, canonical populations and model identities,
incompatible or resumed sessions, false receipts, unknown required arms, refusal
of partial scores, WDDM VRAM reservation, actual Windows telemetry and both
historical selector and prospectively unapproved paired-CI boundaries.

## 2. Exact physical host observations and native refusal

**Bounded three-sample original Windows witness:**
`evidence/reap-v2-live-native-resource-bounded-20261027.json`.

| Value | Genuine sampled measurement | Claim scope |
|---|---:|---|
| Minimum physical RAM available | **10,727,972,864 bytes (~9.99 GiB)** | Host-wide min across 3 snapshots |
| Minimum Windows commit headroom | **25,682,505,728 bytes (~23.92 GiB)** | Host-wide min across 3 snapshots |
| Maximum NVIDIA GPU physical memory used | **12,325 MiB** | Host-wide observed maximum, not isolated model peak |
| Minimum native GPU free | **3,726 MiB** | Host-wide observed minimum, not unused class VRAM guarantee |
| Maximum sampled native llama-server working set | **20,600,455,168 bytes (~19.19 GiB)** | Maximum among all observed native llama-server processes; not lifetime or isolated Q4 peak |
| WDDM reserved/unavailable VRAM | **260 MiB** in a separately observed NVIDIA sample | `total − used − free`; **never** added to usable VRAM |

Owned, active **Q4 LiveBench** evaluator PID40080 and original Q4 llama-server
PID55032 on loopback8890 were independently inspected; the server had
`--fit on -np 1 -c 16384 --jinja --reasoning off` and is **not** a frozen
four-slot qualification. Ollama native model server and unrelated Freebuff,
Podman/Hermes services were also active and untouched. These PIDs are an
observed snapshot, **not** durable process ownership identities for a future
launch. No services were stopped, restarted or reconfigured, no GPU model
was loaded, and Freebuff was never proxy-routed.

**Genuine original Q4 dedicated admission:**
`evidence/reap-v2-native-dedicated-admission-refused-20261027.json`.
Native result **REFUSED**: active servers/listeners, free physical RAM and
Windows commit below dedicated 48GiB targets, GPU below both the 48GiB-class
target and native 12,288MiB free allocation floor, and non-idle GPU. Reported
unverified model/server SHA is a consequence of deliberately **not hashing
or starting** huge files after physical preflight refusal; it is **not a
claim of SHA mismatch or corrupt model bytes**. Historical direct native
four-slot Q4 admission had ≥29.315GiB free RAM, ≥34.315GiB free commit,
≥12,288MiB free VRAM, ≥8GiB disk; these were not relaxed. Dedicated targets
are stricter *engineering provisioning*, **not newly frozen scientific cutoffs**.

## 3. Original model identity, math performance and genuine research

| Original REAP128_v2 fact | Source-backed value or limitation |
|---|---|
| Stored params | **18,543,997,568** |
| Approx active compute params/token | **3,444,503,168** |
| Expert routing | 128 stored experts, 8 active MoE experts/token |
| BF16 pruned original shard data | **37,981,266,120 bytes** in 26 shards |
| BF16 intermediate GGUF | **37,122,439,488 bytes**; no full candidate chained logit gate pass |
| Q8_0 original GGUF | **19,748,435,968 bytes**, SHA256 `89dcaf33b61f59c83a31b718ba98b73c33437d0320ae0cc56737a5d7e43519c6` |
| Q4_K_M original GGUF | **11,563,915,264 bytes**, SHA256 `2d64b786461b99a335fdbbdf7c4d3a6758b21409bce755751714ab7cd8f63bf7` |
| Q4 native placement | Real original 41/41 decoder layers CUDA; **zero** CPU expert offload in verified Windows 4-slot GSM8K research |
| Q8/BF16/dense all-GPU native placement | **UNKNOWN / insufficient physical 16GiB GPU** for whole GGUF; no fabricated placement |
| Windows Q4 original GSM8K | **1224/1319 (92.797574%)**, full original candidate source-bound |
| Windows Q4 original first200 | **189/200 correct**, mean **40.770297720291175 native predicted tokens/s**; source-bound four-slot candidate but same-host original dense first200 missing |
| Original Q4 HumanEval164 | **151/164 (92.07317%)** research, 162 safely executed plus 2 original unterminated nonpasses |
| Original Q8 HumanEval164 | **152/164 (92.68293%)** research, 162 safely executed plus 2 original unterminated nonpasses |
| Original Q8-vs-Q4 HumanEval research | Same-item 147 both correct, 5 Q8 only, 4 Q4 only, 8 neither; separate model sessions, **NOT Q8-vs-dense Gate2** |
| Separate original Linux GSM8K research | Q4 **1229/1319**, Q8 **1220/1319**, random128 **964/1319**; no cross-host pooling into Windows frozen gate |
| BF16 40 isolated layer CPU/CUDA study | **127806/163840** bit-exact scalars; max abs deviation **0.052734375**; NOT chained logits, 704-row full-model CUDA parity or quantized end-to-end numerical pass |
| AutomationBench v1.0.6 original Q4 | Guard.v3 **87/600**, **3 strict**; independent historical guard.v1 **43/600**, **0 strict**. Different source policy and serving sessions; **not pooled** or complete 600 |
| Model-specific isolated lifetime peak physical RAM/VRAM and sustained four-slot Q8/dense tokens/s | **UNKNOWN**; a host sampled max is not a model allocation peak, and one-slot experimental speed cannot be substituted |

Earlier full original v1 historical calibration D0 and Gate0 passed, **but
its strict Gate1 paired lower bound −0.03780741197234255 failed**. Original
REAP160 historical GSM8K research 1244/1319 in a different protocol exists;
same-platform REAP160 real native speed and original matched qualified
quality remain UNKNOWN.

## 4. All originally registered eleven frontier suites

| Registered original suite | Actual source-bound work | Final official full score |
|---|---|---|
| 1. HLE no tools, original 128 text items | Q4 128 responses with nonofficial normalized heuristic 3/128; Q8 99, missing29 after resource refusal; no official judge | UNKNOWN |
| 2. Terminal-Bench4.0, 66 tasks | Original Q4 3 native agent/tool steps, context overflow before Harbor scoring | UNKNOWN |
| 3. DeepSWE1.1, 113 tasks | No eligible complete model-generated/original verifier receipts | UNKNOWN |
| 4. ProgramBench v1.0.1, 200 tasks | Exact official source and 248,089 genuine nonignored behavioral tests authenticated; zero original model programs graded | UNKNOWN |
| 5. NL2Repo-Bench, 104 tasks | Original OpenHands/repo-test scoring absent | UNKNOWN |
| 6. AutomationBench1.0.6, 600 tasks | v3 87/600 strict3 and unrelated v1 43/600 strict0 are partial experimental profiles | UNKNOWN |
| 7. Agents' Last Exam1.1, 122 tasks | Original app/tool worlds and official results absent | UNKNOWN |
| 8. ApexBench1.1 | Exact preregistered source/version cannot be authenticated | UNKNOWN |
| 9. Chartography with tools, 100 visual tasks | Original text-only model lacks native image decoder | NATIVE INELIGIBLE |
| 10. BabyVision with tools, 388 visual tasks | No native image-token input; unrelated VLM/OCR forbidden substitute | NATIVE INELIGIBLE |
| 11. ZeroBench-main with tools, 100 visual tasks pass@5 | Native image input unsupported, no borrowed VLM judge | NATIVE INELIGIBLE |

Original GSM8K and HumanEval are **separate frozen quality gate cohorts**,
not an extra 2 completed official frontier suites. Live Q4 LiveBench
instruction-following 139/200 locally strict-judged partial subset is
similarly outside the original eleven, outside frozen serving geometry,
and not a complete official LiveBench result.

## 5. Deterministic frozen selector — genuine historical result

Source-pinned historical replay:
`evidence/reap-v2-historical-amendment4-authentic-replay-20261027.json`.
The actual original `reap_select.select` was re-executed with the immutable
historical v1 evidence from original SHA `7043980b...7528bb42e`:

| Scientific frozen condition | Latest admissible state | Requirement still missing |
|---|---|---|
| Original source, selection construction, model hashes | Original provenance preserved | New full native gates require fresh same-platform receipts |
| Original v1 D0 and Gate0 | PASS historically | Not transferable to v2 |
| Historical REAP128-v1 Gate1 > −0.03 | **FAIL**: LB −0.03780741197234255 | Forces both v2 and REAP160 evaluation |
| v2 BF16 704×40 original D0 PPL<20 | UNKNOWN frozen current complete chained/calibration-sealed qualification | New exact 704-row full-model source/observer/admission |
| v2 Gate0 EOS and distinct trigram vs dense | UNKNOWN | Paired Q8/dense original full1319 native serving |
| v2 Gate1 GSM8K lower bound > −0.03 | UNKNOWN | Original dense_v2 vs Q8 v2 source-bound full1319 same-platform |
| v2 Gate2 HumanEval164 lower bound > −0.10 | UNKNOWN | Original generated responses from frozen Q8/dense, independent isolated executor and 164 execution attestations |
| Random control paired bound > 0 | UNKNOWN frozen | Original native Q8 vs random128 full1319 |
| Candidate/dense first200 ≥1.3× and zero Q4 expert offload | UNKNOWN ratio | Fresh original dense Q4 + candidate Q4, paired canonical 200 IDs, single frozen host/profile, distinct complete native ownership/session receipts |
| REAP160 original fallback quality and speed | UNKNOWN | Full source-matched native original gate panel and measured speed/fit |
| Optional Amendment4 nonzero-margin one-sided 95% paired McNemar CI > −0.02 | **BLOCKED prospective methodology conflict** | Independently specify exact paired risk-difference CI construction/inversion and boundary behavior **before** observing new optional cohort; original §4 sample-SD Wald is NOT an interchangeable McNemar CI |
| Eleven original frontier results | INCOMPLETE | Exact original tools/harness/official judges and admissible scores |

**Important Amendment4 distinction:** Original historical v1 Gate1 FAIL
already triggers the mandatory full v2+REAP160 fallback. Optional
v2-over-passing-v1 CI method conflict does not authorize outcome-conditioned
statistical selection; the defined method must be settled prospectively.
No threshold, comparison arm, input prompt, scorer, quantizer or frozen
`--fit on -np 4 -c 16384 --jinja` flag was changed.

```json
{"status":"BLOCKED","chosen":null,"release_approved":false}
```

## 6. Reproducible *future* dedicated Windows runner procedure

**No dedicated runner has been provisioned or tested this turn.** The frozen
quality wave is deployable *only after* source identities and fresh, physical
admission succeed on an explicitly authorized, isolated Windows host.

1. Provision at least **48GiB genuinely free physical RAM and Windows commit**
   (prefer ≥64GiB free), **≥48GiB physical CUDA VRAM with ≥40GiB genuinely
   free** on a suitable card, dedicated model storage (five frozen GGUFs total
   roughly **109.1GB**, plus BF16 intermediate ~37.1GB, with ≥8GiB untouched
   disk reserve), and the exact pinned Windows llama-server binary. Prove
   GPU readiness with actual WDDM reserved VRAM accounting and no competing
   GPU resident server; nominal hardware specifications do not prove admission.
2. Copy the **original immutable bytes** of the mapped model arms to the
   dedicated host, then independently verify each registered SHA and size:
   `dense_v2_q4` = dense Q4 source `2c6d946...`; `reap128_v2_q4` = Q4
   `2d64b786...`; `dense_v2` = original dense Q8 `2ff57a74...`;
   `reap128_v2` = Q8 `89dcaf33...`; `random128_v2` = random Q8 `601871e...`.
   Do not alter weights, tokenizer, prompt lists or quantization.
3. On the **dedicated host**, create a new output parent and an independently
   monitored native capacity window. Run the following **read-only** commands
   before *each* model arm (use a new unique admission JSON filename each time):

```powershell
python -m chowder_batch.reap_v2_frozen_wave_20261027 `
  --models-dir C:\reap\models --out C:\reap\eval-wave-1 `
  --server C:\llama.cpp\bin\llama-server.exe --port 8790

python -m chowder_batch.reap_v2_dedicated_runner_admission_20261027 `
  --arm dense_q4 --model C:\reap\models\dense_v2_q4.gguf `
  --server C:\llama.cpp\bin\llama-server.exe --port 8790 `
  --out C:\reap\receipts\dense-q4-admission-1.json
```

The admission output must have no reasons and status
`READY_FOR_CAPACITY_PROBE_ONLY`, with every original source hash verified.
**This status does not authorize automatic unattended launch.** Before a
new native run, independently qualify the owned process-tree supervisor,
bounded RAM/commit/KV floors, restart/teardown and GPU exclusivity, then
maintain monitoring during the actual full run. The wave's `--execute` mode
deliberately refuses until that owner-bound launch capability is integrated.
Once qualified, the EXISTING original source-locked runner supports this
explicit single-arm sequence (fresh experiment output, without mixed resumes):

```powershell
# Frozen dense-Q4 first200 (fresh, owned server/session)
python -m chowder_batch.reap_serve `
  --gguf C:\reap\models\dense_v2_q4.gguf --arm dense_v2_q4 `
  --port 8790 --out C:\reap\eval-wave-1 `
  --server C:\llama.cpp\bin\llama-server.exe -- `
  --port 8790 --arm dense_v2_q4 --out C:\reap\eval-wave-1 `
  --suites gsm8k --parallel 4 --limit 200

# Frozen candidate-Q4 first200, after first server exits and its receipts verify
python -m chowder_batch.reap_serve `
  --gguf C:\reap\models\reap128_v2_q4.gguf --arm reap128_v2_q4 `
  --port 8790 --out C:\reap\eval-wave-1 `
  --server C:\llama.cpp\bin\llama-server.exe -- `
  --port 8790 --arm reap128_v2_q4 --out C:\reap\eval-wave-1 `
  --suites gsm8k --parallel 4 --limit 200
```

Likewise run original `dense_v2`, `reap128_v2`, `random128_v2` arms
**sequentially**, same host with original `--suites gsm8k --parallel 4
--limit 0` for all 1,319. Each Q4 first200 arm is a **distinct model and
serving session**, not one model process shared between incompatible
weights; comparison requires identical original source/flags/physical
hardware, full per-arm sealed receipts and no session pooling. Build an
approved separate **capture-only** original HumanEval native pipeline with
exact prompts/decoding and outside-host sandboxed program execution before
calling any HumanEval correctness official; existing `reap_eval._humaneval_pass`
uses local `python -I` and is **not** an approved generated-code sandbox.
Do not use it for frozen Gate2.

Finally, have an independent reviewer source-pin the original cohort files,
Windows serving owners, session/physical-hardware and execution witness SHA,
and call the new `reap_v2_frozen_gate_evidence_bridge_20261027.py` API;
it delegates to unchanged `reap_select.evaluate` and refuses if prerequisites
are absent. Resolve Amendment4 optional paired method **prospectively** if
it becomes material. Only then can the original deterministic fallback order
be decided. No self-reported `PASS`, fixture, Linux score, research-only
HumanEval pairing or 200-row candidate-alone speed is acceptable.

### Explicit operational and science blockers

Current workstation lacks physical RAM/commit/VRAM and has protected Q4
LiveBench/Ollama processes. Dedicated GPU hardware, owner-bound run admission,
chained BF16 logits/D0, frozen Q8+dense, original HumanEval safe executable
scoring, random/REAP160 and remaining feasible official agent frontiers are
unexecuted or incomplete. No existing expensive completed evaluation needs
to be rerun just to refresh evidence. The PR stays **OPEN/DRAFT**, and
Gen-0 stays **BLOCKED/chosen:null/release_approved:false**.
