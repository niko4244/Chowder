# REAP128_v2 Phase41 — native original-model benchmark execution

Source-bound freeze: 2026-10-27 13:38:35 CDT. The qualification objective remains
OPEN. No benchmark partial prefix is a completed official suite score; no
research-only execution counts as frozen Windows Gen-0 Gate2.

## Actual new REAP128_v2 Q4 and Q8 HumanEval164 model-generated code

Original source-bound Q4 and Q8 candidate answers from independent prior model
inference sessions were executed with the native Python3.13 original HumanEval
test code inside disposable Podman research containers. V38 loader SHA256:
3319565d5d5a6aa210151a32bd367a136a08cfb530402c51f311143de2f08e66.
Outside-VM host independent witness SHA256:
c4e5fd10ae77338337357ee250d8f5e19d925f74cf520c42b2e25b94cc519425.
Original immutable Q4 GGUF SHA:
2d64b786461b99a335fdbbdf7c4d3a6758b21409bce755751714ab7cd8f63bf7.
Original immutable Q8 GGUF SHA:
89dcaf33b61f59c83a31b718ba98b73c33437d0320ae0cc56737a5d7e43519c6.

Each genuine completed original generated program has its own fsynced scored
source-bound receipt, externally witnessed hash, memfd 4-bit seals, independent
cgroup, process, namespace, network/mount and cleanup checks. The native
adversarial test replaced mutable candidate.py after the outside-host witness,
but executed the originally sealed immutable Python source successfully, not
the replacement; no mutable source pathname is executed. Independent
research-only admission was source-SHA pinned, without changing frozen V2
security approval constants.

Two initial Q4/Q8 processes, executing concurrently on a limited native Podman
VM (4 virtual CPUs, 4GiB RAM, no swap), genuinely scored their original
prefixes, then both correctly aborted on teardown/postcheck timeouts.
No interrupted case was assigned an incorrect score. The only confirmed
owned EXITED capsule was inspected and removed by exact image+name+owner
and no unrelated container or protected service was touched. Serialized
single-owner guarded resume verifies the ORIGINAL score journal before each
new source-bound case, appends and fsyncs only fully externally witnessed
results, and retains partial complete-prefix meaning on failure.

| Original arm | Real independently verified prefix | Correct in prefix | Unknown remaining |
| --- | ---: | ---: | ---: |
| Q4_K_M original 164 HumanEval candidates | 34 / 164 | 34 / 34 | 130 |
| Q8_0 original 164 HumanEval candidates | 3 / 164 | 3 / 3 | 161 |

The 34-case Q4 verified receipt snapshot lives at:
F:/reap-v2-human164-independent-resume-20261027-1339

Original scored journal snapshot SHA256:
a4da95d51e9b31ad541311e5dc7ce1acba643062dfd98137de4d25b03417b0c8.
Independent audit JSON SHA256:
4ae57f62160499d5e8cc137278c4172037be9c3106ab13eb3370cf65f6c0421f.

The 3-case Q8 independent snapshot lives at:
F:/reap-v2-human164-v38-independent-worker13-q8-audit-20261027-1342
Q8 original journal SHA256:
804e7646a4b70adbd16e593f59315f9db083298d1470696da6882061b655ab56.

The source-bound original research auditor is
chowder_batch/reap_v38_independent_receipt_audit_worker13_20261027.py;
the Windows cross-process exclusive serialized resume is
chowder_batch/reap_humaneval_v38_guarded_resume_20261027.py.
Both are additive, not replacements for the frozen original v2 Gate2 or
official benchmark graders. Previously recorded 151/164 Q4 and 152/164 Q8
were separate Linux research references and cannot be passed off as new scores.

## Original AutomationBench v1.0.6, genuine score and new guard policy

The ORIGINAL guard.v1 source-owner SHA256
849309859a5c1ef8736ee67668b33d8fbbf1dc3334b246451af4c4e5d46f220a
was independently rescored at 43 / 600 authentic original tasks,
0 / 43 strict original WorldState passes, 35 / 240 positive assertions,
223 / 223 exclusion requirements, 1314 original model calls, 1751 original
tool calls and 22,069,396 original input tokens.
Its exact source is original AutomationBench v1.0.6 commit
4a8e1061254004d9dac807054eed33fad7d1ff14. The original prompt SHA is
1c8feeec1f680b05b392a5d00fc634edb143c3e7bdd152c476af50cebcec57b5.
The source-locked audit replays historical original guard.v1 policy identity
for read-only scoring only and binds the original scoring Python/Git to commit
ea453b2, rather than mislabeling v1 results as guard.v2.

Immutable exclusive F: audit:
F:/reap-v2-original600-guardv1-historical-audit-20261027-1339.json
SHA256:
41c79dfabe975dfcd9394e5c1ad92bf0d81add09a78c408b21dc6acd4b57fabf.

Concurrent contributor original guard.v2 experimental full600 has separate
source owner SHA
332f51a2a4cc5cab8ab2abf3e88ac4a3634b8868be93de4a793cf5969c4df73c.
At 13:36:57 CDT its original index0 was running and 0 scored tasks had
finished. This is UNKNOWN score, not a 0% official outcome. Do not pool
guard.v1 and guard.v2 or their native sessions. Both research runs were one-slot
73,728-context experimental servers. Original frozen serving for scientific
comparison remains --fit on -np 4 -c 16384 --jinja.

## Eleven original registered frontier suites

| # | Benchmark and official original identity | Actual verified position |
| --- | --- | --- |
| 1 | HLE, text only, no tools | Q4 128 source answers only, heuristic 3/128 not official; Q8 first99 source answers, missing29 cgroup-aborted; official UNKNOWN |
| 2 | Terminal-Bench 4.0.0 | Q4 original interleaved-vigenere agent ran three native action rounds before 8192-context overflow; verifier not reached; original score UNKNOWN |
| 3 | DeepSWE1.1 | No original full113 grading; UNKNOWN |
| 4 | ProgramBench v1.0.1 | Original200 tasks and248089 behavioral tests source authenticated, no genuine REAP-generated program scored, UNKNOWN |
| 5 | NL2Repo-Bench | Original104 unscored; UNKNOWN |
| 6 | AutomationBench v1.0.6 | Q4 authentic partial43/600 guard.v1 plus separate live guard.v2, neither full600 nor frozen gate |
| 7 | Agents' Last Exam v1.1 | Original122 unscored; UNKNOWN |
| 8 | ApexBench v1.1 | Exact public registered original v1.1 source not independently confirmed; UNKNOWN |
| 9 | Chartography with tools | Native image input unsupported by text model, not adapted |
| 10 | BabyVision with tools | Native image input unsupported by text model, not adapted |
| 11 | ZeroBench-main with tools | Native image input unsupported by text model, not adapted |

An unavailable complete score is UNKNOWN, not failure and not a 0/100
score. Do not use another VLM, image OCR/caption pipeline, benchmark proxy
or benchmark fixture as an official replacement.

## Original architecture and numerical/resource controls

Original BF16 source: 18,543,997,568 stored parameters,
approximately 3,444,503,168 active parameters per token, 26 original
pruned BF16 safetensors shards totaling 37,981,266,120 bytes. Q8 GGUF
19,748,435,968 bytes; original Q4 GGUF 11,563,915,264 bytes. The
authentic all40 independent single-layer CPU/CUDA numerical study compared
163840 values (78.0066% bit-exact, max difference 0.052734375);
this is NOT full chained BF16 logits or Gate0 numerical acceptance.

Original Windows frozen-flag Q4 GSM8K1319 is genuine completed
1224 / 1319 (92.797574%). A separate Linux original research Q4 1229/1319,
Q8 1220/1319 cannot be pooled for Windows frozen Gate1.
Original Q4 research first200 mean throughput approximately 40.7703 tok/s;
registered same-session dense first200 paired speed and valid ≥1.3x ratio
remain UNKNOWN.

The Q4 experimental native allocator occupied CUDA tensors10619.88 MiB,
KV 1440MiB, recurrent62.81MiB, CUDA compute172.05MiB, with no CPU expert
offload. Windows sampled max working-set24.16GiB/private22.96GiB,
host min RAM available12.5GiB, free commit31.25GiB, min free VRAM234MiB.
An explicit user-authorized conservative startup threshold override does
not bypass physical CUDA allocator fit (at least12555MiB free for a second
fresh 72K Q4 start), owner/port exclusivity, OS reserve or source integrity.
Do not load Q8/BF16/dense on the occupied 16GiB GPU.

## Frozen Gate0/1/2 and Amendment4

| Original preregistered scientific rule | Status |
| --- | --- |
| Original deterministic source/selection identities | PASS identity/provenance only |
| D0 BF16 40-layer full forward, PPL strictly below20 | UNKNOWN |
| Gate0 EOS and distinct-trigram dense-relative | UNKNOWN |
| Gate1 original Q8 paired GSM8K1319 sample-Wald LB > -0.03 | UNKNOWN v2; v1 historical FAIL at -0.03780741197234255 |
| Gate2 original Q8 generated HumanEval164 vs dense LB > -0.10 with frozen v2 security | UNKNOWN |
| Original random128 negative control paired LB >0 | UNKNOWN |
| Frozen original Q4 first200 candidate/dense mean speed ≥1.3x | UNKNOWN |
| Candidate Q4 41 CUDA layers and zero CPU experts | PASS original research serving only |
| REAP160 fallback original quality/offload/speed | UNKNOWN |
| Amendment4 optional v2-v1 nonzero-margin paired McNemar LB > -0.02 | BLOCKED: CI method mathematically underspecified |
| All11 authentic frontier suites | INCOMPLETE |

Amendment4 optional replacement cannot cherry-pick the original §4
sample-SD Wald formula or a favorable post hoc paired McNemar CI
construction. Separate source-anchored prospective diagnostic and method
conflict memo preserve both margins and the original fallback order:
v1 fails Gate1, so v2 and REAP160 must be qualified; no optional substitution
or final selection can be certified without complete genuine original data.

Frozen determination:

    {"status":"BLOCKED","chosen":null,"release_approved":false}

## Next executable work

Let the source-bound serialized Q4 HumanEval runner complete and independently
rescore all164. When the first original 164 is finished or has exited, source-
verify the next exact Q8 index3 and resume its separate Q8 journal serially.
Continue the distinct currently running native Q4 AutomationBench guard.v2
original index sequence and audit its original source-bound receipts; resume
historical guard.v1 on a separate exclusive same-policy serving session only
when the shared GPU is actually free. Qualify original dense/Q8/BF16 and
frozen four-slot, native first200 on a physically admissible dedicated runner,
then complete other authentic official eligible frontier suites.
No release, deployment, Gen1, altered weights or PR merge.
