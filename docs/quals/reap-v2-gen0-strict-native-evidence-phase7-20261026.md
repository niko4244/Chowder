# REAP128_v2 Gen-0 — strict native evidence and terminal safety, phase 7

**Original decision: `BLOCKED`, `chosen:null`, `release_approved:false`.**
Recorded 2026-10-26. This is a CPU-only integrity and previously
**unlaunched dense runner** hardening wave. No original GGUF, frozen
statistical method, prompts, scorer, serving flags or original
historical 16 Kaggle snapshots were changed.

## Live source, model, control and service state

At entry both feature HEAD and origin were
`d61d7b8e1ccd4651a66fed6ee47bb9893a29d053`,
`niko4244/Chowder#282` OPEN / DRAFT / UNMERGED, feature working
tree clean. The two independently checked original **private** jobs:

- `nikmarco/reap128v2-q8-full-postload-reclaim`: **COMPLETE**
  with original full Q8 source and native evidence already privately
  recovered, preserved and separately qualified.
- `nikmarco/reap128v2-random128-full-postload`: **RUNNING**
  on the most recent source-side check; no replacement or duplicate.

Original Spark :8767, Ollama :11434 and model router :8766
returned HTTP 200. Observed RTX5060Ti free VRAM fell to
**3,920 MiB** in a live check, below the original **12,288 MiB**
safe CUDA start. Available Windows physical RAM was around
14 GiB; observed commit **120.627 GiB against limit
129.023 GiB** (only **8.396 GiB** unused commit).
Those are time-variable host counters and do not certify a
model-running high-water mark. No model or GPU service was touched.
The five protected original BF16, Q8, Q4, dense Q4 and random128
weight files were confirmed present at their separately
known byte sizes; full source SHA receipts were reused instead of
needlessly rehashing 100+ GB.

## Native original identity and per-token scale

| Protected source (GGUF) | Bytes | Immutable separately source-hashed SHA-256 |
| --- | ---: | --- |
| REAP128_v2 BF16 intermediate | 37,122,439,488 | `80341aa776ad9b5f937c21b081b6d76de4a2ea277e5fb88c9ae2634a9da8f4c6` |
| REAP128_v2 Q8_0 | 19,748,435,968 | `89dcaf33b61f59c83a31b718ba98b73c33437d0320ae0cc56737a5d7e43519c6` |
| REAP128_v2 Q4_K_M | 11,563,915,264 | `2d64b786461b99a335fdbbdf7c4d3a6758b21409bce755751714ab7cd8f63bf7` |
| Original dense Q4 reference | 21,166,757,984 | `2c6d9466129e8802839566b35728e2506aaf1e8705cd137009e2f08feb9dd049` |
| Independent random128 Q8 control | 19,748,435,872 | `601871e00403658708eb4eebf3b05f9dc54a93b326e31f2f87b48cb59f92241b` |

40 decoder layers × 128 routed experts per layer,
eight selected experts/token: **18,543,997,568 total stored**
weight elements, **2,437,870,208 non-routed**, **16,106,127,360
all routed**, **1,006,632,960 selected routed weights/token**,
approximate **3,444,503,168 active weight elements/token**
(18.575% of stored weights). These are weight-element counts,
not measured CUDA FLOPs/token.

Previously qualified numerical source sample: direct matched-source
Q8↔Q4 dequantized 122,880 values MAE **0.0005579055**,
RMSE **0.0007166865**, max |diff| **0.0039260387**.
BF16↔Q8 MAE **0.0000478359**; BF16↔Q4 MAE
**0.0005555331**. Weight samples do not establish forward
CUDA logits, all-token numerical parity or equivalent output
across architectures. Original Q4 Windows vs Linux first
GSM8K answer already differs at first character offset 34.

## Full original Q4/Q8 Linux research results — preserved

| Original full item source | Q4_K_M | Q8_0 |
| --- | ---: | ---: |
| GSM8K complete scored correct | **1,229 / 1,319** | **1,220 / 1,319** |
| GSM8K natural EOS | 1,281 | 1,280 |
| GSM8K generated tokens | 456,671 | 467,560 |
| GSM8K wall time | 4,639.150 sec | 5,726.947 sec |
| GSM8K sustained tokens/sec | **98.4385** | **81.6421** |
| HumanEval original-scored correct | **151 / 164** | **152 / 164** |
| HumanEval natural EOS | 162 | 162 |
| HumanEval generated tokens | 40,803 | 40,281 |
| HumanEval wall time | 457.068 sec | 538.774 sec |
| HumanEval sustained tokens/sec | **89.2712** | **74.7642** |

Original Q4 independent low-privilege HumanEval re-execution
matched **164/164** original-scored row decisions, although
its older sandbox's full namespace/descendant containment is
not proved by `python -I`/resource limits alone. Q8's **152/164**
remains **original-scorer executed only; fresh independently
sandboxed 164-item execution PENDING** because the separate
private CPU-only Kaggle capability probe did not establish
the necessary network/mount/PID isolation. All 1,319 Q8 GSM8K
rows were independently rescored. Original Q4 and Q8 native
CUDA records each show **41/41** layers offloaded,
`--fit on -np 4 -c 16384 --jinja` and **320 MiB total KV**
for four × 4,096-token native contexts. Shared-object CUDA
library versions/hashes were **not** captured in those
historical GPU logs and cannot now be inferred from offload.

Original full GSM8K Q8−Q4 pairing: 1,198 both correct,
68 both wrong, 22 Q8-only correct, 31 Q4-only correct,
observed **−0.006823351** absolute accuracy, Linux descriptive
only, no paired CI or Gen-0 promotion. Random128 fresh pair
remains unknown while original job runs. Historical REAP160
GSM8K accuracy **0.94313874147**, HumanEval accuracy
**0.95731707317**; original REAP128-v1 Gate1 **FAIL**.
Historical separate REAP160 3.852× speed and 41/41 fit
evidence do not bind to original canonical 200-item serving
session, so canonical REAP160 speed remains **UNKNOWN**.

Q8 complete native resource observations: **1,284 samples**,
of which 1,242 cover scoring time **6,263.805 sec**
versus independent original GSM8K+HumanEval wall sum
**6,265.721 sec**, max sampling gap **5.058 sec**.
Minimum sampled unused cgroup **10.467 GiB**,
available physical RAM **21.101 GiB**, GPU-free
**4,967/5,363 MiB** from original two-T4 startup
**14,912/14,912 MiB**. These are bounded sampled
free-memory minima, not original Windows commit or
instrumented instantaneous allocator peaks.
Independent full Q8 strict UTC-watchdog audit SHA
`107395f480693922bd279a50138338db14c78441320a7060914f0d82cdd73683`
is retained in private F: evidence.

## Newly closed trust and termination gaps

**Strict original evidence JSON parsing.** Python's default JSON
decoder accepts duplicate object keys (last value wins) and
nonstandard `NaN/Infinity`. A malicious/torn native receipt
could have two distinct fields named `model`, `status`,
`correct` or `session` yet be silently reinterpreted as
one asserted certificate. The full Q4/Q8/random128 source
auditor now rejects duplicate keys at *any nesting depth*
and all nonfinite JSON constants before model/source/row
scoring. The separate Q8/random128 HumanEval
**non-executing** private preflight and downstream
Q8/Q4/control descriptive comparator use exactly the same
strict source parser. Tests demonstrated multiple RED
acceptances before the fix and GREEN refusals afterward,
including a forged row whose edited SHA was recomputed
to make ordinary row-integrity checks otherwise pass.

The exact full Q8 original machine and both 1,319/164 row
files passed the stronger source-bound parser without
changes to their original content. Reproduced same
strict full Q8 research audit SHA
`107395f480693922bd279a50138338db14c78441320a7060914f0d82cdd73683`.
No HumanEval generated Python was executed by this
read-only re-audit.

**Unlaunched dense native runtime OOM and provenance.**
Independent scrutiny found the additional source-only dense
runner checked cgroup OOM counters merely around its
postload cache advisory, allowing a fresh OOM event during
hash, model loading or 200-item inference to escape terminal
admission. It previously also wrote terminal
`COMPLETE_DENSE_FIRST200...` **before**
the owned server had exited and before any final server-log
SHA was computed; the unrestricted `read_bytes()`
could also itself exceed a constrained cgroup's reserve
if native logs grew without bound.

The new unlaunched source now snapshots cgroup
`memory.events` OOM/kill counters **before the first
GGUF read**, fails closed on nonzero baseline or any
observed new event at every original `check_runtime`
during source hash, model load, postload and inference
and after owned termination. It rejects symlinked source
models and changes to original model file
device/inode/size/mtime/ctime across hash, prelaunch,
postload and completed scoring. It holds a nonterminal
`SCORED_AWAITING_NATIVE_SHUTDOWN_AND_LOG` status until
the owned server has stopped, a bounded **32-MiB** complete
native log has been incrementally SHA-256-hashed and
the final OOM witness is clean. A missing/oversized
log, live process or cleanup failure produces
`REFUSED_OR_ABORTED`, not source-gate PASS.
This is a safety improvement, not proof that existing
private Kaggle cgroups are truly process-isolated.

The new runner remains **SOURCE-ONLY, never executed**;
its separate exact SHA witness is retained in
`scripts/reap_v2_kaggle_qualification_20261026/SOURCE-SHA256-UNLAUNCHED-DENSE-20261026.json`.
Its 23,235-byte normalized-LF source SHA-256 is
`603c789422a0a2e40e3d7eb76a4594aad7335ba5c61e501ea132ca34be754930`.
Previously protected historical source manifest SHA
`f52c2f2862b8f3acc3f690e0fca71ed50864b0f10ce17129343a4b8b8f73462a`
has not changed.

**Reproducible validation:** 650 adjacent CPU-only scientific tests
passed; one Windows symlink-permission test skipped, one environment-bound
qualification intentionally deselected, five existing config/deprecation
warnings. Strict JSON adversarial tests were observed RED before parser
changes, then GREEN; dense lifetime/terminal pre-change RED, then GREEN.
Ruff and Git whitespace checks passed. No remote model or GPU was started.

## Remaining original admission and scientific gates

| Original registered requirement | Current fact |
| --- | --- |
| BF16, Q8, Q4 source/weight ancestry and tensor census | PASS |
| Full Q4 and Q8 original Linux 1,319 GSM8K quality | PASS (Linux provisional) |
| Q4 independently run 164 HumanEval | PASS historical replay, sandbox limitations disclosed |
| Q8 HumanEval original executed score | 152/164; independent replay **PENDING** |
| Fresh random128 full source-matched control | **RUNNING / UNKNOWN** |
| Fresh dense_v2 reference quality and 200-item original throughput | **UNKNOWN** |
| Candidate Q4 frozen original first-200 rate | 101.426 generated tokens/sec (registered Q4 session only) |
| Original first-200 candidate/dense throughput ≥1.3× | **UNKNOWN**; dense reference absent |
| REAP160 canonical first-200 speed | **UNKNOWN**, unrelated historical probe cannot supersede |
| Original RTX5060Ti commit/VRAM safe admission | **FAIL at observed GPU headroom**, no new run |
| BF16↔Q8/Q4 forward CUDA, Linux/Windows output parity | **UNKNOWN**, actual first answer divergent |
| Optional Amendment4 §2 paired McNemar-style nonzero-margin CI | **UNRESOLVED** independently approved prospective method |
| Frozen original Gate0 / Gate1 / Gate2 complete evidence | **INCOMPLETE**, historical REAP128-v1 Gate1 FAIL |
| Deterministic original Gen-0 selector/release | **BLOCKED**, chosen null, release false |

Original §4 preregistration defines a paired sample-Wald
one-sided bound, while Amendment4 optional §2 invokes
a nonzero-margin paired McNemar-style interval without
fixing a unique paired risk-difference construction.
Independent read-only statistical review rejected
post-hoc switching. A new prospective method requires
authorization before **new confirmation outcomes** are
seen; previously unblinded originals remain exploratory
on that optional interpretation. Mandatory Amendment4
fallback is separate and retains the original selector
order; no optional interval may be smuggled into it.

**Precisely reproducible remaining work:** finish the
already running unique random128 full original private
experiment, preserve exact its own 1,319+164 row/session,
native receipt and full log, independently audit/cross-pair
without combining original sessions. Obtain a genuinely
verified **≥33.452 GiB unused cgroup** private runner
(preferred ≥48 GiB unused cgroup, ≥64 GiB host RAM,
large real CUDA VRAM), but do not repeat the earlier
29.444-GiB failed model load. Before any new dense launch,
capture and independently pin native executable **and
all dynamically loaded CUDA/shared-library hashes**,
isolated container/cgroup identity, complete original
first-200 native session, recorded Q4 candidate/dense
same-serving-geometry sustained output rate. Use exact
`--fit on -np 4 -c 16384 --jinja`.
For HumanEval independent correctness use a different
private runner with verified network, user, PID, mount,
cgroup and process-tree isolation; the earlier Kaggle
probe demonstrated no adequate sandbox. Advance
BF16/CUDA logit and Windows parity only under the
original GPU/RAM/commit safety floors and verified
resident-service restoration if any targeted pause is
actually required.

**No model or private source release, draft PR merge,
deployment, or Gen-1 training.**
