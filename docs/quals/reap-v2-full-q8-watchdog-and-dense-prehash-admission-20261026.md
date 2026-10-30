# REAP128_v2: independent Q8 sustained-run and dense pre-hash safety wave

**Scientific status: BLOCKED; chosen: null; release_approved: false.**
Date 2026-10-26. Original frozen Amendment 4 selector, thresholds, sources,
decoding and full 1319+164 scoring remain unchanged. The original
REAP128-v1 Gate1 remains FAIL and canonical REAP160 speed remains UNKNOWN.
No private weights or benchmark generations are in this repository.

## Live reconciliation / inputs

At the entry HEAD and origin both matched
`faf28f2b56dda0908a6ca1178662cc70e71c692d`,
PR `niko4244/Chowder#282` was OPEN and DRAFT.
The complete original Q8 private Kaggle job
`nikmarco/reap128v2-q8-full-postload-reclaim` remained **COMPLETE**.
Its separate negative-control sibling
`nikmarco/reap128v2-random128-full-postload` remained **RUNNING**
at repeated checks, and was never restarted or replaced.

All five immutable model files were present at their recorded sizes,
including the original BF16 37,122,439,488-byte intermediate, Q8_0
19,748,435,968-byte, Q4_K_M 11,563,915,264-byte, original dense Q4
21,166,757,984-byte and independent random128 Q8 19,748,435,872-byte
weights. Earlier full-file SHA attestations were reused instead of
needlessly rehashing 100+ GB of protected model data. Original Windows
Spark, router and Ollama returned HTTP 200 throughout these checks.
RTX 5060 Ti free memory ranged approximately 10,842–11,193 MiB,
below the **12,288 MiB** frozen CUDA admission and therefore insufficient
for an unapproved new original full-model GPU serving attempt.
Observed physical memory available was around 13.4–14.0 GiB and Windows
commit headroom is dynamic; no local model was loaded or resident
service paused.

## Q8 original full 1319+164 independent requalification

Previously recovered original private evidence directory:
`F:\reap-v2-q8-postload-full-complete-original-evidence-20261026-phase4`.
Original model Q8 SHA:
`89dcaf33b61f59c83a31b718ba98b73c33437d0320ae0cc56737a5d7e43519c6`.
Single original serving session
`95876063255e45159f57e29306e0e885`; pinned source evaluator/scorer,
full canonical prompt and input digests, original 41/41 native CUDA
offload and 4×4096 geometry; original
`--fit on -np 4 -c 16384 --jinja`.

| Original Q8 outcome | GSM8K (complete 1319) | HumanEval (complete 164) |
|---|---:|---:|
| Original-scored correct | **1220 / 1319** | **152 / 164** |
| Natural EOS | 1280 | 162 |
| Generated tokens | 467,560 | 40,281 |
| Source-bound complete-cohort wall time | 5726.947s | 538.774s |
| Sustained generated tokens/s | **81.6421** | **74.7642** |

GSM8K was independently **rescored** for all 1319 exact items.
The original-scored HumanEval 152/164 remains
**PENDING independently contained executable 164-program replay**:
the separate private CPU-only Kaggle harmless sandbox probe proved
neither a usable isolated network/user/PID/mount launcher nor
delegated cgroup descendant process limits; generated code was not
executed by the newer independent preflight or probe.
Never promote its currently reported correctness to independently
executed/pass merely because rows and SHA receipts are complete.

The earlier Q8 original full native source audit SHA
`bc7eb4ba83475db3b82f648a3ce50b8b7ef8ef4db030a285a227837e5dffdb48`
and original Q4 independent full audit SHA
`163d2189f220fd04620f1d94a79e899624d43ef1e8bc6a5d462159fd2e178b6f`
were rechecked without modifying their private files.
The old paired original GSM8K descriptive receipt SHA
`125df9301e222f3f9220742514f279b1b6ff4b31444b709afa0ab4bfe8e13b93`
remains 1198 both correct, 68 both wrong, Q8-only 22,
Q4-only 31; **Q8−Q4 −0.0068233510** accuracy difference
without a post-hoc or claimed official noninferiority interval.

The Q8 original 1284 native GPU/cgroup observations were parsed from
its raw source receipt. **1242 actual UTC-stamped scoring-time samples**
run from `2026-10-26T09:08:18.459718+00:00` through
`2026-10-26T10:52:42.264790+00:00` and span
**6263.805072s**, versus the separate measured original GSM8K +
HumanEval wall sum **6265.721s** — a difference of only **1.916s**.
The largest recorded sample-to-sample gap was **5.058341s**,
median 5.047s and no reversal or >30s holes.
This is meaningful **continuous monitoring coverage** of the
original complete candidate, not merely two startup telemetry points.

New independent external full-audit code rejects incomplete
Q8/random128 watchdog coverage, missing/wrong exact
1319/164 suite summaries, nonfinite/bool/zero wall durations, reordered,
duplicated, skipped or overly sparse runtime UTC timestamps, and
incorrect measured-vs-sampled total wall duration. The strict
original full-Q8 re-audit passed with these new exact safeguards,
preserved privately as
`F:\reap-v2-q8-original-independent-full-watchdog-continuity-audit-20261026-phase5.json`,
SHA **`107395f480693922bd279a50138338db14c78441320a7060914f0d82cdd73683`**.
Its exact report fields: 1242 active samples, 6263.805 measured seconds,
6265.721 registered seconds, maximum gap 5.058 seconds; native initial
GPU-free 14912/14912 MiB, sampled minimum 4967/5363 MiB and
sampled change 9945/9549 MiB. Cgroup unused minimum 10.467 GiB;
physical MemAvailable minimum 21.101 GiB. These sampled deltas
are NOT instantaneous instrumented VRAM allocation peaks, nor
Windows commit values or forward CUDA numerical parity.

## Unlaunched dense_v2 reference — actual safety defects fixed

Independent read-only engineering review identified a real
**pre-admission container pressure bug** in the NEW source-only
`dense_q4_first200_material_postload.py`: despite requiring
**33.452 GiB** genuinely unused cgroup capacity at *native model load*,
it previously streamed and hashed the **entire 21,166,757,984-byte
dense Q4 model before measuring/admitting cgroup or GPU resources**.
Hash-stage page-cache charges could therefore breach the runtime
floor even on the already-rejected 29.444-GiB allocation.

The guarded research runner now:

1. Performs the exact 33.452-GiB-unused-cgroup,
   ≥24-GiB available-RAM, 12,288-MiB-free-per-two-GPU and ≥8-GiB-disk
   **admission BEFORE the first model GGUF read**, then checks the
   original runtime floors during hashing every 256 MiB, evicts only
   clean cache by advisory, and **re-admits** before native serving.
2. Attributes mandatory ≥4-GiB post-load recovery not merely to
   `memory.max - memory.current` improvement, but to an actual
   supporting ≥4-GiB `memory.stat file` charge decrease,
   a reasonable match between charge improvement and file decrease,
   no native `memory.events oom/oom_kill`, and ≥14 GiB actual postload
   headroom. Rejects malformed OOM/cgroup/file amounts, including
   booleans, NaN and missing sources.
3. Applies the 7-GiB physical/10-GiB cgroup runtime floor before
   and during each second of the original five-second
   `POSIX_FADV_DONTNEED` recovery observation.
4. Quiesces and joins the real resource/ownership supervisor and
   refuses a terminal receipt on `fatal` or a live watchdog before
   writing `COMPLETE_DENSE_FIRST200_PROVISIONAL_NOT_FROZEN_WINDOWS_GATE`.
5. Preserves source GGUF SHA, original 200 GSM8K items, exact
   pinned four-way serving flags and source-pinned original scorer
   and evaluator. The historical sixteen immutable Kaggle snapshots
   and original aborted experiment were NOT changed.

RED adversarial tests demonstrated the prehash ordering and cache
false-attribution flaws. Updated scoped **46 tests PASS** including
historical immutable-manifest continuity. The modified **new unlaunched**
dense source is separately byte-hashed in
`SOURCE-SHA256-UNLAUNCHED-DENSE-20261026.json`.
The complete adjacent CPU-only scientific matrix subsequently passed
**634 tests**, with one skipped Windows symlink-permission test,
one intentionally deselected native/pinned environment test and
five existing pytest/deprecation warnings. Ruff and `git diff --check`
passed. The new unlaunched dense source SHA-256 is
`dab245e795018943cdc9d27a57fdb715bd3910dcb8f235785b3009b42d3f56f0`
(20,437 source bytes).
The original 16-snapshot manifest SHA
`f52c2f2862b8f3acc3f690e0fca71ed50864b0f10ce17129343a4b8b8f73462a`
is unchanged.

**No dense job was launched**, no original 29.444-GiB runner reattempted,
and there is still no original dense first-200 throughput, same-session
1.3× candidate/dense speed ratio or proven model fit.
The new dense source *still does not* independently bind the 17,888-byte
pinned launcher to all CUDA shared-object hashes or prove a new runner
with ≥33.452 GiB actually exists; these remain explicit engineering and
resource blockers, not inferred PASS.

## Gate and next execution status

Frozen historical REAP128-v1 Gate1 remains **FAIL**. REAP160 quality
is historical only; historical separately executed 41/41 fit and
3.852× descriptive speed ratio are not bound to the original
200-item `serve.json` owner, so **canonical REAP160 speed UNKNOWN**.
The old dense first-200 reference aborted before full model load,
fresh full random128 original control is **RUNNING** and must not be
duplicated, original BF16/RTX5060Ti versus T4 forward numerical parity
is unestablished (Q4 first output diverged), and the original
Amendment4 optional nonzero-margin McNemar-versus-Wald paired-CI
construction requires **prospective independently authorized
method adjudication before outcome-informed method selection**.
No approval has been issued or assumed.

Reproducible next execution requires:
an actually independently measured new private dense-capable
runner meeting at least 33.452 GiB *unused* cgroup at startup,
prefer 48+ GiB unused cgroup and 64+ GiB available host memory,
adequate CUDA VRAM (prefer 48-GiB class), frozen
`--fit on -np 4 -c 16384 --jinja`, bound native CUDA shared libraries
and clean OOM counters. Never substitute the
postload cache into truly unused pre-load cgroup space.
Preserve each 200-item serving session independently, then
compare the candidate Q4 first-200 and truly completed
matching dense-Q4 first-200 only when same-platform and preregistered
constraints genuinely match.

When the existing original random128 job finishes, recover **its own**
native JSON, full native log, all 1319/164 rows and session SHA
separately; audit, pair on exact source IDs and leave generated
HumanEval execution unverified until safe independent OS isolation.
Original deterministic Amendment4 selector remains unmodified and
must run only against fully eligible evidence.

**No original model modification, new GPU server, disruption of
Spark/Ollama/router, release, PR merge, deployment or Gen-1 training.**
