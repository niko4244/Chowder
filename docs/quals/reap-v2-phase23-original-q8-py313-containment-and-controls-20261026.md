# REAP128_v2 original-Q8 HumanEval v2 admission and controls — phase 23

**2026-10-26. Public-safe, source-bound engineering report.**
This document covers a *new* independent nonexecuting admission/security
wave, not a Gen-0 selection. It extends the original source, runtime and
memory qualification recorded in
`reap-v2-phase19-windows-q4-resource-and-gen0-state-20261026.md`.
No generated HumanEval program was executed in phase 23, and historical
model-weight bytes, generated output texts, evaluator and scorer were not
modified.

## Actual original-Q8 source provenance: passed without execution

The **unchanged** complete original Q8 164-item HumanEval original native
machine SHA `92ca9969ef5959714f4787ed10a4375c69c2c4a80a4842daabc01c2360543fc0`,
strict independent full audit
`107395f480693922bd279a50138338db14c78441320a7060914f0d82cdd73683`,
canonical full population SHA
`5144d31cf337b25ce262b780ed103b128b1e9d289d3c0d15f4fe34933bb705d5`,
original row SHA `cd3c9d1577611a032b36e69a3f96bd272d118819074352503f143bcd68cac301`,
original frozen evaluator/scorer SHA references and pinned prior harmless
256-MiB capability census were checked with
`reap_humaneval_podman_replay.py` source-only mode.

Result: `SOURCE_ONLY_PREEXECUTION_ADMISSION`,
`generated_programs_executed=0`, `private_output_written=false`,
`gen0_status=BLOCKED`. The original source/provenance hash compatibility
issue is resolved; the v2 *execution* qualification remains separate.
Neither the source-only check nor the previous self-asserted six-field
receipt establishes independently safe execution.

## New actual fixed-source Python 3.13 OCI census

A distinct public `docker.io/library/python:3.13-slim` image was
provisioned without restarting, reconfiguring or touching existing
Hermes/Brainz containers. Its **exact local image ID**:

`72036a705a7cff38924efd674e60654666cb283faf7b3fad160b34007f7395bd`

A fixed harmless Python source observed **CPython 3.13.17** with isolated
mode enabled. The 768-MiB native replay-profile container was rerun using
the new pinned image with *only benign census code*, verifying rootless
UID/GID 65534, seccomp mode 2, no effective capabilities, no-new-privileges,
read-only root, no default route, empty secrets mount, cgroup memory
`805306368` bytes and pids.max 16. Container was removed after normal
benign completion. Exclusive **private** source-bound harmless-census
receipt SHA-256:

`e7ff8133498997f7e1867ef62717e291001a4636feb9dc9afef21754c5add7cd`

The current original replay runner is **still bound to its earlier pinned
Python 3.11 image and harmless 256-MiB evidence**; installing an independent
Python 3.13 image does not silently replace the original preflight/image
hash. A new exact-profile v2 evidence handshake is still required before
the runner can use the 3.13 image safely.

## Hard containment failure: no independent VM descendant-PID witness

Two additional *fixed harmless parent-spawns-child* timeout probes used
the pinned Python 3.13 image (never model output), disposable unique
container names and the native rootless/PID-private/768-MiB profile.
Both observed the fixed parent/child marker, native Podman timeout with
Windows CLI status `4294967295`, and the isolated container name absent
afterward. The first private receipt SHA:
`ecbbd6a738d999c3c721e0056d873f9785d3255b6b44eb6c3c1080b86270bc62`.
The second (external Linux-VM `wsl -u root` read-only PID-census) SHA:
`da86c4ae0e31555169902dd0d532c43d4c8145655ae3e84ac4ce4dc232f99416`.

**Neither probe could independently locate the OCI inspect-reported PID
inside the WSL guest's /proc or process tree**, even with a root read-only
observer. Accordingly the external attestor did not witness any child PID
starttimes or prove those exact host-visible descendants absent afterward.
Both private receipts explicitly say `REFUSED_UNQUALIFIED`. This is a
*visibility and independent-qualification failure*, not evidence that
the processes actually escaped. Absence of the Podman container name
cannot substitute for independent child/cgroup extinction.

Do not disable host namespaces, remount host paths, weaken `--network=none`
or kill/restart unrelated containers to manufacture a passing probe.
The next safe implementation route is an **isolated dedicated Linux host**
where the trusted observer is outside and above the rootless OCI worker PID
namespace and can bind each child PID + creation time + cgroup ID to an
exclusive owner before/after timeout. Alternatively establish the correct
rootless Podman VM PID mapping with a trusted host-level observer and test
it with fixed harmless processes first.

## Another independent original-scoring contract mismatch

The registered original scorer executes the generated program as
`python -I <temporary-directory>/candidate.py`, inside that temporary
working directory, under `subprocess.run(..., timeout=10)`.
The as-yet **unqualified** OCI replay instead uses `python3 -I -`
(stdin source rather than a file), an OCI `--timeout=14` and a
23-second host CLI timeout. `__file__`, current working directory,
possible import behavior, and whether a 10- to 14-second completion is
counted are not necessarily equivalent. No v2 independent PASS may be
claimed until the exact original registered execution semantics are
implemented and benign/adversarially verified without modifying the
original evaluator or prompt/output rows.

## Consequential RED/GREEN infrastructure provenance fix

Previous `podman_exec()` could turn a Podman **125/126/127** launch error,
**137/143** containment/resource termination, or nonportable **255/-1 /
4294967295** process error into an ordinary `NONZERO_EXIT` failed program.
An unexecuted original-failing HumanEval row would then appear to match
the historical scorer, making a claimed 164/164 replay potentially false.

Eight adversarial tests first proved this as a real **RED**. The corrected
runner now refuses the **entire** scientific replay on negative or
greater-than-124 subprocess statuses, while retaining legitimate original
Python `exit 0` success and `exit 1/2` failed-test handling. This is a
deliberate fail-closed policy: an unusually high genuine Python exit must
be independently investigated rather than counted as a model failure.

A separate RED/GREEN source gate now also refuses execution if **any** of
these six conditions is not satisfied:

1. `PINNED_PY313_REPLAY_IMAGE_REQUIRED` — actual replay binding must use
   a verified original-compatible 3.13 image; earlier 3.11 source-bound
   capability evidence cannot be automatically reused.
2. `INDEPENDENT_VM_PID_TREE_EXTINCTION_REQUIRED` — actual external
   child PID/cgroup start/end evidence, failed in both real VM probes.
3. `ORIGINAL_10_SECOND_PYTHON_TIMEOUT_EQUIVALENCE_REQUIRED`.
4. `ORIGINAL_CANDIDATE_FILE_AND_CWD_EQUIVALENCE_REQUIRED`.
5. `TRUSTED_PER_TASK_REPLAY_V2_ATTESTATION_REQUIRED` — original source,
   exact executed program SHA, launch identity, Python version, exit vs
   infrastructure/error classification, and an immutable externally
   verified per-task execution witness; the legacy six-field JSON
   is categorically insufficient.
6. `GENERATED_EXECUTION_SOURCE_DISABLED` until independent human-approved
   runtime qualification.

Changing just the `--execute-qualified` CLI flag or the old Boolean gate
does not make the rest of the requirements pass. The strict original-Q8
source-only admission now lists its failed requirements and still writes
**no** execution receipt or model-generated source.

## Original quality/control and remaining scientific gates

The independent complete original **Linux research** source rechecks
remain Q4 GSM8K **1,229/1,319**, HumanEval **151/164** and Q8 GSM8K
**1,220/1,319**, HumanEval **152/164**, all under their own exact source
and serving sessions. A separate **read-only** actual original HumanEval
boolean pairing (without any code execution) newly verified **147 both
correct, 8 both incorrect, 5 Q8-only correct, 4 Q4-only correct**.
Both models naturally terminated on the same 162 items; two were
unterminated on both. Q8−Q4 is **+1/164 = +0.609756 percentage points**
on *recorded original scorer decisions*, not independently isolated v2
code outcomes. The separate private research-only pairing SHA:
`42661438b44bd35d3225fd5fa5817a7c5de521a13da59e1e9efefd8489fd8756`.
No confidence interval was selected, and the strict paired HumanEval
consumer stays blocked until independently qualified execution.

The official frozen local Q4 admission was retried **read-only**:
observed available RAM 16.340 GiB, Windows commit 22.669 GiB and GPU
10,578 MiB versus required minimum **29.315 GiB, 34.315 GiB, and
12,288 MiB**. All three mandatory source-bound checks **REFUSED**, so
no additional shared-host model execution was attempted. The previously
source-qualified genuine Windows Q4 first-200 candidate continues to be
187/200, approx. 183.98 generated tokens/sec, 41/41 CUDA placement and
the complete 148-sample resource audit; do not rerun it unmodified.

The original independent **random128** full Kaggle kernel remained
`RUNNING` at the last check. The original dense first200 cannot be
run on the 4-GiB Podman VM or a shared occupied 16-GiB 5060 Ti.
For Linux's *provisional* snapshot the existing prehash source-bound
admission requires **>=33.452 GiB truly unused cgroup**, >=24 GiB
available physical and **two GPUs each >=12,288 MiB free** plus
postload clean-cache recovery before the first scored item. These
Linux research conditions do **not** establish a matched original
Windows comparison. The original Windows same-hardware candidate-vs-dense
first200 should instead be attempted in a **dedicated** Windows environment
with abundant physical+commit headroom (preferably >=96 GiB RAM and
matching pagefile), a full-device GPU with >=48 GiB VRAM for BF16/dense
studies, pinned native CUDA build/runtime, ample NVMe/disk reserve,
exclusive native servers, and original frozen flags and source datasets.
This hardware plan is a **proposed capacity configuration**, not a
measured admission guarantee or newly completed benchmark.

Retain the **original Amendment 4 selector unchanged**. Historical
REAP128-v1 Gate1 FAIL and REAP160 throughput UNKNOWN still govern
fallback. Resolve the preregistered Wald vs Amendment4 paired McNemar
nonzero-margin CI ambiguity **prospectively on a new blinded confirmatory
cohort**, not via post-outcome method shopping on the known Q8/Q4 rows.

**Scientific decision: `BLOCKED`, `chosen=null`,
`release_approved=false`. PR #282 OPEN/DRAFT/UNMERGED. No deployment,
release, Gen-1, protected GGUF edits, unpublished answers, untrusted
generated execution or protected service disruption occurred.**
