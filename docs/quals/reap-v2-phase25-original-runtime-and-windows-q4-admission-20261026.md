# REAP128_v2 phase 25 — actual WSL containment and Windows Q4 admission

**2026-10-26. Public-safe, source-bound continuation from HEAD `337338526a130a46008cfa83b08436a67e2a42e3`.**
This is a qualification *refusal and engineering correction*; never a new
HumanEval quality score, a changed registered scorer, a memory-floor waiver,
a protected service restart or deterministic Gen-0 model selection.

## Reconciled original data before executing anything

Original REAP128_v2 candidate construction remains 40 layers, 128 routed
experts per layer, eight selected per layer. Weight-element totals are
18,543,997,568 stored, 2,437,870,208 non-routed and 16,106,127,360 stored
routed; 1,006,632,960 routed weights selected per token, or approximately
3,444,503,168 active weight elements per token (18.575% of stored);
320 expert selections/token. This is **not** a FLOP or exact inference
operation count. Immutable GGUF byte sizes: BF16 **37,122,439,488**;
Q8_0 **19,748,435,968**; Q4_K_M **11,563,915,264**; dense Q4
**21,166,757,984**; random128 Q8 **19,748,435,872**.

Two original continuous Linux research populations were already completed
and independently source-audited: Q8 1,220/1,319 GSM8K, 152/164 original
HumanEval, observed 81.6421 / 74.7642 generated tok/s; Q4 1,229/1,319
GSM8K, 151/164 HumanEval, observed 98.4385 / 89.2712 tok/s. These are
different original serving sessions and **NOT** original Windows gate
decisions or a matched speed ratio. Paired original GSM8K Q8 vs Q4 has
1,198 both correct, 68 both wrong, 22 Q8-only and 31 Q4-only;
Q8−Q4 = −0.6823 percentage points. Recorded original HumanEval booleans
have 147 both correct, eight both wrong, five Q8-only and four Q4-only,
not a safe independent original Q8 164-program replay.

The already qualified registered Windows Q4 candidate first 200 is
**187/200 correct**, **192/200 terminated**, 73,224 generated tokens / about
398 seconds ≈ **183.98 tok/s**, all 41/41 model layers on CUDA with no routed
experts offloaded; 4 concurrent slots of 4,096 tokens each (16,384 total),
native GPU KV buffer 320 MiB and native CUDA model buffer 10,619.88 MiB.
Its immutable 148-sample watchdog observed prompt-cache maximum 8,150.615
MiB (native limit 8,192 MiB); minimum sampled available host RAM
10.064 GiB, free Windows commit 14.724 GiB and GPU VRAM 2,859 MiB;
largest sampled *host availability decline* 21.043 GiB, 23.382 GiB and
11,487 MiB, respectively. These are NOT independently isolated model
allocation peaks and must not be added to cache/KV as disjoint quantities.
Do **not** rerun this completed candidate session or splice it into a
purported Windows full 1,319 result.

Historical initial Windows 12-row run source receipt:
`F:\reap128-v2-q4-registered-speed200-20261025-supervisor.json`, source
SHA `836e4e886f11819075d75a461eec028f0453b9fd3d00e2c9553e0d3177dd81ed`;
native start/log SHA `94057821a77c7eaf15384850d1b155cb540904ef10e206a5f394289faa6012f1`.
This was an **abort**, not quality or throughput proof: available physical RAM
fell from 21.41 GiB to **6.98 GiB**, below original 7-GiB runtime floor,
and available Windows commit declined from 28.84 GiB to **10.41 GiB**.
No local workaround may reinterpret the 12 partial rows as completion.

## Actual current Windows admission result

Fresh, CPU-only host counters without stopping anyone's processes:

| Counter | Actual available | Unchanged original Q4 full admission |
|---|---:|---:|
| Physical RAM | ~15.18 GiB | >=29.315 GiB |
| Windows commit headroom | ~11.73 GiB | >=34.315 GiB |
| RTX 5060 Ti GPU VRAM | 565 MiB | >=12,288 MiB |

`nvidia-smi` recorded RTX 5060 Ti total 16,311 MiB, used 15,486 MiB,
free 565 MiB. Windows memory counters recorded committed 123,777,400,832
bytes of commit limit 136,371,810,304; physical available 15,605 MiB.
Transient host counters and WDDM accounting do not establish per-process
peak device allocation. Protected concurrent listeners were present on
11434 (Ollama), 7050 (Hermes), 8766 (router), 8767 (Spark).
No arbitrary kill, pagefile adjustment, prompt-cache weakening, four-slot
reduction, CPU expert spill, model rerun, CUDA consumption or service restart
was admitted. The actual current measured host fails all three entry floors.

## An actually tested alternative HumanEval sandbox: still refused

The original private Kaggle CPU namespace capability probe remains an
unchanged source-hashed **REFUSED** receipt: no user/PID/mount/network
namespaces or delegated cgroup could be demonstrated. The old restricted
Q4 151/164 Kaggle replay is not a v2 independent sandbox certificate.

A distinct **harmless fixed-code** candidate probe was run as an existing
non-root (`uid 1000`) ordinary Ubuntu WSL2 user. A new temporary unshare
process used `--user --map-root-user --mount --pid --net --fork --mount-proc`
without executing any generated program. Actual WSL kernel
`6.18.33.2-microsoft-standard-WSL2` and fixed Python 3.12.3 were observed.
The child had **different user, PID, mount, network namespace identifiers**
from the outer observer, unlike the failed Kaggle CPU environment.
But the child **still saw Windows drive mounts**, and this root filesystem
was not made into an independent read-only, credential-free jail. The
non-root user had no independently delegated cgroup memory/PID enforcement;
network egress was not independently proven impossible; original Python
3.13 executable/semantics were absent. Source-backed actual result:

```json
{
  "status": "CAPABILITY_PROBE_ONLY_NOT_EXECUTION_APPROVAL",
  "new_user_pid_mount_network_namespaces_observed": true,
  "unresolved_gaps": [
    "WINDOWS_DRIVE_MOUNTS_NOT_HIDDEN",
    "ISOLATED_READ_ONLY_ROOTFS_UNPROVEN",
    "CGROUP_MEMORY_PID_DELEGATION_UNPROVEN",
    "KERNEL_ENFORCED_NO_EGRESS_UNPROVEN",
    "ORIGINAL_PYTHON_313_REQUIRED"
  ],
  "verified_complete_humaneval_execution_sandbox": false,
  "execution_approved": false,
  "gen0_status": "BLOCKED"
}
```

The new separate `reap_humaneval_wsl_benign_probe_20261026.py` runs a
**fixed source-hashed harmless child only**, independently inspects
host-visible descendant PID/start identities, never writes a private
generation, retains no private responses, changes no cgroup/mount/service,
and now emits nonzero CLI status even when the harmless probe itself
completes: another tool must NOT interpret program exit zero as authorization.
Adversarial synthetic tests verify this never promotes to a HumanEval PASS.

Independent v2 verifier review also confirmed the two actual external
capability/witness SHA approval pins are both **None**; the current known
rootless Podman runner uses incompatible Python 3.11/stdin and cannot be
made Python3.13 `candidate.py` equivalent by setting an execution flag.
New adversarial RED/GREEN repaired a potential *future*, source-resealed
trace-contradiction acceptance: unexpected trace fields, time elapsed vs
monotonic start/end mismatch, and reused process PID/starttime are refused.
This remains research-only source verification, **not** witness credibility
without an independently examined external OS observer and full raw evidence.

## Registered science not replaced or selectively reinterpreted

- Original full `random128_v2` 1,319 GSM8K +164 HumanEval full Kaggle kernel
  last reported `RUNNING`; no second job or partial-result substitution.
- Original full dense Q4 reference first-200 is not certified; past prehash
  abort and current scarce shared GPU/RAM prevent matching the existing
  original Windows Q4 200 at >=1.3x on current workstation.
- Genuine original REAP160 same-source quality/speed fallback remains
  unproven; no default speed number may be inferred from other serving flags.
- Original BF16/CUDA forward and original Q8/Q4 Windows/Linux logit/platform
  parity remain unproven; sampled weight numerical error does not substitute.
- Freeze the original §4 paired Wald lower bound for its originally specified
  gates and separately treat Amendment 4's optional nonzero-margin paired
  McNemar-style wording as unresolved. Qualified independent statistical
  prospective method approval must precede **new blinded confirmatory data**;
  no post-hoc favorable method or CI conjunction retroactively rescues the
  previously visible results. Mandatory Amendment4 fallback order stays
  REAP128_v2, REAP160, conditional REAP160_v2 after required original
  complete assessments; retain historical REAP128-v1 Gate1 **FAIL**.

**Reproducible next genuine execution:** provision a disposable dedicated
Linux VM/container-host with an independent host PID/cgroup/mount observer
outside the worker's namespaces; original-compatible Python3.13 `python -I
candidate.py` in per-task temporary cwd with 10-second Python subprocess
timeout, 2 GiB address-space limit, no Windows-mounted paths or credentials,
unprivileged jailed identity, no egress, immutable canonical input/rows,
enforced cgroup memory/PIDs and externally proven full descendant teardown.
Validate fixed benign negative canaries, exact raw observations and source
hashes **before** changing either approved v2 SHA pin or executing any
original model-generated code. For dense/BF16/full comparisons, separately
admit a truly large-memory CUDA host and full original per-arm sources.

**Gen-0: `{"status":"BLOCKED","chosen":null,"release_approved":false}`.**
PR #282 must remain OPEN/DRAFT/UNMERGED. No deployment, model release,
private response or model publication, Gen-1 training, scorer/threshold
change, unsafe rerun or external-service interruption was authorized.
