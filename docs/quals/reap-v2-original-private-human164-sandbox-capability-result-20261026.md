# Original HumanEval private CPU sandbox capability probe — 2026-10-26

**Result: `REFUSED_SANDBOX_UNAVAILABLE` for arbitrary generated-program execution.**
This is a distinct, source-only, harmless CPU capability experiment, NOT the
original model inference benchmark, NOT original HumanEval scoring and NOT
a release or original Windows gate attestation. Original Gen-0 remains
`BLOCKED`, `chosen:null`, `release_approved:false`.

The original Q8 full native evaluation **completed**, with all 1,319
GSM8K and 164 original-scored HumanEval rows recovered and independently
audited. Original Q8 HumanEval summary **152/164** awaits a genuinely
isolated independent 164-program replay. The non-executing Q8 source-bound
HumanEval admission has already passed: exclusive private receipt SHA
`f66fb7bd205d9ed43c072c3524c891905d50e5138c29f718d7190312d815fbfc`.

## Harmless Kaggle CPU-only isolation census

Separate source: `scripts/reap_v2_humaneval_sandbox_probe_20261026/sandbox_capability_probe.py`,
SHA `a78fb69d06ac15ffba481582f07eb1d2855807c66b53aabd05c1f383df678942`.
Private kernel `nikmarco/reap128v2-humaneval-sandbox-capabilities-20261026`,
version 1, **COMPLETE**. Metadata: `is_private:true`,
`enable_internet:false`, `enable_gpu:false`, **zero model, dataset,
competition or kernel sources**. No original weights, generated programs,
hidden prompts, scored rows or keys were transferred to this probe.

The notebook executed **only a fixed, locally hashed benign Python
namespace-inspection snippet**, not arbitrary model-generated code. Its
child string SHA `fdacc215f9aae207a476f4229b7e92414bc332d70d58b0f9f721262b2b97ea8d`.
It made no network request, wrote no cgroup, and never loaded a model.
The exact private returned native JSON was retrieved to:

```text
F:\reap-v2-original-human164-sandbox-capability-evidence-20261026-phase4\reap-v2-human164-sandbox-capability-native.json
```

File SHA-256:
`66c47b94f23c05a739242357bfa4f4628c51ec9f1ebe75e5eb4597e3b89e0991`.
Its actual evidence reports:

- Actual notebook effective UID **0**.
- Fixed harmless `unshare` user/PID/mount/network namespace attempt:
  **UNAVAILABLE**; no successful isolated launch.
- Bubblewrap: no available demonstrated launcher.
- `available_namespaced_launchers: []`.
- `new_user_pid_mount_network_namespaces_proven:false`.
- `cgroup_delegation_proven:false`.
- `verified_complete_humaneval_execution_sandbox:false`.
- `no_generated_python_executed:true`.
- `status:CAPABILITY_PROBE_ONLY_NOT_AN_EXECUTION_APPROVAL`.

**Interpretation:** Kaggle `enable_internet:false` and a root runner
with `python -I` or `setrlimit` do not alone prove kernel-enforced
network isolation, minimal read-only mounts, a separate PID namespace,
whole-descendant termination, or delegated cgroup memory/PID caps. The
harmless fixed-probe results cannot authorize execution of arbitrary
candidate Python. Do **not** weaken the frozen HumanEval scorer or silently
promote its reported pass booleans to independently verified outcomes.

## Reproducible next safe HumanEval execution environment

Provision a **separate authorized private Linux runner with a verified
rootless/privileged sandbox facility**, such as configured Bubblewrap or
rootless OCI, that actually proves disjoint user/PID/mount/network
namespaces and no egress, read-only canonical source/tests, a private
per-item scratch mount, no-new-privileges, independently verified
nonroot host credentials, strict cgroup-v2 `memory.max`/`pids.max`
and effective `cgroup.kill` or namespace-init teardown.
Probe each property first with harmless code and record a source/runtime
receipt; otherwise refuse.

Only then execute **terminated** Q8 and independent random128 candidates
in the exact original source-bound 164-item order with CPU ~10s, wall
12s, 2-GiB address-space and 1-MiB output limits, no core dump, bounded
descriptors and descendants; compare all 164 correctness decisions to
the original immutable session/row hashes and independent first audit.
Use a different new private replay receipt per original model arm. Never
reuse the original Q4 replay SHA, session or 151-score.

Source-level adversarial tests for the fixed capability probe passed
**4/4 CPU-only**, including exact full notebook-source SHA and exact
fixed harmless-child SHA checks; this tests source identity and declared
constraints, not actual isolation. The full adjacent original scientific
matrix passed **606**, one Windows symlink test skipped, one intended
deselection and five warnings. No model release, PR merge, production
deployment or Gen-1.
