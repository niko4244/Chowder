# REAP128_v2: independent HumanEval replay fail-closed boundary and benign OCI capability

**October 26, 2026.** This is a public-safe SOURCE/SECURITY qualification, not a
new full-code replay, a Windows quality pass, or a selector approval. Frozen
Amendment 4 remains **BLOCKED / chosen null / release_approved false**.

## Severe source-integrity defect: demonstrated RED, corrected GREEN

The original optional external HumanEval auditor accepted six assertions
inside a supplied JSON: status, original row SHA, session, 164 claimed rows,
164 claimed matches and correct count. Its output then promoted the SHA of
**that self-asserted file** into a `PASS_FULL_164_LOW_PRIVILEGE_REEXECUTION`
marker. The downstream paired consumer accepted that marker by regex.
Neither step verified 164 actual program executions, trusted sandbox
isolation, an independently anchored producer or per-task results.

New CPU-only adversarial tests reproduced both false positives **RED** using
a fully synthetic six-field JSON with no executed programs. The production
auditor now rejects the optional self-asserted legacy replay as independent
execution proof, and the paired HumanEval consumer no longer accepts even
two well-formed historical `PASS_FULL_164...` markers as independent v2
verification. Both tests now **GREEN**. The original frozen scorer,
predictions, official prompts, decoder settings, model weights, full
GSM8K results and historical research receipts remain unchanged. Historical
Q4 Kaggle 164-item independent re-execution remains **provisional research**,
not a proven isolated/sandboxed v2 replay; original Q8 and random128 remain
unqualified for independently isolated HumanEval re-execution.

This intentionally fail-closed interval lasts until a **new** independently
anchored v2 replay proof can validate the actual per-task outcomes,
program/source hashes, sandbox image/runtime profile, process-tree teardown,
exact original interpreter policy and no-privilege/no-network containment.
An existing v1 hash or source-linked file name must not substitute for any
of these checks.

## Current original Q8 exact-source preflight

The privately pinned complete original Q8 machine, source audit (phase5),
canonical HumanEval 164 items, full original HumanEval 164 rows, original
evaluator/scorer and SHA-pinned harmless container probe independently
passed the source-only `check_source_binding` path with **zero generated
programs executed and no replay output written**.

The new private rootless-Podman research runner's CPU-only mocked adversarial
tests exposed and fixed two consequential hazards:

- An unsuccessful `podman kill` / `rm --force` previously returned
  `WALL_TIMEOUT_CONTAINER_KILLED` without verifying the actual container
  owner disappeared. It now checks `container exists` before executing
  (rejecting preexisting names) and after a timeout (rejecting surviving or
  unverifiable owners), and only uses bounded run-owned names.
- Original **unterminated** HumanEval rows were incorrectly passed through
  program formation/size checks even though production scored them as
  nonpasses without execution. They now retain a source-bound nonpass,
  no generated-program hash, and **no executable rendering**.

The runner's `EXECUTABLE_REPLAY_QUALIFIED=False` is a source-enforced hard
gate. Neither the `--execute-qualified` command-line flag nor a successful
source preflight permits generated code to run while full independent
containment, process-tree and interpreter verification remain unfinished.

## Actually observed benign 768-MiB Podman profile

A **fixed harmless** Python census (not a model response, and no model, GPU,
network calls or private mounted files) ran inside the installed rootless
`python:3.11-slim` OCI image, exact pinned image ID
`274d131132dbdc6058ed77088525d77adaf0749a6d99f2cf7d42feb9a10737c8`.
The real replay-profile census verified:

- Non-root UID/GID **65534/65534**, zero effective capabilities,
  seccomp mode **2**, `NoNewPrivs=1`, read-only root.
- Actual cgroup memory.max **805306368 bytes** (768 MiB), pids.max
  **16**, with no default network route and an empty `/run/secrets`.
- Exact `/run/podman-init` mount originating from the
  replay profile's `--init`; the prior generic mount guard had rejected
  even this intentional runtime init path. It now permits this **one exact
  target**, while continuing to reject every other `/run/podman*` target,
  /host and /mnt host bind prefixes. Adjacent tests cover forbidden variants.

An exclusive private, immutable-profile receipt from this benign census has
SHA-256 `fe94c504634db2d160c53b399a03d5f460473caca3db01d3669009dfc8f5c533`.
It explicitly reports `generated_programs_executed:0`,
`sandbox_qualified_for_generated_code:false`,
`full_child_process_tree_teardown_proven:false`, and
`python_original_3_13_interpreter_equivalence_proven:false`.

This establishes the **observed fixed-source cgroup/profile properties
only**. It is not an independent HumanEval result, a general security
guarantee or authorization to execute generated code. The installed image
is Python 3.11; the original Kaggle HumanEval scorer was Python 3.13.
Full descendant reaping and genuine interpreter parity remain unproven.

## Independent Windows Q4 benchmark and remaining gates

Previously source-certified genuine original Windows Q4 first-200,
session `e3f9c268b2bb4d958cc55648c0c4c8e0`: **187/200**,
**73,224 generated tokens**, **398 s** rounded scoring wall time
(approximately **184.0 generated tokens/s**; not a dense comparison);
**148 watchdog samples**, min available physical/commit **10.064/14.724
GiB**, measured full GPU offload and same-session production speed
certificate. The candidate-only original evidence is intact.
The historical first-200 wrapper failure is preserved separately and
unqualified. The native dense first-200 comparator is still absent, so
the **1.3× ratio remains UNKNOWN**.

The independent original random128 full model experiment was still RUNNING
at the latest check; do not duplicate it or invent a result. Original v1
Gate1 FAIL, original REAP160 registered speed UNKNOWN, BF16/CUDA forward
parity, exact dense speed/reference and complete mandatory noninferiority
remained unresolved. Do not merge, deploy, release, select Gen-0 or train Gen-1.

**Validation at publication:** 301 adjacent CPU-only tests passed with 1
guarded skip; Ruff and `git diff --check` passed. No generated code was
executed in the new rootless OCI replay; no model file or private generation
was added to Git.
