# Compute providers: opportunistic external compute under the research layer

Status: **the scheduler seam, three providers, successive-halving screening and
the evidence rules are implemented and tested.** Kaggle kernel-push, session
polling, output fetch and quota reconciliation run against the real Kaggle API
package (stub-injected in tests; no paid GPU-hours have been spent by this
code). RunPod runs against REST v2 through an injectable transport (same
honesty: stub-tested, no real pod launched). The ResearchDirector does not
care where an experiment runs; the scheduler decides, from a provider's
declared availability and quota model.

## 1. What the audit found (why this design, not another)

Chowder is not starting from zero on external compute:

| Existing capability | Where | State |
|---|---|---|
| Real 2×T4 DDP training/eval on Kaggle-class hardware | `docs/DDP_ACCEPTANCE.md`, `tests/test_ddp_acceptance.py` | **proven** (gated real-hardware tests, `accelerate launch --multi_gpu`, world_size 2 verified, cancellation tears down every rank) |
| Kaggle evaluation notebooks | `kaggle/` directory (bootstrap/acquire/evaluate/equivalence scripts) | **proven pattern**: pinned-commit install, secret hygiene (HF token via Kaggle Secrets, never logged), refuses to run outside a real kernel |
| Environment fingerprinting | `kaggle_launcher.capture_environment_fingerprint` → `BackendFingerprint` | production, imported unchanged here |
| Backend equivalence + qualification | `kaggle_equivalence.py` (`EquivalenceReport`, `qualify_backend` — never relaxes the item-score bar; digest divergence must be *declared*) | library, tested |
| VRAM preflight with measured reference | `kaggle_launcher.preflight_vram_headroom` (T4 usable ≈ 14.8 GiB; measured parent-A peak 16 004 MiB) | production |

What does **not** exist, and what this pass adds:

- A **provider seam** the research layer can route through: the ResearchDirector
  compiled experiments to specs (Phase 3), but nothing chose *where* they run.
- A **screening lane**: the tree's branch competition names successive halving
  as its designated allocator, but there was no cheap first-stage lane.
- **Hardware-context discipline for efficiency claims** (§3 below).

## 2. The seam

```
ResearchDirector / ResearchTree
        │  compiled experiments (Phase 3 specs)
        ▼
ExperimentScheduler                     (scientist/compute.py)
   ComputeProvider Protocol:
     name, available(), quota(), estimate_cost(request),
     submit(request), poll(submission), fetch_result(submission)
        │
   ┌────┼──────────────────┐
   ▼    ▼                  ▼
LocalCudaProvider  KaggleProvider   RunPodProvider   (later: Vast, SSH,
 (the machine       (T4×2 kernels,    (REST v2 pods,    Lambda)
  Chowder runs on)   push/poll/       injectable
                      quota-reconcile)  transport)
        │
        ▼
Chowder Evidence Layer (RunRegistry)  — unchanged, all runs land there
```

Rules (all fail-closed):

1. **No default provider.** The scheduler is constructed with an explicit
   provider list; an empty list refuses (`NO_PROVIDER_AVAILABLE`).
2. **Order = preference.** Providers are tried in declared order; the first
   `available()` provider whose quota admits the request wins. `LocalCudaProvider`
   is usually first for substantial runs, `KaggleProvider` first for screening.
3. **`require_provider` pins the choice.** A request may name a provider
   explicitly (replication-on-different-hardware requests do this); a request
   that names a provider the scheduler does not know is refused, not rerouted.
4. **LocalGPUHours are not free elsewhere.** A mission budget spends the same
   ledger regardless of provider; `KaggleProvider` reports cost in GPU-hours
   honestly (T4-hours on 2×T4 = 2 device-hours per wall hour, matching the
   growth accounting's `accelerator_seconds = wall × active_accelerator_count`).
5. **No sidecar trust**: a provider returns typed submissions/results; it never
   touches policy, the protected set, or thresholds. Same threat model as the
   scientist providers (docs/SCIENTIST_THREAT_MODEL.md).

## 3. Capability results vs efficiency results (the hardware-context rule)

The mission's rule, enforced in the evidence types: **quality findings may be
compared across hardware when the protocol is identical; efficiency findings
never leave their hardware context.**

- `ExperimentObservation.measurements` already carry `surface`/`benchmark`;
  they now may carry `hardware_class` (default `None` = local control). The
  scheduler stamps every observation it records with the provider's
  `hardware_class` (e.g. `kaggle_2x_t4_16gb`, `local_rtx_5060ti_16gb`).
- `Claim` gains `hardware_dependent: bool`. A claim about quality with
  identical protocol is `hardware_dependent: False` and may cite runs from
  different hardware classes; a claim about tokens/sec, VRAM, wall time or
  time-to-quality **must** be `hardware_dependent: True`, and the memory
  layer refuses a `replicated` status for such a claim unless every cited run
  shares one hardware class (a throughput number is a number about that
  hardware, not about the model).
- The tree scores transferability on **quality** deltas only; an efficiency
  delta from a different hardware class never feeds `capability_delta`.

## 4. Scheduling policy (screening → survivors → substantial → replication)

The scheduler implements the screening shape from the design discussion:

```
8 hypotheses → Kaggle screening (short/cheap runs, kill weak branches)
            → survivors → substantial runs (usually local, provider order)
            → promising → replication (deliberately cross-hardware where
                           quality is the claim; same-hardware where
                           efficiency is the claim)
```

- `SchedulingRequest.experiment_class`: `exploratory | screening | substantial | replication`.
  Screening requests get the screening provider (a `KaggleProvider` declared
  as `screening: true`); replication requests get `require_provider` set by the
  director when the claim being replicated is efficiency-sensitive.
- The director's `submit_screening_batch` / `submit_survivor_batch` helpers
  wrap this: screening runs are capped by the provider's own quota model
  (see below), and the tree's branch competition consumes the results.
- **Successive halving is wired as the screening lane's allocator**
  (`scientist/screening_halving.py` + `ResearchDirector.run_screening_halving`):
  round r submits every surviving candidate at `min(budget_cap,
  initial × step_multiplier^r)` GPU-hours, the loop records the round's
  observations through the `record_results` seam, and settlement is
  mechanical — tree-scored candidates compete (score descending,
  experiment-id-ascending tiebreak), candidates with no usable observation
  are eliminated **by gate**, the rest **by cutoff**, and only the FINAL
  round's survivors graduate to `submit_survivor_batch`. The schedule
  semantics mirror the growth library's `run_successive_halving` exactly
  (survivor counting, stop rules); what differs is the execution substrate:
  the library runs against the local ExperimentCycleRunner with checkpoint
  resume, the screening lane against the scheduler and its providers. The
  driver refuses **before any compute** when no `record_results` seam is
  provided — rounds advance only on recorded observations — and journals
  every elimination to the refusals ledger.

## 5. KaggleProvider's honest quota model

Kaggle notebooks: up to 12 h sessions, ~20 GB `/kaggle/working`, T4×2 (two
16 GB devices) with weekly GPU quota that varies with demand. The provider:

- is `available()` only when the operator configured a Kaggle username/key
  (Kaggle API v2.2.3 is already a working tool on this machine) — or when a
  fake/recorded provider is explicitly injected in tests;
- models **quota as a declining weekly budget** the operator declares
  (`weekly_gpu_hours`): `quota()` reports remaining hours; `submit` refuses
  (`QUOTA_EXHAUSTED`) rather than promising capacity Kaggle may not grant;
- records `hardware_class = kaggle_2x_t4_16gb`, device-hours honestly
  (wall × 2), and carries the `BackendFingerprint` discipline (environment
  captured per run) forward — a Kaggle result without an environment
  fingerprint is refused at observation-recording time;
- is **opportunistic, never load-bearing**: the scheduler falls back along the
  declared order when Kaggle is unavailable, and the mission budget is charged
  the same either way.

### 5b. The real push path (proven against the API contract, stub-tested)

With `push=True` (the default), `submit` is a real `KaggleApi.kernels_push`
call against the installed kaggle package (2.2.3, kagglesdk-based):

- **Three things must be true or nothing ships**: credentials (explicit,
  `KAGGLE_USERNAME`/`KAGGLE_KEY`, or a `kaggle.json` — resolved *without*
  importing kaggle, because the package's `authenticate()` calls `exit(1)`
  when unauthenticated), a pinned **40-hex** chowder commit
  (`KAGGLE_KERNEL_PIN_REQUIRED` otherwise — never a branch, mirroring
  `kaggle/bootstrap_environment.py`), and an operator-supplied
  `kernel_command` (`KAGGLE_EXECUTOR_NOT_CONFIGURED` otherwise — the
  compiled `campaign_spec` is a declaration and Chowder ships no default
  remote executor that would silently guess what to run).
- The kernel folder gets `kernel-metadata.json` (script, private, GPU,
  `machine_shape: NvidiaTeslaT4` — the T4×2 shape since the P100 retirement)
  and a generated `kernel.py` that: installs chowder at the pinned commit,
  cross-checks the resolved commit from `direct_url.json`, captures the
  `BackendFingerprint` into `/kaggle/working/environment.json`, persists the
  campaign spec, runs the operator command, and writes
  `/kaggle/working/chowder_result.json` (the command's own result file wins;
  the template only fills status/exit-code). HF tokens stay in Kaggle
  Secrets and are the operator command's concern, as in the repo's existing
  notebook scripts.
- `poll` maps the real `KernelWorkerStatus` (QUEUED/RUNNING/COMPLETE/ERROR
  via `kernels_status`); on COMPLETE it fetches output (`kernels_output`) —
  `chowder_result.json` becomes the submission's result, `environment.json`
  its fingerprint; on ERROR/CANCEL the failure message travels with the
  submission. A `push=False` submission has no `provider_ref` and `poll`
  leaves it exactly as it was (spec-only mode is for offline scheduling
  tests; production constructs with push).
- `sync_quota_from_api()` reconciles the declared weekly budget with the
  operator's **real** Kaggle GPU quota (`quota_view` → `ApiAcceleratorQuota`):
  reserved time counts against availability like used time; a missing/zero
  response keeps the declared model untouched. Refusals (`KAGGLE_PUSH_REFUSED`,
  `KAGGLE_API_ERROR`, `KAGGLE_PACKAGE_MISSING`) never charge quota.

## 6. What is proven vs scaffolded (honest)

- **Proven (unit-tested, no network/GPU)**: provider protocol, scheduler
  preference/fallback/pinning rules across three providers, quota refusal,
  capability-vs-efficiency claim gating, hardware-class stamping,
  successive-halving screening (schedule arithmetic, deterministic gate/cutoff
  settlement, journaling, graduation), screening/survivor batching, the
  two-ledger integrity (mission ledger charged identically regardless of
  provider), LocalCudaProvider availability detection from the real
  `HardwareSnapshot`.
- **Proven against the real API contract, stub-tested**: the Kaggle path
  (`kernels_push`/`kernels_status`/`kernels_output`/`quota_view` — push
  folder + metadata construction, status mapping, output collection, quota
  reconciliation, refusal ladder) and the RunPod path (REST v2
  `POST /v2/pods`, `GET /v2/pods/{id}`, `DELETE`, through an injectable
  transport). Tests inject a stub client/transport; no live call is made in
  CI. The first real push/pod should be done once, observed, and recorded in
  DDP_ACCEPTANCE-style notes before the lane is trusted.
- **Scaffolded**: a RunPod pod's EXITED status never reports `complete`
  without a `result_fetcher` confirming an artifact
  (`RUNPOD_EXIT_UNVERIFIED`); fetching pod artifacts out-of-band (volume/S3)
  is the operator's seam. Kaggle-side, everything up to the fetched output
  is real; enriching `chowder_result.json` beyond exit status is the
  operator command's job.
- **Not started**: Vast/Lambda/SSH providers (the protocol accepts them;
  nothing pretends they exist).

## 7. RunPodProvider (REST v2)

- **Endpoint discipline**: `POST /v2/pods` with `name`, `image`,
  `gpu: {id, count}`, `env` (campaign spec + ids + operator command as JSON),
  `cloud`/`disk`; poll `GET /v2/pods/{id}` → `PodStatus`; `terminate` →
  `DELETE` (a dead or refused run must not keep billing). REST v1 was
  deprecated upstream and is not used.
- **Refuses to create an unrunnable pod**: missing `RUNPOD_API_KEY`,
  `gpu_type_id` (from `GET /v2/catalog/gpus`), `image`, or `command` are
  named in the refusal. Creating a pod is a billing event; the declared
  weekly GPU-hour quota gates every submit.
- **No fake success**: PROVISIONING/STARTING/RUNNING map to `running`;
  ERROR/TERMINATED to `failed`; EXITED only completes through a confirmed
  artifact (§6). The transport is injectable — tests never touch the
  network, and the default is stdlib `urllib` (no new dependency).
