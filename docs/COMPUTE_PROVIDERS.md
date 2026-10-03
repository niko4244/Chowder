# Compute providers: opportunistic external compute under the research layer

Status: **the scheduler seam, the two providers and the evidence rules are
implemented and tested; no real Kaggle session has been launched by this
code.** The ResearchDirector does not care where an experiment runs; the
scheduler decides, from a provider's declared availability and quota model.

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
   ┌────┴─────────────┐
   ▼                  ▼
LocalCudaProvider   KaggleProvider          (more later: RunPod, Vast, SSH)
 (the machine        (T4×2 notebooks,
  Chowder runs on)    12 h sessions, weekly
                      GPU quota, opportunistic)
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
- **Successive halving stays the designated allocator** for the in-branch
  parameter/recipe competition (docs/SCIENTIST_MODE.md §6, unchanged): the
  screening lane *finds* the branches worth spending; successive halving (once
  wired) would decide how they spend. This pass does not wire it.

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

## 6. What is proven vs scaffolded (honest)

- **Proven (unit-tested, no network/GPU)**: provider protocol, scheduler
  preference/fallback/pinning rules, quota refusal, capability-vs-efficiency
  claim gating, hardware-class stamping, screening/survivor batching, the
  two-ledger integrity (mission ledger charged identically regardless of
  provider), LocalCudaProvider availability detection from the real
  `HardwareSnapshot`.
- **Scaffolded**: the actual Kaggle submission path. `KaggleProvider.submit`
  constructs the submission (job spec + notebook-facing command lines using
  the existing `kaggle/` script conventions) and returns it as `queued`;
  the kernel-push/session management is **not implemented** — it requires the
  Kaggle API plumbing this pass deliberately leaves out. Until then,
  `KaggleProvider.available()` on this machine reports the operator
  configuration state honestly, and tests use an injected fake.
- **Not started**: RunPod/Vast/Lambda/SSH providers (the protocol accepts
  them; nothing pretends they exist).
