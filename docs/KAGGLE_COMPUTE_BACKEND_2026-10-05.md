# Kaggle as a first-class compute backend (2026-10-05)

Status: **implemented and production-wired** — the backend contract, its
Kaggle implementation, the production CLI transport, the campaign wiring and
the declared-input upload, verified offline. What remains is the kernel-side
payload and a manifest declaration (see Remaining).

## Why

`docs/KAGGLE_BACKEND_REQUIREMENTS.md` states the bar a 0.5 campaign must
clear before it may spend Kaggle quota on candidate training: a dispatched
job's identity, inputs, outputs, cost and failure must all be provable after
the fact. The open dispatcher (PR #200) pushes a *script*; a campaign needs to
dispatch a *scientific attempt*. This change adds that layer as a first-class
`ComputeBackend`, behind the declared-input contract the campaign already
uses (`campaign_prepare.PREPARED_INPUT_FIELDS`), so the Kaggle path is held
to the same guarantees as the local subprocess path.

## What landed

| module | role |
| --- | --- |
| `src/chowder/growth/compute_backend.py` | the shared contract: `SourceBinding`, `DeclaredInput` / `bind_declared_inputs` / `verify_declared_inputs`, `ArtifactEntry` / `verify_artifact_manifest`, `AttemptRequest`, `AttemptOutcome` (+ `to_evidence()`), the `ComputeBackend` protocol |
| `src/chowder/growth/kaggle_compute.py` | `KaggleComputeBackend`, `KaggleTransport` (injectable API seam), `KaggleJobSpec` / `KaggleJobRecord`, `KaggleQuota`, declared accelerator shapes |
| `src/chowder/growth/kaggle_cli_transport.py` | the production transport: stage, push, poll, pull and quota over the official `kaggle` CLI, every call through an injectable runner |
| `src/chowder/growth/kaggle_kernel.py` | the kernel-side half, imported after the generated entry script installs the pinned commit: commit echo, input re-hash across the declared candidate mount paths, command run, artifact manifest, environment, record writing |
| `src/chowder/growth/kaggle_inputs.py` | the declared-input upload: stage the bound inputs flat, publish a content-addressed dataset (`create`, else `version`), return the reference and every candidate kernel path |
| `src/chowder/growth/kaggle_campaign.py` | the campaign wiring: `build_attempt_request` from a `PreparedCampaign`, and `KaggleTrainingFn` — a `TrainingFn` with `admit` that dispatches through a `ComputeBackend` and writes `training-evidence.json` |
| `tests/test_growth_kaggle_compute_backend.py` | 37 offline tests; every production-bar check has a test that makes it fail |
| `tests/test_growth_kaggle_cli_transport.py` | 14 offline tests over a scripted fake CLI, ending in a full dispatch through `KaggleComputeBackend` |
| `tests/test_growth_kaggle_kernel.py` | 12 offline tests of the kernel-side runner |
| `tests/test_growth_kaggle_inputs.py` | 12 offline tests of the publisher over a scripted `kaggle datasets` CLI |
| `tests/test_growth_kaggle_campaign.py` | 17 offline tests: the builder, admission, dispatch, evidence, never-reused attempts, resume, source honesty, and the publisher-to-kernel path integration |

### The declared-input contract

A campaign binds the paths its manifest declares — the same names
`campaign_prepare` produces — with `bind_declared_inputs({"training_material":
path, ...})`. Each entry carries a sha256 and a byte count; the backend
re-hashes every declared input before push (`verify_declared_inputs`) and
refuses the whole attempt if a file moved. The kernel must echo every
declared input with the same digest; the backend compares the echo exactly,
so an input the kernel did not verify, or one the campaign never declared,
refuses before any artifact is trusted. A declaration may name several
candidate kernel paths per input (Kaggle mounts under the bare slug or the
owner-qualified path), and the input is the candidate whose bytes match the
declared digest -- never merely the first path that exists.

### The production bar, as code

| bar | enforced by |
| --- | --- |
| R1 source binding | spec carries repo/commit/version/attempt ids; the record's echoed commit must equal the declared one |
| R2 manifest + admission | inputs verified pre-push and echoed; outputs verified against the kernel manifest (missing / unlisted / hash-mismatched all refuse), then an injectable `artifact_admission` check runs, so a downloaded artifact goes through the same gate as a local one |
| R3 settlement | quota reading before/after × declared accelerator device multiplier, wall from the job record, settled through `compute_cost.settle_cost` against the same projection/ceilings; refusals classify BUDGET_EXHAUSTED |
| R4 resume | declared `resume_from` must be echoed; `resume_state` must be `resumed` or `not-a-resume`; `not-a-resume` fails the attempt and is reported, never rewritten as a fresh start; an undeclared resume refuses |
| R5 classified failures | a job that did not complete goes through `attempt_failure.classify_failure`; its class, not its log line, is what the attempt records, and its quota still settles. The completed-job provenance checks apply only to completed records, so an install failure surfaces as the install error itself rather than as a derived mismatch |
| R6 recorded environment | `python_version`, `packages`, `model_commit` (extensible list) must ship in the job record and travel into the evidence |
| R7 one kernel, one attempt | fresh destination per dispatch, job ids never reused, mounts recorded and compared |
| R8 quota ceiling | quota read before push; the projection must fit both the remaining balance and the campaign's declared Kaggle ceiling, accumulated across dispatches |

`AttemptOutcome.to_evidence()` renders the same keys the growth loop reads
(`status`, `candidate_succeeded`, `refused_by`, `budget_settlement`,
`resume_state`, `artifact_ref`/`artifact_sha256`, `compute_cost`), so
selection and failure classification need no remote-specific branch.

## The production transport and the kernel entry point

`src/chowder/growth/kaggle_cli_transport.py` implements the transport over
the official `kaggle` CLI: it stages the kernel folder (generated entry
script, job document, `kernel-metadata.json` with the declared mounts and
machine shape), pushes with the declared timeout, polls `kaggle kernels
status` to a terminal state under timeout + grace, pulls the output, reads
the kernel's `job-record.json`, and copies the pulled artifacts into the
attempt destination. `kaggle quota` is parsed into remaining GPU hours;
an unparseable balance refuses rather than guessing. Every CLI call goes
through an injectable runner (`clock`/`sleep` too), so the whole transport is
tested offline. Operational failures raise `KaggleTransportError`, which the
backend records as an INFRASTRUCTURE attempt failure with the cause attached
— never an unhandled exception, never a fabricated record.

`src/chowder/growth/kaggle_kernel.py` is the kernel-side half. The generated
entry script installs the pinned commit (`pip install` of the URL the job
document carries) and then imports this installed, reviewed package code
rather than running a generated blob: it reads back the installed commit
from `direct_url.json`, re-hashes every declared input at its declared
kernel path, runs the payload command, collects the artifact manifest
(excluding the record and bytecode caches), records the kernel's own
environment, resolves the resume vocabulary (a marker file may say
`resumed`; otherwise the honest answer is `not-a-resume`), and writes
`job-record.json` even on failure. It runs on a laptop, so all of it is
unit-tested without a GPU.

Two failure shapes are deliberately distinct. A job that ran and failed
writes its full record from the kernel-side runner, which the backend
classifies. A job that failed *before* the installed package could be
imported — the pinned commit did not install, or the installed package
cannot be imported — writes a minimal record from the generated entry
script: the declared identity (`spec_id`, `mounts`, `resume_from` with the
honest `not-a-resume`) comes from the job document, the commit echo and
input verification stay empty because nothing ran, and the failure text is
the record's error. The backend classifies that record from its own error
and settles the quota its session burned. A record that does echo other
code still refuses as a source mismatch; an unidentifiable source is the
reported failure, never a fabricated commit echo or a derived refusal.

The platform log and the record are pulled into a sibling
`<destination>-kaggle-output` directory for the audit; only the kernel's
artifacts reach the attempt destination, so the manifest check sees exactly
what the kernel claims to have emitted.

## Campaign wiring: a prepared manifest to a dispatched attempt

`src/chowder/growth/kaggle_campaign.py` closes the loop between
`campaign_prepare` and the backend:

- `build_attempt_request(prepared, ...)` binds **every** declared prepared
  input (`PREPARED_INPUT_FIELDS` plus the contamination manifest) by digest;
  a prepared set missing one of them, or carrying an unknown one, refuses. It
  requires the declared command and at least one mount (the surface the
  inputs arrive on), checks that the declared kernel paths cover the declared
  inputs exactly, fills the projection from the recipe's own projections and
  the ceilings from the growth envelope -- the same envelope the local
  executor admits against -- and carries the search's `resume_from` when the
  recipe declares a continuation.
- `KaggleTrainingFn` is a `TrainingFn` (`__call__` plus `admit`) that the
  campaign runner accepts exactly where it accepts the local executor:
  `run_campaign(manifest, train_fn=KaggleTrainingFn(...))`. Every attempt
  reserves the smallest unused `attempt-NN` directory (never reused, so a
  refused or failed attempt keeps its evidence), rebuilds the request
  (re-binding the inputs, so a file that moved refuses *that* attempt),
  dispatches through the injected `ComputeBackend`, and writes
  `training-evidence.json` beside the result. The evidence is
  `AttemptOutcome.to_evidence()` merged into the vocabulary the cycle and
  runner already read, plus `actual_cost` (the measured device-hours the
  runner's ceiling settlement needs), the declared inputs, and a
  `source_identity` block computed from the kernel's *echoed* commit -- never
  inferred from a status, so an install failure that identified no source
  cannot be read as a verified one.
- `admit` delegates to `training_binding.check_growth_envelope`, the
  admission rule extracted from the local executor so both executors admit
  exactly the same recipes.

## Input delivery: the declared bytes on a Kaggle mount

`src/chowder/growth/kaggle_inputs.py` uploads the declared set as a Kaggle
dataset. `KaggleInputPublisher.publish(inputs, dataset=..., title=...)`
stages one flat file per declared name (a basename collision refuses rather
than renaming), writes `dataset-metadata.json` (title, `owner/slug`, and one
declared license -- default `unknown`, because the publisher will not stamp a
license the campaign never chose), runs `kaggle datasets create` and falls
back to `kaggle datasets version` when the dataset already exists, attaching
both failures when neither works. The slug is content-addressed
(`<dataset>-<digest12>`), so the same declaration always names the same
dataset and a changed input set can never silently reuse the previous one.
The result carries the reference (for `spec.mounts`) and, per declared input,
the kernel paths it may appear at: both the bare-slug and the owner-qualified
mount, so the kernel resolves the bytes by digest rather than assuming one
path. The publisher does not parse `datasets files` / `datasets status`
output; the byte-level guarantee is the kernel's re-hash compared by the
backend, and a listing parser would be a weaker claim in the same vocabulary.

## Verification

- `tests/test_growth_kaggle_compute_backend.py`: **37 passed** — contract
  primitives (commit validation, input binding/verification, manifest
  missing/extra/hash/escape, unserializable payload) plus every R1–R8 refusal,
  the transport-failure paths (quota unreadable, run raised), the
  failed-record route (no echoed commit and no verified input, yet classified
  from its own error with its spend settled; a failed record that echoes
  other code still refuses as a source mismatch), the echoed-commit evidence
  (present on success, `None` when no record identified one), and the success
  path including the settlement verdict readable by `settlement_refusal`.
- `tests/test_growth_kaggle_cli_transport.py`: **14 passed** — quota/status
  parsing forms, slug and shape rules, staged identity, the generated entry
  script's install-failure record (identity carried, commit echo honestly
  empty, install error preserved), push/poll/pull orchestration,
  timeout+grace, transport-failure classification, an install failure
  surfaced end to end through the backend, and an end-to-end
  `KaggleComputeBackend` dispatch through the real transport over a scripted
  fake CLI.
- `tests/test_growth_kaggle_kernel.py`: **12 passed** — install spec, commit
  echo, input verification (including digest resolution across conflicting
  mount candidates), command status, artifact manifest, resume vocabulary,
  bytecode exclusion.
- `tests/test_growth_kaggle_inputs.py`: **12 passed** — content addressing,
  flat staging and metadata, create-then-version, basename collisions, title
  and slug rules, moved bytes (before and during the copy), a missing CLI,
  and both mount candidates.
- `tests/test_growth_kaggle_campaign.py`: **17 passed** — the builder's
  declared-input/command/mount/path refusals, the projection and ceilings,
  admission shared with the local executor, an end-to-end dispatch through
  the real backend, never-reused attempt directories, resume passthrough,
  source-honesty on install and transport failures, a builder refusal
  recorded as evidence, and the publisher-to-kernel path integration.
- `ruff check src tests`: clean. The new Kaggle tests also pass under the
  light-CI import block (`-p no_heavy`, torch/transformers refused): **92
  passed**, so the dev-only CI jobs collect them.
- The full suite caught `tests/test_worker_env.py`'s launch guard on the new
  transport module: its only `sys.executable` reference is inside the entry
  script it generates for the *remote* kernel, and the `kaggle` CLI launch is
  not a Chowder worker, so the module joined the documented non-worker
  exemptions rather than passing a `worker_env` that would claim otherwise.
- Full suite on the frozen tree: **3095 passed, 77 skipped, 0 failed** in
  639.47s (`FULL_EXIT=0`).

## Remaining

- **The kernel-side payload**: the declared command now travels to the
  kernel with the declared inputs and the mount paths, but what runs *inside*
  the kernel -- the production entry points against the mounted inputs,
  curriculum-corpus materialization, evaluation -- is the campaign's payload
  declaration; this wiring will not invent it. Porting the local executor's
  corpus writing into a declared kernel-side command is the next concrete
  step.
- **Declaring the executor in the manifest**: `run_campaign` already accepts
  an injected `train_fn`, so a Kaggle run is wired today. Making the campaign
  manifest itself declare the Kaggle backend (with the mounts, timeout and
  dataset title it needs) is a deliberate schema change, not smuggled in here.
- **Upload availability**: `create`/`version` returning success is not the
  same as the dataset being ready to mount. The kernel's declared-input
  verification fails such an attempt visibly instead of masking it, and the
  publisher does not poll a status format it cannot pin.
