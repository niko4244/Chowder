# Kaggle backend: the production bar

Status: **requirements** (Phase 12 of the 0.5 architecture preparation).
This document states what `feat/kaggle-dispatch` (PR #200, all CI green)
already provides and what it must add before a 0.5 campaign is allowed to
spend Kaggle quota on candidate training. It is a contract for the backend,
not a review of the PR: the dispatch plumbing that exists is sound, and the
bar below turns a working smoke path into a backend a scientist can trust.

## What PR #200 already provides

`src/chowder/kaggle_dispatch.py` (with `tests/test_kaggle_dispatch.py`):

- push / poll / pull through the official `kaggle` CLI, every call through
  an injectable runner (fully testable, no network, no token);
- a hard `timeout_seconds` on every job (Kaggle's 12 h session cap), because
  every second a kernel runs is charged to the weekly quota;
- kernel state machine handled: `complete` is the only success, every other
  terminal state is a failure **with its log pulled** before the error is
  raised;
- `job_record.json` persisted beside the output — wall seconds, accelerator,
  push output, error;
- `quota` readout of the weekly GPU balance;
- slug validation, private-by-default kernels, explicit source mounts.

That is the right skeleton. The gap is that it dispatches *a script*, and a
0.5 campaign needs to dispatch *a scientific attempt* — one whose identity,
inputs, outputs, cost and failure are provable after the fact.

## The production bar

### R1 — Job identity is bound to source, and the binding is verified on both ends

A job is not identified by its kernel slug. Every `KaggleJobSpec` must carry
a `source_binding`: repository URL, exact commit SHA, campaign id, recipe
id, attempt id, and the chowder package version. The kernel-side bootstrap
(`kaggle/bootstrap_environment.py` already installs at a pinned commit)
must verify after install that the checked-out SHA equals the declared one
and echo both into the output; after download, the dispatcher refuses a job
whose returned SHA differs from the spec's. A kernel that ran different
code than the campaign declared has produced no evidence.

**Status in #200:** absent. `KaggleJobSpec` carries no source binding; the
smoke kernel installs the repo but nothing compares SHAs end-to-end.

### R2 — Artifacts are hash-verified before they exist

`pull_output` lists files; it does not prove them. The kernel must write a
manifest (`path, sha256, bytes`) for everything it emits; after download
the dispatcher verifies every manifest entry and refuses missing, extra, or
hash-mismatched files. Only a verified artifact may be referenced by an
attempt's `artifact_ref` and enter the ledger. The verification must reuse
the same admission checks a local artifact passes (checkpoint manifest
binding, adapter key guard) — a downloaded artifact is not a second-class
citizen with a second-class gate.

**Status in #200:** absent. No manifest, no hashes.

### R3 — Remote spend settles through the same accounting as local spend

A remote run must be priced like any other attempt: record the quota
reading before and after the job (`kaggle quota` already exists), the
accelerator shape, and the wall seconds; convert to device-GPU-hours with
the declared shape's multiplier; settle against the recipe's projection
through `growth.compute_cost.settle_cost`. The existing settlement-refusal
semantics (ACTUAL_EXCEEDS_PROJECTION) must apply unchanged — the Phase 0
audit already found projection laundering at the fixture level, and a
remote backend with its own accounting would multiply it. Settlement
refusals on a remote attempt feed `failure_taxonomy` as BUDGET_EXHAUSTED,
never silently re-queued.

**Status in #200:** partial. Wall seconds and quota text exist; no quota
delta, no device-GPU-hour conversion, no settlement path.

### R4 — Resume semantics speak the search's vocabulary

Kernels are stateless, so a continuation ships the survivor's checkpoint as
a dataset/model mount. The job record must carry `resume_from` (the bound
checkpoint identity) and `resume_state` in exactly the vocabulary
`candidate_search.run_search` already reads (`"resumed"` /
`"not-a-resume"`), produced by the kernel-side loader the same way the local
executor produces it: the checkpoint's manifest must verify inside the
kernel before training continues, and a loader failure is a reported
`not-a-resume`, never a silent restart. A silent restart that survives the
round would be precisely the failure the Phase 2 lineage rules exist to
make impossible.

**Status in #200:** absent. The smoke kernel trains nothing resumable.

### R5 — Failures come back classified, not just logged

The pull-on-failure path is right. The requirement is that the pulled
record (log + status + error) is *classified* through
`growth.failure_taxonomy.classify_failure` before the attempt is recorded:
INFRASTRUCTURE gets retried once with the cause attached, BUDGET_EXHAUSTED
stops the lineage, and no failure may be dropped without a class. A job
whose output pull itself failed is a failure with an error, not a gap in
the ledger.

**Status in #200:** partial. Errors are raised and persisted; no
classification, so the next generation cannot learn from a remote failure.

### R6 — The environment is pinned and its identity ships with the evidence

The CI root cause proven on 2026-10-04 (PR #205) applies verbatim inside a
kernel: a fresh `pip install` resolves today's dependency versions, and a
model helper that used to populate a provenance field may not tomorrow. The
kernel must install from a pinned constraints file, record the resolved
package versions (`pip freeze`) and the resolved base-model commit (via
`hf_resilience.resolve_model_commit`) into the output manifest, and the
evidence record's `software_runtime` must carry what the kernel recorded —
not what the dispatcher assumed.

**Status in #200:** partial. `bootstrap_environment.py` pins the *repo*
commit; package versions and model provenance are not captured.

### R7 — One kernel, one attempt, no shared state

A kernel writes only to its own output directory and never reads another
attempt's output. Cross-attempt contamination through a shared Kaggle
dataset mount is a contamination-gate failure: the mounts a job declares
are recorded, and the contamination policy's rules apply to them the same
way they apply to local data.

**Status in #200:** satisfied by construction (stateless script kernels,
explicit mounts) — keep it that way, and assert it in tests as the mount
surface grows.

### R8 — Quota is a campaign ceiling, not a hope

The weekly GPU balance is a hard bound: a campaign declares its Kaggle
budget, `quota` is read before each dispatch, and a dispatch that could
exceed the remaining balance refuses before push. Combined with R3 this
makes remote spend a first-class part of the retention/budget contract
rather than an externally-tracked courtesy.

**Status in #200:** partial. `quota` exists; nothing enforces it.

## Gap summary

| Requirement | In #200 | Blocking a 0.5 campaign? |
| --- | --- | --- |
| R1 source binding + SHA verification | absent | yes |
| R2 artifact manifest + hashes + admission | absent | yes |
| R3 settle through compute_cost | partial | yes |
| R4 resume vocabulary + in-kernel manifest check | absent | yes for round ≥ 1 |
| R5 classified failures | partial | yes |
| R6 pinned env + recorded provenance | partial | yes |
| R7 one attempt per kernel | satisfied | no — keep asserted |
| R8 quota as a ceiling | partial | yes |

The smoke path (R1-lite through its own commit pin, R7) is already valuable
and should merge; the bar above gates *campaign* use, not the PR.
