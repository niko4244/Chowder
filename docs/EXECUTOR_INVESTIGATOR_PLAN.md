# Executor Investigator — implementation plan (remaining work)

Status: **DRAFT, revised after independent review.** Builds on `incident.py`
and `investigation.py` (merged into `feature/executor-investigator`, both
tested against real incidents from a live Kaggle DPO training session).

**Revision note:** this plan was reviewed by three independent passes
(engineering soundness, benchmark rigor, scope-vs-original-request) before
any of the remaining code was written. All three found real, plan-changing
issues, not polish. This version incorporates all of them — see "What
changed in this revision" at the bottom for a record of what each review
caught, since silently rewriting a plan after review defeats the point of
having reviewed it.

## Context

**What exists (`feature/executor-investigator`, 2 commits):**
- `incident.py` — `FailureCapture`, `EnvironmentSnapshot`, `IncidentFingerprint`,
  rule-based `classify_signature`/`compute_fingerprint`. **10** real fixtures
  in `tests/fixtures_incidents.py` (not 9 — the first draft of this plan had
  this wrong; corrected here, and see the contamination-guarding note in
  Task 5 for why an inaccuracy like that matters more here than usual).
- `investigation.py` — `RemediationRegistry` (exact-fingerprint-only lookup,
  deliberately conservative — same-`signature_kind`-but-different-incident
  never auto-applies), `Investigation`/`HypothesisTrial` (reuses
  `models.Hypothesis`), `route_failure` (the known-remediation fork).
  `add_hypothesis` already rejects a `config_patch` that already failed for
  this incident's signature class.

**Original request coverage — read this before any task below.** The
person who proposed this benchmark gave an 8-row table of incident classes.
Six are addressed by this plan (directly or via the dev/hidden fixtures).
**Two are explicitly out of scope here, not silently dropped:**

| Row | In scope? | Where |
|---|---|---|
| CUDA/library crash (cuBLAS etc.) | yes | dev fixtures + Task 6 hidden set |
| Wrong/unknown machine_shape | yes | dev fixtures + Task 6 hidden set |
| HF Hub transient failures | yes | dev fixtures + Task 6 hidden set |
| Library incompatibility | yes | dev fixtures + Task 6 hidden set |
| CUDA OOM | yes | dev fixtures (2 already exist) — **Task 6 must add a hidden-set OOM variant; the first draft of this plan did not** |
| Corrupted download (hash validation) | yes, new | Task 1 gets a 4th probe (`ArtifactIntegrityProbe`); Task 6 hidden set gets a corrupted-download case |
| Undocumented backend error | yes | structural — `UNKNOWN` signature routes to a real `Investigation` by design; Task 6's disk-full hidden case exercises this path directly |
| Worker crash after training starts (checkpoint reconciliation, avoid double-training) | **out of scope for this plan** | needs Chowder's artifact/registry lifecycle semantics — a genuinely separate design question (what does "the same experiment" mean across a crash-resume boundary, how does cost accounting avoid double-charging), not an extension of the incident/investigation schema. Flagging as a named follow-up, not building it here on top of an already-large plan. |

**Design decisions already made** (repeating so this plan doesn't
re-litigate them):

1. **Chowder does not generate hypotheses itself.** `HypothesisGenerator`
   is a pluggable Protocol, mirroring `TrainingExecutor`. The benchmark uses
   a small deterministic rule-based generator (Task 4), not an LLM, so the
   benchmark measures the *investigation machinery* independently of any
   particular hypothesis-generation quality. A real LLM-backed generator is
   out of scope for this plan.
2. **The benchmark runs against replayed fixtures, not live infrastructure.**
   Today's real incidents cost 15 minutes to several hours each on real
   hardware. A `ReplayExecutor` looks up the ground-truth outcome of a given
   `(fingerprint, config_patch)` pair instead of training anything.
3. **Build one incident through the full loop before building the full
   apparatus.** The first draft of this plan specified probes, a
   remediation runner, ranking, closeout, a generator, and a 9-dimension
   scorer as six tasks that all had to land before anything could run
   end-to-end once — "build everything, then run it," a known-risky
   pattern for exactly this kind of pipeline. This revision inserts a
   walking skeleton (Task 5) between the primitives (Tasks 1-4) and the
   full benchmark (Tasks 6-8), specifically so design mistakes in ranking or
   the runner surface after one cheap round-trip, not after building all of
   Tasks 1-4 blind.

## Tasks (ordered)

### Task 1 — Diagnostic probes + explicit probe context

- **Files:** new `src/chowder/probes.py`.
- **What:** `ProbeContext` (new, frozen dataclass: `capture: FailureCapture`,
  `fingerprint: IncidentFingerprint`, `registry: RemediationRegistry`) —
  named explicitly rather than reusing `executors.py`'s `ExecutionContext`,
  which is the *training*-config context and means something different.
  `DiagnosticProbe` Protocol (`name: str`, `run(context: ProbeContext) ->
  DiagnosticProbeResult`), matching the `TrainingExecutor` Protocol pattern.
  Four concrete probes (one more than the first draft, covering the
  corrupted-download row from the original request):
  - `InstalledPackageProbe` — reads `capture.environment.installed_packages`.
  - `HardwareCompatibilityProbe` — checks `hardware_summary` against a small
    known-floor table (e.g. sm_60 < PyTorch's sm_70 floor). This exact table
    gets frozen before Task 6's hidden fixtures are written — see Task 5.
  - `KnownWorkingConfigProbe` — searches `context.registry` for any prior
    *resolved* environment with this `signature_kind` absent, surfaced as
    evidence only (registry lookups already refuse to auto-apply
    cross-incident fixes; this probe makes that history usable as evidence
    instead of hiding it).
  - `ArtifactIntegrityProbe` — given an expected sha256 and
    `capture.partial_artifact_ref`, reports a match/mismatch observation.
    Covers the original request's "corrupted download... validate artifact
    hashes" row directly, reusing the same sha256-fingerprint pattern
    `contamination.py` already established elsewhere in this codebase.
- **Test:** `tests/test_probes.py` — one test per probe; `HardwareCompatibilityProbe`
  against `WRONG_ACCELERATOR_PROVISIONED` must flag the sm_60 mismatch;
  `ArtifactIntegrityProbe` gets a synthetic mismatch case (no real corrupted
  download in the dev set today).
- **Dependencies:** none.

### Task 2 — Shared replay ground-truth type (pulled forward)

- **Files:** new `src/chowder/replay.py`.
- **What:** The first draft had Task 2's tests inventing an ad hoc
  `(fingerprint, config_patch) -> outcome` lookup that Task 6 would have
  redefined as `BenchmarkCase` later — two shapes for one concept.
  Defining it once, early: `ReplayGroundTruth` (frozen dataclass:
  `fingerprint_sha256: str`, `outcomes: Mapping[str, RemediationOutcome]`
  keyed by `config_patch_digest`) and `ReplayExecutor` (test-support code,
  lives in `src/chowder/replay.py` since both Task 3 and Task 6 need it —
  not fixture-specific despite only being used with fixtures today).
  `ReplayExecutor.run(config_patch) -> RemediationOutcome` looks up the
  digest; an unlisted patch is a hard error (a benchmark case must specify
  ground truth for every patch a generator might plausibly propose, not
  silently default to failure).
- **Test:** `tests/test_replay.py` — lookup hit/miss, digest collision
  safety (two structurally-different patches never share a digest).
- **Dependencies:** none beyond the pre-existing `incident.py`/
  `investigation.py` — despite appearing after Task 1 above, this does not
  actually need `ProbeContext` and can be built in parallel with it.

### Task 3 — Bounded remediation experiment runner

- **Files:** new `src/chowder/remediation_runner.py`.
- **What:** `RemediationExperiment.run(trial, executor, max_attempts) ->
  RemediationRecord`. Two limits, both now real: `max_attempts` (new
  parameter here — the first draft claimed this was "already partially
  modeled" and it wasn't; only the budget half existed) and
  `investigation.remaining_budget()` (exists). **Crash handling, specified
  here rather than left implicit:** if invoking the executor itself raises
  (not just returns a failing outcome), the runner captures a *new*
  `FailureCapture` from that exception and returns a `RemediationRecord`
  with `outcome=PARTIALLY_RESOLVED` and the new capture attached via
  `notes` (a fingerprint, not the full capture — see Task 4 for why the
  full record doesn't belong in a string field). The runner does **not**
  recursively open a new `Investigation` itself — that decision belongs to
  whatever's driving the loop (Task 5), calling `route_failure` again on
  the new capture, the same way the first failure was routed. Keeping this
  out of the runner keeps `investigation.py`'s state machine simple and
  matches the existing separation between "produce evidence" and "decide
  what to do with it."
- **Test:** `tests/test_remediation_runner.py`, using `ReplayExecutor`
  against dev fixtures: a correct patch resolves; a wrong one produces
  `DID_NOT_RESOLVE`; a patch not present in `ReplayGroundTruth.outcomes`
  raises immediately (per Task 2); attempt/budget caps enforced;
  `PARTIALLY_RESOLVED` handling exercised via a fixture designed to trigger
  it.
- **Dependencies:** Task 2 only (needs `ReplayGroundTruth`/`ReplayExecutor`
  directly; does not call probes itself — Task 1 can land before or after
  this one).

### Task 4 — Reproducibility record (structured, not string-embedded)

- **Files:** new `src/chowder/closeout.py`.
- **What:** The first draft proposed embedding "the full trial history...
  as a canonical, hashable audit trail" inside `RemediationRecord.notes:
  str` — a free-text field on an existing frozen dataclass. That requires
  inventing an undocumented serialization scheme and mixes a structured
  audit trail into a field meant for a human-readable note. Instead:
  `AuditTrail` (new frozen dataclass — `investigation_id: str`,
  `trials: tuple[TrialSummary, ...]` where `TrialSummary` captures each
  trial's hypothesis, config_patch, rank, probe evidence, and outcome in
  typed form) plus `trail_sha256` (canonical digest, same pattern as
  `incident.py`'s `_canonical_digest`). `finalize_investigation(investigation)
  -> tuple[RemediationRecord, AuditTrail]` requires `status is RESOLVED`,
  returns the record (unchanged shape, so `RemediationRegistry.with_record`
  keeps working as-is) alongside the trail as a separate, first-class
  artifact — not smuggled into a string.
- **Test:** `tests/test_closeout.py` — finalize a resolved investigation
  built from a dev fixture; assert the trail contains every trial
  (including failed ones, in order); assert the resulting `RemediationRecord`
  makes the fingerprint immediately resolvable via `RemediationRegistry.lookup`
  on a fresh registry.
- **Dependencies:** Task 3.

### Task 5 — Walking skeleton: one real incident, end to end

- **Files:** new `src/chowder/hypothesis_generation.py` (minimal version
  only — a single hardcoded candidate for one `signature_kind`, not the
  full table yet); new `src/chowder/ranking.py` (minimal version — sort by
  probe-corroboration count only, no cost tiebreak yet); new
  `tests/test_walking_skeleton.py`.
- **What:** Drive exactly one dev-set incident
  (`QWEN3_5_CONV1D_NO_ENGINE` — chosen because its real fix is known and
  simple: disabling cuDNN) through the entire loop: `route_failure` →
  `Investigation` → `HardwareCompatibilityProbe` + `InstalledPackageProbe`
  → one hardcoded hypothesis from the minimal generator → `rank_trials`
  (trivial, one trial) → `RemediationExperiment` via `ReplayExecutor` →
  `resolve` → `finalize_investigation` → `RemediationRegistry.with_record`.
  **This is the actual first checkpoint of the plan** — nothing in Tasks
  6-8 starts until this passes, because this is what proves the primitives
  from Tasks 1-4 actually compose the way the diagram claims, on one real
  case, before multiplying that design across nine more.
- **Test:** `tests/test_walking_skeleton.py` — the full sequence above,
  asserting the final registry resolves the incident's fingerprint. If this
  reveals a design problem in `ranking.py`'s or `remediation_runner.py`'s
  API, **fix it here** rather than carrying the flaw into Tasks 6-8's
  larger surface.
- **Dependencies:** Tasks 1-4.

### Task 6 — Full generator and ranking, run against all 10 dev fixtures

- **Files:** extend `hypothesis_generation.py` to the full
  signature_kind-keyed table (documented in its module docstring as an
  intentionally thin placeholder — its job is to make the benchmark
  runnable and prove the investigation machinery works, not to claim
  agent-like reasoning); extend `ranking.py` with the cost tiebreak; new
  `tests/test_dev_fixture_run.py`.
- **What:** Run all 10 dev fixtures (not just the one from Task 5) through
  the same loop, with `ReplayGroundTruth` supplied for each. This is still
  entirely within the dev set — no scoring, no hidden fixtures, just
  confirming the machinery handles the full, already-known variety (OOM
  twice, kernel-unavailable vs. execution-failed staying distinct per the
  existing regression tests, device-mismatch, config-invalid, etc.)
  end-to-end, not just the one case Task 5 proved.
- **Test:** `tests/test_dev_fixture_run.py` — all 10 dev fixtures resolve
  or are correctly abandoned (per fixture-specific ground truth); the two
  existing regression properties (conv1d/cuBLAS never collapse; the two
  real OOMs never collapse) still hold when run through the *investigation*
  path, not just the fingerprinting path checked in `test_incident.py`.
- **Dependencies:** Task 5.

### Task 7 — Hidden fixture set, built under a contamination guard

- **Files:** new `tests/fixtures_incidents_hidden.py`; new
  `docs/HIDDEN_SET_FREEZE.md` (the freeze record, see below).
- **The contamination risk, named plainly:** the same author (agent or
  human) who wrote `classify_signature`'s rule table, the hardware-floor
  table, and the hypothesis generator's rule table is also about to write
  the hidden fixtures meant to test whether those rules generalize. Knowing
  the rules' exact shape while writing "held out" test cases makes it easy
  — even unintentionally — to write cases that don't stress the rules'
  actual blind spots. File separation (`fixtures_incidents_hidden.py`
  vs. `fixtures_incidents.py`) prevents literal training-on-the-answer; it
  does not prevent this.
- **What, concretely:**
  1. **Freeze first.** Before writing a single hidden fixture, compute and
     record sha256 hashes of `incident.py`, the hardware-floor table in
     `probes.py`, and `hypothesis_generation.py` into
     `docs/HIDDEN_SET_FREEZE.md`. A test (`test_freeze_intact`) asserts
     these hashes match at collection time — if a rule changes after the
     freeze, the test fails loudly rather than the hidden set silently
     drifting to match a moved target.
  2. **Pre-register predictions before writing outcomes.** For each hidden
     case, write down the expected `signature_kind` (or explicitly
     "expected UNKNOWN") *before* running anything, in this file. Six
     cases, matching the original request's table exactly, plus the two
     rows added in the Context table above:
     - Qwen3.5 + A10, same library-incompatible-architecture message →
       expect `DEPENDENCY_INCOMPATIBLE` (tests whether the classifier
       generalizes past "T4" specifically — nothing in the rule table
       references hardware for this signature kind, so this should pass
       cleanly; a failure here would mean the rules are more
       hardware-coupled than intended).
     - Llama + T4x2, a different CUDA error under the same accelerator
       family → expect a specific *different* signature_kind stated here
       once the exact synthetic error text is drafted (not "some CUDA
       error" — vague prediction is the same failure mode Agent 2's review
       flagged for the scoring dimensions).
     - Qwen3.5 + T4x2, disk-full mid-download → expect `UNKNOWN` (no rule
       in `_SIGNATURE_RULES` names this; this case exists specifically to
       confirm the system fails safely into a real `Investigation` instead
       of misclassifying into a near-miss bucket, per the original
       request's "escalate from known remediation rules into an
       investigation loop" row).
     - Qwen3.5 + T4x2, HF 503 → expect `NETWORK_TRANSIENT` via a status-code
       phrase, not today's `RemoteProtocolError` wording — tests the rule
       list's phrase coverage, not just its category count.
     - Qwen3.5 + T4x2, incompatible transformers version, different exact
       wording than today's `qwen3_5`-not-recognized case → expect
       `DEPENDENCY_INCOMPATIBLE`.
     - Qwen3.5 + T4x2, wrong `machine_shape`, a *different* incorrect
       string than today's `"NvidiaTeslaT4x2"` → P100 → expect
       `HARDWARE_INCOMPATIBLE`. Using the same string as the dev fixture
       would test memorization, not classification — this must be a
       genuinely different wrong value.
     - **New, covering the CUDA OOM gap the first draft of this plan
       missed:** a third real memory-capacity failure shape (e.g. an OOM
       during a *different* stage than either dev-set OOM — attention
       computation rather than kbit-prep or fp32 upcast) → expect
       `CUDA_OOM`.
     - **New, covering the corrupted-download row:** a downloaded artifact
       whose sha256 doesn't match its expected manifest value → expect a
       new `SignatureKind` this plan does not currently define
       (`ARTIFACT_CORRUPTED`, distinct from `ARTIFACT_NOT_FOUND` — a
       missing file and a wrong-content file need different remediations,
       redownload vs. re-check-the-source, and conflating them would be
       exactly the kind of near-miss-bucket misclassification the
       disk-full case above is designed to catch elsewhere).
  3. **Independent authorship, if practical.** Ideally these fixtures are
     drafted by a party (a separate session, a separate reviewer) who has
     not read `incident.py`'s rule table, working only from the original
     request's plain-English row descriptions. If that's not practical for
     this iteration, note the deviation explicitly in
     `HIDDEN_SET_FREEZE.md` rather than silently accepting the weaker
     guarantee.
- **Test:** `test_freeze_intact` (above). No scoring test here — scoring is
  Task 8.
- **Dependencies:** Task 6 (rules must be stable/tested on the full dev set
  before freezing them).

### Task 8 — Benchmark scorer and honest reporting

- **Files:** new `src/chowder/benchmark.py`; new `tests/test_benchmark.py`.
- **What:** `BenchmarkCase` (`FailureCapture` + `ReplayGroundTruth` +
  expected `SignatureKind`). `BenchmarkScorer`, the 9 original dimensions,
  with two fixes from review:
  - **False blame**, redesigned per review rather than left as the
    known-weak keyword check the first draft proposed: score against
    whether a trial's `config_patch` touches a namespace pre-registered as
    "infrastructure" (e.g. keys under `driver.*`, `library.*`,
    `hardware.*`, `network.*`) versus "model/data" (e.g. `dataset.*`,
    `hyperparameters.*`) for cases whose ground truth is infrastructure —
    a structured check against the patch a hypothesis actually proposes,
    not free text the same generator that's being graded also wrote. The
    namespace list is fixed in this plan now, not decided during
    implementation.
  - **No aggregate hidden-set pass rate.** `BenchmarkReport` presents a
    per-case table (predicted vs. ground truth vs. what happened), not a
    fraction. Any summary text describing hidden-set results must not
    read as "N/6 passed" — six cases is far too small to support a
    generalization claim, and presenting it as a score invites treating
    noise as signal.
  - `run_benchmark(cases, generator, registry) -> BenchmarkReport` — same
    orchestration Task 6 already exercises against the dev set, now also
    run against the hidden set (Task 7) and scored.
- **Test:** `tests/test_benchmark.py` — full scored run against dev cases
  (10, expected clean given Task 6 already proved this); full run against
  hidden cases (7, per Task 7), asserting the report is produced and
  matches the pre-registered predictions field-for-field — a prediction
  mismatch is a **finding to write up** (which case, why, what class of
  rule-set gap it reveals), not a test to quietly loosen until it passes.
- **Dependencies:** Task 7.

## Verification

- Every new module has a matching test file, per the existing 1:1
  convention.
- `ruff check src/ tests/` clean.
- Full suite green except the 6 pre-existing unrelated failures already
  confirmed present on `main` (`replay_history`/`repair_candidates`/
  `autonomous_repair`) — re-confirm the count hasn't grown.
- Task 5's walking skeleton passes before Task 6 starts; Task 6's full
  dev-set run passes before Task 7's freeze; Task 7's freeze is intact
  before Task 8 scores anything against the hidden set. This ordering is
  load-bearing, not just tidy — it's what keeps the hidden-set result
  honest.
- The benchmark's hidden-set report is read and written up as a finding
  (what generalized, what didn't, why) — not treated as a bar this plan
  requires clearing to be considered done.

## What changed in this revision

Three independent reviews, each finding real issues (full reviews not
reproduced here, summarized for record-keeping):

- **Engineering review:** crash-handling inside a remediation experiment
  was undefined despite `PARTIALLY_RESOLVED` already existing in the data
  model (→ specified in Task 3); the audit trail was designed to be
  crammed into a free-text field (→ `AuditTrail` as a first-class type,
  Task 4); the benchmark's ground-truth format was going to be invented
  twice, once ad hoc in the old Task 2 and once formally in the old Task 6
  (→ unified into `ReplayGroundTruth`, Task 2, built once); "max attempts
  per hypothesis" was claimed to already exist and didn't (→ real parameter,
  Task 3); the old Task 6 was actually four tasks wearing one number (→
  split into Tasks 5-8); `DiagnosticProbe`'s context type was undefined (→
  `ProbeContext`, Task 1).
- **Rigor review:** the dev/hidden split guarded against textual leakage
  but not against the same author knowing the rule table while writing
  "held out" cases (→ freeze-and-hash discipline, Task 7); false-blame
  scoring checked output written by the same generator being graded (→
  redesigned against structured config-patch namespaces, Task 8); n=6
  hidden cases can't support a generalization claim (→ per-case reporting
  only, no aggregate score, Task 8); nothing pre-registered what a hidden
  result should look like before running it (→ predictions written into
  this plan now, Task 7).
- **Scope review:** two of the original request's 8 incident rows
  (checkpoint reconciliation after a worker crash; corrupted-download hash
  validation) were silently absent from the first draft (→ corrupted
  download is now in scope via a new probe and hidden case; checkpoint
  reconciliation is explicitly named out-of-scope with a stated reason,
  Context table); no hidden-set OOM variant existed despite two dev
  fixtures (→ added, Task 7); the plan built the entire pipeline before
  running any of it once (→ Task 5's walking skeleton inserted specifically
  to catch this).
