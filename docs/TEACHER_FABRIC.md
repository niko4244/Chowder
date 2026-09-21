# Teacher Fabric / Remote Intelligence Distillation — Priority 8 design

Status: **architecture documented; Slices A–B implemented; Slices C–J not
started.** This document is the design record for the mission brief
(preserved verbatim at the time of writing in the PR that introduced it).
Nothing in the "implemented" column may be read as "commissioned": Slice A
contains zero network code and no real provider. Every unimplemented slice
below is labeled as such, and the open research questions at the end are
genuinely open.

## 1. Architecture audit of current Chowder (as of PR #110)

The seams Teacher Fabric composes with, all verified in source:

- **Executor protocol shape** (`executors.py`): `TrainingExecutor` /
  `EvaluationExecutor` are `@runtime_checkable` Protocols with a `name`
  attribute and a `profile → run/evaluate → cancel` lifecycle, with
  `CostEstimate` as the quoted-cost type and frozen, self-validating
  artifact dataclasses (`TrainingArtifact`, `EvaluationOutcome`) that
  cross-check GPU-hour accounting against `ResourceUsage`.
- **Resource accounting** (`resources.py`): `ResourceUsage` is the
  accounting primitive (`accelerator_seconds`, `gpu_hours` property,
  `from_wall_time`), covering GPU-hour cost but **not** monetary or token
  cost — Teacher Fabric must extend the cost vocabulary without breaking
  this one.
- **Provenance** (`provenance.py`): `EvidenceManifest.digest()` over
  canonical JSON, `sha256_file` / `sha256_directory` for content hashes.
  Registry evidence rows are append-only (`_insert_immutable`).
- **Persistence** (`registry.py`): SQLite, append-only tables
  (`experiments`, `training_runs`, `evaluation_runs`, `results`,
  `execution_incidents`, `run_events`, `teacher_signals`), schema-versioned
  with migration history. Slice B added `teacher_signals` (an append-only
  ledger of stored signals, keyed by content address) as migration 4.
- **Experiment lifecycle** (`cycle.py`, `engine.py`): propose → reserve →
  train → evaluate → `EvolutionEngine.adjudicate()` → hard gate. The gate
  (`gate.evaluate_candidate`) is the sole promotion authority;
  `CandidateCycleOutcome.executor_analysis` already carries structured
  failure evidence out of the cycle.
- **Evidence views** (`intervention_outcomes.py`, `censored_outcomes.py`):
  normalized per-arm rows for scored and result-less experiments, with an
  explicit honesty rule (read from stored evidence or `None`). Teacher
  effectiveness data must land in compatible per-arm shape.
- **Search/selection** (`candidate_selection.py`, `tournament.py`,
  `successive_halving.py`): UCB1 bandit over arms defined by
  `dotted_paths(config_patch)`; proven library capabilities, not yet the
  default project-runner controller.
- **Local-first model handling** (PR #104: `docs/LOCAL_MODELS.md`,
  `hf_resilience.py`, `moe_planning.py`): local cache/disk discipline and
  MoE planning that Teacher Fabric must compose with, never overwrite.
- **Failure handling** (`incident.py`, `executor_investigator.py`,
  `investigation.py`): structured `FailureCapture` → `IncidentFingerprint`
  → routing; the pattern a remote-teacher failure path should reuse rather
  than reinvent.

## 2. Gap analysis (what the mission needs that does not exist yet)

1. No teacher abstraction at all: every training path assumes the model
   being trained or evaluated is local and executable.
2. No monetary/token cost dimension: `ResourceUsage` accounts GPU-seconds
   only; the ledger cannot express "this step cost $0.04".
3. No signal store: teacher outputs are not persisted anywhere, so they
   cannot be deduplicated or reused across experiments.
4. No tokenizer-compatibility gate: nothing today compares tokenizer
   identity between two models, and nothing fails closed on it.
5. No remote-job notion: manifests, idempotent retries, and partial-shard
   recovery do not exist.
6. No teacher-effectiveness evidence: `InterventionOutcome` rows have no
   place where "this arm's signal came from teacher X" is recorded.

## 3. Teacher Fabric architecture

Provider-neutral, in the brief's own component vocabulary (names kept; the
one deliberate adjustment is noted):

```
TeacherRegistry        registration + capability negotiation entry point
TeacherCapabilities    what a provider declares it can do (never inferred
                       from the provider's name/type)
TeacherRequest         what Chowder wants: signal kind + inputs + digests
TeacherSignal          the returned value + provider-reported usage
TeacherSignalArtifact  the durable, fully-provenanced record of one signal
TeacherProvider        @runtime_checkable Protocol (profile/query/cancel,
                       mirroring TrainingExecutor's shape; "run" is named
                       "query" because that is the domain verb)
TeacherBroker          picks the provider for a request and returns its
                       artifact (Slice C+; negotiation primitives land
                       first in Slice A)
TeacherQueryController deterministic escalation policy: which teacher
                       tier, if any, a situation deserves (Slice D)
TeacherSignalStore     content-addressed, budgeted local cache (Slice B)
teacher cost/resource accounting  extends the ledger (Slice D)
provider adapters      real remote providers (Slice F+)
tokenizer compatibility validation (Slice A: identity-hash gate)
remote-job manifests   (Slice G)
teacher-effectiveness history (Slice D+)
```

Dependency direction: Teacher Fabric depends on `models`, `resources`,
`provenance`, `executors` types. Nothing in Chowder's training path
depends on Teacher Fabric; the hard gate never consults it. A disabled
(unconfigured) Teacher Fabric is behaviorally absent.

## 4. Data contracts (Slice A — implemented in `src/chowder/teacher_fabric.py`)

- `SignalKind` (str enum, the brief's 10-item taxonomy): `scalar_reward`,
  `generated_answer`, `critique`, `revised_answer`, `candidate_ranking`,
  `sampled_token_logprobs`, `topk_logprobs`, `full_logits` (research-only),
  `hidden_projection` (experimental), `remote_adapter`.
- `TeacherCapabilities`: the brief's exact flag set (generate, critique,
  revise, rank, scalar_reward, selected_token_logprobs, topk_logprobs,
  hidden_projection, remote_training), strict booleans, plus a
  `supports(signal_kind)` mapping so negotiation never string-matches.
- `TeacherRequest`: teacher id, signal kind, input payload, optional
  student trajectory, student tokenizer identity, parameters; canonical
  `digest()` (the `EvidenceManifest` pattern) so equal requests are
  recognizably equal before any cache exists.
- `TeacherSignal`: signal kind + payload + provider-reported usage
  (`ResourceUsage | None` for GPU-backed teachers), declared monetary cost
  and token counts — provider-reported evidence, recorded as such.
- `TeacherSignalArtifact`: every metadata field the brief requires
  (teacher stable id, provider type, model/revision, tokenizer identity,
  schema version, request/prompt/student-trajectory digests, generation
  parameters, timestamp, monetary/token/GPU cost, latency via
  `ResourceUsage.wall_seconds`, payload content hash, parent job-manifest
  digest, provenance/licensing metadata) + canonical `digest()`.
- `TeacherProvider` protocol: `profile`/`query`/`cancel` +
  `capabilities()`, `@runtime_checkable`.
- `TokenizerIncompatibilityError`: token-aligned signal kinds
  (`sampled_token_logprobs`, `topk_logprobs`, `full_logits`,
  `hidden_projection`) against a provider whose tokenizer identity hash
  differs from the student's **fail closed** — rejection, never
  approximation. Downgrade to text/ranking/reward/critique supervision is
  available only as an explicit, caller-invoked `downgrade_request()`
  choice, never an automatic substitution.

## 5. File/module plan

| Slice | Module(s) | Status |
|---|---|---|
| A | `src/chowder/teacher_fabric.py`, `tests/test_teacher_fabric.py` | **implemented** |
| B | `src/chowder/teacher_signal_store.py` + registry migration | **implemented** |
| C | `src/chowder/teacher_blackbox.py` (Regression Surgeon integration) | not started |
| D | `src/chowder/teacher_query_controller.py`, ledger extension | not started |
| E | selected-token scorer protocol + objective-level tests | not started |
| F | first real remote provider adapter (separately gated tests) | not started |
| G | `src/chowder/teacher_remote_jobs.py` (manifests, retry/resume) | not started |
| H | remote distillation microjob (adapter return + local re-eval) | not started |
| I | multi-teacher experiments (budgets, staleness) | not started |
| J | teacher-selection meta-controller research | not started |

## 6. Integration map (into existing modules)

- **Hard gate**: untouched. Teacher signals inform candidate *generation*
  and repair-material *proposal*; only `gate.evaluate_candidate` over real
  independent evaluation promotes. (Regression rule 5/6.)
- **Regression Surgeon**: black-box critique/revision/correction signals
  enter as repair material through the existing repair-request pipeline
  (`repair_requests.py`, `repair_sources.py`), not a parallel repair
  system. (Slice C.)
- **Resource accounting**: teacher queries become ledger entries with
  monetary/token/GPU dimensions alongside `ResourceUsage` GPU-hours.
  (Slice D.)
- **Evidence views**: teacher effectiveness lands as per-arm rows joining
  the same `dotted_paths` arm identity `intervention_outcomes.py` and
  `censored_outcomes.py` use. (Slice D/J.)
- **Project runner**: a `teacher_fabric` config section, validated by the
  existing fail-closed config validation; absent section = feature off,
  byte-identical behavior. (Regression rule 1/2.)
- **Contamination rules**: teacher-generated training data never touches
  protected/holdout material (`contamination.py` owns that boundary).
- **Registry**: new append-only tables via the existing migration path;
  artifacts carry digests that make registry rows verifiable.

## 7. Phased roadmap

See §5's table; the slice order is the brief's own (A → B → C → D → E → F
→ G → H → I → J), each with tests + full regression suite + documentation
+ truthful roadmap update, none marked complete before its evidence
exists. Slice F (real remote commissioning) additionally requires its own
env-gated test class, per regression rules 11/12.

## 8. Threat / failure model

- **Provider failure** → structured `FailureCapture`-style evidence; a
  failed teacher query may fail a candidate's *signal acquisition*, never
  corrupt experiment history (append-only registry; no in-place mutation).
- **Teacher revision drift** → model/revision is part of every artifact's
  identity and digest; a re-queried teacher at a new revision produces a
  distinguishable artifact, not a silent replacement.
- **Late/partial results** → artifacts are complete-or-absent. Slice B
  implements atomic writes (temp file + `os.replace`) and
  interrupted-write recovery (temp files and unreferenced payloads swept
  at store open); a payload that fails its content hash is never served
  (`verified-or-absent`), and whole-payload integrity is the Slice B
  primitive — streamed shard *transport* remains F/G work.
- **Overclaimed capabilities** → negotiation trusts only the declared
  `TeacherCapabilities` block; a provider whose response does not match
  its declared signal kind is a hard error, not a best-effort parse.
- **Secrets** → API keys live only in provider adapters' configuration
  resolution (environment/config, never in `TeacherRequest` payloads,
  artifacts, manifests, logs, or committed config). (Rule 10.)
- **Untrusted remote results** → every artifact carries payload content
  hash + provenance; Slice G adds worker signatures. Remote success is
  never promotion evidence (rule 13 + H's local re-evaluation).
- **Data leakage** → prompt/input digests identify without containing;
  whether raw prompts may be persisted at all is a config decision with a
  privacy-preserving default (digests only). Flagged in §16.

## 9. Storage / cost model

- Teacher **weights never download locally** as a side effect (rule 3);
  the disk preflight for teacher-enabled projects reasons about signal
  space, not model space.
- Local cache ceiling is configurable; Slice B takes it as a **required**
  `local_cache_max_bytes` constructor argument with no default, and
  measures its footprint (`disk_bytes`) rather than modeling it (rule 14).
  The exact default remains a genuine open question — see §16.1.
- Slice B implements the storage model for whole payloads: artifacts are
  content-addressed (cache key = digest over request digest + payload-file
  hash; payload files named by their own sha256), so dedup is exact and
  eviction is safe (a re-queried signal is re-derivable). The ledger row
  is evidence and survives eviction; the cache copy is bookkeeping.
  Streamed shard *transport* is Slice F/G work.
- Cost accounting is triple-dimension (monetary, token, GPU-hour where
  applicable) because teacher tiers differ in which resource dominates.

## 10. Test matrix

| Layer | Slice A | Later slices |
|---|---|---|
| Schema validation (fail-closed `__post_init__`) | ✔ | B (corruption), G (manifests) |
| Capability negotiation (supported/unsupported) | ✔ | C, E |
| Tokenizer fail-closed rejection + explicit downgrade | ✔ | E (real scorer) |
| Request/artifact digest determinism | ✔ | B (dedup), G (idempotency) |
| Fake provider end-to-end artifact construction | ✔ | C (black-box ops), D (controller), I (multi-teacher) |
| Local-cache budget/eviction/atomicity | — | B ✔ |
| Real remote commissioning (env-gated) | — | F |
| Network-free CI | ✔ (no network anywhere) | enforced every slice |

## 11. Benchmark protocol (to be run when slices C–G exist; designed now so
the measurement is not invented afterwards)

The brief's A–G arms (no teacher / local teacher / black-box critique /
black-box ranking / selected-token scoring / sparse top-K / remote
microjob) measured on: target benchmark improvement, regression count,
hard-gate pass rate, teacher tokens, monetary cost, GPU-hours, wall time,
network bytes, local disk consumed, cached-signal reuse, retries/failures,
improvement per teacher dollar, improvement per teacher token. No strategy
is declared better from one example; conclusions require the same
evidence-gated discipline as every other Chowder claim.

## 12–14. Implemented first safe slice (A), tests and results,
documentation

Slice B is `src/chowder/teacher_signal_store.py` +
`tests/test_teacher_signal_store.py` (34 tests): the content-addressed,
budgeted `TeacherSignalStore` with atomic writes, interrupted-write
recovery, verified-or-absent reads (payload re-hashed on every read;
corruption raises `SignalIntegrityError`, never serves), exact dedup over
`(request_digest, payload_file_sha256)`, a required (no-default)
`local_cache_max_bytes` with measured `disk_bytes()`, explicit
caller-chosen `discard` (no silent eviction policy — that is Slice D's),
registry migration 4 adding the append-only `teacher_signals` ledger
(evidence survives cache eviction; re-acquiring identical evidence after
eviction replays idempotently — `stored_at` is first-acquisition
bookkeeping, and genuine divergence raises `RegistryInvariantError`), and
lossless artifact round-trips including GPU-backed `ResourceUsage`
(`canonical_payload` now carries `peak_vram_gb_by_accelerator`; nothing
persisted artifacts before Slice B, so the canonical form change has no
compatibility surface).

Slice A is `src/chowder/teacher_fabric.py` + `tests/test_teacher_fabric.py`
(see the introducing PR for exact counts): schemas with fail-closed
validation, capability negotiation incl. the tokenizer identity gate,
provider protocol, artifact format with full provenance metadata, a
deterministic `FakeTeacherProvider` (explicitly a test double — no network
code exists in the module), and the test matrix's Slice A column. Full
existing suite green; roadmap updated truthfully (Priority 8 added under
RESEARCH, Slice A done, B–J not started).

## 15. Truthful roadmap status

`docs/ROADMAP.md` RESEARCH section carries Priority 8 with exactly this
status: Slices A–B implemented; C–J not started; nothing remote commissioned;
no claim of student improvement exists anywhere — no student has been
taught by a remote teacher yet.

## 16. Remaining unresolved research questions (genuinely open)

1. **Local cache default size.** The brief requires a justified default
   for `local_cache_max_gb`. Honest answer: it depends on the user's disk
   pressure vs. re-query cost tradeoff, which we have no data for.
   Slice B therefore makes `local_cache_max_bytes` a required argument
   (no default) and exposes the measured footprint (`disk_bytes()`) and
   hit counters the experiment needs. **Experiment (unruns, not started):**
   instrument hit-rate and re-query cost across candidate ceilings
   (0.25 / 1 / 4 GB) on the Phase A campaign workload once real queries
   exist; pick the knee, document it as measured, not chosen.
2. **Hidden-projection distillation: now or deferred?** The brief's
   specialist perspectives genuinely disagree: the distillation-researcher
   view values dense representation-level signal; the
   reliability/review view notes it is the least standardizable
   (architecture-coupled, tokenizer-and-layer-index dependent) signal with
   the weakest remote-provider support. **Resolution: deferred behind the
   `hidden_projection` experimental flag; design an experiment (Slice E/F
   timeframe) comparing student improvement-per-cost against
   selected-token scoring on the same workload before any broader
   investment.**
3. **Sparse top-K logit caching is biased.** Real literature (Sparse Logit
   Sampling, ACL 2025) shows naive top-K caching gives a biased estimate
   of the teacher distribution (truncated mass does not renormalize
   correctly). Do not implement it as a solved mechanism; if used, carry
   the bias correction from that literature and validate the corrected
   estimator empirically. Untouched until Slice E/F.
4. **Cross-tokenizer alignment.** Hard, separately-solved problem
   (NeMo-RL #3206; DistillKit delegates to a separate tool). Chowder fails
   closed instead. Any future alignment method must be separately
   researched and validated before the gate accepts it.
5. **Multi-teacher balancing.** Open-MOPD's real "integration gap" result
   (3.50 → 0.31) confirms naive averaging fails, but the right balancing
   policy for *this* system's workloads is unknown. Slice I, behind an
   experimental flag, with per-domain/token budget accounting first.
6. **Verifier/reward and critique/revision distillation frameworks, and
   hidden/representation distillation literature** — the research audit is
   not exhaustive in these areas; more literature work is owed before
   Slices C/E design their signal payloads in detail.
7. **Prompt-privacy default.** Whether raw prompts (not just digests) may
   be persisted in artifacts by default is a real tension between
   debuggability and leakage risk; the current default stores digests +
   declared parameters only. Revisit with the security review before
   Slice C makes prompts operationally necessary.
8. **Teacher-proxy research** (small local model approximating teacher
   judgments): designed in the brief, deliberately not implemented until
   real teacher-query history exists to train/evaluate it against (same
   evidence-first rule as Priority 6's policy learning).

## 17. Pull request

The introducing PRs contain the regression-safe changes only: Slice A +
this document + the truthful roadmap update, with the full existing test
suite green and no network code added.
