# Training backends: unified growth-loop contract (Option A)

Status: implemented (contract + LOCAL + UNSLOTH + KAGGLE re-exposure). Real
golden-path runs through each backend are a separate verification step that
needs real CUDA / Kaggle quota.

## Why this exists

The growth loop already had two backends — LOCAL (`SubprocessTrainingFn`) and
KAGGLE (`KaggleTrainingFn` over `ComputeBackend`) — and the repo already had
real LOCAL and UNSLOTH *trainers* (`TransformersPeftExecutor`,
`UnslothPeftExecutor` via `backend_selection.create_training_executor`). What
was missing was a single declared `training_backend` surface the campaign
runner dispatches through, so downstream code never writes
`if kaggle / elif unsloth / elif local`.

## The declaration

A campaign declares *where and how* it executes, as a first-class field on the
manifest (absent means `local`: the historical subprocess path, so every
manifest predating the field runs exactly as before):

```json
"training_backend": {"provider": "local", "config": {"device": "auto"}}
```

- `provider` is a closed enum: `local | unsloth | kaggle | auto`. Unknown
  providers refuse at load; a provider is never inferred from what happens to be
  installed.
- `config` is provider-specific, and unknown keys refuse (an unsupported key
  would silently change what the backend is told to do):
  - `local` / `unsloth`: `device` (`auto | cuda | cpu`) and optional
    `measured_evidence` (a directory, or a list of directories, containing prior
    attempt evidence -- see AUTO below).
  - `kaggle`: required `repository`, `commit_sha`, `entry_point`, `mounts`,
    `attempts_root`, `timeout_seconds`, `accelerator`, `payload`
    (`{"kind": "corpus-training", ...}` or `{"kind": "command", "command": [...]}`),
    and optional `owner`, `input_paths`, `model_commit`, `pip_extras`,
    `projection_tolerance`, `declared_quota_ceiling_gpu_hours`,
    `measured_evidence`, and `prepared_path`. **`prepared_path` is optional**: without it the provider
    assembles the prepared campaign from the manifest's *own* declared input
    fields (the same ones `prepare_campaign` writes, and the same ones the run
    phase requires), so a campaign can be dispatched to Kaggle from one
    declaration instead of two documents that can disagree. A declared input
    the manifest omits refuses, naming it.
- `auto`: `candidates` (the ordered provider list to try; default
  `local, unsloth, kaggle`), plus `measured_evidence` as above.
- `measured_evidence` names directories holding prior attempt evidence
  (`training-evidence.json`). A missing directory refuses
  (`TRAINING_BACKEND_EVIDENCE_MISSING` -- an undeclared directory must not read
  as "no measurements"), and evidence stamped by another provider refuses
  (`TRAINING_BACKEND_EVIDENCE_INVALID` -- counting another backend's cost would
  fabricate the comparison). Attempts whose evidence exists but records no cost
  are excluded and *named*, never counted as zero.
- `TrainingBackendDeclaration.from_mapping` re-spoken as a
  `CampaignManifestError` at manifest load, and
  `campaign_runner.FIELD_ENFORCEMENT["training_backend"]` names the behavior it
  drives (`assert_every_field_enforced` fails if the schema and that table
  diverge).

`training_backend` is **not** the project template's `backend.type`. The
template selects the *trainer engine inside* a backend (`backend.type: peft`,
`engine: unsloth`); `training_backend` selects *where and how the campaign
executes*. A campaign that runs Unsloth locally declares
`provider: unsloth` **and** a template with `engine: unsloth`, and the provider
refuses a mismatch rather than discovering it mid-run.

## The unified contract

`src/chowder/growth/training_backends.py` owns:

```python
class TrainingBackend(Protocol):
    provider: str
    trainer: str

    def capabilities(self) -> BackendCapabilities: ...
    def preflight(self, manifest) -> PreflightResult: ...
    def estimate(self, manifest, recipe) -> BackendEstimate: ...
    def admit(self, manifest, recipe) -> tuple[str, str] | None: ...
    def build_training_fn(self, manifest, *, state_root=None, runner=None) -> TrainingFn: ...
    def resume(self, manifest, recipe, checkpoint) -> TrainingRecipe: ...
    def collect(self, manifest, attempt_dir) -> Mapping[str, Any]: ...
    def verify(self, manifest, attempt_dir) -> tuple[str, str] | None: ...
```

`build_training_fn` is the seam the cycle already uses, so `GrowthCycle` and
`run_campaign` are not rewritten. The other methods answer questions the
campaign already asks, so the *runner* is the only code that branches — once,
at the backend boundary (`campaign_runner.build_executor_with_selection`).

Two rules make the boundary safe:

- **Structural refusals stop the campaign before compute.**
  `STRUCTURAL_PREFLIGHT_CODES` (unknown provider, unsupported config key,
  undeclared/invalid template, trainer mismatch, an Unsloth knob the engine
  refuses, incomplete Kaggle wiring) make the runner refuse. A *hardware* fact
  (no accelerator visible) is reported by the panel and enforced where the
  declaration demands it (`config.device: cuda`) or by `auto`, which only
  chooses providers whose preflight admitted.
- **AUTO ranks on measured costs where they exist, declared claims otherwise.**
  An `auto` declaration may choose only among candidates whose preflight passes.
  Each candidate's `config.measured_evidence` (or, for kaggle, its declared
  wiring) names prior attempt evidence; the cheapest measured per-attempt cost
  in it (`wall_seconds` or `measured_gpu_hours`) is that candidate's measured
  cost. Measured costs are a different instrument than claims, so they always
  outrank them: a candidate with real measured evidence beats every candidate
  ranked on declared attach overhead, regardless of magnitude. Among candidates
  with no measured evidence, the cheapest known attach overhead wins; when no
  passable candidate reports either, the declared candidate order decides and
  the record says so. The chosen provider, the reason, which basis decided
  (`ranked_on`), the full preflight panel, a `cost_comparison` (one entry per
  passable candidate: its overhead, its measured cost where it has one, and the
  basis for each), a `cost_basis` string stating that this is *not* a measured
  end-to-end attach cost, and every candidate it refused are written as the
  run's own `training-backend` phase. No figure is fabricated: a provider that
  reports no measured cost is not ranked as if it reported zero. `auto` with no
  passable candidate refuses (`TRAINING_BACKEND_AUTO_UNRESOLVED`) rather than
  dispatching anything.

## Seeing it before spending it

```bash
chowder growth campaign preflight <manifest>        # the declared backend
chowder growth campaign preflight <manifest> --all  # every backend, side by side
```

Without `--all`, prints one JSON document: the declaration, the `auto` record
when auto chose, the panel, the capability matrix, the declared provider's
declared and measured attach overhead, each planned recipe's estimate, and any
refusal -- with `stops_the_run` separating a declaration error the runner
enforces from a hardware fact it only reports (and `recipes_unavailable` saying
why the recipes could not be planned, rather than letting an empty estimate
list read as "nothing to estimate"). Exit status is 0 only when the declared
backend admitted the campaign. No compute starts.

With `--all`, prints one comparison document with a row per executable backend
(`local`, `unsloth`, `kaggle`): each row carries the declaration it was
evaluated under (the declared one when it matches, else one derived from its
measured-evidence wiring), whether it is the declared choice, its panel,
capability matrix, per-recipe estimates (an uncomputable estimate is an
`available: false` row, never a crash), declared and measured overhead,
overrides and notes, and its refusal -- `stops_the_run` is true only for the
provider the campaign actually declared. Construction-time refusals produce a
`REFUSED` row, never a crash. The comparison is read-only reconnaissance and
always exits 0: a provider that would refuse is a row in the report, not an
error in it.

## Preflight panel and per-recipe estimate

`probe_local_panel()` reports, without refusing: accelerator identity,
compute capability and VRAM per device, CUDA/runtime and framework version,
system and available RAM, free disk on the state root's volume, the measurement
method, and warnings (e.g. "no CUDA device is visible"). Device facts come from
the framework; RAM/disk from the operating system (no new dependency). It is a
*device* probe and says so — step timings remain
`campaign_prepare.probe_hardware()`'s measured rung, used by the recipe planner.

`estimate(manifest, recipe)` reports the per-recipe memory picture:

- model parameter count derived from the base model's own `config.json`
  (standard decoder layout) and the template's quantization;
- LoRA adapter and optimizer estimates (adapter-only state for LoRA, full
  optimizer state for a full fine-tune);
- activation approximation from batch × sequence × hidden × layers;
- total, the largest sequence length that fits the visible VRAM with headroom
  (`safe_sequence_length`), and whether offload is required
  (`requires_offload`);
- the expected training strategy from the declared template (e.g.
  `qlora (peft/transformers/4bit)`);
- and `available=False` with a reason when the material to compute it is not
  declared — an estimate nobody can compute is reported, never invented.

Every probe is injectable (`probes={provider: callable}`), which is how the
admission path is proven on a machine with no accelerator.

## Providers

### LOCAL

The production CLI through `SubprocessTrainingFn`
(`campaign_runner.build_local_training_fn`). Admission is
`check_growth_envelope` against the manifest's per-recipe ceilings — the same
envelope the remote path uses. Capabilities declare LoRA/QLoRA/bf16/fp16,
gradient accumulation, gradient checkpointing, checkpoint resume,
activation-offload / optimizer-tiering / frozen-layer-streaming, successive
halving and search as supported (they are Chowder's own mechanisms, resolved by
the transformers executor). An explicit `config.device: cpu` is honored as a
deliberate operator override, recorded on the result, and noted as unscatterable
device-time.

### UNSLOTH

First-class provider over the same execution path: the template's
`engine: unsloth` selects the isolated executor in its own environment. The
capability matrix declares, in code, where each decision lives:

- `supported`: learning_rate, lora_rank, lora_alpha, target_modules,
  gradient_accumulation, quantization, max_steps, batch_size,
  lr_scheduler_type, warmup_steps, warmup_ratio, max_length, seed.
- `data-layer`: replay_mix (declared through `backend.replay`).
- `model-hardware-dependent`: full_finetune.
- `supported`: checkpoint_resume — `resume_from_checkpoint` is resolved into
  `UnslothPeftRunSpec`, and the executor refuses a missing checkpoint, a changed
  dataset or a changed environment against its own
  `chowder-unsloth-checkpoint-manifest.json`. Its acceptance basis is named in
  the matrix row itself: `tests/test_unsloth_peft.py`'s mocked-worker resume
  suite, with real-CUDA acceptance still outstanding.
- `capability-dependent`: custom_objective.
- `refused`: activation_offload, optimizer_tiering, frozen_layer_streaming —
  mirroring `UnslothPeftRunSpec.from_resolved_config`, refused *before*
  training, never a silent no-op.

Chowder stays authoritative for promotion, benchmark selection, contamination,
retention, campaign stopping, identity and evidence interpretation; the engine
only executes the declared intervention, and the isolation boundary is
preserved.

### KAGGLE

Re-exposed, not re-implemented: `KaggleTrainingFn` +
`KaggleComputeBackend` over the already-committed `kaggle_campaign` /
`kaggle_kernel` / `kaggle_payload` wiring, with the same
`check_growth_envelope` admission and the same evidence vocabulary. When no
injected backend/transport and no usable `kaggle` CLI is available, preflight
reports it honestly:

```
KAGGLE_BACKEND: IMPLEMENTED
LOCAL_VERIFICATION: PASS
REMOTE_PRODUCTION_VERIFICATION: BLOCKED_BY_OPERATOR_CREDENTIAL_OR_QUOTA
```

A pulled artifact takes the same admission, contamination, evaluation and
lineage path as a local one; the capability matrix refuses `quota_exemption` and
`interactive_session`.

## The uniform outcome

`UniformAttemptOutcome` wraps `compute_backend.AttemptOutcome` and adds backend
+ version, candidate and intervention ids, parent checkpoint, training metrics,
wall time, warnings and evidence references. `uniform_outcome_from_evidence()`
maps a `TrainingFn`'s evidence mapping onto it, classifying failures with
`attempt_failure.classify_failure` — so a settlement refusal is
`BUDGET_EXHAUSTED` on every backend, and a backend cannot classify its own
failures differently. `to_evidence()` renders the whole thing in the vocabulary
`cycle`, `attempt_failure` and `compute_cost` already read.

## Where the backend is recorded

The declared backend is written down twice, both before/with the work rather
than reconstructed after it:

- the campaign run record's own `training-backend` phase carries the
  declaration, the provider, the trainer, the preflight panel, whether the
  refusal (if any) was enforced, and the `auto` selection when auto chose; a
  caller-supplied executor is recorded as `executor: "caller-supplied"` rather
  than mislabelled as the declared provider's;
- every attempt's evidence carries a `backend` block (provider, trainer,
  declaration, version) stamped by the runner, which is what reaches the
  `attempts` summary in `campaign-run.json`, and what
  `UniformAttemptOutcome.to_evidence()` renders for a single attempt.
- each generation's lineage ledger entry carries a `backend` block: the
  declared declaration (or `{"provider": "undeclared"}` for entries that
  predate backend provenance), the provider that actually ran
  (`caller-supplied` when a caller supplied the executor rather than the
  declared provider's), the trainer, and the `auto` selection (with its
  `ranked_on` basis and full `cost_comparison`) when auto chose. A reader can
  see which generator produced a generation without joining the run record.

## What this cut does not do

- No real golden-path campaign through each backend yet (needs real CUDA /
  Kaggle quota). Unsloth checkpoint-resume is `supported` on the strength of the
  mocked-worker acceptance suite, not a real-CUDA run.
- No UNSLOTH memory panel: `estimate()` reports it as unmeasured rather than
  guessing; the executor's own refusals remain the control.
- The measured-cost mechanism is implemented and contract-verified, but no
  real attempt evidence exists yet: every measured figure in a cost comparison
  today comes from test fixtures. The first real CUDA/Kaggle runs are what turn
  `measured_evidence` from a contract into a basis auto has actually ranked on.

## Verification

- `tests/test_growth_training_backends.py` (77 tests, no GPU, no network):
  declaration parsing and refusals, panel shape and honesty, model-config-derived
  estimates, capability matrix contents *and* its drift guard for every backend
  (Unsloth, local and transformers: each supported mechanism must reach the
  executor's spec or the runner machinery that executes it; each refused knob
  must actually raise), envelope admission, AUTO selection/refusal/tiered cost
  ranking, uniform outcome mapping and classification, attempt evidence
  collect/verify, resume semantics, the preflight report, the compare-all
  report (report-level and end-to-end through the CLI's `--all` flag), and the
  Kaggle built-from-the-manifest path.
- `tests/test_growth_campaign_runner.py` extends lineage: the generation ledger
  entry names the declared backend, the auto choice with its basis, and a
  caller-supplied executor as such.
- Existing growth + worker-env suites: 970 passed (940 at the previous cut +
  30 new).
- `chowder growth campaign preflight <manifest>` exercised through the CLI on a
  real manifest (admitted on this CUDA machine; `recipes_unavailable` reported
  the two undeclared plan inputs rather than printing an empty estimate list).
