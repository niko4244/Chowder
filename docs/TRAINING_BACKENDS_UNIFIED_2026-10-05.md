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
  - `local` / `unsloth`: `device` (`auto | cuda | cpu`).
  - `kaggle`: required `prepared_path`, `repository`, `commit_sha`, `entry_point`,
    `mounts`, `attempts_root`, `timeout_seconds`, `accelerator`, `payload`
    (`{"kind": "corpus-training", ...}` or `{"kind": "command", "command": [...]}`),
    and optional `owner`, `input_paths`, `model_commit`, `pip_extras`,
    `projection_tolerance`, `declared_quota_ceiling_gpu_hours`.
  - `auto`: `candidates` (the ordered provider list to try; default
    `local, unsloth, kaggle`).
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
- **AUTO records its choice before compute.** An `auto` declaration resolves to
  the first candidate whose preflight passes; the chosen provider, the reason,
  the full preflight panel and every candidate it refused are written as the
  run's own `training-backend` phase. `auto` with no passable candidate refuses
  (`TRAINING_BACKEND_AUTO_UNRESOLVED`) rather than dispatching anything.

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
- `verify`: checkpoint_resume (the executor's own
  `chowder-unsloth-checkpoint-manifest.json` is the check).
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

## What this cut does not do

- No real golden-path campaign through each backend yet (needs real CUDA /
  Kaggle quota).
- No UNSLOTH memory panel: `estimate()` reports it as unmeasured rather than
  guessing; the executor's own refusals remain the control.
- No AUTO policy scoring beyond "the first preflight-passable candidate, in the
  declared order".

## Verification

- `tests/test_growth_training_backends.py` (35 tests, no GPU, no network):
  declaration parsing and refusals, panel shape and honesty, model-config-derived
  estimates, capability matrix contents, envelope admission, AUTO selection and
  refusal, uniform outcome mapping and classification, attempt evidence
  collect/verify, resume semantics.
- Existing growth suites (`test_growth_campaign_runner`, `test_growth_campaign`,
  `test_growth_kaggle_*`, `test_growth_campaign_prepare`,
  `test_growth_campaign_readiness`): 137 passed.
