# P11 rung 2 — small-CUDA router qualification result (2026-09-14)

Judged against the committed preregistration
[`P11_CUDA_PREREG_2026-09-13.md`](P11_CUDA_PREREG_2026-09-13.md), pushed as
`ea0458d` **before** the implementation and the run. Verdict: **QUALIFIED —
all seven thresholds pass on measured evidence.** The router device path
(load, train, freeze, publish, evaluate, account) now has evidence on a real
accelerator; `cuda` joins `cpu` in `QUALIFIED_DEVICES` behind the measured
preflight contract. No champion promotion, no quality claim, no 9B
authorization follows from this rung.

## Run

- Route: `chowder project-validate` then `chowder train` on
  `router-project-cuda.json` (derived from the hash-verified CPU builder;
  the only recipe change is `device: "cuda"`), inside the isolated
  worktree at `Chowder-p7-evidence-fixes-20260912` (uncommitted head at run
  time = `ea0458d` + guard-lift implementation, `QUALIFIED_DEVICES = ("cpu", "cuda")`).
- Hardware: NVIDIA GeForce RTX 5060 Ti (16 GiB, device 0). Desktop/compositor
  processes were using ~11.6 GiB; 15.9 GiB was reported free to the worker's
  measured preflight at run time (WDDM accounting differs from nvidia-smi's
  process table). No competing training/eval job ran before or during.
- Exit: `train` exit 0; candidate outcome `succeeded: true`; gate verdict
  **rejected** (measured, not forced).

## Thresholds, judged from durable artifacts

| # | Threshold | Measured result | Verdict |
|---|---|---|---|
| 1 | Preflight measured, not configured | `free_memory_bytes` 15,906,897,920; `step_cost_probe.measured: true` (`step_seconds` 0.6080, `peak_step_bytes` 17,077,760, `projected_oom: false`, `would_exceed_budget: false`) — taken before optimizer step 1, state restored before training | **PASS** |
| 2 | Trainability on device | `trainability.ok: true`; both intended gates (`model.layers.{0,1}.mlp.gate.weight`) `trainable: true`, gradient state `grad-nonzero` on all 12 observed steps, updates observed on all 12; coverage `ok: true`, expected 2, missing `[]` | **PASS** |
| 3 | Frozen unchanged | `frozen.ok: true`, `changed: {}`, 23 frozen tensors, `digest_strategy: ["full"]` (no sampled-vs-full ambiguity) | **PASS** |
| 4 | Limits honoured | 12/12 steps, 384/384 tokens, `stop_reason: "max_steps"`, 24 samples | **PASS** |
| 5 | Gate recorded with ledger | `baseline → passed`, `router-pilot → rejected`; base holdout loss 4.174509048461914 vs candidate 4.174547910690308 (Δ +3.89e-05 — worse, honestly rejected); ledgers present on both arms | **PASS** |
| 6 | Accounting | `active_accelerator_count: 1`, `visible_accelerator_count: 1`, measured `peak_vram_gb_by_accelerator: {"0": 0.015971}` (~16.3 MiB — tiny model, real measurement); training ledger measured phases: `model_load`, `steady_state_steps`, `checkpoint_publication`, `closeout`; candidate eval ledger measured: `model_load`, `baseline_generation`, `candidate_generation` | **PASS** |
| 7 | Identity chain | `worker_kind router_healing_worker_result.v1`, `spec_digest` bound, `source_identity` (source_root = isolated worktree, source_sha256 over 138 files), base content identity `903331fbdae1…` verified before loading | **PASS** |

GPU-hours attributed: **0.000861** (≤ 0.1 ceiling).

## Gate detail (scientific result, not a qualification input)

The gate's accept/reject is whatever the measurements produce (prereg
threshold 5). The candidate trained on 12 steps at lr 0.05 and scored
**worse** than the untouched base by 3.89e-05 holdout loss; dead experts rose
1 → 3. The tiny random-init MoE has no learnable structure for 12 steps to
find, and the run says so. This matches the CPU pilot's qualitative result
(+3.89e-05 there too — same magnitude, same direction).

## Provenance notes

- Preregistration commit `ea0458d` precedes the implementation commit and the
  run (GitHub-timestamped order preserved).
- The builder (`build_tiny_router_pilot.py`) was re-hashed before use:
  `bfa954eec0f838fc70ae66d07fc67c529a552b8af0e7a6e67329a36eca2a22e6`, matching
  the preregistration. Base model and corpora are the same hash-verified
  artifacts the CPU pilot used; the CUDA project builder only rewrote
  `work_dir`, `registry_path`, project name, and `device`.
- One stale seam was found and fixed during this rung: `project.py` still
  carried its own CPU-only refusal with the retired "digest is not
  device-safe" justification. It now defers to the backend's
  `QUALIFIED_DEVICES` (single source of truth, failing-test-first).
- Reproduction: `build_cuda_router_project.py` → `project-validate` →
  `train`; judge with `judge_against_prereg.py`. The `work/` directory keeps
  the registry, worker results, payloads, and logs.

## Interpretation rules honored

- The rung is qualified by a **completed run** meeting all thresholds; the
  gate rejection is a measured scientific outcome, not a rung failure.
- No threshold was reinterpreted after the fact; nothing was loosened to
  pass.
- What this does NOT establish: model quality, the 9B pilot's fit, champion
  promotion, or release permission. Rung 3 (the 9B-derived pilot) still
  requires its own preregistration and a measured memory/cost preflight on
  this hardware.
