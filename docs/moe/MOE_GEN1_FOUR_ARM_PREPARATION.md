# Gen-1 MoE: four executable Fable/control × learning-rate trials

The frozen Gen-1 project template declares **two** learning-rate recipes. The
production candidate search varies *learning rate*, not *training data*; its
`dataset_manifest` is explicitly an **unsafe search field**. Therefore changing
Fable versus control inside a single campaign's search would misrepresent what
was preregistered. The tool below prepares **four independent Chowder projects**,
not four candidates eligible for a single automatic promotion verdict.

## Required evidence before preparing any project

1. A resolved **production** `campaign_prepare` project template, including a
   fixed measured parent baseline, a concrete REAP base, and an actual GPU-hour
   ceiling. `docs/moe/MOE_GEN1_CAMPAIGN_TEMPLATE.json` still contains TODOs and
   **is not** a project file.
2. A clean `build_moe_gen1_mix.py` result: `manifest.json`,
   `train-fable.jsonl`, and `train-control.jsonl`.
3. A **READY** `moe_gen1_train_preflight.py --max-length 2048` report for the
   exact mix and tokenizer. The builder's historical 4096-token screening does
   **not** satisfy this requirement. A rejected report is a blocker, not a
   reason to disable the screening or raise the frozen training length. The
   receipt must include `tokenizer_assets_sha256` from the current preflight;
   the preparer checks that the actual local `backend.base_model` carries the
   identical tokenizer and chat-template files used in screening.
4. For `cpu-transient-grad-int8`, all recorded placement gates, including the
   WDDM operator gate, must be qualified **for the exact base checkpoint**.
   Preparation uses the authoritative placement gate; it cannot waive it.
5. Every prepared project must declare a **finite, positive `goal.gpu_hour_budget`
   no greater than 2.5 GPU-hours per trial**, and its experiment estimate must
   fit inside that same budget. The project's `config.goal_lifecycle` must be
   explicit: `mode: bounded` with finite numerical target(s), or the existing
   `mode: legacy_unbounded` **only for direction-only assessment goals**. The
   latter is a native objective-assessment protocol (not an exemption from
   GPU-hour budgets). This preparation never inserts the legacy mode, changes
   an objective threshold, or invents a budget. A missing/unknown lifecycle
   declaration is a refusal.
6. The concrete production template must declare **exactly 1,000 optimizer
   steps**, batch size 1, accumulation 1, gradient checkpointing enabled,
   BF16 and 2,048 tokens. A measured optimizer-step/load receipt for the
   same `backend.base_model`, expert placement, and sequence length is required.
   Both `config.json` and `model.safetensors.index.json` SHA-256 digests must
   match current local files. These are inexpensive **metadata** identities;
   they are not full model-weight hashes. The preparer uses the slowest
   measured step × 1,000 plus measured model loading, with a separate
   evaluation reserve, and checks the trainer timeout. It refuses an estimate
   that cannot fit the unchanged 2.5 GPU-hour cap.

The existing int8 A/B receipt at `C:\llm-work\moe-gates\ab-int8-seq2048.json`
does **not** qualify these trials. It predates the metadata identity fields,
and its recorded 27.48 seconds/step implies at least 7.63 GPU-hours for
1,000 steps **before loading and evaluation**, well beyond the 2.5-hour
ceiling. The source base and current GPU must be remeasured if a different
placement or host is proposed; copying favorable figures into a receipt is
not a measurement. The historical campaign-prepared default of 30 steps
also fails the registered 1,000-step check.

```powershell
python -B chowder_batch/moe_gen1_train_preflight.py `
  --mix C:\llm-work\train-sets\moe-gen1-v1 `
  --tokenizer G:\llm-cold\bf16\Qwen3.6-35B-A3B `
  --max-length 2048 `
  --report C:\llm-work\train-sets\moe-gen1-2048-admission.json

python -B chowder_batch/moe_gen1_four_arm.py `
  --project-template C:\llm-work\gen1-prepared\project-template.json `
  --mix C:\llm-work\train-sets\moe-gen1-v1 `
  --train-preflight C:\llm-work\train-sets\moe-gen1-2048-admission.json `
  --timing-receipt C:\llm-work\moe-gates\measured-gen1-2048.json `
  --out C:\llm-work\gen1-four-independent-trials
```

The timing-receipt path above is an **example destination for a new real
measurement**, not evidence that this file exists or qualifies. Supported
generic receipts declare `schema: chowder.moe-gen1-step-timing.v1`,
`base_model`, `base_config_sha256`, `base_index_sha256`, `placement`, `gpu`,
`seq: 2048`, `batch_size: 1`, `gradient_accumulation_steps: 1`,
`gradient_checkpointing: true`, `load_seconds`, and at least three measured
positive `step_seconds`. The existing int8 A/B phase schema is also parsed,
but is refused without metadata identity and qualifying measured cost.

Preparation performs **no** GPU training and does not load model shards. It
refuses an existing output root. On success, `four-arm-plan.json` contains four
independently verifiable `chowder project-validate` and `chowder train`
commands. Run those commands **only** after reviewing the real budget and
material provenance. The 2048-token horizon, LoRA preset, base, placement,
evaluation protocol, measured baseline, objective lifecycle, and independently
enforced per-trial GPU-hour budget are preserved in every project.
Admission establishes a necessary measured cost condition. An unrun 1,000-step
training job can still incur additional overhead, so preparation never claims
that a complete GPU run is already proven to fit.
Every project uses a distinct dataset SHA, experiment ID, working directory,
and registry; the two LRs are `5e-5` and `1e-4` in both data arms.

The **2.5-hour cap applies separately** to each of four trials; it is NOT
evidence that a four-trial total/campaign ceiling has been measured or settled.
The four trials are **not** proof of a successful Gen-1 campaign. Their
checkpoint/adapter reload, execution-verified evaluations, cross-arm comparison,
cumulative budget settlement, lineage ancestry, frozen judge, and N5 owner
approval remain independent mandatory evidence. Do not use the project files
to bypass campaign admission or promote any candidate.

Checkpoint provenance is fail-closed: a pre-existing `trainer/checkpoint-*`
without its original `chowder-checkpoint-manifest.json` cannot be claimed by a
new training run. When checkpoints already exist, a manifest cannot gain
previously absent recipe, dataset, parent-adapter, base or expert-arithmetic
bindings retroactively. Use a fresh trial directory instead; merely rewriting
an old manifest would not prove the optimizer state came from those inputs.
