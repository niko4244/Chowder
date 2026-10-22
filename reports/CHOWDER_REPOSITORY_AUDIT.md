# Chowder repository audit

> Audit snapshot: 2026-10-22 UTC. Main 0bc566c5db052b90e3aa1223fef94445d1045dc5; PR #268 24f2f8865b215b477796364b1ec30e944d5ec087; recovery PR #269 e3f0fbd77bc679ea9a856461294cd26b40316562. GitHub code and published evidence were inspected; local Windows workspace inaccessible (Chat On Steroids Core root /strata invalid). Source claims below are **historical reports**, not rerun hardware measurements. No model weights or protected local evidence were accessed in this audit.

## Verified Git state

- Main HEAD `0bc566c5` dated 2026-10-20; compare the source against the [main commit](https://github.com/niko4244/Chowder/commit/0bc566c5db052b90e3aa1223fef94445d1045dc5).
- Open PRs at snapshot: #269 recovery journal, #268 MoE int8, #227 watchdog cadence, #226 metric semantics, #214 unvetted literature drop, #210 campaign controller split, #203 rescued workspace, #202 teacher-free plan, #201 teacher-free pilot, #200 Kaggle dispatch. One GitHub issue #228 flags the literature drop. Historical PR test counts are not fresh CI proof.
- The latest local working tree, live processes, attached storage, complete checkpoint bytes, external Hub policy documents and machine-specific state cannot be inspected: approved /strata root changed. No claim of a clean local git tree or complete model package is warranted.

## Implemented on main

- Transformers/PEFT training, independent evaluation, resumable checkpoint contracts, QLoRA and previously measured Unsloth pilot; real-model CI job exists (`README.md`, `pyproject.toml`, `.github/workflows/ci.yml`).
- Budgeted growth loop with CLI/TUI facade, immutable state and refusal gates (`src/chowder/growth`, `docs/AUTONOMOUS_GROWTH_LOOP.md`).
- REAP observation/pruning/eval/serving scripts; BF16 layer-bulk placement merged in #264 (equivalence tests; GPU speed remained unspecified in merged record).
- Gen-1 training mix builder supports the optional 250-row selfoss_code source and drops over-2048-token rows rather than truncating them; documentation corrected on this audit branch (`docs/MOE_GEN1_MIX_V1.md`).

## Branch work and defects

- #268 holds the int8 loader, exact shape/quantization tests, live census, real-card measurements, and witness mechanisms. Audit commit 24f2f886 additionally validates flags and dataset/token batches *before* an expensive full-checkpoint load; CPU regression CI pending. Not merged; its gate table lacks gradient, smoke, router-balance and WDDM operator evidence, and its recorded sanctioned gate belongs to pre-default-retention code. See `reports/MOE_INT8_16GB_QUALIFICATION.md`.
- #269 addresses an actual Gen-2 hard termination; its earlier parser could mistake an older terminal marker for a later interrupted run. Added fail-closed parser, atomic-verdict fsync and tests on PR branch at `e3f0fbd77bc679ea9a856461294cd26b40316562`. New CI was in progress when inspected. No automated continuation was added.
- #226 and #210 target older bases and must be rebased/reviewed; do not mass-merge. #214 requires human review under the literature curation policy.
- Gen-2 run 10 had a late ancestor raw-artifact refusal after historical 2.16 GPU-h; amendment 24 and merged #266 describe the evidence-location repair and earlier readiness verification. Gen-2 run 11 was terminated externally, not certified.

## Scope and trust boundary

Two model programs are distinct. Primary Qwen3.8 Native Sparse forbids distillation; REAP-MoE permits declared Fable-derived data only under its own charter and requires Fable-free controls while legal decision N2 is open. No checkpoint, adapter, judge record or metric can be transferred between lineages for certification (`docs/REAP_MOE_LINEAGE.md`).

## Assessment

Platform foundations **PARTIAL** (implementation and historical real tests; fresh integrated verification missing); MoE int8 **PARTIAL**; complete custom REAP model **BLOCKED** by unfinished frozen gates, unverified package and budget feasibility; production-grade autonomous continuation **BLOCKED** by unqualified recovery/resume and safety evidence.


## Local verification addendum — 2026-10-22 (supersedes earlier GitHub-only findings)

The approved Windows workspace became accessible. The original main clone is a dirty rescue worktree and was not modified or reset. Isolated detached worktrees were created for testing. The physical REAP160 checkpoint at G:\llm-cold\bf16\Qwen3.6-REAP160 exists: 26 shards, 1,026 index-matched BF16 tensor headers, 46,039,444,192 payload bytes and 46,039,573,000 shard-file bytes; no header offset/index defects. Locally hashed config SHA256 matches 15413a791b9eff423755743bb05b997c93f6ae6f4ce02a1966c9bdfdc2cdc3b9. Full weight bodies remain unhashed.

Existing frozen experiment artifacts were reconciled. REAP128 fails GSM8K Gate 1 (paired bound -0.037807 versus -0.03), triggering the registered REAP160 fallback. REAP160 historical Gate 0, GSM8K, HumanEval and negative-control records PASS. A separate speed-resolution record reports 3.8523x Q4 speed versus dense with zero experts offloaded; all 12 raw comparison input files matched their recorded SHA256 hashes. This supports the fallback but does not constitute a new immutable Gen-0 freeze or a fresh CUDA run.

The local 24-step int8 witness SHA256 20e466d5362b8492fb9b55c312a57486283e641d46844e41c3ae4a85799617d7 records gradient PASS (19/19 LoRA B leaves), smoke PASS (mean loss 1.421 -> 0.551), router-balance PASS (worst entropy change 1.8%), WDDM spill NOT MEASURED. PR #268 revision 108f9322 fixes both the Markdown |g| table parsing defect and a witness --check that wrote despite claiming read-only; the real witness was imported through the validated generated-table mechanism. 7/8 placement gates now register measured values, with only WDDM unmeasured, and campaign admission still correctly refuses. Current-retention VRAM remeasurement and the 2.5 GPU-hour budget remain open.

Fresh Windows Python 3.11.9 focused CPU tests: recovery 90 PASS; int8/preflight 54 PASS; witness/gate/preflight 75 PASS; targeted Ruff clean. An initial 30-failure probe with artificially hidden CUDA was traced to a TorchAO import-time device query and succeeded when rerun with CHOWDER_NO_GPU=1 but CUDA device visibility restored, without GPU training.

The Gen-1 mix file checks passed against its current template-validated, 2048-token manifest: Fable 767 rows SHA256 384b6f10ae579af10ce36cbbe5278716b681d3223beab7343742b51c1857024d, Fable-free control 647 rows SHA256 681b2aa957d042028db8fa8710be6538b23f6563e088b9f752410bf5da174c9e. License decision N2 remains open. The card had approximately 12 GiB occupied and 4 GiB free, so no expensive CUDA tests or promotions were authorized or run.
