# Chowder repository audit

> Final audit 2026-10-22. Main 0bc566c5; MoE PR268 184291f; recovery PR269 cf54480 with local 82-test verification; REAP PR271 73f1593. GitHub and read-only Windows workspaces inspected, 39 model files independently SHA-256 verified twice. No GPU training was started in this audit. Frozen gates and required approvals remain binding.

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

- #268 holds the int8 loader, exact shape/quantization tests, live census, real-card measurements, and witness mechanisms. Audit commit 24f2f886 additionally validates flags and dataset/token batches *before* an expensive full-checkpoint load; CPU regression CI pending. Not merged; gradient, smoke and router-balance are now imported PASS, while WDDM operator evidence is still missing and the sanctioned VRAM measurement belongs to pre-default-retention code. See `reports/MOE_INT8_16GB_QUALIFICATION.md`.
- #269 addresses an actual Gen-2 hard termination; its earlier parser could mistake an older terminal marker for a later interrupted run. Added fail-closed parser, atomic-verdict fsync and tests on PR branch at `cf544801af525826c5bd507581b27580feff23f2`. New CI was in progress when inspected. No automated continuation was added.
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

## Final local reconciliation (2026-10-22)

Full cryptographic inventory: 39 physical REAP160 files, including 26 BF16 shards, hashed twice independently with matching SHA-256 for each file. Package inventory digest: 06aaa74bab80d6a025e25428eacfe8006d0d43fad2406370e563ff1a0c2f1c36. Per-file hashes are in artifacts/custom_moe_model_manifest.json; a sharded checkpoint has no claimed monolithic SHA-256.

Frozen REAP observer D0 locally PASS (40 layers, perplexity 10.646096, threshold less than 20). REAP128 v1 Gate1 FAIL; REAP160 v1 quality gates PASS and historical separate Q4 fit/speed proof. Amendment 4 requires REAP128_v2 scoring before Gen0 can be selected. PR271 adds fail-closed D0/cardinality/duplicate and ordered-arm disclosure; 15 tests PASS. No v2 model/score was located.

Local int8 witness imported 24-step gradient, loss-smoke and router PASS; seven of eight gate rows measured; WDDM operator no-spill remains unmeasured. Historical best 27.48s/step implies 7.63h per 1000 steps versus frozen 2.5h ceiling. Current retention-default VRAM remeasurement remains outstanding.

Additional P1 findings: automatic safe resume remains unavailable, promotion ledger commit can precede terminal verdict, failed evaluation charges need stronger reconciliation, and model placement gate identity currently uses config/size rather than weight contents. Recovery work includes budget fallback and cross-process writer locking with focused tests but awaits complete CI/merge. No expensive GPU run, Gen1 training or promotion occurred.

## Independent integrated engineering reconciliation (2026-10-22)

Current main is 0bc566c5, unchanged. PR #268 at 184291f, #269 at cf54480 and #271 at 73f1593 each passed their original six-job GitHub CI matrix. Their code merges without conflicts on draft integration PR #273. The integration branch f099d3a additionally preserves previously published interrupted campaign verdicts, charges failed evaluation work via independently valid reported cost when available or conservative measured wall time when not, and implements a durable PREPARED/COMMITTED promotion transaction with lineage/terminal digest verification. Recovery is also bound to the complete immutable campaign declaration digest so changed preregistrations cannot replay an old promotion. Crash/manifest integrity targeted tests: 8 PASS. All changes are in open review branches, NOT merged to main.

Draft stacked PR #272 at c2202c1 provides content-bound MoE SHA inventories, enrollment CLI, and fail-closed verification in placement, witness and campaign admission. Lightweight repeat checks are filesystem identity checks plus semantic-file hashes; not a security guarantee against adversarial in-place rewrites with restored metadata. Integrated CPU inventory/gates/witness tests: 64 PASS. The full 39-file REAP160 SHA inventory from two earlier independent reads remains verified; the new gate-specific enrollment is being run as a separate non-destructive operation. Do not assume the old config-only gate table satisfies this new contract.

Fresh integrated CPU testing: CLI/TUI/budget/adversarial suite 151 PASS. The original broad integration run had 187 PASS and one source-identity failure during concurrent edits; the single failed real CPU identity test passed on a stable pinned commit in 74.62 seconds. A second pinned broad run had 188 PASS and two accounting-assertion failures. The code now preserves valid reported evaluation cost and charges conservative wall fallback only when actual cost is missing; the four relevant targeted assertions passed. Full integration PR #273 CI at the latest pushed SHA remains to be confirmed. Test suites overlap and pass counts must not be added as unique tests.

The frozen v2-Ornith calibration source at C:\llm-work
eap\calib-v2-ornith\calibration.jsonl is verified: 717 rows, SHA-256 516f2e4a931fd3f025c6206e1f7e96b53ed68801ec503cb3d3367ca43f540c9f. No REAP128_v2 model/evaluation was found, so amendment-4 Gen-0 selection remains BLOCKED despite historical REAP160 v1 quality and Q4 speed. WDDM spill is still unmeasured. Present-code retention-default VRAM has not been requalified. Historical 27.48 seconds/step implies 7.63 GPU-hours for 1000 steps against the 2.5-hour budget. Fable/Ornith data-rights qualification and human promotion approval remain open.

Full automatic interrupted-training continuation is not implemented: an incomplete journal refuses before re-spending attempted GPU work, and a completed transaction is recovered only with matching terminal and ledger evidence. Model promotion and expensive GPU experiments have not been performed.

## Completed one-time model SHA enrollment (2026-10-22)

The read-only local enrollment completed using the new PR272 code and a NEW file outside the checkpoint directory: C:\Users\nikma\_audit\reap160_content_inventory_verified_20261022.json. It measured 36 actual model-loading payload files, including all 26 BF16 weight shards, and matched every digest against the two independent prior hash inventories. Content SHA256: a1ba0e434eea76fdfc4b85843ab90d7957fa2ffd08ac319c2d2f8aed3ae4aa92. Inventory SHA256: ece6b1c359c357450539826d4183edc5b80182ca6aeaa703d7ce091b9c142b41. A subsequent stat/semantic-file identity check PASSED without re-reading every weight shard. This is a **verified input artifact**, not a substitute for fresh VRAM/WDDM gates or LoRA training. The remaining three package metadata files, LICENSE, dataset-metadata.json and reap-prune.json, are still covered by the existing 39-file release-manifest hashes but are not part of the loader gate payload inventory. Full GPU acceptance remains BLOCKED.
