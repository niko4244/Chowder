# Custom Qwen MoE completion register

<!-- CHOWDER_LATEST_VERIFIED_20261022 -->
> **Latest verified review:** PR #277 `2be95ceb28afdc4aa0d4b4787bc70c3c801e1194` (PENDING). See Gen-0/Gen-1 addendum. Older evidence retained under original revisions.

> Final audit 2026-10-22. Main 0bc566c5; MoE PR268 184291f; recovery PR269 cf54480 with local 82-test verification; REAP PR271 73f1593. GitHub and read-only Windows workspaces inspected, 39 model files independently SHA-256 verified twice. No GPU training was started in this audit. Frozen gates and required approvals remain binding.

## Scope and lineage

The custom MoE in active engineering is Qwen3.6-35B-A3B REAP pruned to 128 routed experts, with 160 only if the frozen fallback criteria demand it. The separate Qwen3.8 Native Sparse Program retains its prohibition on distillation and its own checkpoint/ancestry; no evidence or adapters are transferred. Sources: `docs/REAP_MOE_LINEAGE.md`, `docs/REAP_QWEN36_PREREG.md`.

## Verified via committed code or PR evidence

- Main contains REAP observation/pruning, eval and serving programs, full REAP promotion judge specification, Gen-1 training-data provider, BF16 layer-bulk transfer code, and preservation of model/evidence lineage.
- #268 contains int8 fused expert loader and quantization, placement instrumentation, real REAP160 26-shard census, 20-step RTX 5060 Ti training history, throughput A/B, router/gradient/smoke witness collectors and Kaggle capacity/real-load attempts. Audit commit 24f2f886 adds early calibration/flag validation; full CI not yet reported complete.
- #268 reports 40 expert layers, top-8 routing, approximately 18.82 GiB int8 expert grid plus scales and 4.55 GiB BF16 non-expert parameters, but live full-file hashes cannot be verified here.

## Incomplete for finished model

1. Pre-registered choice among REAP128, permitted REAP160 fallback and dense/random control lacks fully inspected raw baseline measurements in this session.
2. The current int8 default-on retention config is not independently certified by the recorded sanctioned table; four required witness/operator gates remain unmeasured.
3. Real training throughput misses the existing 2.5 GPU-hour/1000-step budget by approximately 5.13 hours even before first-touch overhead.
4. Exact file hashes, tokenizer, complete model card, serving Q4 artifact, functional local loader reproduction and deployment tests are inaccessible/unverified.
5. Gen-0 trusted ancestor has not been proven frozen here. Gen-1 Fable/control experiments, protected benchmark results and signed promotion approval have not been established.

## Completion tests

- Hash all checkpoint shards/tokenizer/config; reinstantiate and compare architecture/parameters, routing, outputs, adapter save/reload and Q4 inference identity.
- Pass every frozen pruning and placement gate on same artifact/code/config/hardware; preserve exact result JSON and code SHA.
- Complete declared contamination audit for both Fable and Fable-free arms; resolve N2 or retain mandatory control and release restrictions.
- Execute authorized bounded Gen-1 under feasible preregistered compute, rescore raw benchmark items, use frozen judge, retain required human approval.

**Status: BLOCKED, not failed as an algorithm.** Insufficient approved compute/access and unpassed or unmeasured acceptance gates prevent a truthful final model release.


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

## Verified Gen-1 mix shortfall (2026-10-22)

The actual file hashes and ordered-subsequence relationship were independently verified: train-control has 647 rows, train-fable 767 rows, and all 647 control rows appear in the same order in the Fable arm, with exactly 120 extra Fable rows. Manifest reports 2,048-token screening and template validation. However the Nemotron agentic source retained **0 of 120** planned rows, OpenThoughts-3 retained **9 of 150**, and Ornith retained **38 of 39**. The current Fable-free control contains no Nemotron tool-use training rows. This is an explicit registered shortfall, not silently repaired by adding unapproved examples. A Fable-vs-no-Fable comparison would also differ in training exposure; do not market it as a dose-matched comparison. Ornith license metadata is undeclared and Fable N2 remains unresolved, so release/data-rights qualification is BLOCKED even though contamination screening dropped 259 rows and all output hashes match.

## Verified deployable GGUF artifact (2026-10-22)

Existing REAP160 Q4_K_M GGUF: C:\llm-work\reap\gguf\reap160-q4_k_m.gguf, 13,821,482,880 bytes, independently calculated SHA-256 8c840461752e1343e81bc12ab9b95d422a4b45c0b0048f00938fe882f299f8f5, identical to the previously saved serve.json digest. Its GGUF header is version 3 with 733 tensors and 41 metadata entries. Historical llama-server load-only evidence reports --fit on -np 4 -c 16384 --jinja, all 41/41 layers GPU-resident, zero routed experts offloaded, and 14,275 MiB VRAM. The archived speed resolution reports 3.8523x the dense Q4 comparison. The file and hash are verified now; loading/generation were not rerun in this audit because the RTX 5060 Ti is occupied. This identifies a real inference artifact, but final custom Gen0 selection remains blocked by amendment-4 REAP128_v2 evaluation and training/release gates.

## Independent integrated engineering reconciliation (2026-10-22)

Current main is 0bc566c5, unchanged. PR #268 at 184291f, #269 at cf54480 and #271 at 73f1593 each passed their original six-job GitHub CI matrix. Their code merges without conflicts on draft integration PR #273. The integration branch f099d3a additionally preserves previously published interrupted campaign verdicts, charges failed evaluation work via independently valid reported cost when available or conservative measured wall time when not, and implements a durable PREPARED/COMMITTED promotion transaction with lineage/terminal digest verification. Recovery is also bound to the complete immutable campaign declaration digest so changed preregistrations cannot replay an old promotion. Crash/manifest integrity targeted tests: 8 PASS. All changes are in open review branches, NOT merged to main.

Draft stacked PR #272 at c2202c1 provides content-bound MoE SHA inventories, enrollment CLI, and fail-closed verification in placement, witness and campaign admission. Lightweight repeat checks are filesystem identity checks plus semantic-file hashes; not a security guarantee against adversarial in-place rewrites with restored metadata. Integrated CPU inventory/gates/witness tests: 64 PASS. The full 39-file REAP160 SHA inventory from two earlier independent reads remains verified; the new gate-specific enrollment is being run as a separate non-destructive operation. Do not assume the old config-only gate table satisfies this new contract.

Fresh integrated CPU testing: CLI/TUI/budget/adversarial suite 151 PASS. The original broad integration run had 187 PASS and one source-identity failure during concurrent edits; the single failed real CPU identity test passed on a stable pinned commit in 74.62 seconds. A second pinned broad run had 188 PASS and two accounting-assertion failures. The code now preserves valid reported evaluation cost and charges conservative wall fallback only when actual cost is missing; the four relevant targeted assertions passed. Full integration PR #273 CI at the latest pushed SHA remains to be confirmed. Test suites overlap and pass counts must not be added as unique tests.

The frozen v2-Ornith calibration source at C:\llm-work
eap\calib-v2-ornith\calibration.jsonl is verified: 717 rows, SHA-256 516f2e4a931fd3f025c6206e1f7e96b53ed68801ec503cb3d3367ca43f540c9f. No REAP128_v2 model/evaluation was found, so amendment-4 Gen-0 selection remains BLOCKED despite historical REAP160 v1 quality and Q4 speed. WDDM spill is still unmeasured. Present-code retention-default VRAM has not been requalified. Historical 27.48 seconds/step implies 7.63 GPU-hours for 1000 steps against the 2.5-hour budget. Fable/Ornith data-rights qualification and human promotion approval remain open.

Full automatic interrupted-training continuation is not implemented: an incomplete journal refuses before re-spending attempted GPU work, and a completed transaction is recovered only with matching terminal and ledger evidence. Model promotion and expensive GPU experiments have not been performed.

## Completed one-time model SHA enrollment (2026-10-22)

The read-only local enrollment completed using the new PR272 code and a NEW file outside the checkpoint directory: C:\Users\nikma\_audit\reap160_content_inventory_verified_20261022.json. It measured 36 actual model-loading payload files, including all 26 BF16 weight shards, and matched every digest against the two independent prior hash inventories. Content SHA256: a1ba0e434eea76fdfc4b85843ab90d7957fa2ffd08ac319c2d2f8aed3ae4aa92. Inventory SHA256: ece6b1c359c357450539826d4183edc5b80182ca6aeaa703d7ce091b9c142b41. A subsequent stat/semantic-file identity check PASSED without re-reading every weight shard. This is a **verified input artifact**, not a substitute for fresh VRAM/WDDM gates or LoRA training. The remaining three package metadata files, LICENSE, dataset-metadata.json and reap-prune.json, are still covered by the existing 39-file release-manifest hashes but are not part of the loader gate payload inventory. Full GPU acceptance remains BLOCKED.

## Local model licensing evidence (2026-10-22)

The verified REAP160 checkpoint includes a full Apache License Version 2.0 text in its LICENSE file, but its accompanying dataset-metadata.json identifies the repository licence as "other" and private. These statements are not equivalent to a verified licence grant for every derivative artifact, hosted model distribution, Fable-derived training example, or adapter. Publication rights remain UNVERIFIED and require a documented independent legal/source-rights decision; N2 is not implicitly cleared by the presence of an Apache-2.0 text.

## Integration CI completion (2026-10-22)

Draft PR #273 at exact commit f099d3ac7d7bc09ef071acb6efd09994ede19667 has now completed all six GitHub checks SUCCESS: Linux Python 3.10, 3.11 and 3.12, Windows Python 3.12, lint/build smoke, and real Transformers/PEFT CPU smoke. GitHub reports mergeable, but the branch is deliberately still draft and NOT merged into main. All frozen hardware/model/licensing/human gates remain in force; CI success is not MoE release certification.

## Engineering continuation — 2026-10-22

Main and the dirty rescue checkout remain unchanged. All six CI jobs passed on PR273 at f099d3a. A new DRAFT integration PR277 at 53ebcab includes:

- PR276 at 8393a3b: verified completed-attempt automatic resume, durable receipt and fsynced journal pairing, exact declaration and input identities, checkpoint SHA256, unique GPU-hour settlement, remaining-budget admission and safe search continuation. In-flight unknown-cost attempts and partial checkpoints still refuse. 74 isolated CPU tests passed, including process kill; 219 PASS and 1 SKIP on pinned post-resume integration 405e3a1. Pre-resume combined baseline 301 PASS on 14e52f9.
- PR274 at 7d79586: new frozen Amendment-4 selector, SHA-pinned raw evaluations, cohort sizes 1319 GSM8K / 164 HumanEval / 200 speed, mandatory REAP128_v2 and REAP160 scoring, and separate v2 observer bound to calibration manifest digest and row census. 29 CPU tests PASS. Current output is BLOCKED: REAP128 v1 Gate1 lower bound -0.0378074 versus strict -0.03; the mandatory v2 model/evals are missing. No selection or release approved.
- PR275 at 3e118c2: hardened indexed shard resolver, fail-closed missing/extra/path traversal detection, CPU metadata lookup speed 8.12 to 3.60 ms on 1026 keys and 112.75 to 11.54 ms on 16000 keys. NOT a GPU training-step speedup. 60 CPU tests passed, 1 Windows symlink skip.
- REAP160 model verified 26 BF16 shards, 1026 indexed tensors, 40 layers, 160 experts top-8, hidden 2048/intermediate 512. Six tokenizer/generation assets are byte-identical to the dense Qwen3.6 parent, including tokenizer.json SHA256 5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42.
- Historical 27.48 seconds/step means 7.63 GPU-hours for 1000 steps, above frozen 2.5 hours. CPU-only transfer self-check computes a 10.23 second host-to-device lower bound at historical 5.52 GiB/s versus 9.0-second whole-step target. No new GPU measurements; WDDM, 14.5 GiB current VRAM, v2 and Gen1 remain unqualified.
- Official Qwen3.6 parent is Apache-2.0 (https://huggingface.co/Qwen/Qwen3.6-35B-A3B); local private derived metadata says license other, and Fable/Ornith N2 rights remain unresolved. No deployment or promotion. Qwen3.8 Native Sparse remains separate.

## Current finalization and owner-approval qualification (2026-10-22)

A new review-only integrated branch, draft PR #277, head a0a678dfa13a2d15f2b3176fb668eebbb8335728, combines PR273, independently implemented receipt-verified campaign resume PR276 (8393a3b), strict frozen REAP selection PR274 (amended by b99582f), safe MoE index resolver PR275 (3e118c2), and REAP owner authorization PR278 (ef866b2). This does not merge to main or alter the dirty rescue checkout.

On PR277's initial head 53ebcab the FULL Linux Python 3.11 CI failed four tests with 3485 passing and 223 skipped: three REAP selector tests imported different GateError classes through package and script routes, and one readiness checker reported READY despite the new resume preflight correctly refusing missing declared evaluation_material_path. These reproducible defects are fixed in later revision d6e378a (canonical package import, readiness source-file SHA preflight and updated expected refusal code). A pinned consolidated targeted run at a0a678d passed 152 and skipped one. Full CI and full local suite on a0a678d require completion before final integrated green can be claimed.

REAP N5 human authorization: the frozen lineage charter requires explicit owner approval AFTER an independent local frozen judge. Previously the mechanical PROMOTED decision was written directly into promoted lineage. Draft PR278 introduces fail-closed REAP gate in campaign runner, direct GrowthCycle.finalize, lineage record and transaction recovery; artifact-bound interactive approval only after separate judge certifies. Production frozen REAP judge currently requires absent docs/reap_moe/reap_moe_campaign.json and a pre-finalization candidate verdict not yet available; judge_certifies_candidate deliberately returns FALSE in production. Thus an algorithmically qualified REAP run remains INCONCLUSIVE/PENDING, no lineage promotion. TTY confirmation is not cryptographic owner identity; no release is approved. Isolated 71/71 owner approval + campaign CPU tests passed; non-REAP pathways retain independent controls.

Safe resume scope is exact: automatic replay only of completed, fsynced, SHA-verified attempts with trustworthy GPU-cost accounting; a partly executing trainer with unknown cost remains BLOCKED from automatic restart. No algorithmic model promotion, current hardware qualification or benchmark improvement claim was made.

<!-- CHOWDER_FINAL_20261022_BEGIN -->
## Custom Qwen3.6 REAP-MoE model completion status — verified closeout 2026-10-22

**Current pinned review code:** PR #277 at `1960b7a241cdbd136affe63534e4fc23b738875c` (UNMERGED to main, no production promotion). Fresh affected-area tests: **24 passed**. Final-head CI: **SIX OF SIX SUCCESS**. Full local suite at final head: **4073 passed and 78 skipped at this head**. The previous green checkpoint `c88f859` had 4,055 passed / 78 skipped and six CI successes; the older `a0a678d` baseline had 3,960 passed / 78 skipped and six CI successes. Neither earlier outcome is a measurement on the current head.

Implemented fixes: `96701b9` rejects ambiguous cost flags and recomputes paid ledger totals; `cce4411` refuses REAP upward adjudication and unmatched legacy promoted records; `55a1109` blocks respend of terminal state roots and aligns CLI/TUI readiness, receipt recovery and executor admission; `9a14f73` requires the settlement CLI to authenticate journal, terminal report and attempt receipts; `6f35d09` requires exact durable PREPARED intent for *every newly promoted* ledger record regardless of claimed model path; `8d30dde` makes frozen REAP/Gen2 judges recompute paid ledger costs. A self-hash or a mutable local marker alone does not prove owner identity or independent model-release certification.

Follow-up hardening at the current tested head requires unique ordered native receipts **or explicitly non-replayable identity/cost-bound completion events** from remote/custom backends, verifies original native trainer evidence and checkpoint bytes, and checks offline cost compliance against the terminal campaign settlement. Completed native checkpoint hashing now streams in bounded 4 MiB chunks rather than loading each shard into memory. Committed lineage reads additionally require a consistent durable terminal verdict and cycle outcome; historical markerless promotions with missing model identity fail closed. Legacy non-native completions without their new cost-binding cannot be independently certified by the strict CLI. These are CPU-tested review-branch controls, not a signed external trust root. Current untrusted all-local-state rewriting remains outside the authenticated threat guarantee.

New ancestry and proof consistency work at this final head: promotion PREPARED and COMMITTED transactions require identical parent-version identities across the candidate generation, campaign record and late cycle outcome. REAP N5 classification accepts both canonical base_model_path and legacy path aliases but treats uninspectable model metadata as unknown, never as an automatic exemption. Local native attempts must retain native receipt proof and may not be downgraded into cheaper non-replayable completion events by modifying the journal; remote/caller-supplied attempts continue to carry their explicit cost-bound proof. Synthetic campaign model fixtures were updated to include real non-REAP model_type declarations; frozen production release approval remains blocked.

The native receipt guard distinguishes a real started GPU job with missing cost/receipt (REFUSED before any further compute or recovery) from an executor preflight refusal where `train_started` is explicitly false, no cost was measured, and no subprocess started. That honest zero-spend terminal REFUSED remains recordable but cannot be used as proof of a paid successful training run. This preserves normal campaign refusal semantics without relaxing receipt requirements for spent attempts.

The superseded head `9ec9985` had four reproducible Linux lineage-correction fixture failures because it claimed an inaccessible Windows-only F:/ synthetic model path as historical non-REAP provenance. At `c88f859` the fixture instead creates an actual temporary inspectable dense `config.json` on the running platform, preserving production fail-closed identity rules. All 28 focused fixture/approval tests passed after the correction. The earlier Linux failures remain historical evidence, never counted as passing checks on the corrected head.

Subsequent full CI at superseded head `c7c1b60` exposed ten Gen2 coupling/dry-run failures (Windows full 4,063 passed, 78 skipped, ten failed). The coupling fixture digested an empty `config.json` before the shared campaign fixture rewrote that exact checkpoint as `{"model_type":"llama"}` under the newer strict N5 ancestry rule. At `1960b7a` it hashes the exact same inspectable non-REAP bytes; all 24 focused coupling/matrix cases then passed. The failed superseded head remains explicitly recorded, not counted as final CI evidence.

Evaluation-cost compatibility: a trustworthy evaluator-reported `ComputeCost` remains chargeable even on a refused or failed evaluation; elapsed-time accounting is the conservative fallback only when that actual report is absent or untrustworthy. A settled complete GPU attempt is replayed exactly once from fsynced receipts, not retrained from arbitrary partial optimizer state; unverifiable in-flight GPU consumption is deliberately refused. The shared runner writer lock remains authoritative over advisory CLI/TUI readiness snapshots.

**Model evidence:** REAP160 BF16 package at `G:\llm-cold\bf16\Qwen3.6-REAP160` (26 indexed safetensors shards, 160 experts, top-8, 40 layers), content inventory `a1ba0e434eea76fdfc4b85843ab90d7957fa2ffd08ac319c2d2f8aed3ae4aa92`. REAP160 inference GGUF at `C:\llm-work\reap\gguf\reap160-q4_k_m.gguf`, 13,821,482,880 bytes, **freshly rehashed SHA-256 `8c840461752e1343e81bc12ab9b95d422a4b45c0b0048f00938fe882f299f8f5`**. Parent-matching tokenizer SHA256 `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`; no Gen-1 release-qualified adapter.

**Release blockers:** Mandatory REAP128_v2 checkpoint/full measured arm absent; REAP128 v1 GSM8K lower bound -0.0378074 violates frozen strict -0.03 and triggers Amendment-4 fallback requirements. Frozen REAP judge declaration missing and real independent precommit judge remains unable to certify, so explicit human N5 authorization has NOT occurred. Historical 27.48 seconds/step implies 7.63 GPU-hours for 1,000 steps versus 2.5 hours; no fresh improvement measurement. Current 14.5 GB VRAM, 20-step, WDDM spill, hardware/LoRA and derivative/Fable/Ornith rights remain unverified or blocked. Qwen3.8 Native Sparse remains separate. Never equate passing CPU CI with Gen-0 freeze or production promotion.
<!-- CHOWDER_FINAL_20261022_END -->

<!-- CHOWDER_POSTFINAL_20261022_BEGIN -->
## Supplemental pinned qualification — Qwen3.6 model candidate and calibration state (2026-10-22)

**Current review revision:** PR #277 `10a0044df73d71a7c40313082f524d13acc8fa1d`; draft and **not merged to `main`**. Pinned final-head Windows REAP/calibration and campaign/accounting regression across disjoint modules: **380 passed, 0 skipped**, no failures (49 selector/calibration in 2.28 seconds; 331 campaign/cost/judge in 104.69 seconds). At the preceding `c73ed0a` revision two non-overlapping broader CPU groups recorded **420 PASS / 1 SKIP** (208.57 seconds) and **136 PASS** (22.38 seconds); those 556 earlier results are NOT fresh measurements on the final head. The earlier `c73ed0a` head independently achieved six of six successful GitHub CI checks. GitHub CI at this exact revision: **SIX_SUCCESS** (six jobs required before acceptance). The earlier 4,073 passed / 78 skipped full suite applies only to `1960b7a241cdbd136affe63534e4fc23b738875c` and has not been represented as a new complete-suite run.

- `71883c2` fixes a reproduced persisted GPU-cost decoding flaw: a self-consistent ledger could previously carry JSON strings such as `"0.25"` for device/wall GPU-hours. `ComputeCost.from_dict` now accepts finite JSON integer/float values (not booleans), rejects strings/null/containers with `ValueError`, and preserves proven reported evaluator costs; conservative elapsed fallback remains only for untrusted/missing cost. The original 13 targeted negative cases failed before the fix; 118 isolated cost regressions passed, and 331 connected campaign/accounting/judge tests passed on `71883c2`.
- `c73ed0a` incorporates isolated patch `dd1c428`: frozen Amendment-4 REAP128_v2 D0 preflight now verifies the calibration SHA, manifest source/row/packed-token census, per-row bounds, and observer `n_seqs` for the complete admitted 717-row corpus. A `--limit-seqs` partial observation cannot silently satisfy v2 D0. The observer retokenizes decoded text, so observer `n_tokens` is validated as integral/plausible rather than equated incorrectly with the manifest's 1,197,289 packed tokens. Isolated worker reported 45 REAP tests passed and 25 loader tests passed/one skipped. Fixture positives are not actual REAP128_v2 GPU D0 evidence. A fresh read-only scan of the actual 717-row, nine-source v2 corpus verified SHA-256, 1,197,289 packed tokens, row bounds and all nine source kept-counts. Fresh manifest-file SHA-256: `d3f8733769a060a924db395b00d3c7c3acf65655ed6ed30a118df3dd58373f81`; the distinct raw calibration JSONL SHA-256 is `516f2e4a931fd3f025c6206e1f7e96b53ed68801ec503cb3d3367ca43f540c9f`.
- `10a0044` integrates isolated commit `3dce236`: frozen Amendment-4 v2 eligibility now checks the Ornith calibration render identity. The preregistered builder default is `bare`; `worked` is a separate optional render absent from the frozen amendment. The REAL existing v2 manifest states `ornith.render=worked`, despite being SHA- and census-valid, so it **must not be admitted as the frozen Gen-0 v2 arm** without an independently approved prospective amendment/new eligible calibration. Script and module entrypoint adversarial tests refuse the variant, while bare/absent positive fixtures pass; isolated 49 REAP tests and final-head selector tests pass. A fresh **negative-only** check against the actual on-disk v2 manifest (with a temporary synthetic observer, never called measured D0) returned the expected `not preregistered: expected bare` refusal. No archived source, model weight, threshold or frozen prereg was changed.
- The REAP160 BF16 candidate exists as 26 indexed shards with 40 MoE layers, 160 experts/layer and top-8 routing; router/expert keys were independently matched across 40/40 layers and its v1 pruning observation digest matched the archived observation. The 13,821,482,880-byte Q4 GGUF `C:\llm-work\reap\gguf\reap160-q4_k_m.gguf` retains its separately previously verified SHA-256 `8c840461752e1343e81bc12ab9b95d422a4b45c0b0048f00938fe882f299f8f5`. No new full GGUF hash or GPU execution was performed for this increment.
- **Frozen release remains BLOCKED.** REAP128 v1 strict GSM8K Gate1 bound failed (-0.0378074119723426 versus required > -0.03). Mandatory distinct REAP128_v2 checkpoint and full D0/evaluation, both fallback arms, frozen independent judge, explicit postjudge owner N5 approval, actual 14.5-GB/20-step GPU qualification, 2.5 GPU-hour recipe, and Fable/Ornith/derivative licensing clearance are missing. Historical 27.48 s/step implies ~7.63 hours/1,000 steps; 10.23-second estimated bulk-transfer floor is an analytical bound, **not** measured improvement. No Gen-0 selection, Gen-1 adapter certification, production promotion or autonomous model improvement is claimed. The separate Qwen3.8 Native Sparse Program is untouched.

**Reproduction / exact-commit checks (PowerShell from review worktree):**

```powershell
git rev-parse HEAD  # 10a0044df73d71a7c40313082f524d13acc8fa1d
$env:CHOWDER_NO_GPU='1'
python -m pytest -q --tb=short tests/test_reap_select.py tests/test_reap_gates.py tests/test_reap_calibration.py
gh pr checks 277 --repo niko4244/Chowder
```

The v2 calibration manifest exists but its worked Ornith variant is not eligible under the frozen default. A separately constructed REAP128_v2 model also does not exist. The selector now refuses incomplete v2 observations and unregistered renders; it does not construct or certify v2 weights. REAP160 availability alone cannot select Gen-0.
<!-- CHOWDER_POSTFINAL_20261022_END -->

<!-- CHOWDER_CONTINUATION_20261022_BEGIN -->
## Current engineering continuation

**New pinned continuation (2026-10-22):** Draft [engineering PR #277](https://github.com/niko4244/Chowder/pull/277) at `84dbb2e9151a0cd9be70a00f167271953ca38d92` (SIX_SUCCESS); not merged to `main`. Exact integrated Windows CPU eight-file suite: **281 PASS / 0 SKIP / 0 FAIL**. Two additional disjoint campaign-runner and settlement adversarial test files: **110 PASS / 0 SKIP / 0 FAIL**, for **391 passes across ten distinct files** at the identical integration HEAD. Older CPU tests cited elsewhere are separate checkpoints, not additional final-head passes.

**Fresh entire CPU-focused pytest suite at the same commit:** **4117 passed, 78 skipped, 0 failed**. This includes tests in the targeted runs; their counts must not be added to the whole-suite count.

- **CH-065, P1:** Frozen REAP speed gate now requires every one of 200 eligible GSM8K request speed measurements for both candidate and dense arms, refusing 1-of-200 false PASS and nonfinite/malformed values.
- **CH-066, P2:** Frozen R3/T23 judges now validate the complete recorded budget failure-reason list against the recomputed production settlement. Historical absent reasons remain UNKNOWN, not promoted.
- **CH-067, P2:** Campaign cost settlement now binds a populated terminal attempt summary to every charged training start; the legitimate paid-failure/no-selection path now writes that summary.

This is **review-only CPU evidence**, not proof of model speed, selection, device-resident training or budget feasibility. Existing v2 Ornith `worked` is outside frozen preregistration, REAP128_v2 and full D0 are absent, v1 Gate1 failed, frozen judge/N5 approval is absent, licensing is unverified, and no release or GPU qualification is authorized. `main`, rescue and the independent Qwen3.8 Native Sparse project are unchanged.
<!-- CHOWDER_CONTINUATION_20261022_END -->

<!-- CHOWDER_SAFETY_FOLLOWUP_20261022_BEGIN -->
## Subsequent CPU safety follow-up

**Current engineering review:** [PR #277](https://github.com/niko4244/Chowder/pull/277) at `0522b688e525852c6bb18ecb613a92b39b963fe0`, **SIX_SUCCESS** CI, draft and not merged. Fresh exact-head CPU regression across ten named files: **411 passed, 0 skipped, zero failed**. Previous complete suite (4,117 passed, 78 skipped) was measured at old `84dbb2e` only, never silently attributed to this updated head.

- **CH-068 (P1):** REAP per-item GSM8K/HumanEval quality requires boolean `correct`; downstream float conversion cannot inflate the frozen paired bound.
- **CH-069 (P2):** Paid native attempt replay must match its original pre-run receipt mode, strictly typed index, recipe, and terminal status.
- **CH-070 (P1):** Frozen REAP/Gen2 judges compare the cycle, parent and candidate record identity to the declared campaign before certifying cost.

These are **CPU-proven code safeguards**, not a successful REAP128_v2 build, frozen paired evaluation, GPU LoRA training, independent frozen judge, owner N5 attestation or licensing clearance. The existing release, Gen-0 selection and Gen-1 certification gates remain **BLOCKED**. Frozen thresholds, weight/data inputs, main, rescue and independent Qwen3.8 Native Sparse program are unchanged.
<!-- CHOWDER_SAFETY_FOLLOWUP_20261022_END -->

<!-- CHOWDER_GATE0_FOLLOWUP_20261022_BEGIN -->
## Subsequent Gate0 frozen evidence repair

**Current reviewed engineering:** [PR #277](https://github.com/niko4244/Chowder/pull/277) at `05117214ba49326749e98a255d4e86dc242c49ca`, **SIX_SUCCESS** CI, unmerged draft. Exact-head ten-file CPU regression: **425 passed, 0 skipped, 0 failed**. The 411-pass run at `0522b68` and the 4,117-pass / 78-skip full suite at `84dbb2e` are earlier evidence, **not** current-head full-suite results.

**CH-071 (P1):** Frozen GSM8K Gate0 wrongly treated string `terminated='false'` as a true termination and nonfinite `trigram_ratio=NaN` as nondegenerate. The adversarial reproductions flipped genuine Gate0 FAIL to PASS. The shared score loader now requires literal Boolean termination and a finite nonboolean numeric trigram ratio from 0 through 1, matching the real evaluator. Adversarial tests also cover self-rehashed summary attestations; existing correct Boolean/finite examples remain admissible. Frozen thresholds are unchanged.

This remains **review-branch CPU-only** assurance. Missing REAP128_v2, registration-eligible Ornith calibration and D0, actual paired benchmarks, GPU throughput/budget evidence, independent frozen judge, N5 approval and derivative rights keep Gen-0 selection, Gen-1 certification and release **BLOCKED**. No GPU run, release, main merge or protected model edit occurred.
<!-- CHOWDER_GATE0_FOLLOWUP_20261022_END -->

<!-- CHOWDER_MODEL_BUILD_FOLLOWUP_20261022_BEGIN -->
## Subsequent frozen REAP model-build readiness

**Current pinned engineering review:** [PR #277](https://github.com/niko4244/Chowder/pull/277) at `4cdc17c51dde11aeb144b88799d60dcdb382d8c3`; **SIX_SUCCESS** CI, draft and not merged. Current exact-commit CPU regression across twelve named files: **462 passed, 0 skipped, zero failed**. Previous 425 focused passes were measured at `0511721` only; the earlier complete 4,117-pass/78-skip suite belongs to `84dbb2e` only and is not a new full-suite run.

**CH-072 (P1):** Fail-closed REAP pruning now verifies the exact indexed-versus-present tensor census and source saliency shape/values, refuses pre-existing output directories, records 40-layer observation hashes when supplied, and atomically publishes a complete checkpoint from a temporary directory. Adversarial tests reproduced overwrites, extra/missing tensor admission and partially published checkpoints in older code.

**CH-073 (P1):** A new read-only arm-aware preflight validates the original 40-layer/256-expert/top-8 model index, v1 or v2 SHA-matched calibration/Ornith evidence, all forty small CPU observer tensor files and full D0 before emitting a deterministic unexecuted prune command. An integrated direct-script import defect was reproduced and corrected before this review checkpoint. Real v1 REAP160 preflight passed (384 sequences, D0 PPL 10.646); the unmatched v2 observer was refused. No model shards were opened, changed or produced by these preflights.

**Release qualification remains BLOCKED:** frozen REAP128_v2 build, full D0, the registered paired evaluations, 14.5 GiB/WDDM/20-step LoRA and 2.5-hour budget, independent judge/N5 and derivative/data licensing have not been established. Neither an input preflight nor CPU regression constitutes selection, training or promotion. Frozen thresholds, main, rescue, weights and Qwen3.8 Native Sparse are untouched.
<!-- CHOWDER_MODEL_BUILD_FOLLOWUP_20261022_END -->

<!-- CHOWDER_GEN0_GEN1_20261023_BEGIN -->
## Gen-0/Gen-1 scientific qualification continuation

**Current code review:** [PR #277](https://github.com/niko4244/Chowder/pull/277) at `2be95ceb28afdc4aa0d4b4787bc70c3c801e1194` (PENDING CI), unmerged draft. At this exact revision, **446 focused CPU tests passed, zero failed**, Ruff passed. The older 462 focused passes and 4,117-pass full suite are at their earlier SHAs, not newly measured here. A superseded 144bd2f CI run failed because a committed MoE test expected the old trainer-template dictionary; the new required 2,048 field is now asserted in the generation and parsed worker spec.

**CH-074:** REAP v2 preflight now requires the *actual* registered 704-row base SHA and only a declared, digest-verified bare Ornith suffix, in addition to complete D0/saliency and model shape checks. The previous self-consistent three-row substitute was reproducibly admitted before repair.

**CH-078 — corrected forged optional suffix:** After the initial base SHA guard, both the direct selector and build preflight still accepted a self-consistent fake 717-row v2 corpus with invented Ornith bare rows. The integrated common validator now enforces the actual pinned source and its zero eligible rows; both entrypoints reject that forgery. This does not produce a full v2 D0 witness.

**CH-079 — observer memory engineering:** The old final 2048-token perplexity step allocated a roughly 2047-by-248320 FP32 vocabulary matrix plus BF16 logits. The new 64-position streaming equivalent reduces the worst projection shape approximately 32-fold without dropping any calibration examples or layers. CPU numerical equivalence was tested; GPU peak and qualifying v2 D0 remain UNMEASURED. A bounded real CUDA one-layer smoke was not started because only 7007 MiB GPU was free at the 8192 MiB conservative launch guard.

**CH-080 — independent Gen-1 2×2 production preparation:** The frozen candidate planner has two learning-rate recipes and rejects dataset identity as an admissible search axis. New strict CPU materialization prepares four *separate* projects for Fable/control crossed with 5e-5/1e-4, pins actual 2048-ready mix/tokenizer/input hashes and preserves the enforced <=2.5 GPU-hour per-recipe budget and hardware refusal. Genuine 767/647-row source files and exact 2048 readiness receipt passed the new independent source check. Production config admission now recognizes declared MoE tools and placement keys. These are prepared independent trial inputs, NOT completed training, a registered four-candidate search or a certified model comparison.

**CH-081 — unresolved paired statistic:** Frozen Amendment 4 names a one-sided 95% paired McNemar noninferiority bound. The current selector calculates a normal paired-difference mean/stdev lower bound without a preregistered McNemar construction. Genuine v2-v1 paired observations do not yet exist; no statistic has been substituted. New controller logic explicitly BLOCKS optional v2 replacement of passing v1 while retaining the diagnostic Wald bound and registered mandatory Gate0/1 failure fallback. Prospective method adjudication is needed before a v2 replacement verdict.

**CH-082 — corrected lean-CI collection:** The first consolidated review at `2ae2d70` failed lean CI because new observer CPU tests unconditionally imported optional `torch`; the new `b027d78` revision skips only those tests when Torch is absent and exercises them when it is installed. Historical CI failure `39928158887` remains a failure and is not attributed to this new head.

**CH-083 — passive WDDM witness, not an INT8 gate:** A new Windows tool records per-PID process identity and dedicated/shared GPU memory counter samples. A real two-sample llama-server witness captured zero shared bytes but this does NOT establish a no-spill verdict for a REAP160 training step; Windows counters can undercount CUDA usage. Real placement-step instrumentation and operator review remain due.

**CH-084 — exact exported adapter reload identity:** Parent baseline and candidate controller/worker paths now compare PEFT LoRA directory bytes to the exported SHA before touching torch/model/GPU, rather than accepting a mutable path or mismatched local base. CPU adversarial tests exercise tampered or missing adapters. No Gen-1 trained adapter or independent real-model reload was executed.

**Exact-head GitHub Linux Python 3.12 broad suite:** At `b027d78`, run `39929171488` job `120332880915` recorded 3,738 passed, 225 skipped and four warnings. This is a GitHub runner result, distinct from the locally measured 354 focused CPU passes.

**CH-077 — unresolved registered-data contradiction:** The exact offline bare-render builder with SHA-pinned admitted Ornith input wrote 704 rows (SHA `f6a09b694ae43a6d8b44b5180a861856501295eabf3d5297eb9749a008cdd929`), identical to the registered base. All 39 admitted SFT items render at 40–173 tokens, so none reach the frozen 256-token minimum. The historical 717-row corpus uses an unregistered worked render. No registered 717-row bare D0 exists; we did not pad, change a threshold or invent it. Distinct REAP128_v2 weights and paired scores remain absent; McNemar wording needs prospective clarification.

**CH-075/076:** The real current Gen-1 manifest is SHA-pinned and the 2,048-token preflight accepts 767 Fable and 647 control rows (120 Fable-exclusive), no clipped assistant answers, max 1,906/1,855 tokens. By contrast the prior generated 512-token Trainer would clip 198 and 151 answers, fully removing 20 and 6. Preparation now carries the declared `max_length: 2048` end to end; legacy undeclared campaigns retain 512. Historic 4,096-screened 1,028-row text is not this mix. Source caps are NOT complete: 0/120 Nemotron, 9/150 OT3 code and 38/39 Ornith survived the declared validation. Do not claim that the requested Nemotron material is present in a trained model.

**Real compute availability:** Kaggle GPU quota 0.00 hours remaining (reported used 31.32 of 30.00, refresh October 24); local RTX 5060 Ti only 7,007 MiB free at the later conservative launch gate with unrelated residents, under the target placement. No new GPU training, optimizer steps, LoRA export, adapter reload or paired decode measurement. Historical 27.48 s/step projects 7.63 h for 1,000 steps versus the frozen 2.5 h ceiling. Four-arm Gen-1, independent evaluation, Gen-2 and quantized deployability remain blocked by selection, resources, frozen judge/N5 and derivative rights. PR #273 is an ancestor of newer PR #277; protected main unchanged.
<!-- CHOWDER_GEN0_GEN1_20261023_END -->
