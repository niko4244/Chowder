# Chowder reliability and security assessment

> Final audit 2026-10-22. Main 0bc566c5; MoE PR268 184291f; recovery PR269 cf54480 with local 82-test verification; REAP PR271 73f1593. GitHub and read-only Windows workspaces inspected, 39 model files independently SHA-256 verified twice. No GPU training was started in this audit. Frozen gates and required approvals remain binding.

## Remediated on open review branch (not merged)

**CH-001 / P1**: PR #269's `_journal_outcome` originally scanned all events and accepted *any* earlier `terminal` even if a subsequent run emitted `campaign_start` and died. A stale terminal plus a matching old verdict could bypass interrupted-entry refusal. On `fix/campaign-journal` commits `ad9d847c` and `e3f0fbd7`, the parser now resets at each start, refuses malformed/torn records, requires a terminal final line and verifies the final verdict file matches. Two regression tests added. **Validation: pending fresh CI**; cannot assert tests passed.

**CH-002 / P1**: PR #269 formerly atomically renamed an *unfsynced* verdict file before writing the journal's fsynced terminal marker. The PR now flushes/fsyncs the temporary verdict file before `os.replace`. Parent-directory metadata durability under all Windows power-loss scenarios is not independently verified; avoid a power-loss guarantee stronger than evidence.

## Not completed

- **CH-003 / P1**: Journaled interrupted campaigns refuse restart, but automatic safe resume and exact partial GPU-hour reconciliation remain explicitly out of scope. Recovery must independently establish attempted recipe IDs, cost, artifact identity and ledger signatures before replay. Never clear or delete the journal automatically.
- **CH-004 / P1**: Concurrent writers against one state root, journal ownership, journal injection, and tamper resistance have not been fully qualified here. Fsync is not an authentication mechanism. Add a single-writer ownership/lease protocol and adversarial tests before calling it multi-process crash-safe.
- **CH-005 / P1**: Gen-2 run 10 completed two attempts (historically 2.16 GPU-h) before ancestor raw-artifact certification refusal; #266 and prereg amendment 24 place/check immutable evidence earlier. Exact local artifacts cannot be rehashed here.
- **CH-006 / P2**: Full CLI/TUI vs service-equivalence, signed cross-generation provenance, budget settlement under interrupted search, checkpoint corruption handling and non-regression promotion invariants require fresh integrated tests. The source and historical tests exist but are not current verification.

## Safety boundaries

No automatic verdict reversal, threshold relaxation, lineage transplant, evidence file regeneration or production-model promotion was made. Human sign-off is mandatory even after the frozen REAP judge succeeds (`docs/REAP_MOE_LINEAGE.md`). Fable outputs and use terms remain subject to unresolved N2; technical controls are not a substitute for a licensing decision.

## Reproduction and verification

Use PR #269 head `cf544801af525826c5bd507581b27580feff23f2` and run:

```shell
python -m pytest tests/test_growth_campaign_runner.py tests/test_growth_campaign.py tests/test_growth_kaggle_campaign.py tests/test_moe_campaign_plumbing.py -q
ruff check src tests
```

Verify the two added stale/torn terminal tests, original hard-exit subprocess tests, refusal at repeated entry, atomic record matching, and CI Windows/Linux matrix. These commands are prescriptions, not claims they executed in this session.

**Verdict: PARTIAL. Critical restart fail-open code patched on review branch; automatic safe resume and full Windows durability unqualified.**


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
## Crash recovery, provenance, accounting and REAP N5 boundaries — verified closeout 2026-10-22

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
