# Custom Qwen MoE completion register

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
