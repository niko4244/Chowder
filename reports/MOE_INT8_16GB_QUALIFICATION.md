# REAP-MoE int8 / 16 GB qualification

> Final audit 2026-10-22. Main 0bc566c5; MoE PR268 184291f; recovery PR269 cf54480 with local 82-test verification; REAP PR271 73f1593. GitHub and read-only Windows workspaces inspected, 39 model files independently SHA-256 verified twice. No GPU training was started in this audit. Frozen gates and required approvals remain binding.

## Artifact and measured evidence

The active fallback research artifact reported by PR #268 is `G:\\llm-cold\\bf16\\Qwen3.6-REAP160`: 26 shards, 46,039,573,000 bytes, config sha256 `15413a791b9eff423755743bb05b997c93f6ae6f4ce02a1966c9bdfdc2cdc3b9`. **Existence, header integrity and SHA-256 hashes for all 26 shards and all 13 auxiliary files were independently verified twice in the Windows workspace.** This is not, by itself, a Gen-0 selection verdict. REAP128 versus REAP160 must follow `docs/REAP_QWEN36_PREREG.md`, including dense/random controls and the applicable gates.

Historical observations reported on the RTX 5060 Ti, not rerun:

| Measure | Prior published observation | Interpretation |
|---|---:|---|
| Live int8 expert grid + scales | 18.82 GiB | 40 layers, 160 routed experts per layer, top-8 architecture |
| Sanctioned seq-2048 checkpointed placement step | 11.345 GiB allocated peak, 11.664 GiB reserved | 14.5 GiB gate **recorded PASS for old code configuration** |
| Sanctioned placement timing | 51.8215 s/step, first-touch included | Placement-only, not LoRA training throughput |
| Real LoRA/AdamW 20-step run | 33.1–35.2 s warm; historical 41.59 s stated aggregate | A 20-step execution, **not** a completed smoke gate |
| Controlled A/B grouped + retained weights | 27.48 s/step, 10.788 GiB allocated, 13.215 GiB reserved | 4 steps/phase and drift control, not a 1000-step campaign |
| Reported PCIe H2D | 5.52 GiB/s, Gen3 x8 | Copy ceiling prevents 9 s/step under reported 3-pass traffic |
| Kaggle full real-model run | 16/26 shards after 707.7 min | No training step; dataset-mounted small-tensor reads bottlenecked |

References: [PR #268](https://github.com/niko4244/Chowder/pull/268), `docs/quals/MOE_INT8_PLACEMENT_PREREG_2026-10-20.md`, `docs/quals/MOE_INT8_THROUGHPUT_2026-10-20.md`, `kaggle/int8_witness/RESULTS-2026-10-21-real-run.md` on its branch.

## Frozen gate register

| Gate | Latest auditable published status |
|---|---|
| load_census | MEASURED, 0 live BF16 expert parameters reported |
| memory_vram <=14.5 GiB | PASS on b4094c1, before default-on retained weights; remeasurement required for shipped config |
| host_rss | MEASURED, but host and run context specific |
| throughput | MEASURED placement step; real recipe remains over 2.5 GPU-h |
| gradient | NOT MEASURED in generated gate table |
| smoke (>=20 optimizer steps and >=1% loss drop) | NOT MEASURED in generated gate table |
| router_balance | NOT MEASURED in generated gate table |
| wddm_spill | NOT MEASURED; operator basis required |

**Budget feasibility:** 27.48 s/step * 1000 / 3600 = 7.63 GPU-h, not 2.5. The 2.5h ceiling is not met at the measured speed, even excluding first-touch. No protocol threshold may be changed after results. Do not launch the declared 1000-step recipe as if it fit.

## Work remaining / precise acceptance

0. Review #268's added preflight tests: malformed/empty/too-short calibration now refuses before expensive full-checkpoint load; verify this in CI.
1. When explicitly authorized and card is idle, rerun sanctioned 3-step placement under the *current retention default*; capture both allocator figures and OS WDDM shared-memory basis.
2. Record 20+ optimizer steps and independent adapter reload with loss windows; collect LoRA-leaf gradients on required layers and entropy windows from real router data, then import its signed/config-matched witness into the generated prereg table.
3. Run BF16 bulk / int8 bulk A/B under matched model/config/seq; preserve noise floor, first-touch, link measurement, RSS and PCIe capacity. Do not confuse per-layer synthetic numbers with whole training-step time.
4. Reconcile compute-budget feasibility by a new preregistered recipe/hardware decision **before** authorizing expenditure; do not retroactively relax the existing gate.
5. Verify REAP128/dense/random quality comparisons, real Q4 deployment speed/GPU residency, tokenizer/model manifest/hash, and all remaining selection evidence.

**Verdict: PARTIAL / not certified for Gen-1 admission.**


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

## Verified deployable GGUF artifact (2026-10-22)

Existing REAP160 Q4_K_M GGUF: C:\llm-work\reap\gguf\reap160-q4_k_m.gguf, 13,821,482,880 bytes, independently calculated SHA-256 8c840461752e1343e81bc12ab9b95d422a4b45c0b0048f00938fe882f299f8f5, identical to the previously saved serve.json digest. Its GGUF header is version 3 with 733 tensors and 41 metadata entries. Historical llama-server load-only evidence reports --fit on -np 4 -c 16384 --jinja, all 41/41 layers GPU-resident, zero routed experts offloaded, and 14,275 MiB VRAM. The archived speed resolution reports 3.8523x the dense Q4 comparison. The file and hash are verified now; loading/generation were not rerun in this audit because the RTX 5060 Ti is occupied. This identifies a real inference artifact, but final custom Gen0 selection remains blocked by amendment-4 REAP128_v2 evaluation and training/release gates.

## Independent integrated engineering reconciliation (2026-10-22)

Current main is 0bc566c5, unchanged. PR #268 at 184291f, #269 at cf54480 and #271 at 73f1593 each passed their original six-job GitHub CI matrix. Their code merges without conflicts on draft integration PR #273. The integration branch f099d3a additionally preserves previously published interrupted campaign verdicts, charges failed evaluation work via independently valid reported cost when available or conservative measured wall time when not, and implements a durable PREPARED/COMMITTED promotion transaction with lineage/terminal digest verification. Recovery is also bound to the complete immutable campaign declaration digest so changed preregistrations cannot replay an old promotion. Crash/manifest integrity targeted tests: 8 PASS. All changes are in open review branches, NOT merged to main.

Draft stacked PR #272 at c2202c1 provides content-bound MoE SHA inventories, enrollment CLI, and fail-closed verification in placement, witness and campaign admission. Lightweight repeat checks are filesystem identity checks plus semantic-file hashes; not a security guarantee against adversarial in-place rewrites with restored metadata. Integrated CPU inventory/gates/witness tests: 64 PASS. The full 39-file REAP160 SHA inventory from two earlier independent reads remains verified; the new gate-specific enrollment is being run as a separate non-destructive operation. Do not assume the old config-only gate table satisfies this new contract.

Fresh integrated CPU testing: CLI/TUI/budget/adversarial suite 151 PASS. The original broad integration run had 187 PASS and one source-identity failure during concurrent edits; the single failed real CPU identity test passed on a stable pinned commit in 74.62 seconds. A second pinned broad run had 188 PASS and two accounting-assertion failures. The code now preserves valid reported evaluation cost and charges conservative wall fallback only when actual cost is missing; the four relevant targeted assertions passed. Full integration PR #273 CI at the latest pushed SHA remains to be confirmed. Test suites overlap and pass counts must not be added as unique tests.

The frozen v2-Ornith calibration source at C:\llm-workeap\calib-v2-ornith\calibration.jsonl is verified: 717 rows, SHA-256 516f2e4a931fd3f025c6206e1f7e96b53ed68801ec503cb3d3367ca43f540c9f. No REAP128_v2 model/evaluation was found, so amendment-4 Gen-0 selection remains BLOCKED despite historical REAP160 v1 quality and Q4 speed. WDDM spill is still unmeasured. Present-code retention-default VRAM has not been requalified. Historical 27.48 seconds/step implies 7.63 GPU-hours for 1000 steps against the 2.5-hour budget. Fable/Ornith data-rights qualification and human promotion approval remain open.

Full automatic interrupted-training continuation is not implemented: an incomplete journal refuses before re-spending attempted GPU work, and a completed transaction is recovered only with matching terminal and ledger evidence. Model promotion and expensive GPU experiments have not been performed.
