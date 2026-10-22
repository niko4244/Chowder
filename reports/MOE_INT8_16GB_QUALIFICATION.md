# REAP-MoE int8 / 16 GB qualification

> Audit snapshot: 2026-10-22 UTC. Main 0bc566c5db052b90e3aa1223fef94445d1045dc5; PR #268 24f2f8865b215b477796364b1ec30e944d5ec087; recovery PR #269 e3f0fbd77bc679ea9a856461294cd26b40316562. GitHub code and published evidence were inspected; local Windows workspace inaccessible (Chat On Steroids Core root /strata invalid). Source claims below are **historical reports**, not rerun hardware measurements. No model weights or protected local evidence were accessed in this audit.

## Artifact and measured evidence

The active fallback research artifact reported by PR #268 is `G:\\llm-cold\\bf16\\Qwen3.6-REAP160`: 26 shards, 46,039,573,000 bytes, config sha256 `15413a791b9eff423755743bb05b997c93f6ae6f4ce02a1966c9bdfdc2cdc3b9`. **Only the config hash is cited from the PR; full shard hashes and existence are not verified in this environment.** This is not, by itself, a Gen-0 selection verdict. REAP128 versus REAP160 must follow `docs/REAP_QWEN36_PREREG.md`, including dense/random controls and the applicable gates.

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
