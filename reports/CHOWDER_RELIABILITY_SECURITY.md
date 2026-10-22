# Chowder reliability and security assessment

> Audit snapshot: 2026-10-22 UTC. Main 0bc566c5db052b90e3aa1223fef94445d1045dc5; PR #268 24f2f8865b215b477796364b1ec30e944d5ec087; recovery PR #269 e3f0fbd77bc679ea9a856461294cd26b40316562. GitHub code and published evidence were inspected; local Windows workspace inaccessible (Chat On Steroids Core root /strata invalid). Source claims below are **historical reports**, not rerun hardware measurements. No model weights or protected local evidence were accessed in this audit.

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

Use PR #269 head `e3f0fbd77bc679ea9a856461294cd26b40316562` and run:

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
