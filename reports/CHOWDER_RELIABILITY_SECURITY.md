# Chowder reliability and security assessment

> Audit snapshot: 2026-10-22 UTC. Main 0bc566c5db052b90e3aa1223fef94445d1045dc5; PR #268 fe94342c1073eb39e69be01ba224c63b38057b95; recovery PR #269 e3f0fbd77bc679ea9a856461294cd26b40316562. GitHub code and published evidence were inspected; local Windows workspace inaccessible (Chat On Steroids Core root /strata invalid). Source claims below are **historical reports**, not rerun hardware measurements. No model weights or protected local evidence were accessed in this audit.

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
