# REAP-MoE N5: candidate certification is not permission to advance lineage

Frozen `docs/REAP_MOE_LINEAGE.md:89-95` requires human owner approval **after**
the independent frozen local judge certifies, on **every** REAP generation.

## Reproducible gap in integration f099d3a

`src/chowder/growth/campaign_runner.py` previously took a mechanical
`PROMOTED` from `_adjudicate`, ran branch-protection certification, wrote
`prepare_transaction`, and immediately called `GrowthCycle.finalize`, which
recorded the new generation in `GenerationLedger`. Neither path asked the
owner. These are two different decisions:

- **Candidate qualification**: the training run's measured evidence satisfies
  machine gates. Its mechanical verdict can be `PROMOTED`.
- **Release / generation promotion**: the independent frozen REAP judge has
  certified the *same run evidence*, and the human owner has explicitly
  approved the exact candidate artifact before the lineage pointer advances.

## Fail-closed gate implemented in this branch

The REAP-prune marker and the Qwen3.5-MoE text config's 128/160 routed expert
count identify the REAP lineage without trusting a `CHOWDER_*` environment
switch. The marker is sufficient but not necessary, so removing only that
marker does not exempt an otherwise recognizable REAP checkpoint. Campaign
qualification and measured costs are preserved even when release is denied.

For REAP, `campaign_runner` now checks the frozen judge prerequisite, then
an interactive **exact cycle ID, candidate version and full 64-hex selected
artifact SHA** confirmation. Missing judge certification, non-TTY input,
incorrect text, EOF, or environment variables cannot grant approval. The
operator grant additionally carries a judge-certified bit in its per-process
MAC; a raw interactive phrase alone, without that prerequisite, cannot be
used as a direct finalize/ledger approval token. The
persisted result is `INCONCLUSIVE` with a `human_approval: pending` phase, the
mechanical adjudication separately recorded, and **no** promotion transaction
or lineage entry. A subsequent attempt at the same pending state root refuses
before compute; the spent evaluation evidence is preserved.

`GrowthCycle.finalize` and `GenerationLedger.record` enforce an in-process,
candidate-bound approval grant for REAP promotions, including direct callers.
The transaction recovery/read path additionally checks the approval phase's
cycle/version/artifact binding against the committed campaign record digest.
Missing model identity cannot be used to promote through `GrowthCycle.finalize`.

**Not yet a full release workflow:** `docs/reap_moe/judge_reap_moe.py` currently
requires `campaign-run.json`, which the original runner publishes after
finalization; its preregistered `docs/reap_moe/reap_moe_campaign.json` is also
not currently present. The available branch-protection certificate is not
equivalent to a frozen REAP judge PASS. Consequently
`owner_approval.judge_certifies_candidate()` intentionally returns False in
production. This branch blocks actual REAP lineage release until a separate
**judge-then-human-approval-then-commit** stage is implemented and validated
without re-running the expensive campaign. A positive CPU fixture injects
both a simulated independent judge PASS and a candidate-bound owner grant
explicitly; those are tests, **not** a production authorization switch.

The local TTY prompt is operator acknowledgement, **not cryptographic proof
of Nik's real-world identity**. The per-process MAC prevents accidental token
replay between candidates but is not resistant to arbitrary malicious Python
code executing in the same process. A trusted pinned public-key signature or
equivalent OS identity approval would be needed for a stronger adversary.
Filesystem tampering that falsifies a base checkpoint's config and removes
every lineage identifier requires independent immutable model-content identity
and lineage binding; the marker/config classifier is not a cryptographic
lineage attestation.

No actual model was trained, evaluated, judged or promoted by this change.
