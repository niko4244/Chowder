# Campaign settlement CLI: required evidence

`chowder growth campaign settle MANIFEST ACCOUNTING` is a read-only,
post-campaign audit of **actual recorded spend**, not a calculator for an
arbitrary cost file. Successful settlement now requires all of:

- The accounting file is `cycle_compute_accounting.json` under the manifest's
  declared `state_root`, not a copied or newly constructed ledger elsewhere.
- A completed `campaign-run.json` in that root matches the declaration's
  cycle, parent and candidate identities, and records the exact accounting
  path, SHA-256 digest, and reaggregated device/wall hours.
- The campaign journal has a durable terminal event matching the run verdict,
  whose independent wall-hour field matches the ledger. Its original campaign
  start identity must also match the supplied frozen declaration, declared
  input bytes, and local source identity: changing the manifest budget cannot
  turn a recorded overrun into compliance. Training entries must match
  journaled attempt starts and recipe identities.
- Any per-attempt receipt files match their separately journaled hashes and
  training costs. The recorded ledger must contain actual positive paid work;
  a self-hashed zero-entry or zero-spend document never demonstrates a paid run.

The command exits **0 only for a verified, budget-compliant settlement**.
Malformed or missing evidence, mismatches, and over-budget settlements exit
**1** and print JSON with `budget_compliant: false`. Fixtures that previously
settled a standalone synthetic ledger must now construct a genuine terminal
campaign run or explicitly test lower-level `settle_campaign` arithmetic.
Historical runs lacking the journaled admission identity intentionally refuse
settlement instead of silently accepting an unattested record.

The general `checked_ledger_total` validator continues to refuse
non-incremental paid entries. This CLI may admit a genuinely zero-cost
non-incremental *evaluation* row in an otherwise authenticated campaign with
positive charged training. Zero-cost rows do not substantiate any paid work.

**Trust boundary:** the ledger's embedded SHA-256 is not a digital signature.
The cross-file terminal record and journal/receipt evidence detect independent
edits and casual forgery. An attacker with write control over the entire
state root could rewrite every local witness. Strong adversarial authentication
requires a separately signed or append-only trusted external digest.
