# Transaction admission for newly promoted lineage records

The public low-level `GenerationLedger.record()` API used to create a new
`PROMOTED` record after examining only a caller-supplied `base_model.path`.
A fabricated non-REAP path thereby bypassed N5 and the durable campaign
promotion transaction. Model families must not be inferred from an optional
caller-controlled field to decide whether new generations require commitment.

From this patch, **every new promoted ledger mutation**, whether REAP or not,
requires an exact candidate/cycle/decision matching `PREPARED` campaign intent.
The existing REAP-specific independently checked frozen-judge and owner gates
still run after the transaction check. `PREPARED` alone cannot make a promoted
generation visible: `GenerationLedger.get()`, `effective_verdict()` and ancestry
continue requiring `COMMITTED` and matching final campaign evidence.

Previously saved non-REAP generations with no transaction remain readable as
historical records. This is a read-only compatibility distinction, not a
permission to create more uncommitted records. Historical REAP promotions
without committed N5 evidence remain blocked pending migration/adjudication.

This is a same-process and local-filesystem fail-closed admission control,
**not** independent cryptographic proof of a frozen judge, operator identity,
or model parent provenance. An adversarial actor that can forge matching
campaign state/transaction files or execute arbitrary Python in the process
can compromise mutable local evidence; strong release certification requires
an out-of-process judge, pinned owner public-key signatures, and externally
anchored append-only evidence. Production REAP N5 remains blocked until then.

CPU regression targets: fabricated non-REAP model paths, missing transaction,
exact PREPARED admission, refusal of pending reads, direct grant-only rejection,
and old non-REAP revision compatibility. No GPU/model change or promotion
was performed as part of this patch.
