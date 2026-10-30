# Original REAP128_v2 prospective external capture signature — research engineering

2026-10-30. **Gen-0 remains `BLOCKED`, `chosen=null`, `release_approved=false`.**
No new native inference, actual capture, independent signature, frozen-gate
approval, F: mutation, model change, or completed HumanEval rerun occurred.

## Why an independent signature is necessary but insufficient

The original frozen `reap_eval._chat` discarded HTTP request and response bytes.
Historical scored rows cannot be retrospectively upgraded to authentic native
request/response evidence. The existing prospective raw-byte and temporal
validators inspect self-reported JSONL; an adversarial collector can reserialize
fabricated events and update their SHA-256 hashes. The temporal checker also
does not itself establish the capture's owner from an external source.

`chowder_batch.reap_v2_native_capture_external_signature_20261030` adds an
Ed25519-signed, domain-separated checkpoint envelope for **future** captures.
The checkpoint binds the exact raw capture and scored-row SHA-256/byte counts, original
registered model SHA-256 and event population, evaluator session, collector
source and instance, server PID and port, witness source and instance, witness
counter, and preceding checkpoint SHA-256. Checkpoint JSON is canonical UTF-8
with one LF and exactly the registered fields. The signature is over a
domain-separated canonical representation of every unsigned field.

The pure test/review helper verifies the signature against an explicitly
supplied public key, then checks the entire event stream with the existing
prospective sequence, concurrency and native response-model consistency
checker. Crucially, it also requires **every event** to agree with the
independently specified collector/session/server tuple. An internally
consistent capture claiming another collector, session, PID or port is refused
even when its checkpoint is re-signed by the test key.

An independent original-byte join also authenticates the **raw source bytes**
of the canonical 1,319-item GSM8K manifest and original `reap_eval.py`, then
checks each request's exact original serialized prompt/body, POST URL and 200
status, response SHA-256, actual native `choices`/content/finish/usage/timings,
and exact scored-row digest. Every scored `correct` label is recomputed with
the source-bound original GSM8K grader; every trigram statistic is recomputed
from the actual returned text. Source-controlled row digest approvals must
match the signed checkpoint, not just the collector's self-reported hash.
Concurrent requests may start in a different order from canonical task IDs;
the join uses the independently verified exact task ID set, not capture order
as a substitute for task identity.

This does **not** prove that the approved signer was independent, that a key
was held by a trusted external observer, that the model actually served the
request, that timestamps reflect reality, or that a PID was not reused. A
signature authenticates a statement by the holder of the corresponding key;
it does not authenticate the underlying observation. Independent capture of
actual HTTP transport (including header bytes), process-instance/creation-time
evidence, independent key custody,
externally fenced monotonic counter/anti-rollback state, and actual frozen
evaluation/grade approvals are still required. A scored row's wall-clock
`seconds` is not authenticated by a signed collector monotonic interval:
the original evaluator uses `time.time()` and rounds to 0.01 seconds; these
clocks cannot be equated or used to invent missing timing evidence.

## Production admission deliberately refuses

The source-controlled production maps for witness public key, witness and
collector source, capture/rows/checkpoint SHA-256, session/collector instance,
previous checkpoint, server PID/port and approved response alias/HTTP worker
count are **empty**. The production function refuses before accepting
caller-provided evidence. The pure helper is for CPU-only adversarial tests;
its caller-supplied test public key cannot install production authority.

All five original registered arm populations and model hashes are pinned
from the existing temporal source registry. This is not integrated into the
original frozen evaluator or the contributor-owned, untracked raw-HTTP witness;
no actual independently authenticated collector exists for these sessions.
`cryptography` is used only when Ed25519 verification is invoked; if missing,
the helper refuses rather than falling back to a hash. Tests use ephemeral
in-memory keys and are skipped in environments without that optional package.

## Verification and research provenance

Run CPU-only tests without model inference:

`python -B -m pytest -q -p no:cacheprovider tests/test_reap_v2_native_capture_external_signature_20261030.py tests/test_reap_v2_native_capture_temporal_contract_20261030.py`

The tests cover valid-but-BLOCKED signatures, altered bytes, foreign keys,
re-signed source/model/session/PID/port claims, contradictory event owner,
reordered records, excess HTTP concurrency, foreign response-model aliases,
forged canonical prompts, missing HTTP completions, altered scored rows,
duplicate JSON keys, noncanonical serialization, incomplete metadata,
replay against an independently expected predecessor, and absent production
approval. No synthetic test signature is a frozen evidence artifact.

Original completed research-only HumanEval164 Q4 151/164 and Q8 152/164,
paired original certificate SHA-256
`5a2ff5e6ab9f371171de96d368672d52854dd47705c6abb6047de432520893c7`,
and Windows Q4 GSM8K 1224/1319 versus separate Linux Q8 1220/1319 remain
unchanged. Original same-session dense Gate1/Gate2, BF16 chained numerics,
REAP160, random128, HLE, and remaining registered frontier suites are not
qualified by this prospective engineering work.

## Separate original frozen Gate0 source-integrity correction

Independent review also reproduced a conditional **fresh-v2 Gate0 false
acceptance** in the original evidence bridge: it independently regraded
GSM8K correctness but did not recompute `trigram_ratio` from the stored text.
An adversarial synthetic fixture changed 40 candidate rows to repetitive
`x x x x x` (true ratio 0.3333), forged the stored ratio as 1.0, and re-pinned
the synthetic artifact/reviewer hashes. The original bridge reported Gate0
PASS even though independently recomputed Gate0 was FAIL. This did not pass
other gates or authorize release; production approval maps remain empty.

The existing original-path bridge now independently recomputes the original
`round(unique_trigrams / total_trigrams, 4)` statistic for every fresh v2
candidate, dense, random128 and Q4 scored row, including HumanEval. It refuses
source-resealed discrepancies **before** invoking the frozen selector.
Adversarial CPU tests re-pin synthetic rows and independent reviewer hashes
across five original cohorts and require `BLOCKED_UNVERIFIED_FROZEN_EVIDENCE`.
The frozen `reap_eval.py`, `reap_gates.py`, `reap_select.py`, weights, original
F: artifacts and certificates remain unchanged. Separate execution-source
attestation of the selector engine is still an open prerequisite.
