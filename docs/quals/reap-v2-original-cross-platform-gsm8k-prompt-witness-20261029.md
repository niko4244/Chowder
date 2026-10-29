# Original cross-platform GSM8K native prompt-length witness (2026-10-29)

This **research-only, CPU-only** verification compares three independently
archived original model-serving sessions: Windows Q4_K_M GSM8K1319, Linux
Q4_K_M GSM8K1319 and Linux Q8_0 GSM8K1319. The original HumanEval164
sessions are used only as immutable Linux native task-population evidence;
neither HumanEval arm is rerun or regraded by this witness.

Each original response row is correlated with one native llama-server task by
the unique archived completion-token/generation-rate timing fingerprint. The
server log then supplies the declared prompt-token length for that task.
All source files are read with existing immutable SHA-256 pins and the
existing independent Windows/Linux graders and native lifecycle verifiers
are required to pass before comparison.

**Measured original result:** all **1,319/1,319** canonical GSM8K item IDs
have identical declared native prompt-token lengths across all three
sessions, with **137,469** declared prompt tokens per arm. The canonical
per-item length map has SHA-256
`d148e024faa57d0df7dfb9b0e8cbdee024b68fdb9f6b505c89d989f4f2e251ef`.
An adversarial swap of two prompt lengths preserving the aggregate histogram
is refused; a consistent fabricated map is also refused by the frozen digest.
A separate adversarial test confirms that changing only response text leaves
the numeric native timing match unchanged; the original archived-file SHA pin
would reject that alteration, but timing alone cannot authenticate text.

**Scope limit:** equal token counts do not authenticate prompt bytes, HTTP
request/response pairs, model weights at execution time, or independent
execution authority. Numeric timing is correlation, not cryptographic
attestation. The three sessions are **not** a frozen same-session Q8-vs-dense
comparison. No Gen-0 gate, release, or selection is qualified. The historical
REAP128-v1 Gate 1 failure, unknown REAP160 speed, and unresolved Amendment 4
statistical authority remain unchanged.

Gen-0: `{"status":"BLOCKED","chosen":null,"release_approved":false}`.
