# AUDIT — `chowder_agent_batch_001.jsonl` (3 SFT + 1 pref pair)

Scope: checked against the CHOWDER_AGENT and CALL_FINALIZER specs, the batch
schema, and the "wrong habit" risk list. Behavioral gates in `validate.py`
were re-run before this audit. No CALL_FINALIZER records exist in batch 001
(all four are CHOWDER_AGENT), so CF-specific rules below are forward-looking.

## Spec compliance

| Rule | Status |
|---|---|
| Exactly 3 SFT + 1 pref pair | OK |
| Schema fields present on all records (incl. top-level `messages` on pref) | OK |
| Pref pair: identical input, chosen more correct, reason + evidence | OK |
| Token budget ~2048 (est. 881/951/874/894) | OK |
| `source: teacher_synthetic` (no user-failure evidence supplied) | OK |
| Verification statuses reflect reality (3 executed `passed`, review `not_run`) | OK |
| Dedup: 4 distinct `task_family` values | OK |
| No fabricated tool results in assistant targets (leakage gate ran) | OK |
| Benchmark/near-duplicate contamination: no public-benchmark items used | OK (synthetic, original) |

## Findings — could these teach a wrong habit?

1. **[Medium] 0003's tool-call format is bare JSON, not an explicit tool-call
   envelope.** If the student's runtime uses native tool-call objects
   (`{"tool_calls": [...]}` or provider-specific syntax), training on bare
   assistant-JSON may produce format mismatch at inference. This is a
   *formatting assumption*, not a correctness error. Recommend: keep the
   records (protocol is stated in-task), but add one batch using your real
   runtime's envelope before fine-tuning.

2. **[Minor] 0001 embeds a full working script as "the" plan.** Risk: the
   student learns to emit maximal solutions when a plan skeleton was asked
   for. Mitigated: the task explicitly requests `write` content and the
   script is the smallest complete verifier (~30 lines), plus a
   counter-habit appears in batch 002 (0005 asks for plans only, no code
   dump).

3. **[Minor] Negative examples are implicit.** Failure modes are documented in
   `failure_mode`, but the SFT targets contain no contrastive turns. DPO on
   the pref pair partially covers this. Recommend 1-2 contrastive examples
   per 10 SFT records.

4. **[Minor] 0002's `usage:` line says `importer.py` while the record never
   names the script.** Harmless (the task prompt shows `sys.argv` handling),
   but a student may overfit to that literal filename. Recommend renaming in
   a future revision or keeping the filename inside the task prompt.

5. **[Info] Error-message coupling.** Expected outputs pin exact
   `JSONDecodeError` message text ("Expecting property name enclosed in
   double quotes"). Verified true for CPython 3.11 here; other Python
   versions could phrase it differently. Fine for a local, pinned runtime;
   do not port these expected strings across interpreters without re-running
   the verification.

## Already-corrected during production (recorded for traceability)

- `JSONDecodeError.lineno` trap (always 1 for single-line parses) — fixed to
  `enumerate` line numbers; the trap itself is now taught in 0001/0002.
- Premature-stop shape in 0003 (stop emitted in the same turn as the final
  `run_tests`) — rebuilt to strict one-action-per-turn, stop only after
  observing `1 passed`.
- Pref pair and all executable expectations now match *observed* behavior,
  not intended behavior.

## Verdict

No record teaches an incorrect capability. Fix #1 (format envelope) and #4
(filenaming) before training if the student runtime differs from the
batch's stated protocol; #2/#3 are batch-002 policy, applied.
