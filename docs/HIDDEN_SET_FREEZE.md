# Hidden evaluation set — freeze record

Part of Task 7, `docs/EXECUTOR_INVESTIGATOR_PLAN.md`. Written before
`tests/fixtures_incidents_hidden.py` was read by anything other than the
author, and before Task 8 runs the investigation loop against any of these
cases.

## Why this file exists

The classifier (`classify_signature` in `src/chowder/incident.py`), the
hardware-floor table (`src/chowder/probes.py`), and the hypothesis
generator's rule table (`src/chowder/hypothesis_generation.py`) were all
written by the same session now writing the "held-out" hidden fixtures.
File separation (`fixtures_incidents_hidden.py` vs. `fixtures_incidents.py`)
prevents literal training-on-the-answer — a hidden fixture can't reuse dev
fixture text verbatim by accident. It does not prevent the author knowing
the rules' exact shape while choosing what to write. The freeze below
guards against one specific failure mode of that risk: the rules changing
*after* this freeze to conveniently match a hidden case, which would let a
correct-looking result actually be "the target moved," not "the rule
generalized." It does not, and cannot, retroactively make the rules'
content unknown to whoever wrote both.

## Authorship deviation, recorded rather than silently accepted

The plan's preferred mitigation — drafting these fixtures via an
independent session or reviewer working only from the original request's
plain-English incident-class descriptions, without reading
`classify_signature`'s rule table first — was not practical for this
iteration (single continuous session, no second reviewer available). This
is the weaker guarantee: file-separation-plus-freeze, not
independent-authorship-plus-freeze. Recorded explicitly per the plan's own
instruction, rather than left unstated.

## Frozen file hashes

Computed 2026-09-01, immediately before `tests/fixtures_incidents_hidden.py`
was written, from the exact working-tree content at that moment (including
this freeze's own Task 7 rule additions — see "What was added before the
freeze" below, which happened first, deliberately).

| File | sha256 |
|---|---|
| `src/chowder/incident.py` | `d43dd2b16f05cf98c6ddd61d913dd489f6386bb1284b1884fd35cdef37eb0cea` |
| `src/chowder/probes.py` | `b10e8b4685d4399f4cc2906f9d21f5e59b0499092298b9feecc7440fc82a84ec` |
| `src/chowder/hypothesis_generation.py` | `21b35ae5f30635d37681d6f72f7e9b29bfad20950d841337b08a5117f9698cfa` |

`tests/test_hidden_set_freeze.py::test_freeze_intact` recomputes these
hashes at collection time and fails loudly if any of the three files has
changed since — a rule change after this point must not be allowed to
silently improve a hidden-set result. If a genuine, unrelated bug fix in
one of these files is ever needed after this freeze, the correct sequence
is: fix it, re-run the *dev* set (Task 6) to confirm nothing broke, decide
explicitly whether the hidden set stays comparable, and re-freeze with a
new hash and a dated note here — never silently update the hash to make
`test_freeze_intact` pass again.

## What was added before the freeze (legitimate scope work, not case-fitting)

Two rule-table extensions landed in `incident.py` *before* the hashes
above were computed, motivated independently of any specific hidden
fixture's exact wording:

- **`SignatureKind.ARTIFACT_CORRUPTED`** (new enum member) plus a
  `classify_signature` rule matching `"checksum mismatch"` /
  `"sha256 mismatch"` / `"hash mismatch"`. Motivated by the original
  request's corrupted-download row and `ArtifactIntegrityProbe` (Task 1),
  which already had no signature_kind of its own to attach evidence to.
- **`NETWORK_TRANSIENT`** gained two needles: `"service unavailable"` and
  `"too many requests"`. The existing rule only covered
  connection-level failures (`RemoteProtocolError`, `ConnectionError`,
  etc.); HTTP 5xx/429 status-code phrasing is an equally common transient
  network failure shape and had no coverage at all.

Both are general-purpose phrase additions (not hidden-fixture-specific
strings), but both were added specifically *because* drafting the hidden
set exposed the gap — which is itself the contamination risk named above,
made concrete: the author noticed a rule gap while writing a test for that
exact gap, then closed it before freezing, rather than an independent party
discovering whether the pre-existing rule already generalized. Recorded
here rather than presented as if the rule always covered this.

## Plan-count correction

`docs/EXECUTOR_INVESTIGATOR_PLAN.md`'s Task 8 section previously said the
hidden set has "7" cases. The Task 7 enumeration it was built from lists
six original rows plus two new ones from the Context table — 6 + 2 = 8,
not 7. Corrected in the plan doc to reference this file. Same category of
mistake as the "9 vs. 10 dev fixtures" inaccuracy an earlier review caught
in this plan; worth naming plainly rather than fixing quietly, since this
document exists specifically to keep this benchmark's own bookkeeping
honest.

## Pre-registered predictions

Written before Task 8 runs `classify_signature`, the generator, or the
investigation loop against any of these 8 cases. A mismatch between a
prediction here and Task 8's actual result is a **finding to write up**
(which case, why, what it reveals about the rules' real coverage) — not a
prediction to quietly edit until it matches, and not a rule to quietly
loosen until the case passes.

| Case | Fixture | Expected `signature_kind` | What it tests |
|---|---|---|---|
| H1 | `HIDDEN_QWEN3_5_ARCH_NOT_RECOGNIZED_A10` | `DEPENDENCY_INCOMPATIBLE` | Hardware-independence: same message as the real dev fixture, different accelerator (A10 vs. T4). Nothing in this rule should reference hardware. |
| H2 | `HIDDEN_LLAMA_CUDNN_EXECUTION_FAILED_T4X2` | `CUDA_EXECUTION_FAILED` | The rule's second needle (`cudnn_status_execution_failed`), never exercised by a real dev fixture (only `cublas_status_execution_failed` was), on a different model (Llama, not Qwen3.5). |
| H3 | `HIDDEN_DISK_FULL_MID_DOWNLOAD` | `UNKNOWN` | Fails safe into a real Investigation rather than misclassifying as `ARTIFACT_NOT_FOUND` — the file's location is correct, there's just no room to write it. |
| H4 | `HIDDEN_HF_503_SERVICE_UNAVAILABLE` | `NETWORK_TRANSIENT` | HTTP-status-code phrasing rather than `RemoteProtocolError` wording — the gap that motivated the pre-freeze `"service unavailable"` needle addition above. |
| H5 | `HIDDEN_QWEN3_5_ATTRIBUTEERROR_VERSION_GAP` | `DEPENDENCY_INCOMPATIBLE` | The rule's `attributeerror` needle, distinct wording from the dev fixture's "does not recognize this architecture" phrasing. |
| H6 | `HIDDEN_WRONG_MACHINE_SHAPE_V100_REQUEST_P100` | `HARDWARE_INCOMPATIBLE` | A genuinely different wrong `machine_shape` string than the dev fixture's `"NvidiaTeslaT4x2"` — tests classification, not memorization of one string. |
| H7 | `HIDDEN_ATTENTION_OOM` | `CUDA_OOM` | A third real memory-capacity failure shape (attention computation), distinct in stage from both dev-set OOMs (kbit-prep; fp32-logits-upcast). |
| H8 | `HIDDEN_CHECKPOINT_CHECKSUM_MISMATCH` | `ARTIFACT_CORRUPTED` | The pre-freeze `ARTIFACT_CORRUPTED` addition above, on a case distinct from `ARTIFACT_NOT_FOUND` (wrong content, not a missing file). |

No prediction is made here about what Task 8's *remediation* layer will do
with any of these (resolve / abandon / a specific config_patch) —
`hypothesis_generation.py` was frozen in its Task-6 state and does not yet
have a candidate for `CUDA_DEVICE_MISMATCH`'s hidden analogue or for
`ARTIFACT_CORRUPTED` at all; whether Task 8 adds one, and whether that
counts as legitimate pre-freeze-style scope work or a same-fixture-fitting
risk, is Task 8's call to make and record, not pre-judged here.
