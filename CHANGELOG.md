# Changelog

## 0.4.0 — 2026-09-21

### Spark 2.5 local-model support

- Explicit SHA-256-pinned local custom-code path (`local_model_compat.py`):
  custom-code files are verified by digest before import, filename traversal
  is rejected, and known Transformers 4.57→5.x incompatibilities (tied-weight
  mapping, causal-mask keyword changes) are patched for pinned models only.
  `trust_remote_code` remains hard-disabled on the autonomous path; the
  digest gate lives inside the patch/import helper itself so a caller cannot
  skip verification. Supported across training, evaluation, memory
  preflight, and architecture preflight, and covered by protocol identity
  (digests flow into the protocol contract).
- New scoring modes `reasoning_answer_match` and
  `reasoning_final_number_match` for templates that open `<think>` in the
  prompt and emit a trailing `</think>` end-of-turn marker (Spark-style);
  existing scoring defaults unchanged.
- Windows adapter-save retry: `save_pretrained` now retries once on the
  transient `os error 32` file lock, catching `SafetensorError` (which is
  not an `OSError`). Regression-tested.

### Goal-lifecycle honesty

- `ProjectRunOutcome.succeeded` is true only when the lifecycle records
  `STOP_GOALS_MET`; promotion, generation limits, budget exhaustion,
  plateaus, refusals, cancellations, and crashes remain non-success.
- Promoted-but-unmet runs report their promotion without claiming
  completion; CLI and TUI status/exit messaging derive exclusively from
  `ProjectRunOutcome`.
- Registry close-out audit (`registry_audit`) runs on every outcome path,
  including persisted-terminal and parent-already-terminal early returns.
- `legacy_unbounded` compatibility mode documented in
  `docs/GOAL_LIFECYCLE_CONTRACT.md`: explicit config opt-in plus all
  metrics unbounded, never combinable with bounded goals.

### Registry

- New `teacher_signals` table backing the existing teacher-signal
  record/list/get methods (previously referenced a missing table).
