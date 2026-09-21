# P12 (scoped slice) — Unsloth/VLM-wrapper result triage with guard-shaped evidence fields

**Date:** 2026-09-12. **Authority:** [2026-09-12 reconciled plan](../../../Chowder-handoff-update/docs/superpowers/plans/2026-09-12-chowder-training-first-reconciled.md), task P12 (Unsloth-inventory portion). **Method:** read-only header/key checks, artifact hashes, registry rows via SQLite `mode=ro`, and re-use of already-recorded measurement evidence. **No generation was reevaluated; no new GPU work; no registry row, artifact, or evidence file was modified.** Corrections are linked as new audit evidence per the plan.

Guard field vocabulary below is the one fixed in `adapter_guard` (2026-09-12): `saved_tensors` / `live_lora_parameters` / `matched_keys` / `lora_B_parameters` / `lora_B_nonzero` (verified nonzero) / `lora_B_zero` (verified zero) / `lora_B_unreadable` (raising storage or nonfinite) + offending names. Historical artifacts predate the split buckets, so fields are marked `(not recorded)` where an older report collapsed or omitted them; `(verified later)` marks values established by post-hoc direct measurement already on file (`guard-real-verification.json`, `unsloth-adapter-diagnosis.json`), not by new computation.

## Trust-status summary

Priority per plan: results that influenced candidate choice/promotion first.

| Run (workspace) | Worker revision class | Adapter sha256 (first 16) | Saved keys namespace | Guard fields at eval time | Score vs baseline (n=20, one run) | Registry status | **Trust status** |
|---|---|---|---|---|---|---|---|
| `level2-unsloth` / `…256027c96e72` (v1) | pre-`bf190e6` VLM wrapper | `c21fa2959953f8d9` | **400/400 `…model.language_model.layers…`** | none recorded; `adapter_loaded: true` (misleading) | 0.30 vs 0.30 | `rejected` | **CONFIRMED-INERT (as evaluated)** |
| `level2-unsloth-v2` / `…f7aae774d09a` | post-fix, `text_only_requested: true` | `781a0089edb1188b` | 256/256 text-decoder | `matched_keys` 256/256, `lora_B_nonzero` 128/128 (recorded in eval evidence) | **0.60 vs 0.30** | `passed` | **VERIFIED-APPLIED** |
| `level2-unsloth-v3` / `…f35e18c0ea5a` | post-fix | `781a0089edb1188b` (= v2's, byte-identical) | text-decoder | n/a — run failed before candidate eval | none (no candidate eval) | `failed` | **NOT A CANDIDATE** (execution failure; artifact anomaly) |
| `level2-unsloth-v4` / `…9a3a5d69ef48` | post-fix, `text_only_requested: true` | `c3da6914a677efe1` | 400/400 text-decoder | `matched_keys` 400/400, `lora_B_nonzero` 200/200; `target_coverage` measured (200 modules, `unmatched: []`, `allow_unmatched: false`) | 0.45 vs 0.30 | `rejected` | **VERIFIED-APPLIED** (gate rejected on quality grounds) |
| Control: `level2-transformers-v4` adapter (used in the real guard verification) | Transformers/PEFT path | n/a (hash in `guard-real-verification.json` context) | 400/400 text-decoder | `saved_tensors` 400, `matched_keys` 400, `lora_B_nonzero` 200 `(verified later)`; `max_abs_logit_delta` 14.5 | +0.15-class gain (its own run lineage) | — | **VERIFIED-APPLIED** |

## Per-run detail

### v1 — `level2-unsloth` (the priority item; a gate result here was void)
- **Saved adapter:** 400 tensors (200 A + 200 B), every key carrying `.model.language_model.layers.` — the `*ForConditionalGeneration` wrapper namespace. sha256 `c21fa2959953f8d9…`.
- **Loadability against the evaluator's text-only CausalLM:** `matched_keys` **0** of 400 saved; live LoRA parameters on the evaluator model 256 (128 modules), all `lora_B` **verified zero** `(verified later via the guard's real-model run: verdict REFUSED, zero-overlap class)`.
- **Output-delta evidence `(verified later)`:** `max_abs_logit_delta: 0.0`, `top_token_unchanged: true` (`unsloth-adapter-diagnosis.json`) — the scored candidate was provably the base model's behavior. A score equal to baseline plus a measured zero logit delta is **confirmed-inert-as-evaluated**, not merely "unresolved".
- **What this does and does not invalidate:** the evaluation and the gate are void (experiment row `rejected` stands as the gate's own decision); the training itself adapted 128 *decoder* leaves under the wrong object's key namespace (injected_by_leaf: gate/up/down 32 each, q/k/v/o 8 each; all 72 linear_attn modules missing). The run's **cost numbers (6.24 GB peak VRAM, 177.9 s accelerator time per its own registry row, with the wrapper-resident vision tower loaded) are the wrapper's cost, not the model's** — `CAN_CHOWDER_TRAIN_THIS_MODEL.md` already carries this correction (commit `f55c442`) and quotes the post-fix 5.84–5.96 GB / 65 s as the honest comparison.
- **Influence:** no promotion resulted (registry `rejected`). Its lingering influence was the "cheaper" backend-preference claim, now disclaimed in the doc.

### v2 — `level2-unsloth-v2` (influenced a recorded preference: the 0.60 figure)
- **Saved adapter:** 256 tensors (128 A + 128 B), text-decoder namespace, sha256 `781a0089edb1188b…`. Leaves: gate/up/down ×32 layers, q/k/v/o ×8 layers (128 modules; linear_attn leaves not requested by this run's config).
- **Guard fields at eval time (recorded in the run's own evidence):** `matched_keys` 256/256; `lora_B_parameters` 128; `lora_B_nonzero` 128; `lora_B_zero`/`lora_B_unreadable` `(not recorded — single-bucket era, all-nonzero implied)`.
- **Outcome:** quality 0.30 → 0.60 (n=20, one run each), experiment `passed` — this is the "0.60" in the backend-comparison table. With a fully matched key set and all-B nonzero, **verified-applied**; the score change is real for this run. The doc's own disclaimer stands: not a controlled quality comparison, n=20.

### v3 — `level2-unsloth-v3` (failed run; artifact anomaly recorded)
- **Registry:** experiment `failed`; events record `TargetCoverageError: these requested target modules adapted NOTHING: ['in_proj_qkv', 'in_proj_z', 'out_proj']` — the coverage guard fired before training; **no candidate evaluation exists**; `failure_records: 0`.
- **Anomaly:** the workspace's adapter directory is **byte-identical** to v2's (sha256 `781a0089edb1188b…` for both `adapter_model.safetensors`). A failed run cannot have produced a trained adapter; this file is a copy of v2's artifact with unexplained provenance. It is **not independent evidence** and must not be cited as a separate trained candidate. (Left in place; flagged for the owner.)

### v4 — `level2-unsloth-v4` (verified-applied; gate rejected on quality)
- **Saved adapter:** 400 tensors, text-decoder namespace, sha256 `c3da6914a677efe1…`; leaves match the full 10-name request (in_proj_qkv/in_proj_z/out_proj ×24, gate/up/down ×32, q/k/v/o ×8 — per-tensor counts ×2 for A+B).
- **Guard fields at eval time:** `matched_keys` 400/400; `lora_B_nonzero` 200/200 `(zero/unreadable split not recorded)`. **Worker-side `target_coverage` present:** `status: measured`, 200/200 modules, `unmatched: []`, `allow_unmatched: false` — the strongest per-component record in this lineage.
- **Outcome:** 0.30 → 0.45, experiment `rejected` — a real, applied adapter that the gate declined. Trust status **verified-applied**; the rejection is a quality-gate outcome, not an evidence defect.

## Boundaries of this triage

- **Covered:** the four `level2-unsloth*` workspaces (the Unsloth lineage that touched candidate-facing claims) plus the Transformers control adapter cited in the real guard verification.
- **Not re-opened here:** the `realtrain-gsm8k` / `-2` pair (already audited 2026-09-12; its reload used text-namespace keys with 400/400 matched — unaffected by the wrapper defect); the level2-transformers v1–v7 lineage beyond the one control; pre-Chowder Unsloth experiments outside `_a4b` if any exist; the fraction-sweep wording corrections (separate P12 items, already corrected in the audit).
- **Rule applied:** a score equal to baseline is *not* by itself proof of inertness (v1 required the direct logit-delta measurement to be confirmed-inert); a changed score is *not* by itself proof of complete key coverage (v2/v4 rely on recorded matched-key/coverage fields).

## Re-verification

Adapter hashes: `sha256sum` on the six `adapter_model.safetensors` paths in the workspaces above. Registry rows: SQLite `mode=ro` on each `level2.db` (`experiments`, `results`, `run_events`). Namespace check: `saved_adapter_keys()` from `chowder.adapter_guard` (or the inlined worker copy) and the `.language_model.` substring count. Any divergence from this file after 2026-09-12 is new evidence and should be linked, not merged silently.
