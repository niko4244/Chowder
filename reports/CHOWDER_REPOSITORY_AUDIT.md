# Chowder repository audit

> Audit snapshot: 2026-10-22 UTC. Main 0bc566c5db052b90e3aa1223fef94445d1045dc5; PR #268 fe94342c1073eb39e69be01ba224c63b38057b95; recovery PR #269 e3f0fbd77bc679ea9a856461294cd26b40316562. GitHub code and published evidence were inspected; local Windows workspace inaccessible (Chat On Steroids Core root /strata invalid). Source claims below are **historical reports**, not rerun hardware measurements. No model weights or protected local evidence were accessed in this audit.

## Verified Git state

- Main HEAD `0bc566c5` dated 2026-10-20; compare the source against the [main commit](https://github.com/niko4244/Chowder/commit/0bc566c5db052b90e3aa1223fef94445d1045dc5).
- Open PRs at snapshot: #269 recovery journal, #268 MoE int8, #227 watchdog cadence, #226 metric semantics, #214 unvetted literature drop, #210 campaign controller split, #203 rescued workspace, #202 teacher-free plan, #201 teacher-free pilot, #200 Kaggle dispatch. One GitHub issue #228 flags the literature drop. Historical PR test counts are not fresh CI proof.
- The latest local working tree, live processes, attached storage, complete checkpoint bytes, external Hub policy documents and machine-specific state cannot be inspected: approved /strata root changed. No claim of a clean local git tree or complete model package is warranted.

## Implemented on main

- Transformers/PEFT training, independent evaluation, resumable checkpoint contracts, QLoRA and previously measured Unsloth pilot; real-model CI job exists (`README.md`, `pyproject.toml`, `.github/workflows/ci.yml`).
- Budgeted growth loop with CLI/TUI facade, immutable state and refusal gates (`src/chowder/growth`, `docs/AUTONOMOUS_GROWTH_LOOP.md`).
- REAP observation/pruning/eval/serving scripts; BF16 layer-bulk placement merged in #264 (equivalence tests; GPU speed remained unspecified in merged record).
- Gen-1 training mix builder supports the optional 250-row selfoss_code source and drops over-2048-token rows rather than truncating them; documentation corrected on this audit branch (`docs/MOE_GEN1_MIX_V1.md`).

## Branch work and defects

- #268 holds the int8 loader, exact shape/quantization tests, live census, real-card measurements, and witness mechanisms. Not merged; its gate table lacks gradient, smoke, router-balance and WDDM operator evidence, and its recorded sanctioned gate belongs to pre-default-retention code. See `reports/MOE_INT8_16GB_QUALIFICATION.md`.
- #269 addresses an actual Gen-2 hard termination; its earlier parser could mistake an older terminal marker for a later interrupted run. Added fail-closed parser, atomic-verdict fsync and tests on PR branch at `e3f0fbd77bc679ea9a856461294cd26b40316562`. New CI was in progress when inspected. No automated continuation was added.
- #226 and #210 target older bases and must be rebased/reviewed; do not mass-merge. #214 requires human review under the literature curation policy.
- Gen-2 run 10 had a late ancestor raw-artifact refusal after historical 2.16 GPU-h; amendment 24 and merged #266 describe the evidence-location repair and earlier readiness verification. Gen-2 run 11 was terminated externally, not certified.

## Scope and trust boundary

Two model programs are distinct. Primary Qwen3.8 Native Sparse forbids distillation; REAP-MoE permits declared Fable-derived data only under its own charter and requires Fable-free controls while legal decision N2 is open. No checkpoint, adapter, judge record or metric can be transferred between lineages for certification (`docs/REAP_MOE_LINEAGE.md`).

## Assessment

Platform foundations **PARTIAL** (implementation and historical real tests; fresh integrated verification missing); MoE int8 **PARTIAL**; complete custom REAP model **BLOCKED** by unfinished frozen gates, unverified package and budget feasibility; production-grade autonomous continuation **BLOCKED** by unqualified recovery/resume and safety evidence.
