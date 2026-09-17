# Generation-0 Evaluation Freeze — Result (2026-09-17)

The dense abliterated parent
(`F:\llm-models\Qwen3.8-9B-abliterated-25-bf16`, content digest
`59e767aa…7555f`) is frozen as **Generation 0** of the growth lineage —
evaluation-only, zero training, per
`GEN0_EVAL_PREREG_2026-09-16.md` as amended by
`GEN0_EVAL_AMENDMENT1_2026-09-17.md`.

## Freeze table

| Artifact | Path | Content |
| --- | --- | --- |
| Eval report | `freeze/eval-report.json` | 7 runs: 2 measured battery + diagnostics + 4 honest UNMEASURED |
| Capability profile | `freeze/capability_profile.json` | 4 skill estimates (all 0.0, confidence 0.267 — two benchmarks) |
| Contamination manifest | `freeze/contamination_manifest.json` | All 6 benchmarks `UNKNOWN / "not checked"` — no training pool exists |
| Frontier snapshot | `freeze/frontier_snapshots.json` | `gen0-frontier`, empty reference set (out of prereg scope) |
| Lineage record | `freeze/generations.json` | `gen0`, parent=None, ledger `PROMOTED` as baseline establishment |
| Scoreboard | `freeze/scoreboard.md` | Registry-qualified rows with honest marks |
| Digest | `freeze/FREEZE_DIGEST.json` | **freeze digest `5c8b18ab691b8bc3…`** over all six artifacts |

## Execution history (both attempts preserved)

| Attempt | Verdict | Aggregate | Cause |
| --- | --- | --- | --- |
| 1 | `REFUSED_BUDGET` (runner's own check) | 2.652 GPU-h vs 1.00 | mgsm ran the 12-language parent task (2.347 GPU-h); raw-model protocol without the frozen chat template |
| 2 | `COMPLETE` | **0.582 GPU-h vs 1.00** | Amendment A1 (mgsm_direct_en), A2 (chat template), A3 (measured sub-budgets) |

## Measured results (attempt 2, protocol: chat template, greedy, seed 1234)

| Measurement | Result | Cost | Sub-budget |
| --- | --- | --- | --- |
| math500@2024-04 (minerva_math, limit 4) | 0.000 exact_match | 0.234 GPU-h | 0.30 ✅ |
| mgsm@2022-11 (mgsm_direct_en, limit 24) | 0.000 exact_match | 0.223 GPU-h | 0.25 ✅ |
| Generation diagnostics (16×128, one batch) | EOS 0.000, cap-hit 1.000, trigram 0.988, loops 0 | 0.110 GPU-h | 0.12 ✅ |
| Load + dispatch (bf16, 32 CPU layers) | 3.94 GiB peak (probe), 0.0148 GPU-h | — | 0.02 ✅ |
| **Aggregate** | — | **0.582 GPU-h** | **1.00 ✅** |

## Honest non-measurements (UNMEASURED is not zero and not pass)

- `ifeval@2023-11`, `mmlu_pro@v2`, `bbh@2023-05-03`: cost arithmetic recorded
  (at the measured ~29 s/request amortized offline cost they exceed the ceiling).
- `gpqa_diamond@2025-05-30`: gated dataset; hub refused the stored credential,
  offline refused on gated metadata.
- Frontier comparison: out of prereg scope; the snapshot preserves that fact
  with an empty reference set rather than inventing gaps.

## What the baseline says

The dense parent, as a raw base model under a chat template: **0.0 on the
measurable math slices** (28/24 greedy samples) and **it never terminates**
(EOS rate 0.0, cap-hit 1.0) — though it does not literally loop (distinct
trigram 0.988, zero obvious loops). Non-termination is the concrete,
measurable deficit a first growth campaign would target; the 0.0 scores on
Minerva-formatted math are consistent with a base (non-instruct) model that
does not emit the graded answer format.

## Discipline record

- Prereg written before any measurement; amendment written after attempt 1's
  refusal and before attempt-2 compute, changing scope/protocol only where
  attempt 1 proved the frozen text unexecutable-as-frozen; the aggregate
  ceiling never moved.
- The builder (`freeze_builder.py`, copied to `docs/gen0/`) mechanically
  verifies: attempt-1 refusal preserved, attempt-2 `COMPLETE`, per-row
  amendment contract (template applied, mgsm scope), sub-budgets, identity
  digest, and the 14.5 GB load ceiling — then refuses with exit 2 otherwise.
- Built entirely on the growth-system APIs merged on `main` (registry,
  `EvalReport`, `Scoreboard`, `ContaminationFirewall`, `SnapshotStore`,
  `GenerationLedger`, `PromotionDecision`, `build_profile`).
- Everything above lives outside the repository in the protected evidence
  root; this document and the builder copy are the only in-repo records.

## Not done here (deliberately)

- No training of any kind; no Generation-1 campaign; no autonomous cycle.
- The growth `TrainingFn` → `chowder train` qualification (PR #167) remains
  the gate before any real first campaign.
- Multilingual MGSM expansion and frontier reference seeding are future
  freezes with their own preregistrations and budgets.
