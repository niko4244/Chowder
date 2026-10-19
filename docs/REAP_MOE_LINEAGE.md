# REAP-MoE lineage — the second sanctioned model program (2026-10-18)

**Owner decision N1 (Nik, 2026-10-18).** The custom Qwen MoE runs as its
own lineage, beside the Qwen3.8 Native Sparse Program (the primary
lineage), never inside it. This document is the lineage's charter. A
change to any rule here is a dated amendment, not an edit in place.

## What the lineage is
- **Gen-0 base:** a REAP-pruned Qwen3.6-35B-A3B, keeping 128 of 256 routed
  experts per layer (~18.5B total parameters), built with
  `chowder_batch/reap_*`.
- **v1 vs v2:** which of v1 (workload calibration, no Fable) and v2
  (calibration v2: workload + Fable + public tool-use data + Ornith
  curriculum) becomes Gen-0 is decided by
  `docs/REAP_QWEN36_PREREG.md` and its amendments, never by preference.
- **Fixed until superseded:** Gen-0's content digest
  (`chowder.model-content.v1`), state root and trusted-ancestor record are
  created once, when Gen-0 is frozen.
- **Goal:** improve agentic tool use, code repair and math reasoning
  without regressing protected skills. The model is served locally at Q4
  on a 16 GB card.

## Separation from the primary lineage
- **No transfer, ever:**
  - no adapter, checkpoint or evaluation report moves between the two
    lineages in either direction;
  - no result from one lineage satisfies a gate in the other.
- **Separate state:** state roots, campaign declarations and judge records
  are separate; a campaign declares which lineage it belongs to.
- **Rule scope:** the primary lineage's "no distillation, ever" rule is
  untouched. The permission below applies to this lineage alone.

## Distillation permission (scoped)
- Fable-derived SFT data is permitted **only** as a declared input:
  - every campaign's pre-registration names each Fable source, its digest
    and its licence;
  - the providers enforce `allow_fable=True` (`OrnithCurriculumProvider`,
    `FableAgenticProvider`);
  - every served row carries `fable_derived` provenance.
- **Licence:** the Fable sources are AGPL-3.0. The source dataset cards
  also describe Fable 5 as an Anthropic model whose usage terms restrict
  training on its outputs. **That second question is owner decision N2 and
  is still open.**
  - Until N2 is recorded, a campaign that uses Fable data must also run a
    **Fable-free control arm**.
  - The control arm measures the effect and isolates the exposure.
- **Never training material** in this lineage, Fable or not: protected
  evaluation items, the held-out Nemotron split (`is_reserved_for_evaluation`),
  and the `tool-use@reap-moe-v1` items.

## Gates (all pre-registered; none waived)
1. **REAP quality and speed:** `docs/REAP_QWEN36_PREREG.md`. Gate 0 is
   termination. Gates 1 and 2 are GSM8K and HumanEval against dense. The
   control is random-128. Speed and fit are measured at Q4_K_M.
2. **Training placement:** `docs/quals/MOE_TRANSIENT_GRAD_PLACEMENT_PREREG_2026-10-18.md`.
   The gates are load census, gradient, memory, throughput, smoke and router
   balance, all measured on reap128 before any campaign declares
   `expert_placement: cpu-transient-grad`.
3. **Tool-use instrument:** `docs/quals/TOOL_USE_REAP_MOE_V1_PREREG_2026-10-18.md`.
   The item file is frozen; any change makes a new version id.
4. **Every growth campaign:** the standard Chowder pre-registration,
   readiness READY, settlement, independent candidate arm and judge. Gen-2
   of the primary lineage is the reference implementation.

## Still open (owner decisions)
- **N2:** Fable / Anthropic usage terms. Until it is recorded, the control
  arm above is mandatory.
- **N3:** the promotion judge. The old plan required a "full Fable judge",
  but Fable 5 is reportedly suspended. The proposal is a frozen local judge
  panel plus execution-verified scoring. This needs a decision before the
  first promotion in this lineage.
- **N5:** promotion approval stays human for every generation.

The roadmap for this lineage lives in the owner's hub:
`~/docs/ROADMAP-CUSTOM-QWEN-MOE-AND-CHOWDER-2026-10-18.md`, phases P0–P8.
