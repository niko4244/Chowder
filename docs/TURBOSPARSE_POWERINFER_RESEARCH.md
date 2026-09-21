# TurboSparse / PowerInfer Research Note — Chowder Sparse-Architecture Track

**Status:** research note (Phase 1 of the sparse-architecture investigation).
**Grounding:** TurboSparse (arXiv 2406.05955) and PowerInfer (arXiv 2312.12456) papers read directly; Chowder code audited same-day (`dense_to_moe.py`, `conversion_exactness.py`, `moe_instrumentation.py`, `parameter_accounting.py`, `activation_census.py`, `activation_experiments.py`, `sparse_accounting.py`). No marketing claims; every transfer statement below names its mechanism.
**Companion tooling (this branch):** `src/chowder/activation_census.py` (Phase 3 measurement), `src/chowder/activation_experiments.py` (Phase 4 structure test), `src/chowder/sparse_accounting.py` (Phase 9 hierarchy).

---

## 1. What the papers actually establish

### TurboSparse (activation sparsity, training-dependent)

| Fact | Evidence | Consequence for Chowder |
|---|---|---|
| **dReLU**: the FFN activation becomes `max(0, xW_gate) · max(0, xW_up)` — ReLU on **both** projections, not just the gate | Paper §3 (Eqs. 5–7) | Architecturally valid for any SwiGLU-style FFN including qwen3_5; changes the *function*, hence a new architecture |
| ~90% neuron activation sparsity on continued-pretrained models | Paper's measured tables | High sparsity is achievable **but only after the training below** |
| **Requires ~150B-token continued pretraining on 64×A800** to reach that sparsity without quality loss | Paper's training setup | A retraining-dependent architecture change. Nothing here is free at conversion time |
| Top-k% magnitude cutoffs (Eqs. 3–4) quantify the sparsity-vs-quality curve *without* retraining | Measurement methodology | Directly reusable: our census measures the same curve on the dense parent |
| MoE experts are themselves ~85% internally sparse | Paper §5 | The compound hypothesis (expert sparsity × intra-expert sparsity) is real and worth measuring — Phase 9 exists for this |

### PowerInfer (hot/cold execution, inference-time)

| Fact | Evidence | Consequence for Chowder |
|---|---|---|
| Neuron activation follows a **power law**: 26–43% of neurons cover ~80% of activations (LLaMA-class dense models) | Paper's measured distributions | Hot/cold structure plausibly exists in Qwen3.8 parents; **must be measured, not assumed** (Phase 3 census) |
| Hot neurons pinned on GPU, cold on CPU, with an **offline ILP placement solver** over a measured profile | Paper's placement pipeline | The placement-policy shape transfers; the solver assumes batch≤32 regimes |
| **CPU-direct compute beats PCIe transfer at batch < 32** — moving activations to weights loses to moving nothing | Paper's transfer-cost analysis | Offloading is NOT automatically beneficial; our F:/NVMe tiers need measured transfer economics first |
| Adaptive online predictors reach ~93% accuracy on neuron activity; neuron-aware sparse kernels are required to realize gains | Paper's predictor + kernel sections | Inference-only machinery; irrelevant to checkpoint construction, relevant to runtime research (Phase 10) |

---

## 2. Transfer classification (the decision table)

### Transfers directly to Chowder (measurement, no model change)
1. **dReLU-counterfactual activity metric** `(gate_pre > 0) & (up_pre > 0)` — a pure measurement over the existing SwiGLU parent; implemented in `activation_census.py`, verified against hand-computed fixtures.
2. **Top-k% magnitude curve** — sparsity-vs-quality measurement on the dense parent without touching weights.
3. **Hot/cold census + Gini concentration + split-half persistence** — the PowerInfer premise (power-law neurons) is checkable on real Qwen3.8 activations with our census.
4. **Compound active-parameter accounting** — `expert sparsity × intra-expert sparsity` hierarchy with explicit definition ids (`sparse_accounting.py`); normalized definitions are a precondition for any cross-paper comparison.

### Inference-only (runtime research, never architecture claims)
- PowerInfer's placement solver, online predictors, neuron-aware kernels, CPU/GPU heterogeneous execution. These change *how fast a fixed model runs*, not *what the model computes*. They must never be mixed into quality gates (program rule).

### Requires retraining (do not claim at conversion time)
- **dReLU as the model's activation** — TurboSparse's 90% sparsity exists only after massive continued pretraining. Converting a checkpoint to dReLU without that training would measurably damage quality; any such arm is a new architecture with full regression obligations (Phase 8's constraint, unchanged).
- Router retraining under changed expert granularity.

### Conflicts with Qwen3.8 architecture (as it stands)
- Qwen3.8's SwiGLU uses SiLU (smooth, unbounded-below gate); dReLU replaces it — a function change, not a configuration.
- The fused `experts.gate_up_proj` layout bakes expert boundaries into tensor shapes; activation-derived experts must go through the permutation path (Phase 5) that preserves gate/up/down row alignment — `dense_to_moe.py`'s contiguous scheme is the only currently-proven layout.

### Speculative (hypotheses until measured — explicitly NOT facts)
- That the real Qwen3.8 parent exhibits **stable, generalizable co-activation structure** strong enough for activation-derived experts to beat mechanical partitioning. *This is the Phase 4 decision checkpoint; the measurement machinery is now built and tested, the real-parent run is next.*
- That hot/cold placement pays off on this specific machine (RTX 5060 Ti + F: NVMe) — needs the Phase 10 transfer-economics measurement.
- That ~3–4B effective active params/token is reachable via the compound hierarchy — recorded as a hypothesis in `sparse_accounting.py`'s formula document, not a target claim.

---

## 3. Gap analysis: Chowder's existing path vs the research frontier

| Capability | Before this branch | After this branch |
|---|---|---|
| Measure per-neuron dense-parent behavior | none (MoE instrumentation covers experts, not dense neurons) | `activation_census.py`: dReLU counterfactual, frequency/magnitude/contribution, Gini, exact hot-set co-occurrence + JL sketch (256-dim) for scalable co-activation, split-half persistence, atomic provenance-bound artifacts |
| Test whether natural experts exist | not possible | `activation_experiments.py`: 5 groupings (contiguous / random / frequency / sketch-cluster / sketch+contribution), held-out half-B evaluation, 3-part decision rule (absolute held-out ratio ≥ 2.0 ∧ stability ≥ 0.90 ∧ ≥ 1.10× random null) |
| Compound active-param accounting | Phase 11 single-level, **with a real defect** | `sparse_accounting.py` hierarchy + **base-module fix**: Phase 11's formula `total − routed − router` excluded the routed top-k share that IS computed every token (and wrongly subtracted the always-on router). Now `total − routed×(1−top_k/E)`; hierarchy composition cross-checks base exactly |
| Activation-derived conversion | not present (by design: opt-in after Phase 4 evidence) | Phase 5, gated on the Phase 4 checkpoint. `convert_checkpoint` currently hardcodes contiguous partitioning; the row-gather/permutation extension is sketched but NOT written — deliberately, pending real-parent evidence |
| Router teacher (dense-parent supervision) | not present | Phase 6, same gating |
| Hardware placement | Memory Fabric research only | Phase 10 plan in §4 below; no code until transfer economics are measured |

**Defects found and fixed during the audit** (regression-proofed):
1. `parameter_accounting.py` active-parameter formula (above) — would have understated every sparse A-label, including any converted Chowder checkpoint.
2. `activation_census.py` initial hook inferred activity from `gated != 0` — SwiGLU output is never exactly zero, so the census would have reported 0% sparsity for every model. Now computes the true dReLU counterfactual from captured gate/up pre-activations.
3. `activation_experiments.py` farthest-point seeding minimized *min*-seed similarity instead of *max* — seeds could collapse into one co-activation direction. Also: the decision rule's relative-stability term could never pass on clean planted data (both signal and null sit at ~1.0); redesigned as absolute-held-out + stability-floor + relative-advantage.

---

## 4. Phase 10 placement plan (runtime track, separate lane)

Tiers for this machine: primary GPU (RTX 5060 Ti 16 GiB), CPU/RAM, NVMe (F:) only if latency justifies. Measurement protocol before ANY placement code:
1. Measure raw PCIe host↔device bandwidth and NVMe→RAM→PCIe latency with a synthetic 100–500 MiB expert-sized block mover (no model needed).
2. Compute the break-even: a cold expert is worth offloading only if `p(active) × t_recompute < p(miss) × t_transfer` under the measured numbers — PowerInfer's own analysis says CPU-direct wins below batch 32, which is our regime; verify locally.
3. Only then wire candidate policies (hot-pin / warm-CPU / cold-NVMe / predictor-prefetch) into a benchmark harness with tokens/sec, first-token latency, peak VRAM, host RAM as reported metrics.

---

## 5. The Phase 4 checkpoint (unchanged, now executable)

> **Does the real Qwen3.8 parent exhibit stable, generalizable neuron co-activation structure strong enough that activation-derived experts outperform mechanical/random partitioning?**

Procedure when GPU time is available (the parent tournament owns it first — this measurement is CPU/GPU-light and MUST NOT contend with a live tournament):
1. Run `ActivationCensus` over parent A (dense, local dir) on a calibration corpus that is **not** protected tournament content (e.g. public-domain text bundles, hashed and recorded in the profile provenance).
2. `run_grouping_comparison` per layer at candidate expert counts (8/16/32).
3. Apply the verdict rule. If no grouping passes: record the negative result HERE and in HANDOFF, keep the contiguous converter, close the investigation phase with evidence. If one passes: proceed to Phase 5 (opt-in converter with reversible permutation manifest) and Phase 6 (router teacher), each with its own regression ladder.

**Scientific constraints carried from the directive:** the frozen parent-evaluation protocol is untouched; protected tournament content is never calibration data; no TurboSparse number is claimed to transfer; negative evidence is preserved.
