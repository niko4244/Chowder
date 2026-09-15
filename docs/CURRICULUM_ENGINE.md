# Curriculum Engine

The curriculum engine answers one question:

> Given Model N's capability profile and failure bank, what should it learn
> next?

It never trains on whatever score is lowest. Priority is weighted evidence,
and every plan item carries the trace that produced it.

## Priority

`CurriculumEngine.prioritize()` ranks skills by:

| Component | Default weight | Source |
| --- | --- | --- |
| weakness magnitude | 0.30 | 1 − skill estimate |
| importance | 0.20 | configurable per-skill importance map |
| confidence | 0.15 | evidence-weighted estimate confidence |
| failure frequency | 0.15 | banked failures for the skill's benchmarks |
| frontier gap | 0.10 | distance to the frontier reference |
| trainability | 0.05 | availability of verifiable material |
| cost efficiency | 0.03 | GPU-hours per expected gain |
| regression risk | 0.02 | estimated disturbance to protected skills |

Weights are configurable (validated against the closed set); the defaults are
the current policy, not a secret.

## The mixture

Each cycle composes five roles from evidence — the proportions are decided,
never hard-coded:

- **TARGET** — examples directed at the current weaknesses;
- **PRESERVE** — consolidation material for existing strengths;
- **GENERAL** — high-quality broad instruction/reasoning;
- **REPLAY** — prior failure classes already repaired (anti-forgetting);
- **STRETCH** — harder examples slightly above current capability.

## Decision provenance

Every `CurriculumItem` carries a `decision_trace`:

- the numeric `components` and the `weights` that produced its priority;
- the `mixture` shares actually chosen and why;
- `weakness_evidence` naming the benchmarks and failures;
- the protected regression set it must not disturb;
- verification method per item (executable tests, symbolic/numeric,
  multi-judge).

The CLI surfaces all of it: `chowder growth curriculum profile.json`.

## Difficulty calibration

`difficulty.py` estimates item difficulty from base-model success rate,
stronger-model success rate, reasoning-step counts, and solution length, then
bands items (trivial / easy / medium / hard / expert). Curricula move upward
progressively rather than presenting only trivial or only impossible
examples.

## Generate analogues, not answers

When the weakness evidence points at a protected benchmark, the pipeline
builds structurally related but non-identical tasks, verifies them
(executable tests / symbolic checks / multi-judge agreement), and trains on
those — never on the protected items themselves. See
`docs/BENCHMARK_CONTAMINATION_POLICY.md`.

## Synthetic pipeline

`synthetic.py` enforces the full chain: seed problem → generator → solution →
independent critic → objective verifier where possible → dedup →
contamination check → difficulty estimation → quality gate → accept/reject.
The generating model never certifies its own output.

## Handoff to recipe search

The plan feeds `recipe_planner.py`, which proposes bounded candidate recipes
(dataset mixture, LR/schedule, LoRA rank/alpha/target modules, sequence
length, batch/accumulation, steps, objective, replay rate) inside the
measured hardware envelope. Candidates then compete through Chowder's
successive-halving search controller — cheap screens first, full budget only
for survivors.
