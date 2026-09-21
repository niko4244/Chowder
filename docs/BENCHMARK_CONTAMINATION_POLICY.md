# Benchmark Contamination Policy

This is non-negotiable. A benchmark score is only evidence if the model never
trained on the thing being measured.

## The three sets

`growth/contamination.py` maintains:

- **PROTECTED EVALUATION SET** — fingerprinted by the firewall only. Training
  pipelines never read it; the firewall sees the material, everything else
  sees digests.
- **DEVELOPMENT EVALUATION SET** — may be inspected for diagnosis, but its
  examples are never copied into training.
- **TRAINING POOL** — registered data sources only, each checked against the
  protected material *before* it can be admitted.

## Detection stack

A candidate training text is checked with, in order:

1. exact SHA-256 digest match;
2. normalized-text digest match (whitespace/case-perturbed copies);
3. canary strings — benchmark publishers embed markers; if a canary appears
   in training text, that is contamination, full stop;
4. substring windows (200 chars) against protected examples;
5. MinHash/LSH over 8-gram shingles with Jaccard estimation (threshold 0.30
   → `POSSIBLE`);
6. declared ancestry — a corpus known to contain a benchmark's source crawl
   is `KNOWN_CONTAMINATION` by declaration, not hope.

## Verdicts and what they do

| Verdict | Meaning | Consequence |
| --- | --- | --- |
| `CLEAN` | No detector fired | May enter the training pool (with the rest of the policy) |
| `POSSIBLE` | Fuzzy overlap | Blocked from admission pending human review |
| `KNOWN_CONTAMINATION` | Declared or proven | Score is `BENCHMARK SCORE TAINTED`; promotion verdict is `TAINTED` |
| `UNKNOWN` | Not checked | The source is not trainable; promotion evidence is `inconclusive` |

Every model generation carries a `contamination_manifest.json` — per
benchmark: CLEAN / POSSIBLE / KNOWN_CONTAMINATION / UNKNOWN. Contaminated
benchmarks are displayed raw and marked, never silently dropped, never
included in skill estimates.

## The critical negative path

The mandatory test: offer a protected benchmark test split to the training
side. The system must **refuse**:

- discovery refuses benchmark-shaped candidates at registration;
- `contamination_relationship="UNKNOWN"` blocks trainability even when every
  other gate passes;
- `admit()` cannot include a source whose license forbids training;
- quarantine downgrade is always available and always blocks training.

`tests/test_growth_contamination_and_data.py` pins all of these.

## Analogues, not answers

If the model fails a protected sample, the fix is never "add question +
answer to training." The curriculum pipeline identifies the underlying skill,
sources independent examples of that skill, generates structurally related
but non-identical tasks, verifies them, trains on those, and retests on the
untouched protected sample. That is how Chowder distinguishes *learned the
skill* from *memorized the answer*.
