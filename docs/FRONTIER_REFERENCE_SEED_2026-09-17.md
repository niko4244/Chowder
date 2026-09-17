# Frontier Reference Seed — procedure and current state (2026-09-17)

The frontier system's code discipline is pinned by tests: unknown levels are
refused, snapshots are immutable, `compare_protocol` renders
`NOT DIRECTLY COMPARABLE` for anything but an exact protocol match at HIGH
confidence, and (new in this closeout) the database refuses duplicates,
unpinned benchmark names, out-of-range scores, and references with no
source URL.

What was missing was operational truth: **the database currently holds zero
reference scores, and this closeout deliberately leaves it that way.**

## Why the seed set is empty

A reference score earns `comparability_confidence = HIGH` only when its
protocol genuinely aligns with Chowder's frozen measurement protocol
(`gen0-freeze-protocol-v1`: lm-eval 0.4.12, `minerva_math` limit 4 /
`mgsm_direct_en` limit 24, chat template applied, greedy, task-native
limits). The closeout searched for importable numbers and found:

1. **Qwen** (blog, model card, GitHub README, technical report): per-size
   tables are published only as rendered images; no text source carries
   MATH-500 or MGSM numbers for Qwen2.5-7B-Instruct with a citable
   harness/shot/split description.
2. **Meta** (Llama-3.1-8B-Instruct): the model card is license-gated and
   returns HTTP 401; its numbers also use Meta's internal harness with
   unknown shot counts for MATH.
3. **lm-eval community numbers**: third-party reproductions with
   unrecorded prompt/format variations — exactly the class
   `compare_protocol` exists to refuse.

Importing any of these as HIGH would violate the program's own rule. They
would be MEDIUM/LOW, and MEDIUM/LOW references are inert by construction
(`best_for_benchmark` surfaces only HIGH; the test
`test_medium_confidence_reference_is_recorded_but_never_decides` pins
that). Decorative rows are worse than no rows.

## How a reference gets added later (the bar to clear)

An entry must carry: exact model/version; exact `benchmark@version`;
metric and scale; dataset split; harness (and version); tool setting;
reasoning setting; sampling/pass@k regime; date; source URL; first-party
flag; comparability confidence; and notes on any known mismatch. The
trusted path is **running the reference protocol ourselves** with the
published model's official config — that produces a first-party-protocol
measurement Chowder fully understands — or waiting for a publication that
documents its protocol completely.

## The gap report

`gen0-frontier` was frozen in the Gen-0 evaluation freeze with an empty
reference set — an honest statement, preserved immutably. Until HIGH-confidence
references exist, the mechanical gap report renders Chowder's rows against
`unavailable` references rather than inventing gaps. The generator script
lives at `docs/gen0/render_gap_report.py` and refuses to hand-wave.

## Test coverage added in this closeout

`tests/test_frontier_reference_seed.py`:
- duplicate reference entries refused (per model × benchmark × level)
- `benchmark@latest` / unpinned names refused
- scores outside normalized 0..1 refused
- empty/whitespace `source_url` refused
- valid references persist across reload
- MEDIUM-confidence references are recorded but never decide comparisons
- HIGH-confidence references round-trip and do decide comparisons
