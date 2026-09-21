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

---

## Addendum (2026-09-17) — the database is no longer empty, and still decides nothing

The rationale above stands: no published number found for `math500@2024-04`
or `mgsm@2022-11` matches Chowder's frozen protocol, so **nothing imported
earns HIGH** and the gate path stays exactly as empty as it was. What changed
is that the *citable* numbers are now recorded instead of living in a search
log.

### What was imported

`docs/gen0/seed_frontier_references.py` (committed, reproducible) writes 15
references and freezes a **new, later-dated** snapshot
`gen0-frontier-context-2026-09-17`. `gen0-frontier` itself is untouched — the
generation-time snapshot stays empty, because enrichment after the fact must
not be rewritten into the record of what the frontier looked like at freeze
time.

| Source | Benchmark rows | Why not HIGH |
| --- | --- | --- |
| DeepSeek-R1 official repo (MATH-500 pass@1 table) | 7 rows, LEVEL_1/2/3/4 peers through absolute frontier (0.891–0.973) | MATH-500 subset + 64-sample pass@1 at temp 0.6 vs Chowder's lm-eval `minerva_math`, 28 items, chat template, greedy |
| Qwen2.5 official blog ("Multi-Mathematics", MGSM 8-shot CoT) | 7 rows, LEVEL_1/2/3 (0.363–0.767) | multilingual 8-shot CoT mean, base checkpoints, vs Chowder's 24-item English `mgsm_direct_en` (no CoT) |
| `lm-evaluation-harness` issue #2646 | 1 row: Llama-3.1-8B-Instruct `minerva_math` 4-shot exact_match **0.346** (MEDIUM) | same task/metric family, but 4-shot on the full split where Chowder ran 0-shot chat-templated 28 items |

### The number worth staring at

The harness-family row is the useful one. Under the *same* `minerva_math` task
and metric, Llama-3.1-8B-Instruct scores **0.346**; its publisher reports MATH
**0.519**, and this 7–9B class posts **0.89–0.93** on MATH-500. Same model,
three plausible-looking protocols, three wildly different numbers. Protocol
dominates this benchmark family at small subsets.

Consequence for reading Generation 0: the frozen `math500@2024-04` row is a
**28-item** `minerva_math` limit (7 subtasks × 4) with `exact_match 0.0` and
`math_verify 0.0`, and `mgsm@2022-11` is 24 items. Those zeros are protocol
artifacts at this budget, not a measured capability floor, and they must never
be used as "the parent cannot regress below 0.0" protection.

### New rendering channel: context, not gaps

`context_rows` / `FrontierDatabase.context_for_benchmark` surface these rows in
the mechanical report's *reference context* table, naming the blocking
protocol dimension per row (`protocol_divergence`). They are structurally
incapable of deciding anything: `best_for_benchmark` still admits only HIGH,
and `gap_rows` remains the only decision path. `render_gap_report.py` reads
gate rows from the generation-time snapshot and context rows from every frozen
snapshot, so both stay honest as the store grows.

Still blank, deliberately: **MGSM has no LEVEL_4 row** (no first-party
absolute-frontier MGSM number was found) and **LEVEL_0_FLOOR has no reference
at all** (that level is Chowder's own measured generation).

### Unresolved evidence finding (recorded, not rewritten)

The frozen Gen-0 `mgsm@2022-11` row is internally inconsistent: it declares
`n_samples: 24` but carries **48** per-sample scores, two of which are `1.0`
(≈0.042), while the recorded aggregate `score` is `0.0`. The raw artifact
(`battery_results_attempt2.json`, `measured[1]`) is the reference of record.
This does not affect any verdict — Gen-1's protected evidence was carried
parent evidence, which the integrity pass already refuses — but the frozen
row cannot be used as an MGSM regression floor until the aggregate and the
vector are reconciled. Gen-2 measures its own slices instead.

Expansion coverage: `tests/test_frontier_reference_expansion.py` (provenance,
no-HIGH pin, context inertness, additive-snapshot immutability, deterministic
seed, snapshot re-write refusal).
