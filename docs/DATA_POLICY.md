# Data Policy

Chowder never ingests anonymous untracked blobs into a production curriculum.
Every source carries accountable provenance: exact revision, license,
permitted-use declaration, verification class, contamination relationship,
quality, PII/secret review, and an explicit inclusion decision.

## Trust classes and verification floors

| Class | Meaning | Acceptable verification |
| --- | --- | --- |
| GOLD | Objectively verified (executable tests, symbolic/numeric checks, authoritative keys) | `executable_tests`, `symbolic_numeric`, `authoritative_key` |
| SILVER | Strongly verified | `multi_judge`, `citation_supported`, `curated_trusted` |
| BRONZE | Useful, weakly verified | `heuristic_filter`, `curated_trusted` |
| QUARANTINE | Unverified; never automatically trained on | `unverified` |

The floors are enforced at construction: a source claiming GOLD with only a
heuristic filter is refused, not warned about.

A source is trainable only when **all** of the following hold:

```python
inclusion_decision == "included"
and permitted_training_use          # license allows it
and trust_class in {GOLD, SILVER, BRONZE}
and contamination_relationship in {CLEAN, POSSIBLE_CLEARED}
```

## Licensing is a registration requirement

`license="unknown"` is refused outright. Discovery-level screening also
rejects non-commercial and research-only declarations for training use, and
`admit(decision="included")` re-checks the permitted-use flag.

## The discovery workflow

```
DISCOVER → INSPECT → LICENSE → QUALITY → CONTAMINATION → REGISTER → (optionally sample)
```

`discovery.py` implements this mechanically:

- every candidate enters as metadata only (id, origin, URL, pinned revision,
  license, domain, language, type, size estimates);
- inspection flags benchmark-shaped candidates (identifiers suggesting
  test/eval/protected material), missing pins, and unknown licenses;
- registration requires completed PII and secret review and records the
  source as `QUARANTINE` with `inclusion_decision="pending"`;
- promotion out of quarantine happens only through an explicit
  `admit()` decision backed by contamination and review evidence.

The one mutating CLI path (`chowder data register`) cannot bypass any of
this; `--revision latest` exits with an error.

## Reservoir sources are sampled, not downloaded

Large corpora (FineWeb-Edu, DCLM, The Stack v2) are registered as reservoirs.
They are streamed and sampled into bounded, fingerprinted slices; nobody
downloads multi-terabyte corpora onto the local machine by default. Licensing
and provenance are re-verified before each campaign, not assumed from a
previous one.

## Pre-registered seed catalog

`seed_registry()` ships reputable open sources across reasoning, math, coding,
general knowledge, and research domains with pinned revisions and honest
initial contamination state (`UNKNOWN` until the firewall checks them). All
start non-trainable.

## Internet data

"It's on the internet" is not a reason to train on it. Source policy checks
licensing, robots/terms where applicable, provenance, privacy, secrets, PII,
quality, duplication, and benchmark contamination. Good internet-derived data
is enormously valuable — with accountable provenance.

## Auditability

`chowder data audit` reports every source's trust class, verification,
license, contamination relationship, inclusion decision, and trainability,
so the answer to "what is this model trained on?" is always one command away.
