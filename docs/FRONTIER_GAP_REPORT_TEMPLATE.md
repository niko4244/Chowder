# Frontier Gap Report

Every serious Chowder generation gets a `FRONTIER_GAP_REPORT.md` generated
from `growth/frontier_reference.py` — mechanically, from the frozen snapshot
and the generation's eval report, never hand-written.

## The five reference levels

```text
FLOOR (Generation 0)
  ↓
PARENT
  ↓
PEER TARGET          — best comparable open model (~5–12B dense or similar
                       active-parameter MoE, deployable on similar hardware)
  ↓
OPEN FRONTIER        — best open-weight model regardless of size (stretch)
  ↓
ABSOLUTE FRONTIER    — best trustworthy score from any current model,
                       under a comparable protocol
```

Chowder should always know where it sits on that ladder. The minimum serious
target for a mature generation: beat Generation 0, hold protected skills,
reach or beat the comparable peer on the majority of categories, stay
competitive on the rest, and demonstrate at least one genuine capability
advantage.

## Comparability gate

A comparison is emitted only when all of the following align:

- benchmark and exact version (`benchmark@version`, never "latest");
- scorer and dataset split;
- tool access (raw model vs agent harness — different measurements, never merged);
- reasoning effort/setting;
- sampling regime (pass@k, n samples);
- environment.

Anything else renders:

```text
NOT DIRECTLY COMPARABLE
```

We would rather have no comparison than a fake one. Parity ratios
(`chowder / reference`) appear only where the zero point and score scale make
a ratio meaningful; otherwise standardized gaps are used.

## Snapshots never rewrite

`SnapshotStore.freeze()` refuses an existing snapshot id. The frontier as it
existed when v1.0 shipped is preserved separately from the frontier when
v1.5 ships, which is what separates "Chowder improved" from "the frontier
moved faster". Each generation records both:

- `absolute_improvement` — this generation vs Generation 0;
- `frontier_gap_change` — the distance-to-frontier trend.

## Report template

```markdown
# Chowder-9B vX.Y Frontier Gap

## Overall
Generation 0:        <score>
Comparable peer:     <score>
Best open-weight:    <score>
Absolute frontier:   <score>
Current Chowder:     <score>

## Reasoning
Chowder:  <score>   Parent: <score>   Peer: <score>   Open: <score>   Frontier: <score>
Gap: <delta>   Trend: <gap change vs last generation>

## Math
...

## Coding
...

## Agentic capability
...

## Self-improvement
...

### Areas already competitive
### Areas approaching frontier
### Largest frontier gaps
### Largest same-size peer gaps
### Capabilities improving fastest
### Capabilities stalled across generations
```

Arrows language everywhere: ↑ improved · → statistically flat · ↓ regressed ·
? unavailable · N/A unsupported · ⚠ contaminated/non-comparable.

## Data sources for reference scores

Preference order: (1) the benchmark's official leaderboard, (2) the model
developer's official report, (3) an independent evaluator with a documented
protocol, (4) a reputable public leaderboard. Every imported score records
model, version, benchmark@version, score, metric, reasoning effort,
agent/tool configuration, sample count, date, source, first-party flag, and
comparability confidence. Only `HIGH`-confidence, protocol-aligned entries
compare; everything else is displayed but gated.
