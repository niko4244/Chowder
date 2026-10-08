# Literature watch

Chowder's job is to keep training and compressing models better than yesterday.
The research frontier moves faster than the code does, so this directory is the
single place where "what is new out there, and does it touch us" gets recorded
-- read first, judged second, and never laundered into a claim the repo cannot
back.

## The split

| Artifact | Who writes it | What it is |
| --- | --- | --- |
| `watch.py` | nobody by hand | a stdlib-only fetcher that emits an **unvetted candidate pool** from the arXiv API, grouped by Chowder surface; `--append-log` screens it through the surface gates and drops what is new |
| `.github/workflows/literature-watch.yml` | nobody by hand | the schedule: runs the pool weekly and opens a pull request carrying an **unvetted drop** into the log when something new matches |
| `WATCH_LOG.md` | a person/agent who read the paper | the **curated** record: what a paper establishes, mapped to a concrete Chowder surface, under the rules below -- plus explicitly-labelled automated drops, which are triage lists and not entries |
| `src/chowder/growth/interventions.py` | a change with tests | the only place a finding becomes a *family the loop may propose* |

A pool entry is a thing to read. A log entry is a finding. A registered family
is a mechanism that has been made runnable and gated. These are three different
things and the watch never collapses them.

## Running the watch

```bash
python docs/literature/watch.py                       # every profile, 14-day window
python docs/literature/watch.py --days 3 --max 8      # tighter, recent
python docs/literature/watch.py --profile compression -o pool.md --json pool.json
python docs/literature/watch.py --list-profiles

# exactly what the schedule runs: screen, drop what is new, write nothing when
# nothing is (safe to rerun -- ids already in the log are never listed twice),
# and prune automated drops older than the horizon in the same write
python docs/literature/watch.py --append-log docs/literature/WATCH_LOG.md \
    --append-limit 120 --prune-drops 30
```

The tool only reads the public arXiv API. It needs no key, writes nothing by
default, and is never imported by the package or the tests, so a build or CI
run with no network is unaffected. arXiv asks for about one request every three
seconds; the fetcher spaces its calls accordingly. Re-run it as often as you
like -- a rerun costs nothing and a stale watch is worse than no watch.

## The recurring run

`.github/workflows/literature-watch.yml` runs the pool every Monday at 06:00
UTC, and on demand through `workflow_dispatch`. Its shape is deliberately
narrow:

1. **Fetch** all six profiles, then **screen** each hit through that profile's
   two coarse gates: the paper's *primary* category has to be one the surface
   watches (arXiv's `cat:` matches cross-lists too, which is how a speech or a
   robotics paper answers an ML query), and one of the profile's surface terms
   has to appear in the title or abstract -- a case-insensitive substring test.
   No ranking, no reading, no numbers.
2. **Drop only what is new.** An id already mentioned anywhere in the log, by a
   curated entry or an earlier drop, is never listed twice, so a rerun, a wider
   window, or a week the schedule skipped cannot duplicate anything.
3. **Write nothing when nothing is new.** Log churn is not evidence the habit is
   running; an empty week shows up in the run summary and nowhere else.
   Inventing a hit to fill a week is the failure the rules below exist to stop.
4. **Land it for review.** When something is new, append a
   `## Automated pool drop -- <date>` section on the machine-owned branch
   `automation/literature-watch` and open or refresh a pull request. `main` is
   protected, so a drop lands only when a person merges it -- which is the
   intent. The branch is regenerated from `main` on every run; do not commit to
   it by hand.
5. **Prune as it goes.** The same write removes `Automated pool drop` sections
   older than 30 days, so a weekly drop cannot grow the log without bound. The
   horizon is longer than the fetch window on purpose: an id inside a pruned
   drop is already too old to be fetched again, so pruning can never make a
   later run list the same paper twice. Curated sections are never touched, and
   a run with no new hit prunes nothing.

An automated drop is a triage list, not a finding. It carries ids, dates, and
matched terms -- deliberately **no abstract text and no number** -- so it cannot
carry a claim into a curated log. It registers nothing, and it is not an entry.

Every run also publishes the **whole unfiltered pool** (`pool.md`, `pool.json`,
and the run's stdout) as a build artifact. That is the mitigation for the screen
being coarse: a keyword miss is possible, so the pool is never silently thrown
away, and widening a profile's terms or categories is a one-line change in
`watch.py`. The screen decides what is worth *listing*; it never decides what is
*true*.

If a drop looks wrong, delete it, with one caveat stated because it looks like a
bug otherwise: deleting a drop takes its ids out of the log, and dedup has no
memory beyond the log, so a hit from that drop which is still inside the fetch
window can honestly be listed again next week. Pruning by hand faster than the
horizon is the only way to see a paper twice; the schedule never does it.

**Operator note.** The pull request is opened with the workflow's own
`GITHUB_TOKEN`, and GitHub parks the `pull_request` run that token creates at
`action_required` until someone approves it, so the required checks never report
on their own. The schedule therefore approves that run itself, which is why the
workflow asks for `actions: write`. Dispatching `ci.yml` separately does **not**
work, and that was measured rather than assumed: the dispatched checks completed
green on the commit while the pull request's check rollup stayed empty, so the
gate stayed blocked. If a run is ever left waiting anyway, one push or an
*Update branch* click from a person covers it; the drop itself is unaffected.

## The rules an entry must follow

These are the same discipline the research notes already use (see
`docs/TURBOSPARSE_POWERINFER_RESEARCH.md`), stated so the watch cannot drift
into marketing.

1. **Cite the source.** Every entry carries its `arXiv:<id>`, submission date,
   and the exact claim the abstract/paper makes. No id, no entry.
2. **Name the mechanism, not the vibe.** Say *what it changes* (the KV cache,
   the optimizer, the data mixture, the judge), not that it is "promising".
3. **Classify the transfer.** One of:
   - *measurement* -- reusable now with no model change (a metric, a probe);
   - *inference-only* -- changes speed, never quality; never enters a quality gate;
   - *requires retraining* -- a claim that only exists after a training run;
   - *watch* -- relevant only if a premise Chowder does not have (a looped
     architecture, a new modality) ever becomes true;
   - *not-applicable* -- recorded so the next watch does not re-surface it.
4. **Map it to a surface.** An existing family id (`compression.ptq`,
   `training.teacher-distillation`, ...), a named backend, or an explicit
   `candidate -- unregistered`. "Interesting" is not a mapping.
5. **No borrowed numbers.** A paper's measured gain is *their* measurement, on
   *their* model, under *their* protocol. It never transfers into a Chowder
   gate without a first-party run. Quote it as theirs or not at all. An
   automated drop carries no number at all, for exactly this reason.
6. **Keep the negatives.** A paper that says "do not do this at our scale" is as
   valuable as one that says "do this" -- record it, and let it stand in for a
   family's rejection basis the way `inference.confidence-routing` does.
7. **The watch registers nothing.** Turning a finding into a proposal still
   requires a mechanism, a smoke row that actually runs
   (`evidence/family_smoke_matrix.json`), and the runnability gate. The watch
   shortens the *discovery* path, never the *evidence* path. A machine-appended
   drop registers nothing twice over: it is only a list, and the paper behind it
   still has to be read before it can become an entry.

## Cadence

The watch is a standing habit, not a milestone. The schedule above does the
mechanical half every week -- fetch, screen, drop when new, stay silent when not
-- and the half a machine cannot do is the reading, which is the part that stays
standing:

- read the drop (or, in an empty week, the run's `pool.json` artifact) for
  anything that touches a live surface;
- promote what matters into a dated curated section under the rules above;
- leave the drop for the horizon to prune, or delete it early if it adds
  nothing -- an empty drop trail is not worth carrying.

An empty week is silent by design: it is recorded in the run summary, never
invented into a hit and never padded into log churn.
