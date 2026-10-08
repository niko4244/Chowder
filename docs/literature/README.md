# Literature watch

Chowder's job is to keep training and compressing models better than yesterday.
The research frontier moves faster than the code does, so this directory is the
single place where "what is new out there, and does it touch us" gets recorded
-- read first, judged second, and never laundered into a claim the repo cannot
back.

## The split

| Artifact | Who writes it | What it is |
| --- | --- | --- |
| `watch.py` | nobody by hand | a stdlib-only fetcher that emits an **unvetted candidate pool** from the arXiv API, grouped by Chowder surface |
| `WATCH_LOG.md` | a person/agent who read the paper | the **curated** record: what a paper establishes, mapped to a concrete Chowder surface, under the rules below |
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
```

The tool only reads the public arXiv API. It needs no key, writes nothing by
default, and is never imported by the package or the tests, so a build or CI
run with no network is unaffected. arXiv asks for about one request every three
seconds; the fetcher spaces its calls accordingly. Re-run it as often as you
like -- a rerun costs nothing and a stale watch is worse than no watch.

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
   gate without a first-party run. Quote it as theirs or not at all.
6. **Keep the negatives.** A paper that says "do not do this at our scale" is as
   valuable as one that says "do this" -- record it, and let it stand in for a
   family's rejection basis the way `inference.confidence-routing` does.
7. **The watch registers nothing.** Turning a finding into a proposal still
   requires a mechanism, a smoke row that actually runs
   (`evidence/family_smoke_matrix.json`), and the runnability gate. The watch
   shortens the *discovery* path, never the *evidence* path.

## Cadence

The watch is a standing habit, not a milestone: run the pool weekly, read what
touches a live surface, and append a dated section to `WATCH_LOG.md` when
something does. A week with nothing relevant is recorded as such -- an empty
week is evidence the habit is running, not a reason to invent a hit.
