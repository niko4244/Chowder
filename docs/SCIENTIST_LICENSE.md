# Scientist mode: license integration strategy

Status: assessed before any AI Scientist v2 code was read into the adapter
(clean-room protocol), and before any vendor code was copied. **Zero AI
Scientist v2 source is included in Chowder.**

## The two licenses

| | Chowder | AI Scientist v2 |
|---|---|---|
| License | MIT (`LICENSE`) | The AI Scientist Source Code License v1.0 (Dec 2025), a derivative of the Responsible AI Source Code License v1.1 |
| Grant | Permissive; derivative works under MIT | §2 copyright license to reproduce/derive/distribute, **subject to §3 restrictions** |
| Flow-down | None | §3.1: distributions must include a complete copy of the license; §3.3: §3.2 restrictions must be passed on in agreements covering derivative works |
| Use restrictions | None | §3.2: surveillance; computer-generated media disclosure; health care; criminal prediction; the "AI Scientist" clause (manuscripts must prominently disclose machine generation) |

## Compatibility analysis

1. **Vendoring upstream source into Chowder is prohibited by this
   assessment.** AI Scientist v2 is source-available, not OSI-licensed.
   Copying its source into an MIT-licensed repository would (a) create a
   derivative work whose restrictions MIT cannot carry, (b) trigger §3.3's
   flow-down obligation on every downstream Chowder recipient, and (c) risk
   misrepresenting the provenance of that code. The Chowder `LICENSE` would
   no longer accurately describe the whole work.
2. **Clean-room adapter instead.** The adapter
   (`src/chowder/scientist/providers/ai_scientist_v2.py`) was written against
   documented behavior and file formats only: the upstream ideas-JSON schema
   (`Name` / `Title` / `Short Hypothesis` / `Related Work` / `Abstract` /
   optional `Description` / `Experiments` / `Risks & Limitations`), the
   upstream `bfts_config.yaml` keys (`desc_file`, `workspace_dir`,
   `data_dir`, `log_dir`, `exec.*`, `agent.*`, `report.*`), and the upstream
   `Journal.to_dict()` export shape (`journal.json`). No upstream source was
   transcribed, translated line-by-line, or mechanically converted.
3. **The sidecar runs upstream code unmodified, in its own runtime, under its
   own license.** The operator obtains it directly from upstream
   (`git clone https://github.com/SakanaAI/AI-Scientist-v2`). Chowder invokes
   it through its documented entry point as a subprocess inside an isolated
   runtime (docker / WSL2 / an explicitly configured local Python
   environment). Chowder never imports it, and the sidecar never imports
   Chowder.
4. **No manuscript pipeline.** Scientist mode uses only the ideation and
   tree-search entry points. `perform_writeup.py`, `perform_llm_review.py`
   and `perform_vlm_review.py` are never invoked. The license's §3.2(e)
   disclosure clause therefore has no occasion to apply; were an operator to
   enable writeup outside Chowder, that operator carries the disclosure
   obligation under the upstream license.

## Operator obligations

An operator who enables `provider: ai_scientist_v2`:

1. obtains AI Scientist v2 themselves and accepts its license with Sakana AI
   (Chowder does not and cannot grant it);
2. runs it in the isolated runtime scientist mode configures;
3. does not redistribute the sidecar as part of a Chowder distribution —
   Chowder's docs state the sidecar is external software under The AI
   Scientist Source Code License;
4. keeps the use restrictions of the upstream license (§3.2) when using the
   sidecar's outputs.

Chowder's own distribution remains purely MIT with no upstream bytes in it:
the `git history` of this branch contains no AI Scientist v2 source, and the
adapter's fixture files in `tests/fixtures/` are **hand-written minimal
documents in the upstream format**, not upstream data.

## Future providers

`FakeDeterministicScientistProvider` (tests) and `AIScientistV2Provider` both
implement the same `ScientistProvider` protocol. A future MIT-compatible
provider (e.g. `LocalLLMScientistProvider`) can be added with no license
exposure; the protocol is the only thing providers share with Chowder.
