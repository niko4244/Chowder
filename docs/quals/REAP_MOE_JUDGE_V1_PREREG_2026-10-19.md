# Pre-registration — `reap-moe-judge-v1` (2026-10-19)

**What this registers.** The promotion decision procedure for the REAP-MoE
lineage: given a finished campaign run, does the pruned candidate deserve to
replace its parent? The rows below, their evidence sources, and the verdict
rule are frozen here. This file is named by `VERSION`/`PREREG` in
`docs/reap_moe/judge_reap_moe.py`; once a candidate names this judge version,
neither the rows nor the rule may be edited to fit the candidate's number.

**Invocation.** `python docs/reap_moe/judge_reap_moe.py <run_root>`. The judge
reads a finished run directory. It loads no model, touches no GPU, and makes
no network call. Reads only; it never writes the run.

**Frozen campaign declaration.** `docs/reap_moe/reap_moe_campaign.json` sits
beside the judge and is pinned by `sha256` in the run record. The judge does
not ship it: an absent or unparseable declaration makes every dependent row
UNKNOWN (hence INCONCLUSIVE), never a default. Where this file speaks of
"declared", it means *that* file — never a constant in this document.

**Evidence sources.** `candidate_evaluation.json` and `parent_evaluation.json`
(arm reports), `chosen_candidate.json` (adapter/artifact identity),
`cycle_compute_accounting.json` (cost ledger), `campaign-run.json` (the run's
own decision), and, for every re-scored benchmark, the raw JSONL prediction
rows the report pins by `sha256`. A missing or unreadable file is UNKNOWN; a
digest or count that disagrees with the pin is FAIL.

## Gating rows

| Row | Rule | Evidence |
|---|---|---|
| R1 | The chosen candidate's `artifact_ref` exists, hashes to its pinned digest, and its adapter binding matches the campaign's declared candidate | `chosen_candidate.json`, artifact bytes |
| R2 | Recomputed campaign cost settles at or under every declared ceiling | `cycle_compute_accounting.json`, declaration |
| R3 | The run's recorded ledger digest equals the recomputed digest of the accounting it pinned | `cycle_compute_accounting.json` |
| R4 | Production `evaluate_retention` accepts the declared profile on the measured arms (candidate `MEASURED_THIS_GENERATION`, parent `MEASURED_PARENT`) | arm reports, declaration |
| R5 | The run's recorded decision is the decision this judge reaches; a refusal is not overturned by a better number | `campaign-run.json`, recomputed R4 |
| R6 | Each arm's reported target score reproduces from its raw items under the declared readout; a non-`tool_call_exact` readout is FAIL | raw JSONL, declaration |
| R7 | The paired target delta (candidate − parent, production `compare`) clears the campaign's declared `min_delta`; the bar is never hard-coded | R6 items, declaration |
| R8 | For each declared protected benchmark, candidate and parent re-score from raw items with that benchmark's declared scorer, and the candidate is not worse beyond its declared allowance | raw JSONL, declaration |
| R9 | Gate 0: candidate EOS termination rate ≥ parent rate − 0.02 | target items, declared `max_new_tokens` |
| R10 | Gate 0: candidate degenerate-item rate ≤ parent rate + 0.02 (loop detector, trigram ratio < 0.9, or cap hit) | target items, declared `max_new_tokens` |

**Verdict rule.** Any FAIL ⇒ `REJECTED`. Else any UNKNOWN ⇒ `INCONCLUSIVE`.
Else `PROMOTED`. No gating rows at all ⇒ `INCONCLUSIVE`: the judge never
certifies nothing.

**Exit codes.** `0` PROMOTED · `1` REJECTED · `2` INCONCLUSIVE. A usage error
(wrong argument count, run root absent) also exits `2`.

**Descriptive evidence.** `reports/*.json` may be printed for the reader. No
row reads them, and no such file can move a status or the verdict.
