# Original guard.v3 AutomationBench: independent WorldState replay (2026-10-29)

**Research-only finding:** 88/600 historical original Q4_K_M tasks are scored,
and **3/88 are strict passes**. Independently re-executing the official
AutomationBench v1.0.6 WorldState rubric on the **source-derived task
assertions, initial states, and archived final world states** reproduces every
one of the 88 strict decisions and all 987 individual assertion verdicts.
This is a materially stronger correctness check than trusting the archived
`score`, `passed`, or `assertion_results` fields. It is not a new model run.

The independent replay is implemented in
`chowder_batch/reap_original600_guardv3_worldstate_replay_20261029.py`.
It runs under the original official Python 3.13 virtual environment, verifies
all 487 tracked official `automationbench/` files against the frozen Git tree,
loads the immutable historical guard.v3 scorer/runner sources, and scores a
Windows handle-locked, byte-identical private snapshot of the original F:
evidence. It does not execute the model, benchmark agent, tool calls, or any
model-generated program. The original F: evidence remains unchanged.

| Authenticated source or evidence | Identity |
| --- | --- |
| Historical original guard.v3 source | Git `84ff9703d6e2aba68eda8585ee461e3d84abec0c` |
| Original guard.v3 owner SHA-256 | `b6aa1f37e81943344217ede4429066c468bce2176a74eb8b15fcc0fe4a4e0529` |
| Official AutomationBench v1.0.6 source | Git `4a8e1061254004d9dac807054eed33fad7d1ff14` |
| Full official tracked source tree SHA-256 witness | `a1aa4d00b1416abf565afd0e25a76f83c4ad8a4d0c95fb3ecc243d898a4a254b` |
| Canonical per-item WorldState replay SHA-256 | `2fb9bb71e5403237bfae0f61072ad0078b12b61720c7868102b5a826de071ef9` |

The only archived assertion-parameter serialization discrepancy is task 47,
`salesforce_field_not_changed`: the canonical source contains an optional
`value: null`, while the official archived assertion record omits that key.
The actual assertion verdict, excluded status, score, and strict pass all
agree. The verifier permits **only this exact task/type/null omission**;
other parameter changes fail closed.

To reproduce from the approved Windows worktree without starting inference:

```powershell
& 'F:\reap-v2-public-automationbench-source-20261026\.venv\Scripts\python.exe' -B -m chowder_batch.reap_original600_guardv3_worldstate_replay_20261029
python -m pytest -q tests/test_reap_original600_guardv3_worldstate_replay_20261029.py
```

**Scientific limits:** The original task evidence is a segmented research
checkpoint, not the frozen four-slot Q8-vs-dense comparison. The source-bound
rubric replay does not authenticate historical HTTP request/response bytes,
prompt text at the native serving boundary, model weights during execution,
or signed independent execution witnesses. It does not certify the remaining
512 unscored tasks or a 600-item official complete result. Original frozen
Gate 1/2, dense same-session comparisons, BF16 chained numerical validation,
REAP160 speed/quality, remaining frontier suites, and prospective Amendment 4
statistical method adjudication remain outstanding. The historical
REAP128-v1 Gate 1 failure and unknown REAP160 native speed are unchanged.

**Gen-0 remains** `{"status":"BLOCKED","chosen":null,"release_approved":false}`.
No original model, source, certificate, or evaluation archive was mutated.
