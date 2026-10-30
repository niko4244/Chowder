# REAP128_v2 — Windows Q4 original first-200 certification crash and source fix

**Decision unchanged:** `BLOCKED`, `chosen:null`, `release_approved:false`.
This report concerns a **failed final certificate step**, not a successful
speed gate, a fresh dense comparator, or a replacement serving session.

## Exact independently inspected original run

The private, source-bound Windows run under
`F:\reap-v2-q4-windows-registered-first200-phase14c-20261026`
executed the frozen `--fit on -np 4 -c 16384 --jinja` Q4 serving flags,
with original Q4 model SHA
`2d64b786461b99a335fdbbdf7c4d3a6758b21409bce755751714ab7cd8f63bf7`,
recorded session `74bcbdde2e85457db06d9f07d2077abf`, original native
server PID `60320`, port `8790`, and measured experts offloaded `0`.
Its saved original GSM8K summary reports 200 rows, 95% recorded accuracy,
97% natural termination and 365.745 mean generated tokens. The original
200-row JSONL SHA is
`b2b9c00e038ded9980e87c427d77a7e44ab7c99e2d68cfda0e8183189fc555fc`.
These are **recorded row/summary observations**, not a new independent
scoring certification or a speed PASS.

The owned wrapper failed **after the evaluator printed 200/200** and
**before** writing the mandatory `speed_session` source proof. The original
native `serve.json` explicitly has `experts_offloaded:null` and
`speed_session:null`; the watchdog terminal has
`status:INCOMPLETE_OR_ABORTED_NO_SPEED_CERTIFICATION`,
`child_exit_code:1`, 217 resource samples and `source_bound_speed_session:null`.
The failure is exact and reproducible:

```text
File "...\chowder_batch\reap_serve.py", line 229, in _certify_q4_speed_session
    from chowder_batch import reap_eval
ModuleNotFoundError: No module named 'chowder_batch'
```

Private original receipts (preserve their exact bytes):

- `watchdog-terminal.json` SHA-256 `2bbabec8d2e01e258322bfe45a6e3da0eed0b6aeb9d8dc0ddca9b8d56074a425`.
- `reap128_v2_q4/server.log` SHA-256 `350d3ecfaaa3216a8419ce11d96073c92fd75e6a1eb41f29abe2ca5f46911781`.
- `reap128_v2_q4/serve.json` SHA-256 `998ac00fe56891917c291c6b514b657c00782db59360badc71186fcdc41dd664`.
- `reap128_v2_q4/summary.json` SHA-256 `2d48474a5026213cab7fc20c9160a0390e410cc91c0ccc44a9103e56a7921d14`.

## Root cause → correction → independent RED/GREEN

The wrapper runs as `python -B chowder_batch/reap_serve.py`. Python puts
the **script's own `chowder_batch` directory**, not its parent, on its
import path. The evaluator child had already received the correct source-root
`PYTHONPATH`, but the *parent wrapper* attempted to import
`chowder_batch.reap_eval` while sealing the same-session speed proof.
Thus the original model and 200-item inference could finish before the
wrapper crashed. The crash cannot be repaired retroactively by editing an
already immutable serving receipt or assuming that 200 recorded rows certify
the terminated native process.

`chowder_batch/reap_serve.py` now derives only the trusted source root from
its own `__file__`, inserts its `src/` and repository root into the wrapper's
local Python import search path immediately before the same-session
certification import, and leaves the evaluator child, model loading, frozen
serving flags, decoding, scoring, dataset, resource admission and threshold
untouched.

New adversarial regression
`tests/test_reap_q4_speed_wrapper_script_import_20261026.py` launches the
**actual wrapper source by absolute file path** in a fresh Python `-I -B`
process without `PYTHONPATH` or a source checkout on its import path. It
calls the original speed-certification entry with missing rows and requires
the legitimate **fail-closed missing native evidence** `ValueError`, not
`ModuleNotFoundError` or an invented PASS.

Before the source fix: **RED** with the exact production
`ModuleNotFoundError`. After the fix: **GREEN**, and the adjacent original
`test_reap_serve.py` / `test_reap_eval_resume_identity.py` suites also passed:
**104 passed** total. The expanded frozen serving, scoring, selector,
native-source audit, paired comparison, admission, and HumanEval preflight
matrix passed **556 tests, 1 Windows-permission skip**, with two existing
configuration/deprecation warnings. Ruff and `git diff --check` passed.

No new CUDA model run was launched by this correction. The old failed
session remains **ineligible**; any future original Windows Q4 first-200
speed certificate must be earned in a **new**, safely admitted and fully
source-bound native serving session, without reusing or altering the failed
phase14c records. No dense first-200 result exists, and the original
candidate/dense ≥1.3× speed gate remains **UNKNOWN**.

The independent random128 full private Kaggle control remained **RUNNING**
on the latest status check; no replacement was launched.
