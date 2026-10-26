# REAP128_v2 Kaggle source snapshots (provisional research)

The nine `.py` files in this directory are exact-byte snapshots of the
private Kaggle run sources used on October 26, 2026. The independent SHA-256
manifest binds their local bytes. The original **GGUF model weights, private
dataset contents, Kaggle credentials, raw completions, and API tokens are not
in this repository**.

These programs are **not** sanctioned replacements for the preregistered
Windows-backed Amendment 4 evaluation. They were created to investigate
isolated Linux CUDA model loading, source integrity, scoring, memory pressure,
and the exact first-200 cohorts. Kaggle's `GGML_CUDA_NO_VMM=ON` change is an
explicit Linux compatibility workaround, not a claim of identical numerical
behavior. The first canonical GSM8K answer in fact differs between original
Windows and the pinned Linux runner. See
`docs/quals/reap-v2-dense-fail-safe-and-single-t4-parity-20261026.md`.

The scripts expect the original, separately stored private Kaggle datasets and
the completed pinned Linux CUDA toolchain notebook. **Do not republish the
model datasets or trigger another evaluation automatically.** Before any
manual rerun, check current remote Kaggle sessions, GPU quota, unique output
identity, source hashes, original resource floors, and private permissions;
never append previous partial rows to a different serving session. The model
source itself must be checked against its original SHA after mounting. The
original `--fit on -np 4 -c 16384 --jinja` server contract and canonical
1,319 GSM8K / 164 HumanEval inputs must remain unchanged.

`candidate_full_1319_164.py` is a **complete-population attempt** but cannot
be called an official frozen Windows gate on a Linux runner. `dense_q4_first200.py`
already generated an original-host-cgroup fail-closed abort; repeat attempts
with the same inadequate host capacity are inappropriate. A standalone
`candidate_q4_first200.py` run is scientifically useful only with a fully
certified original dense comparator from a compatible serving setup.

`human164_lowpriv_independent.py` independently re-executes all 164 recorded
HumanEval programs in an isolated Kaggle CPU notebook without attaching a
model, under limited CPU/memory/filesystem resources and reduced subprocess
privileges. It matched all 164 original recorded correctness decisions (151
passed), but this **cross-checks Linux scoring only**; it does not establish
Windows/Linux numerical parity or qualify the frozen reference-model gates.
