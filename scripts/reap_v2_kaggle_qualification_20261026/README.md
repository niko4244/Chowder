# REAP128_v2 Kaggle source snapshots (provisional research)

The sixteen `.py` files in this directory are exact-byte snapshots of the
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

The new original Q8 workflows (`q8_native_original_smoke.py`,
`q8_complete_1319_164.py`) use exactly the original 40-layer Q8 SHA and
construction sidecar plus the same serving topology. The native Q8 smoke
**passed** but retained only **10.28 GiB free cgroup** post-inference.
Therefore the full Q8 notebook was started under a separately registered,
stricter **10 GiB cgroup stop**, preserving the original 8 GiB minimum. An
aborted full Q8 session must not be merged into prior complete Q4 cohorts.
`q8_random128_private_hash_only.py` verified the complete original private Q8
and independent random128 source weights; it deliberately did not run quality
inference and cannot count toward any gate.

The first complete Q8 source-bound attempt **resource-aborted**, with no full
suite summary, at **9.895 GiB unused cgroup** under the stricter 10 GiB stop.
`q8_postload_cache_resource_probe.py` is a *materially different*, separately
registered original Q8 experiment measuring whether a **read-only clean GGUF
cache-eviction hint after CUDA model load** genuinely recovers at least 4 GiB
and establishes at least 14 GiB free headroom. It refuses further inference
if either condition fails; no previous partial rows may be reused. The
`random128_original_smoke.py` has the independently registered frozen original
random128 SHA and no invented Q8 candidate-construction ancestry. None of
these scripts alters original model bits or frozen service flags.

Independent read-only measurements subsequently showed the original Q8 full
trial safety-aborted at **9.895 GiB** free container memory with *zero*
certifiable complete quality suites. A **separately registered original Q8**
post-load cache-only advisory recovered **17.930 GiB**, raising original
postload cgroup headroom from **10.471 to 28.401 GiB**. Its source-bound native
receipt SHA-256 is
`4fb21d64e01855bfed723294e8b5cb7a111892ffc9fef2f224b1df24cd84727b`.
The read-only POSIX_FADV_DONTNEED advisory operates *only after* the same
SHA-pinned model and original GPU topology are fully loaded; it does not change
the original GGUF, prompts, scoring, CUDA flags or context sizes. The two new
`q8_complete_postload_reclaim.py` and
`random128_complete_postload_reclaim.py` scripts each require a fresh original
serve session and at least 4 GiB independently measured recovered headroom,
with at least 14 GiB cgroup unused, before starting any full benchmark work.
Original Q8 full and independent random128 full **new trials** were submitted
only after their original native one-prompt loads and the postload cache
recovery were verified. These trials still cannot count as complete or official
until every session/row/score/result is independently checked. Do not reuse
any aborted partial cohort.

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
