# REAP128_v2 additional fail-closed preparation — October 26, 2026

**Gen-0 status remains BLOCKED, chosen:null, release_approved:false.**
Original frozen model, scorer, thresholds, serving flags, dense abort and Q4
research evidence are unchanged. This note supplements rather than overwrites
the historical immutable scientific receipts.

## Unchanged original running experiments

Private `nikmarco/reap128v2-q8-full-postload-reclaim` and
`nikmarco/reap128v2-random128-full-postload` were rechecked twice and both
were still RUNNING. Neither full Q8 nor independently constructed random128
control quality is certified. No replacement kernel was launched.

## New source-only dense admission, not execution

Independent read-only verification of original dense abort
`b2ef325bb9c3165439e669326746fdb9165a5a66975890c99cbfd56dda8c6c0a`
confirmed cgroup unused **29.444 to 9.992 GiB** before the model completed
loading, with **zero eligible 200-item dense speed rows**. The historical
`dense_q4_first200.py` only checked 24 GiB pre-load; leave its actual
historical source unchanged and never repeat the old insufficient capacity.

A separate **unlaunched** `dense_q4_first200_material_postload.py` now embeds
the independent bound **19.452 GiB observed incomplete-load demand plus
10 GiB conservative runtime stop plus 4 GiB postload allowance = 33.452 GiB**
actual initial unused cgroup minimum. It also checks real RAM, 2-GPU free
VRAM, and disk. A read-only postload `POSIX_FADV_DONTNEED` must actually
recover at least 4 GiB and leave at least 14 GiB before the first original
GSM8K item. A live monitor still stops before cgroup unused falls below
10 GiB. The model SHA, pinned source and original
`--fit on -np 4 -c 16384 --jinja`, first 200 rendered GSM8K prompts,
decoding, scorer and source identity are unchanged. The new source is
separately hash-manifested, not a completed remote result. Tests: **24/24
CPU-only adversarial PASS**, following a source-absence RED phase.

No sufficiently large real runner was verified. Preferred 48 GiB initial
cgroup/64 GiB host availability/48 GiB GPU capacity are provisioning
recommendations, not proven requirements or measured successful fit.
Windows local CUDA admission remains unavailable at the observed dynamic
VRAM capacity. The original candidate/dense 1.3× speed requirement is UNKNOWN.

## HumanEval independence enforced before paired comparison

Independent review identified a consequential downstream gap: the full
external auditor may legitimately emit a complete original-scored cohort
with `human_eval_independent_execution=PENDING_INDEPENDENT_LOW_PRIVILEGE_CODE_EXECUTION`
while a separate restricted 164-program HumanEval replay has not yet run.
Previously the descriptive pairing utility could then pair original
HumanEval-scored booleans and overstate the independence of that result.

`reap_external_paired_descriptive.read_verified_cohort` now **fails closed**
for HumanEval unless **both arms** carry the complete independent replay
marker `PASS_FULL_164_LOW_PRIVILEGE_REEXECUTION_SHA256:<64 lowercase hex>`,
which the original full auditor emits only after validating the independently
replayed result's original 164 outcomes, exact row SHA, serving session and
correct count. Seven adversarial missing, pending and malformed variants
failed before the patch and now reject; a positive two-arm replay fixture
remains descriptive with no gate promotion. This checks the downstream
full-auditor proof format; it is **not authentication against forged input
audit JSON**, nor a substitute for verifying the full separate private
low-privilege execution receipt and sandbox environment.

Original Q4's 164/164 replay, 151 correct and immutable original source
binding remain preserved, not rerun. The new Q8 and random128 replay notebooks
must each independently bind their own new full native session, exact original
row SHA, frozen suite/scorer and actual observed pass count. Existing GPU
runners call the unchanged original HumanEval scorer inside their own
notebook; an after-the-fact independent replay does not retroactively sandbox
that first execution. Do not falsely claim a complete low-privilege sandbox.

These changes do not resolve Windows/Linux numerical divergence, optional
prospective McNemar-vs-Wald governance, fresh full dense_v2 quality,
REAP160 speed, BF16 forward parity or all mandatory original gates. No
Gen-0 model selection or release is authorized.
