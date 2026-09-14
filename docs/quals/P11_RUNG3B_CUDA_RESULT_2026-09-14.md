# P11 rung 3b — CUDA arm result: the 9B router pilot under the amendment (2026-09-14)

Judged against the committed preregistration
[`P11_RUNG3B_CUDA_PREREG_2026-09-14.md`](../../../Brainz/workspaces/Chowder-p7-evidence-fixes-20260912/docs/quals/P11_RUNG3B_CUDA_PREREG_2026-09-14.md)
(pushed as `4456e75`, **before** the run). Run: `chowder project-validate` →
`chowder train`, fresh registry, project `router-healing-9b-pilot-rung3b-cuda`,
run directory `router-pilot-9b-ed626dc17192`. Total wall ≈ 29 min (budget 40).

## Verdict

**QUALIFIED — with one honestly recorded exceedance on threshold 7's
measurement basis.** Every structural threshold passes on measured evidence;
the attributable-GPU-hours ceiling is met on the prereg's own workload basis
and exceeded under a total-device-time basis that the prereg did not
anticipate. Both numbers are recorded below; nothing was renegotiated.

## The measured outcome

| Metric | Baseline | Candidate (router pilot) |
|---|---|---|
| Holdout loss (64 disjoint sentences) | **2.9327** | **2.8369** (−0.0958) |
| Dead experts (of 512 expert slots) | 480 | **428** |
| Experts per token | 2.0 | 2.0 |

The gate **PROMOTED** `router-pilot-9b` (`PromotionEvent`, event_id 8). Per the
prereg's interpretation rules this completes the amendment's CUDA arm at pilot
scale; it does not qualify model quality.

## Threshold-by-threshold judgment (durable artifacts only)

1. **Validate before train** — PASS. `project-validate` exit 0
   (including a full-mode identity re-hash); `train` invoked only after.
2. **Policy contract** — PASS. All three worker results (train, base eval,
   candidate eval) carry `load_policy_report.policy = "bf16-offload-transient"`,
   `dtype = torch.bfloat16`, `placement_census.verified = true`
   (`expert_params_on_device: 0/64`, `gate_params_on_device: 32/32` on
   `cuda:0`), 32 patched expert modules before and after freeze.
3. **Measured preflight** — PASS. `device_preflight.step_cost_probe` is a real
   on-device measurement: resident-before-step 9.964 GB, peak 11.973 GB,
   incremental 2.009 GB, 2.804 s/step, `projected_oom=false`,
   `would_exceed_budget=false`. The ×1.5 refusal lines (14.0 GB memory, 3.0
   s/step) were enforced by the projection; nothing was refused because
   nothing exceeded.
4. **Trainability** — PASS. Every router gate reports
   `gradient_states=["grad-nonzero"]`, `nonzero_steps=[0..11]`, real
   `update_steps` on all 12 steps, `trainable=true`; 32 layers with trainable
   gate. Frozen evidence: 523 frozen parameters, `changed={}`, `ok=true`,
   digest strategy `[full, sampled]`.
5. **Horizon** — PASS. 12/12 steps, `stop_reason="max_steps"`, 1536 tokens
   consumed (= 12 steps × 64 seq × 2 batch, exactly the fixed workload).
6. **Gate** — PASS. Base 2.9327 and candidate 2.8369 both measured with phase
   ledgers; identity control recorded (`identity_payload=false`, 32 parameters
   applied, before/after logits digests); routing demonstrably changed
   (per-layer top-1 tables); decision recorded as a promotion with its ledger.
7. **Accounting** — PASS on structure; **exceedance recorded on one basis**.
   - Phase ledgers measured on both sides: train — model_load 12.87 s,
     steady_state 24.99 s, checkpoint_publication, closeout; eval —
     model_load 12.77 s, baseline_generation 3.43 s, candidate_generation
     4.00 s.
   - Peak VRAM measured: **13.50 GiB** (train), **9.48 GiB** (each eval) —
     both within the preregistered 14.0 GB refusal line, on a 17.1 GB card,
     with the experts verifiably CPU-resident.
   - **The exceedance**: total measured device time across the three workers
     is 58.1 s ≈ **0.0161 GPU-h**, over the 0.01 ceiling; the workload-only
     basis the prereg actually derived from (steady-state steps + publication
     + eval generations ≈ 32.4 s ≈ 0.0090 GPU-h) is within it. The difference
     is three model loads at 12.8 s each *on device* — the prereg expected
     load cost to be dominated by CPU hashing. Recorded, not renegotiated;
     any successor prereg must budget loads explicitly.
   - Ledger charges are wall-on-device and conservative: results table
     charges 0.0154 (baseline) + 0.0401 (candidate) = 0.0554 GPU-h against
     the 0.05 goal envelope; reservations were checked at proposal time and
     settled, `execution_incidents` is empty, the stranded-result audit was
     clean at closeout, exit 0.
8. **Identity chain** — PASS. The eval worker re-derived base manifest
   SHA-256 **`77520eda…c4` — exact match to the pinned hash**; payload bound
   to base content `a6b20aaf…`, eval re-derived the identical content digest,
   and `payload_verification.base_content_sha256` matches the manifest. The
   published payload is bf16 (the trained dtype), 2,097,152 elements, 12
   steps, manifest body SHA `89625484…` (file-with-newline `ff94498a…`).

## Defects this rung caught and fixed (each behind failing tests + mutation checks)

Three production defects were exposed by moving from tiny models to the 9B
artifact, exactly the escalation discipline working as intended:

1. **Preflight double-count** (`12dcde5`): the memory projection compared a
   step peak that *includes* the resident model against free memory *after*
   the model is resident — demanding the model fit twice. Attempt 3 was
   falsely refused; the fix compares incremental demand.
2. **Deterministic publication lock loss** (`fac3df2`, `22874c5`): all 12
   steps completed, then `save_file`'s temp-file-rename serialization lost a
   Windows sharing-violation race (os error 32) on every attempt — retries
   re-ran the race (attempt 5). The fallback serializes in memory and writes
   the final path directly; the manifest-last rule keeps an interrupted write
   ineligible.
3. **Dtype-dishonest publication** (`5453dd3`): publication silently upcast
   trained bf16 gates to fp32 while the manifest recorded bfloat16 — a
   payload that could never pass its own dtype/hash verification, and that
   `apply_router_payload` would refuse against the bf16 model anyway
   (attempt 6 died exactly there; the eval worker's verified load correctly
   refused). Publication now stores the trained dtype; the hash rule is
   byte-identical for fp32 payloads, so the CPU pilots' verifiability is
   unchanged.

One operational refusal was also exercised: attempt 5 reused attempt 4's
registry and was refused with `duplicate persisted experiment id: baseline`
before any worker ran — the registry invariant doing its job.

## Evidence inventory (this directory)

- `preflight_probe.py` + rung-3 prerefusal artifacts (sibling directory
  `2026-09-14-router-healing-p11-rung3/`): strategy A/B/C/D measurements.
- `train-stdout.attempt{1..7}*.log` — every attempt preserved, including the
  aborted fp32-baseline attempt (the base-arm policy defect), the budget
  refusal, the preflight false refusal, two os-error-32 failures, and the
  dtype refusal.
- `work.attempt{1..6}-*/` — preserved work trees for every failed attempt.
- `work/` — the successful run: `runs.db` (registry, results, promotion
  event), `.chowder/runs/router-pilot-9b-ed626dc17192/` (worker result with
  preflight/trainability/frozen/lifecycle evidence, published bf16 payload,
  run failure-free), `.chowder/evals/{baseline,router-pilot}-*/`
  (per-arm worker results with load-policy census and phase ledgers).
- `router-corpus-9b-pilot.txt` SHA-256 `15d5f5f5…`, `router-holdout-9b-pilot.txt`
  SHA-256 `97e87c22…` — the pinned corpora.

In-repo: preregistration `4456e75`, this report committed to
`docs/quals/` on `docs/p11-rung3-prereg` (PR #161).
