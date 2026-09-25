# Teacher-free distillation pilot — status report

Branch: `experiment/teacher-free-distillation-pilot` (base audit SHA `3d5c0101e40406e01defa7591e025f28178a7be9`)
Date: 2026-09-25. All numbers below are from runs executed during this session; nothing is extrapolated.

## IMPLEMENTED

- **Phase 1 audit** (`AUDIT.md`): PR #201 reviewed file-by-file; infra map (PEFT backend, completion-only masking, memory planner, hardware detection, evaluators, TUI); gaps recorded.
- **Source catalog** (`sources.json`): OT3 approved (apache-2.0, revision `61bcf9d4…` = live HEAD, QwQ-32B-trace caveat), MoT BLOCKED (no dataset license — verified via HF API; model license does not cover it), SWE-smith-trajectories approved (MIT, revision `08e109b4…` = live HEAD; `resolved` flag documented as a claim).
- **Ingest + normalization** (`prepare.py` upgraded): per-example provenance records (source, revision, license, original id, content SHA-256, char/token counts, split), banded-minhash near-duplicate quarantine (`quarantine.jsonl`), per-source quality report with rejection reasons, deterministic dev split (unchanged semantics for prior tests).
- **Fetcher** (`fetch_smith.py`): bounded seeded reservoir, retry/backoff, resume via sidecar, `claimed_resolved` preserved as a claim only.
- **Sandboxed replay** (`replay_smith.py`): Podman isolation (2 GB / 2 CPU / 256 PIDs), network ON only for clone+install, network OFF for replay+tests, red→green protocol (pre-patch failure must reproduce, then patch, then green), per-task raw execution logs, digest-evidenced trace records only on genuine success, SWE-smith `instance_id` parser (repo@commit extraction).
- **Student selection** (`student.py`): Qwen3-1.7B primary (pinned `70d244cc…`, apache-2.0), Qwen2.5-Coder-3B control (license gated on operator acceptance), workload estimates wired through the production `plan_memory` on real `detect_hardware()` output.
- **Training preflight** (`preflight.py`): 8-check CPU suite (tokenization, production completion-only masking, nonempty targets, LoRA coverage, gradient flow adapter-only, finite loss, checkpoint roundtrip, resume integrity) on a 1-layer model.
- **Recipes** (`recipes/*.json`): A (SFT on verified traces), B (repair continuation with 20% general mix + forgetting probe), C (preference, GATED until genuine matched pairs with execution evidence on both sides exist).
- **Evaluation protocol** (`evaluate.py`): repair split-by-repository leakage check, prompt-overlap check, GSM8K extraction/scoring, repair behavior metrics (nonexistent reads, recovery, premature success, repeated calls, green rate), paired comparison with advisory regression flags — never auto-promotion.
- **TUI screen** (`tui_teacher_free.py`): 8-stage workflow whose displayed state is derived exclusively from real on-disk artifacts; refresh re-reads the filesystem; CPU preflight button runs the real script.
- **Pilot data on disk** (`C:\Users\nikma\chowder_teacher_free\`): OT3 pilot exports (300 + 8000 sampled), prepared pilot_v1/v2 with manifests, SWE-smith fetch samples (6 + 20), 8 real sandbox replays with raw logs.

## TESTED

- `tests/test_teacher_free_prepare.py`: **6 passed** (existing, verified by fresh run — not by trusting the PR claim).
- `tests/test_teacher_free_phase2.py`: **10 passed** (new: near-dup quarantine, provenance fields, fetch gating, fetch claim/evidence separation, replay refuses without podman, replay skips without instance metadata, verified-record → repair examples, repair split leakage, GSM8K scoring, repair behavior metrics).
- `tests/test_tui_teacher_free.py`: **3 passed** (artifact-derived stage state incl. real repo dir; app mount).
- CPU preflight: **8/8 checks passed** (executed twice, once persisted to `preflight_result.json`).
- Two real bugs were caught and fixed *by* these tests during development (mask-check indexing; near-dup key too weak for suffix variants) — evidence the suite has teeth.

## TRAINED

**Nothing.** No student weights were modified. Training requires: (a) a verified dataset large enough to matter, (b) exclusive GPU access, and (c) explicit operator authorization. None were present. The preflight proves the training path is correct; it does not constitute training.

## EVALUATED

**No model evaluation ran** (no trained student and no authorized baseline GPU run). What *was* measured, against real data:

- OT3 pilot (n=300, scan 4,000; then n=8,000, scan 100,000): **296/300 (98.7%) and 7,986/8,000 (99.8%) rows exceed the char caps** (24k / 8k). OT3's mean row size ≈ 47 KB — the binding constraint for distilling from OT3 is context budget, not licensing or duplication. At 8k chars the pipeline accepted 6 rows (5 train / 1 dev) — pilot_v2 manifest `088e3abe15b16420…`.
- SWE-smith replay (n=8 sandboxed replays, real Podman execution): **0 verified / 6 not_green / 2 skipped**. Failure modes: 4× missing per-instance test deps (`anyio`, `apispec`) — plain `pip install -e .` is insufficient, SWE-smith's published per-repo environment specs are required; 2× empty/mismatched patch vs. repo (dataset-integrity finding: `bottlepy__bottle` task carrying a `paramiko` patch — the upstream `resolved` flags are unreliable). The verification gate certified nothing without genuine red→green evidence, which is the intended fail-closed behavior.
- Student memory plan (static estimate on real hardware): Qwen3-1.7B LoRA fits the RTX 5060 Ti (primary pool 15.9 GB; workload 3.44 GB frozen + 0.34 GB trainable + 11.27 GB activations worst-band + 0.17 GB optimizer; bottleneck `compute_or_kernel`, not memory).

## BLOCKED

- **GPU training of any condition (A/B/C)**: requires explicit operator authorization and device exclusivity (another campaign is currently using GPU memory on this machine). Recipes and preflight evidence are ready.
- **Mixture-of-Thoughts ingestion**: no dataset license exists to review. Needs an upstream licensing decision.
- **Verified repair corpus**: needs per-instance SWE-smith environments (their published per-repo setup scripts/requirements) installed in the sandbox before tests can run; then red→green replay at pilot scale.
- **gen-2 comparison**: must load and measure the actual frozen gen-2 reference under the same protocol — an operator-authorized GPU task.

## NOT YET ATTEMPTED

- Condition A/B/C training runs, checkpoint lineages beyond preflight scale.
- Real GSM8K / held-out perplexity / instruction-following measurement for any model (untouched student included).
- Preference-pair construction at scale (Condition C stays gated; genuine evidence-backed pairs are rare in the current fetch sample).
- Catastrophic-forgetting measurement (needs a trained condition-B student first).

## Answer to the principal question, honestly

Not answerable yet by measurement — no student has been trained. What this session established: the pipeline exists, is fail-closed end to end, and its gates have already intercepted two real problems (unlicensed source content; unverifiable repair claims). The immediate technical blockers before a first meaningful GPU run are (1) per-instance SWE-smith environments for repair data and (2) an OT3 context-length strategy (filter to short-trace subsets or accept low yield). Operator authorization is the remaining gate for everything GPU-side.
