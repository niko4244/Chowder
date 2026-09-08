# Kaggle as a qualified parallel evaluation backend for Parents C/D

Handoff for this isolated branch/worktree. Written for whoever picks this
up next (human or agent) with zero other context. Everything below is
either implemented-and-tested tooling, or an explicitly-labeled real
measurement -- nothing here claims a Kaggle qualification run or a real
C/D evaluation happened, because none did.

## Mission and isolation

Built in parallel with the live, local `retry7` tournament, which was
explicitly off-limits and never touched:

- Branch: `feat/kaggle-cd-acquisition-and-equivalence`
- Base: `origin/main` at `b8fdf496a0fc3a4317effe294bf6928afed002aa`
  (includes PR #142, the four-parent freeze pipeline + C/D acquisition
  tooling from the prior parallel task).
- Fresh `git worktree` (`../Chowder-kaggle-cd`), separate from every
  other checkout on this machine.
- `Chowder-Protected` was only ever read (a copied sqlite registry, and a
  directory listing) to confirm real, current tournament state:
  **Parent A is complete** (real measured `peak_gpu_mib_sampled: 16004`,
  `wall_seconds: 7425.1`, `quantization: 4bit`, `precision: bf16`);
  **Parent B is still running** (6 of 9 protected suites have produced
  predictions as of this branch's work; it was never stopped, restarted,
  or otherwise touched). No file under `Chowder-Protected` was written.
- No GPU use, no CUDA, no model downloads, no Kaggle API calls, no
  `kaggle` CLI invocation, anywhere in this branch's own work. The `kaggle`
  CLI and Kaggle credentials are present on this machine, but were never
  invoked -- every network/GPU-touching step in the new code is an
  injected callable, exercised only against synthetic fixtures in tests.

## A real, evidence-based finding that shapes everything below

Parent A's real retry7 run peaked at **16004 MiB** VRAM on the local RTX
5060 Ti -- a 16 GiB-class card -- under exactly the configuration this
program uses (4-bit + bf16, protocol v2, 256-token budget). That is
**~97.7% of a 16 GiB card's nominal capacity, roughly 2.3% headroom**,
read directly from that run's persisted evidence, not estimated.

A Kaggle T4 is also a 16 GiB-class card. This means the *local* run
itself is already right at the edge of what a 16 GiB GPU can hold for
this workload -- Kaggle is not being asked to do something local hardware
handles comfortably. `kaggle_launcher.estimate_required_vram_gib`, given
this real measurement plus a 10% cross-architecture margin, estimates
**~17.2 GiB required** -- above a T4's usable capacity by default. The
preflight this branch built (`kaggle_launcher.preflight_vram_headroom`)
will therefore very likely **refuse to launch on a single T4 by default**,
and that is intentional: this branch does not shrink the generation
budget, weaken quantization, or otherwise change evaluation semantics to
force a fit (explicitly forbidden by the mission). If a real attempt is
made and the preflight blocks it, that is a genuine, real capacity
constraint to report to a human, not a bug in this tooling. A different
Kaggle accelerator with more headroom (e.g. a P100, or reducing
`_VRAM_ESTIMATE_MARGIN` after genuinely confirming a T4 handles it) are
the two honest paths forward if this blocks in practice.

Separately, confirmed by reading `evaluators.base_text_worker._dtype`
directly: **T4 GPUs cannot run bf16** (`torch.cuda.is_bf16_supported()`
is False on compute capability 7.5; that function raises
`RuntimeError` on a bf16 request there). Kaggle runs must use `fp16`
instead, which changes the protocol digest (`precision` is a spec
field) -- handled explicitly via
`kaggle_launcher.PRECISION_DIVERGENCE_REASON` and
`kaggle_equivalence.qualify_backend`'s `declared_digest_divergence_reasons`
mechanism, never silently.

Also confirmed: `base_text_worker.evaluate` only ever pins a model to
**one** CUDA device index (`device_map={"": index}`); there is no
multi-GPU device-map path in that (off-limits) file. A Kaggle "T4 x2"
allocation is therefore used as **one T4 per evaluation job** throughout
this tooling -- never assumed to behave like a single 32 GiB card, per
the mission's own instruction.

## What was implemented (all four phases have real tooling; none has a real Kaggle run)

### Phase 1 -- C/D acquisition on Kaggle

- [`src/chowder/qwen38_acquisition.py`](../src/chowder/qwen38_acquisition.py) (already on `main` from the prior task) gained `PARENT_A_PIN`, `PARENT_B_PIN`, `ALL_PARENT_PINS`, and `PARENT_LABELS` so every script below shares one pin/label table.
- [`kaggle/acquire_parent.py`](../kaggle/acquire_parent.py) -- real Hugging Face Hub wiring (`HfApi`/`snapshot_download`) around the already-tested `qwen38_acquisition.acquire_parent`. Reads the HF token only from a Kaggle Secret (`HF_TOKEN`), never prints or logs it. Refuses to run outside a real Kaggle kernel (no `kaggle_secrets` available) -- verified by test.
- [`kaggle/upload_protected_suite.py`](../kaggle/upload_protected_suite.py) -- the one script that would move protected suite content off the local machine, as a **private** Kaggle dataset by default; refuses to publish publicly without a separate, explicit acknowledgement flag. **Not run.**

### Phase 2 -- Kaggle parent-evaluation runner

- [`src/chowder/kaggle_launcher.py`](../src/chowder/kaggle_launcher.py) -- reuses `parent_tournament.evaluate_parent` / `.tokenizer_evidence` / `.LocalParent` unchanged; adds only environment fingerprinting, the VRAM preflight described above, and the one-field (`precision`) spec adaptation.
- [`kaggle/bootstrap_environment.py`](../kaggle/bootstrap_environment.py) -- installs `chowder-ai` from an exact pinned git commit (never a branch), cross-checks pip actually resolved that commit (via `direct_url.json`'s `vcs_info.commit_id`) rather than trusting a possibly-stale cached wheel, and writes the environment fingerprint.
- [`kaggle/run_parent_evaluation.py`](../kaggle/run_parent_evaluation.py) -- the one evaluation entry point used for both Phase 3 (re-running Parent A) and Phase 4 (Parents C/D); wires real paths/registry/tokenizer measurement to `kaggle_launcher.run_kaggle_parent_evaluation`.

### Phase 3 -- mandatory equivalence qualification

- [`src/chowder/kaggle_equivalence.py`](../src/chowder/kaggle_equivalence.py) -- builds the full item-level (54-item), dimension-level, and environment-level comparison and turns it into a machine-readable `BackendQualificationRecord`. Reuses `evaluators.base_text_worker._final_answer`/`._normalize` directly (not reimplemented) and `parent_freeze.classify_delta`/`ITEMS_PER_SUITE` (promoted from private to public so both modules share the exact tie/weak-signal/clear-difference convention). Never relaxes the item-score bar; a protocol-digest mismatch is only tolerated when the caller declares and justifies it (e.g. the precision divergence above) -- an undeclared mismatch still fails closed.
- [`kaggle/build_equivalence_report.py`](../kaggle/build_equivalence_report.py) -- the CLI driver; exit code 0 when qualified (with or without acknowledged differences), 2 when not. This script has no Kaggle/GPU dependency and was exercised **fully, end to end**, with synthetic fixtures.

### Phase 4 -- evaluate C and D

Covered by the same `kaggle/run_parent_evaluation.py` (Phase 2 above) --
no separate script needed; the tooling does not distinguish "the
qualification re-run of A" from "the real evaluation of C/D" beyond
which `--parent`/`--model-dir` is passed. **Not run.**

## What is explicitly NOT done

- No Kaggle kernel was pushed, started, or run. No Kaggle dataset was
  created. No Hugging Face download happened, on Kaggle or locally, for
  C or D.
- **No equivalence qualification has actually been performed.** The
  qualification comparison logic is fully built and tested against
  synthetic data; it has never been run against a real Parent-A-on-Kaggle
  result, because no Parent-A-on-Kaggle result exists yet.
- **Parents C and D have not been acquired, verified, or evaluated
  anywhere.** No claim of tournament validity is made for them.
- Parent B's local tournament run was not stopped, restarted, inspected
  beyond a read-only directory listing, or otherwise touched.
- `docs/HANDOFF.md` (the shared, actively-updated continuity doc for
  `main`) was deliberately **not edited** by this branch -- it documents
  `main`-landed work and Parent B's tournament is actively appending
  evidence there/adjacent to it right now; a separate, clearly-scoped
  document (this one) avoids any collision with that live process or
  with whatever session is tracking retry7's progress.

## Verification run in this branch

```
python -m pytest tests/test_kaggle_equivalence.py tests/test_kaggle_launcher.py \
    tests/test_kaggle_scripts.py tests/test_qwen38_acquisition.py tests/test_parent_freeze.py -q
# 13 + 14 + 24 + 14 + 16 = 81 passed

python -m ruff check src/chowder/kaggle_equivalence.py src/chowder/kaggle_launcher.py \
    src/chowder/qwen38_acquisition.py src/chowder/parent_freeze.py kaggle/ tests/
# All checks passed!

python -m pytest -q
# 1324 passed, 77 skipped (GPU-gated), 0 failed
```

## Next steps for whoever continues this (the mission's own checkpoints)

1. **Kaggle environment + C/D acquisition path verified** -- run
   `kaggle/bootstrap_environment.py` then `kaggle/acquire_parent.py
   --parent C` (and `--parent D`) for real, in a private Kaggle GPU
   notebook with an `HF_TOKEN` secret attached. Confirm the acquisition
   summary's manifest digest and architecture metadata look right before
   trusting the checkpoint.
2. **Local-vs-Kaggle parent equivalence result** -- acquire Parent A on
   Kaggle too (`acquire_parent.py --parent A`), upload the frozen suite
   privately (`upload_protected_suite.py`, after a deliberate human
   decision to do so), run `run_parent_evaluation.py --parent A` on
   Kaggle, download its predictions/evidence back locally, and run
   `build_equivalence_report.py` against the real local Parent A result
   already in `Chowder-Protected`'s registry. **Expect the VRAM preflight
   to be the first real obstacle** given the finding above -- do not
   route around it by changing quantization/precision/budget without
   first genuinely confirming (e.g. by trying and reading the real error)
   that a T4 cannot hold this workload; a different Kaggle accelerator
   tier may be the honest fix.
3. **Parent C full 54-item result** -- only after step 2 reaches
   `qualified` or `qualified_with_acknowledged_differences`.
4. **Parent D full 54-item result** -- same gate.
5. **A/B/C/D evidence ready for final parent selection** -- once B
   finishes locally and C/D are qualified-and-evaluated on Kaggle, the
   already-merged `parent_freeze.build_selection_packet`/
   `freeze_selected_parent` (PR #142) is the consumer; nothing new is
   needed there.

No PR was opened and nothing was pushed to `origin` from this branch.
