# P11 rung 5 — resume check: fan cannot start, and why (2026-09-17)

Prereg: `P11_RUNG5_PREREG_2026-09-16.md` (commit `54490e3`, this branch).
This note records what was checked before any arm was launched, and the
refusal that followed. Nothing here changes the prereg.

## Contention and identity preconditions: clear

| Check | Result |
| --- | --- |
| Device free | RTX 5060 Ti `668 MiB / 16311 MiB`, 6% util; no training process resident (checked before launch) |
| Second card | RTX 2060 idle (`0 MiB`) — not part of this rung (`accelerator_count = 1`) |
| Base artifact present | `F:\llm-models\Qwen3.8-9B-HotCore-CW-E16-k2-h2176` exists |
| Training corpus SHA-256 | `15d5f5f51a739ceee2712fe5b7b550982aba7272f633781f06d5a7ea64f47941` — **matches the frozen pin** |
| Independent holdout SHA-256 | `2e99668207319a1d2b702408bcc659a525fd226d027eebeaebea918e2ad21e97` — **matches the frozen pin** |
| Concurrent CUDA work | none; no Ollama/local-model load started for this rung |

So the rung is **not** hardware-blocked or contention-blocked.

## The refusal: the intervention it preregisters does not exist yet

§3 of the prereg freezes four new spec fields and §4 freezes an upgraded
saturation instrument. Neither is implemented on this branch:

```
$ for f in gate_initialization gate_init_std router_logit_scale router_logit_soft_cap; do
    grep -rl "$f" src/ | tr '\n' ' '; echo; done
   (no matches under src/ for any of the four)
```

They appear only in the preregistration bundle itself
(`P11_RUNG5_PREREG_2026-09-16.md`, `judge_rung5_2026-09-16.py`,
`build_rung5_campaign.py`, `judge_rung5_mutation_probe.py`). No other branch
or worktree in this checkout carries them either.

Consequence, stated plainly:

- **Arm A** (`gate_initialization="artifact"`, no scale, no cap) would run on
  today's code — it is a rung-4 reproduction, and spending a device window to
  re-measure a known result is not what this rung is for.
- **Arms B, C and D** cannot run at all: the knobs they are defined by do not
  exist, so the campaign builder's four project configs would be rejected (or,
  worse, silently ignored) rather than executed as preregistered.

The prereg is explicit that the implementation "lands afterwards on the same
branch, and the four arms run against the frozen pins below". It has not
landed, so the arms are **not launched**. Adapting the experiment to the code
that happens to exist — approximating a reseeded gate with some other
scaling, or dropping the cap — is exactly the silent-adaptation failure the
program forbids.

## What has to land before the four arms can run

From the prereg (not invented here):

1. §3.1 four spec fields with refusal-not-coercion validation, folded into the
   spec digest like every other knob.
2. §3.2 the frozen forward composition `z → u → v → softmax/top-k`, with the
   null path a **bit-identical** no-op so arm A reproduces rung 4 exactly.
3. §3.3 the seeded `small_normal` initialization, with
   `gate_init_std` / `gate_init_seed` / `initialized_gate_digest` recorded and
   the frozen-weight digest taken *after* initialization.
4. §4 the saturation instrument: rows-exactly-one-hot, top-2 gap, gradient
   magnitude and implied update in bf16 half-ULP units — replacing the binary
   grad-nonzero reading that reported L31 as healthy while its gradient was
   exactly zero.
5. The §4.5 unit tests, including the arm-A bit-identity test and the
   degeneracy-broken preflight that makes arm D refuse before step 1.

Then: four sequential 48-step runs (`arm-A`..`arm-D`), each with its own
project file, work directory and registry — never a reused run directory — and
`judge_rung5_2026-09-16.py` against the real evidence root.

## Status

- Rung-5 arms executed: **none** (0 of 4).
- Blocker: implementation absent — **not** hardware, not contention, not
  budget.
- Frozen pins re-verified this date and unchanged.
- Rung 4 / 4b negative results stand untouched; this rung supersedes their
  *contract*, not their record.
