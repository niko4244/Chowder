# REAP128_v2 phase 27 — throughput-integrity hardening, re-audit, scorecard, and HLE launch readiness

**2026-10-26.** This phase does four things and deliberately refuses a fifth:
it closes a real throughput-integrity hole in the original-cohort audit, re-runs
that audit over the complete Q8_0 cohort, publishes a measured-versus-reference
scorecard with every unmeasured cell marked UNKNOWN, and brings a text-only
HLE research kernel to a launch-ready, dry-run-tested state. It does **not**
launch that kernel, because doing so requires moving access-gated benchmark
data (or a Hugging Face credential) to a third party, which is a data-handling
decision for the operator rather than an engineering one.

Nothing here changes the frozen selector, the source models, weights, datasets,
prompts, decoding or statistical thresholds, and no historical result is
revised.

## Live reconciliation

| Item | State |
| --- | --- |
| Working tree base | `6786d34` on `feature/reap128v2-selection-20261025` |
| PR #282 | **OPEN, DRAFT, MERGEABLE, UNMERGED** (head `6786d34`) |
| Local commits this phase | token-ceiling regression; phase-27 evidence (unpushed by design) |
| Privacy | no push, no merge, no release, no raw prompts or private outputs published |
| Experimental branch | separate research worktree remains separate; not scientific authorization |

The unpushed state is intentional: this phase adds evidence and a launch-ready
harness, and does not claim a new gate outcome that would justify moving the PR.

## 1. Closed hole: rows could exceed the frozen completion-token ceiling

`chowder_batch/reap_external_full_audit.py` validated that a scored row's
`completion_tokens`, `seconds` and `predicted_per_second` were *numbers*, but
never checked them against the frozen per-suite completion limits. A row that
reports more generated tokens than the frozen decoding settings permit cannot
have produced the throughput it advertises, so an inflated row would have been
accepted as valid throughput evidence.

The audit now rejects any row whose `completion_tokens` exceeds
`reap_eval.MAX_TOKENS[suite]` (GSM8K 1024, HumanEval 768), and the regression is
covered by `tests/test_reap_external_full_audit_20261026.py`, including the
boundary case (exactly at the limit is admitted, one token over is refused).

Suite result: **123 passed** for that file.

## 2. Re-audit of the complete Q8_0 cohort under the hardened rules

The complete `reap128v2-q8-full-postload-reclaim` output was re-downloaded and
re-audited with the hardened auditor.

| Suite | n | correct | accuracy | generated tokens | wall | tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GSM8K | 1,319 | 1,220 | 92.49431% | 467,560 | 5,726.947 s | 81.6421 |
| HumanEval | 164 | 152 | 92.68293% | 40,281 | 538.774 s | 74.7642 |

Verdict: `COMPLETE_LINUX_COHORT_VERIFIED_PROVISIONAL_NOT_WINDOWS_GATE`, with
row digests `9ff2b418…` (GSM8K) and `cd3c9d15…` (HumanEval) and session
`95876063255e45159f57e29306e0e885`.

**Honest scope of this result.** The session id is a fresh `uuid4` minted by the
kernel and it equals the phase-26 Q8 session, and the row digests are identical.
This is therefore a **re-verification of the same completed artifact**, not a
second independent execution. What it newly proves is that the recorded rows
comply with the frozen token ceilings and survive the stricter auditor, which is
exactly the throughput concern above. It does not constitute a fresh
measurement and is not reported as one.

## 3. Measured quantized arms (unchanged, re-read from the phase-26 artifacts)

| Suite | Q8_0 | Q4_K_M |
| --- | ---: | ---: |
| GSM8K (1,319) | 1,220 — 92.49431% | 1,229 — **93.17665%** |
| HumanEval (164) | 152 — 92.68293% | 151 — **92.07317%** |
| GSM8K tokens/s | 81.6421 | **98.4385** |
| HumanEval tokens/s | 74.7642 | **89.2712** |

Both arms remain **PROVISIONAL** against the Windows serving-owner gate, and the
HumanEval figures remain the original scorer booleans rather than a new
isolated-sandbox execution.

## 4. Scorecard: measured scores and reference scores kept apart

Two artifacts, both committed and regenerable:

- `docs/quals/reap-v2-frontier-scorecard-20261026.json` — the numbers plus the
  comparability rules and source URLs.
- `docs/quals/reap-v2-frontier-scorecard-20261026.png` — rendered by
  `scripts/reap_v2_frontier_scorecard_chart.py`.

Panel A plots only what this repository measured, on the full frozen
populations, from one harness. Panel B covers the 11 screenshot environments and
prints **REAP: UNKNOWN** in every row, because no REAP score exists for any of
them. Vendor marks are drawn hollow with an asterisk where the vendor's
benchmark version or tool setting does not match the screenshot row:

- Qwen3.8-27B HLE 30.8 (no tools), DeepSWE 1.1 42.2, NL2Repo-Bench 42.3,
  Agents' Last Exam Pass@1 20.4; its Terminal-Bench figure 73.0 is
  **Terminal-Bench 2.1**, not 4.0, and is marked accordingly.
- Opus 5.5 Terminal-Bench 4.0 66.4 (xhigh), Chartography with tools 89.0,
  AutomationBench 40.0 (Zapier official test, harness not necessarily 1.0.6),
  and HLE 67.7 **with tools**, which is not the screenshot's no-tools row.

No vendor number is presented as a REAP score and no cell is filled by
extrapolation.

## 5. HLE: frozen text-only protocol, launch-ready kernel, and the refusal to launch

### Frozen protocol (`chowder_batch/reap_hle_protocol.py`)

Declared prospectively, before any outcome was observed:

| Setting | Value |
| --- | --- |
| Dataset | `cais/hle` @ `5a81a4c7271a2a2a312b9a690f0c2fde837e4c29` |
| File | `data/test-00000-of-00001.parquet`, 274,276,147 bytes, sha256 `6d0ee060…726a` |
| Subset | text-only only (empty `image` field), **2,158** of 2,500 |
| Answer types | 1,645 exactMatch, 513 multipleChoice |
| Order | ascending `sha256(id)`; a completed prefix is an unbiased sample |
| Budget | 1,200 items and 32,400 s (9 h) of a 12 h session |
| Prompt | official HLE `Explanation / Answer / Confidence` format, no tools |
| Decoding | greedy, temperature 0, seed 0, max 8,192 completion tokens |
| Serving | `--fit on -np 2 -c 16384 --jinja` (8,192 tokens per slot) |
| Grading | deterministic normalized match, **non-official** |

The geometry differs from the original cohort (`-np 4`) on purpose: 4 slots
leave only 4,096 tokens per request, which truncates legitimate multi-step
reasoning. Because this is a *new* protocol rather than a re-run of the original
one, changing it prospectively is legitimate and is recorded in the manifest
digest.

### Honesty boundary

HLE's official grading is an LLM judge (`o3-mini`). No judge is invoked here,
so this protocol can only ever produce a deterministic non-official signal. The
receipt carries `official_hle_score_claimed: false`,
`amendment4_gate_eligible: false`, `chosen: null` and
`release_approved: false`, and an official HLE score remains **UNAVAILABLE**.

### Guards actually exercised

`scripts/reap_v2_kaggle_qualification_20261026/q4_hle_textonly.py` pins the
imported protocol source bytes *and* the protocol manifest digest separately,
pins the parquet bytes/SHA, the Q4_K_M GGUF bytes/SHA and the server binary SHA,
enforces the 24 GiB host and 12,288 MiB per-GPU floors before load and every 5 s
while serving, refuses to overwrite an existing bin stage, tears down only its
own process group, and writes its receipt atomically.

- 34 local tests pass (`tests/test_reap_hle_protocol_20261026.py`,
  `tests/test_reap_hle_kernel_20261026.py`), including drift pins that fail if
  the kernel's constants diverge from the frozen protocol.
- Two sandboxed dry runs executed the kernel with no GPU and no benchmark data.
  The first refused at the protocol guard. The second exposed a **real defect**:
  the kernel conflated the protocol *file* digest with the *manifest* digest, so
  a correct install would have aborted at startup. Both digests are now separate
  constants with separate tests, and the dry run proceeds to the dataset guard
  and refuses there, which is the intended ordering.

### Why it was not launched

The only GPU environment that has actually admitted this model is Kaggle T4×2.
Feeding HLE to it requires one of:

1. uploading the access-gated parquet as a **private** Kaggle dataset, which the
   dataset card asks users not to re-upload; or
2. storing a Hugging Face token in Kaggle Secrets and downloading at runtime,
   which transmits a personal credential to a third party through a UI that
   needs the operator present.

Both are data-handling decisions with external effects. Neither is a technical
blocker, and neither is authorised by "enable access on Hugging Face", so this
phase stops short of it and asks instead of assuming.

Once the operator chooses a path, the launch sequence is: stage the frozen
protocol file as an input dataset, create the private HLE input (or the secret),
push the kernel with T4×2 and the pinned toolchain input, then collect
`reap-q4-hle-textonly-*.jsonl` and its summary and audit them locally.

## 6. Runtime and admission state

Windows host, read-only counters at the end of this phase:

| Resource | Available | Requirement |
| --- | ---: | ---: |
| Physical RAM | 10.57 GiB of 63.90 GiB | ≥ 29.3146 GiB |
| Commit | 14.51 GiB of 127.01 GiB | ≥ 34.3146 GiB |
| VRAM (RTX 5060 Ti) | 4,823 MiB of 16,311 MiB | ≥ 12,288 MiB |

The largest live consumer is `llama-server` PID 49124 at 4.62 GiB working set,
which also holds the bulk of the resident VRAM. Suspending it would raise free
VRAM above the floor but free physical RAM only to roughly 15 GiB — still about
14 GiB short — so the original Windows Q4 launch would remain unsafe. This is
recorded rather than attempted: the instruction is not to relaunch a known
unsafe workload, and the low-RAM abort already demonstrated the real behaviour.

Kaggle GPU quota at the last read: **15.55 h remaining** of 30 h. The original
`reap128v2-random128-full-postload` control remains **RUNNING** and was not
duplicated.

## 7. Gate state (unchanged)

```json
{"status":"BLOCKED","chosen":null,"release_approved":false}
```

- Windows Gate0/Gate1/Gate2: **UNKNOWN**, no eligible Windows serving attestation.
- Random128 independent full control: **RUNNING/UNKNOWN**.
- Original same-first-200 dense-vs-REAP Q4 ≥ 1.3× matched native speed: **UNKNOWN**.
- Original REAP160 quality/speed fallback: **UNKNOWN**.
- BF16/CUDA logits and cross-platform parity: **UNKNOWN**.
- Historical REAP128-v1 Gate1: **FAIL remains FAIL**.
- Wald versus paired non-zero-margin McNemar: prospectively **UNRESOLVED**.

No gate passes, so the selector stays BLOCKED with `chosen: null`.

## 8. Concrete next steps

1. Decide the HLE data-handling path (private Kaggle input dataset, or Kaggle
   Secrets with an HF token); then launch `q4_hle_textonly.py` and collect the
   text-only cohort.
2. Collect the random128 control receipt when it reaches a terminal state and
   audit it against the hardened auditor.
3. For the coding/agent rows (Terminal-Bench, DeepSWE, ProgramBench,
   NL2Repo-Bench, AutomationBench, Agents' Last Exam, Apex), obtain an
   independently verified no-egress rootless isolation host before any
   model-generated shell or Python is allowed to run.
4. For BF16, dense and REAP160, obtain a sufficiently provisioned CUDA host so
   numerical parity and the dense speed control can be measured rather than
   assumed.
