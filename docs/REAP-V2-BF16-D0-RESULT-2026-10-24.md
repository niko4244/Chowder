# REAP128_v2 original BF16 704×40 — measured D0 result

**Status: COMPLETE and MEASURED. D0 gate `PASS` on the frozen threshold
`0 < perplexity < 20`.**

First complete, resumable, original-BF16 704-sequence × 40-layer observation of
`Qwen/Qwen3.6-35B-A3B` at the pinned revision, produced by the versioned
supervised observer on PR #280.

## Result

| Gate | Threshold | Measured | Verdict |
|---|---|---:|---|
| D0 frozen next-token perplexity | `< 20` | **15.451270210051794** | **PASS** |

Per-source perplexity of the same run (informational, not a gate):

| Calibration source | PPL |
|---|---:|
| `openr1_math` | 2.3723 |
| `openstax_prose` | 5.1677 |
| `ot3_code_reasoning` | 7.4230 |
| `fable_agentic` | 9.1776 |
| `fable_tool_use_lordx64` | 13.4607 |
| `swe_smith_trajectories` | 15.1869 |
| `nemotron_agentic` | 59.5288 |
| `fable_traces_swarm` | 89.7263 |

The two high-perplexity sources are a minority of tokens and are reported for
transparency; the gate is the whole-calibration next-token perplexity above.
The distribution is consistent with an unmolested base model: the frozen
calibration is dominated by SWE/agentic trajectories, which sit near the
aggregate value.

## Run identity

| Field | Value |
|---|---|
| run_id | `20261024-222950-223f34d3` |
| branch / HEAD at completion | `fix/reap-bf16-atomic-resume-20261024` @ `e239096` |
| output (C:) | `C:\llm-work\reap\observe-v2-bf16-atomic-20261024-222950-223f34d3` |
| scratch (F:) | `F:\reap-v2-bf16-atomic-20261024-222950-223f34d3` |
| sealed result | `observation.json` (76,673 bytes) |
| `observation.json` SHA-256 | `799cc170767f6b17b40c3b01eb3f95ec71310a1154a1ce0570fb07468dde76ed` |
| model / weights | `G:\llm-cold\bf16\Qwen3.6-35B-A3B` |
| calibration | `C:\llm-work\reap\calib-v2-ornith-bare-20261023\calibration.jsonl` |

Frozen inputs were **not** modified. The run consumed them exactly as pinned:

- calibration SHA-256 `f6a09b694ae43a6d8b44b5180a861856501295eabf3d5297eb9749a008cdd929`
- ordered BF16 token IDs SHA-256 `78b8f482f61cfd4b20bba6982a20c8eeb5a757b20069f2520273d4a315335f92`
- 704 sequences, **1,192,632** actual retokenized tokens, 704 ordered row lengths
- 40 decoder layers, 256 routed experts, top-8
- upstream pin `Qwen/Qwen3.6-35B-A3B` @ `995ad96eacd98c81ed38be0c5b274b04031597b0`,
  manifest `54e7e0d121cab5fd7e331be1ee38c5b4deddd7eee7f0cc7b112a4322f8418dca`
- all 26 shards (71,903,776,776 bytes) plus 11 metadata files digested and matched
  to that independently pinned upstream manifest — twice (admission/resume and
  again before publication)

## Coverage and integrity

| Check | Evidence |
|---|---|
| Layers | 40 committed, 40 distinct indices `0..39` |
| Checkpoint generation | `40` (embeddings = generation 0) |
| Hash chain length | 41 (generations 0–40) |
| Receipt schema | `chowder.reap-bf16-layer-checkpoint.v1` |
| Receipt `root_sha256` | `514fdc7b0c0949120e2ec58c4eb11469313fdb6d526a91b17670e41034f3709a` |
| `source_identity_sha256` | `a60613d8eafc0708f652e11b68792ae0255cda7ebb66907983a3bae3fd7b86bf` |
| `active_state_sha256` | `3326c73ac9766c524f8b99e74257ab0593f33b6ead2f72658180ba7385375c27` |
| Hidden storage | dual file-backed CPU BF16 slabs, `numeric_conversions: False` |
| Slab size | 4,885,020,672 bytes each (9,770,041,344 total) |
| Router census | `routed_tokens = 9,541,056` (= 1,192,632 × 8) on **all 40** layers |
| Expert coverage | `experts_hit = 256` on **all 40** layers |
| Per-layer stats mirrored to `out/` | `layer-00.pt` … `layer-39.pt` (10,333 bytes each) |

The exact per-layer routed-token count is the strongest available internal
consistency check: top-8 routing over 1,192,632 tokens must produce
9,541,056 routed-token events, and it does on every layer, with all 256 experts
exercised on every layer.

## Compute behavior

| Metric | Value |
|---|---:|
| Sum of per-layer compute | 9,720.0 s (2.70 h) |
| Per-layer compute | 217.8 – 348.1 s (mean 243.0 s) |
| Wall clock, first admission → publication | 17:29:50 → 23:52:40 local (6 h 22 m 50 s) |
| Layer types observed | 30 `linear_attention`, 10 `full_attention` |
| Peak per-layer CUDA allocation | 2,383.7 MiB |
| GPU global free after each layer | ~15.1 GiB |

The 6 h 23 m wall clock exceeded the 2.70 h of layer compute because of three
full 71.9 GB source-shard digest passes, per-layer 4.55 GiB slab commits to the
USB `F:` volume, one abort-and-resume cycle, and the closing 704-sequence NLL
pass.

## Incidents during the run (all recoverable, no output lost)

1. **Telemetry abort at generation 20.** The guard stopped the observer with
   `resource telemetry failed closed: TimeoutExpired` — the Windows process-census
   CIM query exceeded its 15 s budget under load. The guard failed closed exactly
   as designed. `CURRENT` stayed at generation 20 with 21 valid manifests, and
   the run resumed from layer 20 rather than layer 0.
2. **Resume pre-screen defect (fixed in `e239096`).** `admit()` refused the
   existing output directory that a legitimate `--resume` requires, reporting a
   valid checkpoint as `frozen_paths_or_source_invalid`. Fixed with regression
   coverage in `tests/test_reap_v2_observer_preflight.py`.
3. **Resource abort at generation 39.** A genuine external burst (67→75 process
   births) dragged free RAM from 17.9 GiB to 6.5 GiB and the runtime compound
   rule `competing_process_fanout_and_ram_decline` aborted the observer. This
   abort was **not** overridden; the run was resumed after RAM recovered.
4. **Admission override, operator-side only.** After incident 3, admission refused
   repeatedly on `foreign_process_fanout_over_60s` (28–30 births vs. a limit of
   11) where the births were Chrome renderer children, `conhost`, `git.exe` and
   similar tiny processes rather than competing model workloads. The operator
   driver waived only that birth proxy and the 60 s RAM-oscillation proxy, and
   **substituted a stricter direct-memory floor** (RAM ≥ 18 GiB, commit ≥ 25 GiB,
   vs. defaults 16/24). No repository safety code, runtime abort rule, GPU floor,
   disk floor or protected-listener rule was changed or weakened. The waiver is
   recorded per attempt in the driver's `admission-retry-report.jsonl`.

## Independent verification

The supervisor's own verdict (`D0_PASS`, `registered_d0_proven: true`) is not the
only evidence. The published artifact was re-checked with versioned repository
code, which does not read the supervisor's state:

```text
$ python -B -c "from chowder_batch.reap_gates import _d0; print(_d0(<observation.json>))"
{'status': 'PASS', 'statistic': 15.451270210051794, 'threshold': 20.0, 'n': 40, 'reason': None}
```

`_d0` requires exactly 40 distinct layer indices and a finite `0 < ppl < 20`; it
returns `FAIL` otherwise. The same pass re-confirmed `n_seqs == 704`,
`n_tokens == 1192632`, the frozen calibration digest, the frozen token-ID digest,
the 704 ordered lengths, the 40 layer identities, and the pinned upstream
revision.

Reproduce the whole verification:

```powershell
python -B -c @"
import json; from pathlib import Path
from chowder_batch.reap_gates import _d0
p = Path(r'C:\llm-work\reap\observe-v2-bf16-atomic-20261024-222950-223f34d3\observation.json')
print(_d0(p))
d = json.loads(p.read_text(encoding='utf-8'))
assert d['n_seqs'] == 704 and d['n_tokens'] == 1192632
assert d['calibration_sha256'] == 'f6a09b694ae43a6d8b44b5180a861856501295eabf3d5297eb9749a008cdd929'
assert d['checkpoint_receipt']['source_identity']['token_ids_sha256'] == \
       '78b8f482f61cfd4b20bba6982a20c8eeb5a757b20069f2520273d4a315335f92'
assert all(r['routed_tokens'] == 1192632 * 8 and r['experts_hit'] == 256 for r in d['layers'])
assert {r['layer'] for r in d['layers']} == set(range(40))
print('coverage + identity cross-checks OK; ppl =', d['calibration_ppl'])
"@
```

## What this result does and does not establish

**Establishes.** A complete 40-layer, 704-sequence, 1,192,632-token observation of
the pinned original BF16 weights and frozen calibration on this host, with exact
top-8 router census, BF16 bit-preserved hidden state, an unbroken 41-generation
hash chain, source identity re-verified against an independent upstream pin
immediately before publication, and a measured next-token perplexity of
15.4513 that passes the frozen `< 20` gate.

**Does not establish** (unchanged release gates, see
`REAP-V2-ORIGINAL-BF16-PROVENANCE-2026-10-24.md`):

- **Independent proof of execution.** A chain of ordinary file hashes is not a
  cryptographic attestation. An actor able to synthesize fully plausible
  256-expert floating-point statistics, matching self-hashes and a fabricated
  perplexity could still produce self-consistent artifacts. Trusted-execution or
  producer attestation remains unresolved.
- **CUDA numerical parity.** The CPU tests prove source/format rejection and
  storage crash behavior, not arithmetic equivalence against a reference
  implementation.
- **Sudden-power-loss durability** of the NTFS atomic-rename + `fsync` protocol
  on the external USB volumes.

The D0 PASS above is a measurement of a real run on this host, not a synthetic
or reconstructed receipt; it is reported with those attestation limits attached.

## Preserved evidence

- `C:\llm-work\reap\observe-v2-bf16-atomic-20261024-222950-223f34d3` — sealed
  `observation.json` plus all 40 per-layer stats (41 files, 0.47 MiB).
- `F:\reap-v2-bf16-atomic-20261024-222950-223f34d3` — dual slabs and 41 generation
  manifests (86 files, 9.1 GiB). Do not delete before the attestation release
  decision; it is the only copy of the committed hidden state.
- `C:\llm-work\reap\supervisor-v2-bf16-atomic-20261024-222950-223f34d3-*` — every
  admission attempt, per-session supervisor state, resource timelines and observer
  stdout/stderr.
- Historical r2 (`observe-v2-full-registered-20261023-r2`,
  `F:\reap-v2-hidden-full-20261023-r2`) — untouched, still not resumable, and
  never counted toward this result.
