# Frozen REAP128_v2 host-RAM guard — operator instructions

The October 24 r2 abort saved eleven full layer-statistics files but no
restorable state. It was an INCOMPLETE observation with D0 NOT_MEASURED.
**Do not resume or overwrite r2.** Resume requires a fresh atomic
two-buffer observation begun on PR #280 or later.

## What changed

The earlier machine-local supervisor at
`C:\llm-work\reap\local-v2-retry-prep-20261023` is intentionally **not**
modified. Its single 11 GiB physical / 20 GiB commit admission check and
two consecutive samples below 2.5 GiB free RAM were insufficient to catch
the abrupt memory collapse before layer 11 lost progress.

The versioned `chowder_batch/reap_host_guard.py` provides a read-only,
CPU-only sampling and decision policy. The opt-in launcher
`chowder_batch/reap_observe_supervised.py` controls **only the subprocess it
creates**. It does not terminate, disable, configure or restore Sharp-Spark,
Ollama, Hermes, Claude, other agents, or unrelated processes.

- Admission requires three ordered snapshots spanning about 60 seconds with
  free physical RAM >=16 GiB, available system commit >=24 GiB, less than
  1 GiB RAM decline in 60 seconds, fewer than 12 competing process births,
  at least 12 GiB free on the selected first GPU, usable GPU counters, no
  8767/11434 service-listener conflict, and sufficient
  C:/F: log and BF16 scratch space. These headrooms are **conservative
  operational choices**, not measured hardware guarantees.
- Once launched, sample about every 30 seconds. Abort the owned observer when
  free RAM falls below 5 GiB at one sample, or below 6 GiB combined with a
  >=2.5 GiB 60-second decline; also abort for a 12-process burst plus
  concurrent RAM pressure/decline. Existing low-commit, disk, GPU memory,
  temperature and unexpected service reactivation remain fail-closed.
- Keep per-process PID, creation time, parent PID, private/working-set bytes
  for the largest recent competing process births (not command arguments or
  secrets). Process identity uses PID **and start time**, not PID alone.
  Inaccessible process metadata is counted; substantial census failure is
  an admission/continuation failure.
- Resource sensors that raise during monitoring abort only the owned observer.
  Supervisor final state cannot mark D0 PASS from a worker exit code alone:
  the full 704x40 sealed observer receipt and measured perplexity are checked.
  PPL >=20 remains a measured D0 **FAIL**, not a missing or passing gate.

## Evidence: original 126-sample offline replay

No live host process/GPU queries are made in replay mode:

```powershell
python -B -m chowder_batch.reap_observe_supervised --replay tests/data/reap_r2_host_126.jsonl
```

Expected first guard abort: **sample 123**, timestamp
`2026-10-24T05:54:53.6901049Z`, RAM free **4.29 GiB**, 60-second RAM decline
**2.95 GiB**. The legacy two-sample sub-2.5-GiB free-RAM shutdown began
around **sample 126**, approximately **98 seconds later**. The 126 archived
records did not contain individual process histories, so the process-fanout
alarm is tested with synthetic PID/start-time snapshots; the historical
causality of the 100+ competing processes remains unproven.

## Launch (operator action, intentionally never automatic)

**Do not launch while restored model services are running.** Arrange
service availability/maintenance separately and keep their own original
restoration mechanisms and status checks. The supervised launcher never
performs service maintenance. Use correct current frozen paths, at least
9.10 GiB for two slabs and another 4 GiB scratch reserve on F:, plus
ample physical RAM. Never use the old r2 scratch/output.

```powershell
python -B -m chowder_batch.reap_observe_supervised --launch `
  --model-dir <FROZEN_MODEL_METADATA_DIR> `
  --weights-dir <ORIGINAL_BF16_WEIGHTS_DIR> `
  --calibration <EXACT_FROZEN_V2_CALIBRATION_JSONL> `
  --out <UNUSED_OUTPUT_DIR> `
  --scratch <UNUSED_F_VOLUME_CHECKPOINT_DIR> `
  --log-dir <UNUSED_SUPERVISOR_LOG_DIR>
```

After an interruption that has committed generation zero or later, use
the **same** `--model-dir`, `--weights-dir`, `--calibration`, `--out`
and `--scratch`, a **new** `--log-dir`, plus `--resume --launch`.
Confirm the previous observer process has fully exited. If no generation
zero `CURRENT` exists, preserve the failed diagnostics and start fresh
under new unique paths instead of attempting recovery.

The launcher enforces an eight-hour elapsed safety limit. Its fresh
preflight rehashes approximately 72 GiB of source BF16 shards and the
observer rechecks those payloads again on resume and completion; this costs
read I/O and time. Plan admission with ample headroom.

## Release gates still blocked

1. A dedicated realistic small **CUDA numerical equivalence** check remains
   outstanding; CPU bitwise storage and crash tests do not certify arithmetic.
2. Windows NTFS atomic rename plus file `fsync` handles common process kills,
   but independent sudden-power-loss metadata durability remains unverified.
3. A monitored successful original 704-sequence, 40-layer, frozen-source
   observation has not run. Gate D0 is **NOT_MEASURED**.
4. The previous supervisor's Ollama configuration restoration reported an
   error in its archived final receipt; subsequent manual restoration was
   reported separately. Verify current restoration through the system's
   original owner procedure, and do not run archived restoration helpers
   automatically.
