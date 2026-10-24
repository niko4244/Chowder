# Frozen BF16 REAP128_v2 — full 704×40 execution plan

**Scope:** run one real, original 704-sequence, 40-layer Qwen3.6-35B-A3B
BF16 observation through PR #280's durable two-slab checkpoints and owned
Windows host supervisor, then actually measure frozen D0 perplexity.
Historical r2 is incomplete and MUST NOT be reused or represented as D0.

## Readiness evidence (2026-10-24, measured on this machine)

- **Inputs: ADMITTED** by the CPU-only registered-source metadata check:
  704 frozen calibration rows; 26 publisher-pinned BF16 shards of
  71,903,776,776 bytes; 40 layers, 256 experts, top-8; original tokenizer
  retokenization 1,192,632 tokens. Full 26-shard content SHA-256 will be
  checked by the actual observer, *before* loading the GPU model.
- **Host: REFUSED while user models remain active.** Available physical
  11.85–14.21 GiB vs 16 minimum; free system commit 17.10–22.16 GiB vs
  24 minimum; GPU 0.79–4.09 GiB available vs 12 minimum.
  Protected ports 8766 (llama-server), 8767 (Sharp-Spark llama-server),
  and 11434 (Ollama) are listening. The operator must arrange a dedicated,
  approved service-maintenance window before an actual launch.
- **Disk: sufficient on currently guarded volumes.** F: has about 81.25 GiB
  free (at least 13.10 GiB needed for two 4.55-GiB slabs plus 4 GiB
  reserve). C: has 12.21 GiB free for output/logs. The Windows supervisor
  currently monitors **C:** and **F:**; out/log must be on C, scratch on F.
- **Physical storage caution:** F: and original BF16 G: are WD My Passport
  external USB disks, not the internal SSD. Keep both powered and connected
  without hub disconnect/sleep. I: is a faster Samsung SATA SSD but hosts
  an active 61-GiB pagefile and is NOT guarded as a scratch drive by this
  particular launcher. Do not redirect scratch to I without separate
  engineering/change review.
- Archived r2 took 3,803.9 seconds for layers 0–10, about 346 seconds per
  completed layer. The linear 40-layer calculation is ~3.84 hours **only
  for layer computation**, excluding all full-shard SHA hashing, two-slab
  checkpoints, NLL, storage variance, and system contention. It is NOT
  a measured complete-run duration or a promise.

## 1. New unique paths and zero-side-effect diagnosis

Open a **persistent local PowerShell terminal**, and keep it open and the
computer awake for any later `--launch`. The launcher owns a Windows Job
Object; if its parent supervisor exits unexpectedly (including a terminal
session being killed), the observer is terminated for safety. The latest
atomic committed layer can subsequently resume.

Run from the checked-out PR #280 feature worktree:

```powershell
Set-Location 'C:\Users\nikma\wt\chowder-bf16-atomic-resume-20261024'
$runId = (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + (New-Guid).Guid.Substring(0,8)
$model = 'G:\llm-cold\bf16\Qwen3.6-35B-A3B'
$calibration = 'C:\llm-work\reap\calib-v2-ornith-bare-20261023\calibration.jsonl'
$out = "C:\llm-work\reap\observe-v2-bf16-atomic-$runId"
$scratch = "F:\reap-v2-bf16-atomic-$runId"
$logs = "C:\llm-work\reap\supervisor-v2-bf16-atomic-$runId"
$runArgs = @(
  '--model-dir', $model,
  '--weights-dir', $model,
  '--calibration', $calibration,
  '--out', $out,
  '--scratch', $scratch,
  '--log-dir', $logs
)
python -B -m chowder_batch.reap_observe_supervised --doctor @runArgs
```

`--doctor` reads frozen source and live resources, outputs JSON and does
**not** create directories, load weights onto GPU, launch a process, or
touch model services. If it reports `NOT_READY_DO_NOT_LAUNCH` (exit 2),
do not launch. Even a green one-shot `--doctor` is NOT the 60-second
multi-sample host admission, which the actual supervisor enforces again.
Do not reuse the names from r2 or any previous candidate run.

## 2. Operator-approved service maintenance (not automated here)

Arrange a dedicated quiescent window with the owners of the active local
model services; preserve their exact current restart/configuration strategy.
The experiment needs the selected GPU free for the entire observation,
including the final NLL projection, and cannot coexist with the three
protected listeners listed above under this protocol. The supervisor will
**never** stop, kill, disable, configure or restart them automatically.
Do not stop the Chat On Steroids communication bridge or unrelated
desktop infrastructure. A read-only way to identify current port owners:

```powershell
Get-NetTCPConnection -State Listen |
  Where-Object { $_.LocalPort -in @(8766,8767,11434) } |
  Select-Object LocalAddress,LocalPort,OwningProcess
nvidia-smi --query-gpu=index,uuid,name,memory.free,utilization.gpu --format=csv
```

After approved owner-controlled quiescence, repeat `--doctor`. A viable
minimum is RAM available >=16 GiB, free system commit >=24 GiB, selected
GPU >=12 GiB free, no protected model-service listeners, C: >=2 GiB
free and F: >=14 GiB free, with stable 60-second process activity. Aim
for margin above minimum rather than barely meeting the limits. **Do not
enlarge or move the active pagefile solely to bypass these gates:** that
cannot release VRAM or physical RAM and increases disk pressure.

## 3. Full launch — one real attempt

Only after the owner has arranged the maintenance window and doctor is
clear, in the **same still-open terminal** with `$runArgs` and `$runId`
unchanged:

```powershell
python -u -B -m chowder_batch.reap_observe_supervised --launch @runArgs
```

This imposes THREE host snapshots about 30 seconds apart, plus a last-moment
recheck; failing ANY gate refuses launch. It then starts only its owned
BF16 observer behind a Windows Job Object, and keeps checking host
memory/commit, selected GPU, protected services, disks and competing process
fanout about every 30 seconds. Its hard safety limit is **8 hours per
observer invocation**. It does not guarantee the full run will complete
within that window. On resource abort or unexpected process crash,
only the owned observer is stopped; unrelated services are untouched.

The observer streams 26 original source-shard digests from G: and checks
against fixed publisher hashes, materializes embedding generation zero,
then atomically commits a full BF16 slab + stats after each layer. The
final registered D0 requires **40 fully committed layers, frozen 704 token
sequences and actual measured perplexity** (PASS only if PPL <20).

## 4. Live status and controlled stop

In a separate PowerShell window, inspect without modifying the observer:

```powershell
Get-Content (Join-Path $logs 'supervisor-state.json')
Get-Content (Join-Path $logs 'resources.jsonl') -Tail 2
Get-Content (Join-Path $logs 'observer.stdout.log') -Tail 5
Get-Content (Join-Path $logs 'observer.stderr.log') -Tail 5
Get-Content (Join-Path $scratch 'CURRENT') -ErrorAction SilentlyContinue
```

Those `$logs` and `$scratch` variables must be copied to a new terminal
if that terminal does not already have them. A deliberate **owned observer
only** stop can be requested by creating `STOP` in the *current* log
directory; the supervisor checks it on the next sample:

```powershell
New-Item -ItemType File -Path (Join-Path $logs 'STOP') -ErrorAction Stop
```

Do not edit `CURRENT`, generation manifests, layer stats, output JSON,
or either state slab. Do not assume that a partial layer can resume:
only completed generations committed before the stop survive.

## 5. Resume from NEW atomic checkpoint (never legacy r2)

If the supervisor exits without complete D0 and `CURRENT` exists, let
the owned worker fully terminate and verify the host is again stable.
Use exactly the SAME `$model`, `$calibration`, `$out` and `$scratch`,
but an entirely NEW supervisor log directory:

```powershell
$resumeLogs = "C:\llm-work\reap\supervisor-v2-bf16-resume-$runId-" +
              (New-Guid).Guid.Substring(0,8)
$resumeArgs = @(
  '--model-dir', $model,
  '--weights-dir', $model,
  '--calibration', $calibration,
  '--out', $out,
  '--scratch', $scratch,
  '--log-dir', $resumeLogs,
  '--resume'
)
python -B -m chowder_batch.reap_observe_supervised --doctor @resumeArgs
python -u -B -m chowder_batch.reap_observe_supervised --launch @resumeArgs
```

The receiver verifies the original upstream weight bytes, precise frozen
token IDs, committed slab, metadata, stats, hash chain and process/runtime
policy before continuing. If no `CURRENT` exists, the previous attempt
has no resumable committed generation; preserve its log for diagnosis and
start fresh with new unique directories instead.

## What is still NOT claimed

No complete 704×40 D0 run has yet passed or been measured. No current
inference that the model meets PPL <20 is authorized. CPU-only crash,
manifest and supervisor tests do NOT establish numerical CUDA parity or
independently authenticated evidence of genuine model execution.
The original external USB disks also do not have verified sudden-power-loss
directory durability. These remain validation/release questions, but the
resource-gated complete-run path is the immediate engineering priority.
