# Kaggle compute admission: actual remote measurements (2026-10-23)

## Scientific status

Kaggle hardware tests are **not** full REAP128_v2 observations or Gen-1
training. The frozen Qwen3.6-35B-A3B BF16 parent has 26 shards totaling
71,903,776,776 bytes. The registered 704-row, 40-layer D0 observation and
its perplexity/saliency remain unmeasured. Never substitute T4 FP16 or
an untested TPU port for a frozen BF16 measurement.

## Private actual Kaggle runs, one authorized account

On the existing, authenticated nikmarco account:

1. CPU private notebook: https://www.kaggle.com/code/nikmarco/chowder-cpu-hardware-preflight
   - Status COMPLETE; downloaded exact receipt at
     C:/Users/nikma/_audit/chowder-kaggle-primary-cpu-receipt-20261023/chowder-hardware-receipt.json
   - SHA256 aaa6cef8c2c62803131fe90747676d96607ff861684e03dbb0a04e4b472e1fca
   - 4 CPU cores, 31.347 GiB total RAM / 30.265 GiB free,
     /kaggle/working 19.502 GiB free, /tmp 1048.801 GiB free.
   - CPU only, no model loads, no full source copy, no CUDA, no D0.

2. TPU-requested private notebook:
   https://www.kaggle.com/code/nikmarco/chowder-tpu-bf16-hardware-20261023
   - Status COMPLETE, but its actual runtime returned **JAX CPU device
     only**, physical 4 cores / 31.347 GiB host RAM. TPU quota remained
     20 hours available. No BF16 TPU multiplication was performed.
   - Pulled remote metadata confirms enable_tpu=true and
     machine_shape=TpuV5E8. This is a **provider accelerator allocation
     mismatch**, not successful native BF16 validation. It needs
     provider/session resolution; do not infer TPU exists from notebook
     metadata alone.
   - Downloaded receipt:
     C:/Users/nikma/_audit/chowder-kaggle-primary-tpu-receipt-20261023/chowder-hardware-receipt.json
   - This reproduces known open Kaggle CLI issue #1197:
     https://github.com/Kaggle/kaggle-cli/issues/1197 .
     The official push API accepts a TPU request but silently schedules CPU.
     The supported operational workaround is to open the notebook in Kaggle's
     interactive editor, explicitly select TPU v5e-8 in Settings / Accelerator,
     then Save & Run All. The available read-only browser connector cannot
     perform that interactive click in this session. Never use the API-generated
     CPU receipt as if it were a TPU BF16 test.

3. Private CPU-only mounted checkpoint I/O probe:
   https://www.kaggle.com/code/nikmarco/chowder-dataset-stream-cpu-20261023
   - Status COMPLETE, mounted existing private
     nikmarco/qwen36-reap160-bf16 (REAP160, not original dense parent).
   - Read exactly 64 MiB sequentially from the first mounted safetensors
     shard in 1.2034 sec: **53.18186 MiB/sec** measured short sample.
     Sample bytes SHA256 6803608593644f37d1190aff5a3a432acc77c4b92bb8c07e5f1c01f53f409a49.
   - Saved receipt:
     C:/Users/nikma/_audit/chowder-kaggle-primary-stream-receipt-20261023/chowder-checkpoint-stream-sample.json
   - Previous DIFFERENT 46 GB REAP160 run recorded effective ~0.69 MB/sec
     for tensor-by-tensor reads and timed out after 12 hours before
     training. This new sequential sample is promising for a bounded
     staging-to-/tmp implementation. It does **not** establish full-shard
     or original 72 GB throughput, model loading speed or frozen D0.
     The measured Kaggle /tmp free space was ~1048.8 GiB on that worker.

## Bounded mounted-shard to /tmp staging verification

Source: chowder_batch/kaggle_checkpoint_stage_probe.py.
CPU-only tests: tests/test_kaggle_checkpoint_stage_probe.py.

The staging verifier copies **at most 128 MiB** from the first real
mounted safetensors shard to a new temporary directory within /tmp,
flushes and fsyncs the bytes, reads and verifies the local copy's SHA256
against the sampled source, and deletes the temporary copy, even on
exception. It emits elapsed read/write/verify times and a small exclusive
JSON receipt. Missing mount, scratch, symlinked or escaped files,
insufficient scratch and over-budget requests fail closed. This is NOT a
full checkpoint copy, weight digest, training step, registered REAP128_v2
observation or gate acceptance.

The repository contains an observed old-model load that read 29.25 GiB
in 707.7 minutes (~0.69 MB/sec effective tensor-by-tensor), versus a
different CPU remote 64 MiB sequential mount sample at 53.18186 MiB/sec.
The bounded copy proof targets the missing mount-to-scratch throughput and
integrity link. A complete 46/72 GB copy needs a **separate run in the same
remote session as the model**, because Kaggle /tmp is ephemeral and not
shared across subsequent notebook sessions. The 20 GB /kaggle/working
saved-output limit does not accommodate the original 71.9 GB checkpoint.

## Reproducible hardware probes

Source: chowder_batch/kaggle_hardware_probe.py.
Builder: chowder_batch/build_kaggle_hardware_probe.py.

Run the builder with --accelerator cpu or --accelerator tpu, --kernel-id
USERNAME/UNIQUE-SLUG and --out-dir NEW-NONEXISTENT-DIRECTORY. It creates
a *private, internet-disabled* kernel-metadata.json and notebook. Push
via the official kaggle kernels push command only when authenticated as
your single authorized account. The builder never sends a request,
downloads a model, consumes quota or modifies an existing directory.
The remote notebook exclusively creates a receipt and rejects claiming
scientific qualification from physical device identity alone.

Kaggle's terms forbid one person holding or operating multiple active
accounts, so no secondary-account GPU quota is pooled here:
https://www.kaggle.com/terms

Kaggle staff confirmed P100 retirement on September 15, 2026,
with regular GPU T4x2 remaining; T4 compute capability 7.5 lacks native
BF16 matrix support:
https://www.kaggle.com/product-announcements/735239

Kaggle/TPU v5e-8 provides a potential native BF16 lane, but our frozen
Qwen PyTorch/CUDA REAP observer still needs a formally validated TPU
port, source staging and end-to-end layer/NLL parity before use as D0.

Run CPU-only local tests:

    python -B -m pytest -q tests/test_kaggle_hardware_probe.py
    python -B -m ruff check chowder_batch/kaggle_hardware_probe.py chowder_batch/build_kaggle_hardware_probe.py tests/test_kaggle_hardware_probe.py
