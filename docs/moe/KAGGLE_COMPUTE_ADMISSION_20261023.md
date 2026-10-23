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
