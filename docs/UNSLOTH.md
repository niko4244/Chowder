# Unsloth isolated runtime

Chowder treats Unsloth as an optional training implementation, not as part of the controller's dependency stack. The controller keeps its tested Transformers/PEFT/TRL versions; Unsloth lives under the workspace's `.chowder/envs/unsloth/` directory and is invoked only through an isolated Python process.

This boundary is intentional. As of 2026-09-05, Chowder's tested training extra and current Unsloth releases have materially different dependency envelopes. Do not weaken Chowder's pins to make the two environments share packages.

## Current supported setup flow

The default version is pinned to the current PyPI release verified on 2026-09-05:

```text
unsloth==2026.9.2
Python 3.13
```

Both are CLI-overridable so a future compatibility investigation can be explicit and reproducible rather than silently changing the default.

Install Astral `uv` first, then from the Chowder workspace run:

```bash
chowder setup unsloth
```

Chowder creates or reuses:

```text
.chowder/
└── envs/
    └── unsloth/
```

For a new environment it executes the upstream-recommended shape:

```text
uv venv <workspace>/.chowder/envs/unsloth --python 3.13
uv pip install --python <isolated-python> unsloth==2026.9.2 --torch-backend=auto
```

Using `--torch-backend=auto` matters for RTX 50-series / Blackwell because current `uv` can choose the PyTorch index that matches the detected NVIDIA driver. Chowder performs an NVIDIA preflight first and refuses setup if `nvidia-smi` exposes no GPU; otherwise `uv` is allowed to fall back to CPU and a CUDA training environment could be falsely reported as healthy.

Existing environments are reused only when their Python major/minor matches the requested version. Chowder does not silently rebuild or mutate an environment across a requested Python-version change.

## Doctor

Run:

```bash
chowder doctor unsloth
```

The controller launches the isolated interpreter and receives a JSON result file. It never imports Unsloth, Torch, Transformers, PEFT, TRL, bitsandbytes, xFormers, or Triton into the Chowder controller process.

The probe reports at least:

- Python version compatibility;
- Unsloth and `unsloth_zoo` import/version;
- Torch import/version;
- CUDA runtime visibility;
- actual NVIDIA accelerator names and CUDA compute capabilities;
- TRL and PEFT import/version;
- bitsandbytes import/version;
- Triton / `triton-windows` import/version;
- xFormers status when it is present in the resolved environment;
- an actual small NF4 `bitsandbytes.nn.Linear4bit` CUDA forward pass.

The last check is deliberately stronger than testing whether bitsandbytes imports. A broken CUDA/bitsandbytes combination must fail before Chowder spends model-loading or training time.

A doctor failure is reported with the original import/runtime detail. Chowder does not silently fall back to another acceleration path and call the installation successful.

## Environment evidence

Successful or capability-failed setup writes:

```text
.chowder/envs/unsloth/chowder-unsloth-manifest.json
```

The manifest records:

- requested Python and Unsloth versions;
- `uv` version;
- `torch_backend=auto` selection request;
- controller platform/Python;
- pre-install hardware inventory;
- the complete `uv pip freeze` result;
- exact critical package versions observed by the isolated probe;
- CUDA runtime and accelerator evidence;
- every doctor check and its pass/fail detail.

`uv pip check` is also run after installation. A dependency-graph conflict is inserted as a required failed capability check and persisted in the manifest.

## Windows / RTX 5060 Ti

Windows is a first-class target. The isolated interpreter path is resolved as:

```text
.chowder\envs\unsloth\Scripts\python.exe
```

On Linux/WSL it is:

```text
.chowder/envs/unsloth/bin/python
```

The setup flow intentionally follows current upstream guidance for Windows and RTX 50-series: Python 3.13, `uv`, and `--torch-backend=auto`.

## What this slice does not claim

A green `chowder doctor unsloth` proves that the isolated package/CUDA/4-bit runtime is coherent enough for the next training slice. It does **not** prove that Chowder's Unsloth executor, checkpoint/resume, cancellation, PEFT adapter portability, independent evaluation, or quality parity work yet.

Those claims require the separate real-CUDA acceptance run described in the Unsloth integration plan. In particular, Chowder must not commission Qwen3.8 merely because environment setup succeeds.
