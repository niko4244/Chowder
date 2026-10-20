"""Kaggle script kernel: can a Kaggle session host ``cpu-transient-grad-int8``?

The int8 placement ran out of *host* memory on the Windows box (28.6 GB after
load, 44.5 GiB private commit during the second backward, pagefile thrashing at
0.4 GiB free), so Kaggle was proposed as the place to run it. This kernel asks
the only question that decides it, before a 43 GB checkpoint is uploaded:

1. What does the session actually give -- total and available host RAM, scratch
   disk, GPU model and compute capability, and whether the card supports bf16
   (the spec is bf16; T4/P100 do not).
2. How does the int8 layer function's host footprint behave on Linux, where the
   Windows growth cannot be WDDM paging? It walks reap160's real expert shape
   (40 layers x 160 experts x 2048 hidden x 512 intermediate) through the same
   autograd function the worker installs, and records host RSS and device peak
   per layer.

**It measures no gate.** The pre-registration's rows stay ``not measured``
unless a real reap160 run finishes and writes ``placement-witnesses.json``; this
kernel's JSON only says whether such a run could fit here, and it says so with
numbers rather than a guess.

The layer count is chosen from the RAM the session actually has, and the JSON
records how many layers ran, so a truncated probe cannot be read as the full
20.1 GB int8 grid.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

# reap160's own dims: config.json's text_config, and the shard header's expert
# shapes ([160, 1024, 2048] gate_up, [160, 2048, 512] down).
HIDDEN = 2048
INTERMEDIATE = 512
EXPERTS = 160
TOP_K = 8
LAYERS = 40
SEQ = 512
GIB = 1 << 30
OUTPUT = Path("/kaggle/working/int8_capacity.json")
INPUT = Path("/kaggle/input")
CODE_FILE = Path("src/chowder/backends/moe_int8_experts.py")


def log(message: str) -> None:
    print(f"[probe] {message}", flush=True)


def _meminfo() -> dict[str, int]:
    """``/proc/meminfo`` in bytes, the figure the kernel's own OOM killer reads."""
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            parts = rest.split()
            if parts:
                values[key] = int(parts[0]) * 1024
    except OSError as error:  # not Linux: say so rather than inventing numbers
        values["error"] = len(str(error))
    return values


def _self_status() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            key, _, rest = line.partition(":")
            if key in ("VmRSS", "VmPeak", "VmHWM"):
                parts = rest.split()
                if parts:
                    values[key] = int(parts[0]) * 1024
    except OSError:
        pass
    return values


def _disk(path: Path) -> dict[str, float]:
    try:
        usage = shutil.disk_usage(path)
        return {
            "path": str(path),
            "total_gb": usage.total / GIB,
            "free_gb": usage.free / GIB,
        }
    except OSError as error:
        return {"path": str(path), "error": f"{type(error).__name__}: {error}"}


def find_dataset() -> Path | None:
    """The mounted dataset that holds the code, found by content rather than by name.

    The mount layout is not stable across images: an older one mounts
    ``/kaggle/input/<slug>``, the current one mounts
    ``/kaggle/input/datasets/<owner>/<slug>``. Searching for the module itself
    finds the root in either layout, and the failure message lists what was
    actually there instead of a path that was only assumed.
    """
    if not INPUT.is_dir():
        return None
    for hit in sorted(INPUT.rglob(CODE_FILE.name)):
        # <root>/src/chowder/backends/moe_int8_experts.py
        root = hit.parents[3]
        if (root / CODE_FILE).is_file():
            return root
    return None


def environment() -> dict:
    """Everything a decision about this placement needs, measured not assumed."""
    meminfo = _meminfo()
    env: dict = {
        "meminfo_bytes": meminfo,
        "host_ram_total_gb": meminfo.get("MemTotal", 0) / GIB,
        "host_ram_available_gb": meminfo.get("MemAvailable", 0) / GIB,
        "disk_working": _disk(Path("/kaggle/working")),
        "disk_temp": _disk(Path("/kaggle/temp")),
        "disk_root": _disk(Path("/tmp")),
        "python": sys.version.split()[0],
        "inputs": sorted(str(p.relative_to(INPUT)) for p in INPUT.rglob("*") if p.is_dir())[:20]
        if INPUT.is_dir()
        else [],
    }
    try:
        env["nvidia_smi"] = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,compute_cap,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        env["nvidia_smi"] = f"{type(error).__name__}: {error}"

    dataset = find_dataset()
    if dataset is None:
        raise SystemExit(
            f"no mounted dataset under {INPUT} carries {CODE_FILE}; "
            f"mounted: {sorted(str(p.relative_to(INPUT)) for p in INPUT.rglob('*') if p.is_dir()) if INPUT.is_dir() else []}"
        )
    env["dataset"] = str(dataset)
    # The dataset keeps the repo's own layout, so ``chowder`` lives under ``src/``
    # and the batch scripts sit flat beside it.
    for entry in (dataset / "src", dataset / "chowder_batch", dataset):
        if entry.is_dir():
            sys.path.insert(0, str(entry))
    import torch

    env["torch"] = torch.__version__
    env["cuda"] = getattr(torch.version, "cuda", None)
    env["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        env["device_name"] = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        env["compute_capability"] = f"{major}.{minor}"
        env["device_total_gb"] = torch.cuda.get_device_properties(0).total_memory / GIB
        # The spec is bf16. A card without it is not the card the gate was written for.
        env["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
    else:
        env["bf16_supported"] = False
    return env


def per_layer_bytes() -> int:
    """int8 expert bytes for one layer plus its fp32 scales, as the loader builds them.

    The checkpoint stores ``gate_up_proj`` as ``[E, 2I, H]`` and ``down_proj`` as
    ``[E, H, I]`` -- read from reap160's own safetensors header, not assumed --
    and ``_empty_int8_experts`` gives each a fp32 scale per row, shaped
    ``(*q.shape[:-1], 1)``. At reap160's dims the 40-layer grid is 18.8 GiB of
    int8, which is the figure the CPU census recorded."""
    gate_up = EXPERTS * 2 * INTERMEDIATE * HIDDEN
    down = EXPERTS * HIDDEN * INTERMEDIATE
    scales = (EXPERTS * 2 * INTERMEDIATE + EXPERTS * HIDDEN) * 4
    return gate_up + down + scales


def plan(available_gb: float) -> dict:
    """How many layers' int8 grid fits in the RAM this session has, leaving 4 GiB
    for the skeleton the encoder also needs. A probe that cannot hold the whole
    grid says so instead of OOM-killing the session."""
    size = per_layer_bytes()
    budget = max(0.0, available_gb - 4.0) * GIB
    return {
        "per_layer_gb": size / GIB,
        "full_grid_gb": size * LAYERS / GIB,
        "layers_affordable": max(0, min(LAYERS, int(budget // size))),
        "budget_gb": budget / GIB,
    }


def build_grid(layers: int) -> list[dict]:
    """Allocate and fill ``layers`` int8 layers, and keep every one of them.

    The real loader holds the whole grid in host RAM for the length of the run,
    so a probe that builds one layer and frees it measures a run that does not
    exist -- its RSS would be the per-pass working set and nothing else. Every
    byte is filled rather than reserved, because lazily mapped pages would not
    count toward RSS and would understate the footprint this is here to measure.
    """
    import torch

    grid: list[dict] = []
    for index in range(layers):
        grid.append(
            {
                "gate_up_q": torch.empty(
                    (EXPERTS, 2 * INTERMEDIATE, HIDDEN), dtype=torch.int8
                ).fill_(1),
                "down_q": torch.empty(
                    (EXPERTS, HIDDEN, INTERMEDIATE), dtype=torch.int8
                ).fill_(1),
                # Per-row scales, shaped exactly as ``_empty_int8_experts`` allocates them.
                "gate_up_scale": torch.ones(
                    (EXPERTS, 2 * INTERMEDIATE, 1), dtype=torch.float32
                ),
                "down_scale": torch.ones((EXPERTS, HIDDEN, 1), dtype=torch.float32),
            }
        )
        rss = _self_status().get("VmRSS", 0) / GIB
        log(f"grid {index + 1}/{layers} resident, rss {rss:.2f} GB")
    return grid


def run_layer_probe(device: str, dtype_name: str, layers: int) -> dict:
    """Walk ``layers`` int8 layers through the worker's own autograd function.

    Forward then backward per layer, which is the arrangement that retains the
    least: if host RSS still climbs here, the growth is not the graph holding
    dequantised copies and the real run has no chance on this box.
    """
    import torch
    from chowder.backends.moe_int8_experts import _int8_layer_function

    dtype = getattr(torch, dtype_name)
    function = _int8_layer_function()
    act_fn = torch.nn.functional.silu
    samples: list[dict] = []
    peak_device = 0.0

    def snapshot(layer: int, phase: str) -> dict:
        status = _self_status()
        return {
            "layer": layer,
            "phase": phase,
            "rss_gb": status.get("VmRSS", 0) / GIB,
            "hwm_gb": status.get("VmHWM", 0) / GIB,
            "device_allocated_gb": torch.cuda.memory_allocated() / GIB,
            "device_reserved_gb": torch.cuda.memory_reserved() / GIB,
        }

    torch.cuda.reset_peak_memory_stats()
    grid = build_grid(layers)
    load_rss = _self_status().get("VmRSS", 0) / GIB
    log(f"grid complete: {len(grid)} layers resident, rss {load_rss:.2f} GB")

    for index in range(layers):
        layer = grid[index]
        hidden = torch.randn(
            SEQ, HIDDEN, device=device, dtype=dtype, requires_grad=True
        )
        top_k_index = torch.randint(0, EXPERTS, (SEQ, TOP_K), device=device)
        top_k_weights = torch.softmax(
            torch.randn(SEQ, TOP_K, device=device, dtype=torch.float32), dim=-1
        )

        started = time.perf_counter()
        out = function.apply(
            hidden,
            top_k_index,
            top_k_weights,
            layer["gate_up_q"],
            layer["gate_up_scale"],
            layer["down_q"],
            layer["down_scale"],
            act_fn,
            EXPERTS,
        )
        forward_seconds = time.perf_counter() - started
        after_forward = snapshot(index, "forward")
        started = time.perf_counter()
        out.backward(torch.ones_like(out))
        backward_seconds = time.perf_counter() - started
        after_backward = snapshot(index, "backward")
        peak_device = max(peak_device, torch.cuda.max_memory_allocated() / GIB)

        finite = bool(torch.isfinite(hidden.grad).all().item())
        samples.append(
            {
                "forward": after_forward,
                "backward": after_backward,
                "forward_seconds": forward_seconds,
                "backward_seconds": backward_seconds,
                "grad_finite": finite,
                "grad_nonzero": bool((hidden.grad != 0).any().item()),
            }
        )
        log(
            f"layer {index + 1}/{layers} rss {after_backward['rss_gb']:.2f} GB, "
            f"device peak {peak_device:.2f} GB, fwd {forward_seconds:.1f}s "
            f"bwd {backward_seconds:.1f}s, grad finite={finite}"
        )
        # The grid stays resident -- that is the whole point -- so only the
        # per-pass tensors are dropped here.
        del hidden, top_k_index, top_k_weights, out
        torch.cuda.empty_cache()

    del function, grid
    return {
        "layers_probed": layers,
        "grid_resident_gb": layers * per_layer_bytes() / GIB,
        "load_rss_gb": load_rss,
        "peak_device_gb": peak_device,
        "peak_host_rss_gb": max(s["backward"]["rss_gb"] for s in samples)
        if samples
        else 0.0,
        "peak_host_hwm_gb": max(s["backward"]["hwm_gb"] for s in samples)
        if samples
        else 0.0,
        "mean_forward_seconds": sum(s["forward_seconds"] for s in samples) / len(samples)
        if samples
        else 0.0,
        "mean_backward_seconds": sum(s["backward_seconds"] for s in samples) / len(samples)
        if samples
        else 0.0,
        "gradients_finite": all(s["grad_finite"] for s in samples),
        "gradients_nonzero": all(s["grad_nonzero"] for s in samples),
        "samples": samples,
    }


def main() -> int:
    env = environment()
    log(
        f"host RAM {env['host_ram_total_gb']:.1f} GB total, "
        f"{env['host_ram_available_gb']:.1f} GB available; gpu {env.get('device_name')}; "
        f"bf16 {env.get('bf16_supported')}; cuda {env['cuda_available']}"
    )
    log(f"working {env['disk_working']}; temp {env['disk_temp']}")

    probe_plan = plan(env["host_ram_available_gb"])
    result: dict = {
        "environment": env,
        "plan": probe_plan,
        "probe": None,
        "verdict": None,
    }

    if not env["cuda_available"]:
        result["verdict"] = (
            "no GPU in this session: the placement refuses to run without CUDA"
        )
    elif probe_plan["layers_affordable"] == 0:
        result["verdict"] = (
            f"host RAM {env['host_ram_available_gb']:.1f} GB available cannot hold even one "
            f"int8 layer ({probe_plan['per_layer_gb']:.2f} GB each) beside a 4 GiB skeleton"
        )
    else:
        dtype_name = "bfloat16" if env.get("bf16_supported") else "float16"
        if dtype_name != "bfloat16":
            log(
                "bf16 unsupported on this card: probing in fp16, which the spec does not use"
            )
        result["probe_dtype"] = dtype_name
        result["probe"] = run_layer_probe(
            "cuda", dtype_name, probe_plan["layers_affordable"]
        )
        peak = result["probe"]["peak_host_rss_gb"]
        result["verdict"] = (
            f"{result['probe']['layers_probed']}/{LAYERS} layers held resident "
            f"({result['probe']['grid_resident_gb']:.2f} GB of int8) in {dtype_name}: "
            f"RSS {result['probe']['load_rss_gb']:.2f} GB after load, {peak:.2f} GB peak "
            f"of {env['host_ram_total_gb']:.1f} GB; "
            f"device peak {result['probe']['peak_device_gb']:.2f} GB; "
            f"{result['probe']['mean_forward_seconds']:.1f}s fwd + "
            f"{result['probe']['mean_backward_seconds']:.1f}s bwd per layer pass; "
            f"gradients finite={result['probe']['gradients_finite']}"
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(f"verdict: {result['verdict']}")
    log(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
