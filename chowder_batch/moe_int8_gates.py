"""Measure the frozen-expert placement gates on a real MoE checkpoint.

  python chowder_batch/moe_int8_gates.py --base-model <dir> --out gates.json \
      [--arms int8,bulk] [--steps 3] [--seq-len 2048] [--device 0] [--cpu]

For each arm this records what the prereg in
``docs/quals/MOE_TRANSIENT_GRAD_PLACEMENT_PREREG_2026-10-18.md`` asks for, or an
explicit "not measured" when this runner cannot answer it:

  load_census     live placement census from the loader itself
                  (``moe_int8_experts.load_model_int8_experts`` for ``int8``,
                  ``moe_transient_grad.prepare_split_model`` for ``bulk``)
  memory_vram     peak allocated VRAM per step, against the <= 14.5 GB gate
  host_rss        process peak working set across load and steps
  throughput      measured s/step of a **placement step**: forward + backward on
                  ``inputs_embeds`` covering every expert layer, no LoRA and no
                  optimizer, so it is the cost of the placement, not of a training
                  step. The prereg's throughput gate is a worker run.

``gradient``, ``smoke`` and ``router_balance`` need a dataset and a real
fine-tune; this runner reports them as not measured rather than guessing.

WDDM shared-memory spill is not observable from here either: the record carries
``peak_vram_gb`` and ``reserved_gb`` and leaves the spill verdict to the operator.

Exit codes: 0 record written, 2 refusal (missing directory, unknown arm, no
CUDA without ``--cpu``), 3 an arm failed to load or to place.

``--self-check`` writes its own tiny synthetic checkpoint to a temporary
directory and runs both arms on the CPU, so the runner itself is verifiable
without a GPU or a real model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

EXPERT_PLACEMENT = "cpu-transient-grad-int8"
VRAM_GATE_GB = 14.5
SCHEMA = "chowder.moe-placement-gates.v1"
ARMS = ("int8", "bulk")


def _peak_rss_bytes() -> int | None:
    """This process's peak working set: the host RAM the arm actually touched."""
    try:  # Windows: psapi gives the peak, not just the current
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(ProcessMemoryCounters)
        # The pseudohandle is pointer sized: untyped, ctypes truncates it to -1
        # and the call fails with an overflow instead of returning memory.
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        query = kernel32.K32GetProcessMemoryInfo
        query.argtypes = [ctypes.c_void_p, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD]
        query.restype = wintypes.BOOL
        if query(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return int(counters.PeakWorkingSetSize)
        return None
    except Exception:  # non-Windows or a stripped-down interpreter
        try:
            import resource

            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return int(peak) * (1024 if sys.platform != "darwin" else 1)
        except Exception:
            return None


def _gib(nbytes: int | None) -> float | None:
    return None if nbytes is None else round(nbytes / 2**30, 2)


def _vocab_size(config: Any) -> int:
    text = getattr(config, "get_text_config", lambda: config)()
    return int(getattr(text, "vocab_size", 0) or getattr(config, "vocab_size", 0) or 1000)


def _load_int8(base: Path, device: Any, dtype: Any) -> tuple[Any, dict[str, Any]]:
    from chowder.backends import moe_int8_experts as mie

    model, census = mie.load_model_int8_experts(base, device=device, dtype=dtype)
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    census = {**census, **mie.prepare_int8_placement(model, device_type=device_type)}
    return model, census


def _load_bulk(base: Path, device: Any, dtype: Any) -> tuple[Any, dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM

    from chowder.backends import moe_transient_grad as mtg

    if str(device).startswith("cuda"):
        device_map = mtg.split_device_map(str(base), revision=None, local_files_only=True, device=device.index or 0)
        model = AutoModelForCausalLM.from_pretrained(str(base), dtype=dtype, device_map=device_map)
        census = mtg.prepare_split_model(model, device_type="cuda", bulk=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(str(base), dtype=dtype)
        census = mtg.prepare_split_model(model, device_type="cpu", bulk=True)
    return model, dict(census)


ARM_LOADERS = {"int8": _load_int8, "bulk": _load_bulk}


def _placement_steps(model: Any, *, steps: int, seq_len: int, device: Any) -> dict[str, Any]:
    """Forward + backward through every expert layer, timed per step.

    Deliberately not a training step: no LoRA, no optimizer. What it isolates is
    the placement's own cost -- the per-layer host->device copy (and, for int8,
    the dequantisation) in forward and again in the backward recompute.
    """
    import torch

    vocabulary = _vocab_size(model.config)
    generator = torch.Generator().manual_seed(0)
    tokens = torch.randint(0, vocabulary, (1, seq_len), generator=generator).to(device)
    embedding = model.get_input_embeddings()

    times: list[float] = []
    for _ in range(steps):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        hidden = embedding(tokens).detach().requires_grad_(True)
        with torch.enable_grad():
            loss = model(inputs_embeds=hidden, use_cache=False).logits.float().square().mean()
            loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
        model.zero_grad(set_to_none=True)

    return {
        "steps": steps,
        "seq_len": seq_len,
        "step_seconds": [round(value, 4) for value in times],
        "s_per_step": round(sum(times) / len(times), 4) if times else None,
        "includes_lora": False,
        "includes_optimizer": False,
        "gradient_on_inputs_embeds": True,
    }


def _vram(torch: Any, device: Any) -> dict[str, Any]:
    if device.type != "cuda":
        return {"peak_vram_gb": None, "reserved_gb": None}
    return {
        "peak_vram_gb": round(torch.cuda.max_memory_allocated(device) / 2**30, 3),
        "reserved_gb": round(torch.cuda.max_memory_reserved(device) / 2**30, 3),
    }


def run_arm(arm: str, base: Path, *, device: Any, dtype: Any, steps: int, seq_len: int) -> dict[str, Any]:
    import torch

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    model, census = ARM_LOADERS[arm](base, device, dtype)
    load_seconds = time.perf_counter() - started
    load_rss = _peak_rss_bytes()

    result: dict[str, Any] = {
        "arm": arm,
        "load_seconds": round(load_seconds, 3),
        "load_peak_rss_gb": _gib(load_rss),
        "census": census,
    }
    if "expert_bytes" in census:
        result["census_meta"] = {
            "expert_bytes_gb": _gib(int(census["expert_bytes"])),
            "patched_expert_modules": census.get("patched_expert_modules"),
        }

    result.update(_placement_steps(model, steps=steps, seq_len=seq_len, device=device))
    result.update(_vram(torch, device))
    result["total_peak_rss_gb"] = _gib(_peak_rss_bytes())
    result["gates"] = {
        "load_census": "measured",
        "memory_vram": (
            "measured"
            if result["peak_vram_gb"] is not None
            else "not measured (CPU arm)"
        ),
        "memory_vram_pass": (
            None
            if result["peak_vram_gb"] is None
            else bool(result["peak_vram_gb"] <= VRAM_GATE_GB)
        ),
        "host_rss": "measured",
        "throughput": "measured (placement step, no LoRA/optimizer)",
        "gradient": "not measured by this runner (needs a LoRA run)",
        "smoke": "not measured by this runner (needs a dataset)",
        "router_balance": "not measured by this runner (needs a smoke run)",
        "wddm_spill": "not observable from here; compare peak_vram_gb to the card",
    }
    return result


def run(
    base: Path,
    *,
    arms: tuple[str, ...],
    device: Any,
    dtype: Any,
    steps: int,
    seq_len: int,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "base_model": str(base),
        "device": str(device),
        "dtype": str(dtype) if dtype is not None else "auto (the checkpoint's)",
        "placement": EXPERT_PLACEMENT,
        "vram_gate_gb": VRAM_GATE_GB,
        "arms": {},
        "note": (
            "Placement steps only: forward+backward on inputs_embeds, no LoRA and no "
            "optimizer. The prereg's throughput, gradient, smoke and router-balance "
            "gates still require a worker run."
        ),
    }
    for arm in arms:
        record["arms"][arm] = run_arm(arm, base, device=device, dtype=dtype, steps=steps, seq_len=seq_len)
    return record


def _self_check_workspace() -> Path:
    """A tiny synthetic Qwen3-MoE checkpoint, so the runner is provable on CPU."""
    import torch
    from transformers import AutoModelForCausalLM, Qwen3MoeConfig

    directory = Path(tempfile.mkdtemp(prefix="moe-int8-gates-"))
    config = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=8,
        moe_intermediate_size=4,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_experts=6,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        max_position_embeddings=32,
        head_dim=8,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    AutoModelForCausalLM.from_config(config).save_pretrained(directory, max_shard_size="20KB")
    return directory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-model", default="", help="local checkpoint directory (safetensors shards)")
    parser.add_argument("--out", default="", help="where to write the gate record (JSON)")
    parser.add_argument("--arms", default="int8,bulk", help=f"comma-separated subset of {ARMS}")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dtype", default="", help="auto (the config's), bfloat16, float16, float32")
    parser.add_argument("--cpu", action="store_true", help="run on the CPU (self-check, small models)")
    parser.add_argument("--self-check", action="store_true", help="tiny synthetic checkpoint, CPU, then print the record")
    args = parser.parse_args(argv)

    import torch

    arms = tuple(arm.strip() for arm in args.arms.split(",") if arm.strip())
    unknown = [arm for arm in arms if arm not in ARM_LOADERS]
    if unknown:
        print(f"unknown arm(s) {unknown}; expected a subset of {ARMS}", file=sys.stderr)
        return 2

    if args.self_check:
        os.environ["CHOWDER_NO_GPU"] = "1"
        args.base_model = str(_self_check_workspace())
        args.cpu = True
        arms = ARMS
        args.steps = min(args.steps, 2)
        args.seq_len = min(args.seq_len, 8)
    elif not args.base_model:
        print("--base-model is required (or --self-check)", file=sys.stderr)
        return 2

    base = Path(args.base_model)
    if not base.is_dir():
        print(f"base model directory not found: {base}", file=sys.stderr)
        return 2

    use_cuda = torch.cuda.is_available() and not args.cpu
    if not use_cuda and not (args.cpu or args.self_check):
        print("no CUDA device; pass --cpu only for a synthetic self-check", file=sys.stderr)
        return 2
    device = torch.device("cuda", args.device) if use_cuda else torch.device("cpu")
    dtype = getattr(torch, args.dtype) if args.dtype else None

    try:
        record = run(base, arms=arms, device=device, dtype=dtype, steps=args.steps, seq_len=args.seq_len)
    except Exception as error:  # a failed arm is a refusal, not a record with holes
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 3

    payload = json.dumps(record, indent=2)
    if args.out:
        Path(args.out).write_text(payload, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
