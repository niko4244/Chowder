"""Kaggle kernel script: 2xT4 compute-worker smoke test for Chowder.

Pushed by `chowder kaggle smoke` (`chowder.kaggle_dispatch`); also runs
unchanged when pasted into a notebook. It proves, and records, three
things in order -- each stage's outcome is written even if a later stage
fails, so a failed run still returns evidence instead of nothing:

1. hardware  -- `nvidia-smi` + torch see two T4s, no bf16 (cc 7.5).
2. load      -- a 4-bit Qwen3-30B-A3B split across *both* GPUs with an
                explicit per-device cap (two 16 GiB pools, never treated
                as one 32 GiB device; see `hardware_bridge`). This is a
                worker-capability probe, deliberately outside the frozen
                parent-evaluation protocol that `kaggle_launcher` guards.
3. generate  -- a few fixed prompts, greedy, written as teacher-style
                JSONL (`generations.jsonl`) with tokens/s per row.

Outputs in /kaggle/working: smoke_result.json, generations.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

DEFAULT_MODEL = "unsloth/Qwen3-30B-A3B-bnb-4bit"
PROMPTS = [
    "A refrigerator's evaporator fan runs but the fresh-food section is warm. List the first three checks a technician should make, in order.",
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "What is 17 * 23? Answer with the number only.",
]


def _sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout.strip()
    except Exception as exc:  # noqa: BLE001 -- evidence capture, never fatal
        return f"<{type(exc).__name__}: {exc}>"


def stage_deps() -> dict:
    """bitsandbytes is not in every Kaggle image; install only what is missing."""
    import importlib.util

    missing = [m for m in ("bitsandbytes", "accelerate") if importlib.util.find_spec(m) is None]
    if missing:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)
    import importlib.metadata as md

    return {"installed": missing, **{n: md.version(n) for n in ("transformers", "bitsandbytes", "accelerate")}}


def stage_hardware() -> dict:
    import torch

    gpus = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        gpus.append({
            "index": i, "name": props.name, "capability": f"{props.major}.{props.minor}",
            "total_gib": round(total / 2**30, 2), "free_gib": round(free / 2**30, 2),
        })
    return {
        "nvidia_smi": _sh(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu_count": len(gpus), "gpus": gpus,
        "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        "disk_free_gib": round(os.statvfs("/kaggle/working" if Path("/kaggle/working").exists() else ".").f_bavail
                               * os.statvfs(".").f_frsize / 2**30, 1),
    }


def stage_load(model_id: str, per_gpu_gib: int, gpu_count: int):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        max_memory={i: f"{per_gpu_gib}GiB" for i in range(gpu_count)} | {"cpu": "24GiB"},
        torch_dtype=torch.float16,  # T4 has no bf16; `torch_dtype` for the image's transformers 4.x
        low_cpu_mem_usage=True,
    )
    placement: dict[str, int] = {}
    for dev in getattr(model, "hf_device_map", {}).values():
        placement[str(dev)] = placement.get(str(dev), 0) + 1
    info = {
        "model_id": model_id, "load_seconds": round(time.time() - t0, 1),
        "device_map_counts": placement,
        "offloaded_to_cpu": any(k in ("cpu", "disk") for k in placement),
        "peak_alloc_gib": {i: round(torch.cuda.max_memory_allocated(i) / 2**30, 2) for i in range(gpu_count)},
    }
    return tok, model, info


def stage_generate(tok, model, max_new_tokens: int, out_path: Path) -> dict:
    import torch

    rows = []
    with out_path.open("w", encoding="utf-8") as fh:
        for i, prompt in enumerate(PROMPTS):
            text = tok.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False,
            )
            inputs = tok(text, return_tensors="pt").to(model.device)
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            dt = time.time() - t0
            new = out[0, inputs["input_ids"].shape[1]:]
            row = {
                "id": f"smoke-{i}", "prompt": prompt,
                "response": tok.decode(new, skip_special_tokens=True),
                "new_tokens": int(new.numel()), "seconds": round(dt, 2),
                "tokens_per_second": round(new.numel() / dt, 2) if dt else None,
            }
            fh.write(json.dumps(row) + "\n")
            rows.append(row)
    return {
        "rows": len(rows),
        "mean_tokens_per_second": round(sum(r["tokens_per_second"] or 0 for r in rows) / len(rows), 2),
        "arithmetic_check_17x23": "391" in rows[2]["response"],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=os.environ.get("CHOWDER_SMOKE_MODEL", DEFAULT_MODEL))
    p.add_argument("--per-gpu-gib", type=int, default=14)
    p.add_argument("--max-new-tokens", type=int, default=160)
    p.add_argument("--output-dir", default="/kaggle/working" if Path("/kaggle/working").exists() else "smoke_out")
    p.add_argument("--hardware-only", action="store_true")
    args = p.parse_args(argv)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result: dict = {"started_at": time.time(), "python": sys.version.split()[0], "stages": {}}

    def save() -> None:
        (out / "smoke_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    def run(name: str, fn):
        try:
            value = fn()
            result["stages"][name] = {"ok": True, **(value if isinstance(value, dict) else {})}
            return value
        except Exception as exc:  # noqa: BLE001 -- record and stop, never hide
            result["stages"][name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                                      "traceback": traceback.format_exc()[-4000:]}
            return None
        finally:
            save()

    hw = run("hardware", stage_hardware)
    if hw and not args.hardware_only:
        run("deps", stage_deps)
    if not hw or hw["gpu_count"] < 1 or args.hardware_only:
        result["ok"] = bool(hw and hw["gpu_count"] >= 1)
        save()
        return 0 if result["ok"] else 1

    loaded = {}

    def _load():
        tok, model, info = stage_load(args.model, args.per_gpu_gib, hw["gpu_count"])
        loaded.update(tok=tok, model=model)
        return info

    run("load", _load)
    if loaded:
        run("generate", lambda: stage_generate(loaded["tok"], loaded["model"], args.max_new_tokens,
                                               out / "generations.jsonl"))
    result["ok"] = all(s.get("ok") for s in result["stages"].values()) and "generate" in result["stages"]
    result["finished_at"] = time.time()
    save()
    print(json.dumps({k: v for k, v in result.items() if k != "stages"}
                     | {"stages": {k: v.get("ok") for k, v in result["stages"].items()}}, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
