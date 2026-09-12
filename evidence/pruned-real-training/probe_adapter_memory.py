"""Where do the ~8.45 GiB go when a 200-module LoRA adapter is attached?

Established by the A/B (evidence/pruned-real-training/ab-adapter-overhead.json):
generating with the adapter attached puts the card at 0.56 GiB free while torch
reports only 5.84 GiB reserved, from a clean start -- ~8.45 GiB exists that torch
does not account for. That is a reproduction, not an explanation.

INSTRUMENT. `torch.cuda.mem_get_info()` reports the DRIVER's free/total from inside
this process, so

    non_torch = (total - driver_free) - torch_reserved

isolates memory held outside torch's caching allocator without nvidia-smi, and
without other processes confusing the number the way whole-card polling does.

DISCRIMINATOR. Memory is measured at several token budgets on the same prompt:

  * scales with generated tokens  -> cache / state growth (KV, or the hybrid's
    linear-attn state if attaching PEFT knocks generation off its specialised cache)
  * constant jump on first generate -> workspace allocation (cuBLAS handles, bnb
    dequantisation buffers) that does not depend on sequence length

FIXES THE EARLIER DESIGN FLAW: one arm per PROCESS. The A/B ran both arms in a single
process, so the adapter arm could not be given the headroom its own decision rule
required. Here the OS reclaims everything between arms.

Run as:  probe_adapter_memory.py --with-adapter 0   then   --with-adapter 1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\Users\nikma\Chowder-router-healing\src")

MODEL = r"F:\llm-models\Qwen3.8-9B-Pruned-CW-3456"
ADAPTER = r"F:\llm-models\_a4b\realtrain-gsm8k-2\.chowder\runs\realtrain-unsloth-a9e4dbb91fa5\adapter"
EVAL = r"F:\llm-models\_a4b\gsm8k_test_50.jsonl"
BUDGETS = (1, 16, 64, 256, 768)
GIB = 1024**3


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def snapshot(torch) -> dict:
    free, total = torch.cuda.mem_get_info()
    reserved = torch.cuda.memory_reserved()
    allocated = torch.cuda.memory_allocated()
    used = total - free
    return {
        "driver_used_gib": used / GIB,
        "driver_free_gib": free / GIB,
        "torch_reserved_gib": reserved / GIB,
        "torch_allocated_gib": allocated / GIB,
        "non_torch_gib": (used - reserved) / GIB,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-adapter", type=int, required=True, choices=(0, 1))
    args = ap.parse_args()
    with_adapter = bool(args.with_adapter)
    label = "adapter" if with_adapter else "no-adapter"
    out = Path(rf"F:\llm-models\_a4b\probe-memory-{label}.json")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

    from chowder.evaluators.generation import resolve_eos_token_ids

    before = snapshot(torch)
    log(f"arm {label}: before load, non-torch {before['non_torch_gib']:.2f} GiB "
        f"(driver used {before['driver_used_gib']:.2f})")

    set_seed(123)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=False, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=False, dtype=torch.bfloat16, local_files_only=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        device_map={"": 0},
    )
    if with_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, ADAPTER, is_trainable=False)
    model.eval()
    device = next(model.parameters()).device
    eos = resolve_eos_token_ids(tokenizer, model)

    after_load = snapshot(torch)
    log(f"    after load: torch reserved {after_load['torch_reserved_gib']:.2f}, "
        f"non-torch {after_load['non_torch_gib']:.2f} GiB")

    row = json.loads(Path(EVAL).read_text(encoding="utf-8").splitlines()[0])
    encoded = tokenizer(str(row["prompt"]), return_tensors="pt")
    encoded = {k: v.to(device) for k, v in encoded.items()}

    steps: list[dict] = []
    cache_kind = None
    with torch.inference_mode():
        for budget in BUDGETS:
            torch.cuda.synchronize()
            started = time.perf_counter()
            generated = model.generate(
                **encoded, max_new_tokens=budget, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=eos,
                return_dict_in_generate=True,
            )
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            if cache_kind is None:
                pkv = getattr(generated, "past_key_values", None)
                cache_kind = type(pkv).__name__ if pkv is not None else "none"
            snap = snapshot(torch)
            new_tokens = int(generated.sequences.shape[1] - encoded["input_ids"].shape[1])
            snap.update({
                "budget": budget, "new_tokens": new_tokens, "seconds": seconds,
                "ms_per_token": seconds / max(new_tokens, 1) * 1000,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / GIB,
                "non_torch_above_load_gib": snap["non_torch_gib"] - after_load["non_torch_gib"],
            })
            steps.append(snap)
            log(f"    {budget:>4} tok -> {new_tokens:>4} gen  {snap['ms_per_token']:6.1f} ms/tok  "
                f"reserved {snap['torch_reserved_gib']:5.2f}  "
                f"non-torch {snap['non_torch_gib']:5.2f} "
                f"(+{snap['non_torch_above_load_gib']:.2f} since load)  "
                f"driver free {snap['driver_free_gib']:5.2f}")

    report = {
        "arm": label, "with_adapter": with_adapter, "model": MODEL,
        "cache_class": cache_kind,
        "before_load": before, "after_load": after_load, "steps": steps,
    }
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    log(f"    cache class: {cache_kind}")
    log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
