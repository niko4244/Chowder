"""Compare gen-2 and a repair bundle across RRSI evolve/held-out splits."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from chowder.local_model_compat import patch_transformers5_custom_model
from chowder.runtime_eval import HELDOUT_TASKS, TASKS, make_transformers_generate, run_live_benchmark


def load_model(base_dir: str, adapter_dir: str | None):
    digests = {
        name: hashlib.sha256((Path(base_dir) / name).read_bytes()).hexdigest()
        for name in ("configuration_spark.py", "modeling_spark.py")
    }
    patch_transformers5_custom_model(base_dir, digests)
    tokenizer = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_dir, trust_remote_code=True, local_files_only=True,
        dtype=torch.bfloat16, device_map="cuda:0",
    )
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
        repair_dir = Path(adapter_dir) / "repair"
        if (repair_dir / "adapter_config.json").is_file():
            model.load_adapter(str(repair_dir), adapter_name="repair", is_trainable=False)
            model.base_model.add_weighted_adapter(
                ["default", "repair"], [1.0, 1.0],
                adapter_name="combined", combination_type="linear",
            )
            model.set_adapter("combined", inference_mode=True)
    model.eval()
    return tokenizer, model


def run_arm(base_dir: str, adapter_dir: str | None, max_turns: int) -> dict:
    tokenizer, model = load_model(base_dir, adapter_dir)
    generate = make_transformers_generate(
        tokenizer, model, max_new_tokens=128, device=next(model.parameters()).device
    )
    result = {}
    for harness in ("plain", "guarded"):
        result[harness] = {
            "evolve": run_live_benchmark(generate, max_turns=max_turns, harness=harness, tasks=TASKS, split="evolve"),
            "heldout": run_live_benchmark(generate, max_turns=max_turns, harness=harness, tasks=HELDOUT_TASKS, split="heldout"),
        }
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-turns", type=int, default=8)
    args = parser.parse_args()
    payload = {
        "parent": run_arm(args.base, args.parent, args.max_turns),
        "candidate": run_arm(args.base, args.candidate, args.max_turns),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        arm: {harness: {split: result[split]["metrics"] for split in result} for harness, result in arms.items()}
        for arm, arms in payload.items()
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
