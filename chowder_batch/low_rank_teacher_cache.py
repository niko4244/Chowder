"""Bounded teacher pass: cache real hidden states + logits, evaluate all ranks.

Runs ONE forward pass over a small prompt set through the full bf16 teacher
(accelerate device offload; no training), capturing for each prompt position:

* the pre-head hidden state (input to ``lm_head`` after the final norm), and
* the teacher's full next-token logit vector.

The cache is small (positions x 4096 + positions x 248320 fp32) and is reused
by every later phase: rank evaluation on *real* hidden states and the recovery
pilot both read this file instead of ever running the teacher again.

Prompt set deliberately mixes prose, code, numbers, and the runtime-repair
domain so class-specific damage is visible; the same prompts are split
train/val for the recovery pilot.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path



import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PROMPTS = [
    # prose
    "The history of the Roman Empire begins with",
    "A good breakfast for a cold morning is",
    "She opened the door and saw",
    "The conference paper concluded that",
    "In the beginning of the novel, the protagonist",
    "Weather forecasts for tomorrow indicate",
    "The recipe requires two cups of flour and",
    "After the meeting, the team decided to",
    # code
    "def parse_version(s):\n    parts = s.split('.')\n    while",
    "for row in csv.reader(handle):\n        if row and",
    "import json\n\ndef load(path):\n    with open(path) as",
    "class Counter:\n    def __init__(self):\n        self.total",
    "git checkout -b feature/ &&",
    "SELECT id, name FROM users WHERE active =",
    "x = [i ** 2 for i in range(10) if",
    "try:\n    result = risky()\nexcept",
    # numbers / structured
    "The total is 1234 + 5678 =",
    "Prices rose from $19.99 to $24.99, an increase of",
    "Version 2.10.3 is newer than version",
    "The coordinate (41.8827, -87.6233) places us in",
    "2026-09-24 13:45:00 UTC corresponds to epoch",
    "Divide 1024 by 8 to get",
    # runtime-repair domain (chowder tasks)
    "def parse_version(s):\n    return tuple(int(p) for p in",
    "def slugify(value):\n    return value.strip().lower()",
    "def total(values):\n    return sum(",
    "TARGET file version.py must report 2 passed",
    "The test suite failed with 1 failed, 1 passed because",
    "run_tests returned FAILED 3 - implementation, so the next",
    # multilingual tail (the 74% of the vocab that is rare unicode)
    "La r\u00e9publique fran\u00e7aise a \u00e9t\u00e9 proclam\u00e9e en",
    "\u4e2d\u56fd\u7684\u9996\u90fd\u662f\u5317\u4eac\uff0c\u4eba\u53e3\u5927\u7ea6\u6709",
    "\u041c\u043e\u0441\u043a\u0432\u0430 \u2014 \u0441\u0442\u043e\u043b\u0438\u0446\u0430",
    "\u0627\u0644\u0633\u0644\u0627\u0645 \u0639\u0644\u064a\u0643\u0645\u060c \u0643\u064a\u0641 \u062d\u0627\u0644\u0643",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-new", type=int, default=1)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)

    print("loading teacher (bf16, device_map=auto)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto", local_files_only=True
    )
    model.eval()

    # The wrapper's text decoder is what produces logits; route through the
    # causal-LM forward so hidden states and logits come from the same pass.
    encoded = tokenizer(PROMPTS, return_tensors="pt", padding=True)
    print("prompt tokens:", {k: tuple(v.shape) for k, v in encoded.items()}, flush=True)

    with torch.no_grad():
        output = model(**{k: v.to(model.device) for k, v in encoded.items()}, output_hidden_states=True)

    # hidden_states[-1] is the final decoder output (post final norm in HF
    # convention for these architectures) -- the actual lm_head input.
    hidden = output.hidden_states[-1].float().cpu()
    logits = output.logits.float().cpu()
    print("captured hidden", tuple(hidden.shape), "logits", tuple(logits.shape), flush=True)

    # Keep only non-padding positions.
    attention = encoded["attention_mask"].bool()
    positions = attention.nonzero(as_tuple=False)  # [N, 2] (batch, seq)
    kept_hidden = hidden[positions[:, 0], positions[:, 1]]
    kept_logits = logits[positions[:, 0], positions[:, 1]]
    token_ids = encoded["input_ids"][positions[:, 0], positions[:, 1]]
    print("kept positions:", kept_hidden.shape[0], flush=True)

    # torch.save's zipfile writer is unreliable on this Windows setup, so the
    # cache is written as safetensors tensors plus a JSON sidecar.
    from safetensors.torch import save_file

    tensors = {
        "hidden": kept_hidden.contiguous(),   # [N, hidden] fp32
        "logits": kept_logits.contiguous(),   # [N, vocab] fp32
        "token_ids": token_ids.to(torch.int32).contiguous(),
        "batch_idx": positions[:, 0].to(torch.int32).contiguous(),
        "seq_idx": positions[:, 1].to(torch.int32).contiguous(),
    }
    save_file(tensors, str(out_path))
    sidecar = out_path.with_suffix(".json")
    sidecar.write_text(
        json.dumps({"n_positions": int(kept_hidden.shape[0]), "prompts": PROMPTS}, indent=2),
        encoding="utf-8",
    )
    print("wrote", out_path, f"({out_path.stat().st_size / 1e6:.0f} MB) + {sidecar.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
