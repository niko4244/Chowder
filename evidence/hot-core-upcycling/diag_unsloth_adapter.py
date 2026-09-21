"""Why does an Unsloth-trained adapter change nothing under plain transformers?

Observed through Chowder's own lifecycle on the pruned 9B:
  * Transformers engine: baseline quality 0.30 -> candidate 0.45, and the
    predictions visibly changed. So the evaluation harness CAN measure an effect.
  * Unsloth engine: trained fine (loss 4.4014 -> 0.3760, peak 6.24 GB), the
    evaluator reported adapter_loaded=True, and yet the predictions were
    byte-identical to baseline and the metric did not move.

"adapter_loaded: True" only means the load call returned. This checks what the
load actually produced, in the same way the evaluator does it (plain
transformers + PeftModel.from_pretrained):

  1 how many LoRA modules PEFT injected, and into which families;
  2 whether the adapter's B matrices are non-zero (an all-zero B is a no-op
    regardless of how many modules were injected);
  3 whether the adapter's saved keys correspond to real module paths -- the
    classic silent failure is keys that never match, which loads "successfully"
    and adapts nothing;
  4 the decisive one: do logits actually change with the adapter enabled?
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\nikma\Chowder-router-healing\src")

BASE = r"F:\llm-models\Qwen3.8-9B-Pruned-CW-3456"
RUNS = {
    "unsloth": Path(r"F:\llm-models\_a4b\level2-unsloth"),
    "transformers": Path(r"F:\llm-models\_a4b\level2-transformers-v4"),
}


def adapter_dir(work: Path) -> Path:
    return next(work.rglob("adapter_config.json")).parent


def main() -> int:
    import torch
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(BASE, local_files_only=True, trust_remote_code=False)
    prompt = "Q: Which letter is assigned to falcon? A:"
    enc = None

    report: dict = {}
    for label, work in RUNS.items():
        ad = adapter_dir(work)
        cfg = json.loads((ad / "adapter_config.json").read_text(encoding="utf-8"))
        weights = load_file(str(ad / "adapter_model.safetensors"))
        b_keys = [k for k in weights if "lora_B" in k]
        nonzero_b = sum(1 for k in b_keys if float(weights[k].abs().max()) > 0)
        info = {
            "adapter_dir": str(ad),
            "target_modules_type": type(cfg["target_modules"]).__name__,
            "saved_tensors": len(weights),
            "lora_B_tensors": len(b_keys),
            "lora_B_nonzero": nonzero_b,
            "example_key": sorted(weights)[0] if weights else None,
        }
        print(f"\n=== {label} adapter ===")
        for k, v in info.items():
            print(f"  {k}: {v}")

        model = AutoModelForCausalLM.from_pretrained(
            BASE, quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True),
            dtype=torch.bfloat16, device_map="cuda:0",
            local_files_only=True, trust_remote_code=False)
        model.eval()
        if enc is None:
            enc = {k: v.to(model.device) for k, v in
                   tok(prompt, return_tensors="pt").items()}
        with torch.no_grad():
            base_logits = model(**enc).logits[0, -1].float().clone()

        peft_model = PeftModel.from_pretrained(model, str(ad))
        peft_model.eval()
        injected = [n for n, _ in peft_model.named_modules() if n.endswith("lora_A.default")]
        fams: dict[str, int] = {}
        for n in injected:
            leaf = n[: -len(".lora_A.default")].split(".")[-1]
            fams[leaf] = fams.get(leaf, 0) + 1
        info["injected_lora_modules"] = len(injected)
        info["injected_by_leaf"] = fams
        print(f"  injected_lora_modules: {len(injected)}")
        print(f"  injected_by_leaf: {fams}")

        with torch.no_grad():
            adapted_logits = peft_model(**enc).logits[0, -1].float()
        delta = float((adapted_logits - base_logits).abs().max())
        same_argmax = int(base_logits.argmax()) == int(adapted_logits.argmax())
        info["max_abs_logit_delta"] = delta
        info["top_token_unchanged"] = same_argmax
        info["base_top_token"] = tok.decode([int(base_logits.argmax())])
        info["adapted_top_token"] = tok.decode([int(adapted_logits.argmax())])
        print(f"  max |logit delta| with adapter: {delta:.6f}")
        print(f"  top token: base {info['base_top_token']!r} -> adapted {info['adapted_top_token']!r}")
        report[label] = info

        del peft_model, model
        torch.cuda.empty_cache()

    print("\nVERDICT")
    for label, info in report.items():
        effective = info["max_abs_logit_delta"] > 1e-3
        print(f"  {label:<13} injected={info['injected_lora_modules']:<4} "
              f"nonzero_B={info['lora_B_nonzero']}/{info['lora_B_tensors']:<4} "
              f"logit_delta={info['max_abs_logit_delta']:.4f}  "
              f"-> {'CHANGES the model' if effective else 'NO EFFECT'}")
    out = Path(r"F:\llm-models\_a4b\unsloth-adapter-diagnosis.json")
    out.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
