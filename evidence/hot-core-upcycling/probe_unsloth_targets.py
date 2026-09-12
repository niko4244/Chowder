"""Which target_modules spec makes Unsloth adapt the linear_attn modules?

Measured facts so far: with text_only=True the adapter's keys match the evaluator's
model, but Unsloth converts an explicit LIST into a regex whose component block is
(self_attn|attention|attn|mixer|mlp|feed_forward|ffn|dense) -- none of which match
`linear_attn` -- so 72 of 200 modules are skipped. vision.py shows a STRING is passed
through to PEFT untouched, while a list is only converted when a finetune_* filter is
set... which Chowder does not set. So measure what actually happens.

PEFT applies a regex with re.fullmatch, and its list semantics are suffix matching,
so `.*\.(?:name1|name2|...)` reproduces a list exactly while bypassing the converter.
"""
import json, sys
from pathlib import Path
MODEL = r"F:\llm-models\Qwen3.8-9B-Pruned-CW-3456"
NAMES = ["q_proj","k_proj","v_proj","o_proj","in_proj_qkv","in_proj_z","out_proj",
         "gate_proj","up_proj","down_proj"]
REGEX = r".*\.(?:" + "|".join(NAMES) + r")"

def counts(model):
    targets = set()
    for name, _ in model.named_modules():
        i = name.find(".lora_A")
        if i > 0: targets.add(name[:i])
    out = {}
    for t in targets:
        leaf = t.rsplit(".", 1)[-1]
        out[leaf] = out.get(leaf, 0) + 1
    return out

results = {}
for label, spec in (("list", list(NAMES)), ("regex_string", REGEX)):
    from unsloth import FastLanguageModel
    model, tok = FastLanguageModel.from_pretrained(
        model_name=MODEL, max_seq_length=64, dtype=None,
        load_in_4bit=True, text_only=True)
    model = FastLanguageModel.get_peft_model(
        model, r=8, target_modules=spec, lora_alpha=16, lora_dropout=0.0,
        bias="none", use_gradient_checkpointing="unsloth", random_state=1)
    cfg = model.peft_config[model.active_adapter]
    by = counts(model)
    results[label] = {
        "passed_in": spec if isinstance(spec, str) else "(list)",
        "peft_target_modules_type": type(cfg.target_modules).__name__,
        "peft_target_modules": (cfg.target_modules if isinstance(cfg.target_modules, str)
                                else sorted(cfg.target_modules)),
        "adapted_by_leaf": by,
        "adapted_total": sum(by.values()),
        "linear_attn_covered": all(by.get(n, 0) > 0 for n in ("in_proj_qkv","in_proj_z","out_proj")),
    }
    print(f"\n=== {label} ===")
    print("  peft target_modules type:", results[label]["peft_target_modules_type"])
    print("  adapted total:", results[label]["adapted_total"], "| by leaf:", by)
    print("  linear_attn covered:", results[label]["linear_attn_covered"])
    del model
    import torch, gc; gc.collect(); torch.cuda.empty_cache()

Path(r"F:\llm-models\_a4b\unsloth-target-probe.json").write_text(
    json.dumps(results, indent=2) + "\n", encoding="utf-8")
print("\nVERDICT: regex_string covers linear_attn:", results["regex_string"]["linear_attn_covered"],
      "| list covers:", results["list"]["linear_attn_covered"])
