"""Does the guard fire on the REAL adapters? Stub tests cannot answer that."""
import json, sys
from pathlib import Path
sys.path.insert(0, r"C:\Users\nikma\Chowder-router-healing\src")
BASE = r"F:\llm-models\Qwen3.8-9B-Pruned-CW-3456"
ADAPTERS = {"transformers": Path(r"F:\llm-models\_a4b\level2-transformers-v4"),
            "unsloth": Path(r"F:\llm-models\_a4b\level2-unsloth")}
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from chowder.adapter_guard import AdapterNotLiveError, assert_adapter_is_live
out = {}
for label, work in ADAPTERS.items():
    ad = next(work.rglob("adapter_config.json")).parent
    model = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True),
        dtype=torch.bfloat16, device_map="cuda:0",
        local_files_only=True, trust_remote_code=False)
    model.eval()
    peft_model = PeftModel.from_pretrained(model, str(ad), is_trainable=False)
    try:
        rep = assert_adapter_is_live(peft_model, ad)
        out[label] = {"verdict": "ACCEPTED", "report": rep}
        print(f"{label:<13} -> ACCEPTED  matched={rep['matched_keys']} nonzero_B={rep['lora_B_nonzero']}")
    except AdapterNotLiveError as exc:
        out[label] = {"verdict": "REFUSED", "error": str(exc)}
        print(f"{label:<13} -> REFUSED")
        print("   " + str(exc).splitlines()[0])
    del peft_model, model
    torch.cuda.empty_cache()
Path(r"F:\llm-models\_a4b\guard-real-verification.json").write_text(
    json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
ok = out["transformers"]["verdict"] == "ACCEPTED" and out["unsloth"]["verdict"] == "REFUSED"
print("\nGUARD BEHAVES CORRECTLY ON REAL ARTIFACTS:", ok)
raise SystemExit(0 if ok else 2)
