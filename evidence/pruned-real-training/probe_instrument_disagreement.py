"""Do nvidia-smi and torch.cuda.mem_get_info disagree during adapter generation?

Every measurement that saw the card nearly full used nvidia-smi whole-card polling:
the candidate eval (559 MiB free) and the A/B (0.56 GiB free). Every measurement
that saw ~8.8 GiB free used mem_get_info from inside the process. That is a
confounded comparison -- instrument and condition changed together -- and it was
mine to notice earlier.

On Windows WDDM the two can legitimately differ: nvidia-smi reports the device's
allocated memory as the driver sees it, which on WDDM can include system-backed
committed memory, while mem_get_info reports device-level free memory to the
process. If they diverge here, the pre-registered "oversubscription" FAIL was
triggered by an instrument artifact rather than by the model not fitting.

Samples BOTH, interleaved, during one 768-token generation with the adapter.
"""
from __future__ import annotations
import json, subprocess, sys, threading, time
from pathlib import Path
sys.path.insert(0, r"C:\Users\nikma\Chowder-router-healing\src")
MODEL = r"F:\llm-models\Qwen3.8-9B-Pruned-CW-3456"
ADAPTER = r"F:\llm-models\_a4b\realtrain-gsm8k-2\.chowder\runs\realtrain-unsloth-a9e4dbb91fa5\adapter"
EVAL = r"F:\llm-models\_a4b\gsm8k_test_50.jsonl"
OUT = Path(r"F:\llm-models\_a4b\probe-instrument-disagreement.json")
GIB = 1024**3

def smi_free_gib() -> float:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.used",
                          "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout.strip().splitlines()[0]
    total, used = (float(x) for x in out.split(","))
    return (total - used) / 1024.0

def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    from chowder.evaluators.generation import resolve_eos_token_ids
    set_seed(123)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=False, local_files_only=True)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=False, dtype=torch.bfloat16, local_files_only=True,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16),
        device_map={"": 0})
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, ADAPTER, is_trainable=False)
    model.eval()
    device = next(model.parameters()).device
    eos = resolve_eos_token_ids(tok, model)

    samples: list[dict] = []
    stop = threading.Event()
    def sampler():
        while not stop.is_set():
            free_dev, total = torch.cuda.mem_get_info()
            samples.append({
                "t": time.time(),
                "mem_get_info_free_gib": free_dev / GIB,
                "nvidia_smi_free_gib": smi_free_gib(),
                "torch_reserved_gib": torch.cuda.memory_reserved() / GIB,
            })
            stop.wait(5.0)
    row = json.loads(Path(EVAL).read_text(encoding="utf-8").splitlines()[0])
    enc = {k: v.to(device) for k, v in tok(str(row["prompt"]), return_tensors="pt").items()}
    th = threading.Thread(target=sampler, daemon=True); th.start()
    with torch.inference_mode():
        model.generate(**enc, max_new_tokens=768, do_sample=False,
                       pad_token_id=tok.pad_token_id, eos_token_id=eos)
    stop.set(); th.join(timeout=10)

    mg = [s["mem_get_info_free_gib"] for s in samples]
    sm = [s["nvidia_smi_free_gib"] for s in samples]
    print(f"samples {len(samples)}")
    print(f"  mem_get_info free  min {min(mg):.2f}  max {max(mg):.2f} GiB")
    print(f"  nvidia-smi   free  min {min(sm):.2f}  max {max(sm):.2f} GiB")
    print(f"  max disagreement   {max(a-b for a, b in zip(mg, sm)):.2f} GiB")
    verdict = ("INSTRUMENTS DISAGREE: nvidia-smi reports far less free memory than the "
               "device actually has free -- the oversubscription reading was an artifact"
               if max(a-b for a, b in zip(mg, sm)) > 2.0 else
               "INSTRUMENTS AGREE: the earlier 0.56 GiB reading is not an instrument artifact")
    print(f"  VERDICT: {verdict}")
    OUT.write_text(json.dumps({"samples": samples, "verdict": verdict}, indent=2) + "\n",
                   encoding="utf-8")
    print(f"wrote {OUT}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
