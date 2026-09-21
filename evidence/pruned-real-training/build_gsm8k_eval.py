"""Build a GSM8K eval jsonl from the locally cached openai/gsm8k test split.

`expected` is the reference final answer after the "####" marker, which is what
GSM8K ships as ground truth. The prompt asks for the answer last so a verbose
chain-of-thought still ends on the number final_number_match reads.
"""
import json, os
from pathlib import Path
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
from datasets import load_dataset

N = 50
ds = load_dataset("openai/gsm8k", "main", split="test")
rows = []
for item in ds.select(range(N)):
    answer = item["answer"]
    assert "####" in answer, "GSM8K answers carry a #### final-answer marker"
    final = answer.rsplit("####", 1)[1].strip().replace(",", "")
    rows.append({
        "prompt": f"Question: {item['question']}\nThink step by step, then end with the final numeric answer.\nAnswer:",
        "expected": final,
    })
out = Path(r"F:\llm-models\_a4b\gsm8k_test_50.jsonl")
out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
print(f"wrote {out} with {len(rows)} problems")
print("first expected values:", [r["expected"] for r in rows[:6]])
print("sample prompt:", rows[0]["prompt"][:140].replace("\n", " | "))
