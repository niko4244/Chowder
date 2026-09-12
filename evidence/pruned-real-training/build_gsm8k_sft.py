"""GSM8K train split as plain-text SFT rows, with the test set excluded by construction."""
import json, os, re
from pathlib import Path
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
from datasets import load_dataset

train = load_dataset("openai/gsm8k", "main", split="train")
test_qs = {i["question"] for i in load_dataset("openai/gsm8k", "main", split="test")}
rows, skipped = [], 0
for item in train:
    if item["question"] in test_qs:      # official splits are disjoint; verify, don't assume
        skipped += 1
        continue
    # Strip the <<...>> calculator annotations GSM8K embeds; keep the #### answer so
    # the model learns to finish on the number the eval reads.
    sol = re.sub(r"<<[^>]*>>", "", item["answer"]).strip()
    rows.append({"text": f"Question: {item['question']}\nThink step by step, then end with the final numeric answer.\nAnswer: {sol}"})
out = Path(r"F:\llm-models\_a4b\gsm8k_train_sft.jsonl")
out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
print(f"wrote {out}: {len(rows)} rows, {skipped} overlapping test questions excluded")
print("sample:", rows[0]["text"][:180].replace("\n", " | "))
