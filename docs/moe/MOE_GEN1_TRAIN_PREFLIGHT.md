# MoE Gen-1 training data admission

The historical Gen-1 set was initially screened at 4,096 tokens while the
training recipe uses 2,048. `training_data._truncate_example` permits a
prefix of an assistant answer to remain labeled after truncation. Before
launching each learning-rate and Fable/control recipe, check the **actual**
mix file with the model's chat tokenizer:

```powershell
python -B chowder_batch/moe_gen1_train_preflight.py `
  --mix C:/llm-work/train-sets/moe-gen1-v1 `
  --tokenizer G:/llm-cold/bf16/Qwen3.6-35B-A3B `
  --max-length 2048 `
  --report C:/llm-work/train-sets/moe-gen1-data-preflight.json
```

Exit **0** means the source files match their builder manifest, the Fable-free
control is a strict ordered subset of the Fable arm, every row rendered with
the same chat-labeling routine used by the Transformers/Unsloth workers, and
every entire rendered row fits the declared sequence length. Exit **2** means
one or more rows would be shortened or could not be safely rendered. Missing
files, altered hashes, an unvalidated template, or a different tokenizer path
fail before the result can be marked ready. The report includes the manifest
hash, tokenizer file hashes, examples of refused rows, and both arms' counts.
The checker is CPU-only and does not load model weights.

## Observed local mix (2026-10-23)

On manifest SHA-256
`9ac93c3763427605cdb0a51d3bb08e70e4278011663800dbdd98622b86df710c`,
the real Qwen3.6 tokenizer yielded:

| Arm | Rows | Max tokens | Over 2,048 | Lost assistant answers |
| --- | ---: | ---: | ---: | ---: |
| Fable | 767 | 1,906 | 0 | 0 |
| Control | 647 | 1,855 | 0 | 0 |

There are exactly 120 Fable-only rows. These figures describe the **current
locally inspected** artifact. The earlier 1,028-row build and its 136 rows
over 2,048 documented in `docs/MOE_GEN1_MIX_V1.md` were a different artifact.
The observed result is sequence-integrity evidence only. It does not certify
Gen-0 model quality, a placement gate, an optimizer step, a complete checkpoint
resume, licensing decision N2, or successful adapter reload/inference.
