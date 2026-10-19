# Ornith curriculum on Kaggle

One round of propose (Ornith-1.5-9B) → cross-check (Qwythos-27B, code-valid proposals only) → admit (hermes, difficulty skipped), on a 2x T4 script kernel. Mirrors `C:\llm-work\ornith\run-ornith-run3.ps1`; both GGUFs are pinned by revision and sha256 in `models.json`.

```powershell
python kaggle/ornith_curriculum/make_dataset.py --owner <user> --hermes-dir C:\Users\nikma\far-hermes-crosscheck\hermes --ornith-phase C:\llm-work\ornith\ornith_phase.py --out C:\llm-work\kaggle\ornith-ds
kaggle datasets create -p C:\llm-work\kaggle\ornith-ds --dir-mode zip   # --dir-mode zip, or hermes/ is skipped
python kaggle/ornith_curriculum/make_kernel.py --owner <user> --out C:\llm-work\kaggle\ornith-k
kaggle kernels push -p C:\llm-work\kaggle\ornith-k
kaggle kernels status <user>/ornith-curriculum-run3
kaggle kernels output <user>/ornith-curriculum-run3 -p C:\llm-work\ornith\curriculum-run3
```

Kaggle runs at most 2 GPU sessions at once. Proposing stops after `PROPOSE_BUDGET_S` (6 h) and keeps every finished round, so the whole kernel fits the 12 h session limit.
