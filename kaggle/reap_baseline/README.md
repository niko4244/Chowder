# Kaggle REAP baseline kit

One private Kaggle script kernel per baseline arm. Each kernel builds llama.cpp
at the frozen commit with CUDA for a 2x T4, downloads that arm's GGUF and
verifies its sha256, then runs the pre-registered eval (`reap_serve.py` +
`reap_eval.py`, attached read-only as the private dataset
`<owner>/chowder-reap-eval`) with the same flags the local arms ran under.

Nothing here uploads anything. The commands below are the upload, and they are
the orchestrator's to run.

## 0. Generate the upload directories

From the repo worktree root (`C:\Users\nikma\oc-kaggle-kit`):

```powershell
python kaggle/reap_baseline/make_dataset.py --owner <kaggle-user> --out C:\llm-work\reap\kaggle\chowder-reap-eval
python kaggle/reap_baseline/make_kernels.py --owner <kaggle-user> --out C:\llm-work\reap\kaggle\kernels
```

`make_dataset.py` prints the file list and a sha256 manifest (also written to
`manifest.json` next to the dataset files). `make_kernels.py` prints the kernel
id it wrote for each arm.

## 1. Create the private eval-code dataset (before any kernel is pushed)

```powershell
kaggle datasets create -p C:\llm-work\reap\kaggle\chowder-reap-eval
kaggle datasets status <kaggle-user>/chowder-reap-eval
```

Check on kaggle.com that it is **Private**: `isPrivate` in the metadata is a
request, and nothing local can confirm what the site did with it.

## 2. Push the kernels

```powershell
kaggle kernels push -p C:\llm-work\reap\kaggle\kernels\qwen38_27b
kaggle kernels push -p C:\llm-work\reap\kaggle\kernels\qwythos_27b
kaggle kernels push -p C:\llm-work\reap\kaggle\kernels\ornith15_35b
```

## 3. Watch status

```powershell
kaggle kernels status <kaggle-user>/reap-baseline-qwen38-27b
kaggle kernels status <kaggle-user>/reap-baseline-qwythos-27b
kaggle kernels status <kaggle-user>/reap-baseline-ornith15-35b
```

## 4. Fetch outputs, then copy each arm into the local eval tree

```powershell
kaggle kernels output <kaggle-user>/reap-baseline-qwen38-27b -p C:\llm-work\reap\kaggle\out\qwen38_27b
kaggle kernels output <kaggle-user>/reap-baseline-qwythos-27b -p C:\llm-work\reap\kaggle\out\qwythos_27b
kaggle kernels output <kaggle-user>/reap-baseline-ornith15-35b -p C:\llm-work\reap\kaggle\out\ornith15_35b

New-Item -ItemType Directory -Force C:\llm-work\reap\eval\qwen38_27b
Copy-Item C:\llm-work\reap\kaggle\out\qwen38_27b\eval\qwen38_27b\* C:\llm-work\reap\eval\qwen38_27b\ -Recurse -Force
New-Item -ItemType Directory -Force C:\llm-work\reap\eval\qwythos_27b
Copy-Item C:\llm-work\reap\kaggle\out\qwythos_27b\eval\qwythos_27b\* C:\llm-work\reap\eval\qwythos_27b\ -Recurse -Force
New-Item -ItemType Directory -Force C:\llm-work\reap\eval\ornith15_35b
Copy-Item C:\llm-work\reap\kaggle\out\ornith15_35b\eval\ornith15_35b\* C:\llm-work\reap\eval\ornith15_35b\ -Recurse -Force
```

Required destination, per arm:

```
C:\llm-work\reap\eval\<arm>\gsm8k.jsonl
C:\llm-work\reap\eval\<arm>\humaneval.jsonl
C:\llm-work\reap\eval\<arm>\summary.json
C:\llm-work\reap\eval\<arm>\server.log     # from the kernel: keep it, it is evidence
C:\llm-work\reap\eval\<arm>\serve.json     # frozen flags, gguf sha256, experts_offloaded
```

The kernel root of each output directory also carries the run evidence:
`run.log`, `provenance.json` (llama.cpp commit actually built, GGUF sha256
verified, GPU names, exit codes per step). Keep both next to the arm's eval
files, and note `serve.json` carries `experts_offloaded`, which `reap_gates.py`
reads for the speed/fit gate. The gate computation itself only needs
`eval\<arm>\{gsm8k,humaneval}.jsonl`.

## Notes

- The two frozen scripts (`chowder_batch/reap_serve.py`, `chowder_batch/reap_eval.py`)
  and the scorer they import are copied byte-identically; the printed sha256
  manifest is the record of that. The kernels never edit them.
- Each kernel needs internet (git clone, HF download, plus `load_dataset` for the
  two suites) and a GPU: `enable_gpu` gives the 2x T4 this eval was scoped for.
- Scratch (`/kaggle/tmp`, falling back to `/tmp` if that path is not writable)
  holds the llama.cpp clone and the 17-21 GB GGUF; `/kaggle/working` holds only
  logs and eval results, because everything there becomes a kernel output.
- Rerunning a kernel in the same session resumes: the build is skipped if the
  binary exists, the download is skipped if its sha256 sidecar is already
  verified, and `reap_eval.py` keeps the rows it already scored.
- The GGUF download uses `HF_TOKEN` from the kernel environment when one is set.
  All three repos are expected to be public; a 401/403 means the repo is gated.
- llama.cpp is checked out at the short commit `760984655` (local build 10985,
  read from the local server binary's version string). If that ref will not
  resolve on a fresh clone the kernel falls back to the release tag `b10985` and
  records whichever ref it actually built in `provenance.json`.
