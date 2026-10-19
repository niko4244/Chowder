"""Kaggle script kernel: build llama.cpp, fetch one REAP baseline GGUF, run the frozen eval.

One kernel per arm. ``ARM`` and ``BASELINE`` at the top of this file are the
only arm-specific values; ``make_kernels.py`` rewrites those two lines and
nothing else. The pre-registered evaluators are never touched here: this script
launches ``reap_serve.py`` with exactly the arguments the local head-to-head
used, so every arm is scored by the same code, the same frozen server flags
(``--fit on -np 4 -c 16384 --jinja``) and the same greedy decoding.

The machine is a 2x T4 (compute capability 7.5), so llama.cpp is built for that
one architecture: the server that scores this arm is the same server that scored
the local arms, which is the whole point of pinning the commit.

Everything expensive is skipped on a rerun inside the same session: the binary
if it already exists, the download if its sha256 sidecar already holds a
verified digest, and the eval itself because ``reap_eval.py`` keeps the rows it
already scored.

Outputs (all under /kaggle/working, so ``kaggle kernels output`` fetches them):
``run.log``, ``provenance.json``, ``eval/<arm>/{gsm8k.jsonl,humaneval.jsonl,summary.json,server.log,serve.json}``.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

# --- arm identity: rewritten by kaggle/reap_baseline/make_kernels.py ---
ARM = "qwen38_27b"
BASELINE = {"arm": "qwen38_27b", "file": "Qwen3.8-27B-Q4_K_M.gguf", "hf_repo": "unsloth/Qwen3.8-27B-GGUF", "revision": "fdd03b8bbd279c1694563650e79d85a2373d9934", "sha256": "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"}

# --- frozen protocol ---
LLAMA_REPO = "https://github.com/ggml-org/llama.cpp"
# Short commit of the local build (10985); verified against the version string the
# local llama-server binary carries. The release tag of the same build is the
# fallback for the case where the abbreviated sha will not resolve on a fresh
# clone -- whichever lands is logged and written to provenance.json.
LLAMA_COMMIT = "760984655"
LLAMA_TAG = "b10985"
WORK = Path("/kaggle/working")
SCRATCH_PRIMARY = Path("/kaggle/tmp")
SCRATCH_FALLBACK = Path("/tmp")
EVAL_INPUT = Path("/kaggle/input/chowder-reap-eval")
OUT = WORK / "eval"
LOG_PATH = WORK / "run.log"
PROVENANCE = WORK / "provenance.json"
PORT = 8790
SUITES = "gsm8k,humaneval"

EXIT_CODES: dict[str, int] = {}
FACTS: dict[str, str] = {}


def log(message: str) -> None:
    stamped = f"[{dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] {message}"
    print(stamped, flush=True)
    try:
        WORK.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(stamped + "\n")
    except OSError as exc:  # the run must not die because its log file is unwritable
        print(f"run.log unwritable ({exc}); continuing with stdout only", flush=True)


def run(step: str, cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run one step, record its exit code in provenance, and fail loudly on nonzero."""
    log(f"{step}: {' '.join(cmd)}")
    done = subprocess.run(cmd, **kwargs)
    EXIT_CODES[step] = done.returncode
    if done.returncode != 0:
        raise RuntimeError(f"step {step!r} exited {done.returncode}: {cmd[0]}")
    return done


def verify_sha256(path: Path, expected: str, chunk_size: int = 1 << 20) -> str:
    """Stream ``path`` through sha256; return the digest, or raise if it is not ``expected``.

    Chunked on purpose: the GGUF files here are 17-21 GB and must never be read
    into memory at once.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(f"sha256 mismatch for {Path(path).name}: expected {expected}, got {actual}")
    return actual


def scratch_dir() -> Path:
    """Writable scratch outside /kaggle/working (a GGUF must not become an output artifact)."""
    for candidate in (SCRATCH_PRIMARY, SCRATCH_FALLBACK):
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            if candidate != SCRATCH_PRIMARY:
                log(f"{SCRATCH_PRIMARY} is unusable; using {candidate} for the clone and the GGUF")
            return candidate
        except OSError as exc:
            log(f"scratch {candidate} unusable: {exc}")
    raise RuntimeError("no writable scratch directory")


def cuda_driver_stub() -> str:
    for candidate in ("/usr/local/cuda/lib64/stubs/libcuda.so",
                      "/usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so"):
        if Path(candidate).exists():
            return candidate
    raise RuntimeError("no libcuda.so stub in the CUDA toolkit; cannot link ggml-cuda")


def build_llama_server(scratch: Path) -> Path:
    source = scratch / "llama.cpp"
    server = source / "build" / "bin" / "llama-server"
    if server.exists():
        log(f"llama-server already built at {server}; skipping clone and build")
        return server

    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git is not installed in this image")
    if not (source / ".git").exists():
        run("git-clone", [git, "clone", LLAMA_REPO, str(source)])
    checked_out = None
    for ref in (LLAMA_COMMIT, LLAMA_TAG):
        done = subprocess.run([git, "-C", str(source), "checkout", "--force", ref], capture_output=True, text=True)
        EXIT_CODES[f"git-checkout-{ref}"] = done.returncode
        if done.returncode == 0:
            checked_out = ref
            break
        log(f"llama.cpp ref {ref!r} did not resolve: {(done.stderr or '').strip().splitlines()[-1:] or 'no stderr'}")
    if checked_out is None:
        raise RuntimeError(f"llama.cpp has neither {LLAMA_COMMIT!r} nor {LLAMA_TAG!r}; cannot match the local server")
    head = subprocess.run([git, "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    FACTS["llama_cpp_checkout"] = checked_out
    FACTS["llama_cpp_commit"] = head.stdout.strip()
    log(f"llama.cpp at {head.stdout.strip()} (requested {LLAMA_COMMIT})")

    run("cmake-configure", ["cmake", "-S", str(source), "-B", str(source / "build"),
                            "-DGGML_CUDA=ON", "-DCMAKE_CUDA_ARCHITECTURES=75", "-DLLAMA_CURL=OFF",
                            # Kaggle's image has no libcuda.so at build time; link the
                            # toolkit stub (the real driver is loaded at runtime).
                            f"-DCUDA_cuda_driver_LIBRARY={cuda_driver_stub()}"])
    run("cmake-build", ["cmake", "--build", str(source / "build"), "--target", "llama-server",
                        "-j", str(os.cpu_count() or 4)])
    if not server.exists():
        raise RuntimeError(f"build reported success but {server} does not exist")
    return server


def download_gguf(scratch: Path) -> Path:
    target = scratch / BASELINE["file"]
    stamp = target.with_suffix(target.suffix + ".sha256")
    if target.exists() and stamp.exists() and stamp.read_text(encoding="utf-8").strip() == BASELINE["sha256"]:
        log(f"{target.name} is present with a verified sha256 sidecar; skipping the download")
        FACTS["gguf_sha256"] = BASELINE["sha256"]
        return target

    from huggingface_hub import hf_hub_download  # shipped in the Kaggle image

    log(f"downloading {BASELINE['hf_repo']}@{BASELINE['revision']} :: {BASELINE['file']} -> {scratch}")
    downloaded = Path(hf_hub_download(repo_id=BASELINE["hf_repo"], filename=BASELINE["file"],
                                      revision=BASELINE["revision"], local_dir=str(scratch)))
    digest = verify_sha256(downloaded, BASELINE["sha256"])
    stamp.write_text(digest + "\n", encoding="utf-8")
    FACTS["gguf_sha256"] = digest
    FACTS["gguf_path"] = str(downloaded)
    log(f"sha256 verified: {digest}")
    return downloaded


def run_eval(server: Path, gguf: Path) -> None:
    harness = EVAL_INPUT / "reap_serve.py"
    if not harness.exists():
        raise RuntimeError(f"{harness} is missing -- attach the private chowder-reap-eval dataset")
    OUT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(EVAL_INPUT) + os.pathsep + env.get("PYTHONPATH", "")
    # reap_eval._load does os.environ.setdefault("HF_DATASETS_OFFLINE", "1"), i.e. the
    # frozen offline default only wins if it is unset. These suites are not in the
    # image, so pre-set "0" and let load_dataset download them (internet is on).
    env["HF_DATASETS_OFFLINE"] = "0"
    run("reap_serve", [sys.executable, str(harness),
                       "--gguf", str(gguf), "--arm", ARM, "--port", str(PORT), "--out", str(OUT),
                       "--server", str(server),
                       "--", "--port", str(PORT), "--arm", ARM, "--suites", SUITES, "--out", str(OUT)],
        env=env)


def gpu_names() -> list[str]:
    try:
        done = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=60)
        return [line.strip() for line in done.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError) as exc:
        log(f"nvidia-smi unavailable: {exc}")
        return []


def write_provenance(started: str, status: str, error: str | None) -> None:
    document = {
        "arm": ARM,
        "hf_repo": BASELINE["hf_repo"],
        "revision": BASELINE["revision"],
        "file": BASELINE["file"],
        "sha256_expected": BASELINE["sha256"],
        "sha256_verified": FACTS.get("gguf_sha256"),
        "llama_cpp_requested_commit": LLAMA_COMMIT,
        "llama_cpp_checkout": FACTS.get("llama_cpp_checkout"),
        "llama_cpp_commit": FACTS.get("llama_cpp_commit"),
        "server_flags": "frozen in reap_serve.py: --fit on -np 4 -c 16384 --jinja",
        "suites": SUITES,
        "port": PORT,
        "out": str(OUT),
        "gpus": gpu_names(),
        "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
        "status": status,
        "error": error,
        "started_utc": started,
        "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "exit_codes": dict(EXIT_CODES),
    }
    PROVENANCE.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    log(f"REAP baseline arm {ARM}: {BASELINE['hf_repo']}@{BASELINE['revision']} :: {BASELINE['file']}")
    status, error = "ok", None
    try:
        scratch = scratch_dir()
        server = build_llama_server(scratch)
        gguf = download_gguf(scratch)
        run_eval(server, gguf)
    except Exception as exc:
        status, error = "failed", f"{type(exc).__name__}: {exc}"
        log(f"FAILED: {error}")
        traceback.print_exc()
    write_provenance(started, status, error)
    log(f"arm {ARM} finished with status {status}")
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
