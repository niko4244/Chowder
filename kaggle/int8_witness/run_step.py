"""Kaggle script kernel: real int8 placement steps on reap160.

The Windows box could not finish this run -- 28.6 GB after load, 44.5 GiB private
commit during the second backward, 0.4 GiB free, step 2 paging -- and the
capacity probe showed the 18.8 GB grid holding at 19.9 GB RSS on a Kaggle
session. This kernel runs the *real* thing: the worker subprocess, the real
checkpoint, the real spec, `max_steps` steps.

What it measures, while the worker runs: the child's peak RSS, the host's lowest
available memory, wall time to load and to step, the worker's own telemetry, and
whether `placement-witnesses.json` appeared beside the adapter. It writes them to
`/kaggle/working/real_step.json` and prints them.

A one-step run can populate the gradient witness; the smoke and router-balance
gates need the full 20 steps and stay `not measured` until a run does them.
"""

from __future__ import annotations

import json
import hashlib
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

GIB = 1 << 30
MIB = 1 << 20
INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
# Local scratch for the staged checkpoint: the dataset mount is read-only and slow.
SCRATCH = Path("/tmp/chowder-model")
CODE_FILE = Path("src/chowder/backends/moe_int8_experts.py")
MODEL_FILE = "model.safetensors.index.json"

# reap160, and the run the local box attempted at seq 512 / batch 1.
HIDDEN = 2048
MAX_STEPS = int(os.environ.get("WITNESS_MAX_STEPS", "1"))
MAX_LENGTH = int(os.environ.get("WITNESS_MAX_LENGTH", "512"))

# The same 24 rows the local attempt bound into its spec: short, neutral text so
# the loss curve is about the placement, not the corpus.
ROWS = [
    "The routed experts of this layer keep their weights frozen; only the adapter trains.",
    "def add(a, b):\n    return a + b",
    "Traceback (most recent call last): ValueError: invalid literal for int()",
    "Call the search tool, then read the top result before answering.",
    "Step 1: gather requirements. Step 2: design. Step 3: implement.",
    "The compiler rejects the template because the type is not deduced.",
    "SELECT id, name FROM users WHERE active = 1 ORDER BY name;",
    "An int8 matrix stores one scale per row; dequantising multiplies in fp32.",
    "Write a test that fails before the fix and passes after it.",
    "The cache is warm, so the second call answers from memory.",
    "Explain the difference between a lock and a semaphore.",
    "assert response.status_code == 200, response.text",
    "The gradient is finite and non-zero on every adapter leaf.",
    "Rotate the key, then restart the service that reads it.",
    "A page table entry maps a virtual page to a physical frame.",
    "Summarise the log lines into one sentence.",
    "for key, value in mapping.items():\n    print(key, value)",
    "The router picked eight experts out of one hundred and sixty.",
    "Measure first, then optimise the part that is actually slow.",
    "The dataset digest is bound into the run so a resume cannot drift.",
    "Delete the stale branch after the merge lands.",
    "The model card lists the license and the intended use.",
    "Two processes raced on the same file and one lost its write.",
    "Keep the evidence next to the artefact it describes.",
]


def log(message: str) -> None:
    print(f"[step] {message}", flush=True)


def find_dataset(marker: Path) -> Path | None:
    """The mounted directory that contains ``marker``, at whatever depth it is mounted."""
    if not INPUT.is_dir():
        return None
    for hit in sorted(INPUT.rglob(marker.name)):
        for depth in range(0, 5):
            try:
                root = hit.parents[depth]
            except IndexError:
                break
            if (root / marker).is_file():
                return root
    return None


def find_model() -> Path | None:
    """The checkpoint directory: it has a shard index and a config that names the arch."""
    for hit in sorted(INPUT.rglob(MODEL_FILE)):
        candidate = hit.parent
        config = candidate / "config.json"
        if not config.is_file():
            continue
        data = json.loads(config.read_text(encoding="utf-8"))
        if data.get("model_type") == "qwen3_5_moe":
            return candidate
    return None


class Sampler(threading.Thread):
    """Watch the worker and the host while the worker runs.

    ``VmHWM`` is the child's own peak RSS, which is a figure the worker cannot
    report on itself after exiting; ``MemAvailable`` is what the kernel's OOM
    killer looks at, and 0.4 GiB of it is what ended the Windows attempt.
    """

    def __init__(self, pid_file: Path) -> None:
        super().__init__(daemon=True)
        self.pid_file = pid_file
        self.stop = threading.Event()
        self.peak_child_rss_gb = 0.0
        self.peak_child_hwm_gb = 0.0
        self.min_available_gb = float("inf")
        self.samples = 0

    def run(self) -> None:
        while not self.stop.is_set():
            try:
                self.min_available_gb = min(self.min_available_gb, _mem_available())
            except OSError:
                pass
            pid = _read_pid(self.pid_file)
            if pid is not None:
                status = _proc_status(pid)
                if status:
                    self.peak_child_rss_gb = max(self.peak_child_rss_gb, status.get("VmRSS", 0) / GIB)
                    self.peak_child_hwm_gb = max(self.peak_child_hwm_gb, status.get("VmHWM", 0) / GIB)
            self.samples += 1
            self.stop.wait(5)


def _mem_available() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024 / GIB
    return float("inf")


def _read_pid(pid_file: Path) -> int | None:
    try:
        text = pid_file.read_text().strip()
    except OSError:
        return None
    return int(text) if text.isdigit() else None


def _proc_status(pid: int) -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            key, _, rest = line.partition(":")
            if key in ("VmRSS", "VmHWM"):
                parts = rest.split()
                if parts:
                    values[key] = int(parts[0]) * 1024
    except OSError:
        return {}
    return values


def stage_model(source: Path, dest: Path = SCRATCH) -> dict:
    """Copy the checkpoint out of the dataset mount onto local scratch first.

    The first real attempt read the 26 shards straight from the mount and spent
    11 h 47 m on 16 of them -- 29.25 GB at **0.69 MB/s**, an extrapolated 18.5 h
    for the checkpoint, against a 12 h session cap. The loader reads one tensor
    at a time; the mount pays a round trip for each. One sequential copy is a
    different access pattern, so the copy is timed and reported here: if it is
    not faster, the next run knows staging bought nothing.
    """
    if dest.is_dir() and any(dest.iterdir()):
        log(f"staging: reusing {dest} from an earlier session")
        return {"staged": str(dest), "reused": True}
    dest.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in source.iterdir() if p.is_file())
    total = sum(p.stat().st_size for p in files)
    log(f"staging {len(files)} files, {total / 1e9:.2f} GB, from {source} to {dest}")
    started = time.time()
    copied = 0
    next_report = 5e9
    for path in files:
        target = dest / path.name
        with path.open("rb") as reader, target.open("wb") as writer:
            while True:
                chunk = reader.read(1 << 24)
                if not chunk:
                    break
                writer.write(chunk)
                copied += len(chunk)
                if copied >= next_report:
                    rate = copied / MIB / (time.time() - started)
                    log(f"staged {copied / 1e9:.1f} GB, {rate:.1f} MB/s")
                    next_report += 5e9
        target.chmod(path.stat().st_mode)
    seconds = time.time() - started
    rate = total / MIB / seconds if seconds else 0.0
    log(f"staged {total / 1e9:.2f} GB in {seconds / 60:.1f} min = {rate:.1f} MB/s")
    return {
        "staged": str(dest),
        "reused": False,
        "files": len(files),
        "bytes": total,
        "seconds": seconds,
        "mb_per_s": rate,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    started = time.time()
    code = find_dataset(CODE_FILE)
    model = find_model()
    if code is None:
        raise SystemExit(f"no mounted code dataset carries {CODE_FILE}")
    if model is None:
        raise SystemExit(f"no mounted checkpoint has {MODEL_FILE} for qwen3_5_moe")
    staging = stage_model(model)

    env = {
        "code_dataset": str(code),
        "model_dataset": str(model),
        "model_staging": staging,
        "max_steps": MAX_STEPS,
        "max_length": MAX_LENGTH,
        "torch": None,
        "gpu": None,
        "bf16_supported": None,
    }
    run_dir = WORK / "run"
    shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset = run_dir / "dataset.jsonl"
    dataset.write_text("".join(json.dumps({"text": row}) + "\n" for row in ROWS), encoding="utf-8")
    digest = sha256(dataset)

    spec = {
        "base_model": str(env["model_staging"]["staged"]),
        "dataset": str(dataset),
        "dataset_sha256": digest,
        "dataset_format": "text",
        "text_field": "text",
        "output_dir": str(run_dir / "out"),
        "max_steps": MAX_STEPS,
        "max_length": MAX_LENGTH,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "epochs": 1.0,
        "learning_rate": 0.0002,
        "lr_scheduler_type": "linear",
        "warmup_steps": 0,
        "logging_steps": 1,
        "gradient_checkpointing": True,
        "precision": "bf16",
        "quantization": "none",
        "expert_placement": "cpu-transient-grad-int8",
        "target_preset": "moe_frozen_experts",
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "seed": 7,
        "offline": True,
        "trust_remote_code": False,
        "expert_pinning": "none",
    }
    spec_path = run_dir / "run-spec.json"
    spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    result_path = run_dir / "result.json"

    sys.path.insert(0, str(code / "src"))
    import torch

    env["torch"] = torch.__version__
    env["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
        env["device_total_gb"] = torch.cuda.get_device_properties(0).total_memory / GIB
        env["compute_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
        env["bf16_supported"] = bool(torch.cuda.is_bf16_supported())

    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(code / "src"), str(code / "chowder_batch"), environment.get("PYTHONPATH", "")]
    )
    environment["PYTHONUNBUFFERED"] = "1"
    environment["CHOWDER_NO_GPU"] = ""  # the worker owns the card for the length of the run

    log(f"spec: {MAX_STEPS} step(s) at seq {MAX_LENGTH}, placement cpu-transient-grad-int8")
    log(f"code {code}; model {model}; dataset sha256 {digest[:16]}...")

    pid_file = run_dir / "worker.pid"
    sampler = Sampler(pid_file)
    sampler.start()
    load_started = time.time()
    process = subprocess.Popen(
        [sys.executable, "-m", "chowder.backends.transformers_worker", "--spec", str(spec_path),
         "--result", str(result_path)],
        cwd=str(code),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    pid_file.write_text(str(process.pid), encoding="utf-8")

    first_step_seconds: float | None = None
    tail: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip()
        tail.append(line)
        tail = tail[-40:]
        if "[int8-load]" in line or "it/s]" in line or "s/it]" in line:
            log(line[:200])
        if first_step_seconds is None and ("1/" in line and "s/it]" in line):
            first_step_seconds = time.time() - load_started
    code_status = process.wait()
    total_seconds = time.time() - load_started
    sampler.stop.set()
    sampler.join(timeout=10)

    witness = run_dir / "out" / "placement-witnesses.json"
    adapter_files = sorted(p.name for p in (run_dir / "out").glob("*")) if (run_dir / "out").is_dir() else []
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else None

    report = {
        "environment": env,
        "spec": {k: spec[k] for k in ("max_steps", "max_length", "expert_placement", "target_preset")},
        "dataset_sha256": digest,
        "worker_exit_code": code_status,
        "wall_seconds": total_seconds,
        "first_step_seconds": first_step_seconds,
        "peak_child_rss_gb": sampler.peak_child_rss_gb,
        "peak_child_hwm_gb": sampler.peak_child_hwm_gb,
        "min_host_available_gb": None if sampler.min_available_gb == float("inf") else sampler.min_available_gb,
        "samples": sampler.samples,
        "result": result,
        "witness_written": witness.is_file(),
        "output_files": adapter_files,
        "log_tail": tail,
        "elapsed_total_seconds": time.time() - started,
    }
    (WORK / "real_step.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(
        f"exit {code_status}; wall {total_seconds / 60:.1f} min; peak child RSS "
        f"{sampler.peak_child_rss_gb:.2f} GB; lowest host available {report['min_host_available_gb']}; "
        f"witness written={report['witness_written']}"
    )
    return 0 if code_status == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
