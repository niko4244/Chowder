"""Condition A GPU training pilot (teacher-free distillation, Qwen3-1.7B).

Runs recipe A (recipes/a_sft_supervised.json) through the PRODUCTION training
path -- chowder.backends.transformers_peft.TransformersPeftExecutor -- against
the pilot_v3 accepted SFT data (5,015 rows; conclusion-boundary chunks from
ot3_chunked.jsonl sha256 f2560954..., train split pinned by dataset_sha256 in
the resolved config that the worker records).

Authorization (operator-approved 2026-09-25, recorded here per the recipe's
``authorization.operator_approval_required`` gate):
  * GPU pilot on the RTX 5060 Ti (GPU 0) was explicitly authorized by the
    operator in-session before this script was written.
  * Device exclusivity is re-checked at launch: the target device must host
    no *Chowder* training processes. Unrelated user inference servers
    (llama.cpp / Ollama) are recorded as contention risk, never killed.
  * Free-VRAM floor: at least 8 GiB free on the target device (measured plan:
    ~3.4 GiB frozen base + ~0.3 GiB trainable + activations under gradient
    checkpointing, worst band ~7 GiB peak).

This script must be run from the worktree checkout so that ``import chowder``
resolves to THIS tree (the worker pins the parent's imported source identity
and refuses a mismatched checkout -- see chowder/worker_env.py).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# --- sys.path bootstrap BEFORE any chowder import -------------------------
# The editable install points at ~/Chowder (a different checkout). Prepend
# this worktree's src/ so both this process and the worker (which inherits
# the pin through worker_env) run the code under review.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

DEFAULT_DATA_DIR = Path(r"C:\Users\nikma\chowder_teacher_free\pilot_v3")
DEFAULT_OUTPUT_DIR = Path(r"C:\Users\nikma\chowder_teacher_free\checkpoints\cond_a")
DEFAULT_RECIPE = Path(__file__).resolve().parent / "recipes" / "a_sft_supervised.json"
MIN_FREE_VRAM_GB = 8.0
EXCLUSIVITY_PAT = re.compile(r"chowder", re.IGNORECASE)
LAUNCH_TIMEOUT_SECONDS = 6 * 3600  # 6 h safety net; expected wall ~15-30 min

#: recipe student aliases -> canonical HF repo ids (student_selection.json).
_STUDENT_REPOS = {
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gpu_query() -> str:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total",
         "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def gpu_compute_apps() -> str:
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
         "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def check_device_exclusivity(device_index: int, *, min_free_gb: float | None = None) -> dict:
    """No Chowder process may hold the target device.

    Unrelated inference servers (llama.cpp, Ollama, desktop apps) are allowed
    and recorded as contention risk -- they are the operator's own workload
    and this pilot was authorized to share the device with them.

    ``min_free_gb`` defaults to the training floor; callers with a different
    measured requirement (evaluation loads a freeze-weight 1.7B model and needs
    far less than training) must pass their own floor EXPLICITLY rather than
    bypassing this check.
    """
    floor_gb = MIN_FREE_VRAM_GB if min_free_gb is None else float(min_free_gb)
    apps = gpu_compute_apps()
    rows = [r.strip() for r in apps.splitlines() if r.strip()]
    chowder_rows = [r for r in rows if EXCLUSIVITY_PAT.search(r)]
    if chowder_rows:
        raise RuntimeError(
            "device exclusivity violated: Chowder processes hold a GPU: "
            + "; ".join(chowder_rows)
        )
    # Free VRAM on the target device.
    for row in gpu_query().splitlines():
        cells = [c.strip() for c in row.split(",")]
        if len(cells) >= 4 and cells[0] == str(device_index):
            used_mib = float(cells[2].split()[0])
            total_mib = float(cells[3].split()[0])
            free_gb = (total_mib - used_mib) / 1024.0
            if free_gb < floor_gb:
                raise RuntimeError(
                    f"GPU {device_index} has only {free_gb:.1f} GiB free "
                    f"(floor {floor_gb:.1f} GiB for this workload)"
                )
            return {
                "device_index": device_index,
                "device_name": cells[1],
                "free_vram_gb": round(free_gb, 2),
                "free_vram_floor_gb": round(floor_gb, 2),
                "total_vram_gb": round(total_mib / 1024.0, 2),
                "chowder_processes_on_device": 0,
                "compute_apps_snapshot": rows,
                "contention_note": (
                    "non-Chowder inference servers present and left untouched; "
                    "recorded as contention risk per pilot authorization"
                ),
            }
    raise RuntimeError(f"could not read memory info for GPU {device_index}")


def sha256_directory(path: Path) -> str:
    """Hash a directory tree by relative path and content.

    Mirrors chowder.provenance.sha256_directory exactly (the training worker
    verifies ``parent_adapter`` with that function before load); kept inline
    so config building never imports chowder from the wrong checkout. A unit
    test asserts byte-for-byte equivalence with the real implementation.
    """
    root = Path(path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"adapter directory not found: {root}")
    h = hashlib.sha256()
    for entry in sorted(root.rglob("*"), key=lambda p: p.relative_to(root).as_posix()):
        if entry.is_symlink():
            raise ValueError(f"artifact directory contains unsupported symlink: {entry}")
        if not entry.is_file():
            continue
        rel = entry.relative_to(root).as_posix().encode("utf-8")
        h.update(len(rel).to_bytes(8, "big"))
        h.update(rel)
        h.update(b"\0")
        with entry.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        h.update(b"\0")
    return h.hexdigest()


def condition_b_bindings(recipe: dict, *, parent_adapter: Path | None,
                         general_mix: Path | None) -> dict:
    """Resolve Condition B's continuation and general-mix bindings.

    Returns config fragments consumed by ``build_resolved_config``:
    ``parent_adapter`` (the Condition A adapter dir the run continues from,
    SHA-pinned over the directory tree) and ``replay`` (the general-capability
    dataset plus the raw ratio that realizes the recipe's final-mix fraction:
    the executor samples ceil(primary * ratio) rows, so a fraction f of the
    final mix corresponds to ratio f / (1 - f)).
    """
    if recipe.get("condition") != "B_repair_specialized":
        raise ValueError("condition_b_bindings applies to the B recipe only")
    adapter_dir = (Path(parent_adapter) if parent_adapter else
                   Path(r"C:\Users\nikma\chowder_teacher_free\checkpoints\cond_a\adapter"))
    if not adapter_dir.is_dir():
        raise SystemExit(
            f"parent adapter not found: {adapter_dir} "
            "(Condition B continues from the Condition A adapter)")
    mix_path = (Path(general_mix) if general_mix else
                Path(r"C:\Users\nikma\chowder_teacher_free\pilot_v3\train.jsonl"))
    if not mix_path.is_file():
        raise SystemExit(f"general-mix dataset not found: {mix_path}")
    fraction = float(recipe.get("data", {}).get("general_mix_fraction", 0.0) or 0.0)
    if not 0 <= fraction < 1:
        raise SystemExit(f"general_mix_fraction out of range: {fraction}")
    bindings: dict = {
        "parent_adapter": {"path": str(adapter_dir),
                           "sha256": sha256_directory(adapter_dir)},
    }
    raw_ratio = fraction / (1.0 - fraction) if fraction else 0.0
    if raw_ratio > 0:
        bindings["replay"] = {"dataset": str(mix_path),
                              "sha256": sha256_file(mix_path),
                              "ratio": round(raw_ratio, 6)}
    return bindings


def build_resolved_config(
    recipe: dict,
    data_dir: Path,
    *,
    micro_batch: int | None = None,
    grad_accum: int | None = None,
    replay: dict | None = None,
    parent_adapter: dict | None = None,
) -> dict:
    train_path = data_dir / "train.jsonl"
    if not train_path.is_file():
        raise SystemExit(f"missing training data: {train_path}")
    t = recipe["training"]
    eff_mb = int(micro_batch if micro_batch is not None else t["micro_batch"])
    eff_ga = int(grad_accum if grad_accum is not None else t["gradient_accumulation"])
    repo = _STUDENT_REPOS.get(recipe["student"], recipe["student"])
    config = {
        "seed": int(t["seed"]),
        "backend": {
            "type": "transformers-peft",
            "base_model": repo,
            "revision": recipe["student_revision"],
            "dataset": str(train_path),
            "dataset_sha256": sha256_file(train_path),
            "dataset_format": "chat",
            "messages_field": "messages",
            "max_length": int(t["max_length"]),
            "quantization": "none",
            "precision": "bf16" if t["dtype"] == "bfloat16" else t["dtype"],
            "lora": {
                "r": int(t["lora_r"]),
                "alpha": int(t["lora_alpha"]),
                "dropout": float(t["lora_dropout"]),
                "target_modules": list(t["target_modules"]),
            },
            "training": {
                "epochs": float(t["epochs"]),
                "learning_rate": float(t["learning_rate"]),
                "lr_scheduler_type": t["scheduler"],
                "warmup_ratio": float(t["warmup_ratio"]),
                "batch_size": eff_mb,
                "gradient_accumulation_steps": eff_ga,
                "logging_steps": 10,
                "gradient_checkpointing": bool(t["gradient_checkpointing"]),
                "save_strategy": "no",
            },
            "runtime": {"timeout_seconds": LAUNCH_TIMEOUT_SECONDS},
        },
    }
    if replay is not None:
        # Continual-learning mix: the executor samples ceil(primary * ratio)
        # rows from the replay dataset (seeded by the run seed) and trains on
        # the concatenation, so general data is replayed alongside repair data
        # in one pass. Digest-pinned like the primary dataset. Lives inside
        # the backend section: that is where the executor reads it.
        config["backend"]["replay"] = dict(replay)
    if parent_adapter is not None:
        # Adapter-to-adapter continuation: LoRA weights are initialized from
        # the parent adapter; the worker verifies the directory digest before
        # loading and records it in the run telemetry.
        config["backend"]["parent_adapter"] = dict(parent_adapter)
    return config


def step_entries(telemetry: dict) -> list[dict]:
    """Extract per-step loss entries from the worker's telemetry.

    The worker publishes ``step_log`` as ``{"entries": [...]}`` (one dict per
    logged step); an older/bare list shape is accepted too. Returning ``[]``
    means the worker produced no usable log.
    """
    for key in ("step_log", "step_log_truncated"):
        value = telemetry.get(key)
        if isinstance(value, dict):
            entries = value.get("entries")
            if isinstance(entries, list) and entries:
                return list(entries)
        elif isinstance(value, list) and value:
            return list(value)
    return []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--device", type=int, default=0, help="CUDA device index to pin")
    ap.add_argument("--yes", action="store_true",
                    help="record operator authorization for this GPU launch")
    ap.add_argument("--micro-batch", type=int, default=None,
                    help="override recipe micro_batch (keeps effective batch via grad_accum)")
    ap.add_argument("--grad-accum", type=int, default=None,
                    help="override recipe gradient_accumulation")
    ap.add_argument("--parent-adapter", type=Path, default=None,
                    help="Condition B: adapter dir to continue from "
                         "(default: the Condition A adapter)")
    ap.add_argument("--general-mix", type=Path, default=None,
                    help="Condition B: general-capability JSONL for the "
                         "forgetting-probe mix (default: pilot_v3 train split)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the resolved plan without launching the worker")
    args = ap.parse_args()

    if not args.yes:
        ap.error("--yes is required: records operator authorization for GPU use")

    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    condition = str(recipe.get("condition"))
    if condition not in ("A_supervised_distillation", "B_repair_specialized"):
        ap.error(f"unexpected recipe condition: {condition}")
    is_cond_b = condition == "B_repair_specialized"
    if not is_cond_b and (args.parent_adapter or args.general_mix):
        ap.error("--parent-adapter/--general-mix apply to the Condition B recipe only")
    if is_cond_b and str(args.output).endswith("cond_a"):
        ap.error("refusing to write Condition B into the Condition A output dir")

    exclusivity = check_device_exclusivity(args.device)
    print(f"[exclusivity] OK: {json.dumps({k: v for k, v in exclusivity.items() if k != 'compute_apps_snapshot'}, indent=2)}")

    replay = parent_adapter = None
    if is_cond_b:
        bindings = condition_b_bindings(
            recipe, parent_adapter=args.parent_adapter,
            general_mix=args.general_mix)
        replay = bindings.get("replay")
        parent_adapter = bindings["parent_adapter"]
        print(f"[parent] continuing from {parent_adapter['path']} "
              f"sha256={parent_adapter['sha256'][:16]}...")
        if replay:
            print(f"[replay] general mix {replay['dataset']} "
                  f"ratio={replay['ratio']} (20% of final mix)")

    config = build_resolved_config(
        recipe, args.data_dir,
        micro_batch=args.micro_batch, grad_accum=args.grad_accum,
        replay=replay, parent_adapter=parent_adapter,
    )
    overrides = {}
    if args.micro_batch is not None or args.grad_accum is not None:
        overrides = {
            "micro_batch": config["backend"]["training"]["batch_size"],
            "gradient_accumulation": config["backend"]["training"]["gradient_accumulation_steps"],
            "note": "operator override to fit device memory; effective batch unchanged",
        }
        print(f"[override] {overrides}")
    print(f"[data] train sha256 = {config['backend']['dataset_sha256']}")

    # Pin this run to one device; the worker subprocess inherits the env.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    # Reduce fragmentation risk next to the operator's resident inference
    # servers; the static memory plan (14.9 GiB) assumed a quieter device.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    print(f"[device] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

    if args.dry_run:
        print(json.dumps(config, indent=2))
        return 0

    # Pre-download the pinned base model so the worker never races a partial
    # download (and so network hiccups fail here, not mid-run).
    from huggingface_hub import snapshot_download

    pin = recipe["student_revision"]
    repo = _STUDENT_REPOS.get(recipe["student"], recipe["student"])
    print(f"[model] snapshot_download {repo} @ {pin} ...")
    snapshot_path = snapshot_download(repo_id=repo, revision=pin)
    print(f"[model] cached at {snapshot_path}")

    from chowder.backends.transformers_peft import TransformersPeftExecutor
    from chowder.executors import ExecutionContext
    from chowder.memory import HardwareProfile
    from chowder.models import Experiment, Hypothesis

    args.output.mkdir(parents=True, exist_ok=True)
    hardware = HardwareProfile(
        vram_gb=exclusivity["total_vram_gb"],
        ram_gb=64.0,
        nvme_gb=500.0,
        pcie_gbps=12.0,
        ram_gbps=40.0,
        nvme_gbps=3.0,
        accelerator_vram_gb=(exclusivity["total_vram_gb"],),
    )
    context = ExecutionContext(
        hardware=hardware,
        work_dir=str(args.output),
        seed=int(recipe["training"]["seed"]),
        resolved_config=config,
    )
    if is_cond_b:
        experiment = Experiment(
            experiment_id="cond-b-repair-specialized-continuation",
            parent_id="cond-a-supervised-distillation-pilot",
            hypothesis=Hypothesis(
                observation=(
                    "The Condition A student regressed on general capability "
                    "(MMLU 0.340 -> 0.235; GSM8K 0.575/0.580 -> 0.105; dev PPL "
                    "5.49) while gaining sandbox-verified repair behavior"
                ),
                suspected_cause=(
                    "Repair-specific updates displaced general-capability "
                    "behavior; if forgetting is monotonic in specialization "
                    "pressure, continuing without replay deepens the regression"
                ),
                intervention=(
                    "LoRA continuation from the Condition A adapter (parent "
                    "digest-pinned) on 55 sandbox-verified SWE-smith repair "
                    "examples mixed with 20% general replay from the pilot_v3 "
                    "train split (executor replay mechanism, seed 2026); "
                    "forgetting probed by re-running the same MMLU/GSM8K/PPL "
                    "evals against the Condition A numbers"
                ),
            ),
            config_patch={},
            estimated_gpu_hours=0.5,
        )
    else:
        experiment = Experiment(
            experiment_id="cond-a-supervised-distillation-pilot",
            parent_id=None,
            hypothesis=Hypothesis(
                observation=(
                    "Untouched Qwen3-1.7B produces no verifiable chowder repairs "
                    "in the zero-shot baseline"
                ),
                suspected_cause=(
                    "The student has never been fine-tuned on verified "
                    "reasoning-to-repair traces"
                ),
                intervention=(
                    "LoRA SFT (r=16, completion-only loss) on 5,015 accepted "
                    "license-gated OT3-distilled examples (pilot_v3, "
                    "manifest f2560954); no teacher model is ever loaded"
                ),
            ),
            config_patch={},
            estimated_gpu_hours=0.5,
        )

    executor = TransformersPeftExecutor()
    #: Points seen by THIS process. They are the fallback loss history when the
    #: worker's own step log is unavailable, so a missing worker log can never
    #: be published as "no loss history" (recipes require one).
    observed_steps: list[dict] = []

    def on_progress(event) -> None:
        loss = f" loss={event.loss:.4f}" if event.loss is not None else ""
        print(f"[step {event.step}"
              + (f"/{event.max_steps}" if event.max_steps else "")
              + f"]{loss}", flush=True)
        entry = {"step": event.step, "loss": event.loss}
        learning_rate = getattr(event, "learning_rate", None)
        if learning_rate is not None:
            entry["learning_rate"] = learning_rate
        observed_steps.append(entry)

    executor.bind_progress_callback(on_progress)

    started = time.perf_counter()
    try:
        artifact = executor.run(experiment, context)
    except Exception as exc:  # noqa: BLE001 - surface worker logs before exiting
        print(f"[FAILED] {exc}", file=sys.stderr)
        runs_dir = args.output / ".chowder" / "runs"
        if runs_dir.is_dir():
            newest = max(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime)
            for name in ("stderr.log", "stdout.log"):
                p = newest / name
                if p.is_file():
                    tail = p.read_text(encoding="utf-8", errors="replace")[-4000:]
                    print(f"----- {newest.name}/{name} (tail) -----\n{tail}",
                          file=sys.stderr)
        return 1
    wall = time.perf_counter() - started

    # Locate the run directory the executor created (adapter + result).
    runs_dir = args.output / ".chowder" / "runs"
    run_dirs = sorted(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime)
    run_dir = run_dirs[-1]

    record = {
        "experiment_id": experiment.experiment_id,
        "condition": recipe["condition"],
        "base_model": recipe["student"],
        "base_revision": recipe["student_revision"],
        "dataset": config["backend"]["dataset"],
        "dataset_sha256": config["backend"]["dataset_sha256"],
        "recipe": str(args.recipe),
        "overrides": overrides or None,
    }
    if parent_adapter is not None:
        record["parent_adapter"] = dict(parent_adapter)
    if replay is not None:
        record["general_mix"] = dict(replay)
    record["authorization"] = {
        "operator_authorized": True,
        "authorized_on": "2026-09-25",
        "device_exclusivity_check": exclusivity,
    }
    record["run_dir"] = str(run_dir)
    record["artifact_ref"] = artifact.artifact_ref
    record["wall_seconds"] = round(wall, 1)
    record["telemetry"] = dict(artifact.telemetry)
    record_path = args.output / "run_record.json"
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    # Publish per recipe outputs: adapter + trainer_state + loss history.
    adapter_src = run_dir / "adapter"
    adapter_dst = args.output / "adapter"
    if adapter_dst.exists():
        shutil.rmtree(adapter_dst)
    if adapter_src.is_dir():
        shutil.copytree(adapter_src, adapter_dst)
    result_path = run_dir / "worker-result.json"
    loss_history: list[dict] = []
    loss_source = "none"
    if result_path.is_file():
        worker_result = json.loads(result_path.read_text(encoding="utf-8"))
        shutil.copy2(result_path, args.output / "worker-result.json")
        loss_history = step_entries(worker_result.get("telemetry", {}) or {})
        loss_source = "worker step_log"
    if not loss_history:
        loss_history = observed_steps
        loss_source = "launcher progress callback"
    if not loss_history:
        # Recipes require a loss history; an empty file would look like a
        # successful artifact while carrying no evidence at all.
        if (recipe.get("outputs", {}) or {}).get("loss_history_required"):
            raise SystemExit(
                "loss_history_required is set but no worker step log or progress "
                "observations exist; refusing to publish an empty loss history"
            )
        print("[warn] no loss history available from the worker or the progress "
              "callback", file=sys.stderr)
    (args.output / "loss_history.json").write_text(
        json.dumps(loss_history, indent=2), encoding="utf-8"
    )

    print(f"[done] wall={wall:.0f}s artifact={artifact.artifact_ref}")
    print(f"[done] adapter -> {adapter_dst}")
    print(f"[done] record   -> {record_path}")
    print(f"[done] loss log -> {args.output / 'loss_history.json'} "
          f"({len(loss_history)} entries from {loss_source})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
