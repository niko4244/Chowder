"""Record and re-verify the completed Condition A adapter's artifacts.

Condition A is the pilot's only finished student run (Qwen3-1.7B + LoRA,
284/284 steps). Its artifacts live outside this checkout, on one machine's
disk: a 25.7 MB adapter plus the launcher's run record, the worker's result
and the recovered loss history. This tool fixes their identity in the repo so
the run can be re-checked later, on this host or another.

``record``
    Hash every published file, re-derive the run summary from the worker's own
    records (never from the narrative in REPORT.md), and write
    ``condition_a_artifacts.json``. It refuses to overwrite an existing record:
    a digest file that changes silently is not evidence.

``verify``
    Recompute the digests and re-derive the summary from whatever is on disk.
    A differing byte, a missing file, or a re-derived number that disagrees
    with the record is a failure. Artifacts absent from this host report
    ``unverifiable`` -- preserved, but never ``ok``.

The ``evaluation`` block in the record carries the measured evaluation
outcome once one has run. The verify gate is evidence-checked, not a frozen
assumption: ``evaluated: true`` is only accepted when the block pins the
comparison artifact (``evaluation.results.artifact``) next to the record and
that file exists, carries decision ``requires_operator_review``, and includes
its recorded leakage section. Training loss is not model quality, and the
record says so in ``evaluation``.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from chowder.teacher_free_evidence import (  # noqa: E402
    CONDITION_A_FORMAT, sha256_file, verify_recorded_files,
)

DEFAULT_RECORD = Path(__file__).resolve().parent / "condition_a_artifacts.json"

#: Files whose bytes identify the run. Paths are relative to the artifacts root.
PINNED_FILES = (
    "adapter/adapter_model.safetensors",
    "adapter/adapter_config.json",
    "adapter/tokenizer.json",
    "adapter/tokenizer_config.json",
    "adapter/chat_template.jinja",
    "run_record.json",
    "worker-result.json",
    "loss_history.json",
)

#: Run values the verifier re-derives from the artifacts and compares with the
#: record. A claim that cannot be re-derived would just be a narrative again.
COMPARED_RUN_KEYS = (
    "base_model", "base_revision", "dataset_sha256",
    "steps_completed", "steps_total", "training_rows", "mean_train_loss",
    "peak_vram_gb", "measured_gpu_hours",
)


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pinned_paths(root: Path) -> list[Path]:
    """Pinned files plus the resolved run spec the worker actually read."""
    root = Path(root)
    paths = [root / name for name in PINNED_FILES]
    run_record = root / "run_record.json"
    if run_record.is_file():
        try:
            run_dir = Path(load_json(run_record)["run_dir"])
        except (KeyError, ValueError, OSError):
            run_dir = None
        if run_dir is not None:
            try:
                spec = run_dir / "run-spec.json"
                spec.relative_to(root)
            except ValueError:  # spec lives outside the recorded root
                spec = None
            if spec is not None:
                paths.append(spec)
    return paths


def derive_summary(root: Path) -> dict:
    """Re-derive the run summary from the worker's own records.

    Everything here comes from files the production training path wrote:
    ``run-spec.json`` (what the worker was told to do), ``worker-result.json``
    (what it measured) and ``loss_history.json`` (the recovered step log).
    """
    root = Path(root)
    run_record = load_json(root / "run_record.json")
    worker_result = load_json(root / "worker-result.json")
    loss_history = load_json(root / "loss_history.json")
    telemetry = worker_result.get("telemetry") or {}
    spec = load_json(Path(run_record["run_dir"]) / "run-spec.json")
    resource_usage = worker_result.get("resource_usage") or {}
    peak = telemetry.get("peak_vram_gb")
    if peak is None:
        peaks = resource_usage.get("peak_vram_gb_by_accelerator") or {}
        peak = max(peaks.values()) if peaks else None
    rows = telemetry.get("training_rows")
    effective_batch = int(spec["batch_size"]) * int(spec["gradient_accumulation_steps"])
    planned = None
    if isinstance(rows, int) and effective_batch:
        planned = math.ceil(rows / effective_batch) * int(math.ceil(float(spec["epochs"])))
    entries = loss_history if isinstance(loss_history, list) else []
    summary = {
        "experiment_id": run_record.get("experiment_id"),
        "condition": run_record.get("condition"),
        "base_model": spec.get("base_model"),
        "base_revision": spec.get("revision"),
        "dataset": spec.get("dataset"),
        "dataset_sha256": spec.get("dataset_sha256"),
        "steps_completed": telemetry.get("global_step"),
        "steps_total": planned,
        "training_rows": rows,
        "mean_train_loss": telemetry.get("train_loss"),
        "peak_vram_gb": peak,
        "measured_gpu_hours": (telemetry.get("lifecycle") or {}).get("measured_gpu_hours"),
        "wall_seconds": run_record.get("wall_seconds"),
        "resolved": {
            key: spec.get(key) for key in
            ("epochs", "batch_size", "gradient_accumulation_steps", "learning_rate",
             "lr_scheduler_type", "warmup_ratio", "max_length", "seed", "precision",
             "gradient_checkpointing", "lora_r", "lora_alpha", "lora_dropout",
             "target_modules")
        },
        "loss_history": {
            "entries": len(entries),
            "truncated": bool((telemetry.get("step_log") or {}).get("truncated")),
            "first": entries[0] if entries else None,
            "last": entries[-1] if entries else None,
        },
        "versions": worker_result.get("versions"),
        "measured": {
            "train_runtime_seconds": telemetry.get("train_runtime_seconds"),
            "measured_seconds": (telemetry.get("lifecycle") or {}).get("measured_seconds"),
            "peak_vram_gb": peak,
            "measured_gpu_hours": (telemetry.get("lifecycle") or {}).get("measured_gpu_hours"),
        },
    }
    return summary


def build_record(root: Path) -> dict:
    root = Path(root)
    missing = [str(p) for p in (root / name for name in PINNED_FILES) if not p.is_file()]
    if missing:
        raise SystemExit("cannot record a run from an incomplete artifact set: "
                         + ", ".join(missing))
    summary = derive_summary(root)
    files = []
    for path in pinned_paths(root):
        if not path.is_file():
            continue
        digest = sha256_file(path)
        files.append({"path": path.relative_to(root).as_posix(), "sha256": digest,
                      "bytes": path.stat().st_size})
    run = {key: summary.get(key) for key in COMPARED_RUN_KEYS}
    run["resolved"] = summary["resolved"]
    run["wall_seconds"] = summary["wall_seconds"]
    run["versions"] = summary["versions"]
    run["measured"] = summary["measured"]
    return {
        "format": CONDITION_A_FORMAT,
        "artifacts_root": str(root),
        "run": run,
        "files": files,
        "loss_history": summary["loss_history"],
        "evaluation": {
            "evaluated": False,
            "note": ("the adapter has never been scored against a baseline; training "
                     "loss is not model quality and no capability claim is made here"),
        },
    }


def verify_record(record: dict, *, root: Path | None = None,
                  manifest_train_sha256: str | None = None) -> dict:
    """Check a record against the artifacts on disk (or report them absent)."""
    failures: list[str] = []
    if record.get("format") != CONDITION_A_FORMAT:
        failures.append(f"unexpected format: {record.get('format')!r}")
    run = record.get("run") if isinstance(record.get("run"), dict) else {}
    if not run:
        failures.append("record carries no run block")
    if run.get("steps_completed") != run.get("steps_total"):
        failures.append(f"run incomplete: {run.get('steps_completed')}/{run.get('steps_total')} steps")
    if not (record.get("loss_history") or {}).get("entries"):
        failures.append("loss history is empty")
    evaluated = (record.get("evaluation") or {})
    if evaluated.get("evaluated") is True:
        # An evaluation claim must point at measurable evidence: a comparison
        # artifact beside this record, carrying the operator-review decision
        # and its recorded leakage checks. Anything else stays a refusal.
        artifact = evaluated.get("results", {}).get("artifact") \
            if isinstance(evaluated.get("results"), dict) else None
        comparison_path = (Path(__file__).resolve().parent / str(artifact)) \
            if artifact else None
        comparison = load_json(comparison_path) if artifact and comparison_path.is_file() else None
        if not isinstance(comparison, dict):
            failures.append("evaluation claims a run but pins no readable "
                            "comparison artifact next to the record")
        else:
            if comparison.get("decision") != "requires_operator_review":
                failures.append(
                    f"comparison decision is {comparison.get('decision')!r}, "
                    "not requires_operator_review")
            leakage = comparison.get("leakage")
            if not isinstance(leakage, dict) or not leakage or not all(
                    isinstance(v, dict) and v.get("ok") is True
                    for v in leakage.values()):
                failures.append("comparison carries no recorded leakage section")
            if evaluated.get("results", {}).get("paired_mean_delta") is None:
                failures.append("evaluation block carries no measured delta")
    dataset_sha = run.get("dataset_sha256")
    if not isinstance(dataset_sha, str) or len(dataset_sha) != 64:
        failures.append("run does not pin the dataset digest")
    elif manifest_train_sha256 and dataset_sha != manifest_train_sha256:
        failures.append("dataset digest differs from the pinned manifest")

    base = Path(root) if root is not None else Path(str(record.get("artifacts_root") or "."))
    check = verify_recorded_files(base, record.get("files"))
    present = bool(check["entries"]) and not check["missing"] and not check["unreadable"]
    derived = None
    if present and not check["mismatches"]:
        try:
            derived = derive_summary(base)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failures.append(f"could not re-derive the summary: {exc}")
        else:
            for key in COMPARED_RUN_KEYS:
                claimed, actual = run.get(key), derived.get(key)
                if claimed != actual:
                    failures.append(f"{key} disagrees: record {claimed!r} vs artifacts {actual!r}")
            for key in ("entries", "first", "last", "truncated"):
                claimed = (record.get("loss_history") or {}).get(key)
                actual = derived["loss_history"].get(key)
                if claimed != actual:
                    failures.append(
                        f"loss_history.{key} disagrees: record {claimed!r} vs artifacts {actual!r}")
    result = {
        "ok": not failures and present and not check["mismatches"],
        "state": ("verified" if (not failures and present and not check["mismatches"])
                  else "mismatch" if (failures or check["mismatches"]) else "unverifiable"),
        "failures": failures,
        "present": present,
        "checked_files": check["checked"],
        "missing_files": check["missing"],
        "mismatched_files": check["mismatches"],
        "derived": derived,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("record", help="hash the artifacts and write the record")
    rec.add_argument("--artifacts", type=Path, required=True)
    rec.add_argument("--out", type=Path, default=DEFAULT_RECORD)
    rec.add_argument("--replace", action="store_true",
                     help="overwrite an existing record (refused by default)")
    ver = sub.add_parser("verify", help="re-check a record against the artifacts")
    ver.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    ver.add_argument("--root", type=Path, default=None,
                     help="artifact root, when the recorded one is not this host's path")
    args = parser.parse_args()

    if args.command == "record":
        out = args.out
        if out.exists() and not args.replace:
            print(f"refusing to overwrite the existing record: {out}", file=sys.stderr)
            return 2
        record = build_record(args.artifacts)
        out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"recorded": str(out), "files": len(record["files"]),
                          "run": {k: record["run"][k] for k in
                                  ("steps_completed", "steps_total", "mean_train_loss")}},
                         indent=2))
        return 0

    record = load_json(args.record)
    result = verify_record(record, root=args.root)
    # The derived block is large and reproducible; the verdict is the point.
    printable = {k: v for k, v in result.items() if k != "derived"}
    print(json.dumps(printable, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
