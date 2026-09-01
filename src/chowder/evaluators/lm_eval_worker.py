from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping

from .lm_eval import LmEvalSpec


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _resolve_device(torch: Any, requested: str) -> str:
    requested = requested.strip().lower()
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"{requested} requested but CUDA is unavailable")
    return requested


def _model_args(spec: LmEvalSpec) -> dict[str, Any]:
    args: dict[str, Any] = {
        "pretrained": spec.base_model,
        "peft": spec.adapter_dir,
        "trust_remote_code": False,
    }
    if spec.revision is not None:
        args["revision"] = spec.revision
    dtype = {
        "auto": "auto",
        "bf16": "bfloat16",
        "fp16": "float16",
        "fp32": "float32",
    }[spec.precision]
    args["dtype"] = dtype
    if spec.quantization == "4bit":
        args["load_in_4bit"] = True
    return args


def _extract_metrics(raw_results: Mapping[str, Any], metric_map: Mapping[str, str]) -> dict[str, float]:
    task_results = raw_results.get("results")
    if not isinstance(task_results, Mapping):
        raise RuntimeError("lm-eval response is missing results mapping")
    metrics: dict[str, float] = {}
    for target, source in metric_map.items():
        task, metric = source.split(":", 1)
        task_payload = task_results.get(task)
        if not isinstance(task_payload, Mapping):
            raise RuntimeError(f"lm-eval task {task!r} is missing from results")
        if metric not in task_payload:
            raise RuntimeError(f"lm-eval metric {metric!r} is missing from task {task!r}")
        value = task_payload[metric]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError(f"lm-eval metric {task}:{metric} is not numeric")
        metrics[target] = float(value)
    return metrics


def evaluate(spec: LmEvalSpec) -> dict[str, Any]:
    try:
        import lm_eval
        import torch
    except ImportError as exc:
        raise RuntimeError("lm-eval dependencies are missing; install chowder-ai[eval]") from exc

    if spec.trust_remote_code:
        raise RuntimeError("trust_remote_code is disabled")
    device = _resolve_device(torch, spec.device)
    if spec.quantization == "4bit" and not device.startswith("cuda"):
        raise RuntimeError("4-bit lm-eval requires a CUDA device")

    raw = lm_eval.simple_evaluate(
        model="hf",
        model_args=_model_args(spec),
        tasks=list(spec.tasks),
        num_fewshot=spec.num_fewshot,
        batch_size=spec.batch_size,
        device=device,
        limit=spec.limit,
        log_samples=False,
        apply_chat_template=spec.apply_chat_template,
        fewshot_as_multiturn=spec.fewshot_as_multiturn,
        random_seed=spec.seed,
        numpy_random_seed=spec.seed,
        torch_random_seed=spec.seed,
        fewshot_random_seed=spec.seed,
    )
    if raw is None:
        raise RuntimeError("lm-eval returned no result on this process")
    metrics = _extract_metrics(raw, spec.metric_map)
    raw_path = Path(spec.output_dir) / "lm-eval-raw.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(raw, sort_keys=True, indent=2, default=str) + "\n", encoding="utf-8")
    return {
        "metrics": metrics,
        "raw_results_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "runtime": {
            "device": device,
            "gpu_count": 1 if device.startswith("cuda") else 0,
        },
        "versions": {
            "lm-eval": _package_version("lm-eval"),
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
            "peft": _package_version("peft"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    raw = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    raw["tasks"] = tuple(raw["tasks"])
    spec = LmEvalSpec(**raw)
    result = evaluate(spec)
    Path(args.result).write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
