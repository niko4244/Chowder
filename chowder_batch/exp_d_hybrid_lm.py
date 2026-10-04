"""Offline runner and artifact collation for the isolated Experiment D model.

This tool never loads a pretrained checkpoint, fetches data, reads a Chowder
campaign registry, initializes a GPU, or writes outside explicitly supplied
paths. ``smoke`` measures random-initialized CPU fixtures only. ``train``
accepts a caller-provided integer-token JSON artifact (``{"token_ids": [...]}``
records) and requires explicit step/token bounds.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import platform
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from chowder.experimental_hybrid_lm import (  # noqa: E402
    ExperimentDError,
    HybridLanguageModel,
    HybridLMConfig,
    load_experiment_d_configs,
)

DEFAULT_CONFIG_DIR = ROOT / "examples" / "experiment_d" / "configs"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _new_output(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _write_exclusive(payload: dict[str, Any], path: str | Path) -> Path:
    output = _new_output(path)
    try:
        handle = output.open("x", encoding="utf-8")
    except FileExistsError:
        raise FileExistsError(f"refusing to overwrite Experiment D artifact: {output}") from None
    with handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return output


def _latency_summary(samples_ms: Sequence[float]) -> dict[str, float]:
    if not samples_ms:
        raise ValueError("at least one measured latency sample is required")
    ordered = sorted(samples_ms)
    p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "mean_ms": statistics.fmean(ordered),
        "median_ms": statistics.median(ordered),
        "p95_ms": ordered[p95_index],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def load_configs(config_dir: str | Path = DEFAULT_CONFIG_DIR) -> dict[str, HybridLMConfig]:
    """Load a config directory through the production validator."""
    return load_experiment_d_configs(config_dir)


def _model_evidence(config_id: str, config_path: Path, model: HybridLanguageModel) -> dict[str, Any]:
    accounting = model.parameter_report()
    return {
        "config_id": config_id,
        "config_sha256": _sha256(config_path),
        "configuration": model.config.to_dict(),
        "configuration_sha256": model.config.digest(),
        "parameters": accounting.to_dict(),
        "flops_decode_context_1": model.flop_estimate(context_length=1).to_dict(),
        "flops_decode_context_64": model.flop_estimate(context_length=64).to_dict(),
    }


def _smoke_one(config_id: str, config_path: Path, config: HybridLMConfig, *, seed: int, sequence_length: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    model = HybridLanguageModel(config).cpu().eval()
    token_ids = torch.randint(0, config.vocab_size, (1, sequence_length), dtype=torch.long)
    with torch.inference_mode():
        started = time.perf_counter()
        output = model(token_ids)
        latency_ms = (time.perf_counter() - started) * 1000.0
        incremental_cache = None
        incremental_logits = []
        for index in range(sequence_length):
            piece = model(
                token_ids[:, index : index + 1],
                cache=incremental_cache,
                use_cache=True,
            )
            incremental_cache = piece.cache
            incremental_logits.append(piece.logits)
    incremental = torch.cat(incremental_logits, dim=1)
    max_abs = float((output.logits - incremental).abs().max())
    if not torch.allclose(output.logits, incremental, atol=1e-5, rtol=1e-5):
        raise ExperimentDError(f"{config_id}: full vs incremental logits differ by {max_abs}")
    route_stats = [
        stat.__dict__
        for stat in output.routing_stats
        if stat is not None
    ]
    return {
        **_model_evidence(config_id, config_path, model),
        "execution": {
            "device": "cpu",
            "seed": seed,
            "batch_size": 1,
            "sequence_length": sequence_length,
            "full_sequence_forward_ms": latency_ms,
            "full_sequence_vs_incremental_max_abs_logit_error": max_abs,
            "routing": route_stats,
            "quality": None,
            "quality_status": "not evaluated",
            "peak_cpu_rss_bytes": None,
            "gpu_vram_bytes": None,
            "measured_hbm_bytes": None,
            "latency_scope": "one tiny random-initialized CPU forward; not a performance benchmark",
        },
    }


def smoke(
    *,
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    output_path: str | Path,
    config_ids: Sequence[str] | None = None,
    seed: int = 123,
    sequence_length: int = 8,
) -> dict[str, Any]:
    """Validate all small configs and write CPU correctness/timing artifacts."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(sequence_length, bool) or not isinstance(sequence_length, int) or sequence_length <= 0:
        raise ValueError("sequence_length must be a positive integer")
    directory = Path(config_dir)
    available = load_configs(directory)
    selected = list(config_ids) if config_ids is not None else sorted(available)
    selected = [Path(config_id).stem for config_id in selected]
    if not selected or set(selected) - available.keys():
        raise ValueError(f"unknown or empty config selection: {selected!r}")
    results = []
    for config_id in selected:
        results.append(
            _smoke_one(
                config_id,
                directory / f"{config_id}.json",
                available[config_id],
                seed=seed,
                sequence_length=sequence_length,
            )
        )
    artifact = {
        "schema_version": 1,
        "experiment": "D",
        "kind": "random_initialized_cpu_smoke",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_revision": None,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "status": "prototype validation only; no training and no quality claim",
        "runs": results,
    }
    _write_exclusive(artifact, output_path)
    return artifact


def _load_token_sequences(path: str | Path, *, vocab_size: int, sequence_length: int) -> tuple[list[list[int]], str]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    rows = payload.get("sequences") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise ValueError("training token artifact must be a nonempty list or {sequences: [...]}")
    validated: list[list[int]] = []
    for index, row in enumerate(rows):
        token_ids = row.get("token_ids") if isinstance(row, dict) else row
        if not isinstance(token_ids, list) or not token_ids:
            raise ValueError(f"sequence {index} must be a nonempty integer list")
        if any(isinstance(token, bool) or not isinstance(token, int) for token in token_ids):
            raise ValueError(f"sequence {index} token ids must be integers")
        if any(token < 0 or token >= vocab_size for token in token_ids):
            raise ValueError(f"sequence {index} token id outside model vocabulary")
        if len(token_ids) < 2:
            continue
        if len(token_ids) < sequence_length:
            repeats = math.ceil(sequence_length / len(token_ids))
            token_ids = (token_ids * repeats)[:sequence_length]
        else:
            token_ids = token_ids[:sequence_length]
        validated.append(token_ids)
    if not validated:
        raise ValueError("token artifact contains no sequence of at least two tokens")
    return validated, _sha256(source)


def train(
    *,
    config_path: str | Path,
    token_data_path: str | Path,
    output_path: str | Path,
    seed: int = 123,
    steps: int = 10,
    sequence_length: int = 32,
    batch_size: int = 1,
    learning_rate: float = 3e-4,
) -> dict[str, Any]:
    """Run a bounded tiny-model CPU pilot on explicit caller-provided token IDs."""
    for name, value in (("seed", seed), ("steps", steps), ("sequence_length", sequence_length), ("batch_size", batch_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)) or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    config_file = Path(config_path)
    config = HybridLMConfig.from_json(config_file)
    if sequence_length > config.max_position_embeddings:
        raise ValueError("sequence_length exceeds config max_position_embeddings")
    token_sequences, data_sha256 = _load_token_sequences(
        token_data_path, vocab_size=config.vocab_size, sequence_length=sequence_length
    )
    torch.manual_seed(seed)
    model = HybridLanguageModel(config).cpu().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate))
    generator = torch.Generator().manual_seed(seed)
    steps_rows = []
    start = time.perf_counter()
    for step in range(steps):
        indices = torch.randint(len(token_sequences), (batch_size,), generator=generator).tolist()
        batch = torch.tensor([token_sequences[index] for index in indices], dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch, labels=batch)
        assert output.loss is not None
        total_loss = output.loss + 0.01 * output.auxiliary_loss
        total_loss.backward()
        optimizer.step()
        steps_rows.append(
            {
                "step": step + 1,
                "task_loss": float(output.loss.detach()),
                "router_auxiliary_loss": float(output.auxiliary_loss.detach()),
                "total_loss": float(total_loss.detach()),
                "tokens_seen": (step + 1) * batch_size * sequence_length,
            }
        )
    elapsed = time.perf_counter() - start
    model.eval()
    artifact = {
        "schema_version": 1,
        "experiment": "D",
        "kind": "tiny_cpu_training_pilot",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_revision": None,
        "config_id": config_file.stem,
        "config_sha256": _sha256(config_file),
        "data_sha256": data_sha256,
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "tokens_seen": steps * batch_size * sequence_length,
        "learning_rate": float(learning_rate),
        "device": "cpu",
        "elapsed_seconds": elapsed,
        "tokens_per_second": steps * batch_size * sequence_length / max(elapsed, 1e-12),
        "step_records": steps_rows,
        "final_loss": steps_rows[-1]["task_loss"],
        "quality_metrics": None,
        "peak_cpu_rss_bytes": None,
        "peak_gpu_vram_bytes": None,
        "gpu_training_cost": None,
        "parameter_accounting": model.parameter_report().to_dict(),
        "flops_per_token_context_1": model.flop_estimate(context_length=1).to_dict(),
        "status": "smoke training only; random initialization and tiny supplied data are not quality evidence",
    }
    _write_exclusive(artifact, output_path)
    return artifact


def collate_registry(
    *,
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    artifact_paths: Sequence[str | Path] = (),
    output_path: str | Path,
) -> dict[str, Any]:
    """Create a new offline registry snapshot; it never mutates a prior registry."""
    configs = load_configs(config_dir)
    entries = []
    for raw_path in artifact_paths:
        path = Path(raw_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("experiment") != "D":
            raise ValueError(f"not an Experiment D result artifact: {path}")
        entries.append({"artifact": str(path), "sha256": _sha256(path), "result": payload})
    registry = {
        "schema_version": 1,
        "experiment": "D",
        "name": "low-active-compute-hybrid-language-model",
        "status": "prototype-not-trained" if not entries else "artifacts-attached",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_revision": None,
        "configurations": [
            {
                "id": config_id,
                "configuration": config.to_dict(),
                "configuration_sha256": config.digest(),
            }
            for config_id, config in sorted(configs.items())
        ],
        "completed_experiments": [
            {
                "artifact": entry["artifact"],
                "sha256": entry["sha256"],
                "kind": entry["result"].get("kind"),
                "config_id": entry["result"].get("config_id"),
            }
            for entry in entries
        ],
        "measurements": [],
        "artifacts": entries,
        "warning": "Design configs and CPU smoke artifacts are not trained-model quality measurements; source revision remains unbound until recorded by an authorized runner.",
    }
    _write_exclusive(registry, output_path)
    return registry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    configs = subparsers.add_parser("configs", help="validate and list small configs")
    configs.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))

    smoke_parser = subparsers.add_parser("smoke", help="run CPU correctness/accounting smoke")
    smoke_parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    smoke_parser.add_argument("--config-id", action="append", help="Config stem or JSON filename; repeatable")
    smoke_parser.add_argument("--out", required=True)
    smoke_parser.add_argument("--seed", type=int, default=123)
    smoke_parser.add_argument("--sequence-length", type=int, default=8)

    train_parser = subparsers.add_parser("train", help="run bounded CPU pilot on explicit token-ID JSON")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--tokens", required=True)
    train_parser.add_argument("--out", required=True)
    train_parser.add_argument("--seed", type=int, default=123)
    train_parser.add_argument("--steps", type=int, default=10)
    train_parser.add_argument("--sequence-length", type=int, default=32)
    train_parser.add_argument("--batch-size", type=int, default=1)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)

    registry_parser = subparsers.add_parser("registry", help="build a new offline registry snapshot")
    registry_parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    registry_parser.add_argument("--artifact", action="append", default=[])
    registry_parser.add_argument("--out", required=True)

    args = parser.parse_args()
    if args.command == "configs":
        result = {
            key: {"configuration": value.to_dict(), "sha256": value.digest()}
            for key, value in sorted(load_configs(args.config_dir).items())
        }
    elif args.command == "smoke":
        result = smoke(
            config_dir=args.config_dir,
            output_path=args.out,
            config_ids=args.config_id,
            seed=args.seed,
            sequence_length=args.sequence_length,
        )
    elif args.command == "train":
        result = train(
            config_path=args.config,
            token_data_path=args.tokens,
            output_path=args.out,
            seed=args.seed,
            steps=args.steps,
            sequence_length=args.sequence_length,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
    else:
        result = collate_registry(
            config_dir=args.config_dir,
            artifact_paths=args.artifact,
            output_path=args.out,
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
