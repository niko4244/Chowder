"""Prune Qwen3.6-35B-A3B routed experts from REAP observations (or a control).

Per decoder layer, keep the K experts with the highest score and slice the
three tensors that index experts -- ``mlp.experts.gate_up_proj`` [E, 2I, H],
``mlp.experts.down_proj`` [E, H, I] and the router ``mlp.gate.weight`` [E, H] --
to those K rows, preserving their original order. Everything else is copied
byte-for-byte except the MTP head, which is dropped (both arms of every
comparison are converted with ``--no-mtp``, so neither carries it).

Methods:
  reap       score = sum(g*||f||) / count        (arXiv 2510.13999)
  frequency  score = count                        (EAN-style baseline)
  random     K experts per layer, seeded          (negative control)

Usage:
  python chowder_batch/reap_prune.py --model-dir <bf16 dir> [--weights-dir <shards>] \
      --observations <observe out> --keep 128 --method reap --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# The preregistered preflight prints a direct-script invocation. Ensure its
# package imports remain available without relying on a caller's PYTHONPATH.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

EXPERT_KEY = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.(experts\.gate_up_proj|experts\.down_proj|gate\.weight)$")
COPY_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja",
              "generation_config.json", "special_tokens_map.json", "preprocessor_config.json",
              "video_preprocessor_config.json", "LICENSE")


def _verify_registered_shard(path: Path, parent: Path, *, size: int, sha256: str) -> str:
    """Verify live original BF16 payload before the model build opens it.

    This intentionally streams the full original shard. The small metadata-only
    preflight does NOT give byte-level attestation of the current weights.
    """
    if (path.is_symlink() or not path.is_file()
            or path.resolve().parent != parent.resolve()
            or path.stat().st_size != size):
        raise ValueError(f"unsafe or wrong-size original BF16 shard: {path.name}")
    from chowder_batch.reap_hidden_checkpoint import sha256_file
    actual = sha256_file(path)
    if actual != sha256:
        raise ValueError(f"original BF16 shard differs from pinned upstream: {path.name}")
    return actual


def kept_experts(observations: Path, n_layers: int, n_experts: int, keep: int, method: str, seed: int) -> dict[int, list[int]]:
    kept: dict[int, list[int]] = {}
    generator = torch.Generator().manual_seed(seed)
    for layer in range(n_layers):
        if method == "random":
            order = torch.randperm(n_experts, generator=generator)
        else:
            stats = torch.load(observations / f"layer-{layer:02d}.pt", map_location="cpu", weights_only=True)
            if not isinstance(stats, dict) or not all(k in stats for k in ("count", "reap_sum")):
                raise ValueError(f"observation layer {layer}: missing expert statistics")
            for name in ("count", "reap_sum"):
                value = stats[name]
                if (not isinstance(value, torch.Tensor) or value.ndim != 1
                        or value.numel() != n_experts or not torch.isfinite(value).all()
                        or (value < 0).any()):
                    raise ValueError(f"observation layer {layer}: invalid {name} shape or values")
            count = stats["count"].double()
            if not (count > 0).any():
                raise ValueError(f"observation layer {layer}: no expert routing measured")
            score = stats["reap_sum"].double() / count.clamp(min=1) if method == "reap" else count
            score = torch.where(count > 0, score, torch.full_like(score, -1.0))  # never-routed experts go first
            order = torch.argsort(score, descending=True, stable=True)
        kept[layer] = sorted(order[:keep].tolist())
    return kept


def _indexed_shards(index: dict, n_layers: int) -> dict[str, set[str]]:
    """Require the source safetensors index to enumerate precisely its weight tensors."""
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("source safetensors index has no tensor map")
    expert_keys = {
        f"model.language_model.layers.{i}.mlp.{name}"
        for i in range(n_layers)
        for name in ("experts.gate_up_proj", "experts.down_proj", "gate.weight")
    }
    missing = expert_keys - set(weight_map)
    if missing:
        raise ValueError(f"source index lacks expected expert tensors: {sorted(missing)[:3]}")
    by_shard: dict[str, set[str]] = {}
    for key, filename in weight_map.items():
        if (not isinstance(key, str) or not isinstance(filename, str) or not filename
                or Path(filename).name != filename or "/" in filename or "\\" in filename
                or filename in {".", ".."}):
            raise ValueError("source index has an unsafe tensor name or shard path")
        if not key.startswith("mtp."):
            by_shard.setdefault(filename, set()).add(key)
    return by_shard


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--weights-dir", default="")
    parser.add_argument("--observations", default="")
    parser.add_argument("--keep", type=int, required=True)
    parser.add_argument("--method", choices=("reap", "frequency", "random"), default="reap")
    parser.add_argument("--selection", default="",
                        help="source-bound REAP128_v2 selection.json (optional for older arms)")
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    weights_dir = Path(args.weights_dir or args.model_dir)
    out = Path(args.out)
    resolved_out = out.resolve()
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing model directory: {out}")
    if not out.parent.is_dir():
        raise SystemExit(f"model output parent does not exist: {out.parent}")
    source_directories = [model_dir.resolve(), weights_dir.resolve()]
    if args.observations:
        source_directories.append(Path(args.observations).resolve())
    if any(resolved_out.is_relative_to(source) or source.is_relative_to(resolved_out)
           for source in source_directories):
        raise SystemExit("model output must be disjoint from model, weights and observations")
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    text = config["text_config"]
    n_layers, n_experts = int(text["num_hidden_layers"]), int(text["num_experts"])
    if not 0 < args.keep < n_experts or args.keep < int(text["num_experts_per_tok"]):
        raise SystemExit(f"--keep must be in [{text['num_experts_per_tok']}, {n_experts})")
    if args.method != "random" and not args.observations:
        raise SystemExit("--observations is required for reap/frequency")
    registered_original = False
    if args.method != "random":
        evidence = Path(args.observations) / "observation.json"
        if n_layers == 40 and n_experts == 256 and not evidence.is_file():
            raise ValueError("original 40x256 REAP pruning requires observation.json")
        if evidence.is_file():
            observed = json.loads(evidence.read_text(encoding="utf-8"))
            if n_layers == 40 and n_experts == 256 and observed.get("n_seqs") not in (384, 704):
                raise ValueError("original 40x256 REAP observation must match a frozen calibration")
            if (n_layers == 40 and n_experts == 256 and args.keep == 128
                    and args.method == "reap" and observed.get("n_seqs") == 704
                    and not args.selection):
                raise ValueError("registered REAP128_v2 build requires --selection manifest")
            if n_layers == 40 and n_experts == 256 and observed.get("n_seqs") == 384:
                from chowder_batch.reap_build_preflight import FROZEN_V1_SHA256
                if observed.get("calibration_sha256") != FROZEN_V1_SHA256:
                    raise ValueError("original 40x256 v1 REAP observation has wrong frozen calibration")
            registered_original = n_layers == 40 and observed.get("n_seqs") == 704
            if registered_original or "checkpoint_receipt" in observed or "hidden_storage" in observed:
                from chowder_batch.reap_hidden_checkpoint import validate_observation_receipt
                validate_observation_receipt(Path(args.observations), observed,
                                             expected_generation=n_layers,
                                             require_registered_original=registered_original)
            if registered_original:
                from chowder_batch.reap_original_bf16_pin import validate_registered_source_metadata
                validate_registered_source_metadata(model_dir.resolve(), weights_dir.resolve())
                source = observed["checkpoint_receipt"]["source_identity"]
                if (source.get("model_dir") != str(model_dir.resolve())
                        or source.get("weights_dir") != str(weights_dir.resolve())):
                    raise ValueError("registered BF16 observation source differs from pruning inputs")
    kept = kept_experts(Path(args.observations), n_layers, n_experts, args.keep, args.method, args.seed)
    selection_sha256 = None
    if args.selection:
        if args.method != "reap" or n_layers != 40 or n_experts != 256 or args.keep != 128:
            raise ValueError("--selection is only for the registered REAP128_v2 build")
        if not observed.get("checkpoint_receipt") or observed.get("n_seqs") != 704:
            raise ValueError("--selection requires the registered 704-row sealed BF16 observation")
        from chowder_batch.reap_rank_experts import validate_selection
        selection_path = Path(args.selection)
        selection_bytes = selection_path.read_bytes()
        selection = json.loads(selection_bytes)
        validate_selection(selection, model_dir=model_dir, weights_dir=weights_dir,
                           observations=Path(args.observations), kept=kept, out=out)
        selection_sha256 = hashlib.sha256(selection_bytes).hexdigest()

    index = json.loads((model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shards = _indexed_shards(index, n_layers)
    pinned_source_shards = {}
    if args.method != "random" and registered_original:
        from chowder_batch.reap_original_bf16_pin import load_original_pin
        # No staging is created until the live 26 shards have been verified.
        pinned = load_original_pin()["weight_shards"]
        for filename, expected in sorted(pinned.items()):
            pinned_source_shards[filename] = _verify_registered_shard(
                weights_dir / filename, weights_dir,
                size=expected["size"], sha256=expected["sha256"],
            )
    # A completed directory must be a single published artifact, never a partly
    # written large checkpoint. The temporary directory is created by this run
    # and cannot collide with another owner's model directory.
    with tempfile.TemporaryDirectory(prefix=f".{out.name}.building-", dir=out.parent) as temp_dir:
        stage = Path(temp_dir)
        weight_map: dict[str, str] = {}
        total_bytes = 0
        for shard, expected in sorted(shards.items()):
            tensors: dict[str, torch.Tensor] = {}
            with safe_open(str(weights_dir / shard), framework="pt", device="cpu") as handle:
                actual = {key for key in handle.keys() if not key.startswith("mtp.")}
                if actual != expected:
                    raise ValueError(
                        f"{shard}: tensor census mismatches index "
                        f"(missing={sorted(expected - actual)[:3]}, extra={sorted(actual - expected)[:3]})"
                    )
                for key in sorted(actual):
                    tensor = handle.get_tensor(key)
                    match = EXPERT_KEY.fullmatch(key)
                    if match:
                        rows = torch.tensor(kept[int(match.group(1))])
                        if tensor.shape[0] != n_experts:
                            raise ValueError(f"{key}: leading dim {tensor.shape[0]} != {n_experts}")
                        tensor = tensor.index_select(0, rows).contiguous()
                    tensors[key] = tensor
            if tensors:
                save_file(tensors, str(stage / shard), metadata={"format": "pt"})
                for key, tensor in tensors.items():
                    weight_map[key] = shard
                    total_bytes += tensor.numel() * tensor.element_size()
            print(f"{shard}: {len(tensors)} tensors", flush=True)

        (stage / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": total_bytes}, "weight_map": weight_map}, indent=2),
            encoding="utf-8",
        )
        text["num_experts"] = args.keep
        text["mtp_num_hidden_layers"] = 0
        (stage / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        for name in COPY_FILES:
            if (model_dir / name).exists():
                shutil.copy2(model_dir / name, stage / name)
        observation_dir = Path(args.observations)
        record = {
            "method": args.method, "keep": args.keep, "of": n_experts, "seed": args.seed,
            "observations": str(args.observations),
            "observation_digest": hashlib.sha256(
                (observation_dir / "observation.json").read_bytes()).hexdigest()
            if args.observations and (observation_dir / "observation.json").exists() else None,
            "observation_layer_sha256": {
                str(i): hashlib.sha256((observation_dir / f"layer-{i:02d}.pt").read_bytes()).hexdigest()
                for i in range(n_layers)
            } if args.method != "random" else {},
            "selection_sha256": selection_sha256,
            "verified_live_upstream_weight_sha256": pinned_source_shards,
            "kept_experts": {str(k): v for k, v in kept.items()},
            "total_bytes": total_bytes,
        }
        (stage / "reap-prune.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        if out.exists():
            raise SystemExit(f"destination created by another process; refusing overwrite: {out}")
        stage.rename(out)
    print(json.dumps({k: record[k] for k in ("method", "keep", "of", "total_bytes")}), flush=True)


if __name__ == "__main__":
    main()
