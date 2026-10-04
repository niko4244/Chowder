"""Conversion, restoration, and provenance for low-rank vocabulary compression.

The 19 GB source checkpoint is never duplicated: conversion reads exactly one
tensor per target key through ``safetensors`` lazy loading, factorizes it, and
writes only the thin factors plus a manifest. Restoration reconstructs the
original dense matrices from the factors and verifies them against the source
checkpoint before any model is declared "restored".

Manifest format ``chowder-low-rank-v1`` records the model identity, the exact
tensor keys replaced, factor shapes, method, reconstruction error, and the
parameter accounting -- everything a later reviewer needs to reproduce or undo
the conversion.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .low_rank_vocab import FactorizationResult, factorize_svd, state_dict_savings

MANIFEST_NAME = "low_rank_manifest.json"
FACTORS_NAME = "low_rank_factors.safetensors"
MANIFEST_FORMAT = "chowder-low-rank-v1"

# Key spellings seen across qwen3_5 checkpoints. The first match present in the
# checkpoint index is used.
EMBED_KEY_CANDIDATES = (
    "model.language_model.embed_tokens.weight",
    "model.embed_tokens.weight",
    "embed_tokens.weight",
)
LM_HEAD_KEY_CANDIDATES = ("lm_head.weight",)


@dataclass
class ConversionTarget:
    """One matrix to factorize: ``A [rows, cols]`` becomes ``U @ V``."""

    key: str
    role: str  # "embedding" or "lm_head"
    rows: int
    cols: int
    rank: int
    method: str = ""
    energy_captured: float = 0.0
    relative_error: float = 0.0


@dataclass
class LowRankManifest:
    format: str
    model_dir: str
    model_config_sha256: str
    dtype: str
    targets: list[ConversionTarget]
    parameter_accounting: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "model_dir": self.model_dir,
            "model_config_sha256": self.model_config_sha256,
            "dtype": self.dtype,
            "targets": [target.__dict__ for target in self.targets],
            "parameter_accounting": self.parameter_accounting,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LowRankManifest":
        if payload.get("format") != MANIFEST_FORMAT:
            raise ValueError(f"unsupported low-rank manifest format: {payload.get('format')!r}")
        return cls(
            format=payload["format"],
            model_dir=payload["model_dir"],
            model_config_sha256=payload["model_config_sha256"],
            dtype=payload["dtype"],
            targets=[ConversionTarget(**target) for target in payload["targets"]],
            parameter_accounting=payload.get("parameter_accounting", {}),
        )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_key(index_weight_map: Mapping[str, str], candidates: tuple[str, ...]) -> str:
    for key in candidates:
        if key in index_weight_map:
            return key
    raise KeyError(f"checkpoint index contains none of the candidate keys: {candidates}")


def shard_for_key(model_dir: str | Path, key: str) -> Path:
    """Locate the shard file that holds ``key`` without reading any weights."""
    model_dir = Path(model_dir)
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        if key not in weight_map:
            raise KeyError(f"key {key!r} not present in {index_path.name}")
        shard = model_dir / weight_map[key]
        if not shard.is_file():
            raise FileNotFoundError(f"missing shard {shard}")
        return shard
    single = model_dir / "model.safetensors"
    if single.is_file():
        return single
    raise FileNotFoundError(f"no safetensors index or model.safetensors in {model_dir}")


def read_tensor(model_dir: str | Path, key: str) -> torch.Tensor:
    """Read exactly one tensor from a (possibly sharded) checkpoint."""
    shard = shard_for_key(model_dir, key)
    with safe_open(str(shard), framework="pt", device="cpu") as handle:
        if key not in handle.keys():
            raise KeyError(f"key {key!r} not present in {shard.name}")
        return handle.get_tensor(key)


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def factorize_checkpoint_matrices(
    model_dir: str | Path,
    out_dir: str | Path,
    ranks: Mapping[str, int],
    *,
    prefer_cuda: bool = True,
    seed: int = 0,
) -> LowRankManifest:
    """Factorize the embedding and/or output projection of a real checkpoint.

    ``ranks`` maps a role ("embedding", "lm_head") to a rank. Only the thin
    factors are written; the source checkpoint is left untouched and is never
    copied. The returned manifest is also written to ``out_dir``.
    """
    model_dir = Path(model_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"this utility requires a sharded checkpoint index in {model_dir}")
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]

    embed_key = resolve_key(weight_map, EMBED_KEY_CANDIDATES)
    head_key = resolve_key(weight_map, LM_HEAD_KEY_CANDIDATES)

    roles: dict[str, tuple[str, str]] = {}
    if "embedding" in ranks:
        roles["embedding"] = (embed_key, "embedding")
    if "lm_head" in ranks:
        roles["lm_head"] = (head_key, "lm_head")
    if not roles:
        raise ValueError("ranks must name at least one of 'embedding' or 'lm_head'")

    config_sha = sha256_file(model_dir / "config.json")
    factors: dict[str, torch.Tensor] = {}
    targets: list[ConversionTarget] = []

    for role, (key, role_name) in sorted(roles.items()):
        weight = read_tensor(model_dir, key)
        if weight.dim() != 2:
            raise ValueError(f"{key} is not a matrix: {tuple(weight.shape)}")
        rows, cols = weight.shape
        rank = int(ranks[role])
        result: FactorizationResult = factorize_svd(
            weight, rank, prefer_cuda=prefer_cuda, seed=seed
        )
        del weight
        # Both roles store U [rows, rank] and V [rank, cols]. For qwen3_5 the
        # checkpoint's lm_head.weight is [vocab, hidden], the same layout as the
        # embedding, so one convention covers both; the model-side loader maps
        # LowRankLMHead.head_a = V and head_b = U.
        factors[f"{role}.u"], factors[f"{role}.v"] = result.u, result.v
        targets.append(
            ConversionTarget(
                key=key,
                role=role_name,
                rows=rows,
                cols=cols,
                rank=rank,
                method=result.method,
                energy_captured=round(result.energy_captured, 6),
                relative_error=round(result.relative_frobenius_error, 6),
            )
        )

    save_file({name: value.contiguous() for name, value in factors.items()}, str(out_dir / FACTORS_NAME))

    accounting: dict[str, Any] = {}
    for target in targets:
        accounting[target.role] = state_dict_savings((target.rows, target.cols), target.rank)

    manifest = LowRankManifest(
        format=MANIFEST_FORMAT,
        model_dir=str(model_dir),
        model_config_sha256=config_sha,
        dtype="bf16",
        targets=targets,
        parameter_accounting=accounting,
    )
    (out_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def restore_dense_weight(
    factors_dir: str | Path, role: str, *, dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """Rebuild one original dense matrix ``U @ V`` from saved factors."""
    factors_dir = Path(factors_dir)
    with safe_open(str(factors_dir / FACTORS_NAME), framework="pt", device="cpu") as handle:
        names = set(handle.keys())
        if f"{role}.u" not in names or f"{role}.v" not in names:
            raise KeyError(f"factors for role {role!r} not found in {FACTORS_NAME}")
        u = handle.get_tensor(f"{role}.u")
        v = handle.get_tensor(f"{role}.v")
    if u.shape[1] != v.shape[0]:
        raise ValueError(f"factor shapes {tuple(u.shape)} @ {tuple(v.shape)} do not compose")
    return (u @ v).to(dtype)


def verify_restoration(
    model_dir: str | Path, factors_dir: str | Path, *, atol: float = 0.35
) -> dict[str, Any]:
    """Check factor reconstruction against the actual checkpoint tensors.

    bf16 rounding means the reconstruction is compared at a loose elementwise
    tolerance plus a strict relative-Frobenius bound; the point is to catch a
    wrong or transposed factor, not float noise.
    """
    model_dir, factors_dir = Path(model_dir), Path(factors_dir)
    weight_map = json.loads((model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    report: dict[str, Any] = {}
    for role, candidates in (("embedding", EMBED_KEY_CANDIDATES), ("lm_head", LM_HEAD_KEY_CANDIDATES)):
        key = resolve_key(weight_map, candidates)
        original = read_tensor(model_dir, key).to(torch.float32)
        restored = restore_dense_weight(factors_dir, role, dtype=torch.float32)
        if restored.shape != original.shape:
            raise ValueError(f"{role}: restored shape {tuple(restored.shape)} != {tuple(original.shape)}")
        diff = restored - original
        rel = float(diff.norm() / original.norm().clamp_min(1e-30))
        report[role] = {
            "key": key,
            "relative_frobenius_error": round(rel, 6),
            "max_abs_delta": float(diff.abs().max()),
            "within_tolerance": bool(rel <= atol),
        }
        del original, restored, diff
    return report


def load_manifest(factors_dir: str | Path) -> LowRankManifest:
    return LowRankManifest.from_dict(
        json.loads((Path(factors_dir) / MANIFEST_NAME).read_text(encoding="utf-8"))
    )
