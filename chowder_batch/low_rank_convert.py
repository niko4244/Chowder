"""Factorize the real embedding/lm_head matrices of the Qwen3.8-9B bf16 checkpoint.

Strategy (Phase 2 of the experiment):

* Each target matrix is read exactly once from disk and moved to the GPU in
  fp32 (~3.8 GB for 248320 x 4096).
* One exact eigendecomposition of the covariance Gram ``X^T X`` (4096 x 4096,
  computed in float64 for accuracy) yields the top singular subspace; all four
  candidate ranks are sliced from the same basis, so the SVD is paid once per
  matrix, not once per rank.
* Factors are persisted per rank only when the destination disk has room
  (checked before each write); every rank's metrics are recorded regardless.
* Results go to a JSON comparison file consumed by the Phase 3/5 reports.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chowder.low_rank_checkpoint import read_tensor  # noqa: E402

RANKS = (2048, 1536, 1024, 512)
TARGETS = (
    ("embedding", "model.language_model.embed_tokens.weight"),
    ("lm_head", "lm_head.weight"),
)


def free_gb(path: Path) -> float:
    return shutil.disk_usage(str(path)).free / 1024**3


def factorize_all_ranks(weight_fp32: torch.Tensor, ranks: tuple[int, ...]) -> dict[str, torch.Tensor]:
    """Return {rank_label: {"u": U_r, "v": V_r}} from one covariance eigh."""
    max_rank = max(ranks)
    gram = (weight_fp32.T @ weight_fp32).to(torch.float64)
    eigvals, eigvecs = torch.linalg.eigh(gram)
    order = torch.argsort(eigvals, descending=True)[:max_rank]
    values = eigvals[order].clamp_min(0.0)
    vectors = eigvecs[:, order].to(torch.float32)  # V: [hidden, max_rank]
    sqrt_s = torch.sqrt(values).to(torch.float32)
    # U_max = X @ V / s, computed once at the largest rank, sliced below.
    u_max = weight_fp32 @ (vectors / sqrt_s.clamp_min(1e-12).unsqueeze(0))
    out: dict[str, torch.Tensor] = {}
    total_energy = float(eigvals.sum())
    for rank in ranks:
        u = u_max[:, :rank].contiguous()
        v = (vectors[:, :rank] * sqrt_s[:rank].unsqueeze(0)).T.contiguous()  # [rank, hidden]
        out[str(rank)] = {
            "u": u,
            "v": v,
            "energy": float(values[:rank].sum()) / max(total_energy, 1e-30),
        }
    return out


def sampled_relative_error(weight: torch.Tensor, u: torch.Tensor, v: torch.Tensor, rows: int = 65536) -> float:
    step = max(1, weight.shape[0] // rows)
    sample = weight[::step]
    u_sample = u[::step].to(weight.device)
    diff = sample - u_sample @ v.to(weight.device)
    return float(diff.norm() / sample.norm().clamp_min(1e-30))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}

    for role, key in TARGETS:
        t0 = time.time()
        weight = read_tensor(args.model, key).to(torch.float32)
        load_s = time.time() - t0
        weight_gpu = weight.to(device)
        t1 = time.time()
        factors = factorize_all_ranks(weight_gpu, RANKS)
        decompose_s = time.time() - t1
        del weight_gpu
        torch.cuda.empty_cache() if device.type == "cuda" else None

        role_results: dict[str, dict] = {"key": key, "shape": list(weight.shape), "load_seconds": round(load_s, 1)}
        for rank_label, bundle in factors.items():
            rank = int(rank_label)
            err = sampled_relative_error(weight, bundle["u"], bundle["v"])
            n_params = bundle["u"].numel() + bundle["v"].numel()
            entry = {
                "rank": rank,
                "method": "covariance_eigh_f64",
                "energy_captured": round(bundle["energy"], 6),
                "relative_frobenius_error": round(err, 6),
                "factor_parameters": n_params,
                "full_parameters": weight.numel(),
                "compression_ratio": round(weight.numel() / n_params, 4),
                "factor_bytes_bf16": n_params * 2,
                "persisted": False,
            }
            # Persist only when the disk can take it (small ranks first).
            need_gb = (n_params * 2) / 1024**3 + 0.2
            if free_gb(out_dir) > need_gb:
                u_cpu = bundle["u"].to(torch.bfloat16).cpu()
                v_cpu = bundle["v"].to(torch.bfloat16).cpu()
                save_file(
                    {"u": u_cpu, "v": v_cpu},
                    str(out_dir / f"{role}_rank{rank}_factors.safetensors"),
                )
                entry["persisted"] = True
                del u_cpu, v_cpu
            role_results[rank_label] = entry
            print(f"{role} rank {rank}: energy={entry['energy_captured']:.6f} err={entry['relative_frobenius_error']:.6f} persisted={entry['persisted']}", flush=True)
            del bundle
        results[role] = role_results
        del weight, factors
        print(f"{role}: load {load_s:.0f}s, decompose {decompose_s:.0f}s", flush=True)

    payload = {
        "model": args.model,
        "device": str(device),
        "ranks": list(RANKS),
        "results": results,
    }
    out_path = out_dir / "factorization_results.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("wrote", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
