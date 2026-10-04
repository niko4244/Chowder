"""Evaluate factorized heads on REAL hidden states from the teacher cache.

Matrix-level probes use embedding rows as a stand-in for the residual stream;
this script replaces that surrogate with the teacher's actual pre-head hidden
states, so the reported KL / top-1 agreement is what the factorized output
projection would really produce on real text. Positions are split train/val
so the recovery pilot can later be judged on untouched positions.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

RANKS = (2048, 1536, 1024, 512)


def kl_metrics(teacher_logits: torch.Tensor, student_logits: torch.Tensor, *, chunk: int = 128, temperature: float = 1.0) -> dict[str, float]:
    kl_parts = []
    correct = 0
    recall_sum = 0.0
    total = teacher_logits.shape[0]
    ce_sum = 0.0
    with torch.no_grad():
        for start in range(0, total, chunk):
            t = teacher_logits[start : start + chunk]
            s = student_logits[start : start + chunk]
            p = torch.softmax(t / temperature, dim=-1)
            q = torch.softmax(s / temperature, dim=-1)
            kl_parts.append(torch.xlogy(p, p / q.clamp_min(1e-12)).sum(-1))
            correct += float((t.argmax(-1) == s.argmax(-1)).sum())
            t5 = t.topk(5, dim=-1).indices.tolist()
            s5 = s.topk(5, dim=-1).indices.tolist()
            recall_sum += sum(len(set(a) & set(b)) / 5.0 for a, b in zip(t5, s5))
            # cross-entropy of the student distribution at the teacher's argmax
            ce_sum += float(torch.nn.functional.cross_entropy(s, t.argmax(-1), reduction="sum"))
    kl = torch.cat(kl_parts)
    return {
        "kl_mean_nats": float(kl.mean()),
        "kl_p95_nats": float(kl.quantile(0.95)),
        "top1_agreement": correct / total,
        "top5_recall": recall_sum / total,
        "ce_at_teacher_argmax": ce_sum / total,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factors", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--val-positions", type=int, default=60)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensors = load_file(args.cache)
    hidden = tensors["hidden"].to(device)
    logits = tensors["logits"].to(device)
    n = hidden.shape[0]
    split = n - args.val_positions
    print(f"{n} positions: {split} train / {n - split} val", flush=True)

    report: dict[str, dict] = {"n_positions": n, "val_positions": n - split, "ranks": {}}
    for rank in RANKS:
        path = Path(args.factors) / f"lm_head_rank{rank}_factors.safetensors"
        with safe_open(str(path), framework="pt") as handle:
            u = handle.get_tensor("u").to(torch.float32).to(device)  # [vocab, rank]
            v = handle.get_tensor("v").to(torch.float32).to(device)  # [rank, hidden]

        def project(h: torch.Tensor, u=u, v=v) -> torch.Tensor:
            out = []
            for start in range(0, h.shape[0], 128):
                out.append((h[start : start + 128] @ v.T) @ u.T)
            return torch.cat(out)

        entry = {
            "train": kl_metrics(logits[:split], project(hidden[:split])),
            "val": kl_metrics(logits[split:], project(hidden[split:])),
        }
        report["ranks"][str(rank)] = entry
        vrow = entry["val"]
        print(
            f"rank {rank}: val KL={vrow['kl_mean_nats']:.4f} top1={vrow['top1_agreement']:.4f} "
            f"top5={vrow['top5_recall']:.4f} CE@teacher={vrow['ce_at_teacher_argmax']:.4f}",
            flush=True,
        )
        del u, v
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
