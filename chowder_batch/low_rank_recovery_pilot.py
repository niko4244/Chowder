"""Phase-4 recovery pilot: can cheap training rescue the factorized head?

Starts from the identical SVD-initialized factors and compares:

* ``recon`` -- no training at all (the reconstruction-only initialization);
* ``distill`` -- fine-tuning the factors on cached teacher data with a
  KL(teacher || student) objective over the *train* split of cached positions.

Both are scored on the untouched val split. The teacher is never re-run: the
pilot reads exclusively from the Phase-3 cache, which is the point of caching
teacher targets.

Scope note: the cache's hidden states were produced by the *original* embedding,
so this pilot isolates recovery of the output projection. A production candidate
would also factorize the embedding, shifting layer-0 inputs and compounding the
damage; if head-only recovery cannot recover, the compounded candidate is
strictly worse.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chowder.low_rank_vocab import LowRankLMHead  # noqa: E402

RANKS = (2048, 1536, 1024, 512)


def build_head(factors_dir: Path, rank: int, device: torch.device) -> LowRankLMHead:
    bundle = load_file(str(factors_dir / f"lm_head_rank{rank}_factors.safetensors"))
    head = LowRankLMHead(4096, 248320, rank)
    # Checkpoint layout: U [vocab, rank], V [rank, hidden] -> head_a = V, head_b = U
    head.load_factors(bundle["v"].float(), bundle["u"].float())
    return head.to(device)


def kl_metrics(teacher: torch.Tensor, student: torch.Tensor) -> dict[str, float]:
    p = torch.softmax(teacher, dim=-1)
    q = torch.softmax(student, dim=-1)
    kl = torch.xlogy(p, p / q.clamp_min(1e-12)).sum(-1)
    ce = torch.nn.functional.cross_entropy(student, teacher.argmax(-1), reduction="mean")
    top1 = float((teacher.argmax(-1) == student.argmax(-1)).float().mean())
    t5 = teacher.topk(5, dim=-1).indices.tolist()
    s5 = student.topk(5, dim=-1).indices.tolist()
    top5 = sum(len(set(a) & set(b)) / 5.0 for a, b in zip(t5, s5)) / teacher.shape[0]
    return {
        "kl_mean_nats": float(kl.mean()),
        "top1_agreement": top1,
        "top5_recall": top5,
        "ce_at_teacher_argmax": float(ce),
    }


def train_head(
    head: LowRankLMHead,
    hidden: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    steps: int,
    batch: int,
    lr: float,
    device: torch.device,
) -> list[float]:
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    losses: list[float] = []
    generator = torch.Generator().manual_seed(0)
    n = hidden.shape[0]
    head.train()
    for step in range(steps):
        idx = torch.randint(0, n, (min(batch, n),), generator=generator)
        x = hidden[idx].to(device)
        target = teacher_logits[idx].to(device)
        student = head(x)
        loss = torch.nn.functional.kl_div(
            torch.log_softmax(student, dim=-1),
            torch.log_softmax(target, dim=-1),
            log_target=True,
            reduction="batchmean",
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss))
    head.eval()
    return losses


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factors", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ranks", default="2048,1024,512")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    factors_dir = Path(args.factors)
    tensors = load_file(args.cache)
    hidden_cpu = tensors["hidden"].float()
    logits_cpu = tensors["logits"].float()
    n = hidden_cpu.shape[0]
    split = n - 60  # same val split as low_rank_real_eval
    train_h, train_t = hidden_cpu[:split], logits_cpu[:split]
    val_h, val_t = hidden_cpu[split:].to(device), logits_cpu[split:].to(device)
    print(f"train {train_h.shape[0]} / val {val_h.shape[0]} positions on {device}", flush=True)

    report: dict[str, dict] = {
        "steps": args.steps,
        "batch": args.batch,
        "lr": args.lr,
        "ranks": {},
    }
    for rank in [int(r) for r in args.ranks.split(",")]:
        head = build_head(factors_dir, rank, device)
        recon_val = kl_metrics(val_t, head(val_h))
        t0 = time.time()
        losses = train_head(
            head, train_h, train_t,
            steps=args.steps, batch=args.batch, lr=args.lr, device=device,
        )
        train_s = time.time() - t0
        distill_val = kl_metrics(val_t, head(val_h))
        report["ranks"][str(rank)] = {
            "reconstruction_only_val": recon_val,
            "distilled_val": distill_val,
            "kl_improvement_nats": round(recon_val["kl_mean_nats"] - distill_val["kl_mean_nats"], 4),
            "final_train_kl": losses[-1] if losses else None,
            "train_seconds": round(train_s, 1),
        }
        print(
            f"rank {rank}: recon KL={recon_val['kl_mean_nats']:.4f} -> distill KL={distill_val['kl_mean_nats']:.4f} "
            f"(top1 {recon_val['top1_agreement']:.3f} -> {distill_val['top1_agreement']:.3f}) in {train_s:.0f}s",
            flush=True,
        )
        del head
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
