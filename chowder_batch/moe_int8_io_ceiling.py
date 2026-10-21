"""Measure the two ceilings the int8 frozen-expert placement runs into on this box.

Neither number is a gate. They exist so a throughput claim in
``docs/quals/MOE_INT8_THROUGHPUT_2026-10-20.md`` has a script behind it that
anyone can re-run on the same hardware:

    --h2d        host-to-device bandwidth for the placement's own copy form:
                 a pageable and a pinned 1 GiB buffer, so the step's int8
                 traffic can be converted into seconds on this link.
    --experts    the two expert forms at the artifact's shapes, uniformly
                 routed: the per-expert ``F.linear`` loop the reference path
                 uses, and the padded two-``bmm`` form ``grouped_experts(True)``
                 uses. Reports ms and effective TFLOP/s for each.

``--self-check`` runs the arithmetic on CPU and needs neither a card nor the
weights; it does not measure anything.
"""

from __future__ import annotations

import argparse
import json
import time


def h2d_seconds(gib_per_pass: float, passes: int, gib_per_s: float) -> float:
    """Seconds of pure host-to-device traffic per training step."""
    return (gib_per_pass * passes) / gib_per_s


def step_ceiling(gib_per_pass: float, passes: int, gib_per_s: float, target_s: float) -> dict:
    """What fraction of a step's target the copy traffic alone consumes."""
    seconds = h2d_seconds(gib_per_pass, passes, gib_per_s)
    return {
        "gib_per_step": round(gib_per_pass * passes, 2),
        "h2d_seconds": round(seconds, 2),
        "target_s_per_step": target_s,
        "target_reachable": seconds < target_s,
        "headroom_seconds": round(target_s - seconds, 2),
    }


def measure_h2d(gib: float, reps: int) -> dict:
    import torch

    n = int(gib * 2**30)
    out = {}
    for name, pinned in (("pageable", False), ("pinned", True)):
        host = torch.empty(n, dtype=torch.uint8, pin_memory=pinned)
        dev = torch.empty_like(host, device="cuda")
        torch.cuda.synchronize()
        best = float("inf")
        for _ in range(reps):
            start = time.perf_counter()
            dev.copy_(host, non_blocking=pinned)
            torch.cuda.synchronize()
            best = min(best, time.perf_counter() - start)
        out[name] = round(gib / best, 3)
        del host, dev
        torch.cuda.empty_cache()
    return out


def measure_experts(
    tokens: int,
    topk: int,
    experts: int,
    hidden: int,
    moe_inter: int,
    reps: int,
    *,
    device: str = "cuda",
) -> dict:
    """Time the reference per-expert loop against the padded two-``bmm`` form.

    Both forms compute the same work on the same rows: the ``tokens * topk``
    routed assignments dealt uniformly over ``experts`` (``counts`` rows each,
    dropping the few assignments that do not divide evenly) and drawn from the
    ``tokens`` rows of ``x``. What is compared is one rectangle of expert work
    done two ways -- 160 kernel launches against two -- so the row *indices*
    have to be built once, outside the timed forms, or the benchmark would be
    timing its own plumbing.

    ``device`` is a parameter so the shapes can be exercised without a card:
    a form that cannot even index its rows is a correctness bug, not a slow
    measurement.
    """
    import torch
    import torch.nn.functional as F

    dtype = torch.bfloat16
    x = torch.randn(tokens, hidden, dtype=dtype, device=device)
    w1 = torch.randn(experts, 2 * moe_inter, hidden, dtype=dtype, device=device)
    w2 = torch.randn(experts, hidden, moe_inter, dtype=dtype, device=device)
    counts = max(1, (tokens * topk) // experts)
    rows_used = counts * experts
    # The fixed routing assignment: `counts` rows per expert, wrapping around
    # the token rows. `counts` rows may exceed `tokens` (102 vs 2048 rows here),
    # so the assignment is an index tensor, not a slice of x.
    index = torch.arange(rows_used, device=device) % tokens
    assignments = x.index_select(0, index)
    flops = 2 * rows_used * (2 * moe_inter * hidden + hidden * moe_inter)

    # The reference path: one small GEMM pair per expert (gate_up, down).
    def loop() -> torch.Tensor:
        total = torch.zeros(rows_used, hidden, dtype=dtype, device=device)
        for e in range(experts):
            rows = slice(e * counts, (e + 1) * counts)
            gate_up = F.linear(assignments[rows], w1[e])
            total[rows] = F.linear(F.silu(gate_up[:, :moe_inter]) * gate_up[:, moe_inter:], w2[e])
        return total

    # The grouped path: pad every expert to the same row count, two bmm calls.
    def grouped() -> torch.Tensor:
        xg = assignments.view(experts, counts, hidden)
        gate_up = torch.bmm(xg, w1.transpose(1, 2))
        act = F.silu(gate_up[:, :, :moe_inter]) * gate_up[:, :, moe_inter:]
        return torch.bmm(act, w2.transpose(1, 2)).view(rows_used, hidden)

    out = {}
    for name, fn in (("loop", loop), ("grouped", grouped)):
        result = fn()
        if result.shape != (rows_used, hidden):
            raise RuntimeError(
                f"{name} produced {tuple(result.shape)}, expected {(rows_used, hidden)}; "
                "the two forms are not comparable and their timings mean nothing"
            )
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        best = float("inf")
        for _ in range(reps):
            start = time.perf_counter()
            fn()
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            best = min(best, time.perf_counter() - start)
        out[name] = {"ms": round(best * 1e3, 2), "tflops": round(flops / best / 1e12, 2)}
    out["speedup"] = round(out["loop"]["ms"] / out["grouped"]["ms"], 2)
    out["rows_per_expert"] = counts
    out["padded_rows_waste"] = round((experts * counts - tokens * topk) / (tokens * topk), 4)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--h2d", action="store_true")
    parser.add_argument("--experts", action="store_true")
    parser.add_argument("--gib", type=float, default=1.0)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--experts-n", type=int, default=160)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--moe-inter", type=int, default=512)
    # The placement's per-step traffic: 18.82 GiB of int8 + scales, three passes.
    parser.add_argument("--gib-per-pass", type=float, default=18.82)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--target-s", type=float, default=9.0)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)

    if args.self_check:
        # 20 GiB over a 5.5 GiB/s link cannot be under 3.6 s; 56.5 GiB cannot be under 10 s.
        assert abs(h2d_seconds(1.0, 1, 5.0) - 0.2) < 1e-12
        assert abs(h2d_seconds(18.82, 3, 5.52) - 10.23) < 0.01
        c = step_ceiling(18.82, 3, 5.52, 9.0)
        assert c["target_reachable"] is False and c["headroom_seconds"] < 0
        assert step_ceiling(18.82, 3, 5.52, 12.0)["target_reachable"] is True
        print(json.dumps({"self_check": "ok", "ceiling_at_5.52_gib_s": c}, indent=2))
        return 0

    report: dict = {}
    if args.h2d:
        report["h2d_gib_per_s"] = measure_h2d(args.gib, args.reps)
    if args.experts:
        report["experts"] = measure_experts(
            args.tokens, args.topk, args.experts_n, args.hidden, args.moe_inter, args.reps
        )
    bandwidth = (report.get("h2d_gib_per_s") or {}).get("pageable")
    if bandwidth:
        report["step_ceiling"] = step_ceiling(args.gib_per_pass, args.passes, bandwidth, args.target_s)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
