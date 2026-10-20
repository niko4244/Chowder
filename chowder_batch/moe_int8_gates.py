"""Measure the frozen-expert placement gates on a real MoE checkpoint.

  python chowder_batch/moe_int8_gates.py --base-model <dir> --out gates.json \
      [--arms int8,bulk] [--steps 3] [--seq-len 2048] [--device 0] [--cpu]

For each arm this records what the prereg in
``docs/quals/MOE_TRANSIENT_GRAD_PLACEMENT_PREREG_2026-10-18.md`` asks for, or an
explicit "not measured" when this runner cannot answer it:

  load_census     live placement census from the loader itself
                  (``moe_int8_experts.load_model_int8_experts`` for ``int8``,
                  ``moe_transient_grad.prepare_split_model`` for ``bulk``)
  memory_vram     peak allocated VRAM per step, against the <= 14.5 GB gate
  host_rss        process peak working set across load and steps
  throughput      measured s/step of a **placement step**: forward + backward on
                  ``inputs_embeds`` covering every expert layer, no LoRA and no
                  optimizer, so it is the cost of the placement, not of a training
                  step. The prereg's throughput gate is a worker run.

``gradient``, ``smoke``, ``router_balance`` and ``wddm_spill`` need a dataset and
a real fine-tune, so this runner reports them as not measured rather than
guessing -- and ``--witness`` is how a worker run's own record of them gets into
the table (``chowder.placement_witnesses`` produces it from the run).

WDDM shared-memory spill is not observable from here either: the record carries
``peak_vram_gb`` and ``reserved_gb`` and leaves the spill verdict to the operator.

The measurement goes straight into the prereg's gate table
(``docs/quals/MOE_INT8_PLACEMENT_PREREG_2026-10-20.md``, or ``--prereg`` /
``CHOWDER_MOE_GATE_TABLE``): only the generated block is replaced, so the prose
stays hand-authored while the numbers cannot drift from the last measurement. A
row this run cannot answer is carried forward from the recorded one rather than
overwritten with "not measured". ``--check`` compares and writes nothing.

Exit codes: 0 table written with no regression, 1 a recorded gate **regressed**
(the table is still updated, so the worse numbers are the ones on record), 2
refusal (missing directory, unknown arm, no CUDA without ``--cpu``), 3 an arm
failed to load or to place, 4 the table was measured on a different artifact and
would be overwritten with incomparable numbers (``--allow-new-baseline`` starts
a new baseline), 5 the header projection disagrees with the recorded measurement
(``--project-only``: the two cannot both be right).

``--project-only`` is the mode a box with no free card can still answer with:
it reads the shard **headers** (no tensor data, no model, no device), derives
what a placement step must occupy, and prints that beside whatever the table
records -- ``load_census`` and ``host_rss`` today, ``memory_vram`` once a
``--steps 3`` run on the card has happened. It writes no table: a projection is
not a measurement, and the block must never carry one. The int8 and non-expert
terms are derived, not estimated, so a mismatch against the recorded census is a
refusal (exit 5) rather than a rounding difference.

``--self-check`` writes its own tiny synthetic checkpoint to a temporary
directory and runs both arms on the CPU, so the runner itself is verifiable
without a GPU or a real model. Without an explicit ``--prereg`` it writes its
gate table beside that checkpoint, never over the placement's prereg: a
synthetic model's numbers are not the placement's numbers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # noqa: E402

from chowder.moe_placement_gates import (  # noqa: E402
    ALL_GATES,
    DEFAULT_PREREG,
    EXPERT_PLACEMENT,
    VRAM_GATE_GB,
    PlacementGateError,
    artifact_identity,
    assert_declaration_allowed,
    code_commit,
    parse_block,
    regressions,
    render_block,
    resolve_prereg_path,
    splice_block,
    splice_witness_rows,
)

SCHEMA = "chowder.moe-placement-gates.v1"
ARMS = ("int8", "bulk")
#: A gate table written by a different artifact cannot back these numbers: the
#: run refuses to overwrite it until the operator says the baseline moved.
ARTIFACT_MISMATCH_EXIT = 4
#: ``--project-only`` found the header arithmetic and the recorded measurement
#: irreconcilable. Nothing is written either way; this says which of the two is
#: the thing to look at before spending the card.
PROJECTION_DISAGREES_EXIT = 5


def _peak_rss_bytes() -> int | None:
    """This process's peak working set: the host RAM the arm actually touched."""
    try:  # Windows: psapi gives the peak, not just the current
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(ProcessMemoryCounters)
        # The pseudohandle is pointer sized: untyped, ctypes truncates it to -1
        # and the call fails with an overflow instead of returning memory.
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        query = kernel32.K32GetProcessMemoryInfo
        query.argtypes = [ctypes.c_void_p, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD]
        query.restype = wintypes.BOOL
        if query(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return int(counters.PeakWorkingSetSize)
        return None
    except Exception:  # non-Windows or a stripped-down interpreter
        try:
            import resource

            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return int(peak) * (1024 if sys.platform != "darwin" else 1)
        except Exception:
            return None


def _gib(nbytes: int | None) -> float | None:
    return None if nbytes is None else round(nbytes / 2**30, 2)


def _vocab_size(config: Any) -> int:
    text = getattr(config, "get_text_config", lambda: config)()
    return int(getattr(text, "vocab_size", 0) or getattr(config, "vocab_size", 0) or 1000)


def _load_int8(base: Path, device: Any, dtype: Any) -> tuple[Any, dict[str, Any]]:
    from chowder.backends import moe_int8_experts as mie

    model, census = mie.load_model_int8_experts(base, device=device, dtype=dtype)
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    census = {**census, **mie.prepare_int8_placement(model, device_type=device_type)}
    return model, census


def _load_bulk(base: Path, device: Any, dtype: Any) -> tuple[Any, dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM

    from chowder.backends import moe_transient_grad as mtg

    if str(device).startswith("cuda"):
        device_map = mtg.split_device_map(str(base), revision=None, local_files_only=True, device=device.index or 0)
        model = AutoModelForCausalLM.from_pretrained(str(base), dtype=dtype, device_map=device_map)
        census = mtg.prepare_split_model(model, device_type="cuda", bulk=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(str(base), dtype=dtype)
        census = mtg.prepare_split_model(model, device_type="cpu", bulk=True)
    return model, dict(census)


ARM_LOADERS = {"int8": _load_int8, "bulk": _load_bulk}


def _placement_steps(model: Any, *, steps: int, seq_len: int, device: Any) -> dict[str, Any]:
    """Forward + backward through every expert layer, timed per step.

    Deliberately not a training step: no LoRA, no optimizer. What it isolates is
    the placement's own cost -- the per-layer host->device copy (and, for int8,
    the dequantisation) in forward and again in the backward recompute.
    """
    import torch

    if steps <= 0:
        # Census-only: the load is the measurement, nothing is stepped.
        return {
            "steps": 0,
            "seq_len": seq_len,
            "step_seconds": [],
            "s_per_step": None,
            "includes_lora": False,
            "includes_optimizer": False,
            "gradient_on_inputs_embeds": False,
        }

    vocabulary = _vocab_size(model.config)
    generator = torch.Generator().manual_seed(0)
    tokens = torch.randint(0, vocabulary, (1, seq_len), generator=generator).to(device)
    embedding = model.get_input_embeddings()

    times: list[float] = []
    for _ in range(steps):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        hidden = embedding(tokens).detach().requires_grad_(True)
        with torch.enable_grad():
            loss = model(inputs_embeds=hidden, use_cache=False).logits.float().square().mean()
            loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
        model.zero_grad(set_to_none=True)

    return {
        "steps": steps,
        "seq_len": seq_len,
        "step_seconds": [round(value, 4) for value in times],
        "s_per_step": round(sum(times) / len(times), 4) if times else None,
        "includes_lora": False,
        "includes_optimizer": False,
        "gradient_on_inputs_embeds": True,
    }


def _vram(torch: Any, device: Any) -> dict[str, Any]:
    if device.type != "cuda":
        return {"peak_vram_gb": None, "reserved_gb": None}
    return {
        "peak_vram_gb": round(torch.cuda.max_memory_allocated(device) / 2**30, 3),
        "reserved_gb": round(torch.cuda.max_memory_reserved(device) / 2**30, 3),
    }


def run_arm(arm: str, base: Path, *, device: Any, dtype: Any, steps: int, seq_len: int) -> dict[str, Any]:
    import torch

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    model, census = ARM_LOADERS[arm](base, device, dtype)
    load_seconds = time.perf_counter() - started
    load_rss = _peak_rss_bytes()

    result: dict[str, Any] = {
        "arm": arm,
        "load_seconds": round(load_seconds, 3),
        "load_peak_rss_gb": _gib(load_rss),
        "census": census,
    }
    if "expert_bytes" in census:
        result["census_meta"] = {
            "expert_bytes_gb": _gib(int(census["expert_bytes"])),
            "patched_expert_modules": census.get("patched_expert_modules"),
        }

    result.update(_placement_steps(model, steps=steps, seq_len=seq_len, device=device))
    result.update(_vram(torch, device))
    result["total_peak_rss_gb"] = _gib(_peak_rss_bytes())
    result["gates"] = {
        "load_census": "measured",
        "memory_vram": (
            "measured"
            if result["peak_vram_gb"] is not None
            else "not measured (CPU arm)"
        ),
        "memory_vram_pass": (
            None
            if result["peak_vram_gb"] is None
            else bool(result["peak_vram_gb"] <= VRAM_GATE_GB)
        ),
        "host_rss": "measured",
        "throughput": (
            "not measured (--steps 0: census only)"
            if not result["step_seconds"]
            else "measured (placement step, no LoRA/optimizer)"
        ),
        "gradient": "not measured by this runner (needs a LoRA run)",
        "smoke": "not measured by this runner (needs a dataset)",
        "router_balance": "not measured by this runner (needs a smoke run)",
        "wddm_spill": "not observable from here; compare peak_vram_gb to the card",
    }
    return result


def run(
    base: Path,
    *,
    arms: tuple[str, ...],
    device: Any,
    dtype: Any,
    steps: int,
    seq_len: int,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "base_model": str(base),
        "device": str(device),
        "dtype": str(dtype) if dtype is not None else "auto (the checkpoint's)",
        "placement": EXPERT_PLACEMENT,
        "vram_gate_gb": VRAM_GATE_GB,
        "artifact": artifact_identity(base),
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_commit": code_commit(),
        "steps": steps,
        "seq_len": seq_len,
        "arms": {},
        "note": (
            "Placement steps only: forward+backward on inputs_embeds, no LoRA and no "
            "optimizer. The prereg's throughput, gradient, smoke and router-balance "
            "gates still require a worker run."
        ),
    }
    for arm in arms:
        record["arms"][arm] = run_arm(arm, base, device=device, dtype=dtype, steps=steps, seq_len=seq_len)
    return record


# --- the prereg table ---------------------------------------------------------
#
# The placement's prereg document owns one generated block, and
# ``chowder.moe_placement_gates`` is the single definition of it: how it is
# rendered, how it is read back, what counts as a regression, and when a campaign
# is allowed to declare the placement at all. The runner replaces only that
# block, so the prose (and the gate definitions) stay authored by hand while the
# numbers cannot drift from the last measurement.


def _self_check_workspace() -> Path:
    """A tiny synthetic Qwen3-MoE checkpoint, so the runner is provable on CPU."""
    import torch
    from transformers import AutoModelForCausalLM, Qwen3MoeConfig

    directory = Path(tempfile.mkdtemp(prefix="moe-int8-gates-"))
    config = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=8,
        moe_intermediate_size=4,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_experts=6,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        max_position_embeddings=32,
        head_dim=8,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    AutoModelForCausalLM.from_config(config).save_pretrained(directory, max_shard_size="20KB")
    return directory


def _record_witness(args: argparse.Namespace) -> int:
    """Record a worker run's four gates in the table. No model, no device."""
    try:
        witness = json.loads(Path(args.witness).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        print(f"cannot read the witness file {args.witness}: {error}", file=sys.stderr)
        return 2
    if not isinstance(witness, dict):
        print(f"{args.witness} is not a witness record (expected a JSON object)", file=sys.stderr)
        return 2
    if args.spill_verdict:
        if not args.spill_basis.strip():
            print(
                "--spill-verdict needs --spill-basis: nobody can observe driver paging "
                "from PyTorch, so the verdict must name what it rests on",
                file=sys.stderr,
            )
            return 2
        gates = dict(witness.get("gates") or {})
        gates["wddm_spill"] = {
            "value": f"operator {args.spill_verdict}: {args.spill_basis.strip()}",
            "verdict": args.spill_verdict,
            "evidence": {"basis": args.spill_basis.strip()},
        }
        witness["gates"] = gates

    prereg = resolve_prereg_path(args.prereg)
    if prereg is None or not prereg.is_file():
        print(
            f"no gate table to record into at {prereg or DEFAULT_PREREG}; run this "
            "runner against the base model first, so the table is bound to an artifact",
            file=sys.stderr,
        )
        return 2
    document = prereg.read_text(encoding="utf-8")
    try:
        updated, recorded = splice_witness_rows(document, witness)
    except PlacementGateError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    prereg.write_text(updated, encoding="utf-8")
    print(f"recorded {', '.join(recorded)} from {args.witness} into {prereg}")

    parsed = parse_block(updated) or {}
    verdicts = parsed.get("verdicts") or {}
    measured = [gate for gate, verdict in verdicts.items() if not verdict.startswith("not measured")]
    print(f"{len(measured)}/{len(ALL_GATES)} gates measured: {', '.join(sorted(measured))}")
    # The artifact binding was enforced where the rows were written (the witness
    # digest must equal the table's), so this only asks whether every gate is now
    # measured. The campaign's own base is checked at prepare time, against the
    # path it actually declares.
    try:
        assert_declaration_allowed(updated, ["a campaign declaring the placement"])
    except PlacementGateError as error:
        print(f"a campaign still cannot declare the placement: {error}")
        return 0
    print("every gate is measured on this artifact: a campaign may declare the placement")
    return 0


def _recorded_numbers(gates_path: str, prereg: Path | None) -> dict[str, Any]:
    """What has already been measured, in the units the projection reports.

    A gate record is the better witness when there is one: it carries the
    census' own ``non_expert_bytes`` and, after a ``--steps 3`` run, the device
    peak. The prereg table carries only the two numbers its rows publish, which
    is why a projection against it can check the int8 grid and the host peak but
    not the non-expert split.
    """
    recorded: dict[str, Any] = {
        "source": None,
        "int8_plus_scales": None,
        "non_expert": None,
        "host_peak": None,
        "step_vram": None,
        "seq_len": None,
    }
    if gates_path:
        try:
            payload = json.loads(Path(gates_path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError(f"cannot read the gate record {gates_path}: {error}") from error
        arm = (payload.get("arms") or {}).get("int8") or {}
        census = arm.get("census") or {}
        meta = arm.get("census_meta") or {}
        recorded.update(
            {
                "source": Path(gates_path).name,
                "int8_plus_scales": meta.get("expert_bytes_gb"),
                "non_expert": _gib(census.get("non_expert_bytes")),
                "host_peak": arm.get("total_peak_rss_gb"),
                "step_vram": arm.get("peak_vram_gb"),
                "seq_len": payload.get("seq_len"),
            }
        )
        return recorded

    if prereg is None or not prereg.is_file():
        return recorded
    parsed = parse_block(prereg.read_text(encoding="utf-8")) or {}
    numbers = parsed.get("numbers") or {}
    recorded.update(
        {
            "source": f"{prereg.name}: load_census, host_rss",
            "int8_plus_scales": numbers.get("load_census"),
            "host_peak": numbers.get("host_rss"),
            "step_vram": numbers.get("memory_vram"),
        }
    )
    return recorded


def _project_only(args: argparse.Namespace) -> int:
    """Print the projection from the headers, beside what is already recorded.

    No model, no weights, no device: this is the mode that answers "how much did
    the dtype fix buy" on a box whose card is busy. It never writes the gate
    table -- the table is for measurements, and a projection in it would be
    indistinguishable from one.
    """
    from chowder import moe_vram_projection as mvp

    if args.self_check:
        args.base_model = str(_self_check_workspace())
    elif not args.base_model:
        print("--project-only needs --base-model (or --self-check)", file=sys.stderr)
        return 2
    base = Path(args.base_model)
    if not base.is_dir():
        print(f"base model directory not found: {base}", file=sys.stderr)
        return 2

    from transformers import AutoConfig

    from chowder.backends import moe_int8_experts as mie

    try:
        config = AutoConfig.from_pretrained(str(base))
        dtype, dtype_source = mie.resolve_non_expert_dtype(
            base, dtype=args.dtype or None, config=config
        )
        inv = mvp.inventory(base, dtype=str(dtype).replace("torch.", ""))
        activations = None if args.activations_gib is None else int(args.activations_gib * mvp.GIB)
        projection = mvp.project(
            inv,
            seq_len=args.seq_len,
            vocab_size=_vocab_size(config),
            activation_bytes=activations,
        )
    except (OSError, ValueError, FileNotFoundError) as error:
        print(f"cannot project {base}: {type(error).__name__}: {error}", file=sys.stderr)
        return 2

    if args.prereg:
        # An explicit table wins even under --self-check: this mode writes
        # nothing, so naming one is asking to be compared against it.
        prereg = Path(args.prereg)
    elif args.self_check:
        # The synthetic checkpoint's table is its own; the placement's prereg is
        # never the comparison for numbers from a 2-layer toy.
        prereg = base / "PREREG-self-check.md"
    else:
        prereg = resolve_prereg_path(None)
    try:
        recorded = _recorded_numbers(args.gates, prereg)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2

    if recorded.get("seq_len") and int(recorded["seq_len"]) != int(args.seq_len):
        print(
            f"note: {recorded['source']} measured at seq {recorded['seq_len']}, projecting at "
            f"seq {args.seq_len}; the step rows are only comparable at the same seq",
            file=sys.stderr,
        )

    artifact = {**artifact_identity(base), "path": str(base)}
    print(mvp.render(projection, recorded, artifact=artifact))
    if dtype_source:
        print(f"  dtype resolved as {str(dtype).replace('torch.', '')} ({dtype_source})")

    problems = mvp.disagreements(projection, recorded)
    if args.out:
        payload = {
            "schema": "chowder.moe-placement-projection.v1",
            "base_model": str(base),
            "artifact": artifact,
            "dtype": str(dtype).replace("torch.", ""),
            "dtype_source": dtype_source,
            "projection": projection,
            "recorded": recorded,
            "disagreements": problems,
        }
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    if problems:
        for problem in problems:
            print(f"DISAGREES: {problem}", file=sys.stderr)
        return PROJECTION_DISAGREES_EXIT
    print("projection and recorded measurement reconcile")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-model", default="", help="local checkpoint directory (safetensors shards)")
    parser.add_argument("--out", default="", help="where to write the gate record (JSON)")
    parser.add_argument("--arms", default="int8,bulk", help=f"comma-separated subset of {ARMS}")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dtype", default="", help="auto (the config's), bfloat16, float16, float32")
    parser.add_argument("--cpu", action="store_true", help="run on the CPU (self-check, small models)")
    parser.add_argument("--self-check", action="store_true", help="tiny synthetic checkpoint, CPU, then print the record")
    parser.add_argument("--prereg", default="", help=f"prereg document to update (default {DEFAULT_PREREG})")
    parser.add_argument("--check", action="store_true", help="compare against the prereg table and write nothing")
    parser.add_argument(
        "--allow-new-baseline",
        action="store_true",
        help="accept a different artifact digest instead of refusing to compare",
    )
    parser.add_argument(
        "--project-only",
        action="store_true",
        help=(
            "print the VRAM projection from the shard headers beside the recorded gate "
            "table; reads no weights, needs no device, writes no table"
        ),
    )
    parser.add_argument(
        "--gates",
        default="",
        help="a gate record JSON to compare the projection against instead of the prereg table",
    )
    parser.add_argument(
        "--activations-gib",
        type=float,
        default=None,
        help=(
            "the measured activation term for every layer at --seq-len, from "
            "chowder_batch/moe_activation_probe.py; without it the step figure stays a floor"
        ),
    )
    parser.add_argument(
        "--witness",
        default="",
        help=(
            "a worker run's placement-witnesses.json to record in the table; no "
            "measurement is taken, only the four gates this runner cannot answer"
        ),
    )
    parser.add_argument(
        "--spill-verdict",
        default="",
        choices=("", "pass", "FAIL"),
        help="operator verdict for wddm_spill (needs --spill-basis)",
    )
    parser.add_argument(
        "--spill-basis",
        default="",
        help="what the spill verdict rests on; PyTorch cannot observe driver paging",
    )
    args = parser.parse_args(argv)

    if args.project_only:
        return _project_only(args)

    if args.witness:
        return _record_witness(args)

    import torch

    arms = tuple(arm.strip() for arm in args.arms.split(",") if arm.strip())
    unknown = [arm for arm in arms if arm not in ARM_LOADERS]
    if unknown:
        print(f"unknown arm(s) {unknown}; expected a subset of {ARMS}", file=sys.stderr)
        return 2

    if args.self_check:
        os.environ["CHOWDER_NO_GPU"] = "1"
        args.base_model = str(_self_check_workspace())
        args.cpu = True
        arms = ARMS
        args.steps = min(args.steps, 2)
        args.seq_len = min(args.seq_len, 8)
    elif not args.base_model:
        print("--base-model is required (or --self-check)", file=sys.stderr)
        return 2

    base = Path(args.base_model)
    if not base.is_dir():
        print(f"base model directory not found: {base}", file=sys.stderr)
        return 2

    use_cuda = torch.cuda.is_available() and not args.cpu
    if not use_cuda and not args.cpu:
        print("no CUDA device; pass --cpu to measure a census without the card", file=sys.stderr)
        return 2
    device = torch.device("cuda", args.device) if use_cuda else torch.device("cpu")
    dtype = getattr(torch, args.dtype) if args.dtype else None

    try:
        record = run(base, arms=arms, device=device, dtype=dtype, steps=args.steps, seq_len=args.seq_len)
    except Exception as error:  # a failed arm is a refusal, not a record with holes
        # The stack, not only the message: a failure inside a loader is
        # otherwise reported from the line that called it and nothing else.
        import traceback

        traceback.print_exc()
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 3

    payload = json.dumps(record, indent=2)
    if args.out:
        Path(args.out).write_text(payload, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(payload)

    if args.self_check and not args.prereg:
        # A synthetic model's numbers are not the placement's numbers: the
        # self-check writes its table beside its own checkpoint, never over the
        # pre-registration a campaign would be read against.
        prereg = base / "PREREG-self-check.md"
    else:
        prereg = resolve_prereg_path(args.prereg)
    if prereg is None:
        print(
            "no prereg document in this checkout and no --prereg; "
            "the JSON record is the only evidence",
            file=sys.stderr,
        )
        return 0
    document = prereg.read_text(encoding="utf-8") if prereg.is_file() else ""
    previous = parse_block(document)

    measured_on = (record.get("artifact") or {}).get("config_sha256")
    recorded_on = (previous or {}).get("config_sha256")
    new_baseline = False
    if previous is not None and recorded_on and measured_on and recorded_on != measured_on:
        if not args.allow_new_baseline:
            print(
                f"{prereg} records gates measured on config sha256 `{recorded_on}`, but this run "
                f"measured `{measured_on}`: refusing to overwrite one artifact's table with "
                "another's numbers (pass --allow-new-baseline to start a new baseline)",
                file=sys.stderr,
            )
            return ARTIFACT_MISMATCH_EXIT
        new_baseline = True
        print(
            f"--allow-new-baseline: replacing the table measured on `{recorded_on}` "
            f"with `{measured_on}`",
            file=sys.stderr,
        )
        previous = None

    problems = regressions(previous, record)
    if problems:
        print("REGRESSION against the recorded gates:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
    elif previous is not None:
        print("no regression against the recorded gates")

    if args.check:
        print(f"(--check: {prereg} was not written)")
        return 1 if problems else 0

    prereg.parent.mkdir(parents=True, exist_ok=True)
    prereg.write_text(
        splice_block(document, render_block(record, previous=previous)), encoding="utf-8"
    )
    print(f"wrote the gate table into {prereg}{' (new baseline)' if new_baseline else ''}")
    carried = (parse_block(prereg.read_text(encoding="utf-8")) or {}).get("carried") or []
    if carried:
        print(
            f"carried forward from the previous measurement: {', '.join(carried)} "
            "(only a run that measures a gate can change its verdict)"
        )
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
