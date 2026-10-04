"""The router-healing worker: the process that actually trains a router.

`router_healing_orchestrator.py` was an evidence scaffold. It received an
already-loaded model *outside* its own timing boundary, optionally serialised
resume state with ``default=str``, and threw the loaded delta away before
calling an evaluator that never saw it. This worker is the real thing: it is
launched as a subprocess by `RouterHealingExecutor`, so everything it claims is
either measured in its own process or absent from its result.

What it does, in order
----------------------
1. Verifies the `chowder` source it imported matches the identity its parent
   declared, *before* reading its spec or touching a weight file.
2. Loads the local base and its tokenizer, timing the load, and refuses when the
   base's **content** identity differs from the one recorded in the spec.
3. Freezes everything except the router gates (``mlp.gate.weight``) using the
   existing `router_healing` policy, then proves the intended set is exactly the
   set that exists -- by path, not by count.
4. Trains with a **real** forward/backward/update, observing each gate's gradient
   after ``backward()`` and its actual optimizer update, under hard step, token
   and wall-clock limits.
5. Publishes the trained router values as a verified payload, and separately
   publishes resumable optimizer/scheduler/RNG/step state, so the artifact a
   reader consumes and the state a resume consumes are never the same file.

Honesty rules this module implements
------------------------------------
* A phase it did not measure is recorded as unavailable **with the reason** --
  never as ``0.0``.
* CPU is the only qualified device. A device-safe digest alone does not qualify
  the accelerator training, interruption, and evaluation path.
* Expert-row utilisation is reported only when it was actually collected; it is
  `not_reported` otherwise, because a made-up routing table is worse than none.
* A run that hits a hard limit still writes its result, with the limit named.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping

from ..base_identity import resolve_base_identity
from ..lifecycle import (
    PhaseTimer,
    cuda_synchronize,
    quantization_reality_report,
    sampling_device,
    tensor_inventory,
    training_lifecycle_ledger,
)
from ..progress_write import write_progress_best_effort
from ..resume_state import assert_resumable, inventory_checkpoint, resume_witness
from ..router_healing import ROUTER_ONLY_SUFFIXES, freeze_for_router_healing
from ..router_payload import save_router_payload
from ..trainability import (
    TrainabilityProbe,
    assert_components_qualified,
    assert_router_only_scope,
    component_path_report,
    resolve_expected_parameter_paths,
    utilization_by_expert,
)
from ..worker_env import chowder_source_identity
from .router_healing import QUALIFIED_DEVICES, RouterHealingRunSpec

RESULT_KIND = "router_healing_worker_result.v1"

#: Bound on the per-step loss log, so a long run cannot turn its own evidence
#: into an unbounded payload. Truncation is recorded, never silent.
LOSS_LOG_LIMIT = 5000

#: How often the worker writes its best-effort progress file.
PROGRESS_EVERY = 10

CHECKPOINT_KIND = "router_healing_checkpoint.v2"
CHECKPOINT_FILES = (
    "optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json",
    "router_state.safetensors",
)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_corpus(spec: RouterHealingRunSpec) -> str:
    path = Path(spec.corpus_path)
    if not path.is_file():
        raise RuntimeError(f"training corpus not found: {path}")
    actual = _sha256_file(path)
    if actual != spec.corpus_sha256:
        raise RuntimeError(
            "training corpus hash mismatch: the spec declares "
            f"{spec.corpus_sha256!r} but {path} is {actual!r}"
        )
    return path.read_text(encoding="utf-8")


def _pack_blocks(input_ids: list[int], seq_len: int) -> list[list[int]]:
    usable = len(input_ids) - (len(input_ids) % seq_len)
    if usable < seq_len:
        raise RuntimeError(
            f"the corpus encodes to {len(input_ids)} tokens, fewer than one sequence "
            f"of {seq_len}; refusing to train on nothing"
        )
    return [input_ids[start : start + seq_len] for start in range(0, usable, seq_len)]


def _learning_rate_at(step: int, spec: RouterHealingRunSpec) -> float:
    """Warmup then the configured decay, computed from the step alone.

    Derived rather than stored so a resumed run recomputes the identical
    schedule position from its restored global step instead of trusting a
    serialised float.
    """
    if spec.warmup_steps and step < spec.warmup_steps:
        return spec.learning_rate * float(step + 1) / float(spec.warmup_steps)
    if spec.scheduler == "cosine":
        total = max(1, spec.max_steps - spec.warmup_steps)
        progressed = max(0, min(total, step - spec.warmup_steps))
        return spec.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progressed / total))
    return spec.learning_rate


def _publish_checkpoint(
    root: Path,
    *,
    step: int,
    spec: RouterHealingRunSpec,
    optimizer: Any,
    tensors: Mapping[str, Any],
    torch: Any,
) -> dict[str, Any]:
    """Write a complete checkpoint directory, then expose it atomically.

    Files land in a ``.partial`` sibling and the directory is renamed into place
    after flushing every required file. This is atomic publication, not a claim
    of power-loss durability of the directory rename on every filesystem.
    Incomplete or existing artifacts are preserved, never overwritten.
    """
    from safetensors.torch import save_file

    root.mkdir(parents=True, exist_ok=True)
    final = root / f"step-{step}"
    partial = root / f".step-{step}.partial"
    if final.exists() or partial.exists():
        raise RuntimeError(f"checkpoint publication would overwrite existing state at {final}")
    partial.mkdir(parents=True)

    torch.save(optimizer.state_dict(), partial / "optimizer.pt")
    torch.save(
        {
            "kind": "router_healing_scheduler.v1",
            "global_step": int(step),
            "max_steps": spec.max_steps,
            "recipe_digest": spec.recipe_digest(),
        },
        partial / "scheduler.pt",
    )
    import numpy as np

    numpy_state = np.random.get_state()
    torch.save({
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        # Primitive lists keep this compatible with weights_only=True.
        "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
    }, partial / "rng_state.pth")
    (partial / "trainer_state.json").write_text(
        json.dumps({"global_step": int(step), "max_steps": spec.max_steps,
                    "tokens_consumed_total": step * spec.batch_size * spec.seq_len,
                    "samples_consumed_total": step * spec.batch_size}) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    save_file(
        {
            name: tensor.detach().to(torch.float32).cpu().contiguous()
            for name, tensor in tensors.items()
        },
        str(partial / "router_state.safetensors"),
    )
    manifest = {
        "kind": CHECKPOINT_KIND,
        "recipe_digest": spec.recipe_digest(),
        "max_tokens": spec.max_tokens,
        "parameter_paths": sorted(tensors),
        "files": {name: _sha256_file(partial / name) for name in CHECKPOINT_FILES},
    }
    (partial / "checkpoint_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    for name in (*CHECKPOINT_FILES, "checkpoint_manifest.json"):
        with (partial / name).open("r+b") as handle:
            os.fsync(handle.fileno())
    partial.rename(final)
    return {
        "directory": str(final),
        "global_step": int(step),
        "max_steps": spec.max_steps,
        "manifest_sha256": _sha256_file(final / "checkpoint_manifest.json"),
    }


def _read_checkpoint(spec: RouterHealingRunSpec, torch: Any) -> dict[str, Any]:
    """Refuse incomplete, altered, or differently bound state before model load."""
    from safetensors.torch import load_file

    root = Path(spec.resume_from)
    try:
        manifest = json.loads((root / "checkpoint_manifest.json").read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("kind") != CHECKPOINT_KIND:
            raise ValueError("missing versioned checkpoint binding")
        if manifest.get("recipe_digest") != spec.recipe_digest():
            raise ValueError("recipe mismatch (base, corpus, schedule, or training settings)")
        if manifest.get("max_tokens") != spec.max_tokens:
            raise ValueError("token budget mismatch")
        if set(manifest.get("files", {})) != set(CHECKPOINT_FILES):
            raise ValueError("incomplete checkpoint file inventory")
        for name in CHECKPOINT_FILES:
            if _sha256_file(root / name) != manifest["files"][name]:
                raise ValueError(f"content hash mismatch for {name}")
        trainer = json.loads((root / "trainer_state.json").read_text(encoding="utf-8"))
        if not isinstance(trainer, dict):
            raise ValueError("trainer state is not an object")
        step = trainer.get("global_step")
        if type(step) is not int or not 0 < step <= spec.max_steps:
            raise ValueError("invalid checkpoint step")
        if trainer != {
            "global_step": step, "max_steps": spec.max_steps,
            "tokens_consumed_total": step * spec.batch_size * spec.seq_len,
            "samples_consumed_total": step * spec.batch_size,
        }:
            raise ValueError("trainer step, horizon, or data position mismatch")
        scheduler = torch.load(root / "scheduler.pt", map_location="cpu", weights_only=True)
        if scheduler != {"kind": "router_healing_scheduler.v1", "global_step": step,
                         "max_steps": spec.max_steps, "recipe_digest": spec.recipe_digest()}:
            raise ValueError("scheduler state disagrees with the bound recipe and step")
        return {
            "manifest": manifest, "trainer": trainer,
            "rng": torch.load(root / "rng_state.pth", map_location="cpu", weights_only=True),
            "optimizer": torch.load(root / "optimizer.pt", map_location="cpu", weights_only=True),
            "tensors": load_file(str(root / "router_state.safetensors")),
        }
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        raise RuntimeError(f"checkpoint at {root} refused: {exc}") from exc


def train(spec: RouterHealingRunSpec) -> dict[str, Any]:
    """Run one bounded router-training attempt and return its measured result."""
    if spec.device not in QUALIFIED_DEVICES:
        raise RuntimeError(
            f"device {spec.device!r} is not qualified for router training; this worker "
            f"has only been qualified on CPU. Qualified: "
            f"{list(QUALIFIED_DEVICES)}"
        )
    if Path(spec.base_model_dir).resolve() == Path(spec.output_dir).resolve():
        raise RuntimeError("output_dir cannot be the base model directory")

    import torch
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(spec.seed)
    random.seed(spec.seed)
    np.random.seed(spec.seed % (2**32))
    synchronize = cuda_synchronize(torch)
    device = torch.device(spec.device)
    accelerator_count = 0 if device.type == "cpu" else 1
    started = time.perf_counter()
    restored_state = _read_checkpoint(spec, torch) if spec.resume_from else None

    base_identity = resolve_base_identity(spec.base_model_dir)
    if base_identity["content_sha256"] != spec.base_content_sha256:
        raise RuntimeError(
            "base identity mismatch: the spec was frozen against content "
            f"{spec.base_content_sha256!r} but {spec.base_model_dir} is "
            f"{base_identity['content_sha256']!r}. Refusing to train a router against a "
            "base nobody proved it belongs to."
        )

    model_load = PhaseTimer(synchronize=synchronize)
    with model_load:
        tokenizer = AutoTokenizer.from_pretrained(spec.base_model_dir, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            spec.base_model_dir, dtype=torch.float32, local_files_only=True
        )
        model.to(device)
        model.train()

    freeze_summary = freeze_for_router_healing(model, suffixes=ROUTER_ONLY_SUFFIXES)
    scope = assert_router_only_scope(freeze_summary.trainable_param_names, model)
    expected_paths, unknown_suffixes = resolve_expected_parameter_paths(model, ROUTER_ONLY_SUFFIXES)
    coverage = component_path_report(
        expected_paths,
        freeze_summary.trainable_param_names,
        unknown_suffixes=unknown_suffixes,
    )
    assert_components_qualified(coverage)

    inventory = tensor_inventory(model)
    quantization = quantization_reality_report(model, requested="none")

    text = _read_corpus(spec)
    encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
    blocks = _pack_blocks(list(encoded), spec.seq_len)
    order = torch.randperm(
        len(blocks), generator=torch.Generator().manual_seed(spec.seed)
    ).tolist()

    trainable_names = list(freeze_summary.trainable_param_names)
    trainable_set = set(trainable_names)
    parameters = dict((str(name), param) for name, param in model.named_parameters())
    trainable_params = [parameters[name] for name in trainable_names]
    frozen_names = [name for name in parameters if name not in trainable_set]

    optimizer = torch.optim.AdamW(trainable_params, lr=spec.learning_rate)

    start_step = 0
    source_inventory = None
    if spec.resume_from:
        source_inventory = inventory_checkpoint(spec.resume_from)
        assert_resumable(source_inventory, require_rng=True)
        if source_inventory.global_step is None:
            raise RuntimeError(
                f"the checkpoint at {spec.resume_from} records no global step, so the "
                "resume point is unknown"
            )
        start_step = int(source_inventory.global_step)
        if start_step >= spec.max_steps:
            raise RuntimeError(
                f"the checkpoint is already at step {start_step} of a declared horizon of "
                f"{spec.max_steps}; there is nothing left to train. A run with no steps "
                "cannot demonstrate trainability, so this is refused rather than "
                "reported as a no-op success."
            )
        restored = restored_state["tensors"]
        if (set(restored) != set(trainable_names)
                or restored_state["manifest"]["parameter_paths"] != sorted(trainable_names)):
            raise RuntimeError("checkpoint router parameter paths differ from the intended set")
        for name in trainable_names:
            tensor = restored[name]
            if (tensor.shape != parameters[name].shape or tensor.dtype != parameters[name].dtype
                    or not bool(torch.isfinite(tensor).all())):
                raise RuntimeError(f"checkpoint router tensor shape, dtype, or values invalid: {name}")
        optimizer.load_state_dict(restored_state["optimizer"])
        with torch.no_grad():
            for name in trainable_names:
                parameters[name].copy_(restored[name].to(parameters[name].device))

    probe = TrainabilityProbe(
        model,
        trainable_names,
        window_steps=max(1, min(spec.probe_window, spec.max_steps - start_step)),
        frozen_names=frozen_names,
    )
    if restored_state is not None:
        rng = restored_state["rng"]
        try:
            torch.set_rng_state(rng["torch"])
            random.setstate(rng["python"])
            numpy_state = rng["numpy"]
            np.random.set_state((numpy_state[0], np.array(numpy_state[1], dtype=np.uint32),
                                 *numpy_state[2:]))
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise RuntimeError(f"checkpoint RNG state cannot be restored: {exc}") from exc

    state = {
        "step": start_step,
        "tokens": 0,
        "losses": [],
        "losses_truncated": False,
        "stop_reason": "max_steps",
        "samples_consumed": 0,
        "step_trace": [],
    }
    checkpoint_seconds = 0.0
    checkpoints: list[dict[str, Any]] = []
    progress_path = Path(spec.output_dir) / "progress.json"
    first_step_seconds: dict[str, float | None] = {
        "forward": None,
        "backward": None,
        "update": None,
    }
    deadline = None if spec.max_seconds is None else started + spec.max_seconds
    batch_tokens = spec.batch_size * spec.seq_len

    Path(spec.output_dir).mkdir(parents=True, exist_ok=True)
    steady_state = PhaseTimer()
    steady_state.__enter__()
    try:
        for step in range(start_step, spec.max_steps):
            if deadline is not None and time.perf_counter() >= deadline:
                state["stop_reason"] = "max_seconds"
                break
            if step * batch_tokens + batch_tokens > spec.max_tokens:
                state["stop_reason"] = "max_tokens"
                break

            indices = [
                order[(step * spec.batch_size + offset) % len(order)]
                for offset in range(spec.batch_size)
            ]
            batch = torch.tensor([blocks[index] for index in indices], device=device)

            optimizer.zero_grad(set_to_none=True)
            forward_started = time.perf_counter()
            loss = model(input_ids=batch, labels=batch).loss
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"non-finite training loss at step {step}")
            if spec.detailed_timing and first_step_seconds["forward"] is None:
                first_step_seconds["forward"] = time.perf_counter() - forward_started

            backward_started = time.perf_counter()
            loss.backward()
            if spec.detailed_timing and first_step_seconds["backward"] is None:
                first_step_seconds["backward"] = time.perf_counter() - backward_started

            probe.record_gradients(step)

            learning_rate = _learning_rate_at(step, spec)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate

            update_started = time.perf_counter()
            optimizer.step()
            if spec.detailed_timing and first_step_seconds["update"] is None:
                first_step_seconds["update"] = time.perf_counter() - update_started

            probe.record_update(step)
            optimizer.zero_grad(set_to_none=True)

            state["step"] = step + 1
            state["tokens"] += batch_tokens
            state["samples_consumed"] += spec.batch_size
            value = float(loss.detach())
            if len(state["losses"]) < LOSS_LOG_LIMIT:
                state["losses"].append(value)
                state["step_trace"].append({"global_step": step + 1, "block_indices": indices,
                                            "learning_rate": learning_rate})
            else:
                state["losses_truncated"] = True

            if spec.checkpoint_dir and spec.checkpoint_every and (step + 1) % spec.checkpoint_every == 0:
                publication_started = time.perf_counter()
                checkpoints.append(
                    _publish_checkpoint(
                        Path(spec.checkpoint_dir),
                        step=step + 1,
                        spec=spec,
                        optimizer=optimizer,
                        tensors={name: parameters[name] for name in trainable_names},
                        torch=torch,
                    )
                )
                checkpoint_seconds += time.perf_counter() - publication_started

            if (step + 1) % PROGRESS_EVERY == 0:
                write_progress_best_effort(
                    {
                        "global_step": step + 1,
                        "max_steps": spec.max_steps,
                        "loss": value,
                        "tokens": state["tokens"],
                    },
                    progress_path,
                )
    finally:
        steady_state.__exit__(None, None, None)

    trainability = probe.assert_qualified()
    frozen = probe.assert_frozen_unchanged()

    publication_started = time.perf_counter()
    tensors = {name: parameters[name].detach() for name in trainable_names}
    payload = save_router_payload(
        tensors,
        Path(spec.output_dir) / "payload",
        base_content_sha256=spec.base_content_sha256,
        spec_digest=spec.digest(),
        steps_completed=int(state["step"] - start_step),
        recipe_digest=spec.recipe_digest(),
    )
    checkpoint_seconds += time.perf_counter() - publication_started
    checkpoint_timer = PhaseTimer()
    checkpoint_timer.seconds = checkpoint_seconds

    closeout_started = time.perf_counter()
    resume = (
        None
        if source_inventory is None
        else resume_witness(
            source_inventory,
            final_global_step=int(state["step"]),
            declared_max_steps=spec.max_steps,
        )
    )
    if resume is not None:
        resume["state_restoration"] = {
            "optimizer": True, "scheduler_position": True,
            "rng_streams": ["torch_cpu", "python", "numpy"],
            "recipe_digest": spec.recipe_digest(),
            "samples_consumed_before_resume": start_step * spec.batch_size,
            "checkpoint_manifest_sha256": _sha256_file(
                Path(spec.resume_from) / "checkpoint_manifest.json"
            ),
        }
    total_wall = time.perf_counter() - started
    ledger = training_lifecycle_ledger(
        accelerator_count=accelerator_count,
        model_load=model_load,
        checkpoint_publication=checkpoint_timer,
        steady_state_steps_seconds=steady_state.seconds,
        detailed_timing_enabled=spec.detailed_timing,
        first_forward_seconds=first_step_seconds["forward"],
        first_backward_seconds=first_step_seconds["backward"],
        first_update_seconds=first_step_seconds["update"],
        resumed_from_checkpoint=bool(spec.resume_from),
        closeout_seconds=time.perf_counter() - closeout_started,
    )

    losses = state["losses"]
    return {
        "kind": RESULT_KIND,
        "spec_digest": spec.digest(),
        "global_step": int(state["step"]),
        "steps_completed": int(state["step"] - start_step),
        "restored_global_step": int(start_step) if spec.resume_from else None,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "losses": losses,
        "step_trace": state["step_trace"],
        "losses_truncated": state["losses_truncated"],
        "loss_log_limit": LOSS_LOG_LIMIT,
        "limits": {
            "max_steps": spec.max_steps,
            "max_tokens": spec.max_tokens,
            "max_seconds": spec.max_seconds,
            "tokens_consumed": int(state["tokens"]),
            "stop_reason": state["stop_reason"],
            "samples_consumed": int(state["samples_consumed"]),
            "tokens_consumed_total": int(state["step"] * batch_tokens),
            "samples_consumed_total": int(state["step"] * spec.batch_size),
        },
        "lifecycle": ledger.to_dict(),
        "trainability": trainability,
        "frozen": frozen,
        "scope": scope,
        "coverage": coverage.to_dict(),
        "freeze_summary": freeze_summary.to_dict(),
        "payload": payload,
        "checkpoints": checkpoints,
        "resume": resume,
        "base_identity": base_identity,
        "tensor_inventory": inventory,
        "quantization_reality": quantization,
        "utilization": utilization_by_expert(None),
        "tokenizer": {
            "source": spec.base_model_dir,
            "vocab_size": int(getattr(tokenizer, "vocab_size", 0)),
            "class": type(tokenizer).__name__,
            "corpus_sha256": spec.corpus_sha256,
            "blocks": len(blocks),
        },
        "source_identity": chowder_source_identity(),
        "resource_usage": {
            "wall_seconds": total_wall,
            "active_accelerator_count": accelerator_count,
            "visible_accelerator_count": accelerator_count,
            "peak_vram_gb_by_accelerator": {},
            "sampling_device": sampling_device(torch),
        },
        "model": {
            "class": type(model).__name__,
            "parameters": int(sum(p.numel() for p in model.parameters())),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument(
        "--chowder-identity",
        default=None,
        help="JSON file with the chowder source identity the controller declared; "
        "verified against the code this process actually imported BEFORE the spec is "
        "read, so a wrong-checkout worker refuses instead of training",
    )
    args = parser.parse_args()

    # P4c: nothing may be loaded, run, or written before the pin checks out.
    from ..worker_env import verify_source_identity

    if args.chowder_identity is not None:
        verify_source_identity(
            json.loads(Path(args.chowder_identity).read_text(encoding="utf-8"))
        )
    else:
        print(
            "WARNING: no --chowder-identity supplied; the worker's source identity is "
            "unverified for this run",
            file=sys.stderr,
        )

    spec_data = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    spec = RouterHealingRunSpec(**spec_data)
    result = train(spec)
    Path(args.result).write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
