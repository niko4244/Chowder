"""Bounded, real-original BF16 CUDA HF-reference / REAP-observer parity probe.

The frozen preregistration contains NO numerical CUDA-parity acceptance gate.
This tool records exact BF16 equality and actual differences at each decoder
layer, final logits and NLL; it never promotes those results into a D0/G0/G1
claim or chooses a new scientific tolerance. It uses the *unmodified* Hugging
Face decoder as the reference and the same actual original layer weights with
the REAP observer expert hook as the comparison. Neither pass prunes experts.

It operates on the first N tokens of frozen calibration row zero, NOT on the
full 704x40 population. Thus a numerical match is a bounded forward check.
The sealed registered-original 704x40 observer receipt is required as input;
its historical source SHA-256 hashes are independently pinned upstream. This
invocation checks source sizes/metadata and pre/post timestamps but DOES NOT
rehash ~72 GB of weight bytes; that provenance limit is recorded explicitly.

Default is a read-only preflight with *no CUDA initialization or model load*:

  python -m chowder_batch.reap_cuda_parity --model-dir MODEL --weights-dir SHARDS \
    --calibration CALIBRATION.jsonl --observation OBSERVER_DIR --out NEW.json

To run on a separately admitted, otherwise idle Windows CUDA host, repeat
the same command with --execute. That collects 3 passive host observations
30 seconds apart, refuses protected service conflicts and checks host health
after every layer. Only the requested GPU is used; no services are changed.
The receipt is written exclusively after all 40 layers and final comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Callable

from chowder_batch.reap_host_guard import (
    assess_admission,
    assess_runtime,
    collect_snapshot,
)
from chowder_batch.reap_original_bf16_pin import (
    REVISION,
    validate_registered_source_metadata,
)
from chowder_batch.reap_v2_source_identity import FROZEN_V2_BASE_SHA256

SCHEMA = "chowder.reap-original-bf16-cuda-observer-parity.v1"
MAX_TOKENS = 64
MAX_CALIBRATION_BYTES = 16 * 1024 * 1024
MAX_OBSERVATION_BYTES = 16 * 1024 * 1024


def sample_admission(
    *, snapshot_fn: Callable = collect_snapshot,
    sleep_fn: Callable = time.sleep,
) -> list:
    """60 seconds of passive host probes (no GPU context or service actions)."""
    samples = [snapshot_fn()]
    for _ in range(2):
        sleep_fn(30)
        samples.append(snapshot_fn())
    return samples


def admission_decision(samples: list):
    """Reuse the frozen observer's proven host admission policy."""
    return assess_admission(samples)


def runtime_decision(samples: list):
    return assess_runtime(samples)


def _digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _require_disjoint_new_output(
    output: Path, *, model: Path, weights: Path,
    calibration: Path, observation: Path,
) -> None:
    if output.exists() or not output.parent.is_dir() or output.is_symlink():
        raise ValueError("parity receipt must be new in an existing directory")
    target = output.resolve()
    for protected in (model.resolve(), weights.resolve(), calibration.resolve(),
                      observation.resolve()):
        if target == protected or target.is_relative_to(protected) or (
            protected.is_relative_to(target)
        ):
            raise ValueError("parity receipt path overlaps original input evidence")


def _read_json_bounded(path: Path, max_bytes: int) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > max_bytes:
        raise ValueError(f"unsafe or oversized provenance file: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON receipt object")
    return value


def publish_receipt(path: Path, result: dict) -> None:
    """Durable same-directory publish without overwriting another receipt.

    A hard link acts as an atomic exclusive create on NTFS and POSIX. If the
    volume does not support hard links, refuse instead of publishing partially
    written JSON or replacing existing experiment evidence.
    """
    raw = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def original_provenance(
    *, model: Path, weights: Path, observation: Path,
    calibration: Path,
) -> dict:
    """Check historical fully hashed source and its sealed registered observer.

    Current weight BYTES are only checked by the earlier observer receipt,
    not rehashed in this invocation. Do not report a fresh-weight-byte proof.
    """
    from chowder_batch.reap_hidden_checkpoint import validate_observation_receipt

    recorded = _read_json_bounded(observation / "observation.json", MAX_OBSERVATION_BYTES)
    validate_observation_receipt(
        observation, recorded, expected_generation=40,
        require_registered_original=True,
    )
    identity = recorded["checkpoint_receipt"]["source_identity"]
    if (Path(identity["model_dir"]).resolve() != model.resolve()
            or Path(identity["weights_dir"]).resolve() != weights.resolve()
            or identity["calibration_sha256"] != FROZEN_V2_BASE_SHA256):
        raise ValueError("source or calibration paths disagree with sealed original receipt")
    validated = validate_registered_source_metadata(model, weights)
    if validated != identity["trusted_upstream_pin"]:
        raise ValueError("registered original model metadata does not match original receipt")
    reference_spec = importlib.util.find_spec(
        "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe"
    )
    if reference_spec is None or reference_spec.origin is None:
        raise RuntimeError("cannot identify installed HF reference implementation")
    reference_hash = _digest_file(Path(reference_spec.origin))
    hook_hash = _digest_file(Path(__file__).with_name("reap_observe.py"))
    original_sources = identity.get("source_sha256", {})
    if (original_sources.get("modeling_qwen3_5_moe.py") != reference_hash
            or original_sources.get("reap_observe.py") != hook_hash):
        raise ValueError("HF decoder or observer hook changed since original 704x40 observation")
    if not math.isfinite(float(recorded.get("calibration_ppl", math.nan))):
        raise ValueError("original BF16 full observation has no finite measured D0")
    if recorded["calibration_ppl"] >= 20:
        raise ValueError("original BF16 frozen Gate D0 did not pass")
    if _digest_file(calibration) != FROZEN_V2_BASE_SHA256:
        raise ValueError("calibration differs from frozen original observer")
    return {
        "registered_original_revision": REVISION,
        "upstream": validated,
        "observer_path": str(observation.resolve()),
        "observation_sha256": _digest_file(observation / "observation.json"),
        "checkpoint_source_identity_sha256": recorded["checkpoint_receipt"]["source_identity_sha256"],
        "hf_reference_sha256": reference_hash,
        "observer_hook_sha256": hook_hash,
        "original_observer_gate_d0_ppl": recorded["calibration_ppl"],
        "source_weight_verification": (
            "historical registered-original observer fully hashed all 26 shards; "
            "this invocation checks current shard names/sizes and metadata, "
            "plus file stat stability, without freshly hashing 72GB of weights"
        ),
        "first_sequence_length": identity["lengths"][0],
    }


def frozen_prefix(calibration: Path, tokenizer, *, limit: int) -> list[int]:
    if type(limit) is not int or not 2 <= limit <= MAX_TOKENS:
        raise ValueError(f"parity token limit must be 2..{MAX_TOKENS}")
    if (calibration.is_symlink() or not calibration.is_file()
            or calibration.stat().st_size > MAX_CALIBRATION_BYTES):
        raise ValueError("frozen calibration unavailable or oversized")
    raw = calibration.read_bytes()
    if hashlib.sha256(raw).hexdigest() != FROZEN_V2_BASE_SHA256:
        raise ValueError("parity requires frozen 704-row calibration bytes")
    rows = [line for line in raw.splitlines() if line.strip()]
    if len(rows) != 704:
        raise ValueError("frozen calibration does not have 704 rows")
    first = json.loads(rows[0])
    if not isinstance(first, dict) or not isinstance(first.get("text"), str):
        raise ValueError("frozen calibration first row invalid")
    ids = tokenizer(first["text"], add_special_tokens=False)["input_ids"]
    if (not isinstance(ids, list) or len(ids) < limit
            or not all(type(item) is int and item >= 0 for item in ids)):
        raise ValueError("tokenizer returned invalid frozen sequence")
    return ids[:limit]


def tensor_difference(reference, observed) -> dict:
    """Pure descriptive comparison: exact BF16 equality, no tunable gate."""
    import torch

    if (reference.shape != observed.shape or reference.dtype != observed.dtype):
        raise ValueError("reference and observer tensor shape/dtype differ")
    if not bool(torch.isfinite(reference).all()) or not bool(torch.isfinite(observed).all()):
        raise ValueError("non-finite reference or observer tensor")
    a, b = reference.float(), observed.float()
    delta = (a - b).abs()
    unequal = reference != observed
    return {
        "shape": list(reference.shape),
        "dtype": str(reference.dtype),
        "exact_equal": not bool(unequal.any()),
        "different_elements": int(unequal.sum().item()),
        "max_abs_diff": float(delta.max().item()),
        "mean_abs_diff": float(delta.mean().item()),
        "reference_max_abs": float(a.abs().max().item()),
    }


def compare_decoder_layer(
    *, layer, reference, observed, config, hf_modeling, layer_id: int,
) -> tuple:
    """Actual Hugging Face decoder forward, then observer hook, same weights.

    Both reference and observed streams are propagated independently to expose
    accumulated mismatch. Restore the reference expert method in ``finally``.
    """
    import torch
    from chowder_batch.reap_observe import observing_experts_forward

    if (reference.shape != observed.shape or reference.dtype != torch.bfloat16
            or observed.dtype != torch.bfloat16):
        raise ValueError("BF16 reference/observed hidden-state inputs differ")
    experts = layer.mlp.experts
    if "forward" in vars(experts):
        raise ValueError("HF reference experts must start uninstrumented")
    n_tokens = reference.shape[1]
    positions = torch.arange(n_tokens, device=reference.device).view(1, -1)
    rotary = hf_modeling.Qwen3_5MoeTextRotaryEmbedding(config).to(reference.device)
    stats = {
        key: torch.zeros(config.num_experts, device=reference.device,
                         dtype=torch.float64)
        for key in ("count", "gate_sum", "norm_sum", "reap_sum")
    }

    def forward(x):
        embeddings = rotary(x, positions[None].expand(3, 1, -1))
        if layer.block_type == "full_attention":
            mask = hf_modeling.create_causal_mask(
                config=config, inputs_embeds=x, attention_mask=None,
                past_key_values=None, position_ids=positions,
            )
        else:
            mask = hf_modeling.create_recurrent_attention_mask(
                config=config, inputs_embeds=x, attention_mask=None,
            )
        return layer(x, position_embeddings=embeddings,
                     attention_mask=mask, position_ids=positions)

    with torch.inference_mode():
        y_ref = forward(reference)
        try:
            experts.forward = observing_experts_forward(experts, stats)
            y_obs = forward(observed)
        finally:
            if "forward" in vars(experts):
                del experts.forward
    diff = tensor_difference(y_ref, y_obs)
    count = int(stats["count"].sum().item())
    if count != n_tokens * config.num_experts_per_tok:
        raise ValueError(f"layer {layer_id} REAP instrumentation routed-token mismatch")
    if any(not bool(torch.isfinite(v).all()) for v in stats.values()):
        raise ValueError(f"layer {layer_id} non-finite REAP observer saliency")
    evidence = {
        "layer": layer_id,
        "block_type": layer.block_type,
        "routed_tokens": count,
        "experts_hit": int((stats["count"] > 0).sum().item()),
        "activation_difference": diff,
    }
    return y_ref.detach(), y_obs.detach(), evidence


def _source_stamps(model: Path, weights: Path) -> dict[str, tuple[int, int]]:
    from chowder_batch.reap_original_bf16_pin import load_original_pin

    pin = load_original_pin()
    return {
        name: (p.stat().st_size, p.stat().st_mtime_ns)
        for location, inventory in ((model, pin["model_metadata"]),
                                    (weights, pin["weight_shards"]))
        for name in inventory
        for p in [location / name]
    }


def _safe_runtime_snapshot(prior: list) -> list:
    last = collect_snapshot()
    samples = [*prior[-3:], last]
    decision = runtime_decision(samples)
    if decision.action != "CONTINUE":
        raise RuntimeError("CUDA parity interrupted by host guard: " +
                           ", ".join(decision.reasons))
    return samples


def run_cuda_parity(
    *, model: Path, weights: Path, calibration: Path,
    source: dict, token_limit: int, samples: list,
) -> dict:
    """Run only after admission; loads actual BF16 weights on GPU sequentially."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoConfig, AutoTokenizer
    import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as M
    from chowder_batch.reap_observe import (
        PREFIX, ShardReader, _load_layer_without_duplicate_weights,
        _next_token_nll_chunked,
    )
    from chowder_batch.reap_hidden_checkpoint import runtime_identity

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable after admission")
    cuda = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(cuda)
    if not properties.major >= 8:  # hardware BF16 requirement, not scientific gate
        raise RuntimeError("CUDA device has no native BF16 support")
    runtime = runtime_identity(properties)
    cfg = AutoConfig.from_pretrained(model, local_files_only=True,
                                     trust_remote_code=False).text_config
    if (cfg.num_hidden_layers != 40 or cfg.hidden_size != 2048
            or cfg.num_experts != 256 or cfg.num_experts_per_tok != 8):
        raise ValueError("expected original frozen 40-layer/256-expert BF16 architecture")
    cfg._attn_implementation = "sdpa"
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True,
                                              trust_remote_code=False)
    tokens = frozen_prefix(calibration, tokenizer, limit=token_limit)
    if source["first_sequence_length"] != len(
        tokenizer(json.loads(calibration.read_bytes().splitlines()[0])["text"],
                  add_special_tokens=False)["input_ids"]
    ):
        raise ValueError("real tokenizer disagrees with sealed original observer sequence")
    token_ids = torch.tensor(tokens, dtype=torch.long, device=cuda)
    reader = ShardReader(model)
    reader.dir = weights
    stamps = _source_stamps(model, weights)
    records = []
    start = time.monotonic()
    with torch.inference_mode():
        embed = reader.load(PREFIX + "embed_tokens.", str(cuda))["weight"]
        if embed.dtype != torch.bfloat16:
            raise ValueError("original embedding checkpoint is not BF16")
        x_ref = F.embedding(token_ids, embed).unsqueeze(0).cpu()
        x_obs = x_ref.clone()
        del embed
        torch.cuda.empty_cache()
        for layer_id in range(40):
            samples = _safe_runtime_snapshot(samples)
            state = reader.load(PREFIX + f"layers.{layer_id}.", str(cuda))
            if any(v.dtype != torch.bfloat16 for v in state.values()):
                raise ValueError(f"original layer {layer_id} has non-BF16 tensor")
            layer = _load_layer_without_duplicate_weights(
                M.Qwen3_5MoeDecoderLayer, cfg, layer_id, torch.bfloat16,
                state,
            ).eval()
            del state
            y_ref, y_obs, result = compare_decoder_layer(
                layer=layer, reference=x_ref.to(cuda), observed=x_obs.to(cuda),
                config=cfg, hf_modeling=M, layer_id=layer_id,
            )
            x_ref, x_obs = y_ref.cpu(), y_obs.cpu()
            records.append(result)
            del layer, y_ref, y_obs
            torch.cuda.empty_cache()
            samples = _safe_runtime_snapshot(samples)
        norm_w = reader.load(PREFIX + "norm.", str(cuda))["weight"]
        norm = M.Qwen3_5MoeRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).to(cuda)
        norm.weight.data.copy_(norm_w)
        head = reader.load("lm_head.", str(cuda))["weight"]
        if head.dtype != torch.bfloat16:
            raise ValueError("original BF16 vocabulary head dtype mismatch")
        ref_logits = F.linear(norm(x_ref.to(cuda))[0, -1:], head).float()
        obs_logits = F.linear(norm(x_obs.to(cuda))[0, -1:], head).float()
        logits_diff = tensor_difference(ref_logits, obs_logits)
        predicted = {
            "reference": int(ref_logits.argmax(-1).item()),
            "observer": int(obs_logits.argmax(-1).item()),
        }
        ref_nll, ref_count = _next_token_nll_chunked(
            norm, x_ref, token_ids.cpu(), head,
        )
        obs_nll, obs_count = _next_token_nll_chunked(
            norm, x_obs, token_ids.cpu(), head,
        )
    if ref_count != obs_count or not all(math.isfinite(v) for v in (ref_nll, obs_nll)):
        raise RuntimeError("CUDA parity NLL was non-finite or scored different positions")
    if _source_stamps(model, weights) != stamps:
        raise RuntimeError("original checkpoint files changed during CUDA parity measurement")
    if _digest_file(calibration) != FROZEN_V2_BASE_SHA256:
        raise RuntimeError("frozen calibration changed during CUDA parity measurement")
    _safe_runtime_snapshot(samples)
    all_exact = all(row["activation_difference"]["exact_equal"] for row in records)
    all_exact = all_exact and logits_diff["exact_equal"]
    return {
        "schema": SCHEMA,
        "status": "MEASURED_EXACT_BF16_MATCH" if all_exact else "MEASURED_NUMERIC_DIFFERENCES",
        "scientific_gate": "NOT_REGISTERED; DESCRIPTIVE CUDA NUMERIC QUALIFICATION ONLY",
        "scope": "first bounded frozen calibration row, actual original BF16, all 40 layers",
        "model_dir": str(model.resolve()),
        "weights_dir": str(weights.resolve()),
        "source_provenance": source,
        "cuda_runtime": runtime,
        "tokenization": {"row": 0, "prefix_tokens": tokens,
                          "sampled_positions": len(tokens),
                          "sampled_next_token_positions": ref_count},
        "reference_forward": "unmodified transformers Qwen3_5MoeDecoderLayer BF16",
        "observer_forward": "same layer tensors with observing_experts_forward hook",
        "layers": records,
        "final_logits_difference": logits_diff,
        "greedy_next_token_id": predicted,
        "next_token_nll": {"reference": ref_nll, "observer": obs_nll,
                           "observed_minus_reference": obs_nll - ref_nll},
        "elapsed_seconds": round(time.monotonic() - start, 3),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, default=None)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--observation", type=Path, required=True,
                        help="directory with finished registered original 704x40 observation")
    parser.add_argument("--out", type=Path, required=True, help="new immutable JSON receipt")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--execute", action="store_true",
                        help="opt in to 60s passive admission and real 40-layer CUDA workload")
    args = parser.parse_args(argv)
    weights = args.weights_dir or args.model_dir
    try:
        _require_disjoint_new_output(
            args.out, model=args.model_dir, weights=weights,
            calibration=args.calibration, observation=args.observation,
        )
        if not 2 <= args.max_tokens <= MAX_TOKENS:
            raise ValueError(f"--max-tokens must be 2..{MAX_TOKENS}")
        source = original_provenance(
            model=args.model_dir, weights=weights, calibration=args.calibration,
            observation=args.observation,
        )
        if not args.execute:
            print(json.dumps({
                "status": "PREFLIGHT_ONLY_CUDA_NOT_MEASURED",
                "source": source, "next": "add --execute on an idle admitted host",
            }, sort_keys=True))
            return 0
        probes = sample_admission()
        decision = admission_decision(probes)
        if decision.action != "ADMIT":
            raise RuntimeError("CUDA parity refused by host admission: " +
                               ", ".join(decision.reasons))
        result = run_cuda_parity(
            model=args.model_dir, weights=weights,
            calibration=args.calibration, source=source,
            token_limit=args.max_tokens, samples=probes,
        )
        publish_receipt(args.out, result)
        print(json.dumps({"status": result["status"], "receipt": str(args.out)}))
        return 0 if result["status"] == "MEASURED_EXACT_BF16_MATCH" else 1
    except (OSError, RuntimeError, ValueError, ImportError, KeyError,
            TypeError, OverflowError) as exc:
        print(f"original BF16 CUDA parity refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
