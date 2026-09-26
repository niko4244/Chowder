from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterator

from ..adapter_guard import assert_adapter_is_live
from ..contamination import write_holdout_fingerprint_index
from ..hf_resilience import cache_status, with_hub_retries
from ..local_model_compat import patch_transformers5_custom_model
from ..lifecycle import (
    PhaseTimer,
    cuda_synchronize,
    evaluation_lifecycle_ledger,
    sampling_device,
)
from .generation import observed_span, resolve_eos_token_ids
from .rendering import render_prompt
from .scoring import (
    SAMPLE_SEPARATOR,
    final_answer,
    final_number,
    normalize,
    observed_score,
    reasoning_answer,
    score,
)
from .vram import MemorySampler, peak_vram as _peak_vram
from .placement import (
    dispatch_offloaded,
    needs_redispatch_after_adapter,
    placement_note,
)
from .transformers_text import (
    EvalSuiteSpec,
    TransformersTextEvalSpec,
    suite_execution_evidence,
)


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


#: Scoring lives in `.scoring` so both workers cannot drift apart again. This worker
#: used to score the RAW generation while base_text_worker discarded an unclosed
#: <think> block first, which meant Chowder's automatic baseline and its candidate
#: were not scored by the same rule. See that module.
#: Re-exported under the historical private names for existing callers.
_normalize = normalize
_final_answer = final_answer
_reasoning_answer = reasoning_answer
_final_number = final_number
_score = score


def _resolve_dtype(torch: Any, precision: str):
    if precision == "fp32":
        return torch.float32
    if precision == "bf16":
        if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 requested but the active CUDA device does not support bf16")
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def _resolve_device(torch: Any, requested: str) -> str:
    requested = requested.strip().lower()
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"{requested} requested but CUDA is unavailable")
    return requested


def _batches(rows: list[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    """Split a suite's rows into the declared generation batches."""
    if size < 1:
        raise RuntimeError(f"evaluation batch_size must be at least 1, got {size}")
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def _load_rows(suite: EvalSuiteSpec) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(suite.dataset).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RuntimeError(f"{suite.dataset}:{line_number} is not a JSON object")
            if suite.prompt_field not in row or suite.expected_field not in row:
                raise RuntimeError(
                    f"{suite.dataset}:{line_number} missing {suite.prompt_field!r} or {suite.expected_field!r}"
                )
            rows.append(row)
    if not rows:
        raise RuntimeError(f"evaluation suite {suite.name!r} is empty")
    return rows


def evaluate(spec: TransformersTextEvalSpec) -> dict[str, Any]:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    except ImportError as exc:
        raise RuntimeError("evaluation dependencies are missing; install chowder-ai[train]") from exc

    if spec.trust_remote_code:
        raise RuntimeError("trust_remote_code is disabled")
    if spec.local_custom_code_digests is not None:
        patch_transformers5_custom_model(spec.base_model, spec.local_custom_code_digests)
    device_name = _resolve_device(torch, spec.device)
    if spec.quantization == "4bit" and not device_name.startswith("cuda"):
        raise RuntimeError("4-bit evaluation requires a CUDA device")

    dtype = _resolve_dtype(torch, spec.precision)
    # P6: generation dominated the completed rerun's cost (1.16 + 1.86 GPU-hours
    # against 0.44 for the 500 steps), and neither evaluation arm reported its
    # own timing, so that cost was invisible in the artifacts.
    load_timer = PhaseTimer(synchronize=cuda_synchronize(torch))
    load_timer.__enter__()
    set_seed(spec.seed)
    model_cache_status = cache_status(spec.base_model, spec.revision)
    tokenizer = with_hub_retries(
        lambda: AutoTokenizer.from_pretrained(
            spec.base_model,
            revision=spec.revision,
            trust_remote_code=spec.local_custom_code_digests is not None,
            local_files_only=spec.offline,
        ),
        label=f"tokenizer download for {spec.base_model}",
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("tokenizer has neither pad_token nor eos_token")
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": spec.local_custom_code_digests is not None,
        "dtype": dtype,
        "local_files_only": spec.offline,
    }
    if spec.revision is not None:
        model_kwargs["revision"] = spec.revision
    if spec.quantization == "4bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        device_index = int(device_name.split(":", 1)[1]) if ":" in device_name else 0
        model_kwargs["device_map"] = {"": device_index}

    base = with_hub_retries(
        lambda: AutoModelForCausalLM.from_pretrained(spec.base_model, **model_kwargs),
        label=f"model download for {spec.base_model}",
    )
    resolved_commit = getattr(base.config, "_commit_hash", None)
    if spec.quantization == "none":
        if spec.placement == "offload":
            base = dispatch_offloaded(base, device_name)
        else:
            base = base.to(device_name)
    adapter_liveness: dict[str, Any] | None = None
    if spec.adapter_dir is None:
        model = base
    else:
        model = PeftModel.from_pretrained(base, spec.adapter_dir, is_trainable=False)
        # Refuse to score an adapter that cannot change the model. PEFT only
        # warns when no saved key matches, leaving every LoRA B at zero.
        adapter_liveness = assert_adapter_is_live(model, spec.adapter_dir)
        base = placement_after_adapter(base, spec=spec, device_name=device_name)
    model.eval()
    if spec.placement == "offload":
        # Reported per run: "offload" means nothing unless the dense weights
        # demonstrably live on the CPU while generation runs.
        print(f"placement: offload active ({placement_note(model)})", flush=True)
    device = next(model.parameters()).device
    resolved_eos_token_id = resolve_eos_token_ids(tokenizer, model)
    load_timer.__exit__()

    output_dir = Path(spec.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, float] = {}
    suite_evidence: dict[str, Any] = {}

    generation_timer = PhaseTimer(synchronize=cuda_synchronize(torch))
    memory_sampler = MemorySampler(device_name=sampling_device(torch))
    memory_sampler.start()
    generation_timer.__enter__()
    with torch.inference_mode():
        for suite in spec.suites:
            rows = _load_rows(suite)
            fingerprint_path = output_dir / f"holdout-fingerprints-{suite.name}.jsonl"
            fingerprint_digest = write_holdout_fingerprint_index(
                (
                    (str(row[suite.prompt_field]), str(row[suite.expected_field]))
                    for row in rows
                ),
                fingerprint_path,
            )

            correct = 0.0
            predictions_path = output_dir / f"predictions-{suite.name}.jsonl"
            with predictions_path.open("w", encoding="utf-8", newline="\n") as output:
                for chunk in _batches(rows, suite.batch_size):
                    rendered_batch: list[tuple[str, str]] = []
                    for row in chunk:
                        prompt = str(row[suite.prompt_field])
                        expected = str(row[suite.expected_field])
                        # One renderer for both text workers (see
                        # evaluators/rendering.py). This arm previously ignored
                        # `canonical_rendering` entirely: a suite asking for the
                        # pinned template silently rendered through the
                        # checkpoint's own instead, so baseline and candidate
                        # scored different prompt bytes under one protocol entry.
                        rendered, render_evidence = render_prompt(
                            tokenizer=tokenizer,
                            prompt=prompt,
                            suite_name=suite.name,
                            use_chat_template=suite.use_chat_template,
                            canonical_rendering=suite.canonical_rendering,
                        )
                        rendered_batch.append((prompt, expected, rendered))
                    # padding=True pads with the tokenizer's pad token on the
                    # declared side; the mask it returns is what keeps the pads
                    # out of attention, so a padded row sees the same context it
                    # would have seen alone.
                    encoded = tokenizer(
                        [item[2] for item in rendered_batch],
                        return_tensors="pt",
                        padding=len(rendered_batch) > 1,
                    )
                    encoded = {key: value.to(device) for key, value in encoded.items()}
                    width = int(encoded["input_ids"].shape[1])
                    # Self-consistency (n_samples > 1): K temperature-sampled
                    # chains per row, seeded by spec.seed so a rerun of the same
                    # protocol reproduces the same rows. Greedy decoding is the
                    # n_samples == 1 path and stays exactly as it was.
                    sampled = suite.n_samples > 1
                    generation_kwargs: dict[str, Any] = {
                        "max_new_tokens": suite.max_new_tokens,
                        "pad_token_id": tokenizer.pad_token_id,
                        "eos_token_id": resolved_eos_token_id,
                    }
                    if sampled:
                        generation_kwargs.update(do_sample=True, temperature=suite.temperature)
                    else:
                        generation_kwargs["do_sample"] = False
                    chain_texts: list[list[str]] = [[] for _ in rendered_batch]
                    chain_observations: list[list[dict[str, Any]]] = [
                        [] for _ in rendered_batch
                    ]
                    for _pass in range(suite.n_samples):
                        generated = model.generate(**encoded, **generation_kwargs)
                        for index, _ in enumerate(rendered_batch):
                            # What the generation did, recorded rather than
                            # re-derived: the completion text cannot say whether
                            # it stopped on EOS or ran into the cap, and the
                            # generation diagnostics (the campaign's target
                            # instrument) are defined over exactly that.
                            own_tokens, produced, stopped = observed_span(
                                continuation=generated[index, width:].tolist(),
                                max_new_tokens=suite.max_new_tokens,
                                # Only a multi-row call can have padded this row
                                # after it finished.
                                pad_token_id=(
                                    tokenizer.pad_token_id
                                    if len(rendered_batch) > 1
                                    else None
                                ),
                            )
                            chain_texts[index].append(
                                tokenizer.decode(own_tokens, skip_special_tokens=True)
                            )
                            chain_observations[index].append(
                                {
                                    "generated_tokens": max(0, produced),
                                    "eos_terminated": bool(
                                        stopped and resolved_eos_token_id is not None
                                    ),
                                }
                            )
                    for index, (prompt, expected, _) in enumerate(rendered_batch):
                        prediction = SAMPLE_SEPARATOR.join(chain_texts[index])
                        observation = (
                            # Per-chain diagnostics for a sampled row; the
                            # aggregate numbers describe the WORST chain, which
                            # is the one that determines whether the vote had a
                            # full panel to draw from.
                            {
                                "generated_tokens": max(
                                    int(c["generated_tokens"])
                                    for c in chain_observations[index]
                                ),
                                "eos_terminated": all(
                                    c["eos_terminated"]
                                    for c in chain_observations[index]
                                ),
                                "chain_observations": chain_observations[index],
                            }
                            if sampled
                            else chain_observations[index][0]
                        )
                        observed = observed_score(observation, suite.scoring)
                        row_score = (
                            observed
                            if observed is not None
                            else _score(prediction, expected, suite.scoring)
                        )
                        correct += row_score
                        record = {
                            "prompt": prompt,
                            "expected": expected,
                            "prediction": prediction,
                            "score": row_score,
                            **observation,
                        }
                        if suite.store_chains and sampled:
                            record["chains"] = chain_texts[index]
                        output.write(
                            json.dumps(
                                record,
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        output.flush()
            metrics[suite.name] = correct / len(rows)
            suite_evidence[suite.name] = {
                **suite_execution_evidence(suite, rows),
                # Execution evidence, not protocol identity: what it took to
                # produce these rows, so a batched arm and a single-row arm are
                # distinguishable in the artifact even though the declared
                # decoding is the same.
                "scoring": suite.scoring,
                "predictions_file": str(predictions_path),
                "holdout_fingerprints_file": str(fingerprint_path),
                "holdout_fingerprints_sha256": fingerprint_digest,
                "resolved_eos_token_id": resolved_eos_token_id,
                **render_evidence,
            }

    # The candidate arm's own generation, timed and sampled separately from the
    # baseline's -- one arm cannot measure the other, and the ledger says so.
    generation_timer.__exit__()
    memory_sampling = memory_sampler.stop()
    lifecycle_data = evaluation_lifecycle_ledger(
        accelerator_count=1 if device_name.startswith("cuda") else 0,
        arm="candidate",
        generation_seconds=generation_timer.seconds,
        model_load_seconds=load_timer.seconds,
    ).to_dict()

    return {
        "metrics": metrics,
        "suites": suite_evidence,
        "runtime": {
            "device": device_name,
            "gpu_count": 1 if device_name.startswith("cuda") else 0,
            "placement": spec.placement,
            "lifecycle": lifecycle_data,
            "memory_sampling": memory_sampling,
            # The training workers have always reported this; the evaluators did
            # not, and a pre-registered "peak VRAM under budget" condition was
            # therefore undecidable for the evaluation leg. Judging it from
            # nvidia-smi instead measures the whole MACHINE -- every browser and
            # service on it -- and that is what produced a spurious
            # oversubscription FAIL (docs/PRUNED_9B_RERUN_RESULT.md). A run must be
            # able to answer "how much VRAM did *I* use" from its own artifacts.
            **_peak_vram(device_name),
        },
        "model_provenance": {
            "requested_base_model": spec.base_model,
            "requested_revision": spec.revision,
            "model_cache_status": model_cache_status,
            "resolved_model_commit": resolved_commit,
            # "an adapter directory was requested" is NOT "an adapter is in
            # effect": PeftModel.from_pretrained succeeds on a total key mismatch.
            # This now reports the measured check, not the request.
            "adapter_requested": spec.adapter_dir is not None,
            "adapter_loaded": adapter_liveness is not None,
            "adapter_liveness": adapter_liveness,
        },
        "versions": {
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
            "peft": _package_version("peft"),
            "bitsandbytes": _package_version("bitsandbytes"),
        },
    }


def placement_after_adapter(base: Any, *, spec: Any, device_name: str) -> Any:
    """Re-assert the declared placement once an adapter has been attached.

    The adapter wrapper re-places the model it wraps, which silently turns a
    bounded arm measurement into an unbounded one (see
    :func:`chowder.evaluators.placement.needs_redispatch_after_adapter` for the
    measurement). The decision lives in that function and this one only applies
    it, so there is exactly one owner of "does the placement need re-applying".
    """
    if not needs_redispatch_after_adapter(
        quantization=spec.quantization,
        placement=spec.placement,
        adapter=True,
    ):
        return base
    # Re-asserted on the bare base the wrapper holds, which is the module graph
    # the adapter's LoRA layers were injected into: the wrapper generates through
    # it, so the placement applies to the model that will actually run.
    return dispatch_offloaded(base, device_name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument(
        "--chowder-identity",
        default=None,
        help="JSON file with the chowder source identity the controller declared; "
        "verified against the code this process actually imported BEFORE the "
        "spec is read, so a wrong-checkout worker refuses instead of scoring",
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
            "WARNING: no --chowder-identity supplied; the worker's source "
            "identity is unverified for this run",
            file=sys.stderr,
        )

    raw = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    raw["suites"] = tuple(EvalSuiteSpec(**suite) for suite in raw["suites"])
    spec = TransformersTextEvalSpec(**raw)
    result = evaluate(spec)
    Path(args.result).write_bytes(
        (json.dumps(result, sort_keys=True, indent=2) + "\n").encode("utf-8")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
