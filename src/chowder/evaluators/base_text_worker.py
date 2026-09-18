from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from ..contamination import write_holdout_fingerprint_index
from ..hf_resilience import cache_status, with_hub_retries
from ..lifecycle import (
    PhaseTimer,
    cuda_synchronize,
    evaluation_lifecycle_ledger,
    sampling_device,
)
from .base_text import BaseTextEvalSpec
from .generation import observed_generation, resolve_eos_token_ids
from .rendering import render_prompt
from .scoring import final_answer, final_number, normalize, observed_score, score
from .vram import MemorySampler, peak_vram as _peak_vram
from .placement import dispatch_offloaded, placement_note
from .transformers_text import EvalSuiteSpec


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


#: Scoring lives in `.scoring` so both workers cannot drift apart again -- they did,
#: and it made the two sides of a comparison incomparable. See that module.
#: Re-exported under the historical private names for existing callers.
_normalize = normalize
_final_answer = final_answer
_final_number = final_number
_score = score


def _dtype(torch: Any, precision: str):
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


def _device(torch: Any, requested: str) -> str:
    requested = requested.strip().lower()
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"{requested} requested but CUDA is unavailable")
    return requested


def _rows(suite: EvalSuiteSpec) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with Path(suite.dataset).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RuntimeError(f"{suite.dataset}:{line_number} is not a JSON object")
            if suite.prompt_field not in row or suite.expected_field not in row:
                raise RuntimeError(
                    f"{suite.dataset}:{line_number} missing prompt or expected field"
                )
            result.append(row)
    if not result:
        raise RuntimeError(f"evaluation suite {suite.name!r} is empty")
    return result


def evaluate(spec: BaseTextEvalSpec) -> dict[str, Any]:
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            set_seed,
        )
    except ImportError as exc:
        raise RuntimeError("baseline dependencies are missing; install chowder-ai[train]") from exc

    device_name = _device(torch, spec.device)
    if spec.quantization == "4bit" and not device_name.startswith("cuda"):
        raise RuntimeError("4-bit baseline evaluation requires CUDA")
    dtype = _dtype(torch, spec.precision)
    # P6: this arm's own load and generation are timed separately, so the
    # baseline leg of a comparison is a measured cost instead of a blind spot.
    load_timer = PhaseTimer(synchronize=cuda_synchronize(torch))
    load_timer.__enter__()
    set_seed(spec.seed)

    model_cache_status = cache_status(spec.base_model, spec.revision)
    tokenizer = with_hub_retries(
        lambda: AutoTokenizer.from_pretrained(
            spec.base_model,
            revision=spec.revision,
            trust_remote_code=False,
            local_files_only=spec.offline,
        ),
        label=f"tokenizer download for {spec.base_model}",
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("tokenizer has neither pad_token nor eos_token")
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": False,
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
        index = int(device_name.split(":", 1)[1]) if ":" in device_name else 0
        model_kwargs["device_map"] = {"": index}

    model = with_hub_retries(
        lambda: AutoModelForCausalLM.from_pretrained(spec.base_model, **model_kwargs),
        label=f"model download for {spec.base_model}",
    )
    resolved_commit = getattr(model.config, "_commit_hash", None)
    if spec.quantization == "none":
        if spec.placement == "offload":
            if spec.quantization == "4bit":
                raise RuntimeError("offload placement does not combine with 4-bit quantization")
            model = dispatch_offloaded(model, device_name)
        else:
            model = model.to(device_name)
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
    evidence: dict[str, Any] = {}

    generation_timer = PhaseTimer(synchronize=cuda_synchronize(torch))
    memory_sampler = MemorySampler(device_name=sampling_device(torch))
    memory_sampler.start()
    generation_timer.__enter__()
    with torch.inference_mode():
        for suite in spec.suites:
            rows = _rows(suite)
            fingerprint_path = output_dir / f"holdout-fingerprints-{suite.name}.jsonl"
            fingerprint_sha = write_holdout_fingerprint_index(
                (
                    (str(row[suite.prompt_field]), str(row[suite.expected_field]))
                    for row in rows
                ),
                fingerprint_path,
            )
            predictions_path = output_dir / f"predictions-{suite.name}.jsonl"
            correct = 0.0
            with predictions_path.open("w", encoding="utf-8", newline="\n") as output:
                for row in rows:
                    prompt = str(row[suite.prompt_field])
                    expected = str(row[suite.expected_field])
                    # One renderer for both text workers (see
                    # evaluators/rendering.py): what was rendered with is
                    # protocol identity, and two copies drift.
                    rendered, render_evidence = render_prompt(
                        tokenizer=tokenizer,
                        prompt=prompt,
                        suite_name=suite.name,
                        use_chat_template=suite.use_chat_template,
                        canonical_rendering=suite.canonical_rendering,
                    )
                    encoded = tokenizer(rendered, return_tensors="pt")
                    encoded = {key: value.to(device) for key, value in encoded.items()}
                    generated = model.generate(
                        **encoded,
                        max_new_tokens=suite.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=resolved_eos_token_id,
                    )
                    prompt_tokens = encoded["input_ids"].shape[1]
                    prediction = tokenizer.decode(
                        generated[0, prompt_tokens:], skip_special_tokens=True
                    )
                    # The same facts and the same scoring rule the transformers
                    # arm uses: a baseline scored by one worker and a candidate
                    # by the other must leave behind comparable observations, or
                    # the two arms' diagnostics measure different things.
                    observation = observed_generation(
                        generated=generated,
                        prompt_tokens=prompt_tokens,
                        eos_token_id=resolved_eos_token_id,
                        max_new_tokens=suite.max_new_tokens,
                    )
                    observed = observed_score(observation, suite.scoring)
                    row_score = (
                        observed
                        if observed is not None
                        else _score(prediction, expected, suite.scoring)
                    )
                    correct += row_score
                    output.write(
                        json.dumps(
                            {
                                "prompt": prompt,
                                "expected": expected,
                                "prediction": prediction,
                                "score": row_score,
                                **observation,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            metrics[suite.name] = correct / len(rows)
            evidence[suite.name] = {
                "rows": len(rows),
                "scoring": suite.scoring,
                "predictions_file": str(predictions_path),
                "holdout_fingerprints_file": str(fingerprint_path),
                "holdout_fingerprints_sha256": fingerprint_sha,
                "resolved_eos_token_id": resolved_eos_token_id,
                **render_evidence,
            }

    # The baseline arm's own generation, timed and sampled separately from the
    # candidate's -- one arm cannot measure the other, and the ledger says so.
    generation_timer.__exit__()
    memory_sampling = memory_sampler.stop()
    lifecycle_data = evaluation_lifecycle_ledger(
        accelerator_count=1 if device_name.startswith("cuda") else 0,
        arm="baseline",
        generation_seconds=generation_timer.seconds,
        model_load_seconds=load_timer.seconds,
    ).to_dict()

    return {
        "metrics": metrics,
        "suites": evidence,
        "runtime": {
            "device": device_name,
            "gpu_count": 1 if device_name.startswith("cuda") else 0,
            "placement": spec.placement,
            "lifecycle": lifecycle_data,
            "memory_sampling": memory_sampling,
            # See transformers_text_worker: both evaluation arms must report their
            # own footprint, or a baseline-vs-candidate VRAM comparison is not
            # possible from run artifacts.
            **_peak_vram(device_name),
        },
        "model_provenance": {
            "requested_base_model": spec.base_model,
            "requested_revision": spec.revision,
            "model_cache_status": model_cache_status,
            "resolved_model_commit": resolved_commit,
        },
        "versions": {
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
            # Reported even though the baseline never loads an adapter and
            # therefore never imports peft: transformers_text_worker.py's
            # candidate-side protocol always includes it, and the protocol
            # fingerprint must cover the same software-version surface on
            # both sides for a baseline/candidate comparison to mean
            # anything -- an installed-but-unused version is real
            # information, an omitted key is just an asymmetry artifact.
            "peft": _package_version("peft"),
            "bitsandbytes": _package_version("bitsandbytes"),
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
    raw["suites"] = tuple(EvalSuiteSpec(**row) for row in raw["suites"])
    spec = BaseTextEvalSpec(**raw)
    result = evaluate(spec)
    Path(args.result).write_bytes(
        (json.dumps(result, sort_keys=True, indent=2) + "\n").encode("utf-8")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
