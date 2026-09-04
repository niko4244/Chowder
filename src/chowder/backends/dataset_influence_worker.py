"""Real, forward-only per-example loss computation for one checkpoint --
the cheap building block `dataset_influence.py`'s approximation is built on.

Deliberately forward-only (no backward, no optimizer step): computing which
training examples plausibly contributed to a regression should cost far
less than the leave-cluster-out re-training this project's own roadmap
research flagged as the GPU-expensive alternative. Reuses the exact same
model+adapter load pattern `evaluators/transformers_text_worker.py` already
proves correct (base model -> optional 4bit quant -> PeftModel.from_pretrained
-> model.eval()), not a new one.

One example per forward pass, not batched: batching same-length-only rows
without padding-aware loss masking would corrupt the per-example average
once real rows of different lengths mix in the same batch. Training sets
in this codebase's own tests/production configs are small (tens of rows),
so the throughput cost of not batching is real but acceptable for this
first slice.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..hf_resilience import cache_status, with_hub_retries


@dataclass(frozen=True)
class PerExampleLossSpec:
    base_model: str
    adapter_dir: str | None
    dataset: str
    text_field: str
    max_length: int
    device: str
    precision: str
    quantization: str
    revision: str | None
    offline: bool


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


def _load_rows(dataset_path: str, text_field: str) -> list[str]:
    texts: list[str] = []
    with Path(dataset_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or text_field not in row:
                raise RuntimeError(f"{dataset_path}:{line_number} missing {text_field!r}")
            texts.append(str(row[text_field]))
    if not texts:
        raise RuntimeError(f"dataset {dataset_path!r} is empty")
    return texts


def compute_per_example_losses(spec: PerExampleLossSpec) -> dict[str, Any]:
    """Real per-example cross-entropy loss for every row in `spec.dataset`,
    under the given checkpoint (base model + optional adapter). Returns
    `{"losses": [float, ...], "model_cache_status": ..., "device": ...}` in
    the same row order the dataset file was read in.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

    device_name = _resolve_device(torch, spec.device)
    if spec.quantization == "4bit" and not device_name.startswith("cuda"):
        raise RuntimeError("4-bit loss computation requires a CUDA device")

    dtype = _resolve_dtype(torch, spec.precision)
    set_seed(0)
    model_cache_status = cache_status(spec.base_model, spec.revision)
    tokenizer = with_hub_retries(
        lambda: AutoTokenizer.from_pretrained(
            spec.base_model, revision=spec.revision, trust_remote_code=False,
            local_files_only=spec.offline,
        ),
        label=f"tokenizer download for {spec.base_model}",
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("tokenizer has neither pad_token nor eos_token")
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": False, "dtype": dtype, "local_files_only": spec.offline,
    }
    if spec.revision is not None:
        model_kwargs["revision"] = spec.revision
    if spec.quantization == "4bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        device_index = int(device_name.split(":", 1)[1]) if ":" in device_name else 0
        model_kwargs["device_map"] = {"": device_index}

    base = with_hub_retries(
        lambda: AutoModelForCausalLM.from_pretrained(spec.base_model, **model_kwargs),
        label=f"model download for {spec.base_model}",
    )
    if spec.quantization == "none":
        base = base.to(device_name)
    model = base if spec.adapter_dir is None else PeftModel.from_pretrained(
        base, spec.adapter_dir, is_trainable=False
    )
    model.eval()
    device = next(model.parameters()).device

    texts = _load_rows(spec.dataset, spec.text_field)
    losses: list[float] = []
    with torch.inference_mode():
        for text in texts:
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=spec.max_length
            ).to(device)
            outputs = model(**inputs, labels=inputs["input_ids"])
            losses.append(float(outputs.loss))

    return {
        "losses": losses,
        "model_cache_status": model_cache_status,
        "device": str(device),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()

    spec_payload = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    spec = PerExampleLossSpec(**spec_payload)
    result = compute_per_example_losses(spec)
    Path(args.result).write_text(json.dumps(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
