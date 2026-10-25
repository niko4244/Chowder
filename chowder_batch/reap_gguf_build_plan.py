"""Read-only frozen BF16→GGUF/Q8_0/Q4_K_M build admission after full BF16 proof.

The registered Amendment 3 route uses the *same* converter and quantizer
choices as dense/v1 arms, with NO imatrix, NO MTP and Q4 quantized directly
from BF16, not from Q8. This plans but does not execute CPU conversions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

GIB = 1024 ** 3
FROZEN_LLAMA_COMMIT = "7609846557c50f9d984719a9e1e8c5f3d02f807b"
RESERVE = 8 * GIB


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan(*, candidate: Path, verification: Path,
         converter: Path, llama_checkout: Path, quantizer: Path,
         bf16_out: Path, q8_out: Path, q4_out: Path,
         allow_existing: bool = False,
         selected_outputs: frozenset[str] | None = None) -> dict:
    if selected_outputs is None:
        selected_outputs = frozenset(("bf16", "q8_0", "q4_k_m"))
    if not selected_outputs or not selected_outputs.issubset(
            {"bf16", "q8_0", "q4_k_m"}):
        raise ValueError("invalid selected frozen GGUF output set")
    receipt = json.loads(verification.read_text(encoding="utf-8"))
    model = candidate.resolve()
    if (receipt.get("schema") != "chowder.reap128-v2-pruned-bf16-verification.v1"
            or receipt.get("status") != "FULL_SOURCE_TENSOR_EQUIVALENCE_PASS"
            or receipt.get("candidate") != str(model)
            or not isinstance(receipt.get("candidate_shard_sha256"), dict)
            or len(receipt["candidate_shard_sha256"]) != 26):
        raise ValueError("full source-exact 26-shard candidate verification is required")
    original_source = receipt.get("original_source")
    if (not isinstance(original_source, str) or not original_source
            or not Path(original_source).resolve().is_dir()
            or Path(original_source).resolve() == model):
        raise ValueError("registered original BF16 source is missing or overlaps candidate")
    original_source = Path(original_source).resolve()
    if (receipt.get("candidate_index_sha256") != _sha(model / "model.safetensors.index.json")
            or receipt.get("candidate_prune_receipt_sha256") != _sha(model / "reap-prune.json")):
        raise ValueError("actual candidate index/prune provenance changed after verification")
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    tc = config.get("text_config", {})
    if tc.get("num_hidden_layers") != 40 or tc.get("num_experts") != 128:
        raise ValueError("only frozen Qwen REAP128_v2 40x128 can enter GGUF route")
    if (not converter.is_file() or not quantizer.is_file()
            or converter.resolve().parent != llama_checkout.resolve()):
        raise ValueError("registered conversion tools not present at expected checkout")
    revision = subprocess.check_output(
        ["git", "-C", str(llama_checkout), "rev-parse", "HEAD"],
        text=True, timeout=10,
    ).strip()
    if revision != FROZEN_LLAMA_COMMIT:
        raise ValueError("llama.cpp converter checkout differs from preregistered commit")
    dirty = subprocess.check_output(
        ["git", "-C", str(llama_checkout), "status", "--porcelain", "--",
         "convert_hf_to_gguf.py", "gguf-py"],
        text=True, timeout=10,
    ).strip()
    if dirty:
        raise ValueError("frozen converter and gguf-py checkout has uncommitted modifications")
    outputs = tuple(p.resolve() for p in (bf16_out, q8_out, q4_out))
    if (len(set(outputs)) != 3 or any(not path.parent.is_dir() for path in outputs)
            or (not allow_existing and any(path.exists() for path in outputs))):
        raise ValueError("each new GGUF output must have its own existing clean parent")
    if (bf16_out.drive.lower() == candidate.drive.lower()
            or any(path.is_relative_to(root) for path in outputs
                   for root in (model, original_source))):
        raise ValueError("GGUF outputs must be disjoint from original and pruned models")
    bytes_model = receipt.get("pruned_tensor_bytes")
    if type(bytes_model) is not int or bytes_model <= 0:
        raise ValueError("missing verified model tensor byte count")
    # Upper planning allowances, not exact expected GGUF output sizes or gate thresholds.
    estimates = {
        "bf16": int(bytes_model * 1.10) + RESERVE,
        "q8_0": int(bytes_model * 0.625) + RESERVE,
        "q4_k_m": int(bytes_model * 0.44) + RESERVE,
    }
    capacity = {}
    for kind, path in zip(estimates, outputs):
        disk = shutil.disk_usage(path.parent)
        if (kind in selected_outputs and not (allow_existing and path.is_file())
                and disk.free < estimates[kind]):
            raise ValueError(f"{kind} output has inadequate free disk+reserve")
        capacity[kind] = {
            "intended": str(path), "free_bytes": disk.free,
            "estimated_required_bytes_with_reserve": estimates[kind],
        }
    return {
        "schema": "chowder.reap128-v2-gguf-prebuild-v1",
        "status": "PREBUILD_ADMITTED_NOT_CONVERTED_OR_QUALIFIED",
        "candidate_verification_sha256": _sha(verification),
        "candidate": str(model), "original_source": str(original_source),
        "converter_revision": revision,
        "converter": str(converter.resolve()), "quantizer": str(quantizer.resolve()),
        "converter_sha256": _sha(converter),
        "quantizer_binary_sha256": _sha(quantizer),
        "capacity": capacity,
        "conversion_command": [
            sys.executable, "-B", str(converter.resolve()), str(model),
            "--no-mtp", "--outtype", "bf16", "--outfile", str(outputs[0]),
        ],
        "q8_quality_command": [
            str(quantizer.resolve()), str(outputs[0]), str(outputs[1]), "Q8_0", "2",
        ],
        "q4_fit_command": [
            str(quantizer.resolve()), str(outputs[0]), str(outputs[2]), "Q4_K_M", "2",
        ],
        "release_approved": False,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("candidate", "verification", "converter", "llama-checkout",
                "quantizer", "bf16-out", "q8-out", "q4-out"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = plan(
            candidate=args.candidate, verification=args.verification,
            converter=args.converter, llama_checkout=args.llama_checkout,
            quantizer=args.quantizer, bf16_out=args.bf16_out,
            q8_out=args.q8_out, q4_out=args.q4_out,
        )
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        print(f"REAP128_v2 GGUF prebuild refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
