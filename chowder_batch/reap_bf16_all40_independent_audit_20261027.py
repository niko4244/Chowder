"""Independently rehash all original BF16 shards used by 40 genuine layer runs.

This verifies *source provenance, completeness and arithmetic* of actual CPU
vs CUDA BF16 layer receipts without rerunning model inference. It is NOT a
chained 40-layer forward/logit numerical parity certification or a Gen0 gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from chowder_batch import reap_bf16_cuda_all40_research_20261027 as all40
from chowder_batch import reap_bf16_cuda_layer_parity_20261027 as layer

SCHEMA = "chowder.reap128-v2-original-bf16-all40-independent-source-audit.v1"
EXPECTED_COUNT = 40
EXPECTED_ELEMENTS_PER_LAYER = 4096


def assert_exact_receipt_sources(
    *, model: Path, claimed: dict[str, str],
    expected: dict[str, str], required: list[str],
) -> None:
    """Reject a resealed receipt that names foreign or incomplete source shards."""
    if (not isinstance(claimed, dict) or not isinstance(expected, dict)
            or not isinstance(required, list) or not required
            or set(claimed) != set(required) or len(required) != len(set(required))):
        raise ValueError("layer receipt shard list is not its actual indexed layer source")
    for name in required:
        if (name not in expected or type(claimed.get(name)) is not str
                or len(claimed[name]) != 64 or claimed[name] != expected[name]):
            raise ValueError("layer source hash does not match immutable full-source proof")
        candidate = model / name
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError("indexed original shard missing, unsafe or not a regular file")


def audit(model: Path, root: Path) -> dict:
    model = model.resolve(strict=True)
    root = root.resolve(strict=True)
    if model.is_symlink() or root.is_symlink() or not model.is_dir() or not root.is_dir():
        raise ValueError("source checkpoint and private receipt directory must be regular")
    if root.is_relative_to(model) or model.is_relative_to(root):
        raise ValueError("audit output may not be nested within original checkpoint")
    owner_path, summary_path = root / "owner.json", root / "all40-summary.json"
    for p in (owner_path, summary_path):
        if p.is_symlink() or not p.is_file() or not 0 < p.stat().st_size < 256 * 1024:
            raise ValueError("original owned private receipt absent or unsafe")
    original_owner = all40.exact_owner(model)
    if owner_path.read_bytes() != all40._canonical_bytes(original_owner):
        raise ValueError("full-40 original source owner hash changed")
    raw = summary_path.read_bytes()
    summary = all40._load_strict(raw)
    if (summary.get("schema") != all40.SCHEMA
            or summary.get("status") != "ALL40_SOURCE_BOUND_LAYER_NUMERICS_RESEARCH_ONLY"
            or summary.get("model") != str(model)
            or summary.get("owner_sha256") != hashlib.sha256(
                all40._canonical_bytes(original_owner)).hexdigest()
            or summary.get("source_full_model_ancestry_sha256") != layer.FROZEN_PROOF_SHA256
            or summary.get("distinct_original_layer_receipts") != EXPECTED_COUNT
            or summary.get("original_prior_layer_receipts_reused") != [0, 3]
            or summary.get("real_cpu_cuda_bf16_compared_elements")
            != EXPECTED_COUNT * EXPECTED_ELEMENTS_PER_LAYER
            or summary.get("entire_model_forward_or_logit_parity_proven") is not False
            or summary.get("native_gguf_original_four_slot_gate_eligible") is not False
            or summary.get("gen0_status") != "BLOCKED" or summary.get("chosen", 1) is not None
            or summary.get("release_approved") is not False):
        raise ValueError("full-40 result changes original research-only owner contract")
    if (not isinstance(summary.get("per_layer"), list)
            or len(summary["per_layer"]) != 40
            or not isinstance(summary.get("source_bound_private_receipt_sha256"), dict)
            or set(summary["source_bound_private_receipt_sha256"]) != {
                str(i) for i in range(40)}):
        raise ValueError("incomplete per-layer receipts or missing source digests")
    if all40._canonical_bytes(summary) != raw:
        raise ValueError("noncanonical duplicate/resealed original summary")
    proof_path = layer.FROZEN_PROOF_PATH
    if (proof_path.is_symlink() or not proof_path.is_file()
            or layer.digest_file(proof_path) != layer.FROZEN_PROOF_SHA256):
        raise ValueError("frozen original 26-shard source-equivalence proof drift")
    proof = all40._load_strict(proof_path.read_bytes())
    pins = proof.get("candidate_shard_sha256")
    if (proof.get("status") != "FULL_SOURCE_TENSOR_EQUIVALENCE_PASS"
            or proof.get("shards") != 26 or not isinstance(pins, dict) or len(pins) != 26
            or Path(proof.get("candidate", "")).resolve() != model):
        raise ValueError("original 26-shard proof no longer applies to this checkpoint")
    all_rows, used_shards, hashes = [], set(), {}
    expected_local_paths = {"owner.json", "all40-summary.json"}
    for number in range(40):
        cfg, _, inv = layer.verify_model_metadata(model, number)
        if number in all40.ORIGINAL_LAYER_RECEIPTS:
            source_path, fixed_sha = all40.ORIGINAL_LAYER_RECEIPTS[number]
            source_path = Path(source_path)
            if all40.digest_file(source_path) != fixed_sha:
                raise ValueError("previous completed original layer SHA changed")
        else:
            source_path = root / f"layer-{number:02d}.json"
            expected_local_paths.add(source_path.name)
        row, digest = all40._receipt(model, number, source_path)
        full = all40._load_strict(source_path.read_bytes())
        original = full.get("source_verification")
        if (not isinstance(original, dict)
                or full.get("layer_type") != cfg["layer_types"][number]
                or full.get("checkpoint_metadata_sha256") !=
                {key: layer.digest_file(model / key) for key in
                 ("config.json", "model.safetensors.index.json", "reap-prune.json") }):
            raise ValueError("original model metadata, layer type or source hash forged")
        assert_exact_receipt_sources(
            model=model,
            claimed=original.get("verified_live_selected_shard_sha256"),
            expected=pins,
            required=inv["read_shard_names"],
        )
        used_shards.update(inv["read_shard_names"])
        if row != summary["per_layer"][number] or digest != summary[
            "source_bound_private_receipt_sha256"][str(number)]:
            raise ValueError("original layer numeric summary or receipt SHA was altered")
        if not all(math.isfinite(row[key]) for key in
                   ("max_abs_error", "mean_abs_error", "rmse")):
            raise ValueError("invalid original nonfinite numerical error")
        all_rows.append(row)
        hashes[str(number)] = digest
    local_files = {p.name for p in root.iterdir() if p.is_file()}
    if local_files != expected_local_paths or any(p.is_symlink() for p in root.iterdir()):
        raise ValueError("unexpected private 40-layer owner or symlinked evidence")
    # Hash each genuine touched immutable BF16 shard once independently, not 40 times.
    for name in sorted(used_shards):
        observed = layer.digest_file(model / name)
        if observed != pins[name]:
            raise ValueError(f"current original BF16 shard differs: {name}")
    if (sum(x["exact_equal_elements"] for x in all_rows)
            != summary["exact_equal_elements"]
            or max(x["max_abs_error"] for x in all_rows)
            != summary["max_absolute_difference"]
            or min(min(x["host_gpu_free_pre_mib"], x["host_gpu_free_post_mib"])
                   for x in all_rows) != summary["min_physical_gpu_free_mib"]):
        raise ValueError("40-layer arithmetic or measured physical headroom differs")
    return {
        "schema": SCHEMA,
        "status": "SOURCE_BOUND_40_LAYER_RESEARCH_RECEIPTS_INDEPENDENTLY_VERIFIED",
        "original_40_layers": 40, "genuine_new_layer_receipts": 38,
        "reused_immutable_prior_layers": [0, 3],
        "registered_total_bf16_shards": 26,
        "unique_source_shards_rehashed": len(used_shards),
        "original_model": str(model),
        "full_source_proof_sha256": layer.FROZEN_PROOF_SHA256,
        "per_layer_receipt_sha256": hashes,
        "all40_summary_sha256": hashlib.sha256(raw).hexdigest(),
        "input_bf16_elements_compared": 163840,
        "exact_equal_elements": summary["exact_equal_elements"],
        "exact_equal_fraction": summary["exact_equal_elements"] / 163840,
        "max_absolute_difference": summary["max_absolute_difference"],
        "min_sampled_physical_gpu_free_mib": summary["min_physical_gpu_free_mib"],
        "entire_model_logit_parity_proven": False,
        "native_gguf_release_gate_eligible": False,
        "independent_generated_code_sandbox_certified": False,
        "frozen_numerical_gate": "UNKNOWN", "gen0_status": "BLOCKED",
        "chosen": None, "release_approved": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        if (args.out.exists() or args.out.is_symlink() or not args.out.parent.is_dir()
                or args.out.suffix != ".json"):
            raise ValueError("fresh private JSON-only evidence path required")
        result = audit(args.model, args.root)
        with args.out.open("x", encoding="utf8", newline="\n") as out:
            json.dump(result, out, sort_keys=True, indent=2, allow_nan=False)
            out.write("\n")
        print(json.dumps({k: result[k] for k in (
            "status", "original_40_layers", "unique_source_shards_rehashed",
            "exact_equal_fraction", "max_absolute_difference", "gen0_status")}))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"status": "BLOCKED_INDEPENDENT_ALL40_EVIDENCE_DRIFT",
                          "reason": type(error).__name__,
                          "gen0_status": "BLOCKED", "release_approved": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
