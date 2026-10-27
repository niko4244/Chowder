"""Independently join authentic original600 scores to real Windows telemetry.

An aggregate original evaluator output-token/model-time ratio is an observed
research efficiency measurement, NOT llama.cpp decode predicted_per_second
and NOT the preregistered dense-comparator first200 speed gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from chowder_batch import reap_original600_live_independent_audit_20261027 as original
from chowder_batch import reap_original600_live_windows_resources_20261027 as native
from chowder_batch import reap_q4_automationbench_full600_resumable_20261027 as run

SCHEMA = "chowder.reap128v2-original600-native-resource-scored-efficiency.v1"
NATIVE_RESOURCE = Path("F:/reap-v2-q4-guardv3-actual-native-windows-resource-20261027-1356.json")
NATIVE_RESOURCE_SHA = "37bb3be908c3ce5473fd707c90a279033ebf05ea15791eee173751e61d8a238a"
ORIGINAL_GUARD_V3 = Path("F:/reap-v2-q4-original-automationbench600-guardv3-20261027")
OWNER_SHA = "b6aa1f37e81943344217ede4429066c468bce2176a74eb8b15fcc0fe4a4e0529"


def _strict_json(data: bytes) -> dict:
    def unique(pairs: list[tuple[str, object]]) -> dict:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key in original native telemetry")
            result[key] = value
        return result

    result = json.loads(data, object_pairs_hook=unique,
                        parse_constant=lambda value: (_ for _ in ()).throw(
                            ValueError("invalid native telemetry nonfinite " + value)))
    if not isinstance(result, dict):
        raise ValueError("original telemetry root not an object")
    return result


def verify_real_native_resource(
    path: Path = NATIVE_RESOURCE, *, expected_source: Path = ORIGINAL_GUARD_V3
) -> dict:
    if path != NATIVE_RESOURCE or path.is_symlink() or not path.is_file():
        raise ValueError("missing/foreign original native Windows resource snapshot")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != NATIVE_RESOURCE_SHA:
        raise ValueError("original measured Windows resource receipt mutated")
    evidence = _strict_json(raw)
    if (evidence.get("schema") != native.SCHEMA or
            evidence.get("source_root", "").lower() != str(expected_source).lower() or
            evidence.get("source_owner_sha256") != OWNER_SHA or
            evidence.get("original_model_sha256") != run.q4.GGUF_SHA or
            evidence.get("native_original_flags") != run.q4.FLAGS or
            evidence.get("status") !=
                "GENUINE_LIVE_ORIGINAL_Q4_600_WINDOWS_TELEMETRY_NOT_EXCLUSIVE_PEAK" or
            evidence.get("host_gpu_used_is_model_exclusive") is not False or
            evidence.get("frozen_gen0_speed_gate_proven") is not False or
            evidence.get("release_approved") is not False or
            evidence.get("chosen") is not None):
        raise ValueError("original Windows resource owner/scientific claims not authenticated")
    rows = evidence.get("raw_native_windows_observations")
    if not isinstance(rows, list) or not 2 <= len(rows) <= native.MAX_SAMPLES:
        raise ValueError("native original resource samples missing")
    if evidence.get("samples") != len(rows):
        raise ValueError("original resource sample count changed")
    for row in rows:
        if (row.get("source_owner_sha256") != OWNER_SHA or
                row.get("original_native_pid") != evidence.get("native_original_pid") or
                row.get("original_evaluator_pid") != evidence.get("original_evaluator_pid") or
                type(row.get("native_wset_bytes")) is not int or
                type(row.get("native_private_bytes")) is not int or
                row["native_wset_bytes"] < 0 or row["native_private_bytes"] < 0):
            raise ValueError("native Windows resource individual sample source divergence")
    aggregate = {
        "max_native_wset_bytes": max(r["native_wset_bytes"] for r in rows),
        "max_native_private_bytes": max(r["native_private_bytes"] for r in rows),
        "min_physical_host_available_bytes": min(r["physical_available_bytes"] for r in rows),
        "min_native_windows_commit_headroom_bytes":
            min(r["commit_headroom_bytes"] for r in rows),
        "max_host_gpu_used_mib": max(r["host_gpu_used_mib"] for r in rows),
        "min_host_gpu_free_mib": min(r["host_gpu_free_mib"] for r in rows),
    }
    if any(evidence.get(name) != measured for name, measured in aggregate.items()):
        raise ValueError("native resource summary differs from actual repeated Windows samples")
    allocation = evidence.get("source_bound_native_allocator")
    if (not isinstance(allocation, dict) or
            allocation.get("native_cuda_offloaded_layers") != 41 or
            allocation.get("native_cuda_total_layers") != 41 or
            allocation.get("native_cpu_expert_offload") is not False or
            allocation.get("native_context_tokens") != 73728 or
            allocation.get("slots") != 1 or
            allocation.get("active_experts_per_token_per_moe_layer") != 8 or
            allocation.get("stored_experts_per_moe_layer") != 128 or
            allocation.get("original_gguf_sha256") != run.q4.GGUF_SHA):
        raise ValueError("native CUDA allocation/expert census changed")
    return {"source_sha256": NATIVE_RESOURCE_SHA, "telemetry": evidence,
            "derived_sample_extrema": aggregate}


def independent_efficiency(
    root: Path = ORIGINAL_GUARD_V3, resource_path: Path = NATIVE_RESOURCE
) -> dict:
    if root != ORIGINAL_GUARD_V3 or root.is_symlink() or not root.is_dir():
        raise ValueError("not original Q4 experimental guard.v3 research owner")
    raw_owner = (root / "owner.json").read_bytes()
    if hashlib.sha256(raw_owner).hexdigest() != OWNER_SHA:
        raise ValueError("original native experimental owner changed")
    source = original.inspect(root)
    if (source["owner_sha256"] != OWNER_SHA
            or source["model_gguf_sha256"] != run.q4.GGUF_SHA
            or source["noncomparable_to_frozen_four_slot"] is not True):
        raise ValueError("original independently scored tool experiment incompatible")
    resource = verify_real_native_resource(resource_path, expected_source=root)
    totals = source["totals"]
    duration = totals["model_time_s"]
    rate = totals["output_tokens"] / duration if duration > 0 else None
    if rate is not None and (not math.isfinite(rate) or rate < 0):
        raise ValueError("nonfinite genuine native research observed output rate")
    return {
        "schema": SCHEMA, "status": "SOURCE_BOUND_ORIGINAL_PARTIAL_EFFICIENCY_RESEARCH_ONLY",
        "official_source_commit": source["official_source_commit"],
        "original_model_sha256": source["model_gguf_sha256"],
        "original_owner_sha256": OWNER_SHA, "independent_native_receipt_sha256":
            resource["source_sha256"], "original_completed_scored_tasks":
            source["completed_scored_tasks"], "original_registered_tasks": 600,
        "strict_original_worldstate_passes": totals["strict_passes"],
        "actual_original_model_calls": totals["model_calls"],
        "original_input_tokens": totals["input_tokens"],
        "original_output_tokens": totals["output_tokens"],
        "original_sum_model_time_seconds": duration,
        "research_effective_output_tokens_per_model_second": rate,
        "rate_is_not_original_llama_cpp_native_decode_tokens_per_second": True,
        "registered_dense_same_session_first200_ratio": None,
        "qualified_frozen_speed_comparison": False,
        "native_allocation": resource["telemetry"]["source_bound_native_allocator"],
        "native_sample_extrema": resource["derived_sample_extrema"],
        "total_sampled_windows_resource_observations": resource["telemetry"]["samples"],
        "host_gpu_usage_is_not_model_exclusive": True,
        "gen0_status": "BLOCKED", "chosen": None, "release_approved": False,
    }


def main() -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--snapshot", type=Path, help="Write only an exclusive private F: JSON")
    args = cli.parse_args()
    try:
        report = independent_efficiency()
        if args.snapshot is not None:
            if (not args.snapshot.is_absolute() or args.snapshot.parent.resolve()
                    != Path("F:/").resolve() or args.snapshot.exists() or args.snapshot.is_symlink()):
                raise ValueError("native original evidence requires exclusive F: destination")
            with args.snapshot.open("xb") as stream:
                stream.write(run._canonical(report))
        print(json.dumps(report, sort_keys=True, indent=2))
        return 0
    except (ValueError, OSError, TypeError, KeyError, OverflowError) as error:
        print(json.dumps({"status": "REFUSED_SOURCE_BOUND_EFFICIENCY_EVIDENCE",
                          "reason": str(error)[:320], "gen0_status": "BLOCKED"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
