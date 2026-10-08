"""Collate isolated Experiment C profiler/routing JSON artifacts.

This offline helper never loads a model, starts a worker/server, inspects a
training registry, or modifies campaign configuration. Supply measured JSON
artifacts explicitly; output files are exclusively created to preserve
previous evidence.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from chowder.conditional_compute import (
    RoutingStats,
    pareto_frontier,
    render_pareto_svg,
    routing_health,
)
from chowder.conditional_profile import write_profile_artifact


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def collate_profiles(paths: list[Path]) -> dict[str, Any]:
    """Summarize input profile artifacts without combining incompatible phases."""
    entries = []
    for path in paths:
        profile = _load_json(path)
        request = profile.get("request", {})
        latency = profile.get("latency", {})
        entries.append(
            {
                "artifact": str(path),
                "model": profile.get("model", {}),
                "phase": request.get("phase"),
                "call_kind": request.get("call_kind"),
                "latency": latency,
                "linear_projection_flops_by_phase": profile.get(
                    "linear_projection_flops_by_phase", {}
                ),
                "linear_projection_flops_by_component_and_phase": profile.get(
                    "linear_projection_flops_by_component_and_phase", {}
                ),
                "component_cpu_wall_ms_by_category": profile.get(
                    "component_cpu_wall_ms_by_category", {}
                ),
                "weight_residency": profile.get("weight_residency", {}),
                "module_inclusive_timings": profile.get(
                    "module_inclusive_timings", {}
                ),
                "memory_bandwidth": profile.get("memory_bandwidth", {}),
            }
        )
    return {
        "schema_version": 1,
        "kind": "experiment_c_profile_collation",
        "profiles": entries,
        "warning": "only compare same model, dtype, input batch/sequence shape, cache length, device, and profiler mode",
    }


def _strict_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _strict_fraction(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def collate_routes(path: Path) -> dict[str, Any]:
    """Aggregate JSON routing-stat rows by named task family/variant."""
    payload = _load_json(path)
    rows = payload.get("routing_rows")
    if not isinstance(rows, list):
        raise ValueError("routing artifact must contain a routing_rows list")
    grouped: dict[tuple[str, str], list[RoutingStats]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"routing_rows[{index}] must be an object")
        family, variant = row.get("task_family"), row.get("variant")
        if not isinstance(family, str) or not family or not isinstance(variant, str) or not variant:
            raise ValueError(f"routing_rows[{index}] requires task_family and variant")
        if "valid_tokens" not in row:
            raise ValueError(f"routing_rows[{index}] requires valid_tokens")
        dense_fallback = row.get("dense_fallback", False)
        if not isinstance(dense_fallback, bool):
            raise ValueError(f"routing_rows[{index}].dense_fallback must be boolean")
        stats = RoutingStats(
            tokens=_strict_int(row.get("tokens"), f"routing_rows[{index}].tokens"),
            selected_tokens=_strict_int(
                row.get("selected_tokens"), f"routing_rows[{index}].selected_tokens"
            ),
            executed_tokens=_strict_int(
                row.get("executed_tokens"), f"routing_rows[{index}].executed_tokens"
            ),
            selected_fraction=_strict_fraction(
                row.get("selected_fraction"), f"routing_rows[{index}].selected_fraction"
            ),
            executed_fraction=_strict_fraction(
                row.get("executed_fraction"), f"routing_rows[{index}].executed_fraction"
            ),
            dense_fallback=dense_fallback,
            fallback_reason=row.get("fallback_reason"),
            valid_tokens=_strict_int(
                row.get("valid_tokens"), f"routing_rows[{index}].valid_tokens"
            ),
        )
        if (
            stats.tokens < 0
            or not 0 <= stats.valid_tokens <= stats.tokens
            or not 0 <= stats.selected_tokens <= stats.valid_tokens
            or not 0 <= stats.executed_tokens <= stats.tokens
            or not math.isfinite(stats.selected_fraction)
            or not math.isfinite(stats.executed_fraction)
            or not 0.0 <= stats.selected_fraction <= 1.0
            or stats.executed_fraction < 0.0
            or not math.isclose(
                stats.selected_fraction,
                stats.selected_tokens / stats.valid_tokens if stats.valid_tokens else 0.0,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            or not math.isclose(
                stats.executed_fraction,
                stats.executed_tokens / stats.valid_tokens if stats.valid_tokens else 0.0,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
        ):
            raise ValueError(f"routing_rows[{index}] has inconsistent token counts or fractions")
        if stats.fallback_reason is not None and not isinstance(stats.fallback_reason, str):
            raise ValueError(f"routing_rows[{index}].fallback_reason must be a string or null")
        grouped.setdefault((family, variant), []).append(stats)
    summaries = [
        {
            "task_family": family,
            "variant": variant,
            **routing_health(stats),
        }
        for (family, variant), stats in sorted(grouped.items())
    ]
    return {
        "schema_version": 1,
        "kind": "experiment_c_routing_collation",
        "source_artifact": str(path),
        "summaries": summaries,
    }


def collate_pareto(path: Path) -> dict[str, Any]:
    """Compute a quality-compute Pareto set from pre-scored measurements."""
    payload = _load_json(path)
    points = payload.get("measurements")
    if not isinstance(points, list) or not points:
        raise ValueError("measurement artifact must contain a nonempty measurements list")
    if not all(isinstance(point, dict) for point in points):
        raise ValueError("each measurement must be a JSON object")
    return {
        "schema_version": 1,
        "kind": "experiment_c_quality_compute_pareto",
        "source_artifact": str(path),
        "measurements": points,
        "frontier": pareto_frontier(points),
        "warning": "quality and compute values are accepted as supplied; this tool does not establish evaluation validity or latency significance",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    profile = sub.add_parser("profiles")
    profile.add_argument("--out", required=True)
    profile.add_argument("artifacts", nargs="+")

    routing = sub.add_parser("routes")
    routing.add_argument("--input", required=True)
    routing.add_argument("--out", required=True)

    pareto = sub.add_parser("pareto")
    pareto.add_argument("--input", required=True)
    pareto.add_argument("--out", required=True)
    pareto.add_argument("--plot", required=True)

    args = parser.parse_args()
    if args.command == "profiles":
        result = collate_profiles([Path(path) for path in args.artifacts])
        write_profile_artifact(result, args.out)
    elif args.command == "routes":
        result = collate_routes(Path(args.input))
        write_profile_artifact(result, args.out)
    else:
        result = collate_pareto(Path(args.input))
        write_profile_artifact(result, args.out)
        render_pareto_svg(result["measurements"], args.plot)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())