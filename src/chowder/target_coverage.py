"""Did the adapter cover the modules that were actually requested?

PEFT matches `target_modules` by suffix and adapts whatever matches. It raises only
when NOTHING matches -- a list where *some* names match nothing trains a smaller
model than asked for, silently. The transformers worker's own comment warns about
this ("guessing wrong here would be a silent partial-coverage bug, not a loud one");
this module is the loud part.

It was not hypothetical either. On the hybrid 9B, an Unsloth run given an explicit
ten-name list adapted **128 modules instead of 200**: all 72 `linear_attn`
(Mamba-style) modules -- `in_proj_qkv`, `in_proj_z`, `out_proj` across 24 layers --
were skipped, because Unsloth converts the list into a regex that did not match
them. Training succeeded, the adapter loaded, the metric moved, and the gate
promoted a candidate that had never touched two thirds of the model's layers.

`adapter_guard` cannot catch this: 128 adapted modules is a live adapter. Liveness
asks "can this change the model"; coverage asks "did it change what you asked for".
They are different questions and both are needed.

The rule has no magic threshold: **a requested name that adapted zero modules is the
defect.** Partial counts within a name are legitimate (a name may appear in only
some layers by design), but a name matching nothing means the request was not
honoured.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

_LORA_MARKER = ".lora_A"


class TargetCoverageError(RuntimeError):
    """A requested target module matched nothing, so training silently shrank."""


def adapted_modules_by_leaf(model: Any) -> dict[str, int]:
    """Count the adapted modules on a PEFT model, keyed by their leaf name.

    Keys off `.lora_A` rather than a specific adapter name, so it works whether the
    adapter is called `default` or something else.
    """
    # Both `...q_proj.lora_A` and `...q_proj.lora_A.default` appear as module
    # names, so counting every match double-counts each adapted module. Collect the
    # distinct TARGET paths first, then count leaves, so the recorded numbers match
    # how many modules were actually adapted.
    targets: set[str] = set()
    for name, _ in model.named_modules():
        index = name.find(_LORA_MARKER)
        if index > 0:
            targets.add(name[:index])
    counts: dict[str, int] = {}
    for target in targets:
        leaf = target.rsplit(".", 1)[-1]
        if leaf:
            counts[leaf] = counts.get(leaf, 0) + 1
    return counts


def coverage_report(
    requested: Iterable[str], adapted_by_leaf: Mapping[str, int]
) -> dict[str, Any]:
    """Compare what was asked for against what was adapted. Pure data."""
    wanted = [str(name) for name in requested]
    matched = {name: int(adapted_by_leaf.get(name, 0)) for name in wanted}
    unmatched = sorted(name for name, count in matched.items() if count == 0)
    adapted_total = sum(int(v) for v in adapted_by_leaf.values())
    extra = sorted(set(adapted_by_leaf) - set(wanted)) if wanted else []
    return {
        "requested": wanted,
        "requested_count": len(wanted),
        "matched_by_name": matched,
        "unmatched": unmatched,
        "adapted_modules_total": adapted_total,
        "adapted_by_leaf": {k: int(v) for k, v in sorted(adapted_by_leaf.items())},
        # Leaves that were adapted without being asked for: not an error (a preset
        # or regex may be broader than the list), but worth seeing.
        "adapted_not_requested": extra,
    }


def assert_targets_covered(
    requested: Iterable[str],
    adapted_by_leaf: Mapping[str, int] | None,
    *,
    allow_unmatched: bool = False,
) -> dict[str, Any]:
    """Raise when an explicitly requested target module adapted nothing.

    Only applies to an explicit list: an empty `requested` means the caller asked
    PEFT or a preset to decide, so there is no stated intent to violate. Returns the
    report either way, so a run records the coverage it achieved rather than only
    that the check passed.

    `adapted_by_leaf=None` means the worker did not report what it adapted --
    coverage is then UNKNOWN, not zero. A measurement gap must never be reported as
    a defect (the same rule `adapter_guard` follows for unreadable weights), so this
    records `status: "not_reported"` and refuses nothing. Both production workers do
    report it; an absent value is visible in evidence and therefore auditable.
    """
    if adapted_by_leaf is None:
        return {
            "status": "not_reported",
            "requested": [str(name) for name in requested],
            "allow_unmatched": bool(allow_unmatched),
            "note": "the worker reported no adapted-module counts, so coverage "
                    "could not be assessed; this is unknown, not zero",
        }
    report = coverage_report(requested, adapted_by_leaf)
    report["status"] = "measured"
    report["allow_unmatched"] = bool(allow_unmatched)
    if not report["requested"] or allow_unmatched or not report["unmatched"]:
        return report

    detail = ", ".join(
        f"{name}={report['matched_by_name'][name]}" for name in report["requested"]
    )
    raise TargetCoverageError(
        "these requested target modules adapted NOTHING: "
        f"{report['unmatched']}. PEFT matches by suffix and only errors when no "
        "name matches at all, so the run would have trained a smaller model than "
        "asked for without saying so.\n"
        f"  per-name adapted counts: {detail}\n"
        f"  total adapted modules:   {report['adapted_modules_total']}\n"
        "Either the names are wrong for this architecture, or the engine rewrote "
        "the list (Unsloth converts it to a regex, which can miss modules a plain "
        "suffix match would find). Fix the list, or set "
        "backend.lora.allow_unmatched_target_modules=true to accept partial "
        "coverage deliberately."
    )
