"""Passive Windows WDDM per-PID dedicated/shared-memory witness.

This does not load a model, query CUDA, or certify a training placement gate.
Only a complete, task- and identity-bound sampling window plus independent
operator review could later inform the frozen WDDM spill requirement.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

_COUNTER = re.compile(
    r"gpu process memory\(pid_(?P<pid>\d+)_luid_(?P<luid>[^)]+)\)"
    r"\\(?P<kind>shared usage|dedicated usage)$", re.IGNORECASE,
)

# Environment-bound numeric arguments avoid interpolating user text into the
# PowerShell program. Only the built-in Windows performance-counter provider is
# accessed, never a model process or its command line.
_POWERSHELL = r"""
$ErrorActionPreference = 'Stop'
$target = [int]$env:CHOWDER_WDDM_TARGET_PID
$requested = [int]$env:CHOWDER_WDDM_SAMPLE_COUNT
$interval = [int]$env:CHOWDER_WDDM_SAMPLE_INTERVAL
$processBefore = Get-Process -Id $target -ErrorAction Stop
$processStart = $processBefore.StartTime.ToUniversalTime().ToString('o')
$processName = $processBefore.ProcessName
$source = Get-Counter -ListSet 'GPU Process Memory'
$paths = @($source.PathsWithInstances | Where-Object {
    $_ -match "\(pid_${target}_luid_" -and
    ($_ -like '*\Shared Usage' -or $_ -like '*\Dedicated Usage')
})
if ($paths.Count -eq 0) { throw "No WDDM GPU Process Memory counters for PID $target" }
$measurements = @(Get-Counter -Counter $paths -SampleInterval $interval -MaxSamples $requested)
$rows = @(foreach ($measurement in $measurements) {
    [pscustomobject]@{
        timestamp = $measurement.Timestamp.ToString('o')
        counters = @($measurement.CounterSamples | ForEach-Object {
            [pscustomobject]@{path = $_.Path; bytes = $_.CookedValue}
        })
    }
})
$processAfter = Get-Process -Id $target -ErrorAction Stop
if ($processAfter.StartTime.ToUniversalTime().ToString('o') -ne $processStart) {
    throw "Target PID changed identity during sampling"
}
$document = [pscustomobject]@{
    pid = $target
    process_name = $processName
    process_start_utc = $processStart
    samples = $rows
}
ConvertTo-Json -InputObject $document -Depth 8 -Compress
"""


class WddmWitnessError(ValueError):
    """No valid memory measurement witness can be constructed."""


def _integer_bytes(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WddmWitnessError("WDDM counter value must be numeric bytes")
    if not math.isfinite(value) or value < 0 or int(value) != value:
        raise WddmWitnessError("WDDM counter has invalid or nonintegral byte count")
    return int(value)


def summarize_samples(raw: Any, *, pid: int, requested_samples: int) -> dict[str, Any]:
    """Validate complete PID-specific samples without inferring a no-spill gate."""
    if not isinstance(raw, list) or len(raw) != requested_samples:
        raise WddmWitnessError("WDDM sample census is incomplete")
    samples: list[dict[str, Any]] = []
    for sample in raw:
        if not isinstance(sample, dict) or not isinstance(sample.get("timestamp"), str):
            raise WddmWitnessError("WDDM sample lacks a timestamp")
        rows = sample.get("counters")
        if not isinstance(rows, list) or not rows:
            raise WddmWitnessError("WDDM sample lacks memory counters")
        by_adapter: dict[str, dict[str, int]] = {}
        for entry in rows:
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                raise WddmWitnessError("malformed WDDM counter")
            match = _COUNTER.search(entry["path"])
            if not match or int(match["pid"]) != pid:
                raise WddmWitnessError("counter does not belong to the requested PID")
            adapter = match["luid"].lower()
            kind = match["kind"].lower().replace(" ", "_") + "_bytes"
            counters = by_adapter.setdefault(adapter, {})
            if kind in counters:
                raise WddmWitnessError("duplicate WDDM counter for one adapter")
            counters[kind] = _integer_bytes(entry.get("bytes"))
        if any(set(v) != {"shared_usage_bytes", "dedicated_usage_bytes"}
               for v in by_adapter.values()):
            raise WddmWitnessError("shared/dedicated WDDM counter pair is incomplete")
        samples.append({
            "timestamp": sample["timestamp"],
            "adapters": by_adapter,
            "shared_usage_bytes": sum(v["shared_usage_bytes"] for v in by_adapter.values()),
            "dedicated_usage_bytes": sum(v["dedicated_usage_bytes"] for v in by_adapter.values()),
        })
    return {
        "schema": "chowder.wddm-passive-pid-samples.v1", "pid": pid,
        "requested_samples": requested_samples, "observed_samples": len(samples),
        "maximum_shared_usage_bytes": max(s["shared_usage_bytes"] for s in samples),
        "maximum_dedicated_usage_bytes": max(s["dedicated_usage_bytes"] for s in samples),
        "samples": samples,
        "qualification_status": "OBSERVATION_ONLY_NOT_A_WDDM_TRAINING_GATE",
        "limitations": (
            "Samples cover only the specified PID and interval. A zero shared-memory "
            "reading does not certify no WDDM spill during an unobserved GPU run. "
            "Windows performance counters may underrepresent CUDA allocations; "
            "corroborate with CUDA and driver telemetry over the same interval. "
            "Training placement, CUDA allocator peaks, exact parent, and operator "
            "verdict must be independently recorded."
        ),
    }


def observe(*, pid: int, samples: int, interval: int, output: Path) -> dict[str, Any]:
    if sys.platform != "win32":
        raise WddmWitnessError("WDDM process counters require Windows")
    if not 0 < pid < 2**31 or not 1 <= samples <= 3600 or not 1 <= interval <= 3600:
        raise WddmWitnessError("invalid PID, sample count, or sampling interval")
    if output.exists():
        raise WddmWitnessError("refusing to overwrite an existing WDDM receipt")
    env = {**os.environ, "CHOWDER_WDDM_TARGET_PID": str(pid),
           "CHOWDER_WDDM_SAMPLE_COUNT": str(samples),
           "CHOWDER_WDDM_SAMPLE_INTERVAL": str(interval)}
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", _POWERSHELL],
        check=False, text=True, capture_output=True, env=env,
        timeout=samples * interval + 60,
    )
    if result.returncode != 0:
        raise WddmWitnessError("Windows WDDM counter collection failed: "
                               + result.stderr.strip()[:300])
    try:
        raw = json.loads(result.stdout)
        if not isinstance(raw, dict) or raw.get("pid") != pid or not isinstance(
            raw.get("process_name"), str
        ) or not isinstance(raw.get("process_start_utc"), str):
            raise WddmWitnessError("Windows process identity was not recorded")
        witness = summarize_samples(raw.get("samples"), pid=pid,
                                    requested_samples=samples)
        witness["process_name"] = raw["process_name"]
        witness["process_start_utc"] = raw["process_start_utc"]
    except (ValueError, TypeError) as exc:
        raise WddmWitnessError("invalid or incomplete Windows WDDM counter output") from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(witness, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return witness


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--interval", type=int, default=1, help="Seconds between samples")
    parser.add_argument("--out", type=Path, required=True, help="New JSON witness file")
    args = parser.parse_args(argv)
    try:
        witness = observe(pid=args.pid, samples=args.samples, interval=args.interval,
                          output=args.out)
    except (WddmWitnessError, OSError, subprocess.TimeoutExpired) as exc:
        parser.exit(2, f"WDDM observation refused: {exc}\n")
    print(json.dumps({key: value for key, value in witness.items() if key != "samples"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
