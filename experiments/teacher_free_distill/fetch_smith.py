"""Bounded, resumable fetch of SWE-smith trajectories for local replay.

Downloads a small deterministic sample of raw SWE-agent trajectories. Nothing
downloaded here is treated as verified: the dataset's ``resolved`` flag is a
self-reported claim from the publishing run and is preserved only as
``claimed_resolved``. Independent sandbox replay (replay_smith.py) must
substantiate success before any training row exists.

Fail-closed: the catalog entry must be approved with a pinned revision and a
reviewed license, exactly like stream_hf.py.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

RETRYABLE = (ConnectionError, TimeoutError, OSError)


def _catalog_source(catalog: Path, source_id: str) -> dict:
    source = json.loads(catalog.read_text(encoding="utf8"))["sources"][source_id]
    if (source.get("approved") is not True or not source.get("license")
            or not source.get("review_reference") or not source.get("revision")):
        raise PermissionError("source license, provenance and revision must be approved")
    if source["kind"] != "repair":
        raise ValueError("fetch_smith requires a repair-kind source")
    return source


def export(catalog: Path, source_id: str, dest: Path, *, limit: int = 100,
           scan_limit: int = 2000, seed: int = 2026, resolved_only: bool = True,
           split: str = "tool", max_retries: int = 4) -> dict:
    source = _catalog_source(catalog, source_id)
    if limit <= 0 or scan_limit < limit:
        raise ValueError("requires 0 < limit <= scan_limit")
    params = {"limit": limit, "scan_limit": scan_limit, "seed": seed,
              "resolved_only": resolved_only, "split": split}
    sidecar = dest.with_suffix(dest.suffix + ".export.json")
    if sidecar.exists():
        prior = json.loads(sidecar.read_text(encoding="utf8"))
        if all(prior.get(k) == v for k, v in params.items()):
            return dict(prior, resumed=True)
    from datasets import load_dataset  # optional dependency, like stream_hf

    def open_stream():
        return iter(load_dataset(source["repo"], split=split, streaming=True,
                                 revision=source["revision"]))

    rng = random.Random(seed)
    reservoir: list[dict] = []
    scanned = dropped = 0
    attempt = 0
    while attempt < max_retries:
        attempt += 1
        try:
            stream = open_stream()
            # Each attempt restarts the stream, so the scan budget and the
            # reservoir candidate count restart with it (a persisted counter
            # would make every retry break immediately, silently shrinking
            # the sample).
            scanned = dropped = 0
            for row in stream:
                if scanned >= scan_limit:
                    break
                scanned += 1
                if resolved_only and str(row.get("resolved", "")).lower() != "true":
                    dropped += 1
                    continue
                if len(reservoir) < limit:
                    reservoir.append(row)
                # Vitter R over the candidates seen so far (accepted rows
                # included exactly once): (scanned - dropped) IS that count.
                elif rng.randrange(scanned - dropped) < limit:
                    reservoir[rng.randrange(limit)] = row
            break
        except RETRYABLE:
            if attempt >= max_retries:
                raise
            time.sleep(2 ** attempt)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf8") as handle:
        for row in reservoir:
            handle.write(json.dumps({
                "traj_id": row.get("traj_id"),
                "instance_id": row.get("instance_id"),
                "claimed_resolved": str(row.get("resolved", "")).lower() == "true",
                "model": row.get("model"),
                "messages": row.get("messages"),
                "patch": row.get("patch"),
            }, ensure_ascii=False) + "\n")
    result = {"source": source_id, "revision": source["revision"], "split": split,
              "scanned": scanned, "claimed_resolved_dropped": dropped,
              "sampled": len(reservoir), "seed": seed, **params,
              "verification": "NONE - raw trajectories only; replay required before training use",
              "representative_of_full_corpus": False}
    sidecar.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--source", default="swe_smith_trajectories")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--scan-limit", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--include-unresolved", action="store_true",
                        help="keep self-reported-unresolved rows too (for failure mining)")
    args = parser.parse_args()
    print(json.dumps(export(args.catalog, args.source, args.out, limit=args.limit,
                            scan_limit=args.scan_limit, seed=args.seed,
                            resolved_only=not args.include_unresolved), indent=2))


if __name__ == "__main__":
    main()
