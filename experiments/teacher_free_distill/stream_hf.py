"""Optional bounded Hugging Face dataset export; never loads a teacher model."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def export(catalog: Path, source_id: str, dest: Path, *, sample_size: int = 1000,
           scan_limit: int = 20000, seed: int = 2026) -> dict:
    sources = json.loads(catalog.read_text(encoding="utf8"))["sources"]
    source = sources[source_id]
    if (source.get("approved") is not True or not source.get("license")
            or not source.get("review_reference") or not source.get("revision")):
        raise PermissionError("source license, provenance and revision must be approved")
    if source["kind"] != "chat" or sample_size <= 0 or scan_limit < sample_size:
        raise ValueError("requires a chat source and 0 < sample_size <= scan_limit")
    from datasets import load_dataset  # optional; no import needed for CPU validation

    stream = load_dataset(source["repo"], source.get("config_name"), split="train",
                          revision=source["revision"], streaming=True)
    rng = random.Random(seed)
    reservoir = []
    scanned = 0
    for row in stream:
        if scanned >= scan_limit:
            break
        scanned += 1
        if len(reservoir) < sample_size:
            reservoir.append(row)
        else:
            index = rng.randrange(scanned)
            if index < sample_size:
                reservoir[index] = row
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf8") as handle:
        for row in reservoir:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    result = {"source": source_id, "revision": source["revision"], "scanned": scanned,
              "sampled": len(reservoir), "seed": seed,
              "representative_of_full_corpus": False,
              "note": "Reservoir sampling is uniform only within scanned prefix; review shards and source distribution."}
    dest.with_suffix(dest.suffix + ".export.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--scan-limit", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    print(json.dumps(export(args.catalog, args.source, args.out,
                            sample_size=args.sample_size, scan_limit=args.scan_limit,
                            seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
