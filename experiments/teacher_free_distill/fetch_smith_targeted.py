"""Targeted (census) extraction of SWE-smith rows for chosen instance prefixes.

Why this exists
---------------
``fetch_smith.py`` draws a bounded uniform sample of the trajectory stream.
A uniform sample is the right tool for measuring the corpus, but it is the
wrong tool for growing a *specific* slice: the pilot's 23-row strict corpus
came from ~240 scanned rows and contained exactly one trajectory for each of
gTTS, thefuzz, word_cloud and mypy -- the only repos whose trajectories
survived the four-layer fit/preimage/test-id diagnosis. Growing that slice
needs every trajectory those repos have, not another random draw.

A uniform streaming scan of the whole ``tool`` split (24,100 rows) at the
sampler's observed throughput (~1.2 s/row) would take hours. The hub's
parquet conversion of the same pinned revision can be scanned in seconds per
shard with column projection, so this module uses it and is explicit about
the provenance consequence: the parquet branch (``refs/convert/parquet``)
tracks the dataset's main, so the live dataset sha must still equal the
catalog's pinned revision at extract time, or the run refuses to proceed.

Honesty rules (same as fetch_smith.py)
--------------------------------------
* The catalog source must be approved with a pinned revision and a license.
* ``resolved`` / ``claimed_resolved`` are the publishing run's self-reported
  claim. They are preserved and never treated as evidence; only sandbox
  replay (replay_smith.py) can certify a repair.
* A census is a census: no reservoir, no seed, no ``representative_of_
  full_corpus`` pretense -- every matching row of the split is emitted.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def _fetch_smith():
    """The approved-repair-source gate lives in exactly one place."""
    try:
        import fetch_smith
    except ImportError:  # loaded by path (tests) or from another cwd
        spec = importlib.util.spec_from_file_location(
            "fetch_smith", Path(__file__).with_name("fetch_smith.py"))
        fetch_smith = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fetch_smith)
    return fetch_smith

#: The hub's auto-converted parquet branch. Regenerated from dataset main, so
#: the revision check below is what keeps the extract pinned to the catalog.
PARQUET_REF = "refs/convert/parquet"
TRAJECTORY_COLUMNS = ("instance_id", "traj_id", "resolved", "model", "messages", "patch")
INSTANCE_COLUMNS = ("instance_id", "repo", "FAIL_TO_PASS", "image_name")
#: Instance metadata is the task definition (which tests must flip to green).
#: The dataset's synthesized ``patch`` column is deliberately never extracted:
#: it is ground-truth content nobody trained on, and replay re-derives the
#: repair from the trajectory itself.
VERIFICATION_NOTE = "NONE - raw rows only; sandbox replay required before training use"


def _flush_print(*args) -> None:
    """Progress must survive block-buffered redirects (long shard scans)."""
    print(*args, flush=True)


def live_revision(repo: str) -> str:
    """The dataset sha currently served by the hub (fail-closed anchor)."""
    from huggingface_hub import HfApi

    return str(HfApi().dataset_info(repo).sha)


def shard_paths(repo: str, split: str) -> list[str]:
    """Sorted parquet shard paths of one split on the conversion branch."""
    from huggingface_hub import HfApi

    entries = HfApi().list_repo_tree(repo, repo_type="dataset", revision=PARQUET_REF,
                                     recursive=True)
    paths = [e.path for e in entries
             if e.path.endswith(".parquet") and e.path.split("/")[:2] == ["default", split]]
    return sorted(paths)


def shard_fs_path(repo: str, path: str) -> str:
    """Hub filesystem path of one shard, pinned to the conversion branch."""
    return f"datasets/{repo}@{PARQUET_REF}/{path}"


def open_shard(repo: str, path: str):
    """Seekable parquet reader for one shard (monkeypatched in tests)."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    return pq.ParquetFile(HfFileSystem().open(shard_fs_path(repo, path), "rb"))


def validate_prefixes(prefixes: list[str]) -> list[str]:
    cleaned = [p.strip() for p in prefixes]
    if not cleaned or any(len(p) < 3 or "/" in p or any(c.isspace() for c in p)
                          for p in cleaned):
        raise ValueError("instance prefixes must be non-empty repo-scoped id prefixes")
    return cleaned


def matching_prefix(instance_id: object, prefixes: list[str]) -> str | None:
    text = str(instance_id or "")
    return next((p for p in prefixes if text.startswith(p)), None)


def trajectory_row(raw: dict) -> dict:
    return {
        "traj_id": raw.get("traj_id"),
        "instance_id": raw.get("instance_id"),
        "resolved": raw.get("resolved"),
        "claimed_resolved": str(raw.get("resolved", "")).lower() == "true",
        "model": raw.get("model"),
        "messages": raw.get("messages"),
        "patch": raw.get("patch"),
    }


def instance_row(raw: dict) -> dict:
    f2p = raw.get("FAIL_TO_PASS") or []
    if isinstance(f2p, str):
        f2p = [f2p]
    return {
        "instance_id": raw.get("instance_id"),
        "repo": raw.get("repo"),
        "image_name": raw.get("image_name"),
        "FAIL_TO_PASS": list(f2p),
    }


def census(repo: str, revision: str, split: str, prefixes: list[str],
           columns: tuple[str, ...], mapper, *, limit: int | None = None,
           log=_flush_print) -> tuple[list[dict], dict]:
    """Every row of one pinned split whose instance_id matches a prefix.

    Two passes per shard: a projected ``instance_id`` scan (cheap, decides
    which rows matter) and then a full read of only the row groups that
    contain a match. Nothing about the row set is random, so reruns are
    deterministic and a truncated output is a truncated census, not a bias.
    """
    if live_revision(repo) != revision:
        raise PermissionError(
            "pinned revision moved: the parquet conversion cannot be asserted "
            "to hold the catalog's revision")
    rows: list[dict] = []
    scanned = 0
    shards = shard_paths(repo, split)
    for path in shards:
        reader = open_shard(repo, path)
        ids = reader.read(columns=["instance_id"]).column("instance_id").to_pylist()
        scanned += len(ids)
        wanted = [i for i, iid in enumerate(ids) if matching_prefix(iid, prefixes)]
        if wanted:
            starts, cursor = [], 0
            for group in range(reader.metadata.num_row_groups):
                starts.append(cursor)
                cursor += reader.metadata.row_group(group).num_rows
            by_group: dict[int, list[int]] = {}
            for index in wanted:
                group = max(g for g, start in enumerate(starts) if start <= index)
                by_group.setdefault(group, []).append(index - starts[group])
            for group in sorted(by_group):
                table = reader.read_row_group(group, columns=list(columns))
                for local in by_group[group]:
                    rows.append(mapper(table.slice(local, 1).to_pylist()[0]))
                    if limit is not None and len(rows) >= limit:
                        break
                if limit is not None and len(rows) >= limit:
                    break
        log(f"  {path.rsplit('/', 1)[-1]}: rows={len(ids)} matched={len(wanted)} "
            f"cumulative={len(rows)}")
        if limit is not None and len(rows) >= limit:
            break
    provenance = {"source_repo": repo, "revision": revision, "split": split,
                  "mode": "census", "prefixes": prefixes, "shards": shards,
                  "rows_scanned": scanned, "matched_rows": len(rows),
                  "limit": limit, "verification": VERIFICATION_NOTE,
                  "representative_of_full_corpus": False}
    return rows, provenance


def _sidecar(dest: Path) -> Path:
    return dest.with_suffix(dest.suffix + ".export.json")


def _write(dest: Path, rows: list[dict], provenance: dict) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    _sidecar(dest).write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n",
                              encoding="utf8")
    return provenance


def _resumed(dest: Path, params: dict) -> dict | None:
    sidecar = _sidecar(dest)
    if not sidecar.exists():
        return None
    prior = json.loads(sidecar.read_text(encoding="utf8"))
    if all(prior.get(key) == value for key, value in params.items()):
        return dict(prior, resumed=True)
    return None


def export_targeted(catalog: Path, source_id: str, dest: Path, *,
                    prefixes: list[str], split: str = "tool",
                    limit: int | None = None, log=_flush_print) -> dict:
    """Census of the trajectory split for the given instance prefixes."""
    source = _fetch_smith()._catalog_source(catalog, source_id)
    prefixes = validate_prefixes(prefixes)
    params = {"prefixes": prefixes, "split": split, "limit": limit,
              "revision": source["revision"], "source": source_id, "kind": "trajectories"}
    if (prior := _resumed(dest, params)):
        return prior
    rows, provenance = census(source["repo"], source["revision"], split, prefixes,
                              TRAJECTORY_COLUMNS, trajectory_row, limit=limit, log=log)
    return _write(dest, rows, {**provenance, "source": source_id,
                               "kind": "trajectories"})


def export_instances(catalog: Path, source_id: str, dest: Path, *,
                     prefixes: list[str], split: str = "train",
                     limit: int | None = None, log=_flush_print) -> dict:
    """Census of the instance-metadata split for the same prefixes.

    FAIL_TO_PASS is the task definition replay executes; ``image_name`` is the
    official per-instance image the harness prefers. Neither is evidence that
    a trajectory repaired anything.
    """
    source = _fetch_smith()._catalog_source(catalog, source_id)
    prefixes = validate_prefixes(prefixes)
    params = {"prefixes": prefixes, "split": split, "limit": limit,
              "revision": source["revision"], "source": source_id, "kind": "instances"}
    if (prior := _resumed(dest, params)):
        return prior
    rows, provenance = census(source["repo"], source["revision"], split, prefixes,
                              INSTANCE_COLUMNS, instance_row, limit=limit, log=log)
    return _write(dest, rows, {**provenance, "source": source_id, "kind": "instances",
                               "note": "FAIL_TO_PASS is a task definition, not evidence"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--source", default="swe_smith_trajectories")
    parser.add_argument("--instances-source", default="swe_smith_instances")
    parser.add_argument("--instance-prefix", action="append", required=True,
                        help="instance-id prefix to census (repeatable)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--instances-out", type=Path)
    parser.add_argument("--split", default="tool")
    parser.add_argument("--instances-split", default="train")
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after N matched rows (smoke runs only; default: census)")
    args = parser.parse_args()
    result = export_targeted(args.catalog, args.source, args.out,
                             prefixes=args.instance_prefix, split=args.split,
                             limit=args.limit)
    print(json.dumps(result, indent=2))
    if args.instances_out:
        print(json.dumps(export_instances(args.catalog, args.instances_source,
                                          args.instances_out,
                                          prefixes=args.instance_prefix,
                                          split=args.instances_split,
                                          limit=args.limit), indent=2))


if __name__ == "__main__":
    main()
