"""Kaggle notebook script: acquire and verify one pinned Qwen3.8 parent.

Run this INSIDE a **private** Kaggle notebook. GPU is not required for
this script -- acquisition and verification are both disk/CPU-only
(`chowder.qwen38_acquisition.acquire_parent`, imported unchanged; no
acquisition logic lives in this file).

Prerequisites
-------------
1. A private Kaggle notebook with internet access enabled.
2. A Kaggle Secret named ``HF_TOKEN`` attached to the notebook, holding a
   Hugging Face access token. The token is never printed, logged, or
   written to any output file by this script.
3. Chowder installed at the exact pinned commit this qualification run
   is meant to match, e.g.::

       pip install "chowder-ai[train,qlora] @ git+https://github.com/niko4244/Chowder.git@<commit-sha>"

   Prefer this over copying source into notebook cells -- the whole
   point of pinning a commit is that the acquisition/verification code
   is provably the same code, not a hand-transcribed copy that can
   silently drift.

Usage
-----
    python acquire_parent.py --parent A --destination /kaggle/working/parent-a
    python acquire_parent.py --parent C --destination /kaggle/working/parent-c
    python acquire_parent.py --parent D --destination /kaggle/working/parent-d

``--parent`` accepts A/B/C/D (A is needed for the Phase 3 equivalence
qualification re-run; C/D are the Phase 1 targets). Every parent is
acquired at its exact pinned revision from
``chowder.qwen38_acquisition.ALL_PARENT_PINS`` -- never a floating
``main``/``latest`` reference (`ParentPin` itself refuses those at
construction).

Output
------
Writes the real, full-mode manifest next to the destination directory
(the identical `<name>.manifest.json` convention parents A/B already
use), plus a small, deliberately non-protected-content JSON summary
(manifest digest, architecture metadata, parameter/shard counts) --
never raw model weights re-printed, never anything from the protected
evaluation suite (this script never touches it).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from chowder.qwen38_acquisition import (
    ALL_PARENT_PINS,
    AcquisitionError,
    RemoteFileInfo,
    acquire_parent,
)


def _get_hf_token() -> str:
    """Kaggle Secrets only. Never read from argv, never logged, never
    included in any exception message or output file."""
    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError as exc:
        raise RuntimeError(
            "kaggle_secrets is only importable inside a Kaggle notebook kernel; "
            "run this script there, with an HF_TOKEN secret attached to the notebook"
        ) from exc
    token = UserSecretsClient().get_secret("HF_TOKEN")
    if not token:
        raise RuntimeError("Kaggle secret 'HF_TOKEN' is empty or not attached to this notebook")
    return token


def _list_files_fn(token: str):
    def _list(repo: str, revision: str) -> list[RemoteFileInfo]:
        from huggingface_hub import HfApi

        info = HfApi(token=token).model_info(repo, revision=revision, files_metadata=True)
        return [
            RemoteFileInfo(path=sibling.rfilename, size_bytes=sibling.size or 0)
            for sibling in info.siblings
        ]

    return _list


def _snapshot_download_fn(token: str):
    def _download(*, repo_id: str, revision: str, local_dir: str) -> str:
        from huggingface_hub import snapshot_download

        return snapshot_download(repo_id=repo_id, revision=revision, local_dir=local_dir, token=token)

    return _download


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parent", choices=sorted(ALL_PARENT_PINS), required=True)
    parser.add_argument("--destination", required=True, help="Local (Kaggle working-dir) destination directory")
    args = parser.parse_args(argv)

    pin = ALL_PARENT_PINS[args.parent]
    destination = Path(args.destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    token = _get_hf_token()

    candidate_roots = [(destination.parent, shutil.disk_usage(destination.parent).free)]

    try:
        result = acquire_parent(
            pin,
            destination,
            list_files_fn=_list_files_fn(token),
            snapshot_download_fn=_snapshot_download_fn(token),
            candidate_roots=candidate_roots,
        )
    except AcquisitionError as exc:
        print(f"ACQUISITION FAILED for parent {args.parent} ({pin.repo}@{pin.revision[:12]}...): {exc}", file=sys.stderr)
        return 1

    summary_path = destination.parent / f"{destination.name}.acquisition-summary.json"
    summary = {
        "parent": args.parent,
        "repo": pin.repo,
        "revision": pin.revision,
        "manifest_sha256": result.manifest.manifest_sha256,
        "total_weight_gib": round(result.manifest.total_weight_bytes / 2**30, 2),
        "weight_shard_count": len(result.manifest.weight_files),
        "already_present": result.already_present,
        "architecture": result.architecture,
        "parameter_accounting": result.parameter_accounting,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Parent {args.parent} acquisition summary written to {summary_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
