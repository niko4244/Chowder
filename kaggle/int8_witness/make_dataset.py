"""Build the private Kaggle dataset that carries the int8 placement's code.

    python kaggle/int8_witness/make_dataset.py --owner <kaggle-user> --out <dir>

The dataset is the *committed* tree, exported with ``git archive HEAD`` rather
than copied from the working directory: the worktree carries other agents'
uncommitted edits, and a kernel must run the commit its evidence names, not
whatever happened to be on disk when the upload was built.

The export lands flat in the dataset root, so the kernel puts the dataset
directory on ``sys.path`` and ``import chowder`` resolves to
``<dataset>/src/chowder`` only if ``src`` is on the path too -- so the kernel
adds ``<dataset>/src`` and ``<dataset>/chowder_batch`` explicitly.

It writes files and prints the commit, the file list and their digests. It never
calls the ``kaggle`` CLI: uploading is the orchestrator's step (see README.md).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DATASET_SLUG = "chowder-moe-int8"
MANIFEST_NAME = "manifest.json"
# Excluded from the dataset: large or irrelevant trees that the kernel never imports.
SKIP_PREFIXES = ("campaign/", "evidence/", "kaggle/", ".git/")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def head_commit(repo: Path = REPO_ROOT) -> str:
    """The full commit sha the export is taken from; a branch name would not identify it."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    commit = result.stdout.strip()
    if len(commit) != 40:
        raise ValueError(
            f"git rev-parse HEAD returned {commit!r}, not a full commit sha"
        )
    return commit


def export_commit(out: Path, commit: str) -> list[str]:
    """Write the commit's tree into ``out``; returns the paths written, sorted."""
    archive = subprocess.run(
        ["git", "archive", "--format=tar", commit],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
    ).stdout
    written: list[str] = []
    with tarfile.open(fileobj=__import__("io").BytesIO(archive)) as tar:
        for member in tar.getmembers():
            if not member.isfile() or member.name.startswith(SKIP_PREFIXES):
                continue
            target = out / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            extracted = tar.extractfile(member)
            if (
                extracted is None
            ):  # pragma: no cover - a regular file always has a reader
                raise RuntimeError(f"{member.name} has no content in the archive")
            target.write_bytes(extracted.read())
            written.append(member.name)
    return sorted(written)


def write_dataset(out: Path, owner: str) -> dict:
    commit = head_commit()
    files = export_commit(out, commit)
    metadata = {
        "id": f"{owner}/{DATASET_SLUG}",
        "title": "Chowder MoE int8 expert placement code",
        "licenses": [{"name": "other"}],
        "isPrivate": True,
    }
    (out / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "id": metadata["id"],
        "commit": commit,
        "source_repo": str(REPO_ROOT),
        "files": {name: sha256(out / name) for name in files},
    }
    (out / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--owner", required=True, help="Kaggle username that will own the dataset"
    )
    parser.add_argument(
        "--out",
        required=True,
        help="dataset directory to write (its name should be the slug)",
    )
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = write_dataset(out, args.owner)
    print(f"{out} -> {manifest['id']} (private) at commit {manifest['commit']}")
    print(f"{len(manifest['files'])} files")
    for name in (
        "src/chowder/backends/moe_int8_experts.py",
        "kaggle/int8_witness/probe.py",
    ):
        if name in manifest["files"]:
            print(f"{manifest['files'][name]}  {name}")
    print(f"manifest: {out / MANIFEST_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
