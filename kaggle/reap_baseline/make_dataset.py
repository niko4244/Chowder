"""Build the private Kaggle dataset that carries the frozen REAP eval code.

    python kaggle/reap_baseline/make_dataset.py --owner <kaggle-user> --out <dir>

The dataset holds the two entry scripts (``reap_serve.py`` and ``reap_eval.py``,
side by side at the root, because ``reap_serve`` finds ``reap_eval`` next to
itself) and every repo module they import, with the ``chowder`` package layout
preserved so ``from chowder.evaluators.scoring import ...`` resolves when the
dataset directory is on ``PYTHONPATH``. The module set is computed by walking
imports -- including imports inside function bodies, which is how ``reap_eval``
pulls in the scorer -- so nothing has to be kept in a hand-written list.

Neither of the two pre-registered scripts is modified, and neither is the
scorer: the copy is byte-identical to the worktree the dataset was built from,
which the printed sha256 manifest records.

It writes files and prints the file list plus their digests. It never calls the
``kaggle`` CLI: uploading is the orchestrator's step (see README.md).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SRC = REPO_ROOT / "src"
BATCH = REPO_ROOT / "chowder_batch"
ENTRYPOINTS = (BATCH / "reap_serve.py", BATCH / "reap_eval.py")
DATASET_SLUG = "chowder-reap-eval"
MANIFEST_NAME = "manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _imports(path: Path) -> list[tuple[int, str]]:
    """Every ``import`` in the file as ``(level, module)``, nested imports included."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(Path(path).read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.extend((0, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            found.append((node.level or 0, node.module or ""))
    return found


def _find_module(base: Path, parts: list[str]) -> Path | None:
    if not parts:
        candidate = base / "__init__.py"
        return candidate if candidate.is_file() else None
    module = base.joinpath(*parts).with_suffix(".py")
    if module.is_file():
        return module
    package = base.joinpath(*parts) / "__init__.py"
    return package if package.is_file() else None


def _resolve(importer: Path, level: int, module: str) -> Path | None:
    """Map one import to the repo file it names, or None if it is not repo code."""
    if level:
        base = importer.parent
        for _ in range(level - 1):
            base = base.parent
        return _find_module(base, [part for part in module.split(".") if part])
    for base in (SRC, BATCH):
        found = _find_module(base, [part for part in module.split(".") if part])
        if found is not None:
            return found
    return None


def dataset_relative(source: Path) -> PurePosixPath:
    """Where a repo file lands in the dataset: chowder's package layout, batch scripts flat.

    ``src/chowder/x/y.py`` -> ``chowder/x/y.py``; ``chowder_batch/z.py`` -> ``z.py``
    (so ``reap_serve`` and ``reap_eval`` stay importable as top-level modules
    beside each other, as the kernel's ``PYTHONPATH`` expects). POSIX either way:
    the dataset is read on a Linux kernel, and the manifest is what a human
    compares against that side.
    """
    source = Path(source).resolve()
    if source.is_relative_to(SRC):
        return PurePosixPath(*source.relative_to(SRC).parts)
    if source.is_relative_to(BATCH):
        return PurePosixPath(source.name)
    raise ValueError(f"{source} is neither under {SRC} nor {BATCH}; refusing to guess a layout")


def closure(entrypoints: tuple[Path, ...] = ENTRYPOINTS) -> dict[Path, PurePosixPath]:
    """Every repo module reachable from the entrypoints, as {source: dataset-relative path}.

    # ponytail: only modules that are imported -- package ``__init__.py`` files are
    # deliberately not pulled in, because ``chowder/__init__`` imports the engine
    # (and through it torch) while the scoring path needs none of it. The copied
    # ``chowder/`` tree is therefore an implicit namespace package (PEP 420) and
    # imports fine with the dataset directory on sys.path.
    # Ceiling: breaks if a copied module ever depends on a side effect of a package
    # ``__init__``. Upgrade: copy the resolved parent ``__init__.py`` files too and
    # accept the heavy import chain in the kernel.
    """
    found: dict[Path, Path] = {}
    queue = [Path(entry).resolve() for entry in entrypoints]
    while queue:
        source = queue.pop()
        if source in found:
            continue
        found[source] = dataset_relative(source)
        for level, module in _imports(source):
            target = _resolve(source, level, module)
            if target is None:
                head = module.split(".")[0]
                if level:
                    raise ValueError(f"{source}: relative import {'.' * level}{module} resolved nowhere")
                if head == "chowder" or (BATCH / f"{head}.py").is_file():
                    raise ValueError(f"{source}: import {module!r} looks like repo code but resolved nowhere")
                continue  # stdlib or a third-party package the image provides
            queue.append(target)
    return found


def write_dataset(out: Path, owner: str) -> dict[str, str]:
    files = closure()
    written: dict[str, str] = {}
    for source, relative in sorted(files.items(), key=lambda item: str(item[1])):
        destination = out / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)  # verbatim: the frozen scripts are not edited on the way
        written[str(relative)] = sha256(destination)

    metadata = {
        "id": f"{owner}/{DATASET_SLUG}",
        "title": "Chowder REAP eval code",
        "licenses": [{"name": "other"}],
        "isPrivate": True,
    }
    (out / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    written["dataset-metadata.json"] = sha256(out / "dataset-metadata.json")

    manifest = {"id": metadata["id"], "source_repo": str(REPO_ROOT), "files": written}
    (out / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--owner", required=True, help="Kaggle username that will own the dataset")
    parser.add_argument("--out", required=True, help="dataset directory to write (its name should be the slug)")
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    written = write_dataset(out, args.owner)
    print(f"{out} -> {args.owner}/{DATASET_SLUG} (private), {len(written)} files")
    for relative, digest in written.items():
        print(f"{digest}  {relative}")
    print(f"manifest: {out / MANIFEST_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
