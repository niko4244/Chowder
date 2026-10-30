"""Plan/verify a PRIVATE offline Linux CPython3.13 wheelhouse from official uv.lock.

No GPU, Kaggle, WSL, Ollama, Docker, package installation, model files or
untrusted code execution. By default it only prints a read-only dependency
plan. An explicit --fetch writes wheels to a NEW exclusive private directory,
fetching *only* already SHA256-locked HTTPS files.pythonhosted.org URLs.
--verify works entirely offline and requires every wheel's exact hash/size.

The optional CPU Kaggle bootstrap uses a separate Python3.13 environment and
`pip --no-index --no-deps --require-hashes --target` (never system installs).
This is an experimental prerequisite, NOT approval to launch the GPU kernel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tomllib
from urllib.parse import urlparse
from urllib.request import urlopen

from pip._vendor.packaging.markers import Marker, default_environment
from pip._vendor.packaging.tags import compatible_tags, cpython_tags
from pip._vendor.packaging.utils import parse_wheel_filename


SOURCE = Path("F:/reap-v2-public-automationbench-source-20261026")
PRIVATE_ROOT = Path("F:/reap-v2-q8-automationbench-private-20261026")
SOURCE_COMMIT = "4a8e1061254004d9dac807054eed33fad7d1ff14"
LOCK_SHA256 = "a5137b22bb4080a9a8f6ac63e65572e01b9315184d695e7958f93314ff96788f"
# SHA-256 of canonical JSON of the exact Linux cp313/glibc-2.35 manifest.
EXPECTED_MANIFEST_SHA256_235 = "5dd86a1e445f3ec4103781c542e7eb248f3e2654e218dd684849a0826a4d3d8a"
PYTHON_VERSION = "3.13"
ARCH = "x86_64"
SOURCE_ARCHIVE_SHA256 = "0d9b2b3ea8d54250bf3f14a0ce1d0b7d766d445047cdd730c88575d19b57f278"
MAX_TOTAL_DOWNLOAD = 3 * 1024**3
MAX_SINGLE_WHEEL = 900 * 1024**2
MAX_WHEEL_COUNT = 200
MANIFEST_FILE = "reap-q8-automationbench-offline-wheelhouse-manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def official_lock(source: Path) -> dict:
    if not source.is_dir() or source.is_symlink():
        raise ValueError("official checkout unavailable")
    revision = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"],
                              capture_output=True, check=True, timeout=15, text=True).stdout.strip()
    if revision != SOURCE_COMMIT:
        raise ValueError("source revision not pinned official commit")
    changes = subprocess.run(["git", "-C", str(source), "status", "--porcelain=v1",
                              "--untracked-files=all"], capture_output=True,
                             check=True, timeout=20, text=True).stdout
    if changes.strip():
        raise ValueError("official source changed (tracked or untracked)")
    path = source / "uv.lock"
    if sha256_file(path) != LOCK_SHA256:
        raise ValueError("official uv.lock SHA256 mismatch")
    lock = tomllib.loads(path.read_text(encoding="utf-8"))
    if lock.get("requires-python") != ">=3.13" or lock.get("version") != 1:
        raise ValueError("official lock Python/format contract changed")
    root = [p for p in lock["package"] if p.get("name") == "automation-bench"]
    if len(root) != 1 or root[0]["version"] != "1.0.6" or root[0]["source"] != {"editable": "."}:
        raise ValueError("official editable project/version unavailable")
    return lock


def target_environment() -> dict[str, str]:
    environment = default_environment()
    environment.update({"implementation_name": "cpython", "implementation_version": "3.13.15",
                        "python_version": PYTHON_VERSION, "python_full_version": "3.13.15",
                        "platform_python_implementation": "CPython", "os_name": "posix",
                        "sys_platform": "linux", "platform_system": "Linux",
                        "platform_machine": ARCH, "extra": ""})
    return environment


def runtime_closure(lock: dict) -> list[dict]:
    """Follow normal + activated extras edges only; exclude dev packages."""
    packages = {p["name"]: p for p in lock["package"]}
    if len(packages) != len(lock["package"]):
        raise ValueError("ambiguous duplicate package resolution")
    if "automation-bench" not in packages:
        raise ValueError("root package missing")
    env = target_environment()
    todo = [("automation-bench", "")]
    visited = set()
    while todo:
        name, extra = todo.pop()
        if (name, extra) in visited:
            continue
        if name not in packages:
            raise ValueError("locked transitive dependency missing: " + name)
        visited.add((name, extra))
        pkg = packages[name]
        edges = list(pkg.get("dependencies", []))
        edges.extend(pkg.get("optional-dependencies", {}).get(extra, []))
        for edge in edges:
            marker = edge.get("marker")
            if marker is not None and not Marker(marker).evaluate({**env, "extra": extra}):
                continue
            requested = edge.get("extra", [])
            extras = edge.get("extras", [])
            if type(requested) is str:
                requested = [requested]
            if (not isinstance(requested, list) or any(type(e) is not str for e in requested)
                    or not isinstance(extras, list) or any(type(e) is not str for e in extras)):
                raise ValueError("dependency extras invalid")
            todo.append((edge["name"], ""))
            todo.extend((edge["name"], e) for e in [*requested, *extras])
    return [packages[name] for name in sorted({name for name, _ in visited} - {"automation-bench"})]


def target_tags(glibc_minor: int) -> list:
    if type(glibc_minor) is not int or not 17 <= glibc_minor <= 36:
        raise ValueError("glibc floor must be a verified version 2.17–2.36")
    linux = [f"manylinux_2_{minor}_{ARCH}" for minor in range(glibc_minor, 16, -1)]
    linux.extend(["manylinux2014_x86_64", "manylinux2010_x86_64", "manylinux1_x86_64"])
    return list(dict.fromkeys([*cpython_tags((3, 13), platforms=linux),
                               *compatible_tags((3, 13), interpreter="cp313", platforms=linux)]))


def select_wheel(package: dict, ranks: dict) -> dict:
    options = []
    for wheel in package.get("wheels", []):
        url = urlparse(wheel.get("url", ""))
        filename = url.path.rsplit("/", 1)[-1]
        if (url.scheme != "https" or url.netloc != "files.pythonhosted.org"
                or url.username or url.password or url.fragment or url.query
                or not filename.endswith(".whl")):
            raise ValueError("locked wheel must have exact PyPI HTTPS artifact URL")
        name, version, _build, wheel_tags = parse_wheel_filename(filename)
        if str(name).replace("-", "_") != package["name"].replace("-", "_") or str(version) != package["version"]:
            raise ValueError("locked wheel metadata name/version mismatch")
        sha = wheel.get("hash", "")
        size = wheel.get("size", -1)
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", sha) or type(size) is not int or not 0 < size <= MAX_SINGLE_WHEEL:
            raise ValueError("locked wheel hash or bounded size invalid")
        matches = [ranks[tag] for tag in wheel_tags if tag in ranks]
        if matches:
            options.append((min(matches), filename, wheel))
    if not options:
        raise ValueError("NO_OFFLINE_LINUX_CP313_WHEEL: " + package["name"] + "==" + package["version"])
    _rank, filename, selected = min(options, key=lambda x: (x[0], x[1]))
    return {"name": package["name"], "version": package["version"],
            "filename": filename, "url": selected["url"], "size": selected["size"],
            "sha256": selected["hash"].split(":", 1)[1]}


def build_manifest(lock: dict, *, glibc_minor: int = 17) -> dict:
    tags = target_tags(glibc_minor)
    ranks = {tag: i for i, tag in enumerate(tags)}
    selected = [select_wheel(package, ranks) for package in runtime_closure(lock)]
    if not selected or len(selected) > MAX_WHEEL_COUNT:
        raise ValueError("unexpected locked runtime wheel population")
    if len({w["filename"] for w in selected}) != len(selected):
        raise ValueError("duplicate wheel filenames")
    total_bytes = sum(x["size"] for x in selected)
    if total_bytes > MAX_TOTAL_DOWNLOAD:
        raise ValueError("selected locked runtime wheels exceed 3GiB private download cap")
    return {"schema": "reap-q8-automationbench-offline-cp313-wheelhouse.v1",
            "official_source_commit": SOURCE_COMMIT, "uv_lock_sha256": LOCK_SHA256,
            "official_source_tar_sha256": SOURCE_ARCHIVE_SHA256,
            "python": "CPython3.13", "platform": "linux_x86_64",
            "minimum_glibc": f"2.{glibc_minor}", "package_count": len(selected),
            "total_locked_wheel_bytes": total_bytes, "wheels": selected,
            "gpu_or_kaggle_launched": False, "original_model_or_scorer_modified": False,
            "gen0_status": "BLOCKED"}


def canonical_manifest_sha(manifest: dict) -> str:
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def private_directory(out: Path, private_root: Path = PRIVATE_ROOT) -> Path:
    if not out.is_absolute() or out.exists() or out.is_symlink():
        raise ValueError("exclusive new absolute private directory required")
    base = private_root.resolve(strict=True)
    resolved = out.resolve(strict=False)
    if base not in resolved.parents or not out.parent.is_dir():
        raise ValueError("private output must be new child of existing owned private root")
    for ancestor in out.parents:
        if ancestor.is_symlink() or (getattr(ancestor.stat(), "st_file_attributes", 0) & 0x400):
            raise ValueError("private directory ancestor reparse/symlink forbidden")
    return resolved


def prepare(out: Path, manifest: dict, *, download: bool = False, private_root: Path = PRIVATE_ROOT) -> dict:
    resolved = private_directory(out, private_root)
    resolved.mkdir(mode=0o700, exist_ok=False)
    result = resolved / MANIFEST_FILE
    with result.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    if download:
        for entry in manifest["wheels"]:
            path = resolved / entry["filename"]
            if path.exists():
                raise ValueError("wheel output collision")
            digest = hashlib.sha256()
            transferred = 0
            with urlopen(entry["url"], timeout=30) as response, path.open("xb") as dest:
                while block := response.read(1 * 1024 * 1024):
                    transferred += len(block)
                    if transferred > entry["size"] or transferred > MAX_SINGLE_WHEEL:
                        raise ValueError("download exceeded exact locked artifact size")
                    digest.update(block)
                    dest.write(block)
            if transferred != entry["size"] or digest.hexdigest() != entry["sha256"]:
                raise ValueError("downloaded wheel differs from original uv.lock SHA/size")
    return {"status": "PRIVATE_LOCKED_WHEELHOUSE_COMPLETE" if download else "PRIVATE_WHEELHOUSE_MANIFEST_PREPARED_NO_DOWNLOAD",
            "directory": str(resolved), "wheel_count": manifest["package_count"],
            "expected_download_bytes": manifest["total_locked_wheel_bytes"],
            "offline_complete": download, "gpu_or_kaggle_launched": False}


def verify(out: Path, expected: dict) -> dict:
    path = out / MANIFEST_FILE
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("private wheelhouse manifest missing/unbounded")
    stored = json.loads(path.read_text(encoding="utf-8"))
    if stored != expected:
        raise ValueError("offline manifest does not match source-pinned original lock")
    allowed = {MANIFEST_FILE} | {w["filename"] for w in expected["wheels"]}
    observed = {x.name for x in out.iterdir()}
    if allowed != observed:
        raise ValueError("wheelhouse incomplete or has unexpected files")
    for item in expected["wheels"]:
        wheel = out / item["filename"]
        if wheel.is_symlink() or not wheel.is_file() or wheel.stat().st_size != item["size"]:
            raise ValueError("locked wheel absent/size mismatch")
        if sha256_file(wheel) != item["sha256"]:
            raise ValueError("locked wheel SHA256 mismatch")
    return {"status": "OFFLINE_WHEELHOUSE_BYTES_VERIFIED", "package_count": len(expected["wheels"]),
            "total_wheel_bytes": expected["total_locked_wheel_bytes"], "gpu_or_kaggle_launched": False}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--official-source", type=Path, default=SOURCE)
    # Q8 CPU preflight MUST independently verify host libc>=2.35; only the
    # corresponding manifest is pinned and recognized by the offline probe.
    p.add_argument("--glibc-minor", type=int, default=35)
    p.add_argument("--prepare", type=Path, help="Create a new PRIVATE manifest-only folder; never download")
    p.add_argument("--fetch", type=Path, help="Explicitly fetch exact source-locked wheels to NEW private folder")
    p.add_argument("--verify", type=Path, help="Verify an existing wheelhouse entirely offline")
    args = p.parse_args()
    if sum(x is not None for x in (args.prepare, args.fetch, args.verify)) > 1:
        p.error("choose at most one action")
    manifest = build_manifest(official_lock(args.official_source), glibc_minor=args.glibc_minor)
    if args.glibc_minor != 35 or canonical_manifest_sha(manifest) != EXPECTED_MANIFEST_SHA256_235:
        raise ValueError("only the source-attested Linux cp313/glibc-2.35 manifest is qualified")
    if args.prepare is not None:
        summary = prepare(args.prepare, manifest)
    elif args.fetch is not None:
        summary = prepare(args.fetch, manifest, download=True)
    elif args.verify is not None:
        summary = verify(args.verify, manifest)
    else:
        summary = {k: value for k, value in manifest.items() if k != "wheels"}
        summary["status"] = "SOURCE_LOCKED_CPU_PLAN_NO_WRITES"
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
