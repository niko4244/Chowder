"""Standalone Kaggle CPU-only original AutomationBench source/wheel bootstrap.

Run with Kaggle GPU disabled. Default --inspect verifies all pinned wheel bytes,
Python3.13 and glibc WITHOUT writing or importing third-party code. An explicit
--install installs wheel packages into ONE NEW private /kaggle/working target,
never a system location, via pip --no-index --no-deps --require-hashes.

It deliberately does NOT start a llama-server, evaluator inference, kernel,
network client, Q8 weight reader, WSL, Podman, Ollama or source mutation.
Do NOT use its results as model scores or Gen-0 admission evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tarfile


SCHEMA = "reap-q8-automationbench-offline-cp313-wheelhouse.v1"
OFFICIAL_TAR_SHA = "0d9b2b3ea8d54250bf3f14a0ce1d0b7d766d445047cdd730c88575d19b57f278"
OFFICIAL_LOCK_SHA = "a5137b22bb4080a9a8f6ac63e65572e01b9315184d695e7958f93314ff96788f"
LOCKED_WHEEL_MANIFEST_SHA = "5dd86a1e445f3ec4103781c542e7eb248f3e2654e218dd684849a0826a4d3d8a"
OFFICIAL_PROMPT_SHA = "1c8feeec1f680b05b392a5d00fc634edb143c3e7bdd152c476af50cebcec57b5"
OFFICIAL_COMMIT = "4a8e1061254004d9dac807054eed33fad7d1ff14"
MAX_ARTIFACTS = 200
MAX_TOTAL_BYTES = 3 * 1024**3
MAX_TAR_BYTES = 64 * 1024**2


def file_sha(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def ensure_private_target(target: Path, *, home: Path = Path("/kaggle/working")) -> Path:
    if target.exists() or target.is_symlink() or not target.is_absolute():
        raise ValueError("exclusive absolute CPU-only scratch target required")
    boundary = home.resolve(strict=True)
    resolved = target.resolve(strict=False)
    if boundary not in resolved.parents or not target.parent.is_dir():
        raise ValueError("isolated CPU site must remain under /kaggle/working")
    for parent in target.parents:
        if parent.is_symlink():
            raise ValueError("symlinked CPU scratch ancestor")
    return resolved


def glibc_minor() -> int:
    implementation, version = platform.libc_ver()
    match = re.fullmatch(r"2\.(\d+)", version)
    if implementation != "glibc" or match is None:
        raise ValueError("glibc platform/version is unavailable; refuse target wheels")
    return int(match.group(1))


def inspect(wheels: Path, manifest_file: Path, source_tar: Path, *,
            require_native_python: bool = True) -> dict:
    if require_native_python and sys.version_info[:2] != (3, 13):
        raise ValueError("this CPU preflight requires REAL Kaggle CPython3.13")
    if not wheels.is_dir() or wheels.is_symlink() or not manifest_file.is_file():
        raise ValueError("PRIVATE mounted wheelhouse/manifest missing")
    if manifest_file.is_symlink() or manifest_file.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("unbounded/linked source manifest")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest_digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                                                ensure_ascii=False).encode("utf-8")).hexdigest()
    if manifest_digest != LOCKED_WHEEL_MANIFEST_SHA:
        raise ValueError("PRIVATE manifest does not equal complete source-locked 108-wheel contract")
    if (manifest.get("schema") != SCHEMA or manifest.get("official_source_commit") != OFFICIAL_COMMIT
            or manifest.get("uv_lock_sha256") != OFFICIAL_LOCK_SHA
            or manifest.get("official_source_tar_sha256") != OFFICIAL_TAR_SHA
            or manifest.get("python") != "CPython3.13" or manifest.get("platform") != "linux_x86_64"):
        raise ValueError("original uv.lock/official evaluator identity mismatch")
    if platform.machine().lower() not in ("x86_64", "amd64"):
        raise ValueError("mounted platform is not Kaggle Linux x86_64")
    expected_libc = manifest.get("minimum_glibc")
    if not isinstance(expected_libc, str) or not re.fullmatch(r"2\.\d+", expected_libc):
        raise ValueError("missing wheelhouse glibc floor")
    if glibc_minor() < int(expected_libc.split(".")[1]):
        raise ValueError("Kaggle glibc older than pinned wheelhouse compatibility")
    entries = manifest.get("wheels")
    if not isinstance(entries, list) or not 0 < len(entries) <= MAX_ARTIFACTS:
        raise ValueError("locked dependency list missing/unbounded")
    if (manifest.get("package_count") != len(entries)
            or manifest.get("total_locked_wheel_bytes") != sum(entry.get("size", 0) for entry in entries)):
        raise ValueError("dependency population/total checksum invalid")
    if manifest["total_locked_wheel_bytes"] > MAX_TOTAL_BYTES:
        raise ValueError("dependency wheelhouse exceeds size limit")
    filenames = set()
    names = set()
    for entry in entries:
        file = entry.get("filename")
        version = entry.get("version")
        name = entry.get("name")
        digest = entry.get("sha256")
        size = entry.get("size")
        if (not isinstance(file, str) or not re.fullmatch(r"[A-Za-z0-9_.+-]+\.whl", file)
                or type(size) is not int or size <= 0
                or not isinstance(name, str) or not re.fullmatch(r"[a-z0-9-]+", name)
                or not isinstance(version, str) or not re.fullmatch(r"[A-Za-z0-9.+!-]+", version)
                or not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest)):
            raise ValueError("unsafe or malformed wheel metadata")
        if file in filenames or name in names:
            raise ValueError("duplicate locked wheel/distribution")
        filenames.add(file)
        names.add(name)
        path = wheels / file
        if path.is_symlink() or not path.is_file() or path.stat().st_size != size:
            raise ValueError("offline wheel missing, linked or wrong byte length: " + name)
        if file_sha(path) != digest:
            raise ValueError("offline wheel SHA256 mismatch: " + name)
    if "verifiers" not in names or "datasets" not in names or "pydantic" not in names:
        raise ValueError("required official evaluator/runtime packages missing")
    if (not source_tar.is_file() or source_tar.is_symlink()
            or source_tar.stat().st_size > MAX_TAR_BYTES or file_sha(source_tar) != OFFICIAL_TAR_SHA):
        raise ValueError("mounted original official .tar.bin source not byte-identical")
    return {"status": "CPU_OFFLINE_SOURCES_AND_WHEELS_VERIFIED_NOT_INSTALLED",
            "platform": "linux_x86_64", "python": sys.version.split()[0],
            "glibc": platform.libc_ver()[1], "official_commit": OFFICIAL_COMMIT,
            "package_count": len(entries), "locked_download_bytes": manifest["total_locked_wheel_bytes"],
            "weights_loaded": False, "gpu_requested": False, "gen0_status": "BLOCKED"}


def unpack_original_source(archive_path: Path, target: Path) -> Path:
    """Extract exact trusted tar bytes with no symlink/hardlink/path escape."""
    if file_sha(archive_path) != OFFICIAL_TAR_SHA:
        raise ValueError("upstream source tar changed")
    target.mkdir(mode=0o700, exist_ok=False)
    with tarfile.open(archive_path, "r:") as archive:
        for member in archive:
            name = member.name
            if (name.startswith("/") or "\\" in name or ".." in Path(name).parts
                    or member.issym() or member.islnk() or not (member.isfile() or member.isdir())):
                raise ValueError("upstream tar contains unsafe path/link")
            child = target / name
            if not child.resolve(strict=False).is_relative_to(target.resolve(strict=True)):
                raise ValueError("upstream tar path escapes private source root")
            if member.isdir():
                child.mkdir(parents=True, exist_ok=True)
            else:
                child.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as src, child.open("xb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
    if file_sha(target / "uv.lock") != OFFICIAL_LOCK_SHA:
        raise ValueError("extracted source lock mismatch")
    return target


def make_requirements(manifest: dict, path: Path) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for item in manifest["wheels"]:
            stream.write(f"{item['name']}=={item['version']} --hash=sha256:{item['sha256']}\n")


def install_and_smoke(wheel_dir: Path, manifest_file: Path, source_tar: Path, target: Path) -> dict:
    proof = inspect(wheel_dir, manifest_file, source_tar)
    ensure_private_target(target)
    probe = subprocess.run([sys.executable, "-m", "pip", "--version"],
                           capture_output=True, text=True, timeout=15)
    if probe.returncode:
        # A python3.13 with no pip cannot perform a safe wheel-only install.
        raise ValueError("offline python3.13 pip module unavailable; install refused")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    target.mkdir(mode=0o700, exist_ok=False)
    packages = target / "packages"
    packages.mkdir(mode=0o700, exist_ok=False)
    source = unpack_original_source(source_tar, target / "official-source")
    req = target / "requirements-wheel-lock.txt"
    make_requirements(manifest, req)
    cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-index",
           "--no-deps", "--only-binary", ":all:", "--require-hashes", "--no-compile",
           "--target", str(packages), "--find-links", str(wheel_dir), "-r", str(req)]
    env = {key: os.environ[key] for key in ("PATH", "HOME", "TMPDIR", "LANG") if key in os.environ}
    env.update({"PIP_NO_INDEX": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                "PIP_NO_INPUT": "1", "PIP_CONFIG_FILE": os.devnull,
                "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
                "HF_DATASETS_OFFLINE": "1", "HF_HUB_OFFLINE": "1"})
    outcome = subprocess.run(cmd, env=env, cwd=target, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, timeout=1800)
    if outcome.returncode:
        raise ValueError("offline locked wheel installation failed; NO model/GPU used")
    installed = {str(d.metadata.get("Name", "")).lower().replace("_", "-"): d.version
                 for d in importlib.metadata.distributions(path=[str(packages)])}
    expected = {item["name"]: item["version"] for item in manifest["wheels"]}
    if any(installed.get(name) != version for name, version in expected.items()):
        raise ValueError("installed CPU-only package versions do not match official uv.lock")
    code = (
        'from automationbench.domains import DEFAULT_DOMAINS,get_combined_dataset;'
        'import hashlib,json;d=get_combined_dataset(DEFAULT_DOMAINS);'
        'rows=[(str(x["example_id"]),str(x["prompt"])) for x in d];'
        'digest=hashlib.sha256(json.dumps(rows,ensure_ascii=False,separators=(",", ":")).encode()).hexdigest();'
        'print(json.dumps({"task_count":len(d),"source_prompt_sha256":digest,"domains":DEFAULT_DOMAINS}))'
    )
    env["PYTHONPATH"] = os.pathsep.join([str(packages), str(source)])
    smoke = subprocess.run([sys.executable, "-s", "-c", code], cwd=source, env=env,
                           capture_output=True, text=True, timeout=900)
    if smoke.returncode:
        raise ValueError("official CPU-only 600 source loader/import failed; NO scored tasks")
    result = json.loads(smoke.stdout)
    if (result.get("task_count") != 600
            or result.get("source_prompt_sha256") != OFFICIAL_PROMPT_SHA
            or result.get("domains") != ["sales", "marketing", "operations", "support", "finance", "hr"]):
        raise ValueError("source-loaded 600 task SHA/population changed")
    return {**proof, "status": "CPU_OFFLINE_600_SOURCE_LOADER_PASS_NOT_A_MODEL_SCORE",
            "site_packages": str(packages), "official_source": str(source),
            "scorer_source_mutated": False, "model_inference_started": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheels", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--official-tar-bin", type=Path, required=True)
    parser.add_argument("--install", action="store_true", help="CPU-only private target; no network or model")
    parser.add_argument("--scratch", type=Path, help="new /kaggle/working private target; required with --install")
    args = parser.parse_args()
    if args.install:
        if args.scratch is None:
            parser.error("--install requires --scratch")
        result = install_and_smoke(args.wheels, args.manifest, args.official_tar_bin, args.scratch)
    else:
        result = inspect(args.wheels, args.manifest, args.official_tar_bin)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
