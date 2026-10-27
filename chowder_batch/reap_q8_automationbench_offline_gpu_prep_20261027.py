"""Prepare private original-Q8 AutomationBench kernels with locked offline dependencies.

This is source preparation, NOT a Kaggle submission, benchmark score, or Gen-0 gate.
The original Q4-matched six/guard2 arms remain separate and non-qualifying pilots.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tarfile

from chowder_batch import reap_q8_automationbench_cpu_receipt_gate_20261027 as gate
from chowder_batch import reap_q8_automationbench_kaggle_20261026 as original


PRIVATE_ROOT = gate.PRIVATE_ROOT
PREFLIGHT_SOURCE = Path(__file__).with_name("reap_q8_automationbench_offline_cpu_preflight_20261027.py")
PREFLIGHT_SHA = "6fe8953ed5ef07065f7983af9abdd1dbf731f3f853871309ad783673af099bce"
WHEEL_DATASET = "nikmarco/reap128v2-automationbench-wheels-cp313-20261027"
SOURCE_DATASET = "nikmarco/reap128v2-automationbench-sourcebin-20261027"
WHEEL_TAR_SHA = gate.WHEEL_TAR_SHA
WHEEL_TAR_BYTES = gate.WHEEL_TAR_BYTES
WORK = Path("/kaggle/working")


def sha_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def stage_wheels(archive, destination):
    """Stage only the 109 exact regular wheelhouse members from the pinned tar.bin.

    Kaggle mounts the uploaded .tar.bin, NOT its contents. Keep the strict
    archive identity, member count, name, type, byte and exclusive-path checks
    before writing any untrusted archive content to private scratch.
    """
    if (not archive.is_file() or archive.is_symlink()
            or archive.name != "automationbench-cp313-glibc235-original-108-locked-wheels.tar.bin"
            or archive.stat().st_size != WHEEL_TAR_BYTES
            or sha_file(archive) != WHEEL_TAR_SHA):
        raise RuntimeError("mounted original locked 108-wheel tar identity mismatch")
    if (destination.exists() or destination.is_symlink()
            or destination.parent != WORK or not WORK.is_dir()):
        raise RuntimeError("wheelhouse scratch must be a new exclusive Kaggle working path")
    with tarfile.open(archive, mode="r:") as source:
        members = source.getmembers()
        if (len(members) != 109 or any(not member.isfile() for member in members)
                or any(not re.fullmatch(r"[A-Za-z0-9_.+-]+", member.name)
                       for member in members)
                or len({member.name for member in members}) != 109
                or sum(member.size for member in members) > 144 * 1024 * 1024
                or any(member.size <= 0 or member.size > 100 * 1024 * 1024
                       for member in members)):
            raise RuntimeError("locked wheelhouse tar members unsafe or incomplete")
        names = {member.name for member in members}
        manifest_name = "reap-q8-automationbench-offline-wheelhouse-manifest.json"
        if (manifest_name not in names
                or sum(name.endswith(".whl") for name in names) != 108
                or names != {manifest_name} | {name for name in names if name.endswith(".whl")}):
            raise RuntimeError("original wheelhouse 108 wheel plus manifest population mismatch")
        destination.mkdir(mode=0o700)
        for member in members:
            src = source.extractfile(member)
            if src is None:
                raise RuntimeError("wheelhouse member has no regular-file stream")
            with src, (destination / member.name).open("xb") as out:
                shutil.copyfileobj(src, out, length=1024 * 1024)
                out.flush()
                os.fsync(out.fileno())
                if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
                    raise RuntimeError("locked wheelhouse page-cache reclaim unavailable")
                os.posix_fadvise(out.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            if (destination / member.name).stat().st_size != member.size:
                raise RuntimeError("wheelhouse extracted member size changed")
    return destination, destination / manifest_name


def reclaim_clean_offline_cache(roots, profile):
    """Advise away clean scratch file cache without changing resource floors.

    Only the three newly owned Kaggle scratch trees are eligible.  A symlink,
    special file, race, missing advisory, or unbounded tree fails closed.
    This cannot reclaim anonymous RAM and never changes the cgroup limits.
    """
    if (not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED")
            or not hasattr(os, "O_NOFOLLOW")):
        raise RuntimeError("safe prelaunch clean-cache reclaim unavailable")
    if profile not in ("six", "guard2") or len(roots) != 3 or len(set(roots)) != 3:
        raise RuntimeError("expected three exclusive offline scratch trees")
    count = total = 0
    for root in roots:
        if (root.parent != WORK or root.is_symlink() or not root.is_dir()
                or root.name not in {
                    "reap-q8-offline108-" + profile,
                    "reap-q8-wheelhouse-" + profile,
                    "automationbench-pinned-official-source",
                }):
            raise RuntimeError("prelaunch reclaim path outside owned scratch")
        for entry in root.rglob("*"):
            if entry.is_symlink():
                raise RuntimeError("symlink in owned prelaunch scratch")
            mode = entry.stat(follow_symlinks=False).st_mode
            if stat.S_ISDIR(mode):
                continue
            if not stat.S_ISREG(mode):
                raise RuntimeError("nonregular prelaunch scratch file")
            count += 1
            if count > 100_000:
                raise RuntimeError("prelaunch scratch file count unbounded")
            fd = os.open(entry, os.O_RDONLY | os.O_NOFOLLOW)
            try:
                current = os.fstat(fd)
                if not stat.S_ISREG(current.st_mode):
                    raise RuntimeError("prelaunch scratch file substituted")
                total += current.st_size
                if total > 8 * 1024 ** 3:
                    raise RuntimeError("prelaunch scratch bytes unbounded")
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
    return {"advised_files": count, "advised_bytes": total}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError("original source anchor missing or ambiguous")
    return source.replace(old, new)


def patch_kernel(source: str, preflight: str, *, profile: str) -> str:
    """Add offline CPU admission and exact-set preflight; preserve official scoring."""
    if profile not in original.PROFILES or sha(preflight.encode()) != PREFLIGHT_SHA:
        raise ValueError("unsupported arm or offline installer source drift")
    ast.parse(source)
    injected = (
        "\n# Exact source-locked CPython3.13 offline bootstrap, before any GPU load.\n"
        f"WHEEL_TAR_SHA = {WHEEL_TAR_SHA!r}\n"
        f"WHEEL_TAR_BYTES = {WHEEL_TAR_BYTES!r}\n"
        "WORK = pathlib.Path('/kaggle/working')\n"
        + "import stat\n"
        + inspect.getsource(stage_wheels) + "\n"
        + inspect.getsource(reclaim_clean_offline_cache) + "\n"
        f"OFFLINE_BOOTSTRAP_SOURCE = {preflight!r}\n"
        f"OFFLINE_BOOTSTRAP_SHA = {PREFLIGHT_SHA!r}\n"
        "OFFLINE_PACKAGES = None\n"
        "OFFLINE_TARGET = WORK / ('reap-q8-offline108-' + CFG['profile'])\n"
        "def bootstrap_offline(archive):\n"
        "    if hashlib.sha256(OFFLINE_BOOTSTRAP_SOURCE.encode()).hexdigest() != OFFLINE_BOOTSTRAP_SHA:\n"
        "        raise RuntimeError('offline source SHA mismatch')\n"
        "    wheel_tar = unique('automationbench-cp313-glibc235-original-108-locked-wheels.tar.bin')\n"
        "    wheels, manifest = stage_wheels(wheel_tar, WORK / ('reap-q8-wheelhouse-' + CFG['profile']))\n"
        "    if len(list(wheels.glob('*.whl'))) != 108:\n"
        "        raise RuntimeError('original offline 108-wheel population changed')\n"
        "    namespace = {'__name__': 'reap_q8_locked_offline_bootstrap', '__file__': '<source-pinned>'}\n"
        "    exec(compile(OFFLINE_BOOTSTRAP_SOURCE, '<source-pinned>', 'exec'), namespace)\n"
        "    proof = namespace['install_and_smoke'](wheels, manifest, archive, OFFLINE_TARGET)\n"
        "    if (proof.get('status') != 'CPU_OFFLINE_600_SOURCE_LOADER_PASS_NOT_A_MODEL_SCORE'\n"
        "            or proof.get('package_count') != 108 or proof.get('weights_loaded') is not False):\n"
        "        raise RuntimeError('original offline CPU source loader not admitted')\n"
        "    return OFFLINE_TARGET / 'packages', proof\n"
    )
    source = replace_once(source, "def snapshot():\n", injected + "\ndef snapshot():\n")
    source = replace_once(
        source,
        "    interpreter = find_offline_python(official)\n",
        "    OFFLINE_PACKAGES, offline_proof = bootstrap_offline(archive)\n"
        "    R['offline_cpu_source_proof'] = offline_proof\n"
        "    interpreter = sys.executable\n",
    )
    source = replace_once(
        source,
        "'PYTHONPATH':str(source),'PYTHONDONTWRITEBYTECODE':'1'",
        "'PYTHONPATH':str(OFFLINE_PACKAGES)+os.pathsep+str(source),'PYTHONDONTWRITEBYTECODE':'1'",
    )
    # The v3 native Kaggle receipt refused at the unchanged 29.443-GiB
    # prelaunch cgroup floor, after successful original source admission.
    # Reclaim only clean file-backed scratch pages, measure both snapshots,
    # and retain the exact same fail-closed GPU/physical/cgroup admission.
    source = replace_once(
        source,
        "    gc.collect()\n    initial = snapshot()\n    check_resources(initial, prelaunch=True)\n",
        "    gc.collect()\n"
        "    R['prelaunch_before_clean_reclaim'] = snapshot()\n"
        "    R['prelaunch_clean_reclaim'] = reclaim_clean_offline_cache((\n"
        "        OFFLINE_TARGET, WORK / ('reap-q8-wheelhouse-' + CFG['profile']),\n"
        "        WORK / 'automationbench-pinned-official-source'), CFG['profile'])\n"
        "    time.sleep(3)\n"
        "    initial = snapshot()\n"
        "    R['prelaunch'] = initial\n"
        "    check_resources(initial, prelaunch=True)\n",
    )
    # The completed v2 native receipt proved that the full 600-question source
    # hash and all six task identities match, but source iteration orders two
    # sales tasks before their frozen Q4 export positions. Require exact unique
    # membership here; the unchanged official-output gate below still requires
    # the Q4 export order. Never alter the official evaluator or task scorer.
    source = replace_once(
        source,
        "    if (result.get('n') != 600 or result.get('prompt_sha') != CFG['public_prompt_sha256']\n"
        "            or result.get('domains') != ['sales','marketing','operations','support','finance','hr']\n"
        "            or [x['name'] for x in result['selected']] != CFG['tasks']):\n"
        "        raise RuntimeError('source prompt/population/order does not match Q4 exact selected tasks')\n"
        "    R['official_task_contracts'] = result['selected']\n",
        "    names = [x['name'] for x in result['selected']]\n"
        "    if (result.get('n') != 600 or result.get('prompt_sha') != CFG['public_prompt_sha256']\n"
        "            or result.get('domains') != ['sales','marketing','operations','support','finance','hr']\n"
        "            or len(names) != len(CFG['tasks']) or len(set(names)) != len(names)\n"
        "            or set(names) != set(CFG['tasks'])\n"
        "            or len({x['example_id'] for x in result['selected']}) != len(names)\n"
        "            or any(not isinstance(x['contract'],str) or\n"
        "                   re.fullmatch(r'[0-9a-f]{64}',x['contract']) is None for x in result['selected'])):\n"
        "        raise RuntimeError('source prompt/population/unique task identities do not match Q4')\n"
        "    by_name = {x['name']:x for x in result['selected']}\n"
        "    R['official_source_selected_order'] = names\n"
        "    R['official_task_contracts'] = [by_name[name] for name in CFG['tasks']]\n",
    )
    source = replace_once(
        source,
        "    R['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()\n",
        "    # Only own temporary scratch trees; preserve official output and native logs.\n"
        "    for owned in (OFFLINE_TARGET, WORK / ('reap-q8-wheelhouse-' + CFG['profile']),\n"
        "                  WORK / 'automationbench-pinned-official-source'):\n"
        "        try:\n"
        "            if owned.is_dir() and not owned.is_symlink(): shutil.rmtree(owned)\n"
        "        except OSError as cleanup_error:\n"
        "            R['scratch_cleanup_error'] = type(cleanup_error).__name__\n"
        "    R['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()\n",
    )
    ast.parse(source)
    compile(source, "q8_automationbench_offline108_private.py", "exec")
    return source


def prepare(profile: str, destination: Path, receipt: Path) -> dict:
    """Fail closed on missing authentic CPU admission, then create fresh private files."""
    proof = gate.admission(receipt)
    if proof["status"] != "CPU_ONLY_SOURCE_AND_DEPENDENCIES_ADMITTED_Q8_LAUNCH_STILL_BLOCKED":
        raise ValueError("genuine CPU admission not established")
    if profile not in gate.Q8_NOTEBOOKS:
        raise ValueError("unknown matched original arm")
    folder, filename, expected_sha = gate.Q8_NOTEBOOKS[profile]
    old_dir = PRIVATE_ROOT / folder
    old = old_dir / filename
    if not old.is_file() or old.is_symlink() or sha(old.read_bytes()) != expected_sha:
        raise ValueError("original source-pinned Q8 kernel changed")
    original_metadata = json.loads((old_dir / "kernel-metadata.json").read_bytes())
    if (original_metadata.get("dataset_sources") != [original.Q8_INPUT_DATASET, SOURCE_DATASET]
            or original_metadata.get("kernel_sources") != [original.TOOLCHAIN_KERNEL]
            or original_metadata.get("is_private") is not True
            or original_metadata.get("enable_gpu") is not True
            or original_metadata.get("enable_internet") is not False):
        raise ValueError("original private kernel admission changed")
    preflight = PREFLIGHT_SOURCE.read_bytes()
    if sha(preflight) != PREFLIGHT_SHA:
        raise ValueError("original offline CPU installer drift")
    text = patch_kernel(old.read_text(encoding="utf-8"), preflight.decode("utf-8"), profile=profile)
    output = original.validate_private_target(
        destination, owned_roots=(original.OFFICIAL_SOURCE, gate.Q4_PRIVATE_ROOT, Path(__file__).resolve().parents[1])
    )
    new_name = f"reap128v2-q8-automationbench-{profile}-offline108-20261027.py"
    # Kaggle derives the actual notebook slug from its title. Match that
    # identity explicitly so a retry updates the same private notebook rather
    # than silently creating a second experiment under a different slug.
    slug = f"nikmarco/reap128v2-q8-{profile}-original-offline108-private"
    meta = dict(original_metadata)
    meta.update({"id": slug, "title": f"reap128v2 Q8 {profile} original offline108 private",
                 "code_file": new_name,
                 "dataset_sources": [original.Q8_INPUT_DATASET, SOURCE_DATASET, WHEEL_DATASET]})
    output.mkdir(mode=0o700, exist_ok=False)
    (output / new_name).write_text(text, encoding="utf-8", newline="\n")
    (output / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return {"status": "PRIVATE_ORIGINAL_Q8_OFFLINE108_SOURCE_PREPARED_UNLAUNCHED",
            "profile": profile, "kernel_id": slug, "kernel_source_sha256": sha(text.encode()),
            "cpu_receipt_sha256": proof["cpu_receipt_sha256"],
            "original_kernel_sha256": expected_sha, "offline_preflight_sha256": PREFLIGHT_SHA,
            "original_official_source_commit": original.OFFICIAL_COMMIT,
            "original_serving_flags": list(original.PROFILES[profile]["flags"]),
            "official_score": None, "gen0_status": "BLOCKED", "chosen": None,
            "release_approved": False, "path": str(output)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, choices=sorted(original.PROFILES))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--original-cpu-receipt", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.profile, args.out, args.original_cpu_receipt), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
