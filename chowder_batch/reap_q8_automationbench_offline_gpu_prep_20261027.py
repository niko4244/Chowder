"""Prepare private original-Q8 AutomationBench kernels with locked offline dependencies.

This is source preparation, NOT a Kaggle submission, benchmark score, or Gen-0 gate.
The original Q4-matched six/guard2 arms remain separate and non-qualifying pilots.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

from chowder_batch import reap_q8_automationbench_cpu_receipt_gate_20261027 as gate
from chowder_batch import reap_q8_automationbench_kaggle_20261026 as original


PRIVATE_ROOT = gate.PRIVATE_ROOT
PREFLIGHT_SOURCE = Path(__file__).with_name("reap_q8_automationbench_offline_cpu_preflight_20261027.py")
PREFLIGHT_SHA = "6fe8953ed5ef07065f7983af9abdd1dbf731f3f853871309ad783673af099bce"
WHEEL_DATASET = "nikmarco/reap128v2-automationbench-wheels-cp313-20261027"
SOURCE_DATASET = "nikmarco/reap128v2-automationbench-sourcebin-20261027"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError("original source anchor missing or ambiguous")
    return source.replace(old, new)


def patch_kernel(source: str, preflight: str, *, profile: str) -> str:
    """Preserve original evaluator, model, flags and scoring; only add CPU install."""
    if profile not in original.PROFILES or sha(preflight.encode()) != PREFLIGHT_SHA:
        raise ValueError("unsupported arm or offline installer source drift")
    ast.parse(source)
    injected = (
        "\n# Exact source-locked CPython3.13 offline bootstrap, before any GPU load.\n"
        f"OFFLINE_BOOTSTRAP_SOURCE = {preflight!r}\n"
        f"OFFLINE_BOOTSTRAP_SHA = {PREFLIGHT_SHA!r}\n"
        "OFFLINE_PACKAGES = None\n"
        "OFFLINE_TARGET = WORK / ('reap-q8-offline108-' + CFG['profile'])\n"
        "def bootstrap_offline(archive):\n"
        "    if hashlib.sha256(OFFLINE_BOOTSTRAP_SOURCE.encode()).hexdigest() != OFFLINE_BOOTSTRAP_SHA:\n"
        "        raise RuntimeError('offline source SHA mismatch')\n"
        "    manifest = unique('reap-q8-automationbench-offline-wheelhouse-manifest.json')\n"
        "    wheels = manifest.parent\n"
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
    source = replace_once(
        source,
        "    R['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()\n",
        "    # Only own temporary scratch trees; preserve official output and native logs.\n"
        "    for owned in (OFFLINE_TARGET, WORK / 'automationbench-pinned-official-source'):\n"
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
    slug = f"nikmarco/reap128v2-q8-automationbench-{profile}-offline108-20261027"
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
