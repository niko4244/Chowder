"""Deterministic, integrity-checked code-only Windows frozen REAP runner bundle.

Reads ONLY exact committed source-controlled files, never weights, local
credentials, dataset caches, evaluator output or GPU. The bundle is an
immutable input package for a separately authorized dedicated runner, NOT an
execution, resource approval or frozen Gen-0 qualification certificate.

Once committed, run:
 python -m chowder_batch.reap_v2_dedicated_portable_bundle_20261027 \
   --out evidence/reap-v2-dedicated-portable-runner-20261027.zip
"""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

REPO = Path(__file__).resolve().parent.parent
SCHEMA = "chowder.reap-v2-dedicated-portable-original-science-runner.v1"
FROZEN_FLAGS = ["--fit", "on", "-np", "4", "-c", "16384", "--jinja"]
PINNED_ORIGINAL_CODE = {
    "chowder_batch/reap_eval.py": "9736714f21834a45c23599a50fa92ef3bebc6cf70967aca7e2329d26b4d73ee1",
    "chowder_batch/reap_serve.py": "f62a0de3669acae33252a9e178eea134f7a31d98f5b8d2e6746ff8a603cab3dd",
    "chowder_batch/reap_gates.py": "09b669c2cbb51082a7271f3279b519056ea237a3ef9dad8fd64dbdb05359ed5e",
    "chowder_batch/reap_select.py": "ec0f85344805743db88dbaa4563326a051131d685fa81e9dd076181dbc0f80aa",
    "src/chowder/evaluators/scoring.py": "d2e59aca5c0a621bac8d4979d522c9b6107f811610567d21c92a2b53960860eb",
}
SOURCES = (
    "chowder_batch/reap_v2_dedicated_portable_bundle_20261027.py",
    "chowder_batch/reap_v2_original_offline_source_lock_20261027.py",
    "chowder_batch/reap_v2_original_artifact_portability_20261027.py",
    "chowder_batch/reap_v2_supervisor_integration_guard_20261027.py",
    "chowder_batch/reap_v2_live_job_lineage_guard_20261028.py",
    "chowder_batch/reap_v2_phase49_resource_diagnostics_20261028.py",
    "chowder_batch/reap_v2_phase49_owned_memory_witness_20261028.py",
    "chowder_batch/reap_v2_phase50_postexit_job_drain_20261028.py",
    "chowder_batch/reap_v2_phase51_job_event_witness_20261028.py",
    "chowder_batch/reap_v2_owner_bound_runtime_supervisor_20261027.py",
    "chowder_batch/reap_v2_original_humaneval_frozen_executor_20261027.py",
    "chowder_batch/reap_v2_humaneval_certificate_verifier_20261027.py",
    "chowder_batch/reap_v2_dedicated_runner_admission_20261027.py",
    "chowder_batch/reap_v2_frozen_wave_20261027.py",
    "chowder_batch/reap_v2_live_resource_witness_20261027.py",
    "chowder_batch/reap_v2_frozen_gate_evidence_bridge_20261027.py",
    "chowder_batch/reap_v2_authentic_amendment4_replay_20261027.py",
    "chowder_batch/reap_serve.py",
    "chowder_batch/reap_eval.py",
    "chowder_batch/reap_gates.py",
    "chowder_batch/reap_select.py",
    "chowder_batch/reap_v2_source_identity.py",
    "chowder_batch/reap_hidden_checkpoint.py",
    "chowder_batch/reap_calibration.py",
    "chowder_batch/reap_external_full_audit.py",
    "chowder_batch/reap_humaneval_v2_attestation.py",
    "chowder_batch/reap_humaneval_v38_actual_execution_20261027.py",
    "chowder_batch/reap_humaneval_independent_witness_20261027.py",
    "chowder_batch/reap_humaneval_podman_replay.py",
    "chowder_batch/reap_humaneval_podman_probe.py",
    "chowder_batch/reap_humaneval_private_preflight.py",
    "src/chowder/__init__.py",
    "src/chowder/evaluators/__init__.py",
    "src/chowder/evaluators/scoring.py",
    "docs/REAP_QWEN36_PREREG.md",
    "docs/quals/REAP128-V2-CONSTRUCTION-AND-GATES-2026-10-25.md",
    "docs/quals/reap-v2-phase46-original-frozen-wave-and-real-resource-admission-20261027.md",
    "docs/quals/reap-v2-phase47-original-native-supervisor-portable-humaneval-20261027.md",
    "docs/quals/reap-v2-phase48-original-source-fit-and-receipt-integrity-20261027.md",
    "docs/quals/reap-v2-phase49-resource-job-lineage-and-human164-frozen-intake-20261028.md",
    "docs/quals/reap-v2-phase50-natural-win32-job-drain-and-source-bound-sustained-resources-20261028.md",
    "docs/quals/reap-v2-phase51-win32-event-provenance-and-open-scientific-qualification-20261028.md",
    "docs/quals/reap-frozen-scoring-source-fingerprints-20261025.json",
)
METADATA_NAME = "ORIGINAL-REAP-PORTABLE-ENTRY-MANIFEST.json"
WHEEL_PACKAGES = ("psutil", "datasets", "huggingface_hub", "pyarrow", "numpy", "torch")


class BundleRefused(ValueError):
    """No unsafe, uncommitted or modified source can be deployed."""


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_blob(name: str, *, root: Path) -> bytes:
    proc = subprocess.run(["git", "show", "HEAD:" + name], cwd=root,
                          capture_output=True, timeout=20, check=False)
    if proc.returncode:
        raise BundleRefused("required original bundle source not committed: " + name)
    return proc.stdout


def committed_payloads(*, root: Path = REPO) -> tuple[dict[str, bytes], str]:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                          capture_output=True, text=True, timeout=15, check=True).stdout.strip()
    if len(head) != 40 or any(c not in "0123456789abcdef" for c in head):
        raise BundleRefused("Git feature HEAD invalid")
    payload = {}
    for name in SOURCES:
        file = root / name
        if not file.is_file() or file.is_symlink():
            raise BundleRefused("original source missing or symlink: " + name)
        expected = _git_blob(name, root=root)
        current = file.read_bytes()
        # This checkout uses Windows Git autocrlf: Git's immutable index blob
        # is LF but the original production evaluator/scorer byte SHA pins
        # refer to CRLF Windows files. Byte-equality against `git show` would
        # incorrectly refuse every valid historical source. Admit ONLY the
        # safe and deterministic LF<->CRLF checkout transformation; a single
        # other changed byte, embedded bare CR, or binary payload refuses.
        if b"\r" in expected or b"\r" in current.replace(b"\r\n", b""):
            raise BundleRefused("mixed or unexpected original source line endings: " + name)
        if current.replace(b"\r\n", b"\n") != expected:
            raise BundleRefused("uncommitted original source change: " + name)
        # Always pack the EXACT frozen original Windows CRLF bytes. This is
        # reproducible on any checkout and preserves original evaluator SHA
        # 9736714..., scorer SHA d2e59..., serve SHA f62a0..., even when the
        # Git transport stored their canonical text as LF.
        win_bytes = expected.replace(b"\n", b"\r\n")
        if name in PINNED_ORIGINAL_CODE and sha(win_bytes) != PINNED_ORIGINAL_CODE[name]:
            raise BundleRefused("historical frozen evaluator/scorer SHA drift: " + name)
        if len(win_bytes) > 4 * 1024 * 1024:
            raise BundleRefused("unbounded non-source file in portable bundle")
        payload[name] = win_bytes
    if len(payload) != len(SOURCES) or len(set(payload)) != len(SOURCES):
        raise BundleRefused("missing or duplicated original source")
    return payload, head


def manifest(payload: dict[str, bytes], head: str) -> dict:
    if set(payload) != set(SOURCES):
        raise BundleRefused("portable source inventory differs from frozen package plan")
    if any(sha(payload[p]) != digest for p, digest in PINNED_ORIGINAL_CODE.items()):
        raise BundleRefused("frozen original scorer/evaluator binary source drift")
    deps = {}
    for name in WHEEL_PACKAGES:
        try:
            deps[name] = metadata.version(name)
        except metadata.PackageNotFoundError as exc:
            raise BundleRefused("offline Python dependency unavailable: " + name) from exc
    return {
        "schema": SCHEMA,
        "git_head": head,
        "files_sha256": {name: sha(payload[name]) for name in SOURCES},
        "python_build": {"major": sys.version_info.major, "minor": sys.version_info.minor},
        "required_runtime_packages_exact_current_builder_versions": deps,
        "registered_flags": FROZEN_FLAGS,
        "model_gguf_bytes_embedded": 0,
        "dataset_rows_or_judge_codes_embedded": 0,
        "native_server_binary_embedded": False,
        "tokenizer_or_model_generated_outputs_embedded": False,
        "resource_admission": "REQUIRED_PER_ARM_AT_DEDICATED_HOST",
        "humaneval_executable_gate": "BLOCKED_PENDING_INDEPENDENT_GENERATED_CODE_SANDBOX",
        "amendment4_ci": "BLOCKED_PENDING_PROSPECTIVE_INDEPENDENT_METHOD_ADJUDICATION",
        "gen0": {"status": "BLOCKED", "chosen": None, "release_approved": False},
        "no_model_execution_occurred": True,
    }


def deterministic_bytes(payload: dict[str, bytes], record: dict) -> bytes:
    if record.get("schema") != SCHEMA or record.get("files_sha256") != {
        name: sha(payload[name]) for name in SOURCES
    }:
        raise BundleRefused("portable source manifest mismatches source bytes")
    body = json.dumps(record, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    content = {**payload, METADATA_NAME: body}
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9, allowZip64=False) as archive:
        for name in sorted(content):
            if Path(name).is_absolute() or ".." in Path(name).parts or "\\" in name:
                raise BundleRefused("portable archive source path unsafe")
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content[name], compress_type=zipfile.ZIP_DEFLATED,
                             compresslevel=9)
    return output.getvalue()


def verify_bytes(data: bytes) -> dict:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
        if sorted(names) != sorted([*SOURCES, METADATA_NAME]) or len(names) != len(set(names)):
            raise BundleRefused("missing, duplicate or unregistered portable archive member")
        if any(Path(name).is_absolute() or ".." in Path(name).parts or "\\" in name
               for name in names):
            raise BundleRefused("unsafe archive path")
        if any(info.file_size > 4 * 1024 * 1024 or info.is_dir()
               for info in archive.infolist()):
            raise BundleRefused("unbounded or invalid archive member")
        contents = {name: archive.read(name) for name in names}
        record = json.loads(contents[METADATA_NAME],
                            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite")))
        if (record.get("schema") != SCHEMA
                or record.get("registered_flags") != FROZEN_FLAGS
                or record.get("gen0") != {
                    "status": "BLOCKED", "chosen": None, "release_approved": False
                }
                or record.get("no_model_execution_occurred") is not True
                or record.get("files_sha256") != {
                    name: sha(contents[name]) for name in SOURCES
                }
                or any(sha(contents[name]) != digest
                       for name, digest in PINNED_ORIGINAL_CODE.items())):
            raise BundleRefused("portable original source integrity or release block failed")
        return record


def verify_runtime(record: dict) -> None:
    """Original code and the builder's exact Python dependency ABI on target."""
    if record.get("python_build") != {
        "major": sys.version_info.major, "minor": sys.version_info.minor
    }:
        raise BundleRefused("dedicated Python major/minor differs from sealed source builder")
    packages = record.get("required_runtime_packages_exact_current_builder_versions")
    if not isinstance(packages, dict) or set(packages) != set(WHEEL_PACKAGES):
        raise BundleRefused("dedicated original Python requirements incomplete")
    for name, expected in packages.items():
        try:
            current = metadata.version(name)
        except metadata.PackageNotFoundError as exc:
            raise BundleRefused("dedicated required Python dependency missing: " + name) from exc
        if current != expected:
            raise BundleRefused("dedicated Python dependency version differs: " + name)


def publish(path: Path, data: bytes) -> None:
    if not path.parent.is_dir() or path.is_symlink() or path.exists():
        raise BundleRefused("portable archive target must be new in an existing directory")
    if verify_bytes(data).get("schema") != SCHEMA:
        raise BundleRefused("portable archive verification refused")
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--out", type=Path, help="Create code-only bundle exclusively")
    action.add_argument("--verify", type=Path, help="Verify immutable code-only bundle and target ABI")
    args = parser.parse_args()
    try:
        if args.verify:
            target = args.verify
            if not target.is_file() or target.is_symlink() or not 0 < target.stat().st_size < 16 * 1024 * 1024:
                raise BundleRefused("portable dedicated archive missing, linked or too large")
            record = verify_bytes(target.read_bytes())
            verify_runtime(record)
            print(json.dumps({"status": "SOURCE_AND_ABI_VERIFIED_NO_MODEL_EXECUTION",
                              "source_git_head": record["git_head"],
                              "sha256": sha(target.read_bytes()),
                              "gen0": record["gen0"]}, sort_keys=True))
            return 0
        payload, head = committed_payloads()
        record = manifest(payload, head)
        data = deterministic_bytes(payload, record)
        publish(args.out, data)
        print(json.dumps({"status": "PORTABLE_CODE_ONLY_BUNDLE_COMPLETE",
                          "git_head": head, "sha256": sha(data),
                          "bytes": len(data), "files": len(payload),
                          "model_executed": False, "gen0": record["gen0"]}, sort_keys=True))
        return 0
    except (BundleRefused, OSError, ValueError, KeyError, subprocess.SubprocessError,
            zipfile.BadZipFile) as exc:
        print(json.dumps({"status": "REFUSED_PORTABLE_BUNDLE", "reason": str(exc)[:350],
                          "gen0": {"status": "BLOCKED", "chosen": None,
                                     "release_approved": False}}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
