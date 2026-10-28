"""Verify the ORIGINAL constructed Q4/Q8 GGUF transport path before model load.

The frozen 733-tensor construction receipts bind the OUTPUT ABSOLUTE PATH,
not merely the GGUF digest. Transferring weights to a different drive while
copying their sidecars does not make a valid original source-bound cohort.
Do not edit/reseal original sidecars to accommodate a new machine. Recreate
the source drive layout and independent same-byte SHA verification instead.

This read-only preflight verifies the sidecar's registered source/quantizer,
tensor inventory, original output path, length, and declared SHA. It does
NOT read/hash multi-GB model data, establish hardware capacity, or launch.
Owner supervisor still verifies the *actual full GGUF bytes* after admission.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from chowder_batch import reap_serve
from chowder_batch import reap_v2_dedicated_runner_admission_20261027 as admission
from chowder_batch import reap_v2_frozen_wave_20261027 as wave

SCHEMA = "chowder.reap-v2-original-weight-transport-path.v1"
PINNED_SERVE_SOURCE_SHA256 = "f62a0de3669acae33252a9e178eea134f7a31d98f5b8d2e6746ff8a603cab3dd"


class OriginalTransportRefused(ValueError):
    """Original frozen source bytes/path cannot be claimed by this runner."""


def _identity(path: Path) -> str:
    # Normalize Windows slash spelling only, not movement to a new directory.
    return os.path.normcase(os.path.normpath(os.path.abspath(str(path))))


def preflight(study: wave.Study, model: Path) -> dict:
    original = wave.study_for(study.arm)
    if (study.arm, study.suite, study.limit, study.model_sha256) != (
        original.arm, original.suite, original.limit, original.model_sha256
    ):
        raise OriginalTransportRefused("unknown original frozen arm/model identity")
    model = Path(model)
    native = admission.FROZEN_MODELS[wave.NATIVE_ADMISSION_ARM[study.arm]]
    if (not model.is_absolute() or model.is_symlink() or not model.is_file()
            or _identity(model) != _identity(Path(native.original_path))):
        raise OriginalTransportRefused(
            f"original path required: {native.original_path}"
        )
    if model.stat().st_size != native.bytes:
        raise OriginalTransportRefused("original GGUF size differs from registered source")
    if native.sha256 != study.model_sha256:
        raise OriginalTransportRefused("original immutable model SHA differs from frozen study")

    result = {
        "schema": SCHEMA,
        "status": "ORIGINAL_PATH_AND_REGISTERED_CONSTRUCTION_RECEIPT_ONLY",
        "arm": study.arm,
        "original_absolute_path": str(model),
        "model_size_bytes": native.bytes,
        "expected_model_sha256": study.model_sha256,
        "actual_full_model_sha256_measured": False,
        "sidecar_sha256": None,
        "read_only": True,
        "model_loaded": False,
        "gen0": {"status": "BLOCKED", "chosen": None, "release_approved": False},
    }
    if study.arm in reap_serve.REGISTERED_V2_CONSTRUCTION:
        if hashlib.sha256(Path(reap_serve.__file__).read_bytes()).hexdigest() != PINNED_SERVE_SOURCE_SHA256:
            raise OriginalTransportRefused("frozen original construction verifier source has changed")
        receipt_path = model.with_name(model.name + ".reap-chunked.json")
        if (not receipt_path.is_file() or receipt_path.is_symlink()
                or not 0 < receipt_path.stat().st_size <= 4 * 1024 * 1024):
            raise OriginalTransportRefused("original immutable GGUF construction receipt absent")
        raw = receipt_path.read_bytes()
        try:
            # The original source verifier performs comprehensive 733-tensor
            # checks. Here also reject silently overwritten JSON duplicate keys.
            def unique(pairs):
                obj = {}
                for name, value in pairs:
                    if name in obj:
                        raise ValueError("duplicate frozen sidecar field")
                    obj[name] = value
                return obj

            doc = json.loads(raw, object_pairs_hook=unique,
                             parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite")))
            if not isinstance(doc, dict) or _identity(Path(doc["output"])) != _identity(model):
                raise OriginalTransportRefused("original construction output path is not preserved")
            reap_serve._verify_registered_v2_construction(model, study.arm, study.model_sha256)
        except (ValueError, KeyError, TypeError, OSError, UnicodeError) as exc:
            raise OriginalTransportRefused("original source-bound construction receipt invalid: " + str(exc)[:240]) from exc
        result["sidecar_sha256"] = hashlib.sha256(raw).hexdigest()
    return result


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = preflight(wave.study_for(args.arm), args.model)
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0
    except (OriginalTransportRefused, ValueError, OSError) as exc:
        print(json.dumps({"status": "BLOCKED_ORIGINAL_MODEL_TRANSPORT",
                          "reason": str(exc)[:400], "full_model_hashed": False,
                          "model_executed": False,
                          "gen0": {"status": "BLOCKED", "chosen": None,
                                     "release_approved": False}}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
