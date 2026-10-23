"""Content-bound adapter reload admission, before any model or GPU load.

The training executor hashes the exported adapter before publishing it, but
the evaluation worker historically received only a path. A file replaced
after controller admission was therefore scored under the original artifact's
identity. A continuation's parent-adapter baseline was weaker: the project
declared a SHA, but its baseline worker never compared it with disk at all.

This check reuses the training executor's exact directory-digest algorithm;
no torch, PEFT, checkpoint tensor deserialisation, model shard reads or GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..provenance import sha256_directory


class AdapterIdentityRefusal(ValueError):
    """The adapter to be scored is not the one that was attested."""


def verify_adapter_reload(
    adapter_dir: str | Path | None,
    expected_sha256: str | None,
    *,
    label: str,
    base_model: str | Path | None = None,
) -> str | None:
    """Require exact adapter bytes when adapter treatment was requested.

    ``None``/``None`` is the intentional base-only evaluation. For a requested
    adapter, a digest is mandatory: a bare path is never a content identity.
    Returned digest can be used as a verified worker-side provenance witness.
    """
    if adapter_dir is None:
        if expected_sha256 is not None:
            raise AdapterIdentityRefusal(
                f"{label}: adapter SHA was supplied for a base-only evaluation"
            )
        return None
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise AdapterIdentityRefusal(
            f"{label}: adapter reload requires a 64-character lowercase SHA-256 "
            "of the exact exported directory"
        )
    directory = Path(adapter_dir)
    if not directory.is_dir() or directory.is_symlink():
        raise AdapterIdentityRefusal(f"{label}: adapter directory missing/linked: {directory}")
    # A directory hash alone would faithfully hash junk. This is the minimum
    # structural evidence PEFT needs to apply a treatment to a model. The
    # runtime liveness guard independently checks model parameter matching.
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        file = directory / filename
        if not file.is_file() or file.is_symlink() or file.stat().st_size == 0:
            raise AdapterIdentityRefusal(f"{label}: missing/empty {filename}: {directory}")
    try:
        config = json.loads((directory / "adapter_config.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise AdapterIdentityRefusal(f"{label}: invalid adapter_config.json: {error}") from error
    if not isinstance(config, dict) or config.get("peft_type") != "LORA":
        raise AdapterIdentityRefusal(f"{label}: expected an exported PEFT LoRA adapter")
    if base_model is not None:
        # No model shards are read. A saved local path can contradict the very
        # parent model the worker is about to load, even if someone rehashes
        # the altered adapter directory. This catches that incompatibility.
        declared_base = config.get("base_model_name_or_path")
        requested = Path(base_model)
        if (
            requested.is_dir()
            and isinstance(declared_base, str)
            and declared_base.strip()
            and Path(declared_base).is_absolute()
            and Path(declared_base).resolve() != requested.resolve()
        ):
            raise AdapterIdentityRefusal(
                f"{label}: PEFT adapter declares local base {declared_base!r}, "
                f"but worker requested {str(requested)!r}; refusing wrong-parent reload"
            )
    try:
        actual = sha256_directory(directory)
    except (OSError, ValueError) as error:
        raise AdapterIdentityRefusal(f"{label}: cannot attest adapter directory: {error}") from error
    if actual != expected_sha256:
        raise AdapterIdentityRefusal(
            f"{label}: adapter content digest changed before PEFT reload "
            f"(expected {expected_sha256}, actual {actual}); refusing to score "
            "a different adapter as the recorded candidate/parent"
        )
    return actual
