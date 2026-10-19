from __future__ import annotations

import hashlib
import importlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping


class LocalModelCompatibilityError(ValueError):
    """Raised when a local custom model is not explicitly and safely pinned."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_local_custom_code(
    model_path: str | Path,
    expected_digests: Mapping[str, str],
) -> dict[str, str]:
    root = Path(model_path).expanduser().resolve()
    if not root.is_dir():
        raise LocalModelCompatibilityError(f"local custom model directory not found: {root}")
    if not expected_digests:
        raise LocalModelCompatibilityError(
            "local custom code requires explicit SHA-256 digests"
        )
    actual: dict[str, str] = {}
    for name, expected in expected_digests.items():
        if not isinstance(name, str) or Path(name).name != name or Path(name).suffix != ".py":
            raise LocalModelCompatibilityError(f"invalid local custom-code file: {name!r}")
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
            raise LocalModelCompatibilityError(f"invalid digest for local custom code: {name!r}")
        path = root / name
        if not path.is_file():
            raise LocalModelCompatibilityError(f"local custom-code file not found: {path}")
        digest = _sha256(path)
        actual[name] = digest
        if digest.lower() != expected.lower():
            raise LocalModelCompatibilityError(
                f"local custom-code digest changed for {name!r}"
            )
    return actual


def patch_transformers5_custom_model(
    model_path: str | Path,
    expected_digests: Mapping[str, str],
) -> None:
    """Patch only known 4.57-to-5.x Spark incompatibilities after verification.

    Verifies the model repo's custom Python against ``expected_digests`` HERE,
    inside the same process that will import it -- importing is arbitrary
    code execution, so the digest gate travels with the import instead of
    trusting each caller to have validated beforehand.
    """
    verify_local_custom_code(model_path, expected_digests)
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    root = str(Path(model_path).expanduser().resolve())
    config = AutoConfig.from_pretrained(
        root, trust_remote_code=True, local_files_only=True
    )
    auto_map = getattr(config, "auto_map", {})
    target = auto_map.get("AutoModelForCausalLM")
    if not isinstance(target, str) or "." not in target:
        raise LocalModelCompatibilityError(
            "local custom model has no AutoModelForCausalLM mapping"
        )
    model_class = get_class_from_dynamic_module(
        target, root, trust_remote_code=True, local_files_only=True
    )
    if isinstance(getattr(model_class, "_tied_weights_keys", None), list):
        index_path = Path(root) / "model.safetensors.index.json"
        source = "model.embedding.weight"
        if index_path.is_file():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            if source not in index.get("weight_map", {}):
                raise LocalModelCompatibilityError(
                    "cannot verify the Spark embedding weight in the model index"
                )
        model_class._tied_weights_keys = {"lm_head.weight": source}

    module = sys.modules.get(model_class.__module__)
    if module is None:
        raise LocalModelCompatibilityError("custom model module was not imported")
    for name in ("create_causal_mask", "create_sliding_window_causal_mask"):
        create_mask = getattr(module, name, None)
        if create_mask is None or getattr(create_mask, "_chowder_spark_compat", False):
            continue

        def compat_create_mask(*args: Any, _create_mask=create_mask, **kwargs: Any) -> Any:
            if "input_embeds" in kwargs and "inputs_embeds" not in kwargs:
                kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
            kwargs.pop("cache_position", None)
            return _create_mask(*args, **kwargs)

        compat_create_mask._chowder_spark_compat = True
        setattr(module, name, compat_create_mask)

    importlib.invalidate_caches()
