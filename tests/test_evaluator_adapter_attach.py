"""CPU-only pre/post-PEFT attachment identity fences."""

import json

import pytest

from chowder.evaluators.adapter_attach import attach_verified_adapter
from chowder.evaluators.adapter_identity import AdapterIdentityRefusal
from chowder.provenance import sha256_directory


def _adapter(tmp_path):
    folder = tmp_path / "adapter"
    folder.mkdir()
    (folder / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "base_model_name_or_path": "hub/model"}),
        encoding="utf-8",
    )
    (folder / "adapter_model.safetensors").write_bytes(b"original")
    return folder


def test_attached_adapter_is_verified_on_both_sides_of_loader(tmp_path):
    folder = _adapter(tmp_path)
    digest = sha256_directory(folder)
    calls = []

    def loader(base):
        calls.append(base)
        return "attached"

    result, verified = attach_verified_adapter(
        "original-model", adapter_dir=str(folder), adapter_sha256=digest,
        base_model="hub/model", loader=loader, label="candidate",
    )
    assert result == "attached"
    assert calls == ["original-model"]
    assert verified == digest


def test_changed_bytes_during_peft_load_cannot_be_reported_as_bound(tmp_path):
    folder = _adapter(tmp_path)
    digest = sha256_directory(folder)

    def loader(base):
        (folder / "adapter_model.safetensors").write_bytes(b"different-after-open")
        return "attached-untrusted"

    with pytest.raises(AdapterIdentityRefusal, match="after PEFT attach.*digest changed"):
        attach_verified_adapter(
            object(), adapter_dir=str(folder), adapter_sha256=digest,
            base_model="hub/model", loader=loader, label="candidate",
        )


def test_mutated_adapter_before_peft_load_prevents_loader_execution(tmp_path):
    folder = _adapter(tmp_path)
    digest = sha256_directory(folder)
    (folder / "adapter_model.safetensors").write_bytes(b"swapped")

    def forbidden(_):
        raise AssertionError("PEFT must not be called")

    with pytest.raises(AdapterIdentityRefusal, match="before PEFT attach.*digest changed"):
        attach_verified_adapter(
            object(), adapter_dir=str(folder), adapter_sha256=digest,
            base_model="hub/model", loader=forbidden, label="parent baseline",
        )
