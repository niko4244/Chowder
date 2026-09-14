"""Verified router payloads: the read/apply failure modes, and the identity control.

These are the guards that make a payload safe to hand to a *different* process
than the one that trained it. Every refusal below prevents a specific silent
corruption, so each test names the corruption it blocks.
"""
from __future__ import annotations

import errno
import json
from pathlib import Path

import pytest

from chowder.router_payload import (
    ADDITIVE,
    MANIFEST_FILE,
    TENSOR_FILE,
    REPLACEMENT,
    RouterPayloadError,
    apply_router_payload,
    load_router_payload,
    payload_matches_model,
    save_router_payload,
    sha256_file,
)

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

_BASE_SHA = "a" * 64
_SPEC_SHA = "b" * 64


class _Tiny(torch.nn.Module):
    """Parameter names shaped like the real Qwen MoE router layout."""

    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList()
        for _ in range(2):
            layer = torch.nn.Module()
            layer.mlp = torch.nn.Module()
            layer.mlp.gate = torch.nn.Linear(4, 4, bias=False)
            layer.mlp.experts = torch.nn.Linear(4, 4, bias=False)
            self.model.layers.append(layer)


ROUTER_NAMES = ("model.layers.0.mlp.gate.weight", "model.layers.1.mlp.gate.weight")


def _model(seed: int = 0) -> _Tiny:
    torch.manual_seed(seed)
    return _Tiny().float()


def _trained_values(model: _Tiny, *, shift: float) -> dict[str, torch.Tensor]:
    values: dict[str, torch.Tensor] = {}
    for name in ROUTER_NAMES:
        parameter = dict(model.named_parameters())[name]
        values[name] = (parameter.detach() + shift).clone()
    return values


def _publish(model: _Tiny, tmp_path: Path, *, shift: float, base: str = _BASE_SHA) -> dict:
    return save_router_payload(
        _trained_values(model, shift=shift),
        tmp_path / "payload",
        base_content_sha256=base,
        spec_digest=_SPEC_SHA,
        steps_completed=3,
    )


# --- the identity control ----------------------------------------------------


def test_a_payload_equal_to_the_base_changes_nothing(tmp_path):
    """Identity control: applying the base's own values must be a no-op.

    If this ever "changes something", the apply path is doing something other
    than what its manifest says, and every non-identity result is suspect.
    """
    model = _model()
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    artifact = _publish(model, tmp_path, shift=0.0)

    payload = load_router_payload(
        artifact["payload_dir"], expected_base_content_sha256=_BASE_SHA
    )
    assert payload["payload_kind"] == REPLACEMENT
    assert payload_matches_model(model, payload)["is_identity"] is True

    report = apply_router_payload(model, payload, expected_parameter_paths=ROUTER_NAMES)
    assert report["applied_count"] == 2
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter.detach(), before[name]), f"{name} moved during an identity apply"


def test_a_differing_payload_changes_exactly_the_named_parameters(tmp_path):
    """Non-identity control: the apply is real, and strictly scoped."""
    model = _model()
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    artifact = _publish(model, tmp_path, shift=1.5)
    payload = load_router_payload(
        artifact["payload_dir"], expected_base_content_sha256=_BASE_SHA
    )

    comparison = payload_matches_model(model, payload)
    assert comparison["is_identity"] is False
    assert comparison["differing"] == sorted(ROUTER_NAMES)

    apply_router_payload(model, payload, expected_parameter_paths=ROUTER_NAMES)
    after = dict(model.named_parameters())
    for name in ROUTER_NAMES:
        assert not torch.equal(after[name].detach(), before[name])
    for name in ("model.layers.0.mlp.experts.weight", "model.layers.1.mlp.experts.weight"):
        assert torch.equal(after[name].detach(), before[name]), f"frozen {name} was touched"


# --- refusals, each blocking one corruption ----------------------------------


def test_a_payload_for_a_different_base_is_refused(tmp_path):
    artifact = _publish(_model(), tmp_path, shift=1.0, base="c" * 64)
    with pytest.raises(RouterPayloadError, match="base mismatch"):
        load_router_payload(artifact["payload_dir"], expected_base_content_sha256=_BASE_SHA)


def test_a_tampered_tensor_file_is_refused(tmp_path):
    artifact = _publish(_model(), tmp_path, shift=1.0)
    tensor_path = Path(artifact["tensor_path"])
    blob = bytearray(tensor_path.read_bytes())
    blob[-1] ^= 0xFF
    tensor_path.write_bytes(bytes(blob))
    with pytest.raises(RouterPayloadError, match="tensor file hash mismatch"):
        load_router_payload(artifact["payload_dir"], expected_base_content_sha256=_BASE_SHA)


def test_a_manifest_whose_tensor_hash_is_edited_is_refused(tmp_path):
    artifact = _publish(_model(), tmp_path, shift=1.0)
    manifest_path = Path(artifact["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tensors"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    assert sha256_file(artifact["tensor_path"]) == artifact["tensor_file_sha256"]
    with pytest.raises(RouterPayloadError, match="content hash mismatch"):
        load_router_payload(artifact["payload_dir"], expected_base_content_sha256=_BASE_SHA)


def test_a_repeated_apply_is_refused(tmp_path):
    """The rule that must hold for additive payloads too."""
    model = _model()
    artifact = _publish(model, tmp_path, shift=0.25)
    payload = load_router_payload(artifact["payload_dir"], expected_base_content_sha256=_BASE_SHA)
    apply_router_payload(model, payload)
    with pytest.raises(RouterPayloadError, match="already been applied"):
        apply_router_payload(model, payload)


def test_a_shape_mismatch_is_refused_before_any_weight_moves(tmp_path):
    model = _model()
    artifact = _publish(model, tmp_path, shift=1.0)
    payload = load_router_payload(artifact["payload_dir"], expected_base_content_sha256=_BASE_SHA)

    other = _model()
    # A model whose gate is a different shape must be refused as a whole.
    other.model.layers[1].mlp.gate = torch.nn.Linear(4, 8, bias=False)
    before = {name: p.detach().clone() for name, p in other.named_parameters()}
    with pytest.raises(RouterPayloadError, match="shape"):
        apply_router_payload(other, payload)
    for name, parameter in other.named_parameters():
        assert torch.equal(parameter.detach(), before[name]), "a refused apply must write nothing"


def test_a_parameter_the_payload_names_but_the_model_lacks_is_refused(tmp_path):
    """A payload trained on two layers cannot be applied to a one-layer model."""
    model = _model()
    artifact = _publish(model, tmp_path, shift=1.0)
    payload = load_router_payload(artifact["payload_dir"], expected_base_content_sha256=_BASE_SHA)

    shorter = _model()
    shorter.model.layers = torch.nn.ModuleList([shorter.model.layers[0]])
    before = {name: p.detach().clone() for name, p in shorter.named_parameters()}
    with pytest.raises(RouterPayloadError, match="does not name a parameter"):
        apply_router_payload(shorter, payload)
    for name, parameter in shorter.named_parameters():
        assert torch.equal(parameter.detach(), before[name]), "a refused apply must write nothing"


def test_a_declared_parameter_set_is_compared_exactly(tmp_path):
    """A broader or narrower declared set than the payload carries is refused."""
    model = _model()
    artifact = _publish(model, tmp_path, shift=1.0)
    payload = load_router_payload(artifact["payload_dir"], expected_base_content_sha256=_BASE_SHA)
    with pytest.raises(RouterPayloadError, match="does not match the declared parameter set"):
        apply_router_payload(model, payload, expected_parameter_paths=ROUTER_NAMES[:1])


def test_a_nonfinite_tensor_is_refused_at_publication(tmp_path):
    model = _model()
    values = _trained_values(model, shift=0.0)
    values[ROUTER_NAMES[0]][0, 0] = float("nan")
    with pytest.raises(RouterPayloadError, match="non-finite"):
        save_router_payload(
            values,
            tmp_path / "payload",
            base_content_sha256=_BASE_SHA,
            spec_digest=_SPEC_SHA,
            steps_completed=1,
        )


def test_publication_never_overwrites_an_existing_payload(tmp_path):
    _publish(_model(), tmp_path, shift=0.0)
    with pytest.raises(RouterPayloadError, match="already exists"):
        _publish(_model(), tmp_path, shift=1.0)


def test_an_unknown_application_kind_is_refused(tmp_path):
    artifact = _publish(_model(), tmp_path, shift=1.0)
    manifest_path = Path(artifact["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["payload_kind"] = "mystery"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    with pytest.raises(RouterPayloadError, match="unknown application kind"):
        load_router_payload(artifact["payload_dir"], expected_base_content_sha256=_BASE_SHA)


def test_the_additive_kind_is_dispatched_not_documented(tmp_path):
    """A payload declaring additive semantics must add, not replace."""
    model = _model()
    artifact = _publish(model, tmp_path, shift=0.0)
    payload = load_router_payload(artifact["payload_dir"], expected_base_content_sha256=_BASE_SHA)
    payload["payload_kind"] = ADDITIVE

    baseline = {name: p.detach().clone() for name, p in model.named_parameters()}
    apply_router_payload(model, payload)
    after = dict(model.named_parameters())
    for name in ROUTER_NAMES:
        assert torch.equal(after[name].detach(), baseline[name] * 2)


def test_publishing_writes_the_tensor_file_before_the_manifest(tmp_path):
    """An interrupted publication must leave an ineligible directory, not a candidate."""
    artifact = _publish(_model(), tmp_path, shift=1.0)
    assert Path(artifact["tensor_path"]).is_file()
    assert Path(artifact["manifest_path"]).is_file()
    assert Path(artifact["tensor_path"]).name == TENSOR_FILE
    assert Path(artifact["manifest_path"]).name == MANIFEST_FILE


# --- the publication file-lock race -------------------------------------------


def test_publication_retries_a_windows_sharing_violation(tmp_path, monkeypatch):
    """A lost file-lock race after training must not discard the finished run.

    The rung-3b 9B CUDA run measured this: all 12 steps trained, then the
    safetensors serialization died with ``I/O error: The process cannot access
    the file because it is being used by another process. (os error 32)`` -- a
    background process (indexer/antivirus) held the temp file. Retrying an
    identical serialization is safe: the manifest is written last, so no
    published payload can be half-overwritten, and the refusal at the top of
    ``save_router_payload`` still guards the whole directory.
    """
    from safetensors import SafetensorError
    import safetensors.torch as st_torch

    real_save = st_torch.save_file
    calls = {"n": 0}

    def flaky_save(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise SafetensorError(
                "Error while serializing: I/O error: The process cannot access "
                "the file because it is being used by another process. "
                "(os error 32)"
            )
        return real_save(*args, **kwargs)

    monkeypatch.setattr(st_torch, "save_file", flaky_save)
    artifact = _publish(_model(), tmp_path, shift=1.0)

    assert calls["n"] == 2, "the sharing violation must be retried exactly once here"
    payload = load_router_payload(
        artifact["payload_dir"], expected_base_content_sha256=_BASE_SHA
    )
    assert payload["manifest"]["steps_completed"] == 3


def test_publication_does_not_retry_unrelated_errors(tmp_path, monkeypatch):
    """Only a file-lock race is transient; everything else must fail loudly."""
    from safetensors import SafetensorError
    import safetensors.torch as st_torch

    calls = {"n": 0}

    def make_broken(error):
        def broken(*args, **kwargs):
            calls["n"] += 1
            raise error

        return broken

    cases = [
        OSError(errno.ENOSPC, "No space left on device"),
        SafetensorError("Error while serializing: header too large"),
    ]
    for error in cases:
        calls["n"] = 0
        monkeypatch.setattr(st_torch, "save_file", make_broken(error))
        with pytest.raises(type(error)):
            save_router_payload(
                _trained_values(_model(), shift=1.0),
                tmp_path / "payload-other",
                base_content_sha256=_BASE_SHA,
                spec_digest=_SPEC_SHA,
                steps_completed=3,
            )
        assert calls["n"] == 1, f"{type(error).__name__} must not be retried"
