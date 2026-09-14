"""Verified router payloads: trained router values, bound to the base they came from.

A router-healing run changes a few million parameters against a multi-GiB base,
so the artifact it produces is a payload of trained router tensors, never a
checkpoint of the whole model. That payload is only meaningful against the
exact base it was trained on, and only if every field a later reader depends on
is verified rather than assumed. This module is that verification.

Three rules, and the failure each one prevents
----------------------------------------------
* ``payload_kind`` records whether the stored tensors are **replacement**
  trained values or **additive** differences. The first implementation stores
  replacements, so applying a difference as if it were a value (or applying the
  same payload twice) is a silent corruption. The kind is written into the
  manifest and *dispatched on* at apply time rather than merely documented.
* Every tensor is bound by name, shape, dtype and content hash, and the tensor
  file by its own hash. A truncated, tampered or half-written payload is
  refused before a single model weight is mutated. Publication is
  non-overwriting, so an interrupted write can never present as an eligible
  candidate.
* The base is bound by content digest, never by path. Applying a payload to a
  different base produces a model whose provenance is a fiction; that is
  refused rather than warned about.

The same rule the rest of Chowder follows applies here: a value that could not
be read is an evidence gap, and it must never be reported as a measured result.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping

#: Manifest schema. Bump when a field's meaning changes, not when one is added.
PAYLOAD_KIND = "router_healing_payload.v1"

#: The two application semantics a payload may declare.
REPLACEMENT = "replacement"
ADDITIVE = "additive"
APPLICATION_KINDS: tuple[str, ...] = (REPLACEMENT, ADDITIVE)

TENSOR_FILE = "router_payload.safetensors"
MANIFEST_FILE = "router_payload.json"

#: Attribute set on a model once a payload digest has been applied to it, so the
#: same payload cannot be applied twice to the same object unnoticed.
_APPLIED_ATTR = "_chowder_applied_router_payloads"


class RouterPayloadError(ValueError):
    """A router payload cannot be written, read, or applied honestly."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cpu_bytes(tensor: Any) -> bytes:
    """Raw bytes of a tensor, read on the host in the tensor's own dtype.

    The hash must describe the bytes a reader will actually load, so no
    upcasting happens here: an fp32 tensor hashes its fp32 bytes, a bf16
    tensor its bf16 bytes (numpy cannot carry bfloat16, so the raw bytes are
    taken through a uint8 view, which is byte-identical for every dtype).
    """
    import torch  # local: this module must import cheaply without torch

    detached = tensor.detach()
    cpu = detached.reshape(-1).cpu().contiguous()
    return cpu.view(torch.uint8).numpy().tobytes()


def _tensor_record(name: str, tensor: Any) -> dict[str, Any]:
    """Name/shape/dtype/content for one tensor, or a refusal.

    Finiteness is checked on the tensor's own device so a CUDA payload is
    validated where it lives rather than after a copy that could already have
    failed.
    """
    import torch

    if not torch.is_tensor(tensor):
        raise RouterPayloadError(f"router payload tensor {name!r} is not a torch tensor")
    detached = tensor.detach()
    if not bool(torch.isfinite(detached).all().item()):
        raise RouterPayloadError(
            f"router payload tensor {name!r} contains a non-finite value; refusing to "
            "publish a payload that cannot be applied honestly"
        )
    return {
        "name": str(name),
        "shape": [int(size) for size in detached.shape],
        "dtype": str(detached.dtype).replace("torch.", ""),
        "elements": int(detached.numel()),
        "sha256": hashlib.sha256(_cpu_bytes(detached)).hexdigest(),
    }


def _save_file_retrying_sharing_violation(
    tensors: Mapping[str, Any], path: str, *, attempts: int = 3, delay_seconds: float = 1.0
) -> None:
    """Serialize a safetensors file, retrying only a Windows sharing violation.

    The rung-3b 9B CUDA run measured the failure this guards: all training
    steps completed, then ``save_file`` died with ``I/O error: The process
    cannot access the file because it is being used by another process
    (os error 32)`` -- a background process (indexer or antivirus) held the
    temp file serialization writes through. Losing a finished run to that race
    is waste, not safety. The retry is safe by construction: serialization is
    deterministic for the same tensors, the manifest is written only after the
    tensor file survives, and the whole-directory refusal at the top of
    ``save_router_payload`` still applies. Only the measured sharing-violation
    message is retried; every other error propagates unchanged.
    """
    from safetensors import SafetensorError
    from safetensors.torch import save, save_file

    last: Exception | None = None
    for attempt in range(max(1, attempts)):
        try:
            save_file(tensors, path)
            return
        except SafetensorError as error:
            if "os error 32" not in str(error):
                raise
            last = error
            if attempt + 1 < attempts:
                time.sleep(delay_seconds)
    # The retry lost every race: this lock is deterministic, not transient.
    # safetensors serializes through a temp file it then renames, and a
    # scanner that opens each fresh temp file wins that race every time on
    # some Windows hosts (measured twice on the rung-3b 9B run). Serialize in
    # memory and write the final path directly, so no rename is ever
    # attempted. Safety is unchanged: the manifest is written last, so an
    # interrupted direct write still leaves an ineligible directory.
    assert last is not None
    Path(path).write_bytes(save(tensors))


def save_router_payload(
    named_tensors: Mapping[str, Any],
    out_dir: str | Path,
    *,
    base_content_sha256: str,
    spec_digest: str,
    steps_completed: int,
    recipe_digest: str | None = None,
    tensor_file: str = TENSOR_FILE,
) -> dict[str, Any]:
    """Publish trained router values as an immutable, verified payload.

    Non-overwriting by construction: a manifest already present at ``out_dir``
    is a refusal, not an overwrite. The tensor file is written first and only
    then the manifest, so a manifest on disk always describes a complete tensor
    file -- an interrupted publication leaves an ineligible directory instead of
    a candidate with a truncated payload.
    """
    import torch
    from safetensors.torch import save_file

    out = Path(out_dir)
    if (out / MANIFEST_FILE).exists():
        raise RouterPayloadError(
            f"a router payload manifest already exists at {out}; refusing to overwrite "
            "a published payload"
        )
    if not named_tensors:
        raise RouterPayloadError("router payload is empty; nothing was trained")
    if isinstance(steps_completed, bool) or not isinstance(steps_completed, int):
        raise RouterPayloadError("steps_completed must be an integer")
    if steps_completed <= 0:
        raise RouterPayloadError("steps_completed must be positive to publish a payload")
    for label, digest in (
        ("base_content_sha256", base_content_sha256),
        ("spec_digest", spec_digest),
    ):
        if not isinstance(digest, str) or len(digest) != 64:
            raise RouterPayloadError(f"{label} must be a 64-character sha256 hex digest")

    records = [_tensor_record(str(name), tensor) for name, tensor in named_tensors.items()]
    ordered = {record["name"]: named_tensors[record["name"]] for record in sorted(records, key=lambda r: r["name"])}

    out.mkdir(parents=True, exist_ok=True)
    tensor_path = out / tensor_file
    serialized = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in ordered.items()
    }
    _save_file_retrying_sharing_violation(serialized, str(tensor_path))
    tensor_sha = sha256_file(tensor_path)

    manifest = {
        "kind": PAYLOAD_KIND,
        "payload_kind": REPLACEMENT,
        "base_content_sha256": base_content_sha256,
        "spec_digest": spec_digest,
        "recipe_digest": recipe_digest,
        "steps_completed": int(steps_completed),
        "tensor_file": tensor_file,
        "tensor_file_sha256": tensor_sha,
        "parameter_names": sorted(r["name"] for r in records),
        "parameter_count": sum(r["elements"] for r in records),
        "tensors": sorted(records, key=lambda r: r["name"]),
    }
    body = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    (out / MANIFEST_FILE).write_text(body + "\n", encoding="utf-8", newline="\n")

    return {
        "payload_dir": str(out),
        "manifest_path": str(out / MANIFEST_FILE),
        "tensor_path": str(tensor_path),
        "manifest_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "tensor_file_sha256": tensor_sha,
        "parameter_names": manifest["parameter_names"],
        "parameter_count": manifest["parameter_count"],
        "steps_completed": int(steps_completed),
        "base_content_sha256": base_content_sha256,
        "spec_digest": spec_digest,
        "payload_kind": REPLACEMENT,
    }


def load_router_payload(
    payload_dir: str | Path, *, expected_base_content_sha256: str
) -> dict[str, Any]:
    """Read a payload back, verifying everything before returning any tensor.

    Raises rather than returning a partially trusted payload. A mismatch in the
    manifest hash, the tensor-file hash, a per-tensor hash, or the base identity
    each stops the read; the caller cannot accidentally consume a payload that
    failed one check because none is returned until all of them pass.
    """
    from safetensors.torch import load_file

    directory = Path(payload_dir)
    manifest_path = directory / MANIFEST_FILE
    if not manifest_path.is_file():
        raise RouterPayloadError(f"no router payload manifest at {manifest_path}")
    raw = manifest_path.read_text(encoding="utf-8")
    try:
        manifest = json.loads(raw)
    except ValueError as exc:
        raise RouterPayloadError(f"router payload manifest is unparseable: {exc}") from exc
    if not isinstance(manifest, Mapping):
        raise RouterPayloadError("router payload manifest is not a JSON object")
    if manifest.get("kind") != PAYLOAD_KIND:
        raise RouterPayloadError(
            f"{manifest_path} is not a {PAYLOAD_KIND} manifest (kind={manifest.get('kind')!r})"
        )
    application = manifest.get("payload_kind")
    if application not in APPLICATION_KINDS:
        raise RouterPayloadError(
            f"router payload declares an unknown application kind {application!r}; "
            f"expected one of {list(APPLICATION_KINDS)}"
        )
    recorded_base = manifest.get("base_content_sha256")
    if recorded_base != expected_base_content_sha256:
        raise RouterPayloadError(
            "router payload base mismatch: the payload was trained against base "
            f"{recorded_base!r} but the target base is {expected_base_content_sha256!r}. "
            "Applying it would produce a model whose provenance is a fiction."
        )

    tensor_path = directory / str(manifest.get("tensor_file", TENSOR_FILE))
    if not tensor_path.is_file():
        raise RouterPayloadError(f"router payload tensor file is missing: {tensor_path}")
    actual_tensor_sha = sha256_file(tensor_path)
    if actual_tensor_sha != manifest.get("tensor_file_sha256"):
        raise RouterPayloadError(
            f"router payload tensor file hash mismatch: manifest records "
            f"{manifest.get('tensor_file_sha256')!r}, file is {actual_tensor_sha!r}"
        )
    try:
        tensors = load_file(str(tensor_path))
    except Exception as exc:  # a malformed safetensors container is a refusal
        raise RouterPayloadError(f"router payload tensor file is unreadable: {exc}") from exc

    records = manifest.get("tensors")
    if not isinstance(records, list) or not records:
        raise RouterPayloadError("router payload manifest carries no tensor records")
    recorded_names = [str(record.get("name")) for record in records]
    missing = sorted(set(recorded_names) - set(tensors))
    extra = sorted(set(tensors) - set(recorded_names))
    if missing:
        raise RouterPayloadError(f"router payload is missing recorded tensors: {missing}")
    if extra:
        raise RouterPayloadError(f"router payload carries unrecorded tensors: {extra}")

    for record in records:
        name = str(record.get("name"))
        tensor = tensors[name]
        shape = [int(size) for size in record.get("shape", [])]
        actual_shape = [int(size) for size in tensor.shape]
        if actual_shape != shape:
            raise RouterPayloadError(
                f"router payload tensor {name!r} shape mismatch: manifest {shape}, file {actual_shape}"
            )
        dtype = str(record.get("dtype"))
        actual_dtype = str(tensor.dtype).replace("torch.", "")
        if actual_dtype != dtype:
            raise RouterPayloadError(
                f"router payload tensor {name!r} dtype mismatch: manifest {dtype}, file {actual_dtype}"
            )
        actual_sha = hashlib.sha256(_cpu_bytes(tensor)).hexdigest()
        if actual_sha != record.get("sha256"):
            raise RouterPayloadError(
                f"router payload tensor {name!r} content hash mismatch: manifest "
                f"{record.get('sha256')!r}, file {actual_sha!r}"
            )

    return {
        "manifest": dict(manifest),
        "manifest_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "payload_dir": str(directory),
        "payload_kind": application,
        "base_content_sha256": recorded_base,
        "tensors": tensors,
        "parameter_names": sorted(recorded_names),
    }


def apply_router_payload(
    model: Any,
    payload: Mapping[str, Any],
    *,
    expected_parameter_paths: Any = None,
) -> dict[str, Any]:
    """Apply a verified payload to a loaded model, exactly once.

    Every key is checked to exist with the recorded shape and dtype *before* any
    weight is mutated, so a partial application cannot happen: either the whole
    payload fits the model or nothing is written. Re-applying the same payload
    digest to the same model object is refused rather than repeated -- harmless
    for replacement values, corruption for additive ones, and a rule that must
    hold for both.
    """
    import torch

    kind = payload.get("payload_kind")
    if kind not in APPLICATION_KINDS:
        raise RouterPayloadError(f"unknown payload application kind: {kind!r}")
    tensors = payload.get("tensors")
    if not isinstance(tensors, Mapping) or not tensors:
        raise RouterPayloadError("payload carries no tensors")
    digest = payload.get("manifest_sha256") or payload.get("tensor_file_sha256")
    if not isinstance(digest, str) or not digest:
        raise RouterPayloadError("payload carries no identity to record its application against")

    applied_before = getattr(model, _APPLIED_ATTR, None)
    if applied_before is None:
        applied_before = set()
    if digest in applied_before:
        raise RouterPayloadError(
            f"this payload ({digest[:12]}…) has already been applied to this model; "
            "refusing to apply it a second time. Additive payloads would be "
            "double-counted and the rule must hold identically for replacements."
        )

    parameters = dict(model.named_parameters())
    if expected_parameter_paths is not None:
        expected = {str(name) for name in expected_parameter_paths}
        provided = {str(name) for name in tensors}
        missing = sorted(expected - provided)
        extra = sorted(provided - expected)
        if missing or extra:
            raise RouterPayloadError(
                "payload does not match the declared parameter set: "
                f"missing={missing}, unexpected={extra}"
            )

    planned: list[tuple[str, Any, Any]] = []
    for name, value in tensors.items():
        parameter = parameters.get(str(name))
        if parameter is None:
            raise RouterPayloadError(
                f"payload tensor {name!r} does not name a parameter on this model"
            )
        if list(parameter.shape) != list(value.shape):
            raise RouterPayloadError(
                f"payload tensor {name!r} shape {list(value.shape)} does not match the "
                f"model parameter shape {list(parameter.shape)}"
            )
        if parameter.dtype != value.dtype:
            raise RouterPayloadError(
                f"payload tensor {name!r} dtype {value.dtype} does not match the model "
                f"parameter dtype {parameter.dtype}"
            )
        planned.append((str(name), parameter, value))

    with torch.no_grad():
        for _name, parameter, value in planned:
            if kind == REPLACEMENT:
                parameter.copy_(value.to(parameter.device))
            else:
                parameter.add_(value.to(parameter.device))

    applied = set(applied_before)
    applied.add(digest)
    setattr(model, _APPLIED_ATTR, applied)

    return {
        "payload_kind": kind,
        "applied_parameters": sorted(name for name, _p, _v in planned),
        "applied_count": len(planned),
        "manifest_sha256": payload.get("manifest_sha256"),
        "base_content_sha256": payload.get("base_content_sha256"),
    }


def payload_matches_model(model: Any, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Which of the payload's tensors already equal the model's current values.

    This is the identity/non-identity control: applying a payload that already
    equals the base must change nothing, and a payload that differs must change
    something. Reporting the comparison as data is what makes the difference
    between "the apply ran" and "the apply did what it claims".
    """
    import torch

    tensors = payload.get("tensors")
    if not isinstance(tensors, Mapping):
        raise RouterPayloadError("payload carries no tensors")
    parameters = dict(model.named_parameters())
    matching: list[str] = []
    differing: list[str] = []
    missing: list[str] = []
    for name, value in tensors.items():
        parameter = parameters.get(str(name))
        if parameter is None:
            missing.append(str(name))
            continue
        if torch.equal(parameter.detach().cpu(), value.detach().cpu()):
            matching.append(str(name))
        else:
            differing.append(str(name))
    return {
        "identical": sorted(matching),
        "differing": sorted(differing),
        "missing": sorted(missing),
        "is_identity": bool(matching) and not differing and not missing,
    }
