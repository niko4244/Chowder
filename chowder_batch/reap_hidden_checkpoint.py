"""Fail-closed, two-slab, layer-atomic BF16 storage for the REAP observer.

Generation zero is the complete embedding output. Generation N is the output
of decoder layer N-1. Only CURRENT authorizes a committed generation; orphan
stats, manifests and partially overwritten standby slabs are not checkpoints.
No operation here converts BF16 arithmetic values to another floating dtype.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import shutil
import struct
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
import torch

GIB = 1024 ** 3
RESERVE_BYTES = 4 * GIB
SCHEMA = "chowder.reap-bf16-layer-checkpoint.v1"


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def runtime_identity(device) -> dict:
    """JSON-serializable runtime fingerprint of the device that will compute.

    ``torch.cuda.get_device_properties().uuid`` is a ``torch._C._CUuuid``
    object, not a string (torch 2.11 / CUDA 12.8). Returning it directly made
    ``_canonical(identity)`` raise ``TypeError: Object of type _CUuuid is not
    JSON serializable`` only *after* the observer had already streamed the full
    26-shard, ~72 GB source digest -- roughly twenty minutes of work discarded at
    the checkpoint constructor. Coerce it to its stable textual form.
    """
    device_uuid = getattr(device, "uuid", None)
    return {
        "python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "numpy": np.__version__,
        "transformers": __import__("transformers").__version__,
        "cuda": torch.version.cuda,
        "device": str(device), "device_name": device.name,
        "gpu_uuid": str(device_uuid) if device_uuid is not None else None,
        "driver_gpu_identity": _driver_gpu_identity(),
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_version": torch.backends.cudnn.version(),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "flash_sdpa": torch.backends.cuda.flash_sdp_enabled(),
        "efficient_sdpa": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math_sdpa": torch.backends.cuda.math_sdp_enabled(),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def _driver_gpu_identity() -> str:
    try:
        info = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,uuid",
             "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("cannot fingerprint NVIDIA driver/GPU identity") from exc
    if not info:
        raise RuntimeError("empty NVIDIA driver/GPU identity")
    return info


def _digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    before = path.stat()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"checkpoint input changed while hashing: {path.name}")
    return h.hexdigest()


def fingerprint_inputs(
    *, model_dir: Path, weights_dir: Path, calibration_sha256: str,
    seqs: list[torch.Tensor], layer_ids: list[int], out: Path,
    num_experts: int | None = None, top_k: int | None = None,
    require_registered_original: bool = False,
) -> dict:
    """Hash actual weight payloads, token IDs, executable and numeric policy.

    Deliberately reads all source shards at admission and again on resume.
    Config/index-only digests cannot detect a changed tensor payload.
    """
    # This is an INDEPENDENT preregistered upstream identity, not a digest
    # derived from the currently supplied source files.
    trusted_upstream = None
    if require_registered_original:
        from chowder_batch.reap_original_bf16_pin import validate_registered_source_metadata
        validate_registered_source_metadata(model_dir, weights_dir)
    index = model_dir / "model.safetensors.index.json"
    weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
    shards = sorted(set(weight_map.values()))
    artifacts = {}
    for name in shards:
        p = weights_dir / name
        if (not isinstance(name, str) or not name.endswith(".safetensors")
                or Path(name).name != name or p.is_symlink()
                or not p.is_file() or p.resolve().parent != weights_dir.resolve()):
            raise ValueError("unsafe or unavailable BF16 weight shard")
        artifacts[f"weights/{name}"] = sha256_file(p)
    for name in (
        "config.json", "configuration.json", "model.safetensors.index.json",
        "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
        "generation_config.json", "merges.txt", "vocab.json",
        "preprocessor_config.json", "video_preprocessor_config.json",
    ):
        p = model_dir / name
        if p.exists():
            if p.is_symlink() or not p.is_file():
                raise ValueError("unsafe model metadata path")
            artifacts[f"model/{name}"] = sha256_file(p)
    tokens_hash = hashlib.sha256()
    for seq in seqs:
        ids = seq.detach().cpu().to(torch.int64).contiguous().numpy()
        tokens_hash.update(struct.pack("<Q", ids.size))
        tokens_hash.update(ids.astype("<i8", copy=False).tobytes())
    if require_registered_original:
        from chowder_batch.reap_original_bf16_pin import (
            validate_registered_source_artifacts, validate_registered_token_identity,
        )
        trusted_upstream = validate_registered_source_artifacts(artifacts)
        validate_registered_token_identity(
            {
                "calibration_sha256": calibration_sha256,
                "lengths": [int(s.numel()) for s in seqs],
                "token_ids_sha256": tokens_hash.hexdigest(),
            },
            n_seqs=len(seqs), n_tokens=sum(int(s.numel()) for s in seqs),
        )
    import transformers
    device = torch.cuda.get_device_properties(0)
    reference = importlib.util.find_spec(
        "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe"
    )
    if reference is None or reference.origin is None:
        raise RuntimeError("cannot fingerprint reference Qwen decoder implementation")
    source_paths = [
        Path(__file__), Path(__file__).with_name("reap_observe.py"),
        Path(reference.origin),
    ]
    return {
        "schema": SCHEMA,
        "calibration_sha256": calibration_sha256,
        "token_ids_sha256": tokens_hash.hexdigest(),
        "lengths": [int(s.numel()) for s in seqs],
        "layers": layer_ids,
        "num_experts": num_experts, "top_k": top_k,
        "output_dir": str(out.resolve()),
        "model_dir": str(model_dir.resolve()),
        "weights_dir": str(weights_dir.resolve()),
        "artifacts": artifacts,
        "trusted_upstream_pin": trusted_upstream,
        "source_sha256": {p.name: sha256_file(p) for p in source_paths},
        "runtime": runtime_identity(device),
    }


def _sync_file(path: Path) -> None:
    # Opening r+b is required for os.fsync/_commit on Windows.
    with path.open("r+b") as fh:
        fh.flush()
        os.fsync(fh.fileno())


def _atomic_bytes(path: Path, payload: bytes) -> None:
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("xb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)  # same-directory, same-volume replacement
        if os.name != "nt":
            descriptor = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_stats(path: Path, stats: dict[str, torch.Tensor]) -> None:
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("xb") as fh:
            torch.save(stats, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


class _WriterLock:
    """OS-held, automatically released process lock (not a stale PID file)."""

    def __init__(self, path: Path) -> None:
        if path.is_symlink():
            raise ValueError("BF16 writer lock must not be a symlink")
        self.handle = path.open("a+b")
        try:
            self.handle.seek(0)
            if self.handle.read(1) != b"L":
                self.handle.seek(0)
                self.handle.write(b"L")
                self.handle.flush()
            self.handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, IOError):
            self.handle.close()
            raise RuntimeError("BF16 checkpoint already has a live writer") from None

    def close(self) -> None:
        if self.handle.closed:
            return
        self.handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()


class BF16CheckpointHidden:
    """Row-indexed immutable committed input + independently writable output."""

    def __init__(self, lengths: list[int], hidden_size: int, scratch: Path,
                 identity: dict, *, resume: bool = False) -> None:
        if not lengths or any(type(n) is not int or n <= 0 for n in lengths):
            raise ValueError("checkpoint requires positive sequence lengths")
        if type(hidden_size) is not int or hidden_size <= 0:
            raise ValueError("checkpoint requires positive hidden size")
        self.lengths = tuple(lengths)
        self.hidden_size = hidden_size
        offsets = [0]
        for n in lengths:
            offsets.append(offsets[-1] + n)
        self.offsets = tuple(offsets)
        self.required_bytes = offsets[-1] * hidden_size * 2
        self.scratch = scratch
        self.identity = identity
        self.identity_hash = _digest_bytes(_canonical(identity))
        self._lock = None
        self._reader = None
        self._writer = None
        self._written: set[int] = set()
        self.generation = -1
        self.manifest_hash = None
        self.timings: list[dict] = []
        self.phase = "embed"

        if resume:
            if not scratch.is_dir() or scratch.is_symlink():
                raise ValueError("missing or unsafe BF16 checkpoint directory")
        else:
            if scratch.exists() or scratch.is_symlink() or not scratch.parent.is_dir():
                raise FileExistsError("checkpoint scratch must be new")
            free = shutil.disk_usage(scratch.parent).free
            if free < self.required_bytes * 2 + RESERVE_BYTES:
                raise RuntimeError("not enough disk for two BF16 buffers plus reserve")
            scratch.mkdir()
        try:
            self._lock = _WriterLock(scratch / "writer.lock")
            if resume:
                self._restore()
            else:
                for p in self._paths:
                    with p.open("xb") as fh:
                        fh.truncate(self.required_bytes)
                        fh.flush()
                        os.fsync(fh.fileno())
                _atomic_bytes(scratch / "layout.json", _canonical({
                    "schema": SCHEMA, "lengths": lengths,
                    "hidden_size": hidden_size, "bytes_per_slab": self.required_bytes,
                    "identity_sha256": self.identity_hash, "numeric_conversions": False,
                    "scientific_gate": "NOT_MEASURED",
                }))
                self._open_writer(0)
        except BaseException:
            self.close()
            raise

    @property
    def _paths(self) -> tuple[Path, Path]:
        return (self.scratch / "state-A.bf16", self.scratch / "state-B.bf16")

    def _open_writer(self, index: int) -> None:
        self._writer = np.memmap(self._paths[index], dtype="<u2", mode="r+",
                                 shape=(self.offsets[-1], self.hidden_size))

    def _restore(self) -> None:
        if any((self.scratch / name).is_symlink() for name in ("layout.json", "CURRENT")):
            raise RuntimeError("unsafe symlink in BF16 checkpoint metadata")
        layout = json.loads((self.scratch / "layout.json").read_bytes())
        expected = {
            "schema": SCHEMA, "lengths": list(self.lengths),
            "hidden_size": self.hidden_size, "bytes_per_slab": self.required_bytes,
            "identity_sha256": self.identity_hash, "numeric_conversions": False,
            "scientific_gate": "NOT_MEASURED",
        }
        if layout != expected:
            raise RuntimeError("BF16 checkpoint layout/input provenance mismatch")
        if any(p.is_symlink() or p.stat().st_size != self.required_bytes for p in self._paths):
            raise RuntimeError("BF16 checkpoint slab missing/truncated/replaced")
        current = json.loads((self.scratch / "CURRENT").read_bytes())
        generation = current.get("generation")
        if type(generation) is not int or generation < 0:
            raise RuntimeError("invalid BF16 CURRENT checkpoint generation")
        previous = None
        timings = []
        for n in range(generation + 1):
            manifest_path = self.scratch / f"generation-{n:03d}.json"
            if manifest_path.is_symlink():
                raise RuntimeError("unsafe BF16 checkpoint manifest alias")
            data = manifest_path.read_bytes()
            if n == generation and _digest_bytes(data) != current.get("manifest_sha256"):
                raise RuntimeError("BF16 CURRENT marker manifest SHA mismatch")
            manifest = json.loads(data)
            if (manifest.get("schema") != SCHEMA or manifest.get("generation") != n
                    or manifest.get("layer") != n - 1
                    or manifest.get("buffer") != n % 2
                    or manifest.get("identity_sha256") != self.identity_hash
                    or manifest.get("previous_sha256") != previous):
                raise RuntimeError("noncontiguous or mismatched BF16 checkpoint chain")
            if n:
                path = self.scratch / f"layer-{n-1:02d}.pt"
                if path.is_symlink() or sha256_file(path) != manifest.get("stats_sha256"):
                    raise RuntimeError("BF16 committed statistics mismatch")
                if path.stat().st_size > 4 * 1024 * 1024:
                    raise RuntimeError("BF16 statistics file exceeds expected size")
                stats = torch.load(path, map_location="cpu", weights_only=True)
                if not isinstance(stats, dict) or set(stats) != {
                    "count", "gate_sum", "norm_sum", "reap_sum",
                }:
                    raise RuntimeError("BF16 committed statistics structure mismatch")
                self._validate_stats(stats, manifest.get("timing"))
                timings.append(manifest["timing"])
            previous = _digest_bytes(data)
        active = self._paths[generation % 2]
        if active.is_symlink() or sha256_file(active) != manifest.get("state_sha256"):
            raise RuntimeError("BF16 active slab checksum mismatch")
        self.generation = generation
        self.manifest_hash = previous
        self.timings = timings
        self.phase = "committed"
        self._reader = np.memmap(active, dtype="<u2", mode="r",
                                 shape=(self.offsets[-1], self.hidden_size))

    def __len__(self) -> int:
        return len(self.lengths)

    def _bounds(self, index: int) -> tuple[int, int]:
        if type(index) is not int or not 0 <= index < len(self.lengths):
            raise IndexError("sequence outside checkpoint layout")
        return self.offsets[index], self.offsets[index + 1]

    def __getitem__(self, index: int) -> torch.Tensor:
        start, stop = self._bounds(index)
        if self.phase != "layer" or self._reader is None:
            raise RuntimeError("committed BF16 layer input not open")
        # A bounded (< one sequence) bitwise copy protects the immutable mmap
        # even if a future PyTorch call mistakenly performs an in-place op.
        # torch.from_numpy on a read-only mmap would otherwise expose UB.
        return torch.from_numpy(self._reader[start:stop].copy()).view(torch.bfloat16).unsqueeze(0)

    def __setitem__(self, index: int, value: torch.Tensor) -> None:
        start, stop = self._bounds(index)
        if (self.phase not in ("embed", "layer") or self._writer is None
                or index in self._written):
            raise RuntimeError("invalid or repeated BF16 checkpoint row write")
        if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                or value.dtype != torch.bfloat16
                or tuple(value.shape) != (1, stop - start, self.hidden_size)):
            raise TypeError("BF16 checkpoint requires exact-shape CPU BF16 tensor")
        self._writer[start:stop] = value[0].contiguous().view(torch.uint16).numpy()
        self._written.add(index)

    def begin_layer(self, layer: int) -> None:
        if (self.phase != "committed" or type(layer) is not int
                or layer != self.generation):
            raise RuntimeError("BF16 resume cannot skip or replay a layer")
        self._open_writer((self.generation + 1) % 2)
        self._written.clear()
        self.phase = "layer"

    def finish_embeddings(self) -> None:
        if self.phase != "embed":
            raise RuntimeError("embeddings are already committed")
        self._commit(layer=-1, stats=None, timing=None)

    def finish_layer(self, layer: int, stats: dict[str, torch.Tensor],
                     timing: dict) -> None:
        if self.phase != "layer" or layer != self.generation:
            raise RuntimeError("BF16 layer commit is out of order")
        if (set(stats) != {"count", "gate_sum", "norm_sum", "reap_sum"}
                or any(t.device.type != "cpu" or t.dtype != torch.float64
                       or t.ndim != 1 or not bool(torch.isfinite(t).all())
                       for t in stats.values())
                or len({t.numel() for t in stats.values()}) != 1):
            raise RuntimeError("invalid BF16 layer statistics")
        self._validate_stats(stats, timing)
        self._commit(layer=layer, stats=stats, timing=timing)

    def _validate_stats(self, stats: dict, timing: dict | None) -> None:
        expected = self.identity.get("num_experts")
        top_k = self.identity.get("top_k")
        if (not isinstance(timing, dict)
                or any(not isinstance(t, torch.Tensor) or t.dtype != torch.float64
                       or t.ndim != 1 or not bool(torch.isfinite(t).all())
                       or not bool((t >= 0).all()) for t in stats.values())
                or (expected is not None and any(t.numel() != expected for t in stats.values()))
                or not bool(torch.equal(stats["count"], stats["count"].round()))):
            raise RuntimeError("invalid BF16 committed statistics semantics")
        if top_k is not None:
            routed = sum(self.lengths) * top_k
            if (int(stats["count"].sum().item()) != routed
                    or timing.get("routed_tokens") != routed):
                raise RuntimeError("incomplete BF16 committed router census")

    def _commit(self, *, layer: int, stats: dict | None, timing: dict | None) -> None:
        if len(self._written) != len(self.lengths):
            raise RuntimeError("incomplete BF16 layer: missing sequence writes")
        next_generation = self.generation + 1
        buffer = next_generation % 2
        self._writer.flush()
        _sync_file(self._paths[buffer])
        # Hash after flushing; never accept a partially written target slab.
        state_sha = sha256_file(self._paths[buffer])
        stats_sha = None
        if stats is not None:
            path = self.scratch / f"layer-{layer:02d}.pt"
            _atomic_stats(path, stats)
            stats_sha = sha256_file(path)
        manifest = {
            "schema": SCHEMA, "generation": next_generation, "layer": layer,
            "buffer": buffer, "state_sha256": state_sha,
            "stats_sha256": stats_sha, "timing": timing,
            "previous_sha256": self.manifest_hash, "identity_sha256": self.identity_hash,
        }
        data = _canonical(manifest)
        _atomic_bytes(self.scratch / f"generation-{next_generation:03d}.json", data)
        # LAST durable step: the single authoritative generation pointer.
        _atomic_bytes(self.scratch / "CURRENT", _canonical({
            "generation": next_generation, "manifest_sha256": _digest_bytes(data),
        }))
        self._writer = None
        self._reader = np.memmap(self._paths[buffer], dtype="<u2", mode="r",
                                 shape=(self.offsets[-1], self.hidden_size))
        self.generation = next_generation
        self.manifest_hash = _digest_bytes(data)
        if timing is not None:
            self.timings.append(timing)
        self._written.clear()
        self.phase = "committed"

    def mirror_layer_stats(self, out: Path, layer: int) -> None:
        if layer >= self.generation or layer < 0:
            raise RuntimeError("cannot publish uncommitted layer statistics")
        source = self.scratch / f"layer-{layer:02d}.pt"
        target = out / source.name
        # Mirrors are convenience output, never an authority for resume.
        _atomic_bytes(target, source.read_bytes())

    def begin_final_read(self) -> None:
        if self.phase != "committed":
            raise RuntimeError("NLL requires committed final states")
        self.phase = "layer"  # enables immutable row reads without new writes

    def sealed_receipt(self) -> dict:
        """Reverify the complete committed chain and active slab before D0."""
        if self.phase not in ("committed", "layer") or self._writer is not None:
            raise RuntimeError("cannot seal an uncommitted BF16 checkpoint")
        pointer = json.loads((self.scratch / "CURRENT").read_bytes())
        if (pointer.get("generation") != self.generation
                or pointer.get("manifest_sha256") != self.manifest_hash):
            raise RuntimeError("BF16 CURRENT changed after checkpoint activation")
        previous = None
        chain = []
        for generation in range(self.generation + 1):
            manifest_path = self.scratch / f"generation-{generation:03d}.json"
            if manifest_path.is_symlink():
                raise RuntimeError("unsafe BF16 commit manifest alias")
            data = manifest_path.read_bytes()
            sha = _digest_bytes(data)
            manifest = json.loads(data)
            if (manifest.get("schema") != SCHEMA
                    or manifest.get("generation") != generation
                    or manifest.get("layer") != generation - 1
                    or manifest.get("buffer") != generation % 2
                    or manifest.get("previous_sha256") != previous
                    or manifest.get("identity_sha256") != self.identity_hash):
                raise RuntimeError("BF16 checkpoint commit chain changed")
            if generation:
                stats = self.scratch / f"layer-{generation-1:02d}.pt"
                if sha256_file(stats) != manifest.get("stats_sha256"):
                    raise RuntimeError("BF16 committed layer statistics changed")
                if stats.stat().st_size > 4 * 1024 * 1024:
                    raise RuntimeError("BF16 statistics file exceeds expected size")
                values = torch.load(stats, map_location="cpu", weights_only=True)
                self._validate_stats(values, manifest.get("timing"))
                if manifest.get("timing") != self.timings[generation - 1]:
                    raise RuntimeError("BF16 committed layer timing changed")
            chain.append({"manifest_sha256": sha, "manifest": manifest})
            previous = sha
        if previous != self.manifest_hash:
            raise RuntimeError("BF16 checkpoint root SHA changed")
        active = self._paths[self.generation % 2]
        if (active.stat().st_size != self.required_bytes
                or sha256_file(active) != chain[-1]["manifest"]["state_sha256"]):
            raise RuntimeError("BF16 committed final hidden state changed")
        return {
            "schema": SCHEMA, "generation": self.generation,
            "root_sha256": previous, "source_identity": self.identity,
            "source_identity_sha256": self.identity_hash,
            "chain": chain,
            "active_state_sha256": chain[-1]["manifest"]["state_sha256"],
            "bytes_per_slab": self.required_bytes,
        }

    def close(self) -> None:
        if self._writer is not None:
            self._writer.flush()
            self._writer = None
        self._reader = None
        if self._lock is not None:
            self._lock.close()
            self._lock = None


def validate_observation_receipt(
    observation: Path, observed: dict, *, expected_generation: int = 40,
    require_registered_original: bool = False,
) -> None:
    """Consumer-side validation without requiring the original scratch volume.

    Legacy in-RAM observations remain separate. A dual-buffer observation may
    never strip its receipt to bypass verification.
    """
    hidden = observed.get("hidden_storage")
    receipt = observed.get("checkpoint_receipt")
    checkpointed = isinstance(hidden, dict) and hidden.get("type") == (
        "dual file-backed CPU BF16 bits, atomic layer checkpoint"
    )
    if require_registered_original and not checkpointed:
        raise ValueError("registered BF16 D0 requires checkpointed original evidence")
    if not checkpointed and receipt is None:
        return
    if not checkpointed or not isinstance(receipt, dict):
        raise ValueError("BF16 checkpoint observer has missing/invalid sealed receipt")
    identity = receipt.get("source_identity")
    chain = receipt.get("chain")
    if (receipt.get("schema") != SCHEMA
            or type(receipt.get("generation")) is not int
            or receipt["generation"] != expected_generation
            or not isinstance(identity, dict)
            or _digest_bytes(_canonical(identity)) != receipt.get("source_identity_sha256")
            or not isinstance(chain, list) or len(chain) != expected_generation + 1
            or hidden.get("last_committed_generation") != expected_generation
            or observed.get("calibration_sha256") != identity.get("calibration_sha256")
            or observed.get("n_tokens") != sum(identity.get("lengths", []))):
        raise ValueError("BF16 observation checkpoint receipt identity/census mismatch")
    if require_registered_original:
        from chowder_batch.reap_original_bf16_pin import (
            OriginalBF16PinError, validate_registered_source_artifacts,
            validate_registered_token_identity,
        )
        try:
            pinned = validate_registered_source_artifacts(identity.get("artifacts"))
            validate_registered_token_identity(
                identity,
                n_seqs=observed.get("n_seqs"),
                n_tokens=observed.get("n_tokens"),
            )
        except OriginalBF16PinError as exc:
            raise ValueError(f"unregistered BF16 original source: {exc}") from exc
        lengths = identity.get("lengths")
        if (identity.get("trusted_upstream_pin") != pinned
                or identity.get("output_dir") != str(observation.resolve())
                or observed.get("n_seqs") != 704
                or identity.get("num_experts") != 256
                or identity.get("top_k") != 8
                or identity.get("layers") != list(range(40))
                or type(lengths) is not list or len(lengths) != 704
                or not all(type(n) is int and 1 <= n <= 2048 for n in lengths)
                or sum(lengths) != 1_192_632):
            raise ValueError("registered BF16 source identity/token census mismatch")
    reported_layers = observed.get("layers")
    if not isinstance(reported_layers, list) or len(reported_layers) != expected_generation:
        raise ValueError("BF16 observation has incomplete committed layer timings")
    previous = None
    for generation, entry in enumerate(chain):
        if not isinstance(entry, dict) or not isinstance(entry.get("manifest"), dict):
            raise ValueError("BF16 receipt entry malformed")
        manifest = entry["manifest"]
        sha = _digest_bytes(_canonical(manifest))
        if (sha != entry.get("manifest_sha256")
                or manifest.get("schema") != SCHEMA
                or manifest.get("generation") != generation
                or manifest.get("layer") != generation - 1
                or manifest.get("buffer") != generation % 2
                or manifest.get("identity_sha256") != receipt["source_identity_sha256"]
                or manifest.get("previous_sha256") != previous):
            raise ValueError("BF16 observation checkpoint chain broken")
        if generation:
            layer = generation - 1
            path = observation / f"layer-{layer:02d}.pt"
            if (path.is_symlink() or not path.is_file()
                    or sha256_file(path) != manifest.get("stats_sha256")
                    or manifest.get("timing") != reported_layers[layer]):
                raise ValueError("BF16 observation layer stats/timings are not committed")
            if require_registered_original:
                if path.stat().st_size > 4 * 1024 * 1024:
                    raise ValueError("registered BF16 layer statistics exceed size limit")
                try:
                    stats = torch.load(path, map_location="cpu", weights_only=True)
                except Exception as exc:
                    raise ValueError(
                        f"registered BF16 layer {layer} statistics cannot be decoded"
                    ) from exc
                if not isinstance(stats, dict) or set(stats) != {
                    "count", "gate_sum", "norm_sum", "reap_sum",
                }:
                    raise ValueError("registered BF16 layer statistics missing fields")
                for name, tensor in stats.items():
                    if (not isinstance(tensor, torch.Tensor)
                            or tensor.dtype != torch.float64
                            or tuple(tensor.shape) != (256,)
                            or not bool(torch.isfinite(tensor).all())
                            or not bool((tensor >= 0).all())):
                        raise ValueError(f"registered BF16 layer invalid {name} tensor")
                count = stats["count"]
                if (not bool(torch.equal(count, count.round()))
                        or int(count.sum().item()) != 1_192_632 * 8
                        or not bool((stats["gate_sum"] <= count + 1e-6).all())
                        or not bool((stats["reap_sum"] <= stats["norm_sum"] + 1e-6).all())
                        or abs(float(stats["gate_sum"].sum()) - 1_192_632) > 1_192_632 * .005
                        or manifest["timing"].get("layer") != layer
                        or manifest["timing"].get("routed_tokens") != 1_192_632 * 8):
                    raise ValueError("registered BF16 layer router census or saliency inconsistent")
                empty = count == 0
                if any(bool((stats[name][empty] != 0).any())
                       for name in ("gate_sum", "norm_sum", "reap_sum")):
                    raise ValueError("registered BF16 layer has nonzero unrouted saliency")
        previous = sha
    if (receipt.get("root_sha256") != previous
            or receipt.get("active_state_sha256") != chain[-1]["manifest"].get("state_sha256")):
        raise ValueError("BF16 observation final checkpoint root mismatch")
